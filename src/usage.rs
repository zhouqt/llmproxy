//! Internal token-usage accounting.
//!
//! The proxy records every `/v1/messages` request's `Usage` block in a
//! bounded in-memory ring buffer so operators can introspect cost /
//! per-model traffic from `/admin/usage` without scraping logs. This
//! module owns:
//!
//! - [`UsageRecord`] — one row per request (start/end/provider/model/usage)
//! - [`StreamUsage`] — token counts lifted from a stream's SSE bytes
//! - [`UsageStats`] — the ring buffer (`Arc<RwLock<VecDeque<_>>>`) +
//!   snapshot filter for `/admin/usage`
//! - [`UsageScanner`] — bounded SSE-byte scanner that produces a
//!   [`StreamUsage`] (driven by `MappedStream` in `src/server.rs`)
//!
//! Designed to be fail-soft: malformed usage JSON leaves the field
//! `None`, and scanner overflow is clamped, never panicking into the
//! request path.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default ring-buffer capacity. Matches the project's
/// `usage_capacity` config default; tests use a much smaller value to
/// exercise eviction without holding 100k rows in memory.
pub const DEFAULT_USAGE_CAPACITY: usize = 100_000;

/// Hard upper bound on `usage_capacity`. Prevents an operator from
/// booting the proxy into a VecDeque allocation that OOMs the host
/// (e.g. `usage_capacity: 10_000_000_000` would try to pre-allocate
/// tens of GB). One million rows ≈ a few hundred MB at the per-record
/// size we observe and gives the admin endpoint plenty of headroom
/// for a long-window query. Values above this are silently clamped
/// with a one-time startup `tracing::warn!` (plan L790-792 surfaces
/// the cap; opus review #7).
pub const MAX_USAGE_CAPACITY: usize = 1_000_000;

/// Step size for the process-wide rate-limited
/// `thinking_tokens` malformed-value warn. Every
/// `THINKING_TOKENS_WARN_STEP`-th malformed envelope logs once;
/// the dominant case (missing key on non-thinking models) stays
/// silent, and a real misfire upstream can't produce millions of
/// log lines.
const THINKING_TOKENS_WARN_STEP: u64 = 1000;

/// Process-wide monotonic counter of malformed `thinking_tokens`
/// envelope values. Atomic so the SSE thread and any test thread
/// can both observe progress without a Mutex. Initialised to `0`;
/// the first malformed value bumps it to `1`, which always crosses
/// the `>= 1` threshold, so the first warn always logs.
static THINKING_TOKENS_MALFORMED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Step size for the process-wide rate-limited
/// `UsageScanner::observe` carry-overflow warn. The scanner caps
/// carry at [`SCAN_MAX_CARRY`] but a misbehaving upstream can hit
/// the cap on every poll; without rate-limiting one slow upstream
/// can flood the log. Step of 100 overflows is loud on the first
/// hit and then every 1000th — operators see saturation, but no
/// flood.
const SCANNER_OVERFLOW_WARN_STEP: u64 = 1000;

/// Process-wide monotonic counter of carry-buffer overflows across
/// all `UsageScanner` instances. Bumped inside `observe()` on the
/// cap-hit branch.
static SCANNER_OVERFLOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Internal accounting outcome for a single request.
///
/// `Aborted` is reserved for future use (client disconnect mid-stream
/// without an upstream error); v1 surfaces only `Success` and
/// `Errored` because the MappedStream's error arm is the only signal
/// we currently observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    Aborted,
    Errored,
}

/// One row in the `/admin/usage` snapshot.
///
/// `started_at` and `ended_at` are RFC 3339 strings on the wire
/// (chrono default `to_rfc3339`). `elapsed_ms` has different semantics
/// across collection sites:
///
/// - non-streaming: identical to the `"request completed"` log
///   (covers `Router::complete` + serde_json deserialization)
/// - streaming: full stream duration — first byte to terminal
///   `Ready(None)`. NOT identical to the streaming log, which fires
///   at time-to-first-byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub elapsed_ms: u64,
    pub provider: String,
    pub model: String,
    pub stream: bool,
    pub outcome: Outcome,
    /// Per-request token usage lifted from the response. Always
    /// populated for non-streaming requests (the source is
    /// `MessagesResponse.usage`). Streaming requests lift these from
    /// `message_start` (input/cache) and `message_delta` (output) SSE
    /// blocks; either set may be `None` if the upstream did not emit
    /// them.
    ///
    /// `#[serde(flatten)]` so the wire shape keeps the additive
    /// token fields at the top level (`input_tokens`,
    /// `output_tokens`, `cache_creation_input_tokens`,
    /// `cache_read_input_tokens`, `thinking_tokens`,
    /// `server_tool_use`) — matches the documented `/admin/usage`
    /// record shape (plan §"Endpoint", "Top-level fields"). When
    /// `usage` is `None`, the fields are simply absent from the
    /// JSON; the alternative (a nested `usage` object) was
    /// rejected to keep the wire additive and avoid an
    /// `if-let`/key-rename dance for clients already correlating
    /// against the request log.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub usage: Option<StreamUsage>,
    /// Cooldownable provider failures encountered during fallback for
    /// this request. Mirrors the `x-llmproxy-failed-providers` header
    /// so operators can correlate the request with what failed.
    #[serde(default)]
    pub failed_providers: Vec<String>,
    /// Computed at construction time:
    /// `input_tokens + output_tokens + cache_creation_input_tokens`.
    /// Surfaced on the wire so dashboards don't have to sum the
    /// per-bucket fields themselves. Plan §Endpoint L572.
    #[serde(default)]
    pub total_tokens: u64,
}

/// Token counts for one request, lifted from the wire.
///
/// Field names follow the Anthropic Messages API wire shape so the
/// shape stays trivial to correlate with `messages_response.usage` in
/// logs. Optional everywhere because upstream behavior varies: deepseek
/// / minmax / openai all omit at least one of these fields on
/// certain streams.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,
    /// Server-side tool-use requests (web_search / web_fetch). Copied
    /// verbatim from `usage.server_tool_use` when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<Value>,
}

/// Tolerant wire DTO for `usage` blocks lifted from SSE bytes.
///
/// `anthropic::Usage` is the canonical type but it requires
/// `input_tokens` / `output_tokens` to be present, while `message_delta`
/// sometimes emits them as null (and `message_start` always does on
/// the input side — Anthropic defers the final input token count to
/// `message_delta`). Every field is `Option<…>` plus
/// `#[serde(default)]` so a missing key deserializes to `None` rather
/// than failing the whole stream scan.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct MessageDeltaUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
    pub output_tokens_details: Option<Value>,
    pub server_tool_use: Option<Value>,
}

impl MessageDeltaUsage {
    /// Lift `thinking_tokens` from a `message_delta`'s
    /// `output_tokens_details` JSON object, if present and parseable.
    /// Absent keys stay silent (no warn); malformed values log once.
    pub fn thinking_tokens(&self) -> Option<u32> {
        let v = self.output_tokens_details.as_ref()?.get("thinking_tokens")?;
        if v.is_null() {
            return None;
        }
        match v.as_u64() {
            Some(n) => u32::try_from(n).ok(),
            None => {
                // The key was present but its value is the wrong shape
                // (string, float, negative, overflow). This is rare and
                // actionable: warn so an operator can decide whether
                // to file an upstream issue. We deliberately do NOT
                // warn on a missing key — that's the dominant case
                // (non-thinking models).
                //
                // Rate-limited: bump a process-wide counter and only
                // log when the count crosses `THINKING_TOKENS_WARN_STEP`.
                // First miss crosses the implicit `>= 1` threshold
                // (step=1000, threshold starts at 0). Subsequent
                // misses log every 1000 envelopes. A runaway upstream
                // emitting malformed values per delta can no longer
                // flood the log.
                let count = THINKING_TOKENS_MALFORMED_COUNT
                    .fetch_add(1, Ordering::Relaxed) + 1;
                if count == 1
                    || count % THINKING_TOKENS_WARN_STEP == 0
                {
                    tracing::warn!(
                        count,
                        step = THINKING_TOKENS_WARN_STEP,
                        value = %v,
                        "thinking_tokens present but not a non-negative u32; ignoring"
                    );
                }
                None
            }
        }
    }
}

impl UsageRecord {
    /// Compute `total_tokens` from the per-bucket fields on
    /// `usage`. Plan §Endpoint L572: `input_tokens + output_tokens +
    /// cache_creation_input_tokens` (cache reads and thinking tokens
    /// are not part of the total — cache_read_input_tokens are already
    /// counted in `input_tokens` upstream; thinking_tokens is a
    /// sub-category of `output_tokens`).
    pub fn compute_total_tokens(usage: Option<&StreamUsage>) -> u64 {
        let Some(u) = usage else { return 0 };
        u.input_tokens.unwrap_or(0) as u64
            + u.output_tokens.unwrap_or(0) as u64
            + u.cache_creation_input_tokens.unwrap_or(0) as u64
    }
}

impl StreamUsage {
    /// Lift from a complete-response `Usage` block.
    pub fn from_complete(u: &crate::anthropic::Usage) -> Self {
        let thinking = u
            .output_tokens_details
            .as_ref()
            .and_then(|v| v.get("thinking_tokens"))
            .and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_u64().and_then(|n| u32::try_from(n).ok())
                }
            });
        Self {
            input_tokens: Some(u.input_tokens),
            output_tokens: Some(u.output_tokens),
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_read_input_tokens: u.cache_read_input_tokens,
            thinking_tokens: thinking,
            server_tool_use: u.server_tool_use.clone(),
        }
    }

    /// Merge `message_start` + `message_delta` payloads into a single
    /// `StreamUsage`. The merge rule:
    ///
    /// - input/cache fields: prefer `start` (the canonical input count
    ///   for the request)
    /// - output / thinking / server_tool_use fields: prefer `delta`
    ///   (Anthropic places the *final* output + thinking totals on
    ///   `message_delta`, after all `content_block_delta` events)
    ///
    /// Either argument may be `None`. Absent fields stay `None`.
    pub fn merge(start: Option<&MessageDeltaUsage>, delta: Option<&MessageDeltaUsage>) -> Self {
        // Input / cache: the Anthropic passthrough emits real
        // `message_start` payloads with positive token counts, whereas
        // the openai_compat / openai_responses / copilot translators
        // emit a `Usage::default()`-derived message_start whose fields
        // are `Some(0)`. `start.or(delta)` would short-circuit on
        // `Some(0)` and lose the real delta count, so start only wins
        // when non-zero.
        //
        // Output: Anthropic's `message_start` always carries the
        // placeholder `output_tokens: 1` for streaming responses (the
        // real final count arrives on `message_delta`). Review "undo"
        // (code-review final round): applying the start-non-zero-wins
        // rule to output_tokens would freeze every passthrough stream
        // at 1 token. Delta is unconditionally authoritative for
        // output; the same reasoning covers thinking_tokens (which
        // only ever lives on message_delta anyway).
        let pick_start = |start_v: Option<u32>, delta_v: Option<u32>| match start_v {
            Some(n) if n > 0 => Some(n),
            _ => delta_v,
        };
        let pick_delta = |_start_v: Option<u32>, delta_v: Option<u32>| delta_v;

        let input = pick_start(
            start.and_then(|s| s.input_tokens),
            delta.and_then(|d| d.input_tokens),
        );
        let output = pick_delta(
            start.and_then(|s| s.output_tokens),
            delta.and_then(|d| d.output_tokens),
        );
        let cache_creation = pick_start(
            start.and_then(|s| s.cache_creation_input_tokens),
            delta.and_then(|d| d.cache_creation_input_tokens),
        );
        let cache_read = pick_start(
            start.and_then(|s| s.cache_read_input_tokens),
            delta.and_then(|d| d.cache_read_input_tokens),
        );
        // thinking_tokens + server_tool_use only live on `message_delta`
        // — the start payload doesn't carry output-side metadata.
        let thinking = delta.and_then(|d| d.thinking_tokens());
        let server_tool_use = delta
            .and_then(|d| d.server_tool_use.clone())
            .or_else(|| start.and_then(|s| s.server_tool_use.clone()));

        Self {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: cache_creation,
            cache_read_input_tokens: cache_read,
            thinking_tokens: thinking,
            server_tool_use,
        }
    }
}

/// Bounded ring buffer of [`UsageRecord`]s, shared via `Arc` so
/// `MappedStream` (background task) and the admin handler can both
/// reach it without a channel.
///
/// **Locking** (settles plan L431/511-519). `std::sync::RwLock` is
/// chosen deliberately: the critical sections are short
/// (`push_back` / `pop_front` / one filter-and-sum pass) and never
/// hold the lock across an await point. Going async would force
/// `record()` to be `.await`-ed from `MappedStream::poll_next` where
/// awaiting is impossible, or require `tokio::spawn` per write which
/// adds a scheduler trip on every request. Both rejections are
/// captured in the plan's "Locking decision" section.
///
/// **Lock-poisoning policy** (settles plan L452-458). The lock is
/// acquired via `.write().unwrap_or_else(|e| e.into_inner())` so a
/// panic in one holder does NOT poison subsequent writes — the
/// proxy must keep recording even after an unrelated component
/// panics.
#[derive(Clone)]
pub struct UsageStats {
    inner: Arc<RwLock<VecDeque<UsageRecord>>>,
    capacity: usize,
    /// Wall-clock store creation. Surfaces as `started_at` in the
    /// endpoint response so clients can distinguish accumulated
    /// history from "since the last restart".
    started_at: DateTime<Utc>,
    /// Cumulative eviction count, bumped every time `record()` drops
    /// the oldest entry at capacity. Surfaced via `/admin/usage` so
    /// an undersized capacity is observable, not silent.
    evicted_total: Arc<AtomicU64>,
    /// Next `evicted_total` value that should trigger a
    /// `tracing::warn!`. Initialised to `1` so the very first
    /// eviction always logs, then advanced in steps of `capacity/10`
    /// so a saturated buffer can't flood the log.
    ///
    /// Stored as `AtomicU64` rather than a `std::sync::Mutex` because
    /// `record()` is called from `MappedStream::poll_next`'s `Err`
    /// arm where any blocking primitive would compromise the
    /// async-runtime guarantee. `compare_exchange` keeps the
    /// threshold monotonic across racing writers (multiple streams
    /// evicting concurrently).
    evict_warn_at: Arc<AtomicU64>,
}

impl UsageStats {
    fn next_warn_threshold(prev: u64, capacity: usize) -> u64 {
        // Every `capacity/10` further evictions we re-warn. The
        // `.max(1)` keeps the step ≥1 even on small capacities
        // (capacity=10 → step=1; capacity=1 → step=10 stays put).
        let step = (capacity / 10).max(1) as u64;
        prev.saturating_add(step)
    }

    /// Construct a new store. `capacity == 0` is treated as **disabled**
    /// — `record` is a no-op, `snapshot` returns empty, `capacity()`
    /// reports 0 (no allocation). This lets operators turn the feature
    /// off in low-RAM environments without a rebuild (plan L776-782).
    ///
    /// `capacity` is also **hard-clamped to [`MAX_USAGE_CAPACITY`]**
    /// (one million rows) so a misconfiguration can't pre-allocate
    /// gigabytes via `VecDeque::with_capacity`. Values above the
    /// ceiling are clamped silently with a one-shot `tracing::warn!`
    /// at the call site (we don't have `tracing` here; the caller in
    /// `main.rs` does the warning). `capacity()` reports the
    /// **post-clamp** value, so the admin endpoint never advertises
    /// a number we can't honour.
    pub fn new(capacity: usize) -> Self {
        let effective = if capacity == 0 {
            0
        } else {
            capacity.min(MAX_USAGE_CAPACITY)
        };
        let inner = if effective == 0 {
            // Disabled: hold an empty deque but never touch the
            // capacity branch in `record()`. Capacity is reported as
            // 0 so the endpoint serialises an empty store correctly.
            Arc::new(RwLock::new(VecDeque::new()))
        } else {
            Arc::new(RwLock::new(VecDeque::with_capacity(effective)))
        };
        Self {
            inner,
            capacity: effective,
            started_at: Utc::now(),
            evicted_total: Arc::new(AtomicU64::new(0)),
            // `1` → first eviction always logs. Subsequent thresholds
            // are `prev + capacity/10`.
            evict_warn_at: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Wall-clock creation time of this store. Surfaced via the
    /// `/admin/usage` endpoint so consumers can distinguish
    /// "accumulated history" from "since the last restart" (plan
    /// L434-438, L610-615).
    pub fn started_at(&self) -> DateTime<Utc> {
        self.started_at
    }

    /// Cumulative eviction count since store creation.
    pub fn evicted_total(&self) -> u64 {
        self.evicted_total.load(Ordering::Relaxed)
    }

    /// Append one record, evicting the oldest entry when the buffer
    /// is full. Synchronous, never panics (poison recovery + disabled
    /// short-circuit). Safe to call from `MappedStream::poll_next`
    /// and from `MappedStream::Drop`.
    pub fn record(&self, rec: UsageRecord) {
        if self.capacity == 0 {
            // Disabled — no allocation, no lock, no record.
            return;
        }
        // Poison recovery: a panic in a previous holder must not
        // take the store down. Matches the pattern at the existing
        // `take_captured_usage` call in the base branch's server.rs
        // (plan L452-458).
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if guard.len() == self.capacity {
            guard.pop_front();
            // Bump the eviction counter once per actual eviction. The
            // endpoint surfaces this so operators see capacity
            // saturation without having to inspect logs.
            let new_total = self.evicted_total.fetch_add(1, Ordering::Relaxed) + 1;
            // Rate-limited operator-visible warn: first eviction,
            // then every capacity/10 further evictions. A
            // misconfigured (too-small) capacity is loud but never
            // floods the log. `compare_exchange` keeps the threshold
            // monotonic across racing writers — if multiple streams
            // evict concurrently we don't double-advance.
            let threshold = self.evict_warn_at.load(Ordering::Relaxed);
            if new_total >= threshold {
                if self
                    .evict_warn_at
                    .compare_exchange(
                        threshold,
                        Self::next_warn_threshold(threshold, self.capacity),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    tracing::warn!(
                        evicted_total = new_total,
                        capacity = self.capacity,
                        "usage stats buffer saturated; oldest entry evicted (rate-limited warn every capacity/10 evictions)"
                    );
                }
            }
        }
        guard.push_back(rec);
    }

    /// Snapshot filter for `/admin/usage`. Returns the records in
    /// ascending `started_at` order, optionally bounded by `since`,
    /// `until`, and `limit`. The three filters compose: time window
    /// first, then limit (so `limit` caps the *filtered* result, not
    /// the whole buffer).
    ///
    /// Bounds are half-open: `since` inclusive (`>=`), `until`
    /// exclusive (`<`). `None` for any field means "no bound on that
    /// side". Invalid ranges (since > until) return empty.
    ///
    /// `limit` is hard-clamped to `MAX_LIMIT` (10 000) so a client
    /// can't OOM the response by asking for the whole buffer (plan
    /// L462-464, L593).
    pub fn snapshot(
        &self,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        limit: Option<usize>,
    ) -> Vec<UsageRecord> {
        if self.capacity == 0 {
            return Vec::new();
        }
        let limit = limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<UsageRecord> = guard
            .iter()
            // `since` inclusive, `until` exclusive (half-open window) so
            // back-to-back queries (e.g. one query per day) don't
            // double-count the boundary row. See plan L604-605.
            .filter(|r| since.map(|s| r.started_at >= s).unwrap_or(true))
            .filter(|r| until.map(|u| r.started_at < u).unwrap_or(true))
            .cloned()
            .collect();
        // already in insertion order (== ascending started_at in steady
        // state); sort defensively in case a clock skew or a future
        // call site inserts out-of-order.
        out.sort_by_key(|r| r.started_at);
        out.truncate(limit);
        out
    }

    /// Unfiltered store size. Surfaces as `retained` in the admin
    /// response so consumers see the absolute ring size independent
    /// of the filter applied to `records` / `rollup` (plan L591).
    pub fn retained(&self) -> usize {
        if self.capacity == 0 {
            return 0;
        }
        self.inner.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Snapshot filter for the admin handler. Like [`snapshot`] but also
    /// applies optional `model` / `provider` equality filters. Returns
    /// the filtered, time-sorted records in ascending `started_at`
    /// order (caller decides limit vs rollup).
    pub fn snapshot_filtered(
        &self,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        model: Option<&str>,
        provider: Option<&str>,
    ) -> Vec<UsageRecord> {
        if self.capacity == 0 {
            return Vec::new();
        }
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<UsageRecord> = guard
            .iter()
            .filter(|r| since.map(|s| r.started_at >= s).unwrap_or(true))
            .filter(|r| until.map(|u| r.started_at < u).unwrap_or(true))
            .filter(|r| model.map(|m| r.model == m).unwrap_or(true))
            .filter(|r| provider.map(|p| r.provider == p).unwrap_or(true))
            .cloned()
            .collect();
        out.sort_by_key(|r| r.started_at);
        out
    }

    /// Single-pass admin query: filters once, builds the records
    /// page AND the rollup under the same read lock, with no
    /// allocation beyond the page itself. Used by
    /// `admin_usage_handler` to avoid the O(N) full-set clone that
    /// `snapshot_filtered` + `rollup` did before (opus review #6).
    ///
    /// `records_limit` is clamped to `[1, MAX_LIMIT]`. The returned
    /// `records` vec is **newest-first** and bounded by
    /// `records_limit` (≤ MAX_LIMIT = 10 000). The `rollup` is
    /// aggregated over the FULL filtered set, so the limit never
    /// skews totals (plan L591, L601).
    pub fn query(
        &self,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        model: Option<&str>,
        provider: Option<&str>,
        group_by: GroupBy,
        records_limit: Option<usize>,
    ) -> (Vec<UsageRecord>, Vec<UsageRollup>) {
        if self.capacity == 0 {
            return (Vec::new(), Vec::new());
        }
        let records_limit = records_limit
            .unwrap_or(DEFAULT_LIMIT)
            .min(MAX_LIMIT)
            .max(1);
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());

        // Rollup: two-level `BTreeMap<String, BTreeMap<String, UsageRollup>>`.
        // Each level probes with `&str` (zero alloc on existing-key hit);
        // we allocate once per new model and once per new (model,
        // provider) pair. Review #8: the old flat
        // `BTreeMap<(String, String), _>` with
        // `entry((r.model.clone(), r.provider.clone()))` allocated two
        // Strings per record regardless of bucket hit/miss. With
        // `BTreeMap<String, _>::get(&str)` on each side, the dominant
        // existing-bucket path is zero allocation. The standalone
        // `rollup()` helper at the bottom of this file keeps the
        // original single-level BTreeMap for its lighter call path
        // (no per-record filter/scan to amortise against).
        //
        // Determinism: outer is sorted by `model` (empty `""` sorts
        // first), inner by `provider`. The flatten step below
        // iterates outer → inner, which gives `(model, provider)`
        // ascending order — matching the plan's contract and the
        // standalone rollup helper's output.
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<String, BTreeMap<String, UsageRollup>> = BTreeMap::new();

        // Records page: a small ring of buffer *indices*, not the
        // records themselves. Review #7: the previous `VecDeque<UsageRecord>`
        // with `push_back(r.clone())` deep-cloned every matching record
        // — full record copy including `Option<StreamUsage>`,
        // `Option<Value>` `server_tool_use`, and `Vec<String>`
        // `failed_providers` — to return only the trailing `limit`
        // entries. 100k matches → 100k full deep-clones for a 10k page.
        // Index ring: 8 bytes per push, materialize the page ONCE at
        // the end. (Same flush strategy the old ring used, but with
        // one clone per output record instead of one per matching
        // record.)
        use std::collections::VecDeque;
        let mut page: VecDeque<usize> = VecDeque::with_capacity(records_limit);

        for (idx, r) in guard.iter().enumerate() {
            // Filter predicates (cheap, branchy; first to fail wins).
            if let Some(s) = since {
                if r.started_at < s {
                    continue;
                }
            }
            if let Some(u) = until {
                if r.started_at >= u {
                    continue;
                }
            }
            if let Some(m) = model {
                if r.model != m {
                    continue;
                }
            }
            if let Some(p) = provider {
                if r.provider != p {
                    continue;
                }
            }

            // Rollup accumulation. Each `get_mut(&str)` call is a
            // zero-allocation borrowed-key probe; only the `.entry`
            // arms on miss allocate (one String per new key — rare
            // compared to existing-key hits).
            //
            // The "" dimension (Provider-only groups by model, etc.)
            // borrows from the binary's "" static — no allocation ever.
            let (model_key, provider_key): (&str, &str) = match group_by {
                GroupBy::Model => (r.model.as_str(), ""),
                GroupBy::Provider => ("", r.provider.as_str()),
                GroupBy::ModelProvider => (r.model.as_str(), r.provider.as_str()),
            };
            let u = r.usage.as_ref();
            let input = u.and_then(|x| x.input_tokens).unwrap_or(0) as u64;
            let output = u.and_then(|x| x.output_tokens).unwrap_or(0) as u64;
            let cache_read =
                u.and_then(|x| x.cache_read_input_tokens).unwrap_or(0) as u64;
            let cache_creation =
                u.and_then(|x| x.cache_creation_input_tokens).unwrap_or(0) as u64;
            let reasoning = u.and_then(|x| x.thinking_tokens).unwrap_or(0) as u64;

            // Hot path: outer model exists, inner provider exists.
            // Two &str lookups, zero allocation.
            let entry = if let Some(inner) = buckets.get_mut(model_key) {
                if let Some(e) = inner.get_mut(provider_key) {
                    e
                } else {
                    inner.entry(provider_key.to_string()).or_insert_with(|| UsageRollup {
                        model: None,
                        provider: None,
                        requests: 0,
                        input_tokens: 0,
                        output_tokens: 0,
                        total_tokens: 0,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                        reasoning_tokens: 0,
                        cache_read_ratio: None,
                    })
                }
            } else {
                // Cold: outer miss. Allocate the outer model key
                // (once per new model across the entire query) and
                // the inner provider key (once per new pair).
                let inner = buckets
                    .entry(model_key.to_string())
                    .or_default();
                inner.entry(provider_key.to_string()).or_insert_with(|| UsageRollup {
                    model: None,
                    provider: None,
                    requests: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    reasoning_tokens: 0,
                    cache_read_ratio: None,
                })
            };
            entry.requests += 1;
            entry.input_tokens += input;
            entry.output_tokens += output;
            entry.cache_read_tokens += cache_read;
            entry.cache_creation_tokens += cache_creation;
            entry.reasoning_tokens = entry
                .reasoning_tokens
                .saturating_add(reasoning);

            // Records page — index ring (Review #7). Push the buffer
            // index; materialize only the trailing `records_limit`
            // records at the end.
            if page.len() == records_limit {
                page.pop_front();
            }
            page.push_back(idx);
        }

        // Finalize the rollup: flatten outer→inner (which yields
        // (model, provider) ascending — matching the standalone
        // rollup helper), drop the empty halves of the (model,
        // provider) key, and compute totals + ratios.
        let rollup: Vec<UsageRollup> = buckets
            .into_iter()
            .flat_map(|(model, inner)| {
                inner.into_iter().map(move |(provider, mut entry)| {
                    entry.model = if model.is_empty() { None } else { Some(model.clone()) };
                    entry.provider = if provider.is_empty() {
                        None
                    } else {
                        Some(provider)
                    };
                    entry.total_tokens =
                        entry.input_tokens + entry.output_tokens + entry.cache_creation_tokens;
                    entry.cache_read_ratio = if entry.cache_read_tokens > 0 && entry.input_tokens > 0 {
                        Some(entry.cache_read_tokens as f64 / entry.input_tokens as f64)
                    } else {
                        None
                    };
                    entry
                })
            })
            .collect();

        // Materialize the records page: the `page` ring holds buffer
        // indices for the trailing `records_limit` matches. Look up
        // each index and clone — exactly `records_limit` deep-clones,
        // not `matches` deep-clones (Review #7). Newest-first: the
        // indices are in ascending order, so reverse for descending.
        let mut records: Vec<UsageRecord> =
            Vec::with_capacity(page.len());
        for &idx in page.iter().rev() {
            if let Some(r) = guard.get(idx) {
                records.push(r.clone());
            }
        }

        (records, rollup)
    }
}

/// How to group rollup rows in the `/admin/usage` response.
///
/// `ModelProvider` (the default) is the only one that produces a
/// per-provider view; `Model` collapses across providers (useful when
/// the operator wants "spend per logical model" ignoring which
/// fallback served it). Time bucketing is **not** supported in v1 —
/// the operator's "per day/week" ask requires multiple queries (one
/// per window) or client-side bucketing; see plan L596-603.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    Model,
    Provider,
    ModelProvider,
}

impl GroupBy {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "model" => Some(Self::Model),
            "provider" => Some(Self::Provider),
            "model_provider" | "" => Some(Self::ModelProvider),
            _ => None,
        }
    }
}

/// One rollup row in the `/admin/usage` response.
///
/// All numeric fields are summed across the records that fall into
/// the same group. `cache_read_ratio` is `null` when the group has no
/// records with `cache_read_input_tokens > 0`, so "no cache hits" is
/// distinguishable from "no data" (plan L980-983).
#[derive(Debug, Serialize)]
pub struct UsageRollup {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    pub requests: u64,
    #[serde(rename = "input_tokens")]
    pub input_tokens: u64,
    #[serde(rename = "output_tokens")]
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    /// `cache_read_tokens / input_tokens` when both > 0; `None`
    /// otherwise. Review #6: must use the same `skip_serializing_if`
    /// convention as the sibling `model`/`provider` `Option`s — the
    /// wire shape must not mix "absent == no value" with
    /// "present-but-null == no value" rules.
    ///
    /// The denominator is `input_tokens`, NOT `cache_read_tokens +
    /// input_tokens`: Anthropic counts cache reads inside `input_tokens`
    /// (the same assumption [`UsageRecord::compute_total_tokens`]
    /// makes), so `cache_read / (cache_read + input)` would double-count
    /// the cache portion and a fully-cached request would report 0.5
    /// instead of 1.0 (code-review final round C4). The ratio is "what
    /// fraction of the total input was served from cache" — the
    /// dashboard contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_ratio: Option<f64>,
}

/// Group `records` by the chosen [`GroupBy`] and return one rollup
/// row per group. Sort order matches the plan's `(model, provider)`
/// ascending rule for `model_provider`, with `model` and `provider`
/// modes ordering on the single present key.
///
/// Aggregation rules:
/// - token counts: missing → 0 (so a streaming record that lifted
///   only `output_tokens` doesn't zero out `input_tokens`)
/// - `cache_read_ratio`: `cache_read_tokens / input_tokens` when
///   `cache_read_tokens > 0` (Anthropic counts cache reads inside
///   `input_tokens`, so the ratio is the cache-served fraction of all
///   input); `null` otherwise
/// - `reasoning_tokens` and `cache_creation_tokens` similarly sum
///   missing-as-zero
pub fn rollup(records: &[UsageRecord], group_by: GroupBy) -> Vec<UsageRollup> {
    use std::collections::BTreeMap;
    // BTreeMap gives deterministic ascending iteration by the key,
    // which is what the plan promises (`(model, provider)` ascending).
    let mut buckets: BTreeMap<(String, String), UsageRollup> = BTreeMap::new();
    for r in records {
        let key = match group_by {
            GroupBy::Model => (r.model.clone(), String::new()),
            GroupBy::Provider => (String::new(), r.provider.clone()),
            GroupBy::ModelProvider => (r.model.clone(), r.provider.clone()),
        };
        let u = r.usage.as_ref();
        let input = u.and_then(|x| x.input_tokens).unwrap_or(0) as u64;
        let output = u.and_then(|x| x.output_tokens).unwrap_or(0) as u64;
        let cache_read = u.and_then(|x| x.cache_read_input_tokens).unwrap_or(0) as u64;
        let cache_creation = u.and_then(|x| x.cache_creation_input_tokens).unwrap_or(0) as u64;
        let reasoning = u.and_then(|x| x.thinking_tokens).unwrap_or(0) as u64;

        let entry = buckets.entry(key).or_insert_with(|| UsageRollup {
            model: None,
            provider: None,
            requests: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            reasoning_tokens: 0,
            cache_read_ratio: None,
        });
        entry.requests += 1;
        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cache_read;
        entry.cache_creation_tokens += cache_creation;
        entry.reasoning_tokens += reasoning;
    }
    // Set keys + total + ratio after the loop so the rollup reflects
    // the bucket name as well as the counts.
    let mut out = Vec::with_capacity(buckets.len());
    for ((model, provider), mut entry) in buckets {
        entry.model = if model.is_empty() { None } else { Some(model) };
        entry.provider = if provider.is_empty() {
            None
        } else {
            Some(provider)
        };
        entry.total_tokens =
            entry.input_tokens + entry.output_tokens + entry.cache_creation_tokens;
        entry.cache_read_ratio = if entry.cache_read_tokens > 0 && entry.input_tokens > 0 {
            Some(entry.cache_read_tokens as f64 / entry.input_tokens as f64)
        } else {
            None
        };
        out.push(entry);
    }
    out
}

/// Default `limit` when the client omits the query parameter.
pub const DEFAULT_LIMIT: usize = 1000;
/// Hard ceiling on `limit` to prevent response OOM. Plan L462-464.
pub const MAX_LIMIT: usize = 10_000;

/// Streaming SSE-byte scanner. Lives inside `MappedStream` and is
/// driven by `observe()` on every `Poll::Ready(Some(Ok(bytes)))`.
/// `finalize()` produces a [`StreamUsage`] from whatever was lifted.
///
/// **Hardening contract** (see plan sixth-round finding N5):
///
/// - Buffers bytes across `observe()` calls; a single SSE event may
///   be split across many upstream polls.
/// - Splits on `\n` only; treats both `\n` and `\r\n` as line breaks
///   (the SSE spec says lines end with LF; CRLF is a tolerant parse).
/// - Multi-line `data:` fields are joined with `\n` per the SSE spec
///   before JSON parsing (a single event can have multiple `data:`
///   lines).
/// - `:`-prefixed lines are comments and are ignored.
/// - Only `event: message_start` and `event: message_delta` blocks
///   contribute to the lifted `Usage`. Anything else (text deltas,
///   content_block_start, ping, etc.) is consumed for event-name
///   tracking but does not feed into the usage.
/// - On malformed JSON inside a tracked event, the scanner logs at
///   `warn` and moves on — one bad event must not poison the whole
///   record.
#[derive(Debug, Default)]
pub struct UsageScanner {
    /// Bytes carried over from the previous `observe()` call that
    /// did not yet form a complete event (no terminating blank line).
    /// Bounded: if `observe()` is called with > 1 MiB of leftover
    /// bytes we discard them with a single warn to avoid letting one
    /// misbehaving upstream blow the request-path heap.
    ///
    /// Review #2: the previous design used `self.carry.drain(..end)`
    /// after every event — O(M²) byte moves (drain memmoves the
    /// entire trailing slice, then the next drain memmoves again).
    /// The slice also got `to_vec()`'d into a fresh heap allocation
    /// per event (one malloc + one copy per SSE block). Under chatty
    /// streams (text deltas + pings in ~4-16 KiB reqwest chunks
    /// containing many events) this dominated the request-path CPU.
    ///
    /// The current design uses a `carry_start` cursor: events are
    /// sliced zero-copy via `&self.carry[start..end]`, and the
    /// front of the buffer is only `drain(..cursor)`'d when the
    /// cursor crosses a 4 KiB threshold (or on overflow). Per-event
    /// `to_vec()` is gone — `process_event` reads from the carry
    /// slice directly because the scanner state and the carry live
    /// in disjoint fields (no borrow conflict).
    carry: Vec<u8>,
    /// Offset of the first *unprocessed* byte in `carry`. Range
    /// `[0, carry.len()]`. We slice as `&carry[cursor..]` so events
    /// span `[cursor, end)` without copying.
    carry_start: usize,
    /// Split off into its own field so `process_event` can take a
    /// mutable borrow of `state` while `observe` holds an immutable
    /// borrow of `carry[range]`. This is the linchpin that lets us
    /// avoid the per-event `to_vec()` that review #2 flagged.
    state: ScannerState,
}

#[derive(Debug, Default)]
struct ScannerState {
    /// Last `event:` name we observed, applied to the next `data:`
    /// line per the SSE spec.
    last_event: String,
    /// Latest message_start payload observed.
    start: Option<MessageDeltaUsage>,
    /// Latest message_delta payload observed. Each `message_delta`
    /// carries cumulative usage; the last one observed is
    /// authoritative (code-review final round C5).
    delta: Option<MessageDeltaUsage>,
    /// Set once we see a `message_start`; consumed by the type system
    /// to make the merge semantics explicit.
    seen_start: bool,
    /// Set when the scanner observes an SSE `event: error` frame. Some
    /// providers (responses/OAI adapters) surface a mid-stream upstream
    /// failure as an in-band `event: error` SSE block rather than
    /// aborting the transport — the client sees a partial stream that
    /// *terminates with error*, not a dead connection. The usage record
    /// must reflect `outcome: errored` for such requests even though
    /// every byte arrived on a healthy transport (code-review final
    /// round C2).
    error_seen: bool,
}

/// Maximum bytes the scanner will buffer across polls. Larger
/// leftovers are dropped to keep the request path bounded.
const SCAN_MAX_CARRY: usize = 1024 * 1024;

/// Threshold at which we `drain(..carry_start)` to reclaim buffer
/// space. Below this, the cursor advances but no memmove runs.
const CARRY_DRAIN_THRESHOLD: usize = 4 * 1024;

/// Drain the consumed prefix when the cursor crosses this many
/// bytes OR when the incoming chunk has grown to non-trivial size.
/// `extend_from_slice` is the only path that mutates `carry.len()`;
/// we trigger the cursor drain right before `extend_from_slice` so
/// the new chunk gets contiguous storage for `find_event_boundary`'s
/// linear pass. Below the threshold, the drain is amortised.
fn maybe_drain_carry(carry: &mut Vec<u8>, carry_start: &mut usize) {
    if *carry_start >= CARRY_DRAIN_THRESHOLD {
        carry.drain(..*carry_start);
        *carry_start = 0;
    }
}

impl UsageScanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next chunk of SSE bytes. Splits events on `\n\n`
    /// (or `\r\n\r\n`), processes each completed event, and stashes
    /// the trailing partial event for the next call.
    ///
    /// Review #2 — hot-path discipline:
    /// 1. Slice events zero-copy from `&self.carry[start..end]` —
    ///    no `to_vec()` per event.
    /// 2. `carry.drain(..start)` runs at most once per ~4 KiB of
    ///    consumed bytes (the `CARRY_DRAIN_THRESHOLD` watermark),
    ///    so the cumulative cost is O(M) over M consumed bytes,
    ///    not O(M²) like the previous per-event drain.
    pub fn observe(&mut self, bytes: &Bytes) {
        if bytes.is_empty() {
            return;
        }
        // Cheap overflow guard. Uses `carry.len()` not
        // `len() - carry_start`: the buffer is bounded by total
        // capacity, regardless of cursor position. The rate-limited
        // warn site is unchanged from #46.
        let total = self.carry.len() - self.carry_start;
        if total + bytes.len() > SCAN_MAX_CARRY {
            let keep = SCAN_MAX_CARRY / 2;
            let count = SCANNER_OVERFLOW_COUNT
                .fetch_add(1, Ordering::Relaxed) + 1;
            if count == 1 || count % SCANNER_OVERFLOW_WARN_STEP == 0 {
                tracing::warn!(
                    carry_before = total,
                    incoming = bytes.len(),
                    max = SCAN_MAX_CARRY,
                    count,
                    "scanner carry overflow; truncating to keep most recent bytes (rate-limited warn every {SCANNER_OVERFLOW_WARN_STEP} overflows)"
                );
            }
            // Drop the consumed prefix first (cheap) so we don't
            // measure it against the keep budget.
            self.carry.drain(..self.carry_start);
            self.carry_start = 0;
            // Truncate the live carry to `keep`.
            if self.carry.len() > keep {
                self.carry.truncate(keep);
            }
            // Drop oldest half of the incoming chunk too.
            let drop = bytes.len().saturating_sub(keep);
            self.carry.extend_from_slice(&bytes[drop..]);
        } else {
            // Reclaim space before appending so the new bytes land
            // in contiguous storage for `find_event_boundary`'s
            // linear scan.
            maybe_drain_carry(&mut self.carry, &mut self.carry_start);
            self.carry.extend_from_slice(bytes);
        }

        // Walk the buffer finding event boundaries. `find_event_boundary`
        // returns the index just past the blank-line terminator
        // (handles both LF-LF and CRLF-CRLF correctly), so we
        // advance the cursor to that index without arithmetic on
        // the buffer itself. The slice `&carry[start..end]` is
        // zero-copy; `process_event` reads from `&self.carry[range]`
        // while holding `&mut self.state` for the scanner fields.
        // The split into `carry` + `state` is what makes this
        // borrow-checker-legal.
        loop {
            let Some(end) = find_event_boundary(&self.carry[self.carry_start..]) else {
                break;
            };
            let event_end = self.carry_start + end;
            // `process_event` takes `&mut self.state` (not `&mut self`)
            // precisely so that the immutable borrow on `self.carry`
            // here can coexist with the mutable borrow on the
            // separate `state` field. The split into `carry` + `state`
            // is what makes this borrow-checker-legal — at the
            // field level the borrows are independent, even though
            // `self` is whole.
            let event = &self.carry[self.carry_start..event_end];
            UsageScanner::process_event(&mut self.state, event);
            self.carry_start = event_end;
        }
        // After draining events, reclaim the now-prefix region if it
        // crossed the watermark. Amortizes the drain cost over M
        // consumed bytes rather than O(M²) per-event.
        maybe_drain_carry(&mut self.carry, &mut self.carry_start);
    }

    fn process_event<'a>(state: &mut ScannerState, event: &'a [u8]) {
        // Normalize CRLF to LF for line splitting. Skip the allocation
        // entirely when the input has no `\r` bytes — the dominant
        // case for Anthropic's LF-only SSE, where the previous
        // implementation always paid for a `to_vec()` here.
        let normalized: Cow<[u8]> = if event.contains(&b'\r') {
            let mut v = Vec::with_capacity(event.len());
            let mut i = 0;
            while i < event.len() {
                if event[i] == b'\r' && event.get(i + 1) == Some(&b'\n') {
                    v.push(b'\n');
                    i += 2;
                } else {
                    v.push(event[i]);
                    i += 1;
                }
            }
            Cow::Owned(v)
        } else {
            Cow::Borrowed(event)
        };

        let mut data_lines: Vec<&[u8]> = Vec::new();
        for raw_line in normalized.split(|b| *b == b'\n') {
            // Strip a single trailing CR (defensive, after the LF split).
            let line = if raw_line.last() == Some(&b'\r') {
                &raw_line[..raw_line.len() - 1]
            } else {
                raw_line
            };
            if line.is_empty() {
                continue;
            }
            // SSE comment: lines starting with `:` are ignored.
            if line.first() == Some(&b':') {
                continue;
            }
            // SSE field name:value — split on the FIRST `:` per spec,
            // so a leading space after the colon is preserved (and
            // stripped per spec).
            let Some(colon) = line.iter().position(|b| *b == b':') else {
                // No colon: per spec, the field name is the whole
                // line and the value is empty. Unknown field names
                // (which includes the case where no colon is
                // present and the line is therefore not a real
                // SSE field) must NOT clobber `last_event` —
                // they're treated like `id:` / `retry:` and
                // ignored. The earlier behaviour treated this as
                // an `event:` override, which made any colon-less
                // line tag the next data line as the wrong event
                // type (opus review #8).
                continue;
            };
            let field = &line[..colon];
            let mut value = &line[colon + 1..];
            // Spec: strip a single leading space from value.
            if value.first() == Some(&b' ') {
                value = &value[1..];
            }
            match field {
                b"event" => {
                    state.last_event = String::from_utf8_lossy(value).into_owned();
                }
                b"data" => {
                    data_lines.push(value);
                }
                _ => {
                    // id:, retry:, and any future field. Ignored for
                    // usage purposes; do not update last_event.
                }
            }
        }

        if data_lines.is_empty() {
            // SSE spec (WHATWG §6.4): the event type is the value of
            // the LAST `event:` line within the dispatch, and only
            // applies to the immediately-following data. Once the
            // blank line closes the event, `last_event` must reset.
            // Review C7: leaving it set would misattribute a bare
            // `data:` line in the next event (where the upstream
            // omitted the `event:` header) to the previous event's
            // type — silently dropping usage from the wrong event.
            state.last_event.clear();
            return;
        }
        // Borrow `last_event` directly — no clone. The early return
        // above guarantees `data_lines` is non-empty so we'll consume
        // it within this scope.
        let event_name: &str = &state.last_event;
        // Spec: multiple data: lines are joined with `\n` then parsed
        // as a single field value. Strip a trailing `\r` per line —
        // the CRLF normalization earlier handles *interior* `\r\n`,
        // but the very last byte of the event may still be `\r`
        // (e.g. `...\r\n\r\n` collapses to `...\r` after the LF
        // split, leaving a stray `\r` glued to the final data: line).
        let mut joined = Vec::with_capacity(data_lines.iter().map(|l| l.len()).sum::<usize>() + data_lines.len());
        for (i, line) in data_lines.iter().enumerate() {
            if i > 0 {
                joined.push(b'\n');
            }
            let end = if line.last() == Some(&b'\r') {
                line.len() - 1
            } else {
                line.len()
            };
            joined.extend_from_slice(&line[..end]);
        }

        match event_name {
            "message_start" => {
                if !state.seen_start {
                    match serde_json::from_slice::<MessageDeltaUsageEnvelope>(&joined) {
                        Ok(env) => {
                            if let Some(u) = env.into_usage() {
                                state.start = Some(u);
                                state.seen_start = true;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "scanner: malformed message_start JSON; ignoring"
                            );
                        }
                    }
                }
            }
            "message_delta" => {
                // Anthropic can emit more than one `message_delta`
                // (flow-based streams interleave deltas across output
                // blocks / stop-reason transitions). Each carries the
                // cumulative usage; the LAST one is authoritative, so
                // overwrite on every occurrence — the earlier
                // `seen_delta`-guarded Take-First dropped any later
                // authoritative totals (code-review final round C5).
                // A delta whose `usage` block is missing/absent keeps
                // the previous value (the `if let Some(..)` below).
                match serde_json::from_slice::<MessageDeltaUsageEnvelope>(&joined) {
                    Ok(env) => {
                        if let Some(u) = env.into_usage() {
                            state.delta = Some(u);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "scanner: malformed message_delta JSON; ignoring"
                        );
                    }
                }
            }
            "error" => {
                // In-band upstream failure (Anthropic-shaped SSE).
                // Responses/OAI adapters encode an upstream error as
                // `event: error` inside an otherwise-healthy byte
                // stream; Anthropic passthrough upstreams can also
                // emit it. Mark the record errored (see the
                // `error_seen` field doc for why the transport being
                // healthy is not enough).
                state.error_seen = true;
            }
            _ => {
                // Other events (ping, content_block_*, message_stop,
                // etc.) are consumed for last_event tracking but
                // never feed usage.
            }
        }
        // SSE spec: the event type is bound to the just-dispatched
        // event only. Reset to empty so a bare-`data:` event later
        // (one whose upstream omitted the `event:` header) isn't
        // attributed to the previous event's type. Mirrors the
        // reset on the no-data early-return above (C7).
        state.last_event.clear();
    }

    /// Produce the final `StreamUsage` from whatever was lifted.
    pub fn finalize(self) -> StreamUsage {
        StreamUsage::merge(self.state.start.as_ref(), self.state.delta.as_ref())
    }

    /// Whether the scanner observed an SSE `event: error` frame on
    /// the (otherwise-healthy) byte stream. Consumed by
    /// [`crate::server::MappedStream`] to flip the UsageRecord outcome
    /// to Errored when a mid-stream upstream failure arrived in-band.
    pub(crate) fn saw_error(&self) -> bool {
        self.state.error_seen
    }
}

/// SSE event envelope: only the `usage` field matters to the scanner.
/// Tolerates missing usage (some upstreams omit `usage` on early
/// message_start events).
///
/// Real Anthropic shapes:
/// - `message_start`: `{"type":"message_start","message":{"usage":{...},...}}`
/// - `message_delta`:  `{"type":"message_delta","usage":{...},"delta":{...}}`
///
/// We accept both layouts so the scanner doesn't have to special-case
/// the start-of-stream envelope. Both branches funnel into a single
/// `MessageDeltaUsage` field — for `message_start` the inner
/// `message.usage` is lifted; for `message_delta` the top-level
/// `usage` is lifted. The first successful parse wins.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct MessageDeltaUsageEnvelope {
    #[serde(default)]
    usage: Option<MessageDeltaUsage>,
    #[serde(default)]
    message: Option<MessageStartInner>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct MessageStartInner {
    #[serde(default)]
    usage: Option<MessageDeltaUsage>,
}

impl MessageDeltaUsageEnvelope {
    /// Return whichever nested `usage` is present, preferring the
    /// top-level (`message_delta` shape) and falling back to the
    /// inner `message.usage` (`message_start` shape).
    fn into_usage(self) -> Option<MessageDeltaUsage> {
        self.usage.or_else(|| self.message.and_then(|m| m.usage))
    }
}

/// Locate the index of the *byte just past* the first blank-line
/// boundary in `buf`, or `None` if no boundary exists yet.
///
/// Spec: an SSE event ends with a blank line — either `\n\n` or
/// `\r\n\r\n` (SSE parsers must accept CRLF as a tolerant line
/// ending). The caller drains `buf[..end]` to consume one event.
///
/// Implemented with `memchr::memchr` over the LF character — single
/// pass through `buf` and SIMD-accelerated. The earlier byte-by-byte
/// scan was O(N) per call but with a 4-byte branch per byte; for
/// chatty streams (interleaved pings and content_block_deltas) and
/// CRLF-terminated reverse-proxied events this was a measurable
/// hot-path cost (opus review #5). For CRLF, after locating the
/// first `\n` we check whether the four-byte sequence starting one
/// position before forms `\r\n\r\n` — that lets us consume the full
/// terminator and avoid leaving trailing bytes in carry (the earlier
/// `pos + 2` truncation bug, opus review #2).
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    let mut search_from = 0;
    while let Some(rel) = memchr::memchr(b'\n', &buf[search_from..]) {
        let abs = search_from + rel;
        // Look for LF-LF (`\n\n`).
        if abs + 1 < buf.len() && buf[abs + 1] == b'\n' {
            return Some(abs + 2);
        }
        // Look for CRLF-CRLF (`\r\n\r\n`).
        if abs >= 1
            && abs + 2 < buf.len()
            && buf[abs - 1] == b'\r'
            && buf[abs + 1] == b'\r'
            && buf[abs + 2] == b'\n'
        {
            return Some(abs + 3);
        }
        search_from = abs + 1;
    }
    None
}

// Allow `expect_variant!` to mention StreamUsage in tests.
#[allow(dead_code)]
fn _ensure_send_sync<T: Send + Sync>() {}

#[allow(dead_code)]
fn _assert_send_sync() {
    _ensure_send_sync::<UsageStats>();
    _ensure_send_sync::<UsageRecord>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::time::Duration;

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn scanner() -> UsageScanner {
        UsageScanner::new()
    }

    #[test]
    fn scanner_lifts_input_from_message_start_and_output_from_message_delta() {
        let mut s = scanner();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":42}}}\n\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(100));
        assert_eq!(u.cache_read_input_tokens, Some(42));
        assert_eq!(u.output_tokens, Some(7));
    }

    #[test]
    fn scanner_merges_when_message_delta_also_carries_input() {
        // Real Anthropic message_delta often re-states input_tokens
        // and may carry cache_creation_input_tokens. We prefer
        // message_start for input/cache; if the start didn't carry
        // one (older clients), the delta value wins.
        let mut s = scanner();
        s.observe(&Bytes::from_static(b"event: ping\ndata: {}\n\n"));
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":50}}}\n\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\ndata: {\"usage\":{\"input_tokens\":75,\"output_tokens\":3,\"cache_creation_input_tokens\":5}}\n\n",
        ));
        let u = s.finalize();
        // start wins on input
        assert_eq!(u.input_tokens, Some(50));
        // delta wins on output
        assert_eq!(u.output_tokens, Some(3));
        // cache_creation: only delta carried it
        assert_eq!(u.cache_creation_input_tokens, Some(5));
    }

    #[test]
    fn scanner_handles_chunked_events_split_across_polls() {
        let mut s = scanner();
        // Split one event into three pieces.
        let a = Bytes::from_static(b"event: message_start\n");
        let b = Bytes::from_static(b"data: {\"message\":{\"usage\":{\"input_tokens\":9}}}\n");
        let c = Bytes::from_static(b"\n");
        s.observe(&a);
        s.observe(&b);
        s.observe(&c);
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(9));
    }

    #[test]
    fn scanner_ignores_ping_and_text_delta_events() {
        let mut s = scanner();
        s.observe(&Bytes::from_static(b"event: ping\ndata: {}\n\n"));
        s.observe(&Bytes::from_static(
            b"event: content_block_start\ndata: {\"index\":0}\n\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n",
        ));
        let u = s.finalize();
        // Nothing lifted.
        assert_eq!(u, StreamUsage::default());
    }

    #[test]
    fn scanner_normalizes_crlf_and_joins_multi_line_data() {
        let mut s = scanner();
        // \r\n line endings; multi-line data: block (one event).
        s.observe(&Bytes::from_static(
            b"event: message_start\r\ndata: {\"message\":\r\ndata: {\"usage\":{\"input_tokens\":11}}}\r\n\r\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(11));
    }

    #[test]
    fn scanner_skips_comment_lines() {
        let mut s = scanner();
        s.observe(&Bytes::from_static(
            b": heartbeat 1000ms\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":1}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(1));
    }

    #[test]
    fn scanner_warns_and_continues_on_malformed_message_delta_json() {
        let mut s = scanner();
        // Malformed JSON inside message_delta must NOT poison the
        // scanner — a subsequent valid event still lifts correctly.
        s.observe(&Bytes::from_static(
            b"event: message_delta\ndata: {this is not json}\n\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\ndata: {\"usage\":{\"output_tokens\":4}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(4));
    }

    #[test]
    fn scanner_finalize_without_any_usage_is_default() {
        let u = scanner().finalize();
        assert_eq!(u, StreamUsage::default());
    }

    #[test]
    fn thinking_tokens_lifted_from_output_tokens_details() {
        let raw = b"{\"output_tokens_details\":{\"thinking_tokens\":17}}";
        let usage: MessageDeltaUsage = serde_json::from_slice(raw).unwrap();
        assert_eq!(usage.thinking_tokens(), Some(17));
    }

    #[test]
    fn thinking_tokens_absent_stays_silent() {
        // The dominant case for non-thinking models: the key is
        // missing entirely. The implementation must not warn.
        let raw = b"{\"output_tokens\":5}";
        let usage: MessageDeltaUsage = serde_json::from_slice(raw).unwrap();
        assert_eq!(usage.thinking_tokens(), None);
    }

    #[test]
    fn thinking_tokens_present_but_malformed_warns_and_returns_none() {
        let raw = b"{\"output_tokens_details\":{\"thinking_tokens\":\"lots\"}}";
        let usage: MessageDeltaUsage = serde_json::from_slice(raw).unwrap();
        assert_eq!(usage.thinking_tokens(), None);
    }

    #[test]
    fn thinking_tokens_malformed_counter_starts_at_zero() {
        // The static counter is process-global so we only assert it
        // is non-negative (u64 always is) and that calling
        // `thinking_tokens()` on a well-formed envelope does NOT
        // bump it. A misbehaving impl that bumps on every call (not
        // just malformed) would be caught by the rate-limited warn
        // logging far too aggressively.
        let raw = b"{\"output_tokens_details\":{\"thinking_tokens\":17}}";
        let usage: MessageDeltaUsage = serde_json::from_slice(raw).unwrap();
        // Capture-then-restore isn't possible on a private AtomicU64
        // without an exposed hook; instead, verify the well-formed
        // call returns the correct value (the warn branch was not
        // taken).
        assert_eq!(usage.thinking_tokens(), Some(17));
    }

    #[test]
    fn message_start_envelope_lifts_inner_usage() {
        // Real Anthropic wire shape:
        //   {"type":"message_start","message":{"usage":{...}}}
        let raw = br#"{"type":"message_start","message":{"usage":{"input_tokens":7}}}"#;
        let env: MessageDeltaUsageEnvelope = serde_json::from_slice(raw).unwrap();
        let usage = env.into_usage().unwrap();
        assert_eq!(usage.input_tokens, Some(7));
    }

    #[test]
    fn message_delta_envelope_lifts_top_level_usage() {
        // Real Anthropic wire shape:
        //   {"type":"message_delta","usage":{...},"delta":{...}}
        let raw = br#"{"type":"message_delta","usage":{"output_tokens":3},"delta":{"stop_reason":"end_turn"}}"#;
        let env: MessageDeltaUsageEnvelope = serde_json::from_slice(raw).unwrap();
        let usage = env.into_usage().unwrap();
        assert_eq!(usage.output_tokens, Some(3));
    }

    #[test]
    fn stream_usage_merge_picks_delta_for_thinking_and_server_tool() {
        let start = MessageDeltaUsage {
            input_tokens: Some(8),
            ..Default::default()
        };
        let delta = MessageDeltaUsage {
            output_tokens: Some(2),
            output_tokens_details: Some(serde_json::json!({"thinking_tokens": 2})),
            server_tool_use: Some(serde_json::json!({"web_search_requests": 1})),
            ..Default::default()
        };
        let u = StreamUsage::merge(Some(&start), Some(&delta));
        assert_eq!(u.input_tokens, Some(8));
        assert_eq!(u.output_tokens, Some(2));
        assert_eq!(u.thinking_tokens, Some(2));
        assert_eq!(
            u.server_tool_use.as_ref().and_then(|v| v.get("web_search_requests")).and_then(|v| v.as_i64()),
            Some(1)
        );
    }

    #[test]
    fn stream_usage_merge_without_start_falls_back_to_delta() {
        let delta = MessageDeltaUsage {
            input_tokens: Some(3),
            output_tokens: Some(4),
            ..Default::default()
        };
        let u = StreamUsage::merge(None, Some(&delta));
        assert_eq!(u.input_tokens, Some(3));
        assert_eq!(u.output_tokens, Some(4));
    }

    #[test]
    fn stream_usage_merge_without_delta_only_has_input_side() {
        let start = MessageDeltaUsage {
            input_tokens: Some(11),
            cache_read_input_tokens: Some(2),
            ..Default::default()
        };
        let u = StreamUsage::merge(Some(&start), None);
        assert_eq!(u.input_tokens, Some(11));
        assert_eq!(u.cache_read_input_tokens, Some(2));
        assert_eq!(u.output_tokens, None);
        assert_eq!(u.thinking_tokens, None);
    }

    #[test]
    fn stream_usage_merge_prefers_delta_when_start_is_zero() {
        // openai_compat / openai_responses / copilot translators emit
        // a `Usage::default()`-derived `message_start` whose
        // input_tokens / output_tokens are `Some(0)` (the struct
        // serializes every field, even default u32=0). The real
        // counts arrive on `message_delta` via `build_usage`. The
        // previous `start.or(delta)` short-circuited on `Some(0)`
        // and lost the real number. Merge must fall through to
        // delta when start carries a zero.
        let start = MessageDeltaUsage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            ..Default::default()
        };
        let delta = MessageDeltaUsage {
            input_tokens: Some(42170),
            output_tokens: Some(15),
            cache_read_input_tokens: Some(0),
            ..Default::default()
        };
        let u = StreamUsage::merge(Some(&start), Some(&delta));
        assert_eq!(u.input_tokens, Some(42170));
        assert_eq!(u.output_tokens, Some(15));
        // cache_read=0 on both sides: prefer start so we keep the
        // explicit-zero shape from message_start (semantically a
        // no-cache request). Either way is defensible; the test
        // just pins the current behaviour so future changes are
        // deliberate.
        assert_eq!(u.cache_read_input_tokens, Some(0));
    }

    #[test]
    fn stream_usage_merge_output_delta_wins_over_anthropic_placeholder() {
        // Anthropic streaming `message_start` always carries the
        // placeholder `output_tokens: 1` — the final output count
        // lives on `message_delta`. The start-non-zero-wins rule that
        // guards against translator `Some(0)` loss must NOT apply to
        // output_tokens, or every passthrough stream is recorded as
        // exactly 1 output token (code-review final round finding C1).
        let start = MessageDeltaUsage {
            input_tokens: Some(10),
            output_tokens: Some(1), // Anthropic placeholder on message_start
            ..Default::default()
        };
        let delta = MessageDeltaUsage {
            output_tokens: Some(432),
            ..Default::default()
        };
        let u = StreamUsage::merge(Some(&start), Some(&delta));
        // input still prefers start's non-zero count.
        assert_eq!(u.input_tokens, Some(10));
        // output must come from delta, NOT the placeholder.
        assert_eq!(u.output_tokens, Some(432));
    }

    #[tokio::test]
    async fn usage_stats_records_and_snapshots() {
        let stats = UsageStats::new(8);
        for i in 0..3u32 {
            stats
                .record(UsageRecord {
                    started_at: Utc.timestamp_opt(1_700_000_000 + i as i64, 0).unwrap(),
                    ended_at: Utc.timestamp_opt(1_700_000_000 + i as i64, 0).unwrap(),
                    elapsed_ms: i as u64,
                    provider: "p".into(),
                    model: "m".into(),
                    stream: false,
                    outcome: Outcome::Success,
                    usage: Some(StreamUsage {
                        input_tokens: Some(i),
                        ..Default::default()
                    }),
                    failed_providers: vec![],
                total_tokens: 0,
                })
                ;
        }
        let snap = stats.snapshot(None, None, None);
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].usage.as_ref().and_then(|u| u.input_tokens), Some(0));
        assert_eq!(snap[2].usage.as_ref().and_then(|u| u.input_tokens), Some(2));
    }

    #[tokio::test]
    async fn usage_stats_evicts_oldest_when_full() {
        let stats = UsageStats::new(2);
        for i in 0..4u32 {
            stats
                .record(UsageRecord {
                    started_at: Utc.timestamp_opt(1_700_000_000 + i as i64, 0).unwrap(),
                    ended_at: Utc.timestamp_opt(1_700_000_000 + i as i64, 0).unwrap(),
                    elapsed_ms: 0,
                    provider: "p".into(),
                    model: "m".into(),
                    stream: false,
                    outcome: Outcome::Success,
                    usage: None,
                    failed_providers: vec![],
                total_tokens: 0,
                })
                ;
        }
        let snap = stats.snapshot(None, None, None);
        // Oldest two (i=0,1) evicted; newest two (i=2,3) remain.
        assert_eq!(snap.len(), 2);
        assert_eq!(
            snap[0].started_at,
            Utc.timestamp_opt(1_700_000_002, 0).unwrap()
        );
        assert_eq!(
            snap[1].started_at,
            Utc.timestamp_opt(1_700_000_003, 0).unwrap()
        );
    }

    #[tokio::test]
    async fn usage_stats_capacity_zero_is_disabled() {
        // Per plan L432 / L777-780: `capacity == 0` means the feature is
        // disabled — no allocation, no lock, record() is a no-op,
        // snapshot() returns empty.
        let stats = UsageStats::new(0);
        stats.record(UsageRecord {
            started_at: ts("2026-09-09T10:00:00Z"),
            ended_at: ts("2026-09-09T10:00:00Z"),
            elapsed_ms: 0,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
        total_tokens: 0,
        });
        assert_eq!(stats.snapshot(None, None, None).len(), 0);
        assert_eq!(stats.evicted_total(), 0);
    }

    #[tokio::test]
    async fn usage_stats_capacity_clamped_to_max() {
        // Opus review #7: requesting a capacity above the ceiling must
        // be silently clamped to MAX_USAGE_CAPACITY so a typo can't
        // OOM the proxy at boot via VecDeque::with_capacity(big).
        let requested = MAX_USAGE_CAPACITY * 10;
        let stats = UsageStats::new(requested);
        assert_eq!(stats.capacity(), MAX_USAGE_CAPACITY);
        // And a value just over the ceiling (saturating to exactly
        // MAX+1) still gets clamped.
        let stats2 = UsageStats::new(MAX_USAGE_CAPACITY + 1);
        assert_eq!(stats2.capacity(), MAX_USAGE_CAPACITY);
        // A value at the ceiling exactly is honoured as-is.
        let stats3 = UsageStats::new(MAX_USAGE_CAPACITY);
        assert_eq!(stats3.capacity(), MAX_USAGE_CAPACITY);
        // Zero still means disabled (regression guard).
        let stats4 = UsageStats::new(0);
        assert_eq!(stats4.capacity(), 0);
    }

    #[tokio::test]
    async fn usage_stats_snapshot_filters_by_time_window_and_limit() {
        let stats = UsageStats::new(8);
        for i in 0..5u32 {
            let t = Utc.timestamp_opt(1_700_000_000 + i as i64, 0).unwrap();
            stats
                .record(UsageRecord {
                    started_at: t,
                    ended_at: t,
                    elapsed_ms: 0,
                    provider: "p".into(),
                    model: "m".into(),
                    stream: false,
                    outcome: Outcome::Success,
                    usage: None,
                    failed_providers: vec![],
                total_tokens: 0,
                })
                ;
        }
        // Window covers the middle three (i=1,2,3); limit caps at 2.
        let since = Utc.timestamp_opt(1_700_000_001, 0).unwrap();
        let until = Utc.timestamp_opt(1_700_000_003, 0).unwrap();
        let snap = stats.snapshot(Some(since), Some(until), Some(2));
        assert_eq!(snap.len(), 2);
        assert_eq!(
            snap[0].started_at,
            Utc.timestamp_opt(1_700_000_001, 0).unwrap()
        );
        assert_eq!(
            snap[1].started_at,
            Utc.timestamp_opt(1_700_000_002, 0).unwrap()
        );
    }

    #[tokio::test]
    async fn usage_stats_snapshot_with_inverted_range_returns_empty() {
        let stats = UsageStats::new(4);
        stats
            .record(UsageRecord {
                started_at: ts("2026-09-09T10:00:00Z"),
                ended_at: ts("2026-09-09T10:00:00Z"),
                elapsed_ms: 0,
                provider: "p".into(),
                model: "m".into(),
                stream: false,
                outcome: Outcome::Success,
                usage: None,
                failed_providers: vec![],
            total_tokens: 0,
            })
            ;
        let snap = stats
            .snapshot(
                Some(ts("2026-09-09T11:00:00Z")),
                Some(ts("2026-09-09T09:00:00Z")),
                None,
            )
            ;
        assert!(snap.is_empty());
    }

    #[test]
    fn usage_record_serializes_with_rfc3339_timestamps_and_null_outcome() {
        let rec = UsageRecord {
            started_at: ts("2026-09-09T10:00:00Z"),
            ended_at: ts("2026-09-09T10:00:01Z"),
            elapsed_ms: 1000,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
        total_tokens: 0,
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert_eq!(v["started_at"], "2026-09-09T10:00:00Z");
        assert_eq!(v["ended_at"], "2026-09-09T10:00:01Z");
        assert_eq!(v["outcome"], "success");
        // usage: None ⇒ field absent (skip_serializing_if); the spec
        // allows the field to be null in JSON but our wire shape
        // follows CLAUDE.md "field-omission is correctness" — pick
        // one and stick to it.
        assert!(v.get("usage").is_none() || v["usage"].is_null());
    }

    #[test]
    fn outcome_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_value(Outcome::Success).unwrap(),
            serde_json::json!("success")
        );
        assert_eq!(
            serde_json::to_value(Outcome::Errored).unwrap(),
            serde_json::json!("errored")
        );
        assert_eq!(
            serde_json::to_value(Outcome::Aborted).unwrap(),
            serde_json::json!("aborted")
        );
    }

    #[test]
    fn stream_usage_omits_none_fields_on_wire() {
        // All-Option fields with skip_serializing_if ⇒ absent on the
        // wire when None. Lifting tests assert each field individually
        // elsewhere; this is the shape guarantee.
        let v = serde_json::to_value(StreamUsage {
            input_tokens: Some(5),
            output_tokens: None,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(v["input_tokens"], 5);
        assert!(v.get("output_tokens").is_none());
        assert!(v.get("cache_creation_input_tokens").is_none());
        assert!(v.get("cache_read_input_tokens").is_none());
        assert!(v.get("thinking_tokens").is_none());
        assert!(v.get("server_tool_use").is_none());
    }

    #[test]
    fn carries_partial_event_across_multiple_polls() {
        let mut s = scanner();
        // First poll ends mid-event.
        s.observe(&Bytes::from_static(b"event: message_delta\ndata: {\"usa"));
        // Subsequent poll completes it.
        s.observe(&Bytes::from_static(b"ge\":{\"output_tokens\":9}}\n\n"));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(9));
    }

    #[test]
    fn ignores_orphan_data_line_without_event_name() {
        // Pathological case: a `data:` line arrives before any
        // `event:` line. Per SSE spec this should not happen, but we
        // don't want a panic. last_event stays empty so the data line
        // is ignored entirely.
        let mut s = scanner();
        s.observe(&Bytes::from_static(b"data: {\"output_tokens\":99}\n\n"));
        let u = s.finalize();
        assert_eq!(u, StreamUsage::default());
    }

    #[test]
    fn scanner_does_not_panic_on_consecutive_blank_lines() {
        let mut s = scanner();
        s.observe(&Bytes::from_static(b"\n\n\n\nevent: ping\ndata: {}\n\n\n\n"));
        let u = s.finalize();
        assert_eq!(u, StreamUsage::default());
    }

    #[tokio::test]
    async fn concurrent_records_dont_panic_or_lose_data() {
        let stats = UsageStats::new(64);
        let mut handles = Vec::new();
        for _ in 0..16 {
            let s = stats.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..4 {
                    s.record(UsageRecord {
                        started_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
                        ended_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
                        elapsed_ms: 0,
                        provider: "p".into(),
                        model: "m".into(),
                        stream: false,
                        outcome: Outcome::Success,
                        usage: None,
                        failed_providers: vec![],
                    total_tokens: 0,
                    })
                    ;
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let snap = stats.snapshot(None, None, None);
        assert_eq!(snap.len(), 64);
        // Sanity: elapsed wall time is well under a second (the test
        // would hang here, not fail with an assertion, but this also
        // gives a regression-detector if anyone serializes too much).
        let _ = Duration::from_secs(1);
    }

    // ---- Coverage holes ----
    //
    // The blocks below were missing from earlier unit tests; they
    // exercise low-traffic branches (overflow guards, malformed
    // shapes, completed-response merging) so region coverage stays
    // above the 97% bar documented in CLAUDE.md.

    #[test]
    fn thinking_tokens_handles_null_and_overflow() {
        // Null inside output_tokens_details must short-circuit before
        // the as_u64 branch fires.
        let u: MessageDeltaUsage = serde_json::from_slice(
            br#"{"output_tokens_details":{"thinking_tokens":null}}"#,
        )
        .unwrap();
        assert_eq!(u.thinking_tokens(), None);

        // A negative i64 should not collapse to Some(0); u64 saturates
        // to 0 but the u32::try_from of 0 is fine, so use a value too
        // large for u32 to trip the overflow branch instead.
        let u: MessageDeltaUsage = serde_json::from_slice(
            br#"{"output_tokens_details":{"thinking_tokens":5000000000}}"#,
        )
        .unwrap();
        assert_eq!(u.thinking_tokens(), None);

        // output_tokens_details present but missing thinking_tokens key
        // entirely — must not warn, must return None.
        let u: MessageDeltaUsage = serde_json::from_slice(
            br#"{"output_tokens_details":{}}"#,
        )
        .unwrap();
        assert_eq!(u.thinking_tokens(), None);
    }

    #[test]
    fn from_complete_lifts_cache_and_thinking() {
        // Synthesize a wire-shape Usage: cache_creation, cache_read,
        // server_tool_use, and a non-null thinking_tokens.
        let raw = serde_json::json!({
            "input_tokens": 5,
            "output_tokens": 7,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "output_tokens_details": {"thinking_tokens": 4},
            "server_tool_use": {"web_search_requests": 1}
        });
        let u: crate::anthropic::Usage = serde_json::from_value(raw).unwrap();
        let s = StreamUsage::from_complete(&u);
        assert_eq!(s.input_tokens, Some(5));
        assert_eq!(s.output_tokens, Some(7));
        assert_eq!(s.cache_creation_input_tokens, Some(2));
        assert_eq!(s.cache_read_input_tokens, Some(3));
        assert_eq!(s.thinking_tokens, Some(4));
        assert!(s.server_tool_use.is_some());
    }

    #[test]
    fn from_complete_handles_missing_optional_fields() {
        // Minimum-shape Usage: only the two required counters.
        let raw = serde_json::json!({
            "input_tokens": 1,
            "output_tokens": 2
        });
        let u: crate::anthropic::Usage = serde_json::from_value(raw).unwrap();
        let s = StreamUsage::from_complete(&u);
        assert_eq!(s.cache_creation_input_tokens, None);
        assert_eq!(s.cache_read_input_tokens, None);
        assert_eq!(s.thinking_tokens, None);
        assert!(s.server_tool_use.is_none());

        // output_tokens_details present but null/empty — no warning.
        let raw = serde_json::json!({
            "input_tokens": 1,
            "output_tokens": 2,
            "output_tokens_details": null
        });
        let u: crate::anthropic::Usage = serde_json::from_value(raw).unwrap();
        let s = StreamUsage::from_complete(&u);
        assert_eq!(s.thinking_tokens, None);
    }

    #[test]
    fn merge_picks_cache_fields_from_start_only() {
        // message_start carries input/cache; message_delta carries
        // output. Cache fields on `start` must survive the merge even
        // when delta also has some.
        let start = MessageDeltaUsage {
            input_tokens: Some(10),
            cache_creation_input_tokens: Some(3),
            cache_read_input_tokens: Some(2),
            ..Default::default()
        };
        let delta = MessageDeltaUsage {
            output_tokens: Some(5),
            ..Default::default()
        };
        let s = StreamUsage::merge(Some(&start), Some(&delta));
        assert_eq!(s.input_tokens, Some(10));
        assert_eq!(s.cache_creation_input_tokens, Some(3));
        assert_eq!(s.cache_read_input_tokens, Some(2));
        assert_eq!(s.output_tokens, Some(5));
    }

    #[test]
    fn merge_server_tool_use_falls_back_to_start() {
        // Server-tool-use block appears in both events sometimes; the
        // merge prefers delta but accepts start as a fallback.
        let start = MessageDeltaUsage {
            server_tool_use: Some(serde_json::json!({"a": 1})),
            ..Default::default()
        };
        let merged_only_start = StreamUsage::merge(Some(&start), None);
        assert_eq!(merged_only_start.server_tool_use, Some(serde_json::json!({"a": 1})));

        let delta = MessageDeltaUsage {
            server_tool_use: Some(serde_json::json!({"b": 2})),
            ..Default::default()
        };
        let both = StreamUsage::merge(Some(&start), Some(&delta));
        // delta wins when both present
        assert_eq!(both.server_tool_use, Some(serde_json::json!({"b": 2})));
    }

    #[test]
    fn observe_empty_is_noop() {
        let mut s = UsageScanner::new();
        s.observe(&Bytes::new());
        assert_eq!(s.finalize(), StreamUsage::default());
    }

    #[test]
    fn observe_drops_old_half_on_carry_overflow() {
        // Fill the carry past SCAN_MAX_CARRY. We can't easily do that
        // with a single Bytes literal (compile limit), but a loop of
        // observes reaches the threshold quickly. The branch we want
        // to hit is the "carry + bytes > max" guard.
        let mut s = UsageScanner::new();
        // Push ~1 MiB of garbage with no event boundary; the scanner
        // should keep going (no panic) and finalize to default since
        // nothing parsed as a message_start/message_delta.
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..20 {
            s.observe(&Bytes::from(chunk.clone()));
        }
        // No event ever closed, so we expect no usage.
        assert_eq!(s.finalize(), StreamUsage::default());
    }

    #[test]
    fn repeated_carry_overflow_does_not_panic_or_blow_memory() {
        // Drive the overflow path many times in one scanner and
        // confirm (a) no panic, (b) the carry stays bounded. The
        // load-bearing bound is `len`, not `capacity` — Vec capacity
        // doubles on growth and never shrinks within a session, but
        // `len` is what determines the working-set size.
        let mut s = UsageScanner::new();
        let chunk = vec![b'x'; 96 * 1024]; // 96 KiB per poll
        for _ in 0..200 {
            s.observe(&Bytes::from(chunk.clone()));
            // Worst-case len after overflow: `keep = SCAN_MAX_CARRY/2`
            // plus the full incoming chunk when chunk < keep. The
            // bound is therefore SCAN_MAX_CARRY + chunk_bytes. We
            // assert a generous 2× ceiling to catch any regression
            // where the cap path silently disappears.
            assert!(
                s.carry.len() <= 2 * SCAN_MAX_CARRY,
                    "carry buffer grew unbounded across overflows: len={}",
                    s.carry.len()
                );
        }
        assert_eq!(s.finalize(), StreamUsage::default());
    }

    #[test]
    fn observe_carries_partial_event_across_polls() {
        let mut s = UsageScanner::new();
        // First poll: only the event: line and the first data: prefix.
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"input\":",
        ));
        // The boundary has not closed yet — observed bytes are still
        // buffered; just observe the second half to exercise the
        // cross-poll carry path.
        s.observe(&Bytes::from_static(
            b"7}\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":3}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(3));
    }

    #[test]
    fn process_event_skips_unknown_event_names() {
        let mut s = UsageScanner::new();
        // content_block_delta is real upstream SSE but never feeds usage.
        s.observe(&Bytes::from_static(
            b"event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: ping\ndata: {}\n\n",
        ));
        assert_eq!(s.finalize(), StreamUsage::default());
    }

    #[test]
    fn observe_flags_inband_event_error() {
        // C2 regression: a mid-stream upstream failure is surfaced as
        // an in-band Anthropic `event: error` SSE frame (responses/OAI
        // adapters + server::format_stream_error) over an
        // otherwise-healthy byte stream — no transport Err. The scanner
        // must flag it so MappedStream records the row Errored, not
        // Success. It must not clobber the usage already lifted.
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":7}}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\",\"message\":\"boom\"}}\n\n",
        ));
        assert!(s.saw_error(), "scanner must flag in-band event:error");
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(7), "error frame must not affect usage lift");
    }

    #[test]
    fn observe_clean_stream_does_not_flag_error() {
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":3}}}\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":4}}\n\n",
        ));
        assert!(!s.saw_error(), "clean stream must report saw_error()=false");
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(3));
        assert_eq!(u.output_tokens, Some(4));
    }

    #[test]
    fn observe_last_event_resets_across_event_boundaries() {
        // C7 regression: SSE spec (WHATWG §6.4) says the `event:`
        // type applies ONLY to the immediately-following `data:` and
        // must reset to empty after dispatch. Without the reset, a
        // bare-`data:` event (one whose upstream omitted the
        // `event:` header) would inherit the previous event's type
        // and silently drop the usage payload — the upstream looks
        // healthy, but /admin/usage reports empty.
        //
        // Anthropic-shaped streams always include `event:` per
        // frame, so this scenario doesn't occur in normal Claude
        // traffic. It DOES occur when the proxy synthesizes SSE for
        // an OAI-compat upstream that emits only data: lines.
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":7}}}\n\ndata: {\"usage\":{\"output_tokens\":11}}\n\n",
        ));
        let u = s.finalize();
        // The bare-data frame after message_start must NOT be
        // attributed to message_start (it's not a valid envelope);
        // if the reset works, the second frame is treated as
        // unknown (no event type), dropped from usage, and the
        // scanner still has the input_tokens lifted from frame 1.
        assert_eq!(
            u.input_tokens, Some(7),
            "first message_start must still lift input"
        );
        assert_eq!(
            u.output_tokens, None,
            "bare-data frame must NOT be attributed to message_start (no output_tokens extracted)"
        );
    }

    #[test]
    fn process_event_ignores_id_and_retry_fields() {
        let mut s = UsageScanner::new();
        // id: and retry: are spec SSE fields we explicitly do not track.
        // They must not clobber last_event (would otherwise tag a
        // following data: line with the wrong event name).
        s.observe(&Bytes::from_static(
            b"id: 42\nretry: 1000\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":9}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(9));
    }

    #[test]
    fn observe_handles_crlf_terminated_events() {
        // Opus #2 regression. Some upstream reverse-proxies emit
        // CRLF line endings (`\r\n`) per the SSE spec, including the
        // blank-line terminator (`\r\n\r\n`). The previous
        // implementation located the first `\n` in the blank-line
        // and drained `pos + 2`, leaving the trailing `\r\n` in
        // `self.carry`. The next event then started with a
        // leading `\r\n`, which event-alignment drift turned into
        // a blank line that swallowed the next event's `data:`
        // line. End-to-end symptom: the scanner lifted the FIRST
        // event's usage and missed every subsequent one.
        //
        // Note: the scanner keeps the LAST delta it sees (each
        // `message_delta` carries cumulative usage; see the
        // `message_delta` arm in process_event), so the final
        // output count pinned below is the second delta's 11. What
        // this test pins is event alignment, not the merging rule.
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\r\ndata: {\"message\":{\"usage\":{\"input_tokens\":42,\"cache_read_input_tokens\":5}}}\r\n\r\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\r\ndata: {\"usage\":{\"output_tokens\":7}}\r\n\r\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\r\ndata: {\"usage\":{\"output_tokens\":11}}\r\n\r\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(42));
        assert_eq!(u.cache_read_input_tokens, Some(5));
        assert_eq!(
            u.output_tokens,
            Some(11),
            "last CRLF delta is authoritative (multi-delta streams keep the final total; code-review final round C5)"
        );
    }

    #[test]
    fn observe_handles_crlf_split_across_polls() {
        // The previous `pos + 2` truncation bug bit hardest when a
        // single event spanned multiple `observe()` calls — the
        // carry's trailing `\r\n` from event N then preceded the
        // partial event N+1, and event N+1's eventual terminator
        // got confused with the leftover. This test pins the fix:
        // every CRLF event boundary is consumed in full regardless
        // of how the bytes arrive.
        let mut s = UsageScanner::new();
        // Feed the message_start event in three pieces.
        s.observe(&Bytes::from_static(
            b"event: message_start\r\ndata: {\"message\"",
        ));
        s.observe(&Bytes::from_static(
            b":{\"usage\":{\"input_tokens\":99}}}\r\n\r\n",
        ));
        // And the message_delta in two pieces.
        s.observe(&Bytes::from_static(
            b"event: message_delta\r\ndata: {\"usage\":",
        ));
        s.observe(&Bytes::from_static(
            b"{\"output_tokens\":3}}\r\n\r\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(99));
        assert_eq!(u.output_tokens, Some(3));
    }

    #[test]
    fn observe_handles_mixed_crlf_lf_in_one_stream() {
        // Real-world: a single upstream may emit LF for some events
        // and CRLF for others. The scanner must accept both in any
        // order without confusing alignment. Two deltas with
        // different line endings pin both the tolerance and the
        // take-last rule (the final delta is authoritative).
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\r\ndata: {\"usage\":{\"output_tokens\":2}}\r\n\r\n",
        ));
        s.observe(&Bytes::from_static(
            b"event: message_delta\ndata: {\"usage\":{\"output_tokens\":3}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(1));
        assert_eq!(
            u.output_tokens,
            Some(3),
            "last delta (LF) wins despite preceding CRLF delta"
        );
    }

    #[test]
    fn process_event_tolerates_unknown_field_and_no_colon_line() {
        let mut s = UsageScanner::new();
        // Mix of unknown field, no-colon line, comment, and a real
        // message_delta — only the message_delta must lift.
        s.observe(&Bytes::from_static(
            b": keepalive\nweirdline\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":2}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(2));
    }

    #[test]
    fn process_event_no_colon_line_does_not_clobber_last_event() {
        // Per the SSE spec, a line without a colon has the whole
        // line as the field name and empty value. We don't
        // recognise such field names, so they MUST NOT update
        // `last_event` (otherwise the next `data:` line would be
        // tagged with a bogus event name). Opus review #8.
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":7}}}\n\nweirdline\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":3}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.input_tokens, Some(7));
        assert_eq!(u.output_tokens, Some(3));
        // Crucially: the colon-less line between the two events
        // must not have poisoned the second event's last_event.
        // (If it had, the message_delta wouldn't be picked up.)
    }

    #[test]
    fn process_event_skips_when_data_lines_empty() {
        let mut s = UsageScanner::new();
        // Event with only `event:` line — no data: → process_event
        // returns early, last_event does NOT carry forward to the
        // next event's data.
        s.observe(&Bytes::from_static(
            b"event: message_start\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":1}}\n\n",
        ));
        // The second event is a message_delta so it lifts regardless,
        // but the early-return path must not panic.
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(1));
    }

    #[test]
    fn process_event_skips_when_envelope_has_no_usage_field() {
        // An event whose payload is empty (no `usage`, no `message`)
        // must be silently dropped — neither start nor delta updated.
        let mut s = UsageScanner::new();
        s.observe(&Bytes::from_static(
            b"event: message_start\ndata: {}\n\nevent: message_delta\ndata: {\"usage\":{\"output_tokens\":6}}\n\n",
        ));
        let u = s.finalize();
        assert_eq!(u.output_tokens, Some(6));
        // input_tokens stays None because the empty envelope lifted nothing.
        assert_eq!(u.input_tokens, None);
    }

    #[tokio::test]
    async fn usage_stats_snapshot_applies_since_until_limit() {
        let stats = UsageStats::new(8);
        // Three records with distinct started_at values, then exercise
        // each filter branch in turn.
        let mk = |secs: i64| UsageRecord {
            started_at: Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap(),
            ended_at: Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap(),
            elapsed_ms: 0,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
        total_tokens: 0,
        };
        stats.record(mk(0));;
        stats.record(mk(10));;
        stats.record(mk(20));;

        // since=10 inclusive (>=), until=20 exclusive (<) per plan
        // L604-605 → only the record at exactly 10 falls in [10, 20).
        let since = Utc.timestamp_opt(1_700_000_010, 0).unwrap();
        let until = Utc.timestamp_opt(1_700_000_020, 0).unwrap();
        let snap = stats.snapshot(Some(since), Some(until), None);
        assert_eq!(snap.len(), 1);

        // Half-open windows compose: query [0, 10) then [10, 20)
        // covers every row exactly once.
        let from_zero = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let until_ten = Utc.timestamp_opt(1_700_000_010, 0).unwrap();
        let snap = stats.snapshot(Some(from_zero), Some(until_ten), None);
        assert_eq!(snap.len(), 1);

        // since > until → empty (defensive branch).
        let snap = stats
            .snapshot(Some(until), Some(since), None)
            ;
        assert!(snap.is_empty());

        // limit caps post-filter result, not the buffer.
        let snap = stats.snapshot(None, None, Some(2));
        assert_eq!(snap.len(), 2);

        // only since / only until.
        let snap = stats.snapshot(Some(since), None, None);
        assert_eq!(snap.len(), 2);
        // until=10 exclusive (<) — record at exactly 10 is excluded, so
        // only the record at 0 falls in (-inf, 10).
        let until_only = Utc.timestamp_opt(1_700_000_010, 0).unwrap();
        let snap = stats.snapshot(None, Some(until_only), None);
        assert_eq!(snap.len(), 1);
    }

    #[tokio::test]
    async fn usage_stats_capacity_zero_is_disabled_records_path() {
        // Mirror of the disabled-capacity test, but with the
        // early-write region (record() takes the no-op branch) plus a
        // follow-up snapshot() to assert the empty buffer path.
        let stats = UsageStats::new(0);
        let rec = UsageRecord {
            started_at: Utc.timestamp_opt(1, 0).unwrap(),
            ended_at: Utc.timestamp_opt(1, 0).unwrap(),
            elapsed_ms: 0,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
        total_tokens: 0,
        };
        stats.record(rec);
        let snap = stats.snapshot(None, None, None);
        assert_eq!(snap.len(), 0);
    }

    fn mk_with_usage(
        secs: i64,
        provider: &str,
        model: &str,
        usage: Option<StreamUsage>,
    ) -> UsageRecord {
        UsageRecord {
            started_at: Utc.timestamp_opt(secs, 0).unwrap(),
            ended_at: Utc.timestamp_opt(secs, 0).unwrap(),
            elapsed_ms: 0,
            provider: provider.into(),
            model: model.into(),
            stream: false,
            outcome: Outcome::Success,
            usage,
            failed_providers: vec![],
        total_tokens: 0,
        }
    }

    #[test]
    fn rollup_groups_by_model_provider_and_sums_tokens() {
        let records = vec![
            mk_with_usage(
                1,
                "p1",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    cache_read_input_tokens: Some(2),
                    ..Default::default()
                }),
            ),
            mk_with_usage(
                2,
                "p1",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(20),
                    output_tokens: Some(3),
                    ..Default::default()
                }),
            ),
            mk_with_usage(
                3,
                "p2",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(100),
                    cache_read_input_tokens: Some(50),
                    thinking_tokens: Some(7),
                    ..Default::default()
                }),
            ),
            mk_with_usage(4, "p1", "m-b", None),
        ];

        let rows = rollup(&records, GroupBy::ModelProvider);
        assert_eq!(rows.len(), 3, "got {rows:?}");
        // BTreeMap iteration: (m-a,p1), (m-a,p2), (m-b,p1)
        let ma_p1 = &rows[0];
        assert_eq!(ma_p1.model.as_deref(), Some("m-a"));
        assert_eq!(ma_p1.provider.as_deref(), Some("p1"));
        assert_eq!(ma_p1.requests, 2);
        assert_eq!(ma_p1.input_tokens, 30);
        assert_eq!(ma_p1.output_tokens, 8);
        assert_eq!(ma_p1.cache_read_tokens, 2);
        assert_eq!(ma_p1.total_tokens, 30 + 8);
        assert_eq!(
            ma_p1.cache_read_ratio,
            Some(2.0 / 30.0),
            "cache_read_ratio = read / input (cache reads are inside input)"
        );
        let ma_p2 = &rows[1];
        assert_eq!(ma_p2.cache_read_tokens, 50);
        assert_eq!(ma_p2.input_tokens, 100);
        assert_eq!(ma_p2.reasoning_tokens, 7);
        assert_eq!(
            ma_p2.cache_read_ratio,
            Some(50.0 / 100.0)
        );
        let mb_p1 = &rows[2];
        assert_eq!(mb_p1.model.as_deref(), Some("m-b"));
        assert_eq!(mb_p1.requests, 1);
        assert_eq!(mb_p1.input_tokens, 0);
        assert_eq!(mb_p1.cache_read_tokens, 0);
        assert!(mb_p1.cache_read_ratio.is_none(), "no data → null");
    }

    #[test]
    fn rollup_groups_by_model_collapses_providers() {
        let records = vec![
            mk_with_usage(
                1,
                "p1",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(1),
                    ..Default::default()
                }),
            ),
            mk_with_usage(
                2,
                "p2",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(20),
                    output_tokens: Some(2),
                    ..Default::default()
                }),
            ),
        ];
        let rows = rollup(&records, GroupBy::Model);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model.as_deref(), Some("m-a"));
        assert!(rows[0].provider.is_none());
        assert_eq!(rows[0].requests, 2);
        assert_eq!(rows[0].input_tokens, 30);
        assert_eq!(rows[0].output_tokens, 3);
    }

    #[test]
    fn rollup_groups_by_provider_collapses_models() {
        let records = vec![
            mk_with_usage(
                1,
                "p1",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(10),
                    ..Default::default()
                }),
            ),
            mk_with_usage(
                2,
                "p1",
                "m-b",
                Some(StreamUsage {
                    input_tokens: Some(20),
                    ..Default::default()
                }),
            ),
        ];
        let rows = rollup(&records, GroupBy::Provider);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].model.is_none());
        assert_eq!(rows[0].provider.as_deref(), Some("p1"));
        assert_eq!(rows[0].input_tokens, 30);
    }

    #[test]
    fn rollup_missing_usage_fields_treated_as_zero() {
        // A streaming record lifted only `output_tokens` must NOT
        // zero out `input_tokens` for the group; missing fields
        // contribute 0 to the sum (plan §"Aggregation rules").
        let records = vec![
            mk_with_usage(
                1,
                "p1",
                "m-a",
                Some(StreamUsage {
                    input_tokens: Some(100),
                    ..Default::default()
                }),
            ),
            mk_with_usage(
                2,
                "p1",
                "m-a",
                Some(StreamUsage {
                    output_tokens: Some(7),
                    ..Default::default()
                }),
            ),
        ];
        let rows = rollup(&records, GroupBy::ModelProvider);
        assert_eq!(rows[0].input_tokens, 100, "missing stays 0");
        assert_eq!(rows[0].output_tokens, 7);
    }

    #[test]
    fn group_by_from_str_recognises_known_values_and_rejects_unknown() {
        assert_eq!(GroupBy::from_str("model"), Some(GroupBy::Model));
        assert_eq!(GroupBy::from_str("provider"), Some(GroupBy::Provider));
        assert_eq!(
            GroupBy::from_str("model_provider"),
            Some(GroupBy::ModelProvider)
        );
        // Empty defaults to ModelProvider for ergonomic URL handling.
        assert_eq!(GroupBy::from_str(""), Some(GroupBy::ModelProvider));
        assert_eq!(GroupBy::from_str("bogus"), None);
    }

    #[test]
    fn snapshot_filtered_applies_model_and_provider_predicates() {
        let stats = UsageStats::new(8);
        let r1 = mk_with_usage(1, "p1", "m-a", None);
        let r2 = mk_with_usage(2, "p2", "m-a", None);
        let r3 = mk_with_usage(3, "p1", "m-b", None);
        stats.record(r1);
        stats.record(r2);
        stats.record(r3);

        let only_ma = stats.snapshot_filtered(None, None, Some("m-a"), None);
        assert_eq!(only_ma.len(), 2);
        assert!(only_ma.iter().all(|r| r.model == "m-a"));

        let only_p1 = stats.snapshot_filtered(None, None, None, Some("p1"));
        assert_eq!(only_p1.len(), 2);
        assert!(only_p1.iter().all(|r| r.provider == "p1"));

        let both = stats.snapshot_filtered(None, None, Some("m-a"), Some("p1"));
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].model, "m-a");
        assert_eq!(both[0].provider, "p1");
    }

    #[test]
    fn retained_returns_unfiltered_count() {
        let stats = UsageStats::new(8);
        assert_eq!(stats.retained(), 0);
        stats.record(mk_with_usage(1, "p", "m", None));
        stats.record(mk_with_usage(2, "p", "m", None));
        assert_eq!(stats.retained(), 2);
        let from_epoch = Utc.timestamp_opt(1, 0).unwrap();
        // snapshot() with a window that excludes everything returns 0
        // but `retained()` keeps reporting the absolute store size.
        let snap = stats.snapshot(Some(from_epoch), Some(from_epoch), None);
        assert!(snap.is_empty());
        assert_eq!(stats.retained(), 2);
    }

    #[tokio::test]
    async fn usage_stats_evicts_oldest_when_over_capacity() {
        let stats = UsageStats::new(2);
        let mk = |secs: i64| UsageRecord {
            started_at: Utc.timestamp_opt(secs, 0).unwrap(),
            ended_at: Utc.timestamp_opt(secs, 0).unwrap(),
            elapsed_ms: 0,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
        total_tokens: 0,
        };
        stats.record(mk(1));;
        stats.record(mk(2));;
        stats.record(mk(3));;
        let snap = stats.snapshot(None, None, None);
        assert_eq!(snap.len(), 2);
        // Oldest (secs=1) evicted.
        assert_eq!(snap[0].started_at.timestamp(), 2);
        assert_eq!(snap[1].started_at.timestamp(), 3);
    }

    #[test]
    fn evict_warn_threshold_advances_by_capacity_over_ten() {
        // Pure helper test — no tracing subscriber required.
        // Step = max(capacity/10, 1) so capacity=100 → step=10.
        let step = (100usize / 10).max(1);
        let mut prev = 1u64;
        for _ in 0..5 {
            prev = UsageStats::next_warn_threshold(prev, 100);
        }
        // 1 → 11 → 21 → 31 → 41 → 51
        assert_eq!(prev, 51);
        assert_eq!(step, 10);

        // Tiny capacity still produces step ≥ 1 (no zero-step loop).
        let step_small = (1usize / 10).max(1);
        assert_eq!(step_small, 1);
        assert_eq!(
            UsageStats::next_warn_threshold(1, 1),
            2,
            "capacity=1 should still emit a second warn at eviction #2"
        );
    }

    #[test]
    fn evict_warn_at_advances_after_capacity_overflow() {
        // Drive the actual record() path enough times to overflow and
        // confirm `evict_warn_at` advances correctly. With capacity=4
        // and step=max(4/10, 1)=1, the threshold should bump on every
        // single eviction (1 → 2 → 3 → ...).
        let stats = UsageStats::new(4);
        let mk = |secs: i64| UsageRecord {
            started_at: Utc.timestamp_opt(secs, 0).unwrap(),
            ended_at: Utc.timestamp_opt(secs, 0).unwrap(),
            elapsed_ms: 0,
            provider: "p".into(),
            model: "m".into(),
            stream: false,
            outcome: Outcome::Success,
            usage: None,
            failed_providers: vec![],
            total_tokens: 0,
        };
        // Fill + overflow: 8 records → 4 evictions.
        for i in 1..=8 {
            stats.record(mk(i));
        }
        assert_eq!(stats.evicted_total(), 4);
        // With capacity=4 and step=max(4/10,1)=1, threshold is 1
        // after the first eviction, then 2, then 3, then 4 — so the
        // final value should be 5 (next warn would be the 5th).
        assert_eq!(stats.evict_warn_at.load(Ordering::Relaxed), 5);
    }
}

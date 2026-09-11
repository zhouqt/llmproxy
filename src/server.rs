//! Axum server: routes for /v1/messages, /v1/models, /health, /v1/messages/count_tokens,
//! and /admin/copilot/auth (Copilot OAuth bootstrap trigger).

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{middleware, Json, Router as AxumRouter};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;

use crate::anthropic::{MessagesRequest, MessagesResponse};
use crate::error::{ProxyError, Result};
use crate::extractor::{AppJson, AppQuery};
use crate::providers::ProviderOutput;
use crate::state::AppState;
use crate::tokenize::estimate_request_tokens;
use crate::usage::{Outcome, StreamUsage, UsageRecord, UsageScanner, UsageStats, DEFAULT_LIMIT, MAX_LIMIT};

pub fn build_router(state: AppState) -> AxumRouter {
    let api = AxumRouter::new()
        .route("/v1/messages", post(messages_handler))
        .route("/v1/messages/count_tokens", post(count_tokens_handler))
        .route("/v1/models", get(list_models_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            crate::auth::require_auth,
        ));

    // Admin routes are gated behind the same auth as the v1 API. Operators
    // trigger Copilot OAuth bootstrap by POSTing here; the proxy prints
    // the user code in stdout and the device flow runs in the background.
    let admin = AxumRouter::new()
        .route("/admin/copilot/auth", post(admin_copilot_auth_handler))
        .route("/admin/status", get(admin_status_handler))
        .route("/admin/models", get(all_models_handler))
        .route("/admin/usage", get(admin_usage_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            crate::auth::require_auth,
        ));

    AxumRouter::new()
        .route("/health", get(health_handler))
        .merge(api)
        .merge(admin)
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn messages_handler(
    State(state): State<AppState>,
    AppJson(req): AppJson<MessagesRequest>,
) -> Result<Response> {
    // StartedAt must be captured before find_model so even an
    // "unknown model" early-return can record a row (Review #11).
    // The model label is what we have, so use it directly with no
    // resolved provider — `served_by` is None on this path.
    let start = StartedAt::now();

    let model_cfg = match state.router.find_model(&req.model) {
        Some(c) => c.clone(),
        None => {
            // Phase 2: emit `request started` even on the 4xx path so
            // operators see every request that reached the handler.
            // Without this, unknown-model requests vanish from the
            // log entirely (only the terminal 400 surfaces). The
            // `provider=unknown` literal distinguishes this from the
            // normal path's `provider=<primary>` so the start+finish
            // pair is always differentiable.
            tracing::info!(
                model = %req.model,
                provider = "unknown",
                stream = req.stream,
                "request started"
            );
            record_errored(
                &state.usage,
                &start,
                "<router>",
                &req.model,
                &[],
                req.stream,
                false,
                format!("unknown model: {}", req.model),
            );
            return Err(ProxyError::BadRequest(format!(
                "unknown model: {}",
                req.model
            )));
        }
    };

    // Phase 2: emit `request started` on the success path. Placed
    // BEFORE any router call so a long fallback chain can't push the
    // start line past the finish line in the log tail. `provider` is
    // the *intended* primary (not the eventual `served_by`) so the
    // start+finish pair aligns via the model label rather than
    // promising a specific provider will serve the request.
    tracing::info!(
        model = %req.model,
        provider = %model_cfg.primary,
        stream = req.stream,
        "request started"
    );

    if req.stream {
        let stream_result = state.router.stream(&model_cfg, &req).await;
        let (provider, output, attempts) = match stream_result {
            Ok(t) => t,
            Err(e) => {
                // Router-level failure (e.g. AllProvidersCoolingDown)
                // must still leave a row in /admin/usage — without
                // this, the failure case disappears from rollups
                // (Review #11). Carry the per-provider attempts so the
                // row shows who failed (review final round C3).
                let failed = extract_failed_providers(&e);
                let is_all_providers_failed = matches!(
                    e,
                    ProxyError::AllProvidersFailed { .. }
                        | ProxyError::AllProvidersCoolingDown { .. }
                        | ProxyError::RouterBadRequest { .. }
                );
                emit_all_providers_failed_warn(&e, &req.model, true);
                record_errored(
                    &state.usage,
                    &start,
                    &extract_provider_label(&e),
                    &req.model,
                    &failed,
                    req.stream,
                    is_all_providers_failed,
                    format!("router.stream failed: {e}"),
                );
                return Err(e);
            }
        };
        // Commit 5: the streaming path no longer emits its own
        // `request completed` INFO line — that vocabulary is now
        // reserved for the non-streaming handler below. The streaming
        // terminal lines (`streaming fallback` / `streaming completed`
        // / `streaming aborted`) are emitted from `stream_response` /
        // `MappedStream::finalize_and_record` so the log order
        // mirrors the wire timeline (header → first byte → last byte
        // → done).
        return Ok(stream_response(
            provider.name(),
            req.model.as_str(),
            output,
            attempts,
            start,
            state.usage.clone(),
        ));
    }

    let complete_result = state.router.complete(&model_cfg, &req).await;
    let (output, attempts, served_by) = match complete_result {
        Ok(t) => t,
        Err(e) => {
            // Router-level failure (e.g. AllProvidersFailed) must
            // still leave a row in /admin/usage — every error path
            // below this point calls `record_errored` already; this
            // site used to be the lone leak (Review #5 partially
            // addressed the post-resolution cases; #11 closes the
            // last hole). Carry the per-provider attempts so the row
            // shows who failed (review final round C3).
            let failed = extract_failed_providers(&e);
            let is_all_providers_failed = matches!(
                e,
                ProxyError::AllProvidersFailed { .. }
                    | ProxyError::AllProvidersCoolingDown { .. }
                    | ProxyError::RouterBadRequest { .. }
            );
            emit_all_providers_failed_warn(&e, &req.model, false);
            record_errored(
                &state.usage,
                &start,
                &extract_provider_label(&e),
                &req.model,
                &failed,
                req.stream,
                is_all_providers_failed,
                format!("router.complete failed: {e}"),
            );
            return Err(e);
        }
    };
    // From here on every error path records a row before returning.
    let provider_label = served_by.unwrap_or_else(|| model_cfg.primary.clone());
    // The serving provider's own in-place retries must NOT show up in
    // the success row's failed_providers — that field is meant to
    // report the failed fallback chain, not how many times we
    // retried the eventual winner. Local skips (cooldown /
    // model_rewrite) DO appear: they explain where the fallback came
    // from (plan §三方 policy — the UsageRecord view keeps full
    // history except `served_by`).
    let failed_providers = filter_failed_providers_for(&attempts, &provider_label);

    // Parse the provider output into a MessagesResponse, recording
    // an Errored UsageRecord on either shape mismatch.
    let resp = match output {
        ProviderOutput::Json(value) => match serde_json::from_value::<MessagesResponse>(value) {
            Ok(mut r) => {
                r.model = req.model.clone();
                r
            }
            Err(e) => {
                record_errored(
                    &state.usage,
                    &start,
                    &provider_label,
                    &req.model,
                    &failed_providers,
                    req.stream,
                    false,
                    format!("MessagesResponse parse failed: {e}"),
                );
                return Err(ProxyError::Internal(format!(
                    "failed to parse provider response: {e}"
                )));
            }
        },
        ProviderOutput::Stream(_) => {
            record_errored(
                &state.usage,
                &start,
                &provider_label,
                &req.model,
                &failed_providers,
                req.stream,
                false,
                "non-streaming provider returned a stream".into(),
            );
            return Err(ProxyError::Internal(
                "non-streaming provider returned a stream".into(),
            ));
        }
    };

    let usage = StreamUsage::from_complete(&resp.usage);
    let total_tokens = UsageRecord::compute_total_tokens(Some(&usage));
    // Read elapsed_ms exactly once so the record's
    // `started_at + elapsed_ms == ended_at` invariant holds (and the
    // tracing field below matches the recorded value to the
    // millisecond — Review #12).
    let elapsed_ms = start.elapsed_ms();
    state.usage.record(UsageRecord {
        started_at: start.started_at(),
        ended_at: Utc::now(),
        elapsed_ms,
        provider: provider_label.clone(),
        model: req.model.clone(),
        stream: false,
        outcome: Outcome::Success,
        usage: Some(usage),
        failed_providers: failed_providers.clone(),
        total_tokens,
    });
    // Commit 6 Option-absent: only emit `failed_providers=` when the
    // fallback chain actually recorded something. On the happy path
    // (no fallback, no retries against the same provider) the field
    // is absent rather than empty — operators grepping for
    // `failed_providers=cp:429` would otherwise catch every line.
    //
    // `last_error` pairs with a non-empty chain: it carries the
    // sanitized body of the last real upstream failure so the
    // completion line reads as a closed loop ("fell back to backup
    // because primary returned THIS"). Absent when the fallback was
    // entirely local skips (plan §Phase 1 §2 schema).
    //
    // Hot-path short-circuit (code-review finding #1/#2/#9): when no
    // fallback happened, `attempts` is empty and BOTH `format_attempts`
    // (allocates a Vec<String> + join) and `last_failed_body`
    // (sanitize scan over the body) would produce values that two of
    // the three match arms discard. Skip them and the header builder
    // entirely — this is the dominant happy-path traffic shape.
    if attempts.is_empty() {
        tracing::info!(
            model = req.model.as_str(),
            provider = %provider_label,
            stream = req.stream,
            elapsed_ms,
            "request completed"
        );
    } else {
        let summary = format_attempts(&attempts);
        let last_error = crate::router::last_failed_body(&attempts)
            .map(|body| crate::util::summarize_for_log(body, "<empty error message>"));
        match (summary, last_error) {
            (Some(s), Some(err)) => {
                tracing::info!(
                    model = req.model.as_str(),
                    provider = %provider_label,
                    stream = req.stream,
                    elapsed_ms,
                    failed_providers = %s,
                    last_error = %err,
                    "request completed"
                );
            }
            (Some(s), None) => {
                tracing::info!(
                    model = req.model.as_str(),
                    provider = %provider_label,
                    stream = req.stream,
                    elapsed_ms,
                    failed_providers = %s,
                    "request completed"
                );
            }
            (None, _) => {
                tracing::info!(
                    model = req.model.as_str(),
                    provider = %provider_label,
                    stream = req.stream,
                    elapsed_ms,
                    "request completed"
                );
            }
        }
    }

    let mut headers = HeaderMap::new();
    // The `x-llmproxy-failed-providers` header is the upstream-only
    // view of the chain (plan §三方 policy): real upstream failures
    // only, minus the serving provider. This is NOT the same as the
    // success row's `UsageRecord.failed_providers` (which keeps
    // LocalSkip history) — header and record serve different
    // consumers and deliberately differ. `format_attempts_header`
    // (router) filters to `is_upstream_failure()` and returns `None`
    // when every step is a LocalSkip, which means the header is
    // absent for the all-cooldown / all-unmappable fallbacks —
    // exactly the schema lock in plan §Phase 1 §2. With the
    // `attempts.is_empty()` short-circuit above, the header path
    // is also dead on the happy path (the function still iterates
    // the empty slice and returns None, but skipping it costs a
    // single branch on the hot path).
    if !attempts.is_empty() {
        if let Some(s) = crate::router::format_attempts_header(&attempts, Some(&provider_label)) {
            if let Ok(v) = s.parse() {
                headers.insert("x-llmproxy-failed-providers", v);
            }
        }
    }

    Ok((StatusCode::OK, headers, Json(resp)).into_response())
}

/// Pair a monotonic `Instant` (for elapsed math) with the
/// wall-clock `DateTime<Utc>` captured at the same moment. Review
/// #4 forbids deriving wall-clock from `Instant` (NTP corrections
/// and host suspend/resume would corrupt it). Review #1 caught that
/// `start.elapsed()` was being read twice — once for `started_at`,
/// once for `elapsed_ms` — causing `started_at + elapsed_ms > ended_at`.
/// The pair is captured once and consumed by reference, so the
/// monotonic read happens exactly once per record.
#[derive(Copy, Clone, Debug)]
pub(crate) struct StartedAt {
    pub instant: Instant,
    pub wall: DateTime<Utc>,
}

impl StartedAt {
    pub fn now() -> Self {
        // Capture wall-clock before the Instant so `wall` is the
        // first observable timestamp — never behind `ended_at`.
        let wall = Utc::now();
        let instant = Instant::now();
        Self { instant, wall }
    }

    /// Wall-clock start of the request. Review #4: never derived from
    /// `Instant`. This is the snapshot we took at entry.
    pub fn started_at(&self) -> DateTime<Utc> {
        self.wall
    }

    /// Milliseconds since `started_at`. Review #1: read `elapsed`
    /// exactly once so `started_at + elapsed_ms == ended_at` (modulo
    /// the `as u64` truncation, which is the only acceptable
    /// rounding). Callers MUST NOT call this more than once per record.
    pub fn elapsed_ms(&self) -> u64 {
        self.instant.elapsed().as_millis() as u64
    }
}

fn parse_failed_providers(attempts: &[crate::router::RouteStep]) -> Vec<String> {
    attempts.iter().map(crate::router::RouteStep::render).collect()
}

/// Like [`parse_failed_providers`] but drops any attempt against the
/// provider that ultimately served the request. This is the
/// `UsageRecord.failed_providers` view of the chain — it keeps **all**
/// history including local skips (cooldown / model_unmapped), so an
/// `/admin/usage` row explains where the fallback came from even when
/// every skipped provider was a no-request skip. Plan §三方 policy:
/// "`UsageRecord.failed_providers` — 全 history except `served_by`
/// (不含 `is_upstream_failure()` filter — `LocalSkip` 也记入)".
///
/// This is deliberately a SEPARATE filter from
/// [`crate::router::format_attempts_header`] (which is
/// upstream-only, for the `x-llmproxy-failed-providers` response
/// header). Plan §三方 policy §policy 单一来源: "HTTP header 的 filter
/// 写在 `router::format_attempts_header` 内部… `UsageRecord` 的
/// `provider() != served_by` filter 在 `record_errored` caller 处
/// 内联(filter 出 `Vec<String>` 后传给 `record_errored`)。
/// **两者不复用同一个 `filter_*` 函数** — 不同语义,不同抽象。这个表
/// covers and supersedes 当前 main 的 C6 注释"。
fn filter_failed_providers_for(
    attempts: &[crate::router::RouteStep],
    served_by: &str,
) -> Vec<String> {
    attempts
        .iter()
        .filter(|s| s.provider() != served_by)
        .map(crate::router::RouteStep::render)
        .collect()
}

/// Per-provider failure labels from a router-level error, for the
/// `/admin/usage` row written by `record_errored` for the `<router>`
/// provider. Mirrors the response-header path
/// (`ProxyError::failed_providers_header` → `x-llmproxy-failed-providers`)
/// but keeps the structured Vec the record needs. All three
/// router-terminal variants carry `attempts` and must surface them
/// here so the `UsageRecord.failed_providers` rollup sees the full
/// chain — plan §Phase 1 round-9 MINOR #1 fix.
fn extract_failed_providers(e: &ProxyError) -> Vec<String> {
    match e {
        ProxyError::AllProvidersFailed { attempts, .. } => parse_failed_providers(attempts),
        ProxyError::AllProvidersCoolingDown { attempts, .. } => parse_failed_providers(attempts),
        ProxyError::RouterBadRequest { attempts, .. } => parse_failed_providers(attempts),
        _ => Vec::new(),
    }
}

/// Pick the `UsageRecord.provider` label for the row that
/// `record_errored` writes on the `<router>` terminal path. Per plan
/// §Phase 1 §三方 policy (round-7 NIT #8 fix):
///
/// - `AllProvidersFailed` → the **last** `RouteStep::Failed.provider`
///   in the chain. This is the provider whose upstream error the
///   caller will see in the response body; surfacing it as the row's
///   `provider` lets `/admin/usage?group_by=provider` attribute the
///   error correctly instead of dumping every all-failed row into the
///   synthetic `<router>` bucket.
/// - `AllProvidersCoolingDown` and `RouterBadRequest` → `"<router>"`.
///   Both variants have zero real upstream calls (cooldown skips and
///   model-rewrite misses respectively); using `last_failed_provider`
///   would fall back to `None` and we'd be tempted to substitute
///   `model_cfg.primary` — that would lie, because primary was never
///   *tried* in either case. `<router>` correctly says "the router
///   decided this request was unsendable".
///
/// Returns `Cow<'static, str>` so the `Failed` case can borrow from
/// the `attempts` slice (no allocation) while the literal `"<router>"`
/// stays `Cow::Borrowed` (also no allocation).
fn extract_provider_label(e: &ProxyError) -> String {
    match e {
        ProxyError::AllProvidersFailed { attempts, .. } => crate::router::last_failed_provider(
            attempts,
        )
        .expect("AllProvidersFailed.attempts must contain at least one Failed step")
        .to_string(),
        ProxyError::AllProvidersCoolingDown { .. } | ProxyError::RouterBadRequest { .. } => {
            "<router>".to_string()
        }
        _ => "<router>".to_string(),
    }
}

/// Emit the single terminal-state `WARN all providers failed` log line
/// for the three router-terminal error variants. Centralized so the
/// stream and complete paths emit the same shape, and so callers can
/// pass `is_all_providers_failed = true` to suppress the redundant
/// `request recorded as Errored` WARN in `record_errored`.
///
/// `fail_reason` is derived from the `attempts` chain (see
/// `router::derive_fail_reason`): `upstream` if any real upstream
/// failure fired, else `cooldown` if every entry was a cooldown skip,
/// `model_skip` if every entry was `ModelUnmapped` (the
/// `RouterBadRequest` case always lands here), or `mixed` for a chain
/// holding both skip classes with no upstream attempt (possible for
/// `AllProvidersCoolingDown`). An empty `AllProvidersCoolingDown`
/// chain (no steps to classify) defaults to `cooldown`.
///
/// `last_error` is only attached when `fail_reason = upstream`; for
/// `cooldown` / `model_skip` there's no real upstream body to log, so
/// the field is absent (the schema contract per plan §Phase 1 commit 4).
fn emit_all_providers_failed_warn(e: &ProxyError, model: &str, is_stream: bool) {
    let (attempts, fail_reason) = match e {
        ProxyError::AllProvidersFailed { attempts, .. } => {
            let reason = crate::router::derive_fail_reason(attempts)
                .map(|r| r.to_string())
                .unwrap_or_else(|| "upstream".to_string());
            (attempts.as_slice(), reason)
        }
        ProxyError::AllProvidersCoolingDown { attempts, .. } => {
            // Derive from the carried skip chain: any upstream `Failed`
            // wins (unlikely here since the all-cooldown terminal is
            // reached after skips), both skip classes → `mixed`,
            // all cooldown → `cooldown`. An empty chain (empty model
            // chain or the admin `select_provider` path) has no steps
            // to classify, so it defaults to `cooldown` — plan §空切片
            // 语义 round-6 decision: "AllProvidersCoolingDown + empty
            // attempts 默认为 cooldown, 无上游尝试信息".
            let reason = crate::router::derive_fail_reason(attempts)
                .unwrap_or(crate::router::FailReason::Cooldown);
            (attempts.as_slice(), reason.to_string())
        }
        ProxyError::RouterBadRequest { attempts, .. } => {
            // model_rewrite excluded every chain entry — by construction
            // this branch contains only `ModelUnmapped` steps, so the
            // fail_reason derivation is constant. We still call
            // `derive_fail_reason` so future schema additions don't
            // silently drift; the result is asserted to be
            // `model_skip` to catch that drift in tests.
            let reason = crate::router::derive_fail_reason(attempts)
                .map(|r| r.to_string())
                .unwrap_or_else(|| "model_skip".to_string());
            (attempts.as_slice(), reason)
        }
        // Non-terminal errors should never reach here; ignore.
        _ => return,
    };
    // `format_attempts` returns `None` for an empty chain, so the
    // `failed_providers=` field stays absent on the not-a-real-chain
    // edge (empty model chain, admin select_provider) — commit 6's
    // Option-absent convention.
    let summary = format_attempts(attempts);
    if fail_reason == "upstream" {
        // Sanitize the body for the log field. After the
        // `last_failed_body` signature change (returns `Option<&str>`),
        // the caller pays the `summarize_for_log` alloc + scan only
        // when the field will actually be emitted (i.e. here, in the
        // `fail_reason = upstream` arm).
        let last_error = crate::router::last_failed_body(attempts)
            .map(|b| crate::util::summarize_for_log(b, "<empty error message>"))
            .unwrap_or_else(|| "<empty error message>".to_string());
        match summary {
            Some(s) => {
                tracing::warn!(
                    model = %model,
                    stream = is_stream,
                    failed_providers = %s,
                    fail_reason = %fail_reason,
                    last_error = %last_error,
                    "all providers failed"
                );
            }
            None => {
                tracing::warn!(
                    model = %model,
                    stream = is_stream,
                    fail_reason = %fail_reason,
                    last_error = %last_error,
                    "all providers failed"
                );
            }
        }
    } else if let Some(s) = summary {
        tracing::warn!(
            model = %model,
            stream = is_stream,
            failed_providers = %s,
            fail_reason = %fail_reason,
            "all providers failed"
        );
    } else {
        tracing::warn!(
            model = %model,
            stream = is_stream,
            fail_reason = %fail_reason,
            "all providers failed"
        );
    }
}

/// Review #5: the non-streaming `?` paths used to short-circuit
/// before `record()`, so an upstream that returned a stream when the
/// client asked for JSON (or returned malformed JSON) left no row in
/// `/admin/usage`. This helper writes an `Outcome::Errored` row at
/// every such site so the row count matches the request count.
/// `is_stream` mirrors the client's `stream` flag so a failed *streaming*
/// request isn't mislabeled as non-streaming (verified in debug-env
/// against the mock — router.stream failure rows used to say
/// `stream: false`).
///
/// `is_all_providers_failed` is the suppression flag for the
/// terminal-state WARN — `messages_handler`'s three all-failed paths
/// already emit their own `all providers failed ... fail_reason=...`
/// WARN before calling this helper. When that flag is `true`, this
/// helper only writes the `UsageRecord` (still required by Review #11
/// for the `/admin/usage` rollups) and skips the redundant WARN
/// emission — plan §Phase 1 commit 3.
fn record_errored(
    stats: &UsageStats,
    start: &StartedAt,
    provider: &str,
    model: &str,
    failed_providers: &[String],
    is_stream: bool,
    is_all_providers_failed: bool,
    reason: String,
) {
    stats.record(UsageRecord {
        started_at: start.started_at(),
        ended_at: Utc::now(),
        elapsed_ms: start.elapsed_ms(),
        provider: provider.to_string(),
        model: model.to_string(),
        stream: is_stream,
        outcome: Outcome::Errored,
        usage: None,
        failed_providers: failed_providers.to_vec(),
        total_tokens: 0,
    });
    // Suppression: when the caller (messages_handler Err arm) has
    // already emitted an `all providers failed` WARN, this WARN is
    // redundant and would double-line the same terminal event.
    if is_all_providers_failed {
        return;
    }
    // Run the upstream body through `summarize_for_log` to keep
    // multi-KB Cloudflare pages out of the log — was `reason = %reason`
    // which emitted `status=503, body=<full Cloudflare page>` for a
    // hundred-KiB HTML page (plan §Phase 1 commit 3).
    let reason_summary = crate::util::summarize_for_log(&reason, "<empty error message>");
    // Field name `stream=` (not `is_stream=`) — keeps the operator's
    // `grep stream=` query consistent with every other log line
    // emitted by `messages_handler`. Code review finding #2.
    tracing::warn!(
        provider,
        model,
        stream = is_stream,
        reason = %reason_summary,
        "request recorded as Errored"
    );
}

/// Render the attempt chain as `provider1:kind1,provider2:kind2` for
/// use as a tracing field. Returns `None` when the chain is empty —
/// the caller should skip the `failed_providers=` field entirely
/// (commit 6's `Option`-absent convention) rather than emit a field
/// with an empty value. The "render-once" half of commit 6: callers
/// that need both the log field and the header derive both from a
/// single render via [`router::format_attempts_header`] or by calling
/// this once and reusing the result.
pub fn format_attempts(attempts: &[crate::router::RouteStep]) -> Option<String> {
    if attempts.is_empty() {
        return None;
    }
    Some(
        attempts
            .iter()
            .map(crate::router::RouteStep::render)
            .collect::<Vec<_>>()
            .join(","),
    )
}

// ---------------------------------------------------------------------------
// Streaming log helpers — plan §Phase 1 §3 (streaming helpers).
//
// All three consume the same `(model, provider, elapsed_ms)` core and only
// vary in which optional fields they emit, so the vocabulary stays stable
// across the three terminal events:
//   * `log_streaming_fallback` — emitted at `stream_response` start when
//     the router had to skip / fail providers before settling on the
//     serving one. Only fires when `failed` is non-empty (the
//     `Option`-absence convention from commit 6 — fallback that didn't
//     happen has no log to leave behind).
//   * `log_streaming_completed` — emitted at `MappedStream::finalize`
//     when the stream terminated cleanly (`Ready(None)`).
//   * `log_streaming_aborted` — emitted at `MappedStream::finalize`
//     when the stream terminated with an upstream error
//     (`Ready(Some(Err))`) OR with a contract violation (Phase 2).
//
// Token fields never appear on any streaming terminal log; only
// `request completed` (non-streaming) carries them. This matches the
// pre-Phase-1 §A.3 invariant and the explicit "no tokens" plan.
fn log_streaming_fallback(
    model: &str,
    provider: &str,
    elapsed_ms: u64,
    failed: &[crate::router::RouteStep],
) {
    // Commit 6's Option-absent convention: a fallback that didn't happen
    // has no log line. Otherwise we'd emit `failed_providers=` with an
    // empty value, which is exactly the noise the rework is removing.
    let summary = match format_attempts(failed) {
        Some(s) => s,
        None => return,
    };
    // `last_error` only meaningful when at least one upstream actually
    // fired — mirrors the `fail_reason=upstream` rule from the
    // non-streaming terminal line (commit 4). Caller summarizes only
    // when the field will actually be emitted (zero-cost when absent).
    let last_error = crate::router::last_failed_body(failed)
        .map(|b| crate::util::summarize_for_log(b, "<empty error message>"));
    if let Some(err) = last_error {
        tracing::info!(
            model = %model,
            provider = %provider,
            stream = true,
            elapsed_ms,
            failed_providers = %summary,
            last_error = %err,
            "streaming fallback"
        );
    } else {
        tracing::info!(
            model = %model,
            provider = %provider,
            stream = true,
            elapsed_ms,
            failed_providers = %summary,
            "streaming fallback"
        );
    }
}

fn log_streaming_completed(model: &str, provider: &str, elapsed_ms: u64) {
    // §A.3 lock: completed never carries `failed_providers` or
    // tokens. The replay of `failed_providers` on the response header
    // is the source of truth for which providers were skipped before
    // the stream started.
    tracing::info!(
        model = %model,
        provider = %provider,
        stream = true,
        elapsed_ms,
        "streaming completed"
    );
}

fn log_streaming_aborted(
    model: &str,
    provider: &str,
    elapsed_ms: u64,
    reason: crate::router::AbortReason,
) {
    match reason {
        crate::router::AbortReason::UpstreamError => {
            tracing::info!(
                model = %model,
                provider = %provider,
                stream = true,
                elapsed_ms,
                "streaming aborted (upstream error)"
            );
        }
        crate::router::AbortReason::ContractViolation => {
            tracing::info!(
                model = %model,
                provider = %provider,
                stream = true,
                elapsed_ms,
                "streaming aborted (contract violation)"
            );
        }
    }
}

fn stream_response(
    provider_name: &str,
    model: &str,
    output: ProviderOutput,
    attempts: Vec<crate::router::RouteStep>,
    start: StartedAt,
    stats: UsageStats,
) -> Response {
    // C9: even when the router returns Ok with a non-Stream output
    // (a provider that ignores `stream` mode and hands back JSON),
    // the 500 guard MUST record a row — otherwise the request vanishes
    // from /admin/usage (Review #5 spirit: every error path leaves a
    // row). Compute failed_providers up-front so both the error row
    // and the success stream path can share it.
    //
    // `provider_name` is the serving provider (router.stream
    // succeeded on it). Drop any attempt against the same provider
    // before writing the row / building the response header. Local
    // skips remain — the row keeps full-history (plan §三方 policy).
    let failed_providers =
        filter_failed_providers_for(&attempts, provider_name);
    let ProviderOutput::Stream(stream) = output else {
        // Phase 2 §3: contract-violation branch (router returned
        // `ProviderOutput::Json` for a `stream:true` request). The
        // request started line was already emitted in
        // `messages_handler`; this site must NOT emit
        // `streaming fallback` / `streaming completed` (we never
        // reached the streaming path) — instead emit
        // `streaming aborted (contract violation)` paired with an
        // `ERROR` so operators can see the upstream misbehaviour, and
        // `record_errored` writes the row (per round-2 MAJOR fix
        // this branch still records with `is_all_providers_failed =
        // false` so the existing
        // `tests/server.rs::admin_usage_records_errored_when_stream_response_guard_fires`
        // keeps passing).
        log_streaming_aborted(
            model,
            provider_name,
            start.elapsed_ms(),
            crate::router::AbortReason::ContractViolation,
        );
        tracing::error!(
            provider = %provider_name,
            model = %model,
            elapsed_ms = start.elapsed_ms(),
            "streaming request returned non-stream output"
        );
        record_errored(
            &stats,
            &start,
            provider_name,
            model,
            &failed_providers,
            true,
            false,
            "stream_response: provider returned non-Stream output".to_string(),
        );
        return ProxyError::Internal("expected stream output".into()).into_response();
    };

    let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
        Box::into_pin(stream);

    // Commit 5: emit the streaming fallback snapshot BEFORE writing the
    // response headers / wrapping the inner stream. Doing it here keeps
    // log order = wire order (header → first byte → … → done) so
    // operators see the fallback context before the first SSE event.
    // Helper is no-op when `attempts` is empty (commit 6
    // Option-absent convention) — no `failed_providers=` empty-string
    // noise on the happy path.
    log_streaming_fallback(
        model,
        provider_name,
        start.elapsed_ms(),
        &attempts,
    );

    let mut resp_headers = HeaderMap::new();
    // Upstream-only view for the response header — see the
    // non-streaming branch's comment. Deliberately different from the
    // `UsageRecord.failed_providers` full-history view written above
    // (plan §三方 policy).
    // Written BEFORE the MappedStream::new move below — the stream
    // wrapper takes ownership of `failed_providers` so we can't borrow
    // it after the move.
    if !failed_providers.is_empty() {
        resp_headers.insert(
            "x-llmproxy-failed-providers",
            failed_providers.join(",").parse().unwrap(),
        );
    }

    let body = Body::from_stream(MappedStream::new(
        provider_name,
        model,
        inner,
        start,
        stats,
        failed_providers, // moved into the stream wrapper
    ));

    let mut resp = Response::new(body);
    let h = resp.headers_mut();
    h.insert(
        "content-type",
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    h.insert("cache-control", "no-cache".parse().unwrap());
    h.insert("x-accel-buffering", "no".parse().unwrap());
    h.extend(resp_headers);
    resp
}

/// Adapter: wraps a `Result<Bytes, ProxyError>` stream as a
/// `Result<Bytes, std::io::Error>` stream for axum's body. Emits an
/// Anthropic `event: error` SSE chunk before terminating so clients
/// don't see an incomplete body with no signal that something went
/// wrong.
///
/// Also drives a [`UsageScanner`] across the bytes so the
/// `/admin/usage` endpoint can report per-request token counts even
/// when the upstream is an Anthropic-shaped SSE source. Scanner state
/// lives here (not in the upstream provider) so adding a new provider
/// never has to thread a usage channel through the converter layer.
pub struct MappedStream {
    /// Provider name, carried into the upstream-error log so operators
    /// can see which provider's stream failed in a multi-provider
    /// deployment.
    provider: String,
    /// Client-requested model name, same purpose as `provider`.
    model: String,
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    done: bool,
    /// Wall-clock start of the request; reused for the UsageRecord
    /// written on stream termination so `started_at` is comparable
    /// across streaming vs non-streaming rows. Review #4: this is
    /// the captured-at-entry wall-clock, not a derivation from
    /// `Instant` (which would be wrong under NTP corrections).
    start: StartedAt,
    /// Shared usage stats; the row is written on terminal `Ready(None)`.
    stats: UsageStats,
    /// Per-request fallback summary mirrored from the
    /// `x-llmproxy-failed-providers` header so admin/usage rows
    /// correlate with the response header.
    ///
    /// Review #9: this was `Arc<Vec<String>>` with zero concurrent
    /// readers — the Arc added an allocation + refcount on every
    /// streaming response even in the common no-failure case, then
    /// `finalize_and_record` deref-cloned the whole inner Vec for the
    /// single record. A plain `Vec<String>` moved out with
    /// `std::mem::take` is cheaper and honest about the ownership.
    failed_providers: Vec<String>,
    /// SSE byte scanner. `Some` while we're driving the inner stream;
    /// taken in `finalize_and_record` so the write path can be infallible.
    usage_scanner: Option<UsageScanner>,
    /// Set true when the upstream emits an error mid-stream, so the
    /// UsageRecord's `outcome` reflects Errored rather than Success.
    /// See plan sixth-round finding C3.
    errored: bool,
}

impl MappedStream {
    pub(crate) fn new(
        provider: &str,
        model: &str,
        inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
        start: StartedAt,
        stats: UsageStats,
        failed_providers: Vec<String>,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            inner,
            done: false,
            start,
            stats,
            failed_providers,
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        }
    }

    /// On terminal poll, build the UsageRecord and push it to the
    /// shared ring buffer. Splits the buffer out of `self` first so
    /// the write path can't re-borrow the stream.
    fn finalize_and_record(&mut self) {
        let Some(scanner) = self.usage_scanner.take() else {
            return;
        };
        let ended_at = Utc::now();
        let saw_error = scanner.saw_error();
        let usage = scanner.finalize();
        let total_tokens = UsageRecord::compute_total_tokens(Some(&usage));
        let stats = self.stats.clone();
        // Capture terminal-state bookkeeping BEFORE moving it into the
        // record so the streaming log helpers below can use it. Code
        // review finding #8: `mem::take` the strings instead of cloning
        // — `finalize_and_record` is terminal (we return right after),
        // so the originals are dead and `take` is cheaper than
        // allocating two fresh `String`s per stream completion.
        let elapsed_ms = self.start.elapsed_ms();
        let provider = std::mem::take(&mut self.provider);
        let model = std::mem::take(&mut self.model);
        let was_errored = self.errored || saw_error;
        let record = UsageRecord {
            started_at: self.start.started_at(),
            ended_at,
            elapsed_ms,
            provider: provider.clone(),
            model: model.clone(),
            stream: true,
            outcome: if was_errored {
                Outcome::Errored
            } else {
                Outcome::Success
            },
            usage: if usage == StreamUsage::default() {
                None
            } else {
                Some(usage)
            },
            failed_providers: std::mem::take(&mut self.failed_providers),
            total_tokens,
        };
        // Synchronous write. `UsageStats::record` is sync per plan
        // L431/511-519 (std::sync::RwLock), so this is just a brief
        // lock acquire — no scheduler trip, no spawn, no risk of
        // the record landing after the request was already
        // acknowledged to the client.
        stats.record(record);
        // Commit 5: streaming terminal log. `errored` reflects a
        // mid-stream upstream `Ready(Some(Err))`; `saw_error` reflects
        // an upstream-emitted `event: error` chunk the scanner caught.
        // Either one routes to `streaming aborted (upstream error)` —
        // both are "stream ended not because the client received
        // message_stop, but because something upstream failed".
        if was_errored {
            log_streaming_aborted(
                &model,
                &provider,
                elapsed_ms,
                crate::router::AbortReason::UpstreamError,
            );
        } else {
            log_streaming_completed(&model, &provider, elapsed_ms);
        }
    }
}

impl Drop for MappedStream {
    /// Streams dropped before reaching `Poll::Ready(None)` — true
    /// client disconnect mid-stream OR premature wrapper teardown
    /// by axum's `Body::from_stream` — are NOT recorded AND NOT
    /// LOGGED. We have no way to tell whether upstream tokens were
    /// billed for bytes the client never saw, so writing a row would
    /// either be (a) a phantom Success for work the client never
    /// received, or (b) mislabel a disconnect as a normal completion;
    /// emitting a `streaming aborted (client disconnect)` line would
    /// be equally misleading (operators can't act on a disconnect —
    /// it's the client's choice, not an upstream fault). Per plan
    /// §Phase 1 §client disconnect we accept the undercount rather
    /// than mislabel. The `Ready(None)` arm of `poll_next` is the
    /// only path that calls `finalize_and_record` (and therefore the
    /// only path that emits a streaming terminal log) now.
    fn drop(&mut self) {
        // Drop the scanner without recording. The previous
        // behaviour called `finalize_and_record` from Drop as a
        // fallback for streams that reached `Ready(None)` in
        // `poll_next`, but that path already records via the
        // `poll_next` arm — this Drop is now strictly a
        // cleanup path.
        let _ = self.usage_scanner.take();
    }
}

impl Stream for MappedStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => {
                if let Some(s) = self.usage_scanner.as_mut() {
                    s.observe(&b);
                }
                Poll::Ready(Some(Ok(b)))
            }
            Poll::Ready(Some(Err(e))) => {
                // Sanitize the upstream body before logging it — the
                // `%e` Display for `ProxyError::Upstream` embeds the
                // full body verbatim, so a 100 KiB Cloudflare error
                // page would land in the log line on every abort.
                // `summarize_for_log` strips HTML/URLs and caps at
                // ~200 chars (plan §Phase 1 §4: unstripped multi-KB
                // bodies must never reach the log).
                let status = e.status_code().as_u16();
                let detail = crate::util::summarize_for_log(
                    &e.to_string(),
                    "<empty error message>",
                );
                tracing::error!(
                    provider = %self.provider,
                    model = %self.model,
                    status = %status,
                    error = %detail,
                    "upstream stream error"
                );
                // Emit a synthetic Anthropic `event: error` SSE chunk so
                // the client can distinguish "stream ended normally"
                // from "stream aborted by upstream failure" — without
                // this, the body just truncates with 200 OK and no
                // message_stop, which Anthropic SDKs report as a
                // confusing parse error. Mark `done` so the next poll
                // terminates the stream instead of emitting the chunk
                // again. Flip errored so the UsageRecord reflects this.
                //
                // Finalize the usage record BEFORE flipping `done`,
                // because once `done` is true the next poll returns
                // `Ready(None)` immediately via the early-return at
                // the top of `poll_next` and skips the finalize path
                // (opus review #1: mid-stream upstream errors must
                // produce a UsageRecord with `outcome: errored`).
                self.errored = true;
                self.finalize_and_record();
                self.done = true;
                Poll::Ready(Some(Ok(format_stream_error(&e))))
            }
            Poll::Ready(None) => {
                self.done = true;
                self.finalize_and_record();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Encode a [`ProxyError`] as an Anthropic SSE `event: error` chunk.
fn format_stream_error(err: &ProxyError) -> Bytes {
    // Code review finding #4: build the SSE error chunk in ONE pass.
    // The previous shape paid for `serde_json::json!` (Value Map +
    // nested Values) + `format!("data: {payload}")` (which calls
    // Display on the Value, serializing it via `to_string()`) +
    // `Bytes::from` (clones the resulting String). Now we write the
    // JSON-shaped bytes via a single `write!` into a `BytesMut`, then
    // freeze it once. One allocation, one walk, one conversion.
    use std::fmt::Write;
    let mut buf = bytes::BytesMut::with_capacity(96 + err.to_string().len());
    buf.extend_from_slice(b"event: error\ndata: ");
    // The shape mirrors the upstream Anthropic `error` event: a JSON
    // object with `type` and an `error.{type,message}` payload. The
    // message is the Display form of the ProxyError — must be
    // JSON-escaped (quotes, backslashes, control chars) so we route
    // through `serde_json::to_string` rather than hand-rolling
    // escaping (which would re-introduce the same alloc cost).
    let inner = serde_json::json!({
        "type": "error",
        "error": {
            "type": "upstream_error",
            "message": err.to_string(),
        }
    });
    let _ = write!(&mut buf, "{inner}");
    buf.extend_from_slice(b"\n\n");
    buf.freeze()
}

async fn count_tokens_handler(
    State(_state): State<AppState>,
    AppJson(req): AppJson<serde_json::Value>,
) -> impl IntoResponse {
    let tokens = estimate_request_tokens(&req);
    Json(serde_json::json!({ "input_tokens": tokens }))
}

/// Per-call timeout for upstream model-catalog fetches in the model-list
/// endpoints. The shared reqwest client's default timeout is 600 s
/// (`proxy_client.rs`), which would let a single hanging upstream stall
/// these endpoints for minutes; metadata must stay snappy.
const MODELS_METADATA_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Fetch a provider's model catalog, bounded by [`MODELS_METADATA_TIMEOUT`].
/// `timeout` is factored out so tests can exercise the timeout branch
/// without waiting 10 seconds.
async fn fetch_models_with_timeout(
    provider: &crate::providers::SharedProvider,
    timeout: std::time::Duration,
) -> Option<Vec<serde_json::Value>> {
    match tokio::time::timeout(timeout, provider.list_models()).await {
        Ok(models) => models,
        Err(_) => {
            tracing::warn!(
                provider = %provider.name(),
                timeout_secs = timeout.as_secs(),
                "list_models timed out"
            );
            None
        }
    }
}

/// Routing priority for every `(upstream model id, provider name)` pair
/// reachable through `config.models`.
///
/// Keyed by **upstream id** (the `id` a provider's `list_models()` reports),
/// not the client-facing `ModelConfig::name`: for each chain position we
/// translate the client name through the provider's configured
/// `model_rewrite` table. The value is `(chain position, declaration seq)`
/// where chain position is 0 for the primary and 1, 2, ... for fallbacks,
/// and `seq` is the visiting order over `config.models` (declaration order,
/// deterministic — never the `HashMap` iteration order of
/// `router.providers()`). The minimum wins on duplicate keys.
///
/// Providers whose `can_serve_model` rejects the **client** name (the same
/// check the router performs at dispatch time) are skipped entirely.
pub(crate) fn build_routing_priority(
    config: &crate::config::Config,
    providers: &std::collections::HashMap<String, crate::providers::SharedProvider>,
) -> std::collections::HashMap<(String, String), (u64, u64)> {
    let empty = std::collections::HashMap::new();
    let mut priority = std::collections::HashMap::new();
    let mut seq: u64 = 0;
    for m in &config.models {
        for (pos, provider_name) in m.chain().enumerate() {
            let Some(provider) = providers.get(provider_name) else {
                continue;
            };
            // Client-facing name — matches the router's own dispatch check.
            if !provider.can_serve_model(&m.name) {
                continue;
            }
            let rewrite = provider.merged_rewrite(&empty);
            let upstream_id = rewrite.get(&m.name).cloned().unwrap_or_else(|| m.name.clone());
            let key = (upstream_id, provider_name.to_string());
            let rank = (pos as u64, seq);
            seq += 1;
            priority
                .entry(key)
                .and_modify(|e: &mut (u64, u64)| {
                    if rank < *e {
                        *e = rank;
                    }
                })
                .or_insert(rank);
        }
    }
    priority
}

/// Selection rank for a dedup candidate. Lower wins. Class order:
/// registered (in a routing chain) < discovered-but-unregistered <
/// static config entry — so the entry the router would actually use beats
/// everything else, and a real upstream catalog entry beats the generic
/// static placeholder.
///
/// The trailing `(provider, json)` strings only matter for the
/// unregistered class: entries there come from unordered sources, so the
/// lexicographically smallest `(provider, serialized entry)` wins to keep
/// the result deterministic across restarts.
type EntryRank = (u8, u64, u64, String, String);

fn entry_rank(
    entry: &serde_json::Value,
    priority: &std::collections::HashMap<(String, String), (u64, u64)>,
) -> EntryRank {
    let provider = entry.get("owned_by").and_then(|v| v.as_str());
    let fallback_json = serde_json::to_string(entry).unwrap_or_default();
    match provider {
        Some("llmproxy") => (2, 0, 0, String::new(), String::new()),
        Some(p) => match priority.get(&(entry["id"].as_str().unwrap_or_default().to_string(), p.to_string())) {
            Some(&(pos, seq)) => (0, pos, seq, String::new(), String::new()),
            None => (1, 0, 0, p.to_string(), fallback_json),
        },
        // No attribution (should not happen post-aggregation): treat as
        // unregistered so a properly attributed entry wins.
        None => (1, 0, 0, String::new(), fallback_json),
    }
}

/// Dedup model entries by `id`, preferring the entry the router would
/// actually use. Replaces the previous last-occurrence-wins pass whose
/// winner depended on `HashMap` iteration order (nondeterministic across
/// restarts).
///
/// Rules (see docs/models-api-split-plan.md §1.2):
/// - entries attributed to a routing-chain position (per
///   [`build_routing_priority`]) beat unregistered and static entries;
/// - among registered entries the smallest `(chain position, declaration
///   seq)` wins — i.e. the primary beats fallbacks;
/// - unregistered discovered entries beat the static placeholder;
/// - static entries (`owned_by: "llmproxy"`) survive only when nothing
///   else claims the id.
///
/// Entries with a missing or empty `id` are filtered out with a warning.
fn dedup_models_by_routing_priority(
    entries: &mut Vec<serde_json::Value>,
    priority: &std::collections::HashMap<(String, String), (u64, u64)>,
) {
    // First-seen order of each id determines output order.
    let mut order: Vec<String> = Vec::new();
    let mut best: std::collections::HashMap<String, (EntryRank, serde_json::Value)> =
        std::collections::HashMap::new();
    for entry in entries.drain(..) {
        let id = match entry.get("id").and_then(|v| v.as_str()) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => {
                tracing::warn!("model entry has empty or missing id, skipping");
                continue;
            }
        };
        let rank = entry_rank(&entry, priority);
        match best.get_mut(&id) {
            Some((best_rank, _)) if *best_rank <= rank => {}
            _ => {
                if !order.contains(&id) {
                    order.push(id.clone());
                }
                best.insert(id, (rank, entry));
            }
        }
    }
    entries.extend(order.into_iter().filter_map(move |id| {
        best.remove_entry(&id).map(|(_, (_, entry))| entry)
    }));
}

async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Routing priority decides which provider's entry wins an id collision
    // (see docs/models-api-split-plan.md §1.2). Built from `config.models`
    // declaration order — deterministic across restarts.
    let priority = build_routing_priority(&state.config, state.router.providers());

    // Static config entries: lowest priority (see dedup rules), they only
    // survive when no provider claims the same id.
    let mut entries: Vec<_> = state
        .config
        .models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "llmproxy",
                "display_name": m.name,
            })
        })
        .collect();

    // Provider-declared order (config declaration order, not HashMap order)
    // for deterministic aggregation; fetches run concurrently, each bounded
    // by MODELS_METADATA_TIMEOUT so one hanging upstream can't stall the
    // endpoint for the shared client's 600 s default.
    let names: Vec<&str> = state.config.providers.iter().map(|p| p.name()).collect();
    let providers = state.router.providers().clone();
    let fetches = names.iter().map(|name| {
        let providers = providers.clone();
        async move {
            let models = match providers.get(*name) {
                Some(p) => fetch_models_with_timeout(p, MODELS_METADATA_TIMEOUT).await,
                None => None,
            };
            (*name, models)
        }
    });
    let results = futures_util::future::join_all(fetches).await;

    for (provider_name, models) in results {
        let Some(models) = models else {
            continue;
        };
        let mut models = models;
        for entry in models.iter_mut() {
            let Some(obj) = entry.as_object_mut() else {
                continue;
            };
            // Invariant (plan §1.1): every aggregated entry carries the
            // configured provider name in `owned_by`. Anything a provider
            // put in `owned_by` is upstream ownership — relocate it.
            if let Some(upstream) = obj.remove("owned_by") {
                obj.entry("upstream_owned_by").or_insert(upstream);
            }
            obj.insert("owned_by".to_string(), json!(provider_name));
        }
        entries.extend(models);
    }

    dedup_models_by_routing_priority(&mut entries, &priority);

    Json(serde_json::json!({
        "object": "list",
        "data": entries,
    }))
}

/// List ALL models from ALL configured providers, grouped per provider and
/// deliberately NOT deduplicated — `/v1/models` shows what the router would
/// use; this endpoint shows everything each upstream advertises (see
/// docs/models-api-split-plan.md §2.1).
///
/// Always returns 200: a provider whose catalog fetch fails or times out
/// still appears with an empty `models` array plus a `cache_state` reason,
/// mirroring the degradation semantics of `/admin/status`.
async fn all_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    use crate::providers::Provider;

    let providers = state.router.providers().clone();
    let fetches = state.config.providers.iter().map(|cfg| {
        let providers = providers.clone();
        let cfg = cfg.clone();
        async move {
            let provider = providers.get(cfg.name());
            let discovered = match &provider {
                Some(p) => fetch_models_with_timeout(p, MODELS_METADATA_TIMEOUT).await,
                None => None,
            };
            (cfg, provider.is_some(), discovered)
        }
    });

    let mut groups = Vec::new();
    for (cfg, configured, discovered) in futures_util::future::join_all(fetches).await {
        let rewrite = cfg.model_rewrite();
        let has_static = !rewrite.is_empty();

        // Merge static (rewrite-table values) + discovered into one map
        // keyed by id; discovered metadata wins on collision. Static
        // duplicates collapse deterministically by smallest client key.
        let mut merged: std::collections::BTreeMap<String, serde_json::Value> =
            std::collections::BTreeMap::new();
        if has_static {
            // Sort by client key: `model_rewrite` is a HashMap with random
            // iteration order, so this keeps duplicate upstream values
            // deterministic (smallest client key wins via or_insert).
            let mut static_pairs: Vec<(&String, &String)> = rewrite.iter().collect();
            static_pairs.sort();
            for (_, upstream) in static_pairs {
                merged
                    .entry(upstream.clone())
                    .or_insert_with(|| {
                        serde_json::json!({
                            "id": upstream,
                            "object": "model",
                            "created": 0,
                            "display_name": upstream,
                        })
                    });
            }
        }

        let cache_state = match (&discovered, state.copilot.as_ref()) {
            (Some(_), _) => "populated",
            (None, Some(c)) if c.name() == cfg.name() => c.cache_state().await,
            (None, _) => "fetch_failed",
        };

        let discovered_ok = discovered.is_some();
        if let Some(models) = discovered {
            for entry in models {
                let Some(id) = entry.get("id").and_then(|v| v.as_str()).map(str::to_string)
                else {
                    continue;
                };
                merged.insert(id, entry);
            }
        }

        let source = match (has_static, discovered_ok) {
            (true, true) => "static+discovered",
            (true, false) => "static",
            (false, _) => "discovered",
        };

        groups.push(serde_json::json!({
            "provider": cfg.name(),
            "type": cfg.type_label(),
            "configured": configured,
            "source": source,
            "cache_state": cache_state,
            "models": merged.into_values().collect::<Vec<_>>(),
        }));
    }

    Json(serde_json::json!({
        "object": "list",
        "providers": groups,
    }))
}

/// Report current health of every configured provider.
///
/// The response shape is the contract for the Claude Code statusline
/// hook — see `docs/PLANS/provider-status-endpoint.md`. Each row carries
/// a stable machine-readable `status` keyword (`available`/`cooling_down`)
/// that mirrors the router's own cooldown state, the provider kind label,
/// served models, and — when cooling — the triggering upstream status and
/// remaining cooldown seconds. Always 200 when the proxy is reachable; the
/// endpoint answers out of in-memory state and never contacts upstreams.
async fn admin_status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let statuses = state.router.provider_status().await;
    let total = statuses.len();
    let cooling_down = statuses
        .iter()
        .filter(|s| s.status == crate::router::ProviderHealth::CoolingDown)
        .count();
    let available = total - cooling_down;
    Json(json!({
        "status": "ok",
        "providers": statuses,
        "summary": {
            "total": total,
            "available": available,
            "cooling_down": cooling_down,
        }
    }))
}

/// Query params for `/admin/usage`. All fields are optional and
/// default to "no bound". Time bounds are RFC 3339 strings — the
/// same shape `started_at` / `ended_at` use on the wire, so a row's
/// `started_at` can be fed straight back in.
#[derive(Debug, Default, Deserialize)]
pub struct AdminUsageQuery {
    pub since: Option<String>,
    pub until: Option<String>,
    pub limit: Option<usize>,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// `model` / `provider` / `model_provider` (default). Any other
    /// value returns 400 (Anthropic-shaped) rather than silently
    /// defaulting.
    pub group_by: Option<String>,
}

/// Serve the buffered usage ring buffer as JSON. Reads only — the
/// admin endpoint never blocks other request handlers on the buffer
/// lock for more than a microsecond, and `cache-control: no-store`
/// signals the response is point-in-time. Operators pair this with
/// `x-llmproxy-failed-providers` header correlation when chasing
/// fallback incidents.
async fn admin_usage_handler(
    State(state): State<AppState>,
    AppQuery(q): AppQuery<AdminUsageQuery>,
) -> Response {
    let since = match q.since.as_deref().map(parse_rfc3339).transpose() {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let until = match q.until.as_deref().map(parse_rfc3339).transpose() {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let group_by = match parse_group_by(q.group_by.as_deref().unwrap_or("model_provider")) {
        Ok(g) => g,
        Err(e) => return e.into_response(),
    };

    // 1. Single-pass query: builds both the records page AND the
    //    rollup under one read lock with bounded allocation
    //    (opus review #6 — the previous two-step flow cloned ALL
    //    matching records just to keep `limit` of them). The
    //    rollup aggregates over the FULL filtered set so `limit`
    //    never skews totals (plan §Endpoint L591, L601).
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);
    let (records, rollup) = state.usage.query(
        since,
        until,
        q.model.as_deref(),
        q.provider.as_deref(),
        group_by,
        Some(limit),
    );

    let body = json!({
        "started_at": state.usage.started_at(),
        "capacity": state.usage.capacity(),
        "retained": state.usage.retained(),
        "evicted_total": state.usage.evicted_total(),
        "filter": {
            "since": q.since,
            "until": q.until,
            "model": q.model,
            "provider": q.provider,
            "group_by": group_by_label(group_by),
            "limit": limit,
        },
        "records": records,
        "rollup": rollup,
    });
    let mut resp = Json(body).into_response();
    resp.headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    resp
}

fn parse_group_by(s: &str) -> crate::error::Result<crate::usage::GroupBy> {
    crate::usage::GroupBy::from_str(s)
        .ok_or_else(|| ProxyError::BadRequest(format!("invalid group_by: {s:?}")))
}

fn group_by_label(g: crate::usage::GroupBy) -> &'static str {
    match g {
        crate::usage::GroupBy::Model => "model",
        crate::usage::GroupBy::Provider => "provider",
        crate::usage::GroupBy::ModelProvider => "model_provider",
    }
}

fn parse_rfc3339(s: &str) -> crate::error::Result<chrono::DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| ProxyError::BadRequest(format!("invalid RFC3339 timestamp: {e}")))
}
/// Trigger GitHub Copilot OAuth bootstrap on demand.
///
/// Returns 200 with the device code info (operator shows it to the user),
/// or 409 if a bootstrap is already in progress, or 404 if no Copilot
/// provider is configured. The actual device flow + token exchange runs
/// in a spawned task; this handler returns as soon as the device code is
/// issued so the operator can move on. See fix-R2.
async fn admin_copilot_auth_handler(State(state): State<AppState>) -> Response {
    let Some(provider) = state.copilot.clone() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "type": "error",
                "error": {
                    "type": "not_found",
                    "message": "no github_copilot provider configured",
                }
            })),
        )
            .into_response();
    };
    match provider.start_bootstrap().await {
        Ok(dc) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "message": "bootstrap started; complete the device flow within the timeout",
                "device_code": dc.device_code,
                "user_code": dc.user_code,
                "verification_uri": dc.verification_uri,
                "expires_in": dc.expires_in,
                "interval": dc.interval,
            })),
        )
            .into_response(),
        Err(e) => {
            // "already in progress" is a normal conflict; surface it as
            // 409 so the operator can retry after the existing flow
            // finishes. Anything else is an internal error.
            let msg = e.to_string();
            if msg.contains("already in progress") {
                (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "type": "error",
                        "error": {
                            "type": "conflict",
                            "message": msg,
                        }
                    })),
                )
                    .into_response()
            } else {
                e.into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use std::collections::HashMap;
    use std::pin::Pin;

    fn make_stream(
        items: Vec<std::result::Result<Bytes, ProxyError>>,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> {
        Box::pin(stream::iter(items))
    }

    fn fresh_mapped() -> MappedStream {
        MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![]),
            done: false,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        }
    }

    /// Single shared panic message for all `assert_matches!`-style helpers.
    /// Keeping the message in one helper means each test call site is free
    /// of its own missed panic-string line.
    fn expect_poll_none(poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>) {
        assert!(matches!(poll, std::task::Poll::Ready(None)), "expected Ready(None)");
    }

    fn expect_poll_pending(poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>) {
        assert!(matches!(poll, std::task::Poll::Pending), "expected Pending");
    }

    fn assert_poll_ready_some_ok(
        poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>,
        label: &str,
    ) -> Bytes {
        match poll {
            std::task::Poll::Ready(Some(Ok(b))) => b,
            other => panic!("{label}: expected Ready(Some(Ok)), got {other:?}"),
        }
    }

    #[test]
    fn mapped_stream_returns_none_when_already_done() {
        // Once `done` is set, poll_next must short-circuit to Ready(None)
        // without touching the inner stream at all.
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![Err(ProxyError::Internal("unused".into()))]),
            done: true,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_none(poll);
    }

    #[test]
    fn mapped_stream_propagates_pending_from_inner() {
        // When the inner stream returns Poll::Pending, the wrapper must
        // also return Poll::Pending (and must NOT mark itself done).
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: Box::pin(stream::pending::<std::result::Result<Bytes, ProxyError>>()),
            done: false,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_pending(poll);
        assert!(!s.done, "Pending must not flip done=true");
    }

    #[tokio::test]
    async fn mapped_stream_emits_error_event_then_terminates_on_inner_error() {
        // An upstream error must NOT just truncate the body — the client
        // would see 200 OK and no message_stop, with no signal that
        // anything went wrong. We inject an Anthropic `event: error`
        // chunk so the SDK can distinguish aborted streams from normal
        // end-of-stream.
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![Err(ProxyError::Internal("boom".into()))]),
            done: false,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);

        // First poll: the synthetic error chunk.
        let b1 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "error event",
        );
        let s1 = std::str::from_utf8(&b1).unwrap();
        assert!(
            s1.contains("event: error"),
            "expected event:error, got: {s1}"
        );
        assert!(s1.contains("boom"), "error body must contain message: {s1}");
        assert!(
            s1.contains("upstream_error"),
            "error type must be upstream_error: {s1}"
        );

        // Second poll: stream ends.
        let p2 = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(p2, Poll::Ready(None)));
        assert!(s.done);
    }

    #[tokio::test]
    async fn mapped_stream_records_errored_when_stream_ends_cleanly_after_inband_error() {
        // C2 regression: a mid-stream failure arriving as an in-band
        // Anthropic `event: error` frame (responses/OAI adapters emit
        // these; see format_stream_error) terminates the byte stream
        // HEALTHILY — no transport Err, body closes normally, HTTP 200.
        // The UsageRecord must still be Errored so the operator doesn't
        // see a "success" row for a failed generation.
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![Ok(Bytes::from_static(
                b"event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":3}}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\",\"message\":\"boom\"}}\n\n",
            ))]),
            done: false,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);

        let b1 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "first poll",
        );
        assert!(
            std::str::from_utf8(&b1).unwrap().contains("event: error"),
            "in-band error chunk must reach the client"
        );

        let p2 = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(p2, Poll::Ready(None)));
        assert!(s.done);

        let recs = s.stats.snapshot(None, None, None);
        assert_eq!(recs.len(), 1, "healthy end must still record one usage row");
        assert_eq!(
            recs[0].outcome,
            Outcome::Errored,
            "in-band event:error must record Errored, not Success"
        );
        assert_eq!(
            recs[0].usage.as_ref().and_then(|u| u.input_tokens),
            Some(3),
            "input usage must survive the error frame"
        );
    }

    #[tokio::test]
    async fn mapped_stream_emits_bytes_then_terminates() {
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![
                Ok(Bytes::from_static(b"event: foo\n\n")),
                Ok(Bytes::from_static(b"event: bar\n\n")),
            ]),
            done: false,
            start: StartedAt::now(),
            stats: UsageStats::new(8),
            failed_providers: Vec::new(),
            usage_scanner: Some(UsageScanner::new()),
            errored: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);

        let b1 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "first poll",
        );
        assert_eq!(&b1[..], b"event: foo\n\n");

        let b2 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "second poll",
        );
        assert_eq!(&b2[..], b"event: bar\n\n");

        let p3 = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(p3, Poll::Ready(None)));
        assert!(s.done);
    }

    #[test]
    fn format_stream_error_contains_event_and_message() {
        // Standalone unit test for the helper so future SSE-format
        // changes are caught here.
        let bytes = format_stream_error(&ProxyError::Internal("disk full".into()));
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with("event: error\n"));
        assert!(s.contains("disk full"));
        assert!(s.contains("upstream_error"));
        assert!(s.ends_with("\n\n"));
    }

    #[test]
    fn fresh_mapped_helper_is_not_done() {
        let s = fresh_mapped();
        assert!(!s.done);
    }

    // ──────────────────────────────────────────────────────────────────
    // admin_copilot_auth: 200 OK / 409 Conflict paths.
    //
    // The 404 arm is covered in tests/server.rs. These unit tests
    // exercise the success arm (device code returned) and the conflict
    // arm (concurrent bootstrap fails fast with the structured 409
    // envelope). They construct a real `CopilotProvider` against a
    // wiremock github device-flow endpoint and exercise the handler
    // through `axum::Router::oneshot`.
    //
    // Note: CopilotState is private to providers/copilot.rs, so we use
    // `CopilotProvider::new` which builds the state from the standard
    // TokenStore path (XDG_DATA_HOME). To redirect the device-flow URL
    // at the wiremock we use the crate-private
    // `LLMPROXY_TEST_GITHUB_BASE_URL` env var; the existing
    // `device_flow::ENV_LOCK` serializes this against parallel tests.
    // ──────────────────────────────────────────────────────────────────
    mod admin_copilot {
        use crate::config::{Config, ModelConfig, ProviderConfig, ServerConfig};
        use crate::cooldown::CooldownCache;
        use crate::providers::Provider;
        use crate::providers::copilot::CopilotProvider;
        use crate::router::Router;
        use crate::state::AppState;
        use crate::usage::UsageStats;
        use axum::body::Body;
        use axum::http::{Method, Request, StatusCode};
        use http_body_util::BodyExt;
        use serde_json::{json, Value};
        use std::collections::HashMap;
        use std::sync::Arc;
        use tower::util::ServiceExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        fn build_app_with_copilot(provider: Arc<CopilotProvider>) -> axum::Router {
            let cfg = Config {
                server: ServerConfig {
                    listen: "127.0.0.1:0".to_string(),
                    api_key: None,
                },
                proxy: Default::default(),
                user_agent: crate::config::default_user_agent(),
                providers: vec![ProviderConfig::GithubCopilot {
                    name: "copilot".to_string(),
                    vscode_version: "1.95.0".to_string(),
                    account_type: "individual".to_string(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                }],
                models: vec![ModelConfig {
                    name: "m".to_string(),
                    primary: "copilot".to_string(),
                    fallback_chain: vec![],
                    cooldown_seconds: 60,
                    max_retries_per_provider: 1,
                    max_retries_total: 1,
                }],
                    ..Config::default()
                };
            let cfg = Arc::new(cfg);
            let cooldown = CooldownCache::new();
            let mut providers = HashMap::new();
            providers.insert("copilot".to_string(), provider.clone() as Arc<dyn Provider>);
            let router = Arc::new(Router::new(cfg.clone(), providers, cooldown.clone()));
            let state = AppState {
                config: cfg,
                router,
                cooldown,
                http: reqwest::Client::new(),
                copilot: Some(provider),
                usage: UsageStats::new(8),
            };
            crate::server::build_router(state)
        }

        /// Build a real `CopilotProvider` whose token store lives in a
        /// private tempdir. The caller is responsible for setting the
        /// github-base-URL env var under `ENV_LOCK`.
        fn new_copilot() -> Arc<CopilotProvider> {
            let dir = tempfile::tempdir().expect("tempdir");
            // XDG_DATA_HOME must be set BEFORE `CopilotProvider::new`
            // is called — `TokenStore::new` reads it.
            std::env::set_var("XDG_DATA_HOME", dir.path());
            Arc::new(
                CopilotProvider::new(
                    "copilot".to_string(),
                    "1.95.0".to_string(),
                    "individual".to_string(),
                    HashMap::new(),
                    reqwest::Client::new(),
                )
                .expect("copilot provider builds"),
            )
        }

        #[tokio::test]
        async fn admin_copilot_auth_returns_200_with_device_code_when_bootstrap_starts() {
            // Hold the github-base-URL env lock and point it at the
            // wiremock for this test only.
            let _env_guard = crate::oauth::device_flow::ENV_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());

            let server = MockServer::start().await;
            std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());

            Mock::given(method("POST"))
                .and(path("/login/device/code"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "device_code": "code-200",
                    "user_code": "USER-200",
                    "verification_uri": "https://example.test/device",
                    "expires_in": 600,
                    "interval": 5,
                })))
                .mount(&server)
                .await;

            let provider = new_copilot();
            let app = build_app_with_copilot(provider);

            let resp = app
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/admin/copilot/auth")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            // Drop the env override immediately so the rest of the
            // suite isn't affected by a stale value.
            std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

            assert_eq!(resp.status(), StatusCode::OK);
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["status"], "ok");
            assert!(
                body["message"]
                    .as_str()
                    .unwrap()
                    .contains("complete the device flow"),
                "message should describe the bootstrap step, got: {body}"
            );
            assert_eq!(body["device_code"], "code-200");
            assert_eq!(body["user_code"], "USER-200");
            assert_eq!(body["verification_uri"], "https://example.test/device");
            assert_eq!(body["expires_in"], 600);
            assert_eq!(body["interval"], 5);
        }

        #[tokio::test]
        async fn admin_copilot_auth_returns_409_when_bootstrap_already_in_progress() {
            // The two requests must be SEQUENCED, not raced. Rationale:
            // `start_bootstrap`'s `try_lock` fast-path guard is dropped
            // immediately after the check — the refresh_lock is only
            // truly *held* by the spawned background task, which acquires
            // it AFTER `request_device_code` returns and then blocks in
            // the poll loop (which sleeps `interval.max(5)+1` ≈ 6s before
            // its first HTTP poll). So the reliable ordering is:
            //   1. request 1 returns 200 (device code issued),
            //   2. its spawned task acquires refresh_lock and parks in
            //      the ~6s poll sleep,
            //   3. request 2 hits try_lock while the lock is held →
            //      "already in progress" → structured 409 Conflict.
            // The device-code mock is instant; the second request must
            // arrive during the spawned task's poll sleep, which is why
            // we await request 1 fully and then poll for the 409 with a
            // short retry budget (well under the ~6s hold window).
            let _env_guard = crate::oauth::device_flow::ENV_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());

            let server = MockServer::start().await;
            std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());

            Mock::given(method("POST"))
                .and(path("/login/device/code"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "device_code": "code-slow",
                    "user_code": "SLOW",
                    "verification_uri": "https://example.test/device",
                    "expires_in": 600,
                    "interval": 5,
                })))
                .mount(&server)
                .await;
            // The poll loop will fire one access_token request after its
            // first sleep; answer with authorization_pending so it keeps
            // holding the lock (never completes) for the test's lifetime.
            Mock::given(method("POST"))
                .and(path("/login/oauth/access_token"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "error": "authorization_pending"
                })))
                .mount(&server)
                .await;

            let provider = new_copilot();
            let app = build_app_with_copilot(provider);

            let mk_req = || {
                Request::builder()
                    .method(Method::POST)
                    .uri("/admin/copilot/auth")
                    .body(Body::empty())
                    .unwrap()
            };

            // Request 1: expect 200 and a device code.
            let first_resp = app.clone().oneshot(mk_req()).await.unwrap();
            assert_eq!(
                first_resp.status(),
                StatusCode::OK,
                "first bootstrap must succeed"
            );

            // The spawned task acquires refresh_lock asynchronously after
            // request 1 returns. Poll request 2 until it observes the
            // held lock (409). Budget stays well under the ~6s hold.
            let mut conflict_resp = None;
            for _ in 0..50 {
                let resp = app.clone().oneshot(mk_req()).await.unwrap();
                if resp.status() == StatusCode::CONFLICT {
                    conflict_resp = Some(resp);
                    break;
                }
                // 200 means the spawned task hasn't grabbed the lock yet
                // (or already released it — impossible here since the
                // poll loop never completes). Give it a moment.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

            let conflict_resp =
                conflict_resp.expect("a concurrent bootstrap must eventually return 409 Conflict");

            // Inspect the 409 body: structured conflict envelope with
            // an "already in progress" message.
            let bytes = conflict_resp.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], "conflict");
            let msg = body["error"]["message"].as_str().unwrap();
            assert!(
                msg.contains("already in progress"),
                "409 message must mention 'already in progress', got: {msg}"
            );
        }

        /// Triggers the `Err(e) => { ... else { e.into_response() } }`
        /// branch of `admin_copilot_auth_handler` — i.e. start_bootstrap
        /// fails for a reason that is *not* "already in progress"
        /// (e.g. GitHub returned 500). The handler must surface it as a
        /// normal internal error response, not a 409.
        #[tokio::test]
        async fn admin_copilot_auth_returns_internal_error_when_bootstrap_fails_for_other_reason() {
            let _env_guard = crate::oauth::device_flow::ENV_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());

            let server = MockServer::start().await;
            std::env::set_var("LLMPROXY_TEST_GITHUB_BASE_URL", &server.uri());

            // GitHub returns 500 on the device-code endpoint. The error
            // message will not contain "already in progress", so the
            // handler takes the `else { e.into_response() }` path.
            Mock::given(method("POST"))
                .and(path("/login/device/code"))
                .respond_with(ResponseTemplate::new(500).set_body_string("upstream error"))
                .mount(&server)
                .await;

            let provider = new_copilot();
            let app = build_app_with_copilot(provider);

            let resp = app
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/admin/copilot/auth")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            std::env::remove_var("LLMPROXY_TEST_GITHUB_BASE_URL");

            // 500 from the upstream device-code endpoint is propagated
            // through `ProxyError::into_response`, not 409 (which is the
            // "already in progress" branch).
            assert_eq!(
                resp.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "non-conflict bootstrap failure must surface as 5xx, not 409"
            );
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["type"], "error");
        }

        #[tokio::test]
        async fn admin_models_copilot_group_reports_cache_state() {
            // Cold Copilot provider (private tempdir token store): the
            // catalog fetch returns None, so the group must degrade to
            // cache_state "cold" — routed through CopilotProvider's own
            // introspection rather than the generic "fetch_failed".
            let provider = new_copilot();

            let cfg = Config {
                server: ServerConfig {
                    listen: "127.0.0.1:0".to_string(),
                    api_key: None,
                },
                proxy: Default::default(),
                user_agent: crate::config::default_user_agent(),
                providers: vec![ProviderConfig::GithubCopilot {
                    name: "copilot".to_string(),
                    vscode_version: "1.95.0".to_string(),
                    account_type: "individual".to_string(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                }],
                models: vec![ModelConfig {
                    name: "m".to_string(),
                    primary: "copilot".to_string(),
                    fallback_chain: vec![],
                    cooldown_seconds: 60,
                    max_retries_per_provider: 1,
                    max_retries_total: 1,
                }],
                    ..Config::default()
                };
            let cfg = Arc::new(cfg);
            let cooldown = CooldownCache::new();
            let mut providers = HashMap::new();
            providers.insert("copilot".to_string(), provider.clone() as Arc<dyn Provider>);
            let router = Arc::new(Router::new(cfg.clone(), providers, cooldown.clone()));
            let app = crate::server::build_router(AppState {
                config: cfg,
                router,
                cooldown,
                http: reqwest::Client::new(),
                copilot: Some(provider),
                usage: UsageStats::new(8),
            });

            let resp = app
                .oneshot(
                    Request::builder()
                        .method(Method::GET)
                        .uri("/admin/models")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let group = &body["providers"][0];
            assert_eq!(group["provider"], "copilot");
            assert_eq!(group["cache_state"], "cold");
            assert_eq!(group["source"], "discovered");
        }
    }

    #[test]
    fn build_routing_priority_prefers_primary_and_translates_rewrite() {
        use crate::config::{Config, ModelConfig, ProviderConfig, ServerConfig};
        use crate::providers::Provider;
        use std::sync::Arc;

        // copilot has a non-empty rewrite table: client "claude" →
        // upstream "gpt-x". compat is a fallback with an empty table.
        let config = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: None,
            },
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "compat".to_string(),
                    api_key: "k".to_string(),
                    api_base: "https://x.test".to_string(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                    provider_ignore: Vec::new(),
                    reasoning_echo: false,
                },
                ProviderConfig::GithubCopilot {
                    name: "copilot".to_string(),
                    vscode_version: "1.95.0".to_string(),
                    account_type: "individual".to_string(),
                    model_rewrite: HashMap::from([
                        ("claude".to_string(), "gpt-x".to_string()),
                        ("other".to_string(), "gpt-y".to_string()),
                    ]),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "claude".to_string(),
                primary: "copilot".to_string(),
                fallback_chain: vec!["compat".to_string()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 1,
            }],
                ..Config::default()
            };
        let providers: HashMap<String, crate::providers::SharedProvider> = HashMap::from([
            (
                "copilot".to_string(),
                Arc::new(crate::providers::copilot::CopilotProvider::new(
                    "copilot".to_string(),
                    "1.95.0".to_string(),
                    "individual".to_string(),
                    config.providers[1].model_rewrite().clone(),
                    reqwest::Client::new(),
                )
                .unwrap()) as Arc<dyn Provider>,
            ),
            (
                "compat".to_string(),
                crate::providers::build(&config.providers[0], reqwest::Client::new()).unwrap(),
            ),
        ]);

        let priority = build_routing_priority(&config, &providers);
        // Client name translated to the upstream id for the rewritten
        // provider; primary gets position 0.
        assert_eq!(
            priority.get(&("gpt-x".to_string(), "copilot".to_string())),
            Some(&(0u64, 0u64))
        );
        // Untranslated fallback keeps the client name as id, position 1.
        assert_eq!(
            priority.get(&("claude".to_string(), "compat".to_string())),
            Some(&(1u64, 1u64))
        );
        // Rewrite keys not referenced by any model chain are absent.
        assert!(!priority.contains_key(&("gpt-y".to_string(), "copilot".to_string())));
    }

    #[test]
    fn build_routing_priority_skips_providers_that_cannot_serve() {        use crate::config::{Config, ModelConfig, ProviderConfig, ServerConfig};

        // copilot can only serve "other" (rewrite key); the chain asks for
        // "claude", which it cannot serve — so no entry for copilot at all,
        // while compat (empty rewrite = serve anything) is registered.
        let config = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: None,
            },
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![
                ProviderConfig::OpenaiCompat {
                    name: "compat".to_string(),
                    api_key: "k".to_string(),
                    api_base: "https://x.test".to_string(),
                    model_rewrite: HashMap::new(),
                    use_proxy: false,
                    provider_ignore: Vec::new(),
                    reasoning_echo: false,
                },
                ProviderConfig::GithubCopilot {
                    name: "copilot".to_string(),
                    vscode_version: "1.95.0".to_string(),
                    account_type: "individual".to_string(),
                    model_rewrite: HashMap::from([("other".to_string(), "gpt-x".to_string())]),
                    use_proxy: false,
                },
            ],
            models: vec![ModelConfig {
                name: "claude".to_string(),
                primary: "copilot".to_string(),
                fallback_chain: vec![],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 1,
            }],
                ..Config::default()
            };
        let providers: HashMap<String, crate::providers::SharedProvider> = HashMap::from([
            (
                "copilot".to_string(),
                crate::providers::build(&config.providers[1], reqwest::Client::new()).unwrap(),
            ),
            (
                "compat".to_string(),
                crate::providers::build(&config.providers[0], reqwest::Client::new()).unwrap(),
            ),
        ]);

        let priority = build_routing_priority(&config, &providers);
        assert!(priority.is_empty(), "copilot cannot serve claude");
    }

    #[test]
    fn build_routing_priority_takes_min_rank_on_duplicate_keys_and_skips_ghosts() {
        use crate::config::{Config, ModelConfig, ProviderConfig, ServerConfig};

        // copilot's rewrite maps both "m1" and "m2" to the same upstream
        // id "u". Row1 uses copilot as fallback (pos 1), row2 as primary
        // (pos 0) — the minimum position must win for ("u", "copilot").
        // Row1 also references a "ghost" provider absent from the map,
        // which must be skipped without panicking.
        let config = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".to_string(),
                api_key: None,
            },
            proxy: Default::default(),
            user_agent: crate::config::default_user_agent(),
            providers: vec![ProviderConfig::GithubCopilot {
                name: "copilot".to_string(),
                vscode_version: "1.95.0".to_string(),
                account_type: "individual".to_string(),
                model_rewrite: HashMap::from([
                    ("m1".to_string(), "u".to_string()),
                    ("m2".to_string(), "u".to_string()),
                ]),
                use_proxy: false,
            }],
            models: vec![
                ModelConfig {
                    name: "m1".to_string(),
                    primary: "ghost".to_string(),
                    fallback_chain: vec!["copilot".to_string()],
                    cooldown_seconds: 60,
                    max_retries_per_provider: 1,
                    max_retries_total: 1,
                },
                ModelConfig {
                    name: "m2".to_string(),
                    primary: "copilot".to_string(),
                    fallback_chain: vec![],
                    cooldown_seconds: 60,
                    max_retries_per_provider: 1,
                    max_retries_total: 1,
                },
            ],
                    ..Config::default()
                };
        let providers: HashMap<String, crate::providers::SharedProvider> = HashMap::from([(
            "copilot".to_string(),
            crate::providers::build(&config.providers[0], reqwest::Client::new()).unwrap(),
        )]);

        let priority = build_routing_priority(&config, &providers);
        // pos 0 from the m2 row wins over pos 1 from the m1 row; the seq
        // component reflects insertion order across config.models.
        assert_eq!(
            priority.get(&("u".to_string(), "copilot".to_string())),
            Some(&(0u64, 1u64)),
            "duplicate key must keep the minimum chain position"
        );
        // The ghost provider never lands in the table.
        assert_eq!(priority.len(), 1);
    }

    fn rank_of(
        priority: &HashMap<(String, String), (u64, u64)>,
        id: &str,
        provider: &str,
    ) -> EntryRank {
        entry_rank(
            &json!({"id": id, "owned_by": provider}),
            priority,
        )
    }

    #[test]
    fn dedup_by_priority_registered_beats_unregistered_and_static() {
        let mut priority = HashMap::new();
        priority.insert(("shared".to_string(), "primary".to_string()), (0u64, 0u64));

        let mut entries = vec![
            json!({"id": "shared", "owned_by": "llmproxy"}),
            json!({"id": "shared", "owned_by": "fallback"}),
            json!({"id": "shared", "owned_by": "primary"}),
        ];
        dedup_models_by_routing_priority(&mut entries, &priority);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["owned_by"], "primary");

        // Without any registered entry, an unregistered discovered entry
        // beats the static placeholder.
        let mut entries = vec![
            json!({"id": "shared", "owned_by": "llmproxy"}),
            json!({"id": "shared", "owned_by": "somewhere"}),
        ];
        dedup_models_by_routing_priority(&mut entries, &priority);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["owned_by"], "somewhere");

        // And with nothing else claiming the id, static survives.
        let mut entries = vec![json!({"id": "shared", "owned_by": "llmproxy"})];
        dedup_models_by_routing_priority(&mut entries, &priority);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["owned_by"], "llmproxy");
    }

    #[test]
    fn dedup_by_priority_smallest_chain_position_and_declaration_seq_win() {
        let mut priority = HashMap::new();
        // Same id served by two providers; "a" is primary (pos 0).
        priority.insert(("m".to_string(), "b".to_string()), (1u64, 3u64));
        priority.insert(("m".to_string(), "a".to_string()), (0u64, 2u64));
        // Cross-ModelConfig tie on position: smaller declaration seq wins.
        priority.insert(("n".to_string(), "y".to_string()), (0u64, 5u64));
        priority.insert(("n".to_string(), "x".to_string()), (0u64, 7u64));

        let mut entries = vec![
            json!({"id": "m", "owned_by": "b"}),
            json!({"id": "m", "owned_by": "a"}),
            json!({"id": "n", "owned_by": "x"}),
            json!({"id": "n", "owned_by": "y"}),
        ];
        dedup_models_by_routing_priority(&mut entries, &priority);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["owned_by"], "a");
        assert_eq!(entries[1]["owned_by"], "y");
    }

    #[test]
    fn dedup_by_priority_unregistered_pick_is_deterministic() {
        // No priority entries: two unregistered providers claim the same
        // id — the lexicographically smallest provider name must win so
        // the result does not depend on HashMap iteration order.
        let priority = HashMap::new();
        let mut entries = vec![
            json!({"id": "m", "owned_by": "zeta"}),
            json!({"id": "m", "owned_by": "alpha"}),
        ];
        dedup_models_by_routing_priority(&mut entries, &priority);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["owned_by"], "alpha");
    }

    #[test]
    fn dedup_by_priority_filters_empty_ids_and_keeps_first_seen_order() {
        let priority = HashMap::new();
        let mut entries = vec![
            json!({"id": ""}),
            json!({"not_id": "c"}),
            json!({"id": "b"}),
            json!({"id": "a"}),
            json!({"id": "b", "dup": true}),
        ];
        dedup_models_by_routing_priority(&mut entries, &priority);
        let ids: Vec<&str> = entries.iter().map(|e| e["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["b", "a"]);
    }

    #[test]
    fn entry_rank_orders_classes() {
        let mut priority = HashMap::new();
        priority.insert(("m".to_string(), "p".to_string()), (0u64, 0u64));
        let registered = rank_of(&priority, "m", "p");
        let unregistered = rank_of(&priority, "m", "q");
        let static_entry = rank_of(&priority, "m", "llmproxy");
        assert!(registered < unregistered);
        assert!(unregistered < static_entry);
    }

    // A slow upstream (wiremock delay) must not stall the endpoint: the
    // per-call timeout converts the hang into a degraded None well under
    // the shared client's 600 s default.
    #[tokio::test]
    async fn fetch_models_with_timeout_returns_none_on_hang() {
        use std::time::Duration;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(30))
                    .set_body_json(json!({"object": "list", "data": []})),
            )
            .mount(&server)
            .await;

        let provider = crate::providers::build(
            &crate::config::ProviderConfig::OpenaiCompat {
                name: "slow".to_string(),
                api_key: "k".to_string(),
                api_base: server.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
            reqwest::Client::new(),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let out = fetch_models_with_timeout(&provider, Duration::from_millis(100)).await;
        assert!(out.is_none(), "timed-out fetch must degrade to None");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "must return at the timeout, not after the upstream delay"
        );
    }
}

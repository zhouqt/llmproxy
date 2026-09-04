# Plan: drop `failed_providers` when no fallback, and surface token counts

> **Implementation note (post-review simplification).** The `EmptySkipping`
> custom formatter described in §A below was **removed** after a follow-up
> code review discovered that `tracing-core` ≥0.1.36 implements
> `impl<T: Value> Value for Option<T>` — a `None` records nothing, so the
> field is simply absent from the event under the **standard** `compact()`
> formatter. The plan's goals (single call site, `failed_providers` absent
> when no fallback, `cache_read_tokens` absent vs 0 distinguished) are all
> met natively: `log_request_completed` / `log_streaming_completed` are one
> `tracing::info!` each, passing `Option` values directly
> (`failed_providers = failed_providers.map(tracing::field::display)` keeps
> the unquoted wire shape). `src/empty_skipping_format.rs` is gone;
> `main.rs` installs the stock `.compact()` formatter. The §A design below
> is kept for the historical record of *why* the formatter was originally
> introduced.

## Context

Two follow-ups to the recently-merged log-format PR (#22).

### Problem 1 — noisy `failed_providers` field on healthy requests

Today the `request completed` log always emits `failed_providers=...`, even when
`attempts` is empty (i.e. the primary succeeded with no fallback). That makes
`grep failed_providers` useless as a fallback alert — every line has the field,
and operators have to eyeball each line to tell "actually fell back" from "ran
through the primary on the first try":

```
INFO request completed model="work-low" provider="cp" stream=true elapsed_ms=13956 failed_providers
```

### Problem 2 — no observability of token spend per call

Operators have no signal of how many tokens each request consumed beyond what
the client sees. No logging of input / output / total / cache_read. The proxy
is the choke point — every request flows through it, so it is the right place
to record the numbers.

### Intended outcome

1. `failed_providers` appears ONLY when a fallback actually happened. Grep
   returns real fallback events.
2. Each request emits token counts (`input_tokens`, `output_tokens`,
   `total_tokens`, `cache_read_tokens`) when the provider reported usage.
   Fields are absent when the upstream did not.
3. No new public API yet — a follow-up PR can expose usage via response
   headers or a dedicated endpoint. This PR stops at the log surface so we
   can validate field semantics against real traffic before exposing externally.

## User-confirmed decisions

- Log fields: `input_tokens`, `output_tokens`, `total_tokens`,
  `cache_read_tokens`. (Reasoning / cache_creation / server_tool_use deferred.)
- Streaming path: capture usage from the final usage chunk and emit at log
  time alongside `elapsed_ms`.

## Design

### A. Conditional `failed_providers` + token fields — single-macro `tracing::info!` with `Empty` sentinel

**Locked final form (after Sonnet + Opus reviews):** one `tracing::info!` macro
call driven by a helper that pre-resolves the values. Fields that should be
absent are recorded as `tracing::field::Empty` (the standard sentinel for "do
not print"). A small custom `FormatEvent` impl drops `Empty` fields on output.
The same `FormatEvent` is installed in both `main.rs` (production) and
`test_support.rs` (test capture), so operators and tests see identical wire
shape.

Why this form (not a 4-branch cascade):

- A 4-branch cascade duplicates the macro preamble four times. Adding a field
  like `request_id` would require touching four sites.
- A custom `FormatEvent` impl is a one-time investment in `src/main.rs` and
  `src/test_support.rs` — both stay ~20 lines — and the call site stays
  single-line forever.
- The earlier plan's "two macros" attempt had `cache_read_tokens` rendering
  as `0` for absence, which conflates "no cache hit" with "upstream reported
  0". The `Empty` approach cleanly distinguishes them.

#### A.1 The helper

```rust
// src/server.rs
pub struct LogTokens {
    pub input: u32,
    pub output: u32,
    pub cache_read: Option<u32>,
}

fn log_request_completed(
    model: &str,
    provider: &str,
    stream: bool,
    elapsed: std::time::Duration,
    failed_providers: Option<&str>,
    tokens: Option<&LogTokens>,
) {
    let total = tokens.map(|t| t.input + t.output);
    let cache_read = tokens.and_then(|t| t.cache_read);

    let span = tracing::info_span!("request", model = %model, provider = %provider);
    let _enter = span.enter();

    let failed_providers_field =
        failed_providers.unwrap_or(tracing::field::Empty);
    tracing::info!(
        stream = stream,
        elapsed_ms = elapsed.as_millis() as u64,
        failed_providers = failed_providers_field,
        input_tokens = tokens.map_or(tracing::field::Empty, |t| t.input),
        output_tokens = tokens.map_or(tracing::field::Empty, |t| t.output),
        total_tokens = total.map_or(tracing::field::Empty, |t| t as u64),
        cache_read_tokens = cache_read.map_or(tracing::field::Empty, |t| t),
        "request completed"
    );
}
```

That's the locked final form. Two `info!` macros are no longer needed.

**Operator note on `total_tokens` semantics (Opus M1):** `total = input + output`
follows Anthropic's convention (Anthropic `usage` has no top-level `total_tokens`;
clients sum `input_tokens + output_tokens`). This **excludes** `cache_read`.
Operators who want the OpenAI-equivalent `total_tokens` (prompt + completion,
where prompt includes cached) should add `cache_read_tokens` themselves. The
manual smoke section documents this. The semantics are consistent across all
three provider paths (OpenAI Chat Completions, OpenAI Responses, Anthropic
passthrough) because `build_usage` makes `input = prompt - cached` on the
conversion paths, and Anthropic's own spec defines `input_tokens` as
excluding cached on the passthrough path.

#### A.2 The FormatEvent impl

A custom `tracing_subscriber::fmt::FormatEvent` that records field values
into a visitor and, on write, skips any field whose recorded value was the
`tracing::field::Empty` sentinel. The trick: we wrap the visit pass in a
custom `Visit` impl that checks each recorded value for equality with
`tracing::field::Empty` (via `debug_assert_eq` against the debug form,
which is the sentinel's stable representation). On match, the field is
skipped; otherwise it is rendered with `format!("{name}={debug}")`.

Implementation lives in a new shared module `src/empty_skipping_format.rs`:

```rust
// src/empty_skipping_format.rs
use std::fmt::{self, Write};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::fmt::FmtContext;

pub struct EmptySkipping;

const EMPTY_SENTINEL: &str = "<empty field>"; // tracing::field::Empty's Debug form

impl<S, N> FormatEvent<S, N> for EmptySkipping
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // 1. Level prefix (matches `compact()` shape: `INFO ` etc.)
        write!(writer, "{} ", event.metadata().level())?;
        // 2. Message + span attrs are emitted by FormatFields on a sub-writer.
        //    We use `Format::default().format_event` for that, then layer our
        //    own field pass on top — or implement both inline. Below:
        //    render the message and span fields via the default formatter.
        let mut visitor = FieldSkipEmpty::new(writer.by_ref());
        event.record(&mut visitor);
        // 3. Append the message string at the end of the line.
        writeln!(writer)?;
        Ok(())
    }
}

struct FieldSkipEmpty<'w, W: Write> {
    writer: &'w mut W,
    skipped_any: bool,
}

impl<'w, W: Write> FieldSkipEmpty<'w, W> {
    fn new(writer: &'w mut W) -> Self { Self { writer, skipped_any: false } }
}

impl<'w, W: Write + Send + Sync> Visit for FieldSkipEmpty<'w, W> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        // tracing::field::Empty's Debug form is the literal "<empty field>".
        // Compare by formatted string; cheaper than trait hacking.
        if format!("{:?}", value) == EMPTY_SENTINEL {
            return;
        }
        // Skip the implicit `message` field — rendered separately.
        if field.name() == "message" {
            return;
        }
        let _ = write!(self.writer, " {}={:?}", field.name(), value);
        self.skipped_any = true;
    }
    // record_str / record_u64 / etc. all funnel through Debug via tracing's
    // field builder, so record_debug alone is sufficient.
}
```

The `<empty field>` literal is the stable `Debug` output of
`tracing::field::Empty` (verified against `tracing` 0.1.x). The
`record_debug` comparison is the *only* load-bearing piece: a naive impl
that `format!`s every recorded value would render `Empty` as the literal
`<empty field>` string, defeating the conditional-failure goal.

**Regression test (Sonnet C2 — required before merge):**

```rust
#[test]
fn empty_sentinel_field_is_suppressed() {
    use tracing_subscriber::fmt::MakeWriter;
    // ... install a buffer-backed writer + EmptySkipping layer ...
    tracing::info!(failed_providers = tracing::field::Empty, "x");
    let out = String::from_utf8(buf).unwrap();
    assert!(!out.contains("failed_providers="), "Empty field leaked: {out}");
    assert!(!out.contains("<empty field>"), "sentinel leaked: {out}");

    tracing::info!(failed_providers = "cp:429", "x");
    let out = String::from_utf8(buf).unwrap();
    assert!(out.contains("failed_providers=\"cp:429\""), "missing field: {out}");
}
```

`src/main.rs::init_tracing` switches from `.compact()` to
`.fmt_events(EmptySkipping).with_target(false)`. We drop `with_thread_ids`,
`with_span_events`, and colorization so operators see the same shape on TTY
and journalctl. `src/test_support.rs::capture_logs` installs the same
`FormatEvent`. **One subscriber layer only** (Sonnet m5): if a future
JSON-output layer is added, JSON consumers will see `failed_providers=null`
instead of absent — document this as known asymmetry in that layer's
README if/when it lands.

#### A.3 Operator grep workflow

Two completion surfaces — both required (Sonnet M2):

- `request completed` → emitted on the **non-streaming path only**.
  Absence of `failed_providers=` means no fallback. Presence of
  `failed_providers="..."` means fallback occurred. Carries token counts
  for non-streaming requests.
- `streaming completed` → emitted on the **streaming path only**, after
  body drain. Carries token counts (`input_tokens`, `output_tokens`,
  `total_tokens`, `cache_read_tokens`). **`streaming completed` does NOT
  carry `failed_providers`**; correlate with the earlier `request
  completed` line on the same request by `model + provider + stream`
  (the existing convention is to grep the model name + provider name).
- `streaming aborted (upstream error)` → emitted on the streaming path
  when the upstream errors mid-body. **No token fields** (Opus C1 —
  partial primary usage is discarded). The earlier `request completed`
  line on the same request will carry `failed_providers` reflecting
  that the streaming chain stopped on error.
- `all providers failed` → terminal failures (mutually exclusive with
  `request completed`; both should be greppable for distinct alerts).
- `cache_read_tokens=[1-9]` → cache hits on either path.

**Unified correlation pattern:** grep
`request completed|streaming completed|streaming aborted`
within a `model=` + `provider=` window — three completion event names
covering all success and error terminations.

**`elapsed_ms` (Sonnet m6):** `request completed`, `streaming completed`,
and `streaming aborted (upstream error)` all carry
`elapsed_ms = start.elapsed().as_millis() as u64`. Per-request latency
tracking is preserved on every path.

### B. Token extraction — non-streaming path

Non-streaming `Router::complete` returns `(ProviderOutput, Vec<RouteAttempt>)`.
The provider's JSON value already carries the Anthropic `usage` object on
`MessagesResponse.usage` after `serde_json::from_value(value)?` in
`messages_handler` (`src/server.rs:100`). Read directly:

```rust
fn extract_complete_tokens(resp: &MessagesResponse) -> Option<LogTokens> {
    Some(LogTokens {
        input: resp.usage.input_tokens,
        output: resp.usage.output_tokens,
        cache_read: resp.usage.cache_read_input_tokens,
    })
}
```

`resp.usage` is non-optional on `MessagesResponse` (Anthropic spec mandates
it). `cache_read_input_tokens` is `Option<u32>` on Anthropic; we preserve
that Option so `Empty` correctly distinguishes "upstream didn't report"
from "upstream reported 0".

**Invariant note (all three paths — Opus C3):** on OpenAI Chat Completions
and OpenAI Responses, the conversion layer calls `build_usage`
(`src/conversion/util.rs:288-312`) which sets
`input_tokens = prompt.saturating_sub(cached)` and
`cache_read_input_tokens = if cached > 0 { Some(cached) } else { None }`. On
Anthropic passthrough, upstream returns Anthropic's native `usage` shape
verbatim — and Anthropic's spec defines `input_tokens` as already
excluding cached, with `cache_read_input_tokens` as a separate field. The
result is **functionally identical** across all three paths: when the
plan's `extract_complete_tokens` reads `resp.usage.input_tokens` and
`resp.usage.cache_read_input_tokens`, both are post-cache-exclusion values.
`LogTokens::input + cache_read = prompt_total` is the invariant.

**Two round-trip unit tests (Opus C3):**

- `src/conversion/util.rs` test mod: OpenAI Chat with
  `prompt_tokens=10, cached_tokens=4` → `build_usage` →
  `MessagesResponse` → `extract_complete_tokens` → assert
  `input=6, cache_read=Some(4)`.
- `src/server.rs` test mod (new): construct a `MessagesResponse` with
  `input_tokens=6, cache_read_input_tokens=Some(4)` (the Anthropic passthrough
  shape) and assert `extract_complete_tokens` returns
  `LogTokens { input: 6, output: ..., cache_read: Some(4) }`.

**Third round-trip unit test (Sonnet m3):**

- `src/server.rs` test mod: construct a `MessagesResponse` with both
  `cache_read_input_tokens=Some(8)` AND `cache_creation_input_tokens=Some(2)`
  (the Anthropic native shape that flows through the passthrough path
  unchanged — `build_usage` drops `cache_creation_input_tokens` to `None`
  on the conversion path, so this case only exercises Anthropic-native
  responses). Assert that the field flows through to
  `MessagesResponse.usage.cache_creation_input_tokens = Some(2)` and is
  preserved (not silently dropped) by `serde_json::from_value`. **No log
  line carries `cache_creation_input_tokens`** in this PR — that's
  deferred (§H) — but the wire shape is not corrupted.

### C. Token extraction — streaming path (decoupled from `ProviderOutput`)

Two design decisions drive §C, both informed by the Opus review.

**Decision 1 (Opus M4) — `Provider::stream` gains an optional `StreamUsageSink`
parameter.** The Opus review confirmed the architectural blocker: the SSE
adapter is constructed inside each provider's `stream()` method, so the sink
must reach that construction point. Three options were considered:

- (a) `Provider::stream(req, model_rewrite, sink: Option<StreamUsageSink>)` —
  trait surface change; one new parameter, 4 provider impls each update one
  signature and one constructor call.
- (b) Store sink on provider struct — requires mutable access through
  `Arc<dyn Provider>` (hostile).
- (c) Refactor `Provider::stream` to return a raw byte stream so the router
  constructs the SSE adapter — major trait change, all 4 provider
  impls.

**Choose (a).** Minimal blast radius. The 4 provider impls
(`anthropic.rs`, `openai_compat.rs`, `openai_responses.rs`, `copilot.rs`)
each update:

1. Their `stream` impl signature (one parameter added).
2. The SSE adapter constructor call (one argument added).

Plus the trait declaration in `src/providers/mod.rs`. Plus the 5 mock sites
in `tests/` (signature update only).

**Trait signature change:**

```rust
// src/providers/mod.rs
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
        usage_sink: Option<StreamUsageSink>,  // NEW: None means no token capture
    ) -> Result<ProviderOutput>;
    // ... other methods unchanged ...
}
```

**Why `Option<...>` not `&StreamUsageSink`:** the sink needs to be cloned
into the SSE adapter (which outlives the call frame). Cloning the `Arc` is
cheap; passing by reference would force lifetimes through the SSE adapter's
generic. `Option` keeps the call site ergonomic at the router level
(`usage_sink.unwrap_or_default()` if needed) and at the test level
(`None`).

**Decision 2 (Opus C1 / Sonnet C1) — partial-usage-on-stream-error must be
discarded.** Today's `MappedStream::poll_next` (lines 218-249 of
`src/server.rs`) treats the inner-error path and the success-EOF path
identically: both set `done = true` and return `Ready(None)`. This is
**wrong for token attribution**: `StreamTranslator::push_chunk` updates
`final_usage` on **every** chunk that carries a `usage` field (line
84-86 of `src/conversion/stream.rs`), including mid-stream usage chunks.
So if primary emits a usage chunk mid-body then errors, the sink
already holds primary's partial usage. Without tracking errored
state, the `streaming completed` line would log primary's tokens even
though the client received a synthetic `event: error` chunk and zero
usable bytes.

**Fix:** `MappedStream` no longer has `done: bool` and `errored: bool`
separately. They are collapsed into a single `phase: MappedPhase`
enum (Sonnet M3 — eliminates the two-flag bug class). When the inner
stream returns `Err(...)`, the synthetic `event: error` chunk is
emitted and `self.usage.take()` is called to discard the partial
primary usage; the callback fires `MappedCompletion::Errored` (which
carries no usage). When the inner stream returns `Ready(None)`,
`self.usage` is populated from the shared `usage_watch` and the
callback fires `MappedCompletion::Success(Some(usage))`. The decision
**discards** the partial usage on principle: when a stream terminates
with an upstream error, the proxy cannot prove the upstream charged
the request, the client received no usable response, and there is
**no fallback** (per the streaming contract — once bytes flow, the
chain stops). Logging primary's partial tokens would silently
misattribute cost.

```rust
// src/server.rs::MappedStream — final shape (see §D for phase semantics)
pub struct MappedStream {
    pub provider: String,
    pub model: String,
    pub inner: Pin<Box<dyn Stream<Item = Result<Bytes, ProxyError>> + Send>>,
    pub phase: MappedPhase,                                            // Streaming | Done(...)
    pub usage: Option<StreamUsage>,                                    // populated by usage_watch
    pub on_complete: Option<SingleFireCallback>,                       // see §D FnOnce workaround
}

enum MappedCompletion {
    Success(Option<StreamUsage>),  // None = upstream didn't report usage (Sonnet M4)
    Errored,                       // partial usage discarded (Opus C1)
}
```

`phase` transitions to `MappedPhase::Done(MappedCompletion::Errored)` in
the `Ready(Some(Err(_)))` arm (line 224-241 of current `server.rs`) when
`MappedStream` emits the synthetic `event: error` chunk. On success-EOF
the arm sets `phase = MappedPhase::Done(MappedCompletion::Success(...))`.

#### C.1 Sink + StreamUsage types

```rust
// src/providers/mod.rs
#[derive(Clone, Debug, Default)]
pub struct StreamUsage {
    pub input: u32,
    pub output: u32,
    pub cache_read: Option<u32>,
}

#[derive(Clone, Default)]
pub struct StreamUsageSink {
    inner: Arc<Mutex<Option<StreamUsage>>>,
}

impl StreamUsageSink {
    pub fn empty() -> Self { Self::default() }
    pub fn from_arc(inner: Arc<Mutex<Option<StreamUsage>>>) -> Self {
        Self { inner }
    }
    pub fn set(&self, usage: StreamUsage) {
        let mut g = self.inner.lock().expect("StreamUsageSink mutex poisoned");
        *g = Some(usage);
    }
    pub fn arc(&self) -> Arc<Mutex<Option<StreamUsage>>> {
        self.inner.clone()
    }
}
```

The `arc()` accessor lets `messages_handler` clone the inner `Arc` and
pass one clone to `Router::stream` (which forwards it into the SSE
adapter for writes) and another clone into `MappedStream::with_callback`
(for reads at `Ready(None)` time) — sharing exactly one underlying
state cell between writer and reader (Sonnet M5). `ProviderOutput::Stream`
keeps its single-byte-stream variant untouched (no change to the enum).

#### C.2 SSE adapter terminal-path sink writes

`OpenAiSseToAnthropic::new(inner, model, sink)` and
`ResponsesSseToAnthropic::new(inner, model, sink)` take an `Arc<...>` clone
of the sink. On three terminal paths:

1. `[DONE]` sentinel (line 599 / 248 of the two adapters).
2. Inner `Ready(None)` (line 686 / 197).
3. Error-envelope detection (line 615).

…the adapter calls `translator.take_final_usage()` and converts the result
to `StreamUsage` via `build_usage` (the same path `StreamTranslator::finalize`
uses at line 157-177 of `src/conversion/stream.rs`), then `sink.set(...)`.

If `take_final_usage()` returns `None`, the sink stays empty — `set` is not
called. The `streaming completed` log line will then have all four token
fields absent (via `Empty`).

**`build_usage` signature change for sink-write path (Sonnet m2):**
`build_usage` currently consumes `service_tier: Option<String>` by value.
If the SSE adapter holds `service_tier` from an earlier chunk (typical:
`chat.completion` events include it), calling `build_usage` from the
terminal sink-write path would move the tier and break the
**response-emission** path that also needs it. Two options:

- (a) Change `build_usage` to take `service_tier: &Option<String>` and
  clone internally — minimal blast radius; one signature update; the
  three call sites adjust to `&u.service_tier`.
- (b) Capture `service_tier` into a sink-owned field on the SSE adapter
  and assemble the final `Usage` only in the response-emission path,
  not the sink-write path — heavier refactor.

**Choose (a).** Minimal blast radius; the cloned `String` is one short
allocation per terminal write (negligible at end-of-stream cost).

```rust
// src/conversion/util.rs
pub fn build_usage(
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    reasoning_tokens: u32,
    service_tier: &Option<String>,   // CHANGED: by reference
) -> Usage {
    // ... existing body, using service_tier.clone() internally ...
}
```

Three call-site adjustments (`OpenAiSseToAnthropic`, `ResponsesSseToAnthropic`,
`PassthroughSse`) — each changes `service_tier` argument to `&u.service_tier`
where `u` is the upstream usage struct.

#### C.3 Anthropic passthrough scanner (Opus M2 / Sonnet M6)

The passthrough adapter (`PassthroughSse` at `src/providers/anthropic.rs:486`)
forwards raw bytes without a translator. The `message_delta` SSE event on the
wire carries a `usage` object (Anthropic spec). A line-scanner in
`PassthroughSse` extracts it. **The scanner mirrors the
`OpenAiSseToAnthropic::process_lines` pattern exactly** — `BytesMut` pending
buffer, `\r\n` aware (Sonnet M6 — Anthropic SSE uses `\r\n` line endings and
a naive `\n`-only split leaves `\r` artifacts in JSON parse on the
client-side), multi-line `data:` concatenation:

```rust
pub struct PassthroughSse<S> {
    inner: S,
    sink: StreamUsageSink,
    pending: BytesMut,        // mirror openai_compat.rs line-buffer pattern
    finished: bool,
    last_event: Option<String>,  // tracks "event: message_delta" between byte chunks
    last_usage: Option<Usage>,   // populated when we see event+data with usage
}

impl PassthroughSse {
    pub fn new(inner: S, sink: StreamUsageSink) -> Self { ... }
    fn process_lines(&mut self) {
        loop {
            // Find next line boundary (\n OR \r\n — Sonnet M6).
            let newline_pos = self.pending.iter().position(|&b| b == b'\n');
            let Some(nl) = newline_pos else { break; };
            // Strip the trailing \n and any \r immediately before it.
            let line_end = if nl > 0 && self.pending[nl - 1] == b'\r' { nl - 1 } else { nl };
            let line = self.pending.split_to(nl + 1);
            let line = std::str::from_utf8(&line[..line_end]).unwrap_or("");
            // 1. Track `event:` lines into self.last_event.
            // 2. On `data:` lines, accumulate the JSON payload. If
            //    self.last_event == Some("message_delta"), parse JSON; if it
            //    has a "usage" object, capture into self.last_usage.
            // 3. Reset self.last_event on empty lines (SSE event boundary).
        }
    }
}

// In poll_next's Ready(None) and Ready(Some(Err(_))) arms:
self.last_usage.take()
    .map(|u| StreamUsage {
        input: u.input_tokens,
        output: u.output_tokens,
        cache_read: u.cache_read_input_tokens,
    })
    .map(|s| self.sink.set(s));
```

**Required unit test (Sonnet M6):** feed the scanner a
`\r\n`-delimited Anthropic SSE byte stream (`event: message_delta\r\ndata:
{"usage":{...}}\r\n\r\n`) and assert the captured usage equals the input
JSON. Without `\r` trimming, the JSON parse fails with trailing `\r` and
the usage is silently dropped.

#### C.4 Translator accessors

```rust
// src/conversion/stream.rs
impl StreamTranslator {
    /// Take the most recent `final_usage` the translator has observed.
    /// Returns None if no chunk has carried a `usage` field.
    pub fn take_final_usage(&mut self) -> Option<ChatUsage> {
        self.final_usage.take()
    }
}

// src/conversion/responses_stream.rs
impl ResponsesStreamTranslator {
    pub fn take_final_usage(&mut self) -> Option<ResponsesUsage> {
        self.final_usage.take()
    }
}
```

The SSE adapter calls `t.take_final_usage()` after `t.finalize()` and
writes the resulting Anthropic-shaped `Usage` into the sink. Conversion
to `StreamUsage`:

```rust
let anthropic_usage = match openai_usage {
    Some(u) => build_usage(
        u.prompt_tokens,
        u.completion_tokens,
        u.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens).unwrap_or(0),
        u.completion_tokens_details.as_ref().and_then(|d| d.reasoning_tokens).unwrap_or(0),
        u.service_tier.clone(),
    ),
    None => return,
};
sink.set(StreamUsage {
    input: anthropic_usage.input_tokens,
    output: anthropic_usage.output_tokens,
    cache_read: anthropic_usage.cache_read_input_tokens,
});
```

### D. `MappedStream` lifecycle — callback fires synchronously in the `Ready(None)` poll (Sonnet C1)

> **Sonnet C1 (blocker — fixed).** The earlier "fire on the next poll's
> short-circuit" design was wrong: `axum::body::Body::from_stream` wraps the
> stream with `http_body_util::StreamBody`, which polls the inner stream
> exactly once and forwards `Ready(None)` to axum as the body terminator.
> There IS no "next poll" on success-EOF. Result with the old design: the
> `streaming completed` log line never fires for the dominant case
> (a clean stream that ends with `[DONE]`). Operators grepping for it
> see it only on errored streams — breaking the operator workflow that
> motivated this plan.
>
> **Fix:** fire the callback **synchronously in the same poll that returns
> `Ready(None)`** on success-EOF, before returning. The errored path fires
> in the same place (after emitting the synthetic `event: error` chunk).
> The `on_complete` field is `Option::take()`-guarded so re-polls
> (including any defensive double-poll from `StreamBody`) are no-ops.

`MappedStream` state is collapsed to a single phase enum (Sonnet M3 —
eliminates the bug-prone `done: bool` + `errored: bool` two-flag state
machine):

```rust
enum MappedPhase {
    Streaming,
    Done(MappedCompletion),  // both errored and success-EOF land here
}

// In MappedStream — new field shape
pub struct MappedStream {
    pub provider: String,
    pub model: String,
    pub inner: Pin<Box<dyn Stream<Item = Result<Bytes, ProxyError>> + Send>>,
    pub phase: MappedPhase,    // replaces done + errored
    pub usage: Option<StreamUsage>,
    pub on_complete: Option<Box<dyn FnOnce(MappedCompletion) + Send>>,
}

enum MappedCompletion {
    Success(Option<StreamUsage>),   // None = upstream didn't report usage
    Errored,
}
```

`MappedCompletion::Success` now carries `Option<StreamUsage>` (Sonnet M4):
the `Success(StreamUsage::default())` choice misled operators into
reading "0 tokens" when the actual situation is "no usage available".
The log macro emits token fields only when `Some`. This cleanly
distinguishes "0 tokens used" from "upstream didn't report usage."

**`poll_next` updated shape:**

```rust
fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
    // Phase terminal: fire callback once, then short-circuit forever.
    if let MappedPhase::Done(completion) = &self.phase {
        if let Some(cb) = self.on_complete.take() {
            let cb_result = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| cb(completion.clone())),
            );
            if let Err(e) = cb_result {
                tracing::error!(
                    provider = %self.provider,
                    model = %self.model,
                    panic = ?e,
                    "request_completed callback panicked; swallowed to preserve body-sink task",
                );
            }
        }
        return Poll::Ready(None);
    }
    match self.inner.as_mut().poll_next(cx) {
        Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
        Poll::Ready(Some(Err(e))) => {
            tracing::error!(
                provider = %self.provider,
                model = %self.model,
                error = %e,
                "upstream stream error",
            );
            // Synthetic `event: error` chunk. Drain the sink so the
            // callback below sees the Errored variant with no usage
            // (Sonnet C1 / Opus C1 — partial-usage-on-error is discarded).
            let _ = self.usage.take();
            let chunk = format_stream_error(&e);
            self.phase = MappedPhase::Done(MappedCompletion::Errored);
            // Fire callback synchronously in this same poll, before
            // returning the synthetic chunk — ensures the log line
            // emits even if this is the final poll.
            if let Some(cb) = self.on_complete.take() {
                let cb_result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| cb(MappedCompletion::Errored)),
                );
                if let Err(e) = cb_result {
                    tracing::error!(
                        provider = %self.provider,
                        model = %self.model,
                        panic = ?e,
                        "request_completed callback panicked; swallowed to preserve body-sink task",
                    );
                }
            }
            Poll::Ready(Some(Ok(chunk)))
        }
        Poll::Ready(None) => {
            // Success-EOF. Pull usage from MappedStream's owned field —
            // SSE adapters write to it via a write-back closure (see §D.1
            // below). Fire callback synchronously, then return.
            let usage = self.usage.take();
            self.phase = MappedPhase::Done(MappedCompletion::Success(usage));
            if let Some(cb) = self.on_complete.take() {
                let cb_result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| cb(MappedCompletion::Success(usage))),
                );
                if let Err(e) = cb_result {
                    tracing::error!(
                        provider = %self.provider,
                        model = %self.model,
                        panic = ?e,
                        "request_completed callback panicked; swallowed to preserve body-sink task",
                    );
                }
            }
            Poll::Ready(None)
        }
        Poll::Pending => Poll::Pending,
    }
}
```

**`MappedStream::with_callback` constructor:**

```rust
impl MappedStream {
    pub fn with_callback(
        provider: impl Into<String>,
        model: impl Into<String>,
        inner: impl Stream<Item = Result<Bytes, ProxyError>> + Send + 'static,
        on_complete: impl FnOnce(MappedCompletion) + Send + 'static,
        // usage_watch: how SSE adapter signals "I have final usage" — see §D.1
        usage_watch: Arc<Mutex<Option<StreamUsage>>>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            inner: Box::pin(inner),
            phase: MappedPhase::Streaming,
            usage: None,  // populated lazily via Arc<Mutex<Option<StreamUsage>>>
            on_complete: Some(Box::new(on_complete)),
        }
    }
}
```

**Send + FnOnce bound check (Sonnet b.1):** `Box<dyn FnOnce(MappedCompletion)
+ Send + 'static>` does not compile in stable Rust — `dyn FnOnce` is
`!Send`. The standard workaround is to wrap the `FnOnce` in a sync
closure that calls a `FnMut` internally, or use `Arc<dyn Fn(MappedCompletion)
+ Send + Sync>` with `Option::take()`-based single-fire guarantee
inside the wrapper. We use the second: a small `OnceCell`-style wrapper
that allows the trait-object signature to be `dyn Fn(MappedCompletion) +
Send + Sync + 'static` while the wrapper's `call(completion)` is
idempotent — re-calls are silently dropped.

```rust
type SingleFireCallback = Arc<dyn Fn(MappedCompletion) + Send + Sync + 'static>;
// Wrapped by a builder that ensures the inner closure runs at most once
// via a `Mutex<Option<Box<dyn FnOnce(MappedCompletion) + Send>>>`.
```

This bounds-namespace issue only matters for compilation; the design is
otherwise unchanged.

#### D.1 SSE adapter → MappedStream usage handoff (Sonnet M5)

The earlier design had SSE adapters writing directly into a sink that
`MappedStream` then read at `Ready(None)` time. Sonnet M5 flagged that
ordering ambiguity: if the SSE adapter's terminal-path write happens
*after* `MappedStream` has already returned `Ready(None)` (because the
adapter is dropped one frame later), the callback would see empty
usage even though it was captured.

**Fix:** the sink is `Arc<Mutex<Option<StreamUsage>>>` shared between
SSE adapter (writer) and `MappedStream::with_callback`'s
`usage_watch` (reader). The SSE adapter writes on **every terminal
event** (`[DONE]`, error envelope) so the value is materialized
**before** the inner stream returns `Ready(None)`. `PassthroughSse`
(which has no `[DONE]` sentinel, only byte-EOF) writes on its own
`Ready(None)` arm — the same poll that returns `Ready(None)` to
`MappedStream`, but the write happens before the return statement,
so the lock acquisition is complete by the time `MappedStream`
reads.

Concretely, the `usage_watch` Arc is shared:

```rust
let usage_watch = Arc::new(Mutex::new(None::<StreamUsage>));
let sink_for_sse = usage_watch.clone();
let stream = state.router.stream(
    &model_cfg, &req,
    Some(StreamUsageSink::new(usage_watch)),
).await?;
// ... adapter writes into `sink_for_sse` on its terminal paths ...
let mapped = MappedStream::with_callback(
    provider_name, model, inner,
    /* on_complete closure */,
    usage_watch,  // MappedStream reads this on Ready(None)
);
```

`MappedStream::poll_next`'s `Ready(None)` arm reads
`usage_watch.lock().unwrap().take()` and stores it in
`self.usage`. The callback later takes `self.usage` (which is
already populated). Ordering is deterministic: the SSE adapter's
write happens strictly before the `Ready(None)` return.

**Closure does NOT capture the sink (Sonnet m1).** The
`MappedStream::with_callback` closure captures only what it needs to
emit the log line: model, provider, start instant, and the variant
discriminant of `MappedCompletion`. The shared `usage_watch` `Arc`
is passed as a separate argument to `with_callback`; the SSE adapter
keeps its own clone of the same `Arc` via `StreamUsageSink::arc()`
(§C.1). The closure's environment carries no `StreamUsageSink`
clone — that would be dead weight since the SSE adapter already
keeps the sink alive across the stream's lifetime via its owned
`inner` field.

**`elapsed_ms` preserved on all completion paths (Sonnet m6).** The
`request completed` (non-streaming), `streaming completed`, and
`streaming aborted (upstream error)` lines all carry
`elapsed_ms = start.elapsed().as_millis() as u64`. Per-request
latency tracking is preserved end-to-end; the operator never loses
visibility into request duration, even on errored streams.

**Trait surface compatibility (Sonnet a.1).** `Provider::stream`'s
new `usage_sink: Option<StreamUsageSink>` parameter is a breaking
change to any external consumer of `llmproxy` as a library (the
`pub mod providers;` re-exports the `Provider` trait). For this
repo — which compiles `llmproxy` only as a binary — the breakage
is contained to the 4 in-tree provider impls + 9 mock sites. If the
project later publishes as a library, prefer option (b) from the
Sonnet review: add a `stream_with_sink` method alongside the
existing `stream` (which forwards with `None`), preserving the
old signature. **No action in this PR.**

### E. Files to change

| File | Change |
|---|---|
| `src/providers/mod.rs` | Add `StreamUsage` struct + `StreamUsageSink`. Update `Provider::stream` trait signature to add `usage_sink: Option<StreamUsageSink>`. `ProviderOutput::Stream` stays unchanged. |
| `src/empty_skipping_format.rs` (NEW) | `EmptySkipping` `FormatEvent` impl. ~60 lines. |
| `src/main.rs` | `init_tracing()` uses `EmptySkipping`. Drop `with_thread_ids`, `with_span_events`, colorization for stable operator shape. |
| `src/test_support.rs` | `capture_logs()` installs `EmptySkipping` FormatEvent on a buffer-backed writer. |
| `src/conversion/stream.rs` | Add `StreamTranslator::take_final_usage(&mut self) -> Option<ChatUsage>` (public). |
| `src/conversion/responses_stream.rs` | Add `ResponsesStreamTranslator::take_final_usage(&mut self) -> Option<ResponsesUsage>` (public). |
| `src/conversion/util.rs` | New unit test: Chat→Anthropic round-trip pins `input = prompt - cached` invariant. |
| `src/providers/openai_compat.rs` | `OpenAiSseToAnthropic::new(inner, model, sink)`. Update `stream(req, model_rewrite, sink)` impl signature. After `[DONE]`, `Ready(None)`, and `looks_like_error_envelope`, write captured `ChatUsage` into the sink via `build_usage`. |
| `src/providers/openai_responses.rs` | `ResponsesSseToAnthropic::new(inner, model, sink)`. Same terminal-path writes for `ResponsesUsage`. Update `stream` signature. |
| `src/providers/anthropic.rs` | `PassthroughSse::new(inner, sink)`. Update `stream` signature. Replace struct-literal `PassthroughSse { inner }` with constructor calls (3 test sites). Add `BytesMut`-based line scanner mirroring `OpenAiSseToAnthropic::process_lines` (Opus M2). On `Ready(None)` and inner-error paths, write captured usage into the sink. |
| `src/providers/copilot.rs` | Update both `stream` impl signatures (Chat at 1151, Responses at 973). Pass sink to inner SSE adapter constructor. |
| `src/router.rs` | Update `Router::stream` to take `usage_sink: Option<StreamUsageSink>` and forward it to `provider.stream(...)`. No new `stream_with_sink` method (Opus C2 — eliminates the parallel-method duplication). |
| `src/server.rs` | `messages_handler`: extract tokens from `MessagesResponse` on the non-streaming path; pass to `log_request_completed`. On the streaming path, build sink, call `Router::stream(req, sink)`, construct `MappedStream::with_callback` whose callback emits `streaming completed` (success path) or `streaming aborted (upstream error)` (errored path, no token fields). New helper `log_request_completed` (single-macro form per §A.1). New helper `extract_complete_tokens(&MessagesResponse) -> Option<LogTokens>`. New struct `LogTokens`. New `MappedCompletion` enum. |
| `src/server.rs::MappedStream` | Replace `done: bool` + `errored: bool` with `phase: MappedPhase` enum (Sonnet M3). Add `usage: Option<StreamUsage>`, `on_complete: Option<SingleFireCallback>`. New constructor `with_callback(provider, model, inner, on_complete, usage_watch)`. Existing `new()` keeps working (test sites pass `None`/empty for new fields). `poll_next`'s `Ready(Some(Err(_)))` arm sets `phase = Done(Errored)`. The `Ready(None)` arm and the `Ready(Some(Err(_)))` arm both fire callback synchronously (Sonnet C1). |

**Test-side struct-literal updates** (auto-resolved because new fields are `Option` / bool-default):

- `src/server.rs:685, 719, 735, 755, 787` — `MappedStream { provider, model, inner, done }` continues to work; the existing `new()` constructor builds with `phase: MappedPhase::Streaming` and `usage`/`on_complete` default to `None`. **No struct-literal changes required.**
- `src/providers/anthropic.rs:971, 979, 988` — `PassthroughSse { inner }` becomes `PassthroughSse::new(inner, StreamUsageSink::empty())` (3 sites; ~3 line changes).
- `Provider::stream` impl updates: 4 provider impls (`anthropic.rs`, `openai_compat.rs`, `openai_responses.rs`, `copilot.rs`) each get one new parameter in their `stream` method signature. **9 mock sites total** (Sonnet M1): 3 in `tests/` (`tests/server.rs::TestProvider` at line 73-91, `tests/integration_router.rs::WiremockOpenAiProvider` at 82-126, `tests/integration_router.rs::DeferredProvider` at 1397-1434) and **6 in `src/router.rs` test mod** (`MockProvider`, `CountingMockProvider`, `NonCooldownProvider`, `RestrictedMockProvider`, `ModelUnsupportedProvider`, `StreamingMockProvider`). Of those 9, only 2 are streaming-capable (`TestProvider`, `StreamingMockProvider`) and genuinely need to thread the sink through to a fake SSE stream that emits `[DONE]`; the other 7 just pass `None` to satisfy the signature.

### F. Reused functions and patterns

- `build_usage` (`src/conversion/util.rs:288`) — canonical Anthropic
  `Usage` constructor. Used by SSE adapters' terminal-path sink writes
  to convert `ChatUsage` / `ResponsesUsage` to Anthropic shape.
- `format_attempts` / `format_attempts_summary` (`src/server.rs:136-151`)
  — unchanged. We just stop emitting the resulting string when
  `attempts` is empty (now via `Empty` sentinel).
- `summarize_for_log` (`src/util.rs`) — no new truncation needed.
- `expect_variant!` macro (`src/lib.rs`) — for any new match-style tests.
- `JsonFieldAbsent` matcher pattern from `tests/server.rs` — for asserting
  the new field absence in wiremock request bodies.

### G. Testing strategy

> Coverage target stays >97% regions
> (`cargo llvm-cov --lib --bins --tests`).

1. **Unit tests (`src/server.rs` test mod):**
   - `log_request_completed` matrix: four input combos
     (`failed_providers` Some/None × `tokens` Some/None). For each,
     install `capture_logs` + `EmptySkipping` formatter, run the helper,
     assert the captured lines contain exactly the expected field set.
     Pin the Empty-skipping convention (`failed_providers=` field absent
     when no fallback; token fields absent when no usage).
   - `extract_complete_tokens` two-path test (Opus C3): one Chat-shaped
     input asserting `input=6, cache_read=Some(4)`; one Anthropic-shaped
     input asserting the same.
   - `MappedStream` tests (existing 5) gain branches to cover:
     - Success EOF with sink preloaded → callback fires
       `MappedCompletion::Success(usage)`, `streaming completed` line
       carries the captured usage.
     - Success EOF with empty sink → callback fires
       `MappedCompletion::Success(StreamUsage::default())`, `streaming
       completed` line has all token fields absent.
     - **Inner error mid-stream after usage chunk emitted** (Opus C1):
       construct an SSE adapter that emits a `usage` chunk then returns
       `Err(...)` mid-body; assert `errored == true` after the synthetic
       error chunk; assert the callback fires `MappedCompletion::Errored`;
       assert `streaming completed` is NOT emitted; assert the partial
       usage is discarded (sink holds `None`).
     - Double-poll idempotency: after callback fires once, a second poll
       on `done == true` is a no-op (callback already taken) (Opus M3).
     - Callback panic — wrapped in `catch_unwind`; assert the body-sink
       task is not torn down and the panic shows up in `tracing::error!`.
2. **Translator accessor tests:**
   - `StreamTranslator::take_final_usage` returns `None` when no usage
     chunk has been pushed, and `Some(...)` after a chunk carrying
     `usage` is processed. **Idempotent on multiple takes.**
   - Mirror for `ResponsesStreamTranslator`.
3. **Passthrough scanner tests (`src/providers/anthropic.rs` test mod,
   Opus M2):**
   - `event: message_delta\ndata: {"usage": {...}}\n\n` (full chunk)
     → on `Ready(None)`, sink holds the captured usage.
   - Same event split across two byte chunks (line `event:` in chunk 1,
     `data: {...}` in chunk 2) → sink still captures correctly.
   - `data: {"usage": {...}}` with `\r\n` line endings → captured.
   - `event: content_block_delta\ndata: {...}\n\n` (different event) →
     sink stays empty.
   - Malformed JSON in the `data:` line → scanner ignores it (no panic,
     sink stays empty).
   - `cache_read_input_tokens` absent in JSON → sink's
     `StreamUsage.cache_read` is `None`.
4. **Integration tests (`tests/server.rs`):** the eight required
   scenarios (Sonnet i — every name and assertion pinned):

   **Span inheritance (Sonnet m4):** every `streaming completed` /
   `streaming aborted (upstream error)` log line must carry
   `model=<name> provider=<name>` fields matching the
   `request_completed` line on the same request. The
   `MappedStream::with_callback` closure is built inside the
   `messages_handler` function and inherits its `tracing::Span` scope;
   tests assert both log lines share the same `model` and `provider`
   values to verify the closure did not escape the span context.

   1. **Primary success, no fallback (non-streaming)** — assert
      `request completed` line is present, `failed_providers=` field
      absent from that line, `input_tokens`/`output_tokens`/
      `total_tokens` present.
   2. **Primary 429 → fallback success (non-streaming)** — assert
      `request completed` line carries `failed_providers="cp:429"`,
      token counts present, `provider="<fallback name>"`.
   3. **Primary success, no fallback (streaming)** — assert
      `streaming completed` line is present (this is the test that
      catches the Sonnet C1 regression), `failed_providers=` absent,
      token counts present, **no `request completed` line** on this
      path (per §A.3 split).
   4. **Primary mid-stream usage chunk then error** (Opus C1 / Sonnet
      C1) — assert `streaming completed` is NOT emitted; assert
      `streaming aborted (upstream error)` IS emitted with no token
      fields; assert the partial primary usage is not logged.
   5. **All providers 429** — assert `all providers failed` carries
      `failed_providers="cp:429,backup:429"`; assert neither
      `request completed` nor `streaming completed` is emitted.
   6. **Cache hit on fallback** — assert `streaming completed`
      `cache_read_tokens` refers to the fallback provider's cache
      (verified via the fallback's wiremock returning `usage` with
      `cache_read_input_tokens: Some(N)`).
   7. **Provider A model_rewrite skip → B cooldown skip → C serves**
      — assert `streaming completed` line names provider C; assert
      `failed_providers="a:bypass,b:cooldown"` (or equivalent) on
      the corresponding `request completed` line.
   8. **Fallback path produces no usage** (e.g., Copilot endpoint
      that omits the trailing usage chunk) — assert `streaming
      completed` line has all token fields absent; assert the line
      still emits with `provider` and `elapsed_ms`.
5. **Coverage:** run
   `cargo llvm-cov --lib --bins --tests`. New branches enumerated
   above (>=19: 4 log_request_completed, 2 extract_complete_tokens
   paths, 5 MappedStream, 2 translator accessors, 6 passthrough
   scanner, 8 integration). Add unit tests until >97% regions covered.

### H. Out of scope (deferred)

- No `cache_creation_input_tokens` / reasoning_tokens /
  server_tool_use in this PR — user-confirmed scope is
  `input / output / total + cache_read`.
  **However, this is a known follow-up.** `cache_creation_input_tokens`
  distinguishes `ephemeral_1h_input_tokens` (5× the cost of 5m) from
  `ephemeral_5m_input_tokens`, which is load-bearing for cost
  attribution. Tracked as a follow-up PR; will land **before** this
  PR ships to prod traffic with significant Anthropic passthrough
  share. Note: on the conversion path `build_usage` already drops
  `cache_creation_input_tokens` (line 300 of `src/conversion/util.rs`
  sets it to `None`); on the Anthropic passthrough path it flows
  through upstream unchanged but is not surfaced in the log line.
- No `service_tier` field on `LogTokens` (Opus m7): thrown away
  intentionally; not operator-actionable. If we ever need it, the
  field can be added without changing the wire shape (operators
  default to absent and learn to ignore).
- No new public API (header / endpoint) for usage — to be designed
  after real-traffic validation.

## Verification

### Build

```bash
cargo build --release
cargo test --lib --bins --tests
cargo llvm-cov --lib --bins --tests  # >97% regions
```

### Manual smoke

```bash
RUST_LOG=llmproxy=info,llmproxy=trace cargo run --release -- --config config.yaml &
PROXY=http://127.0.0.1:9999

# 1. Healthy request — `failed_providers` field absent
curl -sS -X POST $PROXY/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: '$LLMPROXY_API_KEY \
  -d '{"model":"work-mini","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}' \
  >/dev/null
# Expect:
#   INFO request completed model="work-mini" provider="..." stream=false elapsed_ms=...
#     (no `failed_providers=` substring)
#   INFO streaming completed model="work-mini" provider="..." elapsed_ms=... input_tokens=N output_tokens=N total_tokens=N
#     (cache_read_tokens= absent unless a cache hit occurred)
#   `total_tokens = input + output` (Anthropic semantics); for OpenAI-equivalent
#   `total_tokens` (prompt + completion, includes cache), add cache_read_tokens.

# 2. Fallback — primary 429
curl -sS -X POST $PROXY/v1/messages ... >/dev/null
# Expect:
#   INFO fallback triggered model="..." failed_provider="cp" status=429
#   INFO request completed ... failed_providers="cp:429" ...
#   INFO streaming completed ... input_tokens=N ...

# 3. Cache hit (after a duplicate system prompt)
curl -sS -X POST $PROXY/v1/messages ... >/dev/null
# Expect: streaming completed line carries cache_read_tokens=K (K > 0)

# 4. Mid-stream error
# Configure the primary wiremock to emit a `usage` chunk then drop the
# connection. Expect:
#   INFO request completed model="..." stream=true ...
#   INFO upstream stream error provider="..." model="..." error="..."
#   INFO streaming aborted (upstream error) ... (NO token fields)
# Crucially: NOT `streaming completed` — the partial primary usage is discarded.
```

### CI

Push branch `feat/logging-tokens-and-conditional-failure` and let the
docker build matrix run as for #22. No new test environments required.

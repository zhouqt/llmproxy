//! Axum server: routes for /v1/messages, /v1/models, /health, /v1/messages/count_tokens,
//! and /admin/copilot/auth (Copilot OAuth bootstrap trigger).

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{middleware, Json, Router as AxumRouter};
use bytes::Bytes;
use futures_util::Stream;
use serde_json::json;

use crate::anthropic::{MessagesRequest, MessagesResponse};
use crate::error::{ProxyError, Result};
use crate::extractor::AppJson;
use crate::providers::{ProviderOutput, StreamUsage, StreamUsageSink};
use crate::state::AppState;
use crate::tokenize::estimate_request_tokens;

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
    let model_cfg = state
        .router
        .find_model(&req.model)
        .ok_or_else(|| ProxyError::BadRequest(format!("unknown model: {}", req.model)))?
        .clone();

    let start = std::time::Instant::now();

    if req.stream {
        // Build a single usage-watch Arc shared between the SSE adapter
        // (writer — fills via terminal-path sink writes) and the
        // MappedStream callback (reader — drains on Ready(None) to
        // emit token counts in the `streaming completed` log line).
        let usage_watch: Arc<Mutex<Option<StreamUsage>>> = Arc::new(Mutex::new(None));
        let sink = StreamUsageSink::from_arc(usage_watch.clone());

        let (provider, output, attempts) = state
            .router
            .stream(&model_cfg, &req, Some(sink))
            .await?;

        // Aggregated fallback summary (when a fallback actually
        // happened): the router already logged per-attempt
        // `fallback triggered` lines; this restores the single
        // `failed_providers="primary:429,..."` string the non-streaming
        // path emits. Token counts land on the later `streaming
        // completed` line from the MappedStream callback.
        if !attempts.is_empty() {
            log_streaming_fallback(
                &req.model,
                provider.name(),
                start.elapsed(),
                &attempts,
            );
        }

        // We log the streaming-completed/aborted lines from inside the
        // MappedStream callback (which fires synchronously in the same
        // poll that returns Ready(None) — see `MappedStream::poll_next`).
        // Operators correlate with the prior `request completed` /
        // `streaming fallback` line on the same request by matching
        // `model` + `provider`.
        return Ok(stream_response(
            provider.name(),
            req.model.as_str(),
            output,
            attempts,
            usage_watch,
            provider.name().to_string(),
            req.model.clone(),
            start,
        ));
    }

    let (output, attempts) = state.router.complete(&model_cfg, &req).await?;
    let ProviderOutput::Json(value) = output else {
        return Err(ProxyError::Internal(
            "non-streaming provider returned a stream".into(),
        ));
    };

    let mut resp: MessagesResponse = serde_json::from_value(value)?;
    resp.model = req.model.clone();

    // For non-streaming, infer the provider that served the request: when
    // no attempts failed, it was the primary; otherwise it's the last
    // chain entry that didn't appear in `attempts`.
    let provider_label = if attempts.is_empty() {
        model_cfg.primary.clone()
    } else {
        model_cfg
            .chain()
            .filter(|n| !attempts.iter().any(|a| a.provider == *n))
            .last()
            .unwrap_or("unknown")
            .to_string()
    };
    let tokens = extract_complete_tokens(&resp);
    log_request_completed(
        &req.model,
        &provider_label,
        false,
        start.elapsed(),
        format_attempts_optional(&attempts).as_deref(),
        tokens.as_ref(),
    );

    let mut headers = HeaderMap::new();
    if !attempts.is_empty() {
        if let Ok(v) = format_attempts(&attempts).parse() {
            headers.insert("x-llmproxy-failed-providers", v);
        }
    }

    Ok((StatusCode::OK, headers, Json(resp)).into_response())
}

fn format_attempts(attempts: &[crate::router::RouteAttempt]) -> String {
    attempts
        .iter()
        .map(|a| format!("{}:{}", a.provider, a.status))
        .collect::<Vec<_>>()
        .join(",")
}

/// Same shape as `format_attempts`, but emits `None` when there were
/// no fallback attempts. The single-macro log helper passes this
/// straight into the `tracing::info!` call, where `None` records
/// nothing (`impl Value for Option<T>`) so the `failed_providers`
/// field is absent on the healthy no-fallback path.
fn format_attempts_optional(attempts: &[crate::router::RouteAttempt]) -> Option<String> {
    if attempts.is_empty() {
        None
    } else {
        Some(format_attempts(attempts))
    }
}

/// Public summary formatter for router attempts. Exposed for `Router` so
/// fallback / "all providers failed" logs can render the same shape that
/// the response header (`x-llmproxy-failed-providers`) emits. Keep the
/// two formats in sync — operators correlate header + log entries by
/// this string.
pub fn format_attempts_summary(attempts: &[crate::router::RouteAttempt]) -> String {
    format_attempts(attempts)
}

/// Token counts captured from upstream usage, ready to emit on the
/// `request completed` / `streaming completed` log lines. `cache_read`
/// is `Option<u32>` so the `Empty`-skipping convention distinguishes
/// "upstream didn't report" from "upstream reported 0".
pub struct LogTokens {
    pub input: u32,
    pub output: u32,
    pub cache_read: Option<u32>,
}

/// Extract token counts from a non-streaming Anthropic `MessagesResponse`.
/// `resp.usage` is non-optional on the wire (Anthropic spec mandates it);
/// we read directly. Conversion paths set `input = prompt - cached` so
/// `LogTokens::input + cache_read = prompt_total` is the invariant.
fn extract_complete_tokens(resp: &MessagesResponse) -> Option<LogTokens> {
    Some(LogTokens {
        input: resp.usage.input_tokens,
        output: resp.usage.output_tokens,
        cache_read: resp.usage.cache_read_input_tokens,
    })
}

/// Emit the canonical `request completed` log line. Conditional fields
/// use `Option`-absence (see the comment inside the function).
/// `total_tokens = input + output` follows Anthropic's
/// convention (Anthropic's `usage` has no top-level `total_tokens`;
/// clients sum input + output). Operators who want the OpenAI-equivalent
/// total (prompt + completion, includes cache) add `cache_read_tokens`.
fn log_request_completed(
    model: &str,
    provider: &str,
    stream: bool,
    elapsed: std::time::Duration,
    failed_providers: Option<&str>,
    tokens: Option<&LogTokens>,
) {
    // Conditional fields ride on `tracing`'s native
    // `impl Value for Option<T>`: a `None` records nothing, so the
    // field is simply absent from the event — no custom formatter and
    // no `tracing::field::Empty` sentinel needed. `failed_providers`
    // goes through `field::display` to keep the unquoted
    // `failed_providers=cp:429` wire shape; `model`/`provider` stay
    // quoted via `record_str`. `cache_read_tokens` as `Option<u32>`
    // distinguishes "no cache hit" (absent) from "upstream reported
    // 0" (=0). Operators correlate lines by matching
    // `model` + `provider`; rendering them as event fields is the wire
    // shape PR #22 produced.
    tracing::info!(
        model = model,
        provider = provider,
        stream = stream,
        elapsed_ms = elapsed.as_millis() as u64,
        failed_providers = failed_providers.map(tracing::field::display),
        input_tokens = tokens.as_ref().map(|t| t.input),
        output_tokens = tokens.as_ref().map(|t| t.output),
        total_tokens = tokens.as_ref().map(|t| (t.input as u64) + (t.output as u64)),
        cache_read_tokens = tokens.as_ref().and_then(|t| t.cache_read),
        "request completed"
    );
}

/// Outcome of a streaming body drain — passed to the
/// `MappedStream::with_callback` closure once per stream. We collapse the
/// success and errored branches into a single enum so the caller can
/// pick the right log line without holding two booleans (Sonnet M3).
#[derive(Clone, Debug)]
pub enum MappedCompletion {
    /// Stream ended normally. `Option<StreamUsage>` is `None` when the
    /// upstream never reported usage (e.g. provider doesn't emit a
    /// `usage` chunk); `Some` when it did. Distinguishing these via
    /// `Option` prevents operators from reading "0 tokens" when the
    /// actual situation is "no usage available" (Sonnet M4).
    Success(Option<StreamUsage>),
    /// Upstream errored mid-body (transport-level `Err`, or an upstream
    /// error-envelope that the SSE adapter encoded as an Anthropic
    /// `event: error` chunk). The synthetic `event: error` SSE chunk has
    /// been emitted to the client. Partial primary usage is discarded
    /// (Opus C1) — we cannot prove the upstream charged the request, the
    /// client received no usable response, and there is no fallback
    /// (streaming contract — once bytes flow, the chain stops). Logging
    /// primary's partial tokens would silently misattribute cost.
    Errored,
    /// The body stream was dropped before reaching EOF — client
    /// disconnect (Ctrl-C / tool abort / connection reset). No terminal
    /// chunk was emitted. Fired from `Drop` so aborted-by-client streams
    /// still leave a trace in the logs.
    Aborted,
}

fn stream_response(
    provider_name: &str,
    model: &str,
    output: ProviderOutput,
    attempts: Vec<crate::router::RouteAttempt>,
    usage_watch: Arc<Mutex<Option<StreamUsage>>>,
    provider_for_log: String,
    model_for_log: String,
    start: std::time::Instant,
) -> Response {
    let ProviderOutput::Stream(stream) = output else {
        // A stream:true request that yields a non-stream output is a
        // provider contract violation and returns 500. Log it — without
        // this the request is completely invisible (no start line, no
        // MappedStream callback) (code-review F1).
        tracing::error!(
            provider = provider_for_log,
            model = model_for_log,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "streaming request returned non-stream output"
        );
        return ProxyError::Internal("expected stream output".into()).into_response();
    };

    let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
        Box::into_pin(stream);

    // The callback fires synchronously in the same poll that returns
    // Ready(None) (success-EOF) or Ready(Some(Err)) (errored mid-body).
    // Operators grep for `streaming completed` / `streaming aborted
    // (upstream error)` to find request-end on the streaming path —
    // the previous design fired the callback on the *next* poll's
    // short-circuit, but `axum::body::Body::from_stream` polls exactly
    // once on success-EOF, so the log line never appeared for the
    // dominant case. See Sonnet C1.
    let on_complete = move |completion: MappedCompletion| {
        let elapsed = start.elapsed();
        match completion {
            MappedCompletion::Success(usage) => {
                let tokens = usage.map(|u| LogTokens {
                    input: u.input,
                    output: u.output,
                    cache_read: u.cache_read,
                });
                log_streaming_completed(
                    &model_for_log,
                    &provider_for_log,
                    elapsed,
                    tokens,
                );
            }
            MappedCompletion::Errored => {
                // Partial-usage-on-error is discarded. We emit the
                // aborted line WITHOUT token fields (Opus C1).
                log_streaming_aborted(&model_for_log, &provider_for_log, elapsed);
            }
            MappedCompletion::Aborted => {
                // Client disconnected before EOF — no error chunk was
                // emitted, no tokens were logged. Emit a distinct line
                // so aborted-by-client streams leave a trace.
                tracing::info!(
                    model = &model_for_log,
                    provider = &provider_for_log,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "streaming aborted (client disconnect)"
                );
            }
        }
    };
    let mapped = MappedStream::with_callback(
        provider_name,
        model,
        inner,
        on_complete,
        usage_watch,
    );
    let body = Body::from_stream(mapped);

    let mut resp = Response::new(body);
    let h = resp.headers_mut();
    h.insert(
        "content-type",
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    h.insert("cache-control", "no-cache".parse().unwrap());
    h.insert("x-accel-buffering", "no".parse().unwrap());
    if !attempts.is_empty() {
        if let Ok(v) = format_attempts(&attempts).parse() {
            h.insert("x-llmproxy-failed-providers", v);
        }
    }
    resp
}

/// Emit the streaming success log line. Mirrors the non-streaming
/// `request completed` field shape so operators can grep a consistent
/// vocabulary across both paths. Notably absent: `failed_providers` —
/// streaming has no fallback (once bytes flow, the chain stops). For
/// streaming the fallback signal lives on the earlier non-streaming
/// `request completed` line emitted by `Router::stream`'s `info!`
/// sites (or in this case by `Router`'s own `tracing::info!` callers
/// before we return).
fn log_streaming_completed(
    model: &str,
    provider: &str,
    elapsed: std::time::Duration,
    tokens: Option<LogTokens>,
) {
    // Same `Option<T> = absent` convention as `log_request_completed`
    // (see its doc comment). Token fields appear only when the SSE
    // adapter reported usage.
    tracing::info!(
        model = model,
        provider = provider,
        elapsed_ms = elapsed.as_millis() as u64,
        input_tokens = tokens.as_ref().map(|t| t.input),
        output_tokens = tokens.as_ref().map(|t| t.output),
        total_tokens = tokens.as_ref().map(|t| (t.input as u64) + (t.output as u64)),
        cache_read_tokens = tokens.as_ref().and_then(|t| t.cache_read),
        "streaming completed"
    );
}

/// Emit the streaming error log line. NO token fields — partial
/// primary usage is discarded (Opus C1).
fn log_streaming_aborted(model: &str, provider: &str, elapsed: std::time::Duration) {
    tracing::info!(
        model = model,
        provider = provider,
        elapsed_ms = elapsed.as_millis() as u64,
        "streaming aborted (upstream error)"
    );
}

/// Emit the aggregated fallback summary on the streaming path. The
/// router already logs a per-attempt `fallback triggered` line; this is
/// the streaming equivalent of the non-streaming `request completed`
/// line's `failed_providers=` field — the one aggregated string
/// operators grep for. Deliberately NOT named `request completed`
/// (that would contradict a later `streaming aborted` line on the same
/// request) and carries no token fields (tokens land on the later
/// `streaming completed` line from the MappedStream callback).
fn log_streaming_fallback(
    model: &str,
    provider: &str,
    elapsed: std::time::Duration,
    attempts: &[crate::router::RouteAttempt],
) {
    let failed_providers = format_attempts(&attempts);
    tracing::info!(
        model = model,
        provider = provider,
        stream = true,
        elapsed_ms = elapsed.as_millis() as u64,
        failed_providers = %failed_providers,
        "streaming fallback"
    );
}

/// Adapter: wraps a `Result<Bytes, ProxyError>` stream as a
/// `Result<Bytes, std::io::Error>` stream for axum's body. Emits an
/// Anthropic `event: error` SSE chunk before terminating so clients
/// don't see an incomplete body with no signal that something went
/// wrong.
pub struct MappedStream {
    /// Provider name, carried into the upstream-error log so operators
    /// can see which provider's stream failed in a multi-provider
    /// deployment.
    provider: String,
    /// Client-requested model name, same purpose as `provider`.
    model: String,
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    /// Collapsed two-flag state machine (Sonnet M3) — both success-EOF
    /// and errored-mid-body land in `Done(_)` so the callback-firing
    /// logic doesn't drift between paths.
    phase: MappedPhase,
    /// Shared `Arc<Mutex<Option<StreamUsage>>>` clone of the same cell
    /// the SSE adapter writes to on its terminal paths. We drain this
    /// at success-EOF and hand the value to the callback via
    /// `MappedCompletion::Success(Some(usage))`. Held directly on
    /// `MappedStream` (not inside `SingleFireCallback`) so the
    /// callback closure's environment stays minimal (Sonnet m1).
    usage_watch: Arc<Mutex<Option<StreamUsage>>>,
    /// Single-fire callback wrapped in a `Mutex<Option<Box<dyn FnOnce>>>`.
    /// The callback fires synchronously in the same poll that returns
    /// `Ready(None)` (success-EOF) or `Ready(Some(Err(_)))` (errored
    /// mid-body) — see `poll_next` for the firing sequence (Sonnet C1).
    on_complete: Option<Mutex<Option<Box<dyn FnOnce(MappedCompletion) + Send>>>>,
}

/// Two-phase lifecycle for [`MappedStream`]. Collapsing `done: bool` +
/// `errored: bool` into a single phase enum eliminates the bug class
/// where the two flags drift out of sync (Sonnet M3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappedPhase {
    Streaming,
    /// Terminal — terminal callback has been (or is being) fired.
    Done,
}

impl MappedStream {
    /// Test-only constructor: builds a stream with no callback and no
    /// usage-watch, so it never emits a log line. Production code must
    /// use `with_callback` — this variant silently skips logging, which
    /// would make a request invisible in the logs (code-review F7).
    pub fn new(
        provider: &str,
        model: &str,
        inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            inner,
            phase: MappedPhase::Streaming,
            usage_watch: Arc::new(Mutex::new(None)),
            on_complete: None,
        }
    }

    /// Constructor that wires up the per-stream logging callback.
    /// `on_complete` fires synchronously in the same poll that returns
    /// `Ready(None)` (success-EOF) or `Ready(Some(Err(_)))` (errored
    /// mid-body) — see `poll_next`. `usage_watch` is the same
    /// `Arc<Mutex<Option<StreamUsage>>>` clone the SSE adapter writes
    /// to; the callback receives the drained value via
    /// `MappedCompletion::Success(Some(usage))`.
    pub fn with_callback(
        provider: &str,
        model: &str,
        inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
        on_complete: impl FnOnce(MappedCompletion) + Send + 'static,
        usage_watch: Arc<Mutex<Option<StreamUsage>>>,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            inner,
            phase: MappedPhase::Streaming,
            usage_watch,
            on_complete: Some(Mutex::new(Some(Box::new(on_complete)))),
        }
    }

    /// Drain captured usage (if any) from the shared watch cell. Called
    /// from `poll_next`'s success-EOF arm. `None` when the SSE adapter
    /// never wrote (upstream didn't report usage).
    fn take_captured_usage(&self) -> Option<StreamUsage> {
        self.usage_watch.lock().ok().and_then(|mut g| g.take())
    }

    /// Fire the single-fire callback once. Wrapped in `catch_unwind` so
    /// a panic in user-supplied log code cannot tear down the body-sink
    /// task (which would otherwise surface as a truncated body with no
    /// signal). After firing, the callback is consumed — re-polls (e.g.
    /// defensive double-polls from `StreamBody`) are silent no-ops.
    fn fire_callback(&mut self, completion: MappedCompletion) {
        let cb = match self.on_complete.as_mut() {
            Some(slot) => slot.lock().ok().and_then(|mut g| g.take()),
            None => return,
        };
        let Some(cb) = cb else {
            return;
        };
        let cb_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cb(completion);
        }));
        if let Err(panic_payload) = cb_result {
            tracing::error!(
                provider = %self.provider,
                model = %self.model,
                panic = ?panic_payload,
                "streaming_completed callback panicked; swallowed to preserve body-sink task",
            );
        }
    }
}

impl Stream for MappedStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Phase terminal: short-circuit forever. The callback has
        // already fired (or was never set). This is the path that
        // Sonnet C1's "fire on the next poll's short-circuit" would
        // have hit but no longer exists — we fire in the SAME poll
        // that returns Ready(None), then transition to Done.
        if self.phase == MappedPhase::Done {
            return Poll::Ready(None);
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                tracing::error!(
                    provider = %self.provider,
                    model = %self.model,
                    error = %e,
                    "upstream stream error"
                );
                // Discard partial primary usage (Opus C1 / Sonnet C1 —
                // we cannot prove the upstream charged the request,
                // the client received no usable response, and there
                // is no fallback on the streaming path).
                let _ = self.take_captured_usage();
                let chunk = format_stream_error(&e);
                // Fire the callback synchronously BEFORE returning the
                // synthetic chunk. This is critical: if we returned
                // the chunk first and then transitioned to Done,
                // `axum::body::Body::from_stream`'s `StreamBody` would
                // call `poll_next` once more, observe Done, and never
                // re-emit. Firing here, in the same poll, guarantees
                // the `streaming aborted (upstream error)` line lands
                // even if this is the final poll.
                self.fire_callback(MappedCompletion::Errored);
                self.phase = MappedPhase::Done;
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(None) => {
                // Success-EOF. Drain the shared watch cell — the SSE
                // adapter has already written its terminal usage into
                // this Arc by the time we observe Ready(None) (Sonnet
                // M5 — ordering is deterministic; the adapter writes
                // on [DONE] / error-envelope / Ready(None) before
                // returning). If the adapter flagged `errored` (an
                // upstream error-envelope encoded as an Anthropic
                // `event: error` chunk), classify as aborted instead of
                // a clean completion.
                let usage = self.take_captured_usage();
                let errored = usage.as_ref().map(|u| u.errored).unwrap_or(false);
                if errored {
                    self.fire_callback(MappedCompletion::Errored);
                } else {
                    self.fire_callback(MappedCompletion::Success(usage));
                }
                self.phase = MappedPhase::Done;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MappedStream {
    fn drop(&mut self) {
        // Client disconnected (or the body was dropped) before the
        // stream reached EOF. Normal completion sets `phase = Done`
        // inside `poll_next` before the body is dropped, so this guard
        // prevents double-firing. `fire_callback` is panic-safe
        // (catch_unwind) and the callback is sync `tracing::info!`.
        if self.phase != MappedPhase::Done {
            let _ = self.take_captured_usage();
            self.fire_callback(MappedCompletion::Aborted);
        }
    }
}

/// Encode a [`ProxyError`] as an Anthropic SSE `event: error` chunk.
fn format_stream_error(err: &ProxyError) -> Bytes {
    let payload = serde_json::json!({
        "type": "error",
        "error": {
            "type": "upstream_error",
            "message": err.to_string(),
        }
    });
    Bytes::from(format!("event: error\ndata: {payload}\n\n"))
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
        MappedStream::new("test", "test", make_stream(vec![]))
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
        // Once phase is done, poll_next must short-circuit to
        // Ready(None) without touching the inner stream at all.
        let mut s = MappedStream::new(
            "test",
            "test",
            make_stream(vec![Err(ProxyError::Internal("unused".into()))]),
        );
        // Drive the inner-error path first: synthetic error chunk is
        // emitted, phase flips to done. Then a second poll must
        // short-circuit to Ready(None).
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let p1 = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(p1, std::task::Poll::Ready(Some(Ok(_)))));
        assert!(s.phase == MappedPhase::Done);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_none(poll);
    }

    #[test]
    fn mapped_stream_propagates_pending_from_inner() {
        // When the inner stream returns Poll::Pending, the wrapper must
        // also return Poll::Pending (and must NOT mark itself done).
        let mut s = MappedStream::new(
            "test",
            "test",
            Box::pin(stream::pending::<std::result::Result<Bytes, ProxyError>>()),
        );
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_pending(poll);
        assert!(
            s.phase == crate::server::MappedPhase::Streaming,
            "Pending must not flip phase to Done"
        );
    }

    #[tokio::test]
    async fn mapped_stream_emits_error_event_then_terminates_on_inner_error() {
        // An upstream error must NOT just truncate the body — the client
        // would see 200 OK and no message_stop, with no signal that
        // anything went wrong. We inject an Anthropic `event: error`
        // chunk so the SDK can distinguish aborted streams from normal
        // end-of-stream.
        let mut s = MappedStream::new(
            "test",
            "test",
            make_stream(vec![Err(ProxyError::Internal("boom".into()))]),
        );
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
        assert!(s.phase == crate::server::MappedPhase::Done);
    }

    #[tokio::test]
    async fn mapped_stream_emits_bytes_then_terminates() {
        let mut s = MappedStream::new(
            "test",
            "test",
            make_stream(vec![
                Ok(Bytes::from_static(b"event: foo\n\n")),
                Ok(Bytes::from_static(b"event: bar\n\n")),
            ]),
        );
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
        assert!(s.phase == crate::server::MappedPhase::Done);
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
        assert!(s.phase == crate::server::MappedPhase::Streaming);
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

    // ──────────────────────────────────────────────────────────────────
    // log_request_completed / log_streaming_completed /
    // log_streaming_aborted / extract_complete_tokens / MappedCompletion
    // / MappedPhase — covering the new helpers added for
    // conditional failed_providers + token counts.
    //
    // We exercise the helpers under a tracing subscriber that captures
    // output to an in-memory buffer (standard `compact()` formatter —
    // the same shape production installs). Each scenario asserts the
    // captured log line carries exactly the expected field set.

    // ──────────────────────────────────────────────────────────────────

    use crate::anthropic::Usage;
    use crate::providers::StreamUsage;
    use crate::test_support::CaptureWriter;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;

    fn run_with_capture<F: FnOnce()>(body: F) -> String {
        let writer = CaptureWriter::default();
        let layer = tracing_subscriber::fmt::Layer::default()
            .with_writer(writer.clone())
            .with_target(false)
            .with_level(true)
            .with_ansi(false)
            .compact();
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, body);
        let buf = writer.0.lock().unwrap().clone();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn log_request_completed_no_fallback_no_tokens() {
        // Healthy path: no fallback, no usage reported.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "primary",
                false,
                std::time::Duration::from_millis(120),
                None,
                None,
            );
        });
        assert!(out.contains("request completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(out.contains("stream=false"), "got: {out}");
        assert!(!out.contains("failed_providers"), "must omit failed_providers");
        assert!(!out.contains("input_tokens"), "must omit input_tokens");
        assert!(!out.contains("output_tokens"), "must omit output_tokens");
        assert!(!out.contains("total_tokens"), "must omit total_tokens");
        assert!(!out.contains("cache_read_tokens"), "must omit cache_read_tokens");
    }

    #[test]
    fn log_request_completed_fallback_with_tokens() {
        // Fallback + usage: all fields present.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "backup",
                false,
                std::time::Duration::from_millis(120),
                Some("cp:429"),
                Some(&LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: Some(4),
                }),
            );
        });
        assert!(out.contains("request completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"backup\""), "got: {out}");
        assert!(out.contains("stream=false"), "got: {out}");
        assert!(out.contains("failed_providers=cp:429"), "got: {out}");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(out.contains("cache_read_tokens=4"), "got: {out}");
    }

    #[test]
    fn log_request_completed_no_fallback_with_tokens_no_cache() {
        // Healthy path WITH usage: tokens present, cache_read absent.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "primary",
                false,
                std::time::Duration::from_millis(120),
                None,
                Some(&LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: None,
                }),
            );
        });
        assert!(!out.contains("failed_providers"), "must omit failed_providers");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(out.contains("stream=false"), "got: {out}");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(!out.contains("cache_read_tokens"), "must omit cache_read_tokens");
    }

    #[test]
    fn log_request_completed_fallback_no_tokens() {
        // Fallback without usage: failed_providers present, tokens absent.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "backup",
                false,
                std::time::Duration::from_millis(120),
                Some("cp:429"),
                None,
            );
        });
        assert!(out.contains("request completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"backup\""), "got: {out}");
        assert!(out.contains("stream=false"), "got: {out}");
        assert!(out.contains("failed_providers=cp:429"), "got: {out}");
        assert!(!out.contains("input_tokens"), "must omit input_tokens");
        assert!(!out.contains("output_tokens"), "must omit output_tokens");
        assert!(!out.contains("total_tokens"), "must omit total_tokens");
    }

    #[test]
    fn log_streaming_completed_with_tokens() {
        let out = run_with_capture(|| {
            log_streaming_completed(
                "work-mini",
                "primary",
                std::time::Duration::from_millis(50),
                Some(LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: Some(4),
                }),
            );
        });
        assert!(out.contains("streaming completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(!out.contains("failed_providers"), "streaming must not carry failed_providers");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(out.contains("cache_read_tokens=4"), "got: {out}");
    }

    #[test]
    fn log_request_completed_fallback_with_tokens_no_cache() {
        // Fallback + usage WITHOUT cache_read: failed_providers and
        // token fields present, cache_read_tokens absent.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "backup",
                false,
                std::time::Duration::from_millis(120),
                Some("cp:429"),
                Some(&LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: None,
                }),
            );
        });
        assert!(out.contains("request completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"backup\""), "got: {out}");
        assert!(out.contains("failed_providers=cp:429"), "got: {out}");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(!out.contains("cache_read_tokens"), "must omit cache_read_tokens");
    }

    #[test]
    fn log_request_completed_no_fallback_with_tokens_with_cache() {
        // Healthy path with usage + cache hit: no failed_providers, all
        // token fields (incl. cache_read_tokens) present.
        let out = run_with_capture(|| {
            log_request_completed(
                "work-mini",
                "primary",
                false,
                std::time::Duration::from_millis(120),
                None,
                Some(&LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: Some(4),
                }),
            );
        });
        assert!(out.contains("request completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(!out.contains("failed_providers"), "must omit failed_providers");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(out.contains("cache_read_tokens=4"), "got: {out}");
    }

    #[test]
    fn log_streaming_completed_with_tokens_no_cache() {
        // Streaming with usage but no cache: token fields present,
        // cache_read_tokens absent.
        let out = run_with_capture(|| {
            log_streaming_completed(
                "work-mini",
                "primary",
                std::time::Duration::from_millis(50),
                Some(LogTokens {
                    input: 6,
                    output: 4,
                    cache_read: None,
                }),
            );
        });
        assert!(out.contains("streaming completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(!out.contains("failed_providers"), "streaming must not carry failed_providers");
        assert!(out.contains("input_tokens=6"), "got: {out}");
        assert!(out.contains("output_tokens=4"), "got: {out}");
        assert!(out.contains("total_tokens=10"), "got: {out}");
        assert!(!out.contains("cache_read_tokens"), "must omit cache_read_tokens");
    }

    #[test]
    fn log_streaming_fallback_emits_failed_providers_only_when_present() {
        let out = run_with_capture(|| {
            log_streaming_fallback(
                "work-mini",
                "primary",
                std::time::Duration::from_millis(50),
                &[crate::router::RouteAttempt {
                    provider: "primary".into(),
                    status: 429,
                    body: String::new(),
                }],
            );
        });
        assert!(out.contains("streaming fallback"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(out.contains("stream=true"), "got: {out}");
        assert!(out.contains("failed_providers=primary:429"), "got: {out}");
    }

    #[test]
    fn log_streaming_completed_no_tokens() {
        let out = run_with_capture(|| {
            log_streaming_completed(
                "work-mini",
                "primary",
                std::time::Duration::from_millis(50),
                None,
            );
        });
        assert!(out.contains("streaming completed"), "got: {out}");
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(!out.contains("input_tokens"), "must omit input_tokens");
        assert!(!out.contains("output_tokens"), "must omit output_tokens");
    }

    #[test]
    fn log_streaming_aborted_emits_no_token_fields() {
        let out = run_with_capture(|| {
            log_streaming_aborted(
                "work-mini",
                "primary",
                std::time::Duration::from_millis(50),
            );
        });
        assert!(
            out.contains("streaming aborted (upstream error)"),
            "got: {out}"
        );
        assert!(out.contains("model=\"work-mini\""), "got: {out}");
        assert!(out.contains("provider=\"primary\""), "got: {out}");
        assert!(!out.contains("input_tokens"), "must NOT carry input_tokens");
        assert!(!out.contains("output_tokens"), "must NOT carry output_tokens");
        assert!(!out.contains("total_tokens"), "must NOT carry total_tokens");
        assert!(!out.contains("cache_read_tokens"), "must NOT carry cache_read_tokens");
    }

    #[test]
    fn extract_complete_tokens_conversion_path_post_cache_subtraction() {
        // OpenAI Chat shape: prompt=10, cached=4 → input_tokens=6,
        // cache_read=Some(4). This is the invariant
        // `LogTokens::input + cache_read = prompt_total`.
        let usage = Usage {
            input_tokens: 6,
            output_tokens: 4,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(4),
            cache_creation: None,
            server_tool_use: None,
            output_tokens_details: None,
            service_tier: None,
            inference_geo: None,
        };
        let resp = MessagesResponse {
            id: "msg_test".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "m".into(),
            content: vec![],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            stop_details: None,
            container: None,
            usage,
            extra: Default::default(),
        };
        let tokens = extract_complete_tokens(&resp).expect("non-empty usage");
        assert_eq!(tokens.input, 6);
        assert_eq!(tokens.output, 4);
        assert_eq!(tokens.cache_read, Some(4));
        // Invariant: input + cache_read = prompt_total (10).
        assert_eq!(tokens.input + tokens.cache_read.unwrap(), 10);
    }

    #[test]
    fn extract_complete_tokens_passthrough_path_preserves_cache_creation() {
        // Anthropic-native shape: cache_creation_input_tokens is
        // preserved on the wire and MUST flow through to
        // MessagesResponse.usage (Opus C3 / Sonnet m3). The log
        // surface in this PR does NOT emit cache_creation — that's a
        // follow-up — but the wire shape must not be corrupted.
        let usage = Usage {
            input_tokens: 8,
            output_tokens: 4,
            cache_creation_input_tokens: Some(2),
            cache_read_input_tokens: Some(8),
            cache_creation: None,
            server_tool_use: None,
            output_tokens_details: None,
            service_tier: None,
            inference_geo: None,
        };
        let resp = MessagesResponse {
            id: "msg_test".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "m".into(),
            content: vec![],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            stop_details: None,
            container: None,
            usage,
            extra: Default::default(),
        };
        // Field still flows through to MessagesResponse.usage — wire
        // shape is not corrupted.
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(2));
        let tokens = extract_complete_tokens(&resp).expect("non-empty usage");
        assert_eq!(tokens.cache_read, Some(8));
        assert_eq!(tokens.input, 8);
    }

    #[test]
    fn mapped_completion_success_carries_optional_usage() {
        // MappedCompletion::Success(Option<StreamUsage>) — None branch
        // is operator-readable as "no usage available", not "0 tokens".
        let c = MappedCompletion::Success(None);
        match c {
            MappedCompletion::Success(None) => {}
            _ => panic!("expected Success(None)"),
        }
        let usage = StreamUsage {
            input: 1,
            output: 2,
            cache_read: Some(3),
            ..Default::default()
        };
        let c2 = MappedCompletion::Success(Some(usage.clone()));
        match c2 {
            MappedCompletion::Success(Some(u)) => {
                assert_eq!(u.input, 1);
                assert_eq!(u.output, 2);
                assert_eq!(u.cache_read, Some(3));
            }
            _ => panic!("expected Success(Some(_))"),
        }
    }

    #[tokio::test]
    async fn mapped_completion_errored_discards_partial_usage_via_callback() {
        // Construct an SSE adapter (OpenAiCompat-style) that emits a
        // usage chunk, then errors mid-body. The MappedStream wrapper
        // must fire MappedCompletion::Errored (NOT Success), and the
        // sink must be empty after the call (partial usage discarded
        // per Opus C1).
        use crate::providers::StreamUsageSink;
        use futures_util::stream;

        let chunks: Vec<std::result::Result<Bytes, ProxyError>> = vec![
            // A normal SSE chunk first (will be processed by the
            // translator if it recognized the shape, but here we
            // short-circuit: any valid bytes are fine — the test
            // focuses on the inner-error path).
            Ok(Bytes::from_static(b"event: ping\ndata: {}\n\n")),
            Err(ProxyError::Internal("boom".into())),
        ];
        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::iter(chunks));

        let sink = StreamUsageSink::empty();
        let usage_watch = sink.arc();

        let captured: Arc<Mutex<Option<MappedCompletion>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |c| {
                *captured_clone.lock().unwrap() = Some(c);
            },
            usage_watch.clone(),
        );

        // Drain to EOF (will hit the inner Err mid-body).
        use futures_util::StreamExt;
        let _ = mapped.next().await; // Some(Ok(ping))
        let _ = mapped.next().await; // Some(Ok(synthetic event: error))
        let none = mapped.next().await; // Ready(None) — phase is Done
        assert!(none.is_none());

        // Callback fired with Errored.
        let c = captured.lock().unwrap().clone();
        match c {
            Some(MappedCompletion::Errored) => {}
            other => panic!("expected Errored, got {other:?}"),
        }
        // Partial primary usage was discarded.
        let sink_after = usage_watch.lock().unwrap().clone();
        assert!(
            sink_after.is_none(),
            "partial usage must be discarded on error, got {sink_after:?}"
        );
    }

    #[tokio::test]
    async fn mapped_completion_success_fires_callback_with_drained_usage() {
        // Pre-populate the shared watch cell with usage. Construct a
        // MappedStream with `inner = empty`. The Ready(None) arm must
        // fire MappedCompletion::Success(Some(usage)) (Sonnet M4).
        use crate::providers::{StreamUsage, StreamUsageSink};
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::empty());

        let sink = StreamUsageSink::empty();
        let usage_watch = sink.arc();
        // Simulate the SSE adapter writing terminal usage before EOF.
        usage_watch.lock().unwrap().replace(StreamUsage {
            input: 7,
            output: 3,
            cache_read: Some(2),
            ..Default::default()
        });

        let captured: Arc<Mutex<Option<MappedCompletion>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |c| {
                *captured_clone.lock().unwrap() = Some(c);
            },
            usage_watch.clone(),
        );
        use futures_util::StreamExt;
        let none = mapped.next().await;
        assert!(none.is_none());

        let c = captured.lock().unwrap().clone();
        match c {
            Some(MappedCompletion::Success(Some(u))) => {
                assert_eq!(u.input, 7);
                assert_eq!(u.output, 3);
                assert_eq!(u.cache_read, Some(2));
            }
            other => panic!("expected Success(Some(_)), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mapped_completion_callback_does_not_fire_twice_on_double_poll() {
        // After the callback fires once on success-EOF, a second poll
        // on phase=Done is a no-op (Sonnet M3 idempotency).
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::empty());

        let counter: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();
        let usage_watch = crate::providers::StreamUsageSink::empty().arc();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |_c| {
                *counter_clone.lock().unwrap() += 1;
            },
            usage_watch,
        );
        use futures_util::StreamExt;
        let _ = mapped.next().await; // fires once
        let _ = mapped.next().await; // short-circuits
        let _ = mapped.next().await; // short-circuits
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn mapped_completion_drop_fires_aborted_callback() {
        // Regression (code-review F3): when the body stream is dropped
        // before EOF (client disconnect), MappedStream::drop must fire
        // MappedCompletion::Aborted so the request leaves a log trace.
        // Normal completion sets phase=Done in poll_next first, so this
        // must not fire for a drained stream.
        use crate::providers::StreamUsageSink;
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::pending());

        let captured: Arc<Mutex<Option<MappedCompletion>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let usage_watch = StreamUsageSink::empty().arc();
        let mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |c| {
                *captured_clone.lock().unwrap() = Some(c);
            },
            usage_watch,
        );

        // Drop without polling to terminal — Drop must fire Aborted.
        drop(mapped);

        let c = captured.lock().unwrap().clone();
        match c {
            Some(MappedCompletion::Aborted) => {}
            other => panic!("expected Aborted from Drop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mapped_completion_drop_does_not_double_fire_after_drain() {
        // A stream that reached Ready(None) (phase=Done) must NOT fire
        // Aborted on drop — normal completion already fired the callback.
        use crate::providers::StreamUsageSink;
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::empty());

        let captured: Arc<Mutex<Option<MappedCompletion>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let usage_watch = StreamUsageSink::empty().arc();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |c| {
                *captured_clone.lock().unwrap() = Some(c);
            },
            usage_watch,
        );

        // Poll to Ready(None) — fires Success; then drop.
        use futures_util::StreamExt;
        let _ = mapped.next().await; // fires Success
        let c1 = captured.lock().unwrap().clone();
        assert!(
            matches!(c1, Some(MappedCompletion::Success(_))),
            "drained stream must fire Success, got {c1:?}"
        );
        drop(mapped);
        // Drop must not overwrite the already-fired completion.
        let c2 = captured.lock().unwrap().clone();
        assert!(
            matches!(c2, Some(MappedCompletion::Success(_))),
            "Drop must not double-fire after drain, got {c2:?}"
        );
    }

    #[tokio::test]
    async fn mapped_completion_errored_flag_classifies_stream_as_aborted() {
        // Regression (code-review F4): when the SSE adapter flags the
        // sink's `errored` (upstream error-envelope), MappedStream's
        // Ready(None) arm must fire MappedCompletion::Errored instead of
        // Success — the client saw an `event: error` chunk.
        use crate::providers::{StreamUsage, StreamUsageSink};
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::empty());

        let sink = StreamUsageSink::empty();
        let usage_watch = sink.arc();
        // Simulate the adapter's error-envelope sentinel write.
        usage_watch.lock().unwrap().replace(StreamUsage {
            errored: true,
            ..Default::default()
        });

        let captured: Arc<Mutex<Option<MappedCompletion>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            move |c| {
                *captured_clone.lock().unwrap() = Some(c);
            },
            usage_watch,
        );
        use futures_util::StreamExt;
        let _ = mapped.next().await;

        let c = captured.lock().unwrap().clone();
        match c {
            Some(MappedCompletion::Errored) => {}
            other => panic!("errored flag must yield Errored, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fire_callback_catches_panicking_on_complete_closure() {
        // Defense-in-depth (MappedStream::fire_callback wraps the user
        // closure in catch_unwind): a panic inside the on_complete log
        // closure must be swallowed, logged, and must NOT propagate to
        // tear down the body-sink task (which would surface as a
        // truncated body with no signal).
        use crate::providers::StreamUsageSink;
        use futures_util::stream;

        let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
            Box::pin(stream::empty());

        let usage_watch = StreamUsageSink::empty().arc();
        let mut mapped = MappedStream::with_callback(
            "primary",
            "m",
            inner,
            // Deliberately panicking closure — fire_callback must catch it.
            move |_c| {
                panic!("boom in on_complete");
            },
            usage_watch,
        );
        use futures_util::StreamExt;
        // Poll to Ready(None); the panicking closure fires inside.
        let _ = mapped.next().await;
        // If the panic propagated, the test would abort here with the
        // panic message. Reaching this assert proves it was swallowed.
        assert!(true, "panic in on_complete must be swallowed");
    }

}

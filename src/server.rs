//! Axum server: routes for /v1/messages, /v1/models, /health, /v1/messages/count_tokens,
//! and /admin/copilot/auth (Copilot OAuth bootstrap trigger).

use std::pin::Pin;
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
use crate::providers::ProviderOutput;
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
        let (provider, output, attempts) = state.router.stream(&model_cfg, &req).await?;
        let summary = format_attempts(&attempts);
        tracing::info!(
            model = req.model.as_str(),
            provider = provider.name(),
            stream = req.stream,
            elapsed_ms = start.elapsed().as_millis() as u64,
            failed_providers = %summary,
            "request completed"
        );
        return Ok(stream_response(provider.name(), req.model.as_str(), output, attempts));
    }

    let (output, attempts) = state.router.complete(&model_cfg, &req).await?;
    let ProviderOutput::Json(value) = output else {
        return Err(ProxyError::Internal(
            "non-streaming provider returned a stream".into(),
        ));
    };

    let mut resp: MessagesResponse = serde_json::from_value(value)?;
    resp.model = req.model.clone();

    let summary = format_attempts(&attempts);
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
    tracing::info!(
        model = req.model.as_str(),
        provider = %provider_label,
        stream = req.stream,
        elapsed_ms = start.elapsed().as_millis() as u64,
        failed_providers = %summary,
        "request completed"
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

/// Public summary formatter for router attempts. Exposed for `Router` so
/// fallback / "all providers failed" logs can render the same shape that
/// the response header (`x-llmproxy-failed-providers`) emits. Keep the
/// two formats in sync — operators correlate header + log entries by
/// this string.
pub fn format_attempts_summary(attempts: &[crate::router::RouteAttempt]) -> String {
    format_attempts(attempts)
}

fn stream_response(
    provider_name: &str,
    model: &str,
    output: ProviderOutput,
    attempts: Vec<crate::router::RouteAttempt>,
) -> Response {
    let ProviderOutput::Stream(stream) = output else {
        return ProxyError::Internal("expected stream output".into()).into_response();
    };

    let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
        Box::into_pin(stream);
    let mapped = MappedStream::new(provider_name, model, inner);
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
    done: bool,
}

impl MappedStream {
    pub fn new(
        provider: &str,
        model: &str,
        inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            inner,
            done: false,
        }
    }
}

impl Stream for MappedStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
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
                // Emit a synthetic Anthropic `event: error` SSE chunk so
                // the client can distinguish "stream ended normally"
                // from "stream aborted by upstream failure" — without
                // this, the body just truncates with 200 OK and no
                // message_stop, which Anthropic SDKs report as a
                // confusing parse error. Mark `done` so the next poll
                // terminates the stream instead of emitting the chunk
                // again.
                self.done = true;
                Poll::Ready(Some(Ok(format_stream_error(&e))))
            }
            Poll::Ready(None) => {
                self.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
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
        MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![]),
            done: false,
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
    async fn mapped_stream_emits_bytes_then_terminates() {
        let mut s = MappedStream {
            provider: "test".to_string(),
            model: "test".to_string(),
            inner: make_stream(vec![
                Ok(Bytes::from_static(b"event: foo\n\n")),
                Ok(Bytes::from_static(b"event: bar\n\n")),
            ]),
            done: false,
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
}

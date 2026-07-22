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

    if req.stream {
        let (_provider, output, attempts) = state.router.stream(&model_cfg, &req).await?;
        return Ok(stream_response(output, attempts));
    }

    let (output, attempts) = state.router.complete(&model_cfg, &req).await?;
    let ProviderOutput::Json(value) = output else {
        return Err(ProxyError::Internal(
            "non-streaming provider returned a stream".into(),
        ));
    };

    let mut resp: MessagesResponse = serde_json::from_value(value)?;
    resp.model = req.model.clone();

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

fn stream_response(output: ProviderOutput, attempts: Vec<crate::router::RouteAttempt>) -> Response {
    let ProviderOutput::Stream(stream) = output else {
        return ProxyError::Internal("expected stream output".into()).into_response();
    };

    let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
        Box::into_pin(stream);
    let mapped = MappedStream::new(inner);
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
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    done: bool,
}

impl MappedStream {
    pub fn new(inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>) -> Self {
        Self {
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
                tracing::error!("upstream stream error: {e}");
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

async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut models: Vec<_> = state
        .config
        .models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "llmproxy",
            })
        })
        .collect();

    // If Copilot is configured and has a cached model list, merge it in.
    // Copilot-discovered entries take precedence over static config entries
    // with the same id. Static entries for non-Copilot providers are kept.
    if let Some(cp) = &state.copilot {
        if let Some(raw) = cp.cached_models().await {
            if let Some(data) = raw.get("data").and_then(|d| d.as_array()) {
                // Build a set of Copilot model ids for dedup.
                let copilot_ids: std::collections::HashSet<&str> = data
                    .iter()
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                    .collect();

                // Remove static entries that Copilot also reports (Copilot wins).
                models.retain(|m| {
                    let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    !copilot_ids.contains(id)
                });

                // Append Copilot-discovered models in OpenAI format.
                for m in data {
                    let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let vendor = m.get("vendor").and_then(|v| v.as_str()).unwrap_or("");
                    let display_name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    models.push(serde_json::json!({
                        "id": id,
                        "object": "model",
                        "created": 0,
                        "owned_by": vendor,
                        "display_name": display_name,
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({
        "object": "list",
        "data": models,
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
    use std::pin::Pin;

    fn make_stream(
        items: Vec<std::result::Result<Bytes, ProxyError>>,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> {
        Box::pin(stream::iter(items))
    }

    fn fresh_mapped() -> MappedStream {
        MappedStream {
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
                logging: Default::default(),
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
    }
}

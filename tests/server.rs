use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use llmproxy::anthropic::MessagesRequest;
use llmproxy::config::{Config, ModelConfig, ProviderConfig, ServerConfig};
use llmproxy::cooldown::CooldownCache;
use llmproxy::error::{ProxyError, Result};
use llmproxy::providers::{Provider, ProviderOutput, SharedProvider};
use llmproxy::router::Router;
use llmproxy::state::AppState;
use llmproxy::usage::{Outcome, UsageRecord, UsageStats};

enum CompleteBehavior {
    Json,
    Stream,
    MalformedJson,
    Error(u16),
    /// First `fail_count` calls return `Upstream{status, body}`,
    /// subsequent calls return a normal success JSON. C6 regression:
    /// lets a single test exercise "primary retried N times before
    /// succeeding" without standing up a per-test provider impl.
    FailThenSucceed {
        status: u16,
        fail_count: u8,
        /// Mutable counter shared via Arc so successive `complete`
        /// invocations see each other's increments.
        counter: Arc<AtomicU32>,
    },
}

enum StreamBehavior {
    Bytes(&'static str),
    Json,
    Error(u16),
    ItemError,
}

struct TestProvider {
    name: String,
    complete: CompleteBehavior,
    stream: StreamBehavior,
    models: Option<Vec<serde_json::Value>>,
}

#[async_trait]
impl Provider for TestProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        match &self.complete {
            CompleteBehavior::Json => Ok(ProviderOutput::Json(json!({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "upstream-model",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 2, "output_tokens": 1}
            }))),
            CompleteBehavior::MalformedJson => Ok(ProviderOutput::Json(json!({
                "unexpected": true
            }))),
            CompleteBehavior::Stream => {
                Ok(ProviderOutput::Stream(Box::new(stream::empty())))
            }
            CompleteBehavior::Error(status) => Err(ProxyError::Upstream {
                status: *status,
                body: "upstream failed".to_string(),
            }),
            CompleteBehavior::FailThenSucceed {
                status,
                fail_count,
                counter,
            } => {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                if n < u32::from(*fail_count) {
                    Err(ProxyError::Upstream {
                        status: *status,
                        body: format!("planned fail #{n}"),
                    })
                } else {
                    Ok(ProviderOutput::Json(json!({
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "model": "upstream-model",
                        "stop_reason": "end_turn",
                        "stop_sequence": null,
                        "usage": {"input_tokens": 2, "output_tokens": 1}
                    })))
                }
            }
        }
    }

    async fn stream(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        match self.stream {
            StreamBehavior::Bytes(body) => Ok(ProviderOutput::Stream(Box::new(stream::iter([
                Ok(Bytes::from_static(body.as_bytes())),
            ])))),
            StreamBehavior::Json => Ok(ProviderOutput::Json(json!({"unexpected": true}))),
            StreamBehavior::Error(status) => Err(ProxyError::Upstream {
                status,
                body: "stream failed".to_string(),
            }),
            StreamBehavior::ItemError => Ok(ProviderOutput::Stream(Box::new(stream::iter([
                Err(ProxyError::Internal("stream item failed".to_string())),
            ])))),
        }
    }

    async fn list_models(&self) -> Option<Vec<serde_json::Value>> {
        self.models.clone()
    }
}

fn provider(
    name: &str,
    complete: CompleteBehavior,
    stream: StreamBehavior,
) -> SharedProvider {
    Arc::new(TestProvider {
        name: name.to_string(),
        complete,
        stream,
        models: None,
    })
}

fn build_app(
    api_key: Option<&str>,
    primary: SharedProvider,
    backup: Option<SharedProvider>,
) -> axum::Router {
    build_app_with_usage(api_key, primary, backup, UsageStats::new(0))
}

fn build_app_with_usage(
    api_key: Option<&str>,
    primary: SharedProvider,
    backup: Option<SharedProvider>,
    usage: UsageStats,
) -> axum::Router {
    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), primary);
    let mut provider_configs = vec![ProviderConfig::OpenaiCompat {
        name: "primary".to_string(),
        api_key: "unused".to_string(),
        api_base: "http://unused".to_string(),
        model_rewrite: HashMap::new(),
        use_proxy: false,
        provider_ignore: Vec::new(),
    reasoning_echo: false,
    }];
    let fallback_chain = if let Some(backup) = backup {
        providers.insert("backup".to_string(), backup);
        provider_configs.push(ProviderConfig::OpenaiCompat {
            name: "backup".to_string(),
            api_key: "unused".to_string(),
            api_base: "http://unused".to_string(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
            provider_ignore: Vec::new(),
        reasoning_echo: false,
        });
        vec!["backup".to_string()]
    } else {
        vec![]
    };
    let capacity = usage.capacity();
    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: api_key.map(str::to_string),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: provider_configs,
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: "primary".to_string(),
            fallback_chain,
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],

        usage_capacity: capacity,};
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,

        usage,})
}

fn test_request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(value) => {
            builder = builder.header("content-type", "application/json");
            Body::from(value.to_string())
        }
        None => Body::empty(),
    };
    builder.body(body).unwrap()
}

fn messages_request(stream: bool) -> Value {
    json!({
        "model": "claude-test",
        "max_tokens": 32,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}]
    })
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_is_public_while_api_routes_are_protected() {
    let app = build_app(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    let health = app
        .clone()
        .oneshot(test_request(Method::GET, "/health", None))
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.into_body().collect().await.unwrap().to_bytes(), "ok");

    let unauthorized = app
        .clone()
        .oneshot(test_request(Method::GET, "/v1/models", None))
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

    let mut authorized_request = test_request(Method::GET, "/v1/models", None);
    authorized_request
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let authorized = app.oneshot(authorized_request).await.unwrap();
    assert_eq!(authorized.status(), StatusCode::OK);
    let body = body_json(authorized).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "claude-test");
}

#[tokio::test]
async fn admin_status_requires_auth() {
    let app = build_app(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        )),
    );

    let resp = app
        .oneshot(test_request(Method::GET, "/admin/status", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn admin_status_lists_providers_with_available_summary() {
    let app = build_app(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        )),
    );

    let mut req = test_request(Method::GET, "/admin/status", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp).await;
    assert_eq!(body["status"], "ok");
    let providers = body["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 2);
    // Config declaration order: primary, then backup.
    assert_eq!(providers[0]["name"], "primary");
    assert_eq!(providers[0]["type"], "openai_compat");
    assert_eq!(providers[0]["status"], "available");
    assert!(providers[0]["last_error_status"].is_null());
    assert!(providers[0]["cooling_down_remaining_secs"].is_null());
    assert_eq!(providers[0]["models"][0], "claude-test");
    assert_eq!(providers[1]["name"], "backup");
    assert_eq!(providers[1]["status"], "available");
    assert_eq!(body["summary"]["total"], 2);
    assert_eq!(body["summary"]["available"], 2);
    assert_eq!(body["summary"]["cooling_down"], 0);
}

#[tokio::test]
async fn admin_status_marks_provider_cooling_down_after_failed_request() {
    let app = build_app(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Error(429),
            StreamBehavior::Bytes("unused"),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        )),
    );

    // Trigger fallback: primary returns 429 (cooldownable), backup serves.
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    assert_eq!(
        msg_resp.headers()["x-llmproxy-failed-providers"],
        "primary:429"
    );

    // The status endpoint must now report primary as cooling down.
    let mut req = test_request(Method::GET, "/admin/status", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp).await;
    let providers = body["providers"].as_array().unwrap();
    let primary = providers.iter().find(|p| p["name"] == "primary").unwrap();
    assert_eq!(primary["status"], "cooling_down");
    assert_eq!(primary["last_error_status"], 429);
    let remaining = primary["cooling_down_remaining_secs"].as_u64().unwrap();
    assert!(
        (55..=60).contains(&remaining),
        "429 cooldown should report ~60s remaining, got {remaining}"
    );
    let backup = providers.iter().find(|p| p["name"] == "backup").unwrap();
    assert_eq!(backup["status"], "available");
    assert_eq!(body["summary"]["total"], 2);
    assert_eq!(body["summary"]["available"], 1);
    assert_eq!(body["summary"]["cooling_down"], 1);
}

#[tokio::test]
async fn count_tokens_returns_word_based_estimate() {
    // R5: the old `len(json) / 4` heuristic under-counted English
    // inputs by up to 27% (e.g. 9-word panagram: 11 estimated vs 14
    // actual). The new estimator walks the JSON tree and counts
    // ceil(word_len / 3.5) per word. See fix-R5 in docs/TEST_ISSUES.md.
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    // 9-word English panagram, documented at 14 actual tokens.
    // Old impl returned 11 (under by 3); new impl returns 14 exactly.
    let input = json!({
        "text": "the quick brown fox jumps over the lazy dog"
    });
    let response = app
        .clone()
        .oneshot(test_request(
            Method::POST,
            "/v1/messages/count_tokens",
            Some(input),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let tokens = body_json(response).await["input_tokens"].as_u64().unwrap();
    assert_eq!(tokens, 14, "panagram should be 14 tokens");

    // 8 digits = one word of length 8 → ceil(8/3.5) = 3 tokens.
    let small = json!({"text": "12345678"});
    let small_response = app
        .clone()
        .oneshot(test_request(
            Method::POST,
            "/v1/messages/count_tokens",
            Some(small),
        ))
        .await
        .unwrap();
    let small_tokens = body_json(small_response).await["input_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(small_tokens, 3, "8-digit word should be 3 tokens");

    // Empty body still floors at 1 (overhead).
    let empty = json!({});
    let empty_response = app
        .clone()
        .oneshot(test_request(
            Method::POST,
            "/v1/messages/count_tokens",
            Some(empty),
        ))
        .await
        .unwrap();
    let empty_tokens = body_json(empty_response).await["input_tokens"]
        .as_u64()
        .unwrap();
    assert!(empty_tokens >= 1, "empty body should floor at 1");
}

#[tokio::test]
async fn complete_returns_anthropic_response_with_requested_model() {
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(false)),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get("x-llmproxy-failed-providers")
        .is_none());
    let body = body_json(response).await;
    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["content"][0]["text"], "ok");
}

#[tokio::test]
async fn complete_fallback_reports_failed_provider() {
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Error(429),
            StreamBehavior::Bytes("unused"),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        )),
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(false)),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-llmproxy-failed-providers"],
        "primary:429"
    );
}

#[tokio::test]
async fn stream_fallback_sets_sse_headers_and_returns_body() {
    let sse = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Error(503),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Bytes(sse),
        )),
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(true)),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream; charset=utf-8");
    assert_eq!(response.headers()["cache-control"], "no-cache");
    assert_eq!(response.headers()["x-accel-buffering"], "no");
    assert_eq!(
        response.headers()["x-llmproxy-failed-providers"],
        "primary:503"
    );
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        sse
    );
}

#[tokio::test]
async fn unknown_model_and_malformed_json_are_rejected() {
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );
    let mut unknown = messages_request(false);
    unknown["model"] = json!("missing");

    let unknown_response = app
        .clone()
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(unknown),
        ))
        .await
        .unwrap();
    assert_eq!(unknown_response.status(), StatusCode::BAD_REQUEST);
    assert!(body_json(unknown_response).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown model"));

    let malformed = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from("{"))
        .unwrap();
    let malformed_response = app.oneshot(malformed).await.unwrap();
    assert_eq!(malformed_response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn malformed_json_returns_anthropic_error_envelope() {
    // R4: axum's default Json extractor returns `text/plain`
    // "Failed to parse the request body as JSON..." for malformed input,
    // which is inconsistent with every other error response (auth,
    // unknown model, etc.) that uses the Anthropic error envelope.
    // AppJson<T> wraps the rejection so the proxy emits the same
    // `{"type":"error","error":{...}}` shape. See fix-R4 in
    // docs/TEST_ISSUES.md.
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    // Malformed JSON: truncated object.
    let bad = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from("{"))
        .unwrap();
    let resp = app.clone().oneshot(bad).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    // Body must be JSON, not text/plain.
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("application/json"),
        "expected JSON content-type, got: {ct}"
    );
    let body = body_json(resp).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "Bad Request");
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("invalid request body"),
        "error message should describe the parse failure, got: {msg}"
    );
    drop(app);
}

#[tokio::test]
async fn missing_content_type_returns_anthropic_error_envelope() {
    // R4: a POST without `Content-Type: application/json` is a
    // malformed request — the proxy should reject it with the same
    // Anthropic envelope instead of returning a default axum error.
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .body(Body::from(r#"{"model":"claude-test","messages":[]}"#))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "Bad Request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("application/json"),
        "error message should mention the required content type"
    );
    drop(app);
}

#[tokio::test]
async fn wrong_provider_output_types_return_internal_errors() {
    let complete_app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Stream,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );
    let complete_response = complete_app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(false)),
        ))
        .await
        .unwrap();
    assert_eq!(complete_response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let stream_app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Json,
        ),
        None,
    );
    let stream_response = stream_app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(true)),
        ))
        .await
        .unwrap();
    assert_eq!(stream_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn upstream_stream_item_error_terminates_body() {
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::ItemError,
        ),
        None,
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(true)),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // The body must contain an `event: error` SSE chunk so the client
    // can distinguish an aborted stream from a normal end-of-stream.
    let body = response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(
        body_str.contains("event: error"),
        "expected event:error in body, got: {body_str}"
    );
    assert!(body_str.contains("upstream_error"));
}

#[tokio::test]
async fn all_providers_failed_includes_header_and_last_body() {
    // Both providers return 500 — the chain is exhausted. The client
    // must see the *last* upstream error in the body (so it knows
    // what really went wrong) AND the per-provider summary in the
    // `x-llmproxy-failed-providers` header (so it knows the chain
    // was exhausted, not just one provider). Without this, callers
    // see a generic "all cooling down" message and lose the real
    // cause — see fix-B in TEST_ISSUES.md.
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Error(500),
            StreamBehavior::Bytes("unused"),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Error(500),
            StreamBehavior::Bytes("unused"),
        )),
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(false)),
        ))
        .await
        .unwrap();

    // The response status reflects the *last* upstream status (500),
    // not a generic 502 — the caller should be able to see what the
    // final upstream actually returned, not a proxy-level error code
    // that hides which provider failed and why.
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        response.headers()["x-llmproxy-failed-providers"],
        "primary:500,backup:500"
    );
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    // Body preserves the last upstream status / body so the caller can
    // diagnose the failure. TestProvider::complete uses "upstream failed"
    // for the body, which the upstream-JSON path then forwards as the
    // response body.
    assert!(
        body_str.contains("upstream failed"),
        "expected last upstream body in response, got: {body_str}"
    );
}

#[tokio::test]
async fn stream_chain_exhaustion_includes_failed_providers_header() {
    // Streaming-path counterpart of
    // `all_providers_failed_includes_header_and_last_body`: both
    // providers fail at stream() time with cooldownable statuses, so
    // the chain is exhausted before any bytes start flowing. The
    // client must see `x-llmproxy-failed-providers` summarising which
    // providers were tried (otherwise it just sees a 5xx with no clue
    // why). See fix-R9 in docs/TEST_ISSUES.md.
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Error(429),
        ),
        Some(provider(
            "backup",
            CompleteBehavior::Json,
            StreamBehavior::Error(503),
        )),
    );

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages",
            Some(messages_request(true)),
        ))
        .await
        .unwrap();

    // Last upstream status (503) is forwarded as the response status;
    // 429 is non-terminal so the chain falls through to backup.
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-llmproxy-failed-providers"],
        "primary:429,backup:503"
    );
}

#[tokio::test]
async fn admin_copilot_auth_returns_404_when_no_copilot_provider() {
    // When the proxy is configured without a github_copilot provider,
    // POST /admin/copilot/auth must return 404, not 500. See fix-R2.
    let app = build_app(
        Some("admin-key"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/admin/copilot/auth")
                .header("authorization", "Bearer admin-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert_eq!(body["error"]["type"], "not_found");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no github_copilot"),
        "body was: {body}"
    );
}

#[tokio::test]
async fn admin_copilot_auth_requires_authentication() {
    // The admin endpoint must be gated behind the same auth as /v1
    // routes — anonymous callers must NOT be able to trigger bootstrap.
    let app = build_app(
        Some("admin-key"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/admin/copilot/auth")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn list_models_aggregates_static_and_provider_discovered_models() {
    // Primary provider exposes models via list_models. Includes a
    // non-object entry and an id-less object — both must be skipped
    // defensively rather than breaking the endpoint.
    let primary = Arc::new(TestProvider {
        name: "primary".to_string(),
        complete: CompleteBehavior::Json,
        stream: StreamBehavior::Bytes("unused"),
        models: Some(vec![
            json!({"id": "gpt-4o", "owned_by": "openai", "display_name": "GPT-4o"}),
            json!({"id": "claude-extra", "owned_by": "openai"}),
            json!("not-an-object"),
            json!({"no_id_here": true}),
        ]),
    }) as SharedProvider;

    // Backup provider returns 503 and contributes no models.
    let backup = Arc::new(TestProvider {
        name: "backup".to_string(),
        complete: CompleteBehavior::Error(503),
        stream: StreamBehavior::Bytes("unused"),
        models: None,
    }) as SharedProvider;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), primary);
    providers.insert("backup".to_string(), backup);

    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: Some("test-key".to_string()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
            reasoning_echo: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
            reasoning_echo: false,
            },
            // Declared but never registered in the router map: both model
            // endpoints must skip it gracefully instead of panicking.
            ProviderConfig::OpenaiCompat {
                name: "ghost".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
        ],
        models: vec![
            ModelConfig {
                name: "gpt-4o".to_string(),
                primary: "primary".to_string(),
                fallback_chain: vec!["backup".to_string()],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 2,
            },
            ModelConfig {
                name: "claude-shared".to_string(),
                primary: "primary".to_string(),
                fallback_chain: vec![],
                cooldown_seconds: 60,
                max_retries_per_provider: 1,
                max_retries_total: 1,
            },
        ],
    
        usage_capacity: 0,};
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    
        usage: llmproxy::usage::UsageStats::new(0),});

    // Auth gate applies: /v1/models requires authentication.
    let unauth = app
        .clone()
        .oneshot(test_request(Method::GET, "/v1/models", None))
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    // Authenticated request.
    let mut req = test_request(Method::GET, "/v1/models", None);
    req.headers_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp).await;
    assert_eq!(body["object"], "list");

    let data = body["data"].as_array().unwrap();
    assert_eq!(
        data.len(),
        3,
        "expected 3 entries: gpt-4o, claude-shared, claude-extra"
    );

    // gpt-4o: the routing primary wins the id collision. Intentional
    // behavior change (docs/models-api-split-plan.md §1.3): the old
    // last-occurrence-wins dedup picked whichever provider happened to
    // iterate last; now the entry from the chain's primary is kept, and
    // the upstream's own `owned_by` is preserved as `upstream_owned_by`.
    let gpt4o = data
        .iter()
        .find(|m| m["id"] == "gpt-4o")
        .expect("gpt-4o must be present");
    assert_eq!(
        gpt4o["owned_by"], "primary",
        "chain-primary entry must win dedup collision on id 'gpt-4o'"
    );
    assert_eq!(
        gpt4o["upstream_owned_by"], "openai",
        "the upstream vendor must be preserved as upstream_owned_by"
    );

    // claude-shared: static-only, no collision.
    let shared = data
        .iter()
        .find(|m| m["id"] == "claude-shared")
        .expect("claude-shared must be present");
    assert_eq!(shared["owned_by"], "llmproxy");

    // claude-extra: from provider only.
    let extra = data
        .iter()
        .find(|m| m["id"] == "claude-extra")
        .expect("claude-extra must be present");
    assert_eq!(extra["owned_by"], "primary");
}

#[tokio::test]
async fn list_models_collision_winner_follows_chain_order_deterministically() {
    // Both providers advertise id "shared-model"; a ModelConfig named
    // "shared-model" chains primary → backup, so `owned_by` must be
    // "primary" regardless of HashMap iteration order (which changes
    // every process restart).
    let mk = |name: &str| {
        Arc::new(TestProvider {
            name: name.to_string(),
            complete: CompleteBehavior::Json,
            stream: StreamBehavior::Bytes("unused"),
            models: Some(vec![json!({"id": "shared-model"})]),
        }) as SharedProvider
    };

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), mk("primary"));
    providers.insert("backup".to_string(), mk("backup"));

    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: Some("test-key".to_string()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
        ],
        models: vec![ModelConfig {
            name: "shared-model".to_string(),
            primary: "primary".to_string(),
            fallback_chain: vec!["backup".to_string()],
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
    
        usage_capacity: 0,};
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    
        usage: llmproxy::usage::UsageStats::new(0),});

    let mut req = test_request(Method::GET, "/v1/models", None);
    req.headers_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp).await;
    let data = body["data"].as_array().unwrap();
    // The registered (chain) entry also beats the same-id static entry,
    // so only one "shared-model" row survives.
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "shared-model");
    assert_eq!(
        data[0]["owned_by"], "primary",
        "collision winner must follow routing chain order"
    );
}

#[tokio::test]
async fn admin_models_lists_every_provider_without_dedup() {
    let mk = |name: &str, ids: &[&str], upstream: Option<&str>| {
        let models: Vec<Value> = ids
            .iter()
            .map(|id| match upstream {
                Some(v) => json!({"id": id, "upstream_owned_by": v}),
                None => json!({"id": id}),
            })
            .collect();
        Arc::new(TestProvider {
            name: name.to_string(),
            complete: CompleteBehavior::Json,
            stream: StreamBehavior::Bytes("unused"),
            models: Some(models),
        }) as SharedProvider
    };

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        mk("primary", &["gpt-4o", "claude-extra"], Some("openai")),
    );
    // Includes an id-less entry, which must be skipped defensively.
    providers.insert(
        "backup".to_string(),
        Arc::new(TestProvider {
            name: "backup".to_string(),
            complete: CompleteBehavior::Json,
            stream: StreamBehavior::Bytes("unused"),
            models: Some(vec![json!({"id": "gpt-4o"}), json!({"missing_id": true})]),
        }) as SharedProvider,
    );

    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: Some("test-key".to_string()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::from([
                    ("alias-a".to_string(), "gpt-4o".to_string()),
                    ("alias-b".to_string(), "static-only".to_string()),
                ]),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
            // Declared but absent from the router map: must appear as a
            // degraded group instead of breaking the endpoint.
            ProviderConfig::OpenaiCompat {
                name: "ghost".to_string(),
                api_key: "unused".to_string(),
                api_base: "http://unused".to_string(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
        ],
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: "primary".to_string(),
            fallback_chain: vec!["backup".to_string()],
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
    
        usage_capacity: 0,};
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    
        usage: llmproxy::usage::UsageStats::new(0),});

    // Auth gate applies to /admin/* too.
    let unauth = app
        .clone()
        .oneshot(test_request(Method::GET, "/admin/models", None))
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let mut req = test_request(Method::GET, "/admin/models", None);
    req.headers_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "a failing provider must not fail the endpoint"
    );

    let body = body_json(resp).await;
    assert_eq!(body["object"], "list");
    let groups = body["providers"].as_array().unwrap();
    assert_eq!(groups.len(), 3, "one group per configured provider");

    // Declaration order of config.providers.
    assert_eq!(groups[0]["provider"], "primary");
    assert_eq!(groups[1]["provider"], "backup");
    assert_eq!(groups[2]["provider"], "ghost");

    // primary: discovered only (empty rewrite table), no dedup.
    assert_eq!(groups[0]["source"], "discovered");
    assert_eq!(groups[0]["cache_state"], "populated");
    let models = groups[0]["models"].as_array().unwrap();
    assert_eq!(models.len(), 2, "no dedup across providers here");

    // backup: non-empty rewrite table → static entries from the rewrite
    // values, merged with the discovered catalog by id; the id-less
    // discovered entry is dropped.
    assert_eq!(groups[1]["source"], "static+discovered");
    assert_eq!(groups[1]["cache_state"], "populated");
    let models = groups[1]["models"].as_array().unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids,
        vec!["gpt-4o", "static-only"],
        "duplicate rewrite value 'gpt-4o' collapses; sorted by id"
    );

    // ghost: declared but unregistered → degraded group, still 200.
    assert_eq!(groups[2]["cache_state"], "fetch_failed");
    assert_eq!(groups[2]["source"], "discovered");
    assert!(groups[2]["models"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn admin_models_reports_fetch_failed_for_provider_without_catalog() {
    // The TestProvider returns None for list_models when `models` is None.
    // A non-empty rewrite table exercises the static-only group shape
    // (source "static", fetch_failed, models from the rewrite values).
    let primary = Arc::new(TestProvider {
        name: "primary".to_string(),
        complete: CompleteBehavior::Error(503),
        stream: StreamBehavior::Bytes("unused"),
        models: None,
    }) as SharedProvider;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), primary);

    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: Some("test-key".to_string()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![ProviderConfig::OpenaiCompat {
            name: "primary".to_string(),
            api_key: "unused".to_string(),
            api_base: "http://unused".to_string(),
            model_rewrite: HashMap::from([("alias".to_string(), "static-upstream".to_string())]),
            use_proxy: false,
            provider_ignore: Vec::new(),
            reasoning_echo: false,
        }],
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: "primary".to_string(),
            fallback_chain: vec![],
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
    
        usage_capacity: 0,};
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    
        usage: llmproxy::usage::UsageStats::new(0),});

    let mut req = test_request(Method::GET, "/admin/models", None);
    req.headers_mut()
        .insert("authorization", "Bearer test-key".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "still a degraded 200");

    let body = body_json(resp).await;
    let group = &body["providers"][0];
    assert_eq!(group["provider"], "primary");
    assert_eq!(group["cache_state"], "fetch_failed");
    assert_eq!(
        group["source"], "static",
        "non-empty rewrite table + failed discovery → static-only group"
    );
    let models = group["models"].as_array().unwrap();
    assert_eq!(
        models.len(),
        1,
        "static rewrite entries survive a failed discovery"
    );
    assert_eq!(models[0]["id"], "static-upstream");
}

// ---------------------------------------------------------------------------
// /admin/usage endpoint tests (plan §Tests L846-857)
//
// Coverage required:
//   - 401 without bearer / 200 with (mirror /admin/status auth tests)
//   - JSON shape snapshot (started_at, capacity, retained, evicted_total,
//     filter, records, rollup)
//   - filter params (since, until, model, provider, group_by, limit)
//   - since=bogus → 400
//   - capacity-0 zero-body invariant
//   - collection path coverage: non-streaming + streaming record a row
// ---------------------------------------------------------------------------

use chrono::{DateTime, Utc};
use llmproxy::usage::StreamUsage;

fn ts(s: &str) -> DateTime<Utc> {
    // RFC 3339 fixed timestamp; the offset is normalized to UTC.
    DateTime::parse_from_rfc3339(s)
        .unwrap()
        .with_timezone(&Utc)
}

fn mk_record(
    started_at: &str,
    provider: &str,
    model: &str,
    input: Option<u32>,
    output: Option<u32>,
    cache_read: Option<u32>,
    cache_creation: Option<u32>,
    reasoning: Option<u32>,
) -> UsageRecord {
    let sa = ts(started_at);
    let usage = StreamUsage {
        input_tokens: input,
        output_tokens: output,
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        thinking_tokens: reasoning,
        server_tool_use: None,
    };
    let total_tokens = UsageRecord::compute_total_tokens(Some(&usage));
    UsageRecord {
        started_at: sa,
        ended_at: sa + chrono::Duration::milliseconds(100),
        elapsed_ms: 100,
        provider: provider.into(),
        model: model.into(),
        stream: false,
        outcome: Outcome::Success,
        usage: Some(usage),
        failed_providers: vec![],
        total_tokens,
    }
}

#[tokio::test]
async fn admin_usage_requires_auth_when_api_key_set() {
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
        UsageStats::new(8),
    );
    let resp = app
        .oneshot(test_request(Method::GET, "/admin/usage", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn admin_usage_open_when_api_key_unset() {
    let app = build_app_with_usage(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
        UsageStats::new(8),
    );
    let resp = app
        .oneshot(test_request(Method::GET, "/admin/usage", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_usage_capacity_zero_returns_empty_body() {
    // Plan L432/777-780: capacity==0 ⇒ the feature is off, so the
    // endpoint returns the empty-store invariant shape.
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
        UsageStats::new(0),
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["capacity"], 0);
    assert_eq!(body["retained"], 0);
    assert_eq!(body["evicted_total"], 0);
    assert_eq!(body["records"].as_array().unwrap().len(), 0);
    assert_eq!(body["rollup"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn admin_usage_shape_snapshot_when_records_present() {
    let usage = UsageStats::new(8);
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(100),
        Some(20),
        Some(0),
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("cache-control").unwrap(),
        "no-store",
        "endpoint must disable intermediate caches"
    );
    let body = body_json(resp).await;
    // Top-level bookkeeping
    assert_eq!(body["capacity"], 8);
    assert_eq!(body["retained"], 1);
    assert_eq!(body["evicted_total"], 0);
    assert!(
        body["started_at"].is_string(),
        "started_at must always be present"
    );
    // Filter echo (defaults)
    assert!(body["filter"]["since"].is_null());
    assert!(body["filter"]["until"].is_null());
    assert!(body["filter"]["model"].is_null());
    assert!(body["filter"]["provider"].is_null());
    assert_eq!(body["filter"]["group_by"], "model_provider");
    assert_eq!(body["filter"]["limit"], 1000);
    // Records page — token fields live at top level via #[serde(flatten)]
    // on the optional `usage: Option<StreamUsage>`. Fields whose
    // underlying value is None are simply absent (additive contract).
    let rec = &body["records"][0];
    assert_eq!(rec["model"], "m1");
    assert_eq!(rec["provider"], "p1");
    assert_eq!(rec["input_tokens"], 100);
    assert_eq!(rec["output_tokens"], 20);
    assert_eq!(rec["total_tokens"], 120);
    // Wire field is `cache_read_input_tokens` (additive — matches the
    // upstream Messages-API `usage` block); plan §Endpoint uses
    // `cache_read_tokens` as a short alias in the example shape.
    assert_eq!(rec["cache_read_input_tokens"], 0);
    assert!(
        rec.get("cache_creation_tokens").is_none(),
        "absent (None in source) rather than null on the wire"
    );
    assert!(
        rec.get("cache_creation_input_tokens").is_none(),
        "absent (None in source) rather than null on the wire"
    );
    assert!(
        rec.get("thinking_tokens").is_none(),
        "absent (None in source) rather than null on the wire"
    );
    // Rollup aggregation (single bucket)
    let roll = &body["rollup"][0];
    assert_eq!(roll["model"], "m1");
    assert_eq!(roll["provider"], "p1");
    assert_eq!(roll["requests"], 1);
    assert_eq!(roll["input_tokens"], 100);
    assert_eq!(roll["output_tokens"], 20);
    assert_eq!(roll["total_tokens"], 120);
    // cache_read_ratio == null because cache_read_tokens == 0 (plan L1039-1042)
    assert!(roll["cache_read_ratio"].is_null());
}

#[tokio::test]
async fn admin_usage_filter_by_since_until_half_open() {
    let usage = UsageStats::new(8);
    // 3 rows across a 3-day window
    usage.record(mk_record(
        "2026-09-07T10:00:00Z",
        "p1",
        "m1",
        Some(1),
        Some(1),
        None,
        None,
        None,
    ));
    usage.record(mk_record(
        "2026-09-08T10:00:00Z",
        "p1",
        "m1",
        Some(1),
        Some(1),
        None,
        None,
        None,
    ));
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(1),
        Some(1),
        None,
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
        usage,
    );
    // since=2026-09-08 inclusive, until=2026-09-09 exclusive
    // → only the 09-08 row should match.
    let uri = "/admin/usage?since=2026-09-08T00:00:00Z&until=2026-09-09T00:00:00Z";
    let mut req = test_request(Method::GET, uri, None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "half-open window should pick exactly one row");
    let at = recs[0]["started_at"].as_str().unwrap();
    assert!(
        at.starts_with("2026-09-08T10:00:00"),
        "expected 09-08 record, got {at}"
    );
    // Rollup must reflect the filtered set too.
    let roll = &body["rollup"][0];
    assert_eq!(roll["requests"], 1);
}

#[tokio::test]
async fn admin_usage_filter_by_model_and_provider() {
    let usage = UsageStats::new(16);
    usage.record(mk_record("2026-09-09T10:00:00Z", "p1", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:01Z", "p2", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:02Z", "p1", "m2", Some(1), Some(1), None, None, None));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    // Filter to provider=p1
    let mut req = test_request(Method::GET, "/admin/usage?provider=p1", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.clone().oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 2, "provider filter keeps p1 rows");
    assert!(recs.iter().all(|r| r["provider"] == "p1"));
    // Filter to model=m2
    let mut req = test_request(Method::GET, "/admin/usage?model=m2", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0]["model"], "m2");
}

#[tokio::test]
async fn admin_usage_group_by_model_only() {
    let usage = UsageStats::new(16);
    usage.record(mk_record("2026-09-09T10:00:00Z", "p1", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:01Z", "p2", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:02Z", "p1", "m2", Some(1), Some(1), None, None, None));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage?group_by=model", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let roll = body["rollup"].as_array().unwrap();
    assert_eq!(roll.len(), 2, "two distinct models");
    let m1 = roll.iter().find(|r| r["model"] == "m1").unwrap();
    let m2 = roll.iter().find(|r| r["model"] == "m2").unwrap();
    assert_eq!(m1["requests"], 2, "m1 served by both p1 and p2");
    assert_eq!(m2["requests"], 1);
    // provider field is suppressed in model-only mode
    assert!(m1["provider"].is_null());
}

#[tokio::test]
async fn admin_usage_cache_read_ratio_when_cache_hit_present() {
    let usage = UsageStats::new(8);
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(100),
        Some(20),
        Some(40),
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let roll = &body["rollup"][0];
    // input=100, cache_read=40. Anthropic counts cache reads inside
    // input_tokens, so the ratio is 40/100 = 0.4 — NOT 40/140 (the old
    // double-counting denominator; see C4).
    let ratio = roll["cache_read_ratio"].as_f64().unwrap();
    assert!((ratio - 0.4).abs() < 1e-9);
}

#[tokio::test]
async fn admin_usage_cache_read_ratio_is_one_for_fully_cached_request() {
    // C4 regression: a request whose input is entirely served from
    // cache (input_tokens == cache_read_input_tokens) must report a
    // cache hit ratio of 1.0. The old `cache_read / (cache_read +
    // input)` denominator double-counted the cache portion and froze
    // full-cache requests at 0.5.
    let usage = UsageStats::new(8);
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(100),
        Some(20),
        Some(100),
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let roll = &body["rollup"][0];
    let ratio = roll["cache_read_ratio"].as_f64().unwrap();
    assert!(
        (ratio - 1.0).abs() < 1e-9,
        "fully-cached request must report ratio 1.0, got {ratio}"
    );
}

#[tokio::test]
async fn admin_usage_cache_read_ratio_absent_when_no_cache_hits() {
    // Review #6 regression. When the group has no cache reads AND no
    // cache creation, `cache_read_ratio` is `None` and must serialize
    // as field-absent (skip_serializing_if = Option::is_none) — the
    // same convention as the sibling `model`/`provider` `Option`s.
    // Mixing absent-with-null in the same response shape breaks
    // consumer field-presence logic.
    let usage = UsageStats::new(8);
    // No cache_read_input_tokens, no cache_creation_input_tokens.
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(100),
        Some(20),
        None,
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let raw = std::str::from_utf8(&bytes).unwrap();
    let roll = &body_json_from_bytes(&bytes)["rollup"][0];
    // Parsed Value: `cache_read_ratio` is missing entirely (not null).
    assert!(
        roll.get("cache_read_ratio").is_none(),
        "cache_read_ratio must be absent when group has no cache activity, got: {raw}"
    );
    // Defensive belt-and-suspenders: confirm the raw JSON doesn't even
    // contain the substring. Catches accidental serialization as null.
    assert!(
        !raw.contains("cache_read_ratio"),
        "raw JSON unexpectedly contained cache_read_ratio: {raw}"
    );
}

fn body_json_from_bytes(bytes: &bytes::Bytes) -> serde_json::Value {
    serde_json::from_slice(bytes).unwrap()
}

#[tokio::test]
async fn admin_usage_limit_caps_records_but_not_rollup() {
    // Plan §Endpoint L465-466 + L591: limit is a records-page cap, but
    // rollup aggregates over the FULL filtered set.
    let usage = UsageStats::new(16);
    for i in 0..5 {
        let stamp = format!("2026-09-09T10:00:0{}Z", i);
        usage.record(mk_record(
            &stamp,
            "p1",
            "m1",
            Some(10),
            Some(2),
            None,
            None,
            None,
        ));
    }
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage?limit=2", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 2, "records page is capped by limit");
    // Newest-first: 10:00:04 then 10:00:03
    assert!(recs[0]["started_at"].as_str().unwrap().contains("10:00:04"));
    assert!(recs[1]["started_at"].as_str().unwrap().contains("10:00:03"));
    let roll = &body["rollup"][0];
    assert_eq!(
        roll["requests"], 5,
        "rollup aggregates over the full filtered set, ignoring limit"
    );
    assert_eq!(roll["input_tokens"], 50);
}

#[tokio::test]
async fn admin_usage_limit_clamps_to_max_limit() {
    let usage = UsageStats::new(8);
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    // Asking for limit=999_999 should be clamped to MAX_LIMIT=10_000.
    let mut req = test_request(Method::GET, "/admin/usage?limit=999999", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    assert_eq!(body["filter"]["limit"], 10_000);
}

#[tokio::test]
async fn admin_usage_invalid_since_returns_400() {
    let usage = UsageStats::new(8);
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage?since=bogus", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp).await;
    // The shape is the project-wide Anthropic-error envelope (C5 fix).
    assert_eq!(body["type"], "error");
}

#[tokio::test]
async fn admin_usage_invalid_group_by_returns_400() {
    let usage = UsageStats::new(8);
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage?group_by=day", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn admin_usage_since_greater_than_until_returns_empty() {
    // Plan §Endpoint L608: since > until is an HTTP 200 with empty
    // records/rollup, not a special-case error.
    let usage = UsageStats::new(8);
    usage.record(mk_record(
        "2026-09-09T10:00:00Z",
        "p1",
        "m1",
        Some(1),
        Some(1),
        None,
        None,
        None,
    ));
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let uri = "/admin/usage?since=2026-09-10T00:00:00Z&until=2026-09-09T00:00:00Z";
    let mut req = test_request(Method::GET, uri, None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["records"].as_array().unwrap().len(), 0);
    assert_eq!(body["rollup"].as_array().unwrap().len(), 0);
    // retained still reflects the unfiltered store size (plan L591).
    assert_eq!(body["retained"], 1);
}

#[tokio::test]
async fn admin_usage_evicted_total_surfaces_in_body() {
    let usage = UsageStats::new(2);
    // Three rows in a buffer of capacity 2 ⇒ 1 eviction.
    usage.record(mk_record("2026-09-09T10:00:00Z", "p1", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:01Z", "p1", "m1", Some(1), Some(1), None, None, None));
    usage.record(mk_record("2026-09-09T10:00:02Z", "p1", "m1", Some(1), Some(1), None, None, None));
    assert_eq!(usage.evicted_total(), 1);
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage,
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    assert_eq!(body["evicted_total"], 1);
    assert_eq!(body["retained"], 2);
    assert_eq!(body["capacity"], 2);
}

#[tokio::test]
async fn admin_usage_collects_via_messages_handler() {
    // End-to-end: a non-streaming /v1/messages request records one
    // UsageRecord visible at /admin/usage. Same Arc wiring the
    // plan prescribes (L850-857).
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("unused")),
        None,
        usage_for_state,
    );
    // Fire a non-streaming request first; the row lands in the
    // shared Arc<UsageStats> that the admin endpoint also reads.
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    // The handler used the TestProvider's default JSON response
    // (input_tokens=2, output_tokens=1), so a row should be visible.
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "non-streaming request recorded one row");
    let rec = &recs[0];
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["provider"], "primary");
    assert_eq!(rec["input_tokens"], 2);
    assert_eq!(rec["output_tokens"], 1);
    assert_eq!(rec["total_tokens"], 3);
    assert_eq!(rec["outcome"], "success");
}

#[tokio::test]
async fn admin_usage_records_errored_when_provider_returns_stream_for_json_request() {
    // review #5 regression. A non-streaming request whose provider
    // returns a `ProviderOutput::Stream` used to short-circuit on
    // the `else` branch (`?`) before reaching `record()`, so the
    // request left NO row in /admin/usage — even though upstream
    // may have billed. Now we must record `outcome: "errored"`.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        // Complete → Stream: the non-streaming handler sees a stream
        // where it expected JSON, hitting the new error branch.
        provider("primary", CompleteBehavior::Stream, StreamBehavior::Bytes("x")),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    // The row must still exist, tagged errored.
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "errored non-streaming request records one row");
    let rec = &recs[0];
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["provider"], "primary");
    assert_eq!(rec["outcome"], "errored");
    assert_eq!(rec["stream"], false);
    // No usage parsed on the error path → usage is null on the wire.
    assert!(rec["usage"].is_null(), "error row carries no usage");
}

#[tokio::test]
async fn admin_usage_records_errored_when_provider_returns_malformed_json() {
    // review #5 second shape: provider returns JSON that fails to
    // parse as a MessagesResponse. Same expectation — one errored
    // row, not zero.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::MalformedJson,
            StreamBehavior::Bytes("x"),
        ),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    // One errored row must be recorded despite the parse failure.
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "malformed-JSON request records one row");
    let rec = &recs[0];
    assert_eq!(rec["outcome"], "errored");
    assert!(rec["usage"].is_null());
}

#[tokio::test]
async fn admin_usage_records_errored_when_router_complete_fails() {
    // Review #11 regression: `state.router.complete(...).await?` at
    // src/server.rs:95 used to propagate `AllProvidersFailed` without
    // writing a row. Now the leak is plugged — every router-level
    // failure must still record exactly one `errored` row.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    // Single provider, returns 503 → router exhausts the chain with
    // `AllProvidersFailed` → handler must call `record_errored`
    // before propagating.
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Error(503),
            StreamBehavior::Bytes("x"),
        ),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(
        msg_resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "router.complete failure must surface as 5xx"
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(
        recs.len(),
        1,
        "router.complete failure records one errored row"
    );
    let rec = &recs[0];
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["outcome"], "errored");
    assert_eq!(rec["stream"], false);
    assert!(rec["usage"].is_null());
    // C3: the `<router>` row must carry the per-provider attempts that
    // actually fired, not an empty list — otherwise the failure is
    // invisible to rollups.
    assert_eq!(
        rec["failed_providers"],
        json!(["primary:503"]),
        "router-level failure row carries the attempted provider:status"
    );
}

#[tokio::test]
async fn admin_usage_records_errored_when_stream_response_guard_fires() {
    // C9 regression. Symmetric to the complete→stream guard at
    // review #5: when the client asks for a stream but the provider's
    // `stream` method returns `ProviderOutput::Json` (or anything
    // non-Stream), `stream_response`'s `else` branch used to return a
    // 500 with NO usage row. The request then vanished from
    // /admin/usage even though the proxy did receive it. Now the
    // guard records an `errored` row before returning.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        // stream → Json: the streaming handler sees JSON where it
        // expected a Stream, hitting the C9 guard.
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Json,
        ),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(true)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(
        msg_resp.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "stream_response guard must surface as 500"
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(
        recs.len(),
        1,
        "stream_response guard must still record one errored row"
    );
    let rec = &recs[0];
    assert_eq!(rec["provider"], "primary");
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["outcome"], "errored");
    assert_eq!(
        rec["stream"],
        true,
        "stream-shaped client request keeps stream=true on the error row"
    );
    assert!(rec["usage"].is_null());
}

#[tokio::test]
async fn admin_usage_records_errored_when_router_stream_fails() {
    // Review #11 second half: same regression for the streaming path.
    // Single provider returns a 429 from `stream` → router exhausts
    // the chain → handler must record before propagating.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Error(429),
        ),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(true)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(
        msg_resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "router.stream failure must surface as 429"
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(
        recs.len(),
        1,
        "router.stream failure records one errored row"
    );
    let rec = &recs[0];
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["outcome"], "errored");
    // Reviewed in debug-env against the mock: the router.stream error
    // row must carry the client's `stream: true`, not the old
    // hardcoded `false` that mislabeled failed streams as non-stream.
    assert_eq!(rec["stream"], true, "failed stream row keeps stream=true");
    // C3: same as the complete path — the `<router>` row carries the
    // per-provider attempts that fired.
    assert_eq!(
        rec["failed_providers"],
        json!(["primary:429"]),
        "router-level stream failure row carries the attempted provider:status"
    );
}

#[tokio::test]
async fn admin_usage_excludes_serving_provider_from_failed_providers_on_retry() {
    // C6 regression: when the serving provider had to be retried
    // in-place (max_retries_per_provider > 1, primary fails N-1 times
    // then succeeds), the SUCCESS row's `failed_providers` must drop
    // the serving provider's own failure attempts — that field is
    // for the failed *fallback chain*, not for in-place retries
    // against the eventual winner.
    //
    // Manually build the app (build_app_with_usage hardcodes
    // max_retries_per_provider = 1) so primary gets 3 attempts:
    // 2 failures + 1 success.
    let counter = Arc::new(AtomicU32::new(0));
    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        Arc::new(TestProvider {
            name: "primary".into(),
            complete: CompleteBehavior::FailThenSucceed {
                status: 429,
                fail_count: 2,
                counter: counter.clone(),
            },
            stream: StreamBehavior::Bytes("unused"),
            models: None,
        }) as SharedProvider,
    );
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".into(),
            api_key: Some("secret".into()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![ProviderConfig::OpenaiCompat {
            name: "primary".into(),
            api_key: "k".into(),
            api_base: "http://x".into(),
            model_rewrite: Default::default(),
            use_proxy: false,
            provider_ignore: Vec::new(),
            reasoning_echo: false,
        }],
        models: vec![ModelConfig {
            name: "claude-test".into(),
            primary: "primary".into(),
            fallback_chain: vec![],
            cooldown_seconds: 60,
            max_retries_per_provider: 3,
            max_retries_total: 3,
        }],
        usage_capacity: usage.capacity(),
    };
    let cfg = Arc::new(cfg);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(cfg.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config: cfg,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
        usage: usage_for_state,
    });
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    // Response header is also filtered — must be absent because the
    // only attempt was against the (served) primary.
    assert!(
        msg_resp.headers().get("x-llmproxy-failed-providers").is_none(),
        "serving provider's own retries must not appear in the header"
    );
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1);
    let rec = &recs[0];
    assert_eq!(rec["provider"], "primary");
    assert_eq!(rec["outcome"], "success");
    assert_eq!(
        rec["failed_providers"],
        json!([]),
        "serving provider's in-place retries must not appear in failed_providers"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        3,
        "primary should have been hit 3 times (2 fails + 1 success)"
    );
}

#[tokio::test]
async fn admin_usage_keeps_fallback_failures_when_serving_provider_is_different() {
    // C6 second leg: when the failed chain contains a DIFFERENT
    // provider than the eventual winner, that provider's failure
    // MUST remain in failed_providers. The filter only drops the
    // serving provider's own attempts — co-existing fallback
    // failures pass through untouched.
    //
    // Build manually so primary returns 429 every time (max
    // retries=1), backup returns a normal JSON, served_by = "backup".
    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        Arc::new(TestProvider {
            name: "primary".into(),
            complete: CompleteBehavior::Error(429),
            stream: StreamBehavior::Bytes("unused"),
            models: None,
        }) as SharedProvider,
    );
    providers.insert(
        "backup".to_string(),
        Arc::new(TestProvider {
            name: "backup".into(),
            complete: CompleteBehavior::Json,
            stream: StreamBehavior::Bytes("unused"),
            models: None,
        }) as SharedProvider,
    );
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".into(),
            api_key: Some("secret".into()),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".into(),
                api_key: "k".into(),
                api_base: "http://x".into(),
                model_rewrite: Default::default(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".into(),
                api_key: "k".into(),
                api_base: "http://x".into(),
                model_rewrite: Default::default(),
                use_proxy: false,
                provider_ignore: Vec::new(),
                reasoning_echo: false,
            },
        ],
        models: vec![ModelConfig {
            name: "claude-test".into(),
            primary: "primary".into(),
            fallback_chain: vec!["backup".into()],
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
        usage_capacity: usage.capacity(),
    };
    let cfg = Arc::new(cfg);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(cfg.clone(), providers, cooldown.clone()));
    let app = llmproxy::server::build_router(AppState {
        config: cfg,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
        usage: usage_for_state,
    });
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(false)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1);
    let rec = &recs[0];
    assert_eq!(rec["provider"], "backup");
    assert_eq!(rec["outcome"], "success");
    // The serving provider (backup) is filtered out, primary's 429
    // stays. This proves the filter is targeted (by served_by) and
    // doesn't accidentally drop all attempts.
    assert_eq!(
        rec["failed_providers"],
        json!(["primary:429"]),
        "non-serving fallback provider's failure must remain"
    );
    // Header must agree with the record.
    assert_eq!(
        msg_resp.headers()["x-llmproxy-failed-providers"],
        "primary:429"
    );
}

#[tokio::test]
async fn admin_usage_records_errored_when_model_is_unknown() {
    // Review #11 third leg: `find_model` returning `None` used to
    // return `BadRequest` via `?` before `start` was even captured,
    // so the row couldn't be written at all. Now `start` is captured
    // first and the unknown-model branch records before propagating.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider("primary", CompleteBehavior::Json, StreamBehavior::Bytes("x")),
        None,
        usage_for_state,
    );
    let mut req_body = messages_request(false);
    req_body["model"] = json!("this-model-does-not-exist");
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(req_body),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::BAD_REQUEST);
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(
        recs.len(),
        1,
        "unknown-model request records one errored row"
    );
    let rec = &recs[0];
    assert_eq!(rec["model"], "this-model-does-not-exist");
    assert_eq!(rec["outcome"], "errored");
}

#[tokio::test]
async fn admin_usage_dropped_stream_does_not_record_phantom_success() {
    // opus #4 regression. When the client TCP socket closes mid-stream
    // (or axum's `Body::from_stream` drops the wrapper before
    // `Ready(None)` ever fires), `MappedStream::Drop` runs without a
    // normal completion. The previous behaviour wrote a row tagged
    // `outcome: "success"` — phantom accounting for work the client
    // never received. Plan L751-757 says v1 accepts the undercount
    // rather than mislabel, so Drop must not write any row.
    //
    // We simulate by firing a streaming request whose provider emits
    // exactly one chunk then `None` — we drop the response body
    // without fully consuming it. axum's body-into-stream machinery
    // plus our `MappedStream::Drop` then fires before the inner
    // stream sees `Ready(None)`.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            // The handler picks the JSON branch for `complete` and
            // the stream branch for `stream`; we want the stream
            // branch, so stream behavior matters here. We emit a
            // tiny SSE chunk and let the stream complete normally
            // — but we'll discard the response body, so the
            // drop-on-the-floor scenario is what axum sees.
            CompleteBehavior::Json,
            StreamBehavior::Bytes(
                "event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
            ),
        ),
        None,
        usage_for_state,
    );
    // Fire the request. We do NOT consume the body — we drop the
    // response after the status check.
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(true)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    // Discard the body without consuming it. axum will then drop
    // the response-body stream → which drops `MappedStream` →
    // which runs our `Drop` impl.
    drop(msg_resp.into_body());

    // Now query /admin/usage. There must be NO row: the stream
    // never reached `Ready(None)`, so the only path that could
    // have written a record was Drop, and we've now made Drop a
    // no-op for incomplete streams.
    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    assert_eq!(
        body["records"].as_array().unwrap().len(),
        0,
        "no phantom Success row from an incomplete stream"
    );
    assert_eq!(body["retained"], 0);
    assert_eq!(
        usage.evicted_total(),
        0,
        "undercount is the plan-aligned v1 behaviour"
    );
}

#[tokio::test]
async fn admin_usage_mid_stream_error_records_errored_outcome() {
    // Opus #1 regression: a streaming request where upstream emits
    // some bytes then errors must produce a UsageRecord with
    // `outcome: "errored"`. Before the fix, the `Some(Err)` arm set
    // `done=true` and relied on `MappedStream::Drop` to record — but
    // Drop is a no-op for incomplete streams (opus #4), so the row
    // never landed. Operators couldn't see mid-stream upstream
    // failures or their token spend.
    //
    // The TestProvider's `ItemError` stream behavior emits one
    // Bytes::from_static chunk then `Err(Internal)` — exactly the
    // shape of an upstream mid-stream error. The proxy emits an
    // `event: error` SSE chunk to the client; the UsageRecord should
    // land with `outcome: "errored"`.
    let usage = UsageStats::new(8);
    let usage_for_state = usage.clone();
    let app = build_app_with_usage(
        Some("secret"),
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::ItemError,
        ),
        None,
        usage_for_state,
    );
    let mut msg_req = test_request(
        Method::POST,
        "/v1/messages",
        Some(messages_request(true)),
    );
    msg_req
        .headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let msg_resp = app.clone().oneshot(msg_req).await.unwrap();
    assert_eq!(msg_resp.status(), StatusCode::OK);
    // Drain the body so the test doesn't leak the stream and Drop
    // fires before we query /admin/usage.
    let _ = msg_resp.into_body().collect().await.unwrap().to_bytes();

    let mut req = test_request(Method::GET, "/admin/usage", None);
    req.headers_mut()
        .insert("authorization", "Bearer secret".parse().unwrap());
    let body = body_json(app.oneshot(req).await.unwrap()).await;
    let recs = body["records"].as_array().unwrap();
    assert_eq!(
        recs.len(),
        1,
        "mid-stream error must produce exactly one UsageRecord"
    );
    let rec = &recs[0];
    assert_eq!(rec["model"], "claude-test");
    assert_eq!(rec["provider"], "primary");
    assert_eq!(
        rec["outcome"], "errored",
        "Outcome must reflect mid-stream upstream failure, not Success"
    );
    // The rollup should also reflect the row.
    let roll = &body["rollup"][0];
    assert_eq!(roll["requests"], 1);
}

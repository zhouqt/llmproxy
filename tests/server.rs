use std::collections::HashMap;
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
use llmproxy::config::{ApiFormat, Config, ModelConfig, ProviderConfig, ServerConfig};
use llmproxy::cooldown::CooldownCache;
use llmproxy::error::{ProxyError, Result};
use llmproxy::providers::{Provider, ProviderOutput, SharedProvider};
use llmproxy::router::Router;
use llmproxy::state::AppState;

enum CompleteBehavior {
    Json,
    Stream,
    Error(u16),
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
}

#[async_trait]
impl Provider for TestProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn api_format(&self) -> ApiFormat {
        ApiFormat::Anthropic
    }

    async fn complete(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        match self.complete {
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
            CompleteBehavior::Stream => {
                Ok(ProviderOutput::Stream(Box::new(stream::empty())))
            }
            CompleteBehavior::Error(status) => Err(ProxyError::Upstream {
                status,
                body: "upstream failed".to_string(),
            }),
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
    })
}

fn build_app(
    api_key: Option<&str>,
    primary: SharedProvider,
    backup: Option<SharedProvider>,
) -> axum::Router {
    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), primary);
    let mut provider_configs = vec![ProviderConfig::OpenaiCompat {
        name: "primary".to_string(),
        api_key: "unused".to_string(),
        api_base: "http://unused".to_string(),
        model_rewrite: HashMap::new(),
        use_proxy: false,
    }];
    let fallback_chain = if let Some(backup) = backup {
        providers.insert("backup".to_string(), backup);
        provider_configs.push(ProviderConfig::OpenaiCompat {
            name: "backup".to_string(),
            api_key: "unused".to_string(),
            api_base: "http://unused".to_string(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        });
        vec!["backup".to_string()]
    } else {
        vec![]
    };
    let config = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: api_key.map(str::to_string),
        },
        proxy: Default::default(),
        providers: provider_configs,
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: "primary".to_string(),
            fallback_chain,
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
        logging: Default::default(),
    };
    let config = Arc::new(config);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(config.clone(), providers, cooldown.clone()));
    llmproxy::server::build_router(AppState {
        config,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    })
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

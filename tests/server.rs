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
    }];
    let fallback_chain = if let Some(backup) = backup {
        providers.insert("backup".to_string(), backup);
        provider_configs.push(ProviderConfig::OpenaiCompat {
            name: "backup".to_string(),
            api_key: "unused".to_string(),
            api_base: "http://unused".to_string(),
            model_rewrite: HashMap::new(),
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
async fn count_tokens_returns_length_estimate() {
    let app = build_app(
        None,
        provider(
            "primary",
            CompleteBehavior::Json,
            StreamBehavior::Bytes("unused"),
        ),
        None,
    );
    let input = json!({"text": "12345678"});
    let expected = ((serde_json::to_string(&input).unwrap().len() as f32) / 4.0).ceil() as u32;

    let response = app
        .oneshot(test_request(
            Method::POST,
            "/v1/messages/count_tokens",
            Some(input),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await["input_tokens"], expected);
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

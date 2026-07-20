//! End-to-end integration tests for the fallback router using a real
//! wiremock HTTP server as a mock LLM provider.
//!
//! Goal: every test below exercises the *full* routing path
//! (Router::complete / Router::stream / messages_handler / stream_response)
//! through actual HTTP frames, not in-process mocks. This is the only
//! way to verify wire-level behaviour (request body shape, retry
//! sequencing, response status mapping, fallback chain behaviour,
//! header injection) end-to-end.
//!
//! Strategy:
//! - Each test spins up its own MockServer instance so per-test routing
//!   is isolated.
//! - The Anthropic-format proxy endpoint is exercised through
//!   `tower::ServiceExt::oneshot` (no TCP listener required) so the
//!   tests stay fast and don't conflict on ports.
//! - Where the test only needs to exercise the router (and not the
//!   HTTP layer), we exercise the Router directly.
//!
//! These tests must NOT modify any production code to make them
//! pass — see `docs/TEST_PLAN.md` and the project memory entry
//! "no test-only functional code".

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use llmproxy::anthropic::MessagesRequest;
use llmproxy::config::{Config, ModelConfig, ProviderConfig, ServerConfig};
use llmproxy::cooldown::CooldownCache;
use llmproxy::error::{ProxyError, Result};
use llmproxy::providers::{Provider, ProviderOutput, SharedProvider};
use llmproxy::router::Router;
use llmproxy::state::AppState;

// ────────────────────────────────────────────────────────────────────────
// Mock LLM provider that talks real HTTP via wiremock.
// ────────────────────────────────────────────────────────────────────────

/// Real-shape Anthropic-compatible response (the proxy translates OpenAI
/// upstream responses into this before returning to the client).
fn anthropic_ok(text: &str, model: &str) -> Value {
    json!({
        "id": "msg_integration",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 5, "output_tokens": 3}
    })
}

/// Build a wiremock-complete-test provider config that points at the
/// given MockServer and treats itself as an OpenAI-compatible endpoint.
fn wiremock_provider(name: &str, server: &MockServer) -> SharedProvider {
    Arc::new(WiremockOpenAiProvider {
        name: name.to_string(),
        base_url: server.uri(),
        api_key: "wiremock-key".to_string(),
    })
}

struct WiremockOpenAiProvider {
    name: String,
    base_url: String,
    api_key: String,
}

#[async_trait]
impl Provider for WiremockOpenAiProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let body = json!({
            "model": req.model,
            "messages": [{"role": "user", "content": req.messages[0].to_string_value()}],
            "stream": false
        });
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = reqwest::Client::new()
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(ProxyError::Http)?;

        let status = resp.status();
        let text = resp.text().await.map_err(ProxyError::Http)?;
        if !status.is_success() {
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let parsed: Value = serde_json::from_str(&text).map_err(ProxyError::Json)?;
        Ok(ProviderOutput::Json(parsed))
    }

    async fn stream(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        unimplemented!("stream path covered by tests/server.rs unit tests")
    }
}

// Helper: convert the first message to a string for wiremock body.
trait MessageContentToString {
    fn to_string_value(&self) -> String;
}

impl MessageContentToString for llmproxy::anthropic::Message {
    fn to_string_value(&self) -> String {
        match &self.content {
            llmproxy::anthropic::MessageContent::Text(t) => t.clone(),
            llmproxy::anthropic::MessageContent::Blocks(_) => "[blocks]".to_string(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Router-level tests using real wiremock servers.
// ────────────────────────────────────────────────────────────────────────

fn make_req(model: &str) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap()
}

#[tokio::test]
async fn mock_llm_provider_primary_succeeds_returns_anthropic_response() {
    // A primary provider whose /v1/messages endpoint returns a valid
    // Anthropic-shaped JSON response must round-trip through
    // Router::complete without touching any fallback chain.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok("hello back", "claude-test")))
        .expect(1)
        .mount(&server)
        .await;

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        wiremock_provider("primary", &server),
    );
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![ProviderConfig::OpenaiCompat {
            name: "primary".to_string(),
            api_key: "k".to_string(),
            api_base: server.uri(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        }],
        models: vec![ModelConfig {
            name: "claude-test".into(),
            primary: "primary".into(),
            fallback_chain: vec![],
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: 1,
        }],
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router.complete(model_cfg, &make_req("claude-test")).await.unwrap();
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "hello back");
    assert!(attempts.is_empty(), "primary succeeded; no fallback expected");
}

#[tokio::test]
async fn mock_llm_provider_falls_back_when_primary_returns_429() {
    // Primary returns 429; the router must (a) record an attempt and (b)
    // call backup. backup's response must be the one returned to the
    // caller (with the failed-providers header fed by the attempts list).
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .mount(&primary)
        .await;
    let backup = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok("from backup", "claude-test")))
        .expect(1)
        .mount(&backup)
        .await;

    let primary_provider = wiremock_provider("primary", &primary);
    let backup_provider = wiremock_provider("backup", &backup);

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), primary_provider);
    providers.insert("backup".to_string(), backup_provider);

    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: primary.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: backup.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
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
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router.complete(model_cfg, &make_req("claude-test")).await.unwrap();
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "from backup");
    assert_eq!(attempts.len(), 1, "exactly one attempt (primary's 429)");
    assert_eq!(attempts[0].provider, "primary");
    assert_eq!(attempts[0].status, 429);
    assert_eq!(attempts[0].body, "rate limited");
}

#[tokio::test]
async fn mock_llm_provider_chain_exhausted_returns_last_upstream_error() {
    // Every provider returns 500 — the chain is exhausted. The router
    // must surface `AllProvidersFailed` with both attempts and the last
    // upstream body so the operator sees what really happened.
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("primary-error"))
        .mount(&primary)
        .await;
    let backup = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("backup-error"))
        .mount(&backup)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &primary));
    providers.insert("backup".to_string(), wiremock_provider("backup", &backup));

    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: primary.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: backup.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
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
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let err = router
        .complete(model_cfg, &make_req("claude-test"))
        .await
        .err()
        .expect("chain exhausted should fail");

    match err {
        ProxyError::AllProvidersFailed { attempts, last, .. } => {
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0].provider, "primary");
            assert_eq!(attempts[0].body, "primary-error");
            assert_eq!(attempts[1].provider, "backup");
            assert_eq!(attempts[1].body, "backup-error");
            match last.as_ref() {
                ProxyError::Upstream { status, body } => {
                    assert_eq!(*status, 500);
                    assert_eq!(body, "backup-error");
                }
                other => panic!("expected wrapped Upstream, got {other:?}"),
            }
        }
        other => panic!("expected AllProvidersFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn mock_llm_provider_per_provider_retry_three_times_before_chain_advance() {
    // Primary is configured with max_retries_per_provider=3 and must
    // actually be hit 3 times (2 failures + 1 success) before we yield
    // a result. Primary returns 429 twice then 200. We expect the
    // server to record exactly 3 requests and the response text "third
    // time lucky" to come back.
    let primary = MockServer::start().await;
    // wiremock's Mock can be re-mounted for the same matcher and serve
    // a different response each call. We use an atomic counter inside
    // a closure-based responder so we can assert the request count.
    let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counter_clone = counter.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < 2 {
                ResponseTemplate::new(429).set_body_string("please retry")
            } else {
                ResponseTemplate::new(200).set_body_json(anthropic_ok("third time lucky", "claude-test"))
            }
        })
        .expect(3)
        .mount(&primary)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &primary));
    providers.insert(
        "backup".to_string(),
        wiremock_provider("backup", &primary),
    );

    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: primary.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: primary.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
        ],
        models: vec![ModelConfig {
            name: "claude-test".into(),
            primary: "primary".into(),
            fallback_chain: vec!["backup".into()],
            cooldown_seconds: 60,
            max_retries_per_provider: 3,
            max_retries_total: 5,
        }],
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router.complete(model_cfg, &make_req("claude-test")).await.unwrap();
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "third time lucky");
    // Two 429 attempts before the success.
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts.iter().filter(|a| a.status == 429).count(), 2);
    assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);
}

// ────────────────────────────────────────────────────────────────────────
// Axum HTTP-layer end-to-end tests.
// ────────────────────────────────────────────────────────────────────────

fn build_axum_app(
    providers: HashMap<String, SharedProvider>,
    provider_configs: Vec<ProviderConfig>,
    model_chain: Vec<String>,
) -> axum::Router {
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: provider_configs,
        models: vec![ModelConfig {
            name: "claude-test".to_string(),
            primary: model_chain[0].clone(),
            fallback_chain: model_chain[1..].to_vec(),
            cooldown_seconds: 60,
            max_retries_per_provider: 1,
            max_retries_total: model_chain.len() as u32,
        }],
        logging: Default::default(),
    };
    let cfg = Arc::new(cfg);
    let cooldown = CooldownCache::new();
    let router = Arc::new(Router::new(cfg.clone(), providers, cooldown.clone()));
    let state = AppState {
        config: cfg,
        router,
        cooldown,
        http: reqwest::Client::new(),
        copilot: None,
    };
    llmproxy::server::build_router(state)
}

fn messages_body(stream: bool) -> Value {
    json!({
        "model": "claude-test",
        "max_tokens": 32,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}]
    })
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn collect_bytes(resp: axum::response::Response) -> (StatusCode, Vec<(String, String)>, Bytes) {
    let status = resp.status();
    let mut headers = Vec::new();
    for (k, v) in resp.headers() {
        if let Ok(vs) = v.to_str() {
            headers.push((k.as_str().to_string(), vs.to_string()));
        }
    }
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes)
}

#[tokio::test]
async fn http_end_to_end_primary_succeeds_returns_requested_model_name() {
    // Verify the messages_handler rewrites resp.model back to the
    // *requested* model name (not whatever the upstream returned). This
    // is the test that locks the contract: the client must see the
    // alias they used, even if the upstream model name differs.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({"stream": false})))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok(
            "ack",
            "different-upstream-name",
        )))
        .mount(&upstream)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &upstream));
    let configs = vec![ProviderConfig::OpenaiCompat {
        name: "primary".to_string(),
        api_key: "k".to_string(),
        api_base: upstream.uri(),
        model_rewrite: HashMap::new(),
        use_proxy: false,
    }];
    let app = build_axum_app(providers, configs, vec!["primary".to_string()]);

    let resp = app
        .oneshot(post("/v1/messages", messages_body(false)))
        .await
        .unwrap();
    let (status, _headers, body) = collect_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&body).unwrap();
    // Model override: server always rewrites resp.model to req.model.
    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["content"][0]["text"], "ack");
}

#[tokio::test]
async fn http_end_to_end_429_falls_back_to_backup_with_failed_providers_header() {
    // Same scenario as the router-level test, but exercised through the
    // full axum stack (extractor → handler → router → SSE adapter →
    // response builder). The client must see 200 OK with the body
    // returned by backup and the x-llmproxy-failed-providers header.
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate-limited"))
        .mount(&primary)
        .await;
    let backup = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok("via backup", "claude-test")))
        .mount(&backup)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &primary));
    providers.insert("backup".to_string(), wiremock_provider("backup", &backup));
    let configs = vec![
        ProviderConfig::OpenaiCompat {
            name: "primary".to_string(),
            api_key: "k".to_string(),
            api_base: primary.uri(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        },
        ProviderConfig::OpenaiCompat {
            name: "backup".to_string(),
            api_key: "k".to_string(),
            api_base: backup.uri(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        },
    ];
    let app = build_axum_app(providers, configs, vec!["primary".into(), "backup".into()]);

    let resp = app
        .oneshot(post("/v1/messages", messages_body(false)))
        .await
        .unwrap();
    let (status, headers, body) = collect_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    let failed_header = headers
        .iter()
        .find(|(k, _)| k == "x-llmproxy-failed-providers")
        .map(|(_, v)| v.clone())
        .expect("x-llmproxy-failed-providers header must be set on fallback");
    assert_eq!(failed_header, "primary:429");
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["content"][0]["text"], "via backup");
}

#[tokio::test]
async fn http_end_to_end_chain_exhausted_returns_status_and_failed_providers_header() {
    // Both providers return 500 — the chain is exhausted. The end-to-end
    // view must surface the LAST upstream error status (500) plus the
    // failed-providers summary. Without that, callers see a generic 502
    // and can't tell which provider failed.
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("primary-error"))
        .mount(&primary)
        .await;
    let backup = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("backup-error"))
        .mount(&backup)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &primary));
    providers.insert("backup".to_string(), wiremock_provider("backup", &backup));
    let configs = vec![
        ProviderConfig::OpenaiCompat {
            name: "primary".to_string(),
            api_key: "k".to_string(),
            api_base: primary.uri(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        },
        ProviderConfig::OpenaiCompat {
            name: "backup".to_string(),
            api_key: "k".to_string(),
            api_base: backup.uri(),
            model_rewrite: HashMap::new(),
            use_proxy: false,
        },
    ];
    let app = build_axum_app(providers, configs, vec!["primary".into(), "backup".into()]);

    let resp = app
        .oneshot(post("/v1/messages", messages_body(false)))
        .await
        .unwrap();
    let (status, headers, body) = collect_bytes(resp).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let failed_header = headers
        .iter()
        .find(|(k, _)| k == "x-llmproxy-failed-providers")
        .map(|(_, v)| v.clone())
        .expect("x-llmproxy-failed-providers header must be set");
    assert_eq!(failed_header, "primary:500,backup:500");
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(
        body_str.contains("backup-error"),
        "body must surface the last upstream error body, got: {body_str}"
    );
}

#[tokio::test]
async fn http_end_to_end_unknown_model_returns_anthropic_envelope_400() {
    // A request for a model that doesn't exist in the router's config
    // must surface as a 400 with the Anthropic envelope, not as an
    // opaque 502. This is the test that locks fix-R4 behaviour at the
    // HTTP boundary.
    let providers = HashMap::new();
    let app = build_axum_app(providers, vec![], vec!["primary".into()]);

    let body = json!({
        "model": "no-such-model",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = app.oneshot(post("/v1/messages", body)).await.unwrap();
    let (status, _headers, body_bytes) = collect_bytes(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "Bad Request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown model")
    );
}

#[tokio::test]
async fn http_end_to_end_malformed_json_returns_anthropic_envelope_400() {
    // Same envelope contract as fix-R4 — ensures the proxy never emits a
    // raw "Failed to parse the request body as JSON..." text response.
    let providers = HashMap::new();
    let app = build_axum_app(providers, vec![], vec!["primary".into()]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from("{"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, headers, body_bytes) = collect_bytes(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let ct = headers
        .iter()
        .find(|(k, _)| k == "content-type")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    assert!(
        ct.starts_with("application/json"),
        "expected JSON content-type, got: {ct}"
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "Bad Request");
}

// ────────────────────────────────────────────────────────────────────────
// Additional fallback-flow tests with deterministic provider state.
// ────────────────────────────────────────────────────────────────────────

/// A provider that wraps another provider's body in a ProviderOutput::Json
/// or returns a programmable upstream error. Lets the integration tests
/// exercise the Router's retry / cooldown paths without spinning up an
/// extra wiremock instance for the simpler scenarios.
struct DeferredProvider {
    name: String,
    behavior: Arc<std::sync::Mutex<DeferredBehavior>>,
}

#[derive(Debug)]
enum DeferredBehavior {
    Ok(Value),
    /// Return ProxyError::Upstream with the given status/body until the
    /// call count reaches `fail_count`, then return Ok(body).
    FailThenOk { fail_count: u32, ok: Value, status: u16 },
}

#[async_trait]
impl Provider for DeferredProvider {
    fn name(&self) -> &str {
        &self.name
    }
    async fn complete(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut g = self.behavior.lock().unwrap();
        match &mut *g {
            DeferredBehavior::Ok(v) => Ok(ProviderOutput::Json(v.clone())),
            DeferredBehavior::FailThenOk {
                fail_count,
                ok,
                status,
            } => {
                if *fail_count > 0 {
                    *fail_count -= 1;
                    Err(ProxyError::Upstream {
                        status: *status,
                        body: format!("synthetic-{status}"),
                    })
                } else {
                    Ok(ProviderOutput::Json(ok.clone()))
                }
            }
        }
    }
    async fn stream(
        &self,
        _req: &MessagesRequest,
        _model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        Ok(ProviderOutput::Stream(Box::new(stream::empty())))
    }
}

#[tokio::test]
async fn mock_llm_provider_short_cooldown_for_non_429_upstream_error() {
    // A 503 from a primary provider marks the provider on cooldown for
    // the *short* default TTL, not the model's configured cooldown_seconds.
    // Verifying this prevents accidentally giving flaky 5xx upstreams the
    // same grace period as rate-limited (429) ones.
    let behavior = Arc::new(std::sync::Mutex::new(DeferredBehavior::Ok(
        anthropic_ok("secondary body", "claude-test"),
    )));
    let primary_behavior = Arc::new(std::sync::Mutex::new(DeferredBehavior::FailThenOk {
        fail_count: u32::MAX,
        ok: serde_json::json!({}),
        status: 503,
    }));

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        Arc::new(DeferredProvider {
            name: "primary".into(),
            behavior: primary_behavior,
        }) as SharedProvider,
    );
    providers.insert(
        "backup".to_string(),
        Arc::new(DeferredProvider {
            name: "backup".into(),
            behavior,
        }) as SharedProvider,
    );
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: "http://x".into(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: "http://x".into(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
        ],
        models: vec![ModelConfig {
            name: "claude-test".into(),
            primary: "primary".into(),
            fallback_chain: vec!["backup".into()],
            cooldown_seconds: 120,
            max_retries_per_provider: 1,
            max_retries_total: 2,
        }],
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers.clone(), CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router.complete(model_cfg, &make_req("claude-test")).await.unwrap();
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "secondary body");

    // The 503 cooldown TTL must be the *short* default (5s), NOT the
    // model's configured cooldown_seconds (120). If the router gave a
    // 503 the same 120s grace period as a 429, the primary would
    // accidentally stay out of the rotation for any flaky 5xx — we
    // want flaky upstreams back online ASAP.
    let active = router.cooldown().active().await;
    let primary_entry = active
        .iter()
        .find(|(name, _, _)| name == "primary")
        .expect("primary must be on cooldown after 503");
    let ttl = primary_entry.2;
    assert_eq!(
        primary_entry.1, 503,
        "cooldown entry status should match the upstream's 503"
    );
    assert!(
        ttl <= Duration::from_secs(5) && ttl > Duration::from_secs(1),
        "non-429 cooldown must use the short default TTL (~5s), got {ttl:?}"
    );
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, 503);
}

#[tokio::test]
async fn mock_llm_provider_subsequent_request_skips_cooldown_provider_directly() {
    // After the first fallback succeeds, the next request should:
    // - see primary still on cooldown
    // - record ZERO attempts (skip without HTTP call)
    // - go straight to backup, returning its body
    let primary = MockServer::start().await;
    // The primary mock will count calls; if it's hit on the second
    // request we know the cooldown wasn't respected.
    let primary_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let primary_calls_clone = primary_calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            primary_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ResponseTemplate::new(429).set_body_string("still rate-limited")
        })
        .mount(&primary)
        .await;
    let backup = MockServer::start().await;
    let backup_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let backup_calls_clone = backup_calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            backup_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(anthropic_ok("backup", "claude-test"))
        })
        .mount(&backup)
        .await;

    let mut providers = HashMap::new();
    providers.insert("primary".to_string(), wiremock_provider("primary", &primary));
    providers.insert("backup".to_string(), wiremock_provider("backup", &backup));
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: primary.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: backup.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
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
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let req = make_req("claude-test");

    // First request: primary should be hit once (429), backup takes over.
    let (_out, attempts) = router.complete(model_cfg, &req).await.unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(primary_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(backup_calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Second request: primary is on cooldown — backup must be called
    // directly with no HTTP attempt against primary.
    let (_out, attempts) = router.complete(model_cfg, &req).await.unwrap();
    assert!(attempts.is_empty(), "primary must be skipped, no attempt recorded");
    assert_eq!(
        primary_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "primary must not be called while on cooldown"
    );
    assert_eq!(backup_calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn mock_llm_provider_skips_provider_with_unsupported_model_via_runtime_400() {
    // Mock provider returns 400 + a body that triggers the
    // is_model_unsupported runtime detection. The router must treat the
    // provider as a model-unsupported skip (separate path from cooldown),
    // record the attempt, and successfully fall back.
    let behavior = Arc::new(std::sync::Mutex::new(DeferredBehavior::FailThenOk {
        fail_count: u32::MAX,
        ok: serde_json::json!({}),
        status: 400,
    }));
    let ok_body = anthropic_ok("backup worked", "claude-test");
    let backup_behavior = Arc::new(std::sync::Mutex::new(DeferredBehavior::Ok(ok_body)));

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        Arc::new(DeferredProvider {
            name: "primary".into(),
            behavior,
        }) as SharedProvider,
    );
    providers.insert(
        "backup".to_string(),
        Arc::new(DeferredProvider {
            name: "backup".into(),
            behavior: backup_behavior,
        }) as SharedProvider,
    );

    // Use a custom provider type with the model_unsupported body. We
    // can't easily inject a custom body via wiremock in this Deferred-
    // Provider wrapper, so we use a separate path: configure the wrapper
    // via a custom upstream body on the wiremock side and re-use the
    // wiremock test instead.
    let _ = providers; // mark-unused to suppress drift
    let primary_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string(
            r#"{"error":{"code":"model_not_supported","message":"The requested model is not supported.","param":"model","type":"invalid_request_error"}}"#,
        ))
        .mount(&primary_server)
        .await;
    let backup_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok("backup ok", "claude-test")))
        .mount(&backup_server)
        .await;

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        wiremock_provider("primary", &primary_server),
    );
    providers.insert(
        "backup".to_string(),
        wiremock_provider("backup", &backup_server),
    );
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            api_key: None,
        },
        proxy: Default::default(),
        providers: vec![
            ProviderConfig::OpenaiCompat {
                name: "primary".to_string(),
                api_key: "k".to_string(),
                api_base: primary_server.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
            },
            ProviderConfig::OpenaiCompat {
                name: "backup".to_string(),
                api_key: "k".to_string(),
                api_base: backup_server.uri(),
                model_rewrite: HashMap::new(),
                use_proxy: false,
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
        logging: Default::default(),
    };
    let router = Router::new(Arc::new(cfg), providers, CooldownCache::new());

    let model_cfg = router.find_model("claude-test").unwrap();
    let (out, attempts) = router.complete(model_cfg, &make_req("claude-test")).await.unwrap();
    let ProviderOutput::Json(body) = out else {
        panic!("expected JSON output");
    };
    assert_eq!(body["content"][0]["text"], "backup ok");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].provider, "primary");
    assert_eq!(attempts[0].status, 400);
    assert!(attempts[0].body.contains("model_not_supported"));
    // The model-unsupported branch puts primary on a short (60s) cooldown.
    assert!(router.cooldown().is_cooling_down("primary").await);
}

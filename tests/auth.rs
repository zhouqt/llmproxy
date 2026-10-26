//! Integration tests for the auth middleware.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::{get, Router};
use tower::ServiceExt;

use llmproxy::auth::require_auth;
use llmproxy::config::{Config, ServerConfig};
use llmproxy::secrets::secret_string;
use llmproxy::state::AppState;
use llmproxy::usage::UsageStats;

fn dummy_state(api_key: Option<String>) -> AppState {
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".into(),
            api_key: api_key.map(secret_string),
        },
        proxy: Default::default(),
        user_agent: llmproxy::config::default_user_agent(),
        secrets: Default::default(),
        providers: vec![],
        models: vec![],
        ..Config::default()
    };
    // We don't exercise providers here — just auth.
    AppState {
        config: Arc::new(cfg),
        router: Arc::new(llmproxy::router::Router::new(
            Arc::new(Config {
                server: Default::default(),
                proxy: Default::default(),
                user_agent: llmproxy::config::default_user_agent(),
                secrets: Default::default(),
                providers: vec![],
                models: vec![],
                ..Config::default()
            }),
            Default::default(),
            llmproxy::cooldown::CooldownCache::new(),
        )),
        cooldown: llmproxy::cooldown::CooldownCache::new(),
        http: reqwest::Client::new(),
        copilot: None,
        usage: UsageStats::new(8),
        auth_limiter: llmproxy::auth::AuthRateLimiter::new(),
    }
}

async fn echo_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

fn build_test_app(api_key: Option<String>) -> Router {
    build_test_app_with_state(dummy_state(api_key))
}

fn build_test_app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/protected", get(echo_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_auth,
        ))
        .with_state(state)
}

fn bad_request() -> Request<Body> {
    Request::builder()
        .uri("/protected")
        .header("authorization", "Bearer wrong")
        .body(Body::empty())
        .unwrap()
}

fn good_request() -> Request<Body> {
    Request::builder()
        .uri("/protected")
        .header("authorization", "Bearer secret-key")
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn no_api_key_configured_passes_through() {
    let app = build_test_app(None);
    let resp = app
        .oneshot(Request::builder().uri("/protected").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn matching_bearer_passes() {
    let app = build_test_app(Some("secret-key".into()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("authorization", "Bearer secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn wrong_bearer_rejected() {
    let app = build_test_app(Some("secret-key".into()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn missing_auth_header_rejected() {
    let app = build_test_app(Some("secret-key".into()));
    let resp = app
        .oneshot(Request::builder().uri("/protected").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn x_api_key_header_accepted() {
    let app = build_test_app(Some("secret-key".into()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("x-api-key", "secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn wrong_x_api_key_rejected() {
    let app = build_test_app(Some("secret-key".into()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("x-api-key", "wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn malformed_authorization_header_rejected() {
    let app = build_test_app(Some("secret-key".into()));
    // No "Bearer " prefix
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/protected")
                .header("authorization", "secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// Use State to avoid unused warnings if helper changes.
#[allow(dead_code)]
async fn _with_state(State(_s): State<AppState>) {}

// ---- failed-auth rate limiting (audit L7) ----

/// Audit L7 T1: the first `AUTH_FAILURE_LIMIT` bad requests get 401; the
/// 31st and later get 429 with a `Retry-After` header and a
/// `rate_limit_error` body.
#[tokio::test]
async fn auth_failures_are_rate_limited_after_the_limit() {
    let app = build_test_app(Some("secret-key".into()));
    for _ in 0..llmproxy::auth::AUTH_FAILURE_LIMIT {
        let resp = app.clone().oneshot(bad_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
    let resp = app.clone().oneshot(bad_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        resp.headers().get("retry-after").is_some(),
        "429 must carry Retry-After"
    );
    let bytes = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], "rate_limit_error");
}

/// Audit L7 T2: valid tokens never touch the limiter — they keep passing
/// even while bad requests are being throttled.
#[tokio::test]
async fn valid_token_unthrottled_while_failures_are_throttled() {
    let app = build_test_app(Some("secret-key".into()));
    for _ in 0..llmproxy::auth::AUTH_FAILURE_LIMIT {
        app.clone().oneshot(bad_request()).await.unwrap();
    }
    // Window is now over the limit: bad → 429, good → 200.
    let resp = app.clone().oneshot(bad_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    for _ in 0..3 {
        let resp = app.clone().oneshot(good_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

/// Audit L7 T4: the limiter lives behind an `Arc` inside `AppState`, so
/// clones of the state (fresh routers) share one failure window.
#[tokio::test]
async fn limiter_state_is_shared_across_app_state_clones() {
    let state = dummy_state(Some("secret-key".into()));
    let app_a = build_test_app_with_state(state.clone());
    let app_b = build_test_app_with_state(state);
    for _ in 0..llmproxy::auth::AUTH_FAILURE_LIMIT {
        app_a.clone().oneshot(bad_request()).await.unwrap();
    }
    let resp = app_a.clone().oneshot(bad_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    // A router built from the cloned state sees the same window.
    let resp = app_b.oneshot(bad_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
}

/// With no api_key configured the limiter is never consulted: requests
/// pass through no matter how many there are.
#[tokio::test]
async fn no_api_key_never_throttles() {
    let app = build_test_app(None);
    for _ in 0..(llmproxy::auth::AUTH_FAILURE_LIMIT + 5) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header("authorization", "Bearer whatever")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

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
use llmproxy::state::AppState;

fn dummy_state(api_key: Option<String>) -> AppState {
    let cfg = Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".into(),
            api_key,
        },
        proxy: Default::default(),
        providers: vec![],
        models: vec![],
        logging: Default::default(),
    };
    // We don't exercise providers here — just auth.
    AppState {
        config: Arc::new(cfg),
        router: Arc::new(llmproxy::router::Router::new(
            Arc::new(Config {
                server: Default::default(),
                proxy: Default::default(),
                providers: vec![],
                models: vec![],
                logging: Default::default(),
            }),
            Default::default(),
            llmproxy::cooldown::CooldownCache::new(),
        )),
        cooldown: llmproxy::cooldown::CooldownCache::new(),
        http: reqwest::Client::new(),
        copilot: None,
    }
}

async fn echo_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

fn build_test_app(api_key: Option<String>) -> Router {
    let state = dummy_state(api_key);
    Router::new()
        .route("/protected", get(echo_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_auth,
        ))
        .with_state(state)
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

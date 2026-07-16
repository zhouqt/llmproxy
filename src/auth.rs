//! Bearer-token auth middleware.

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

pub async fn require_auth(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.server.api_key.as_deref() else {
        // No api_key configured → auth disabled.
        return next.run(req).await;
    };

    let header_value = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            // Anthropic SDK uses x-api-key for the provider-side auth; allow
            // it as a fallback so existing clients work unchanged.
            req.headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
        });

    match header_value {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(req).await
        }
        _ => (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "missing or invalid API key"
                }
            })),
        )
            .into_response(),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(!constant_time_eq(b"hello", b"helloo"));
        assert!(constant_time_eq(b"", b""));
    }
}

//! Bearer-token auth middleware.

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use secrecy::ExposeSecret;

use crate::state::AppState;

pub async fn require_auth(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state
        .config
        .server
        .api_key
        .as_ref()
        .map(|k| k.expose_secret())
    else {
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
    // Hash both sides before comparing (audit L3): the raw loop used to
    // return early on a length mismatch, which leaks the expected API
    // key's length through timing. SHA-256 digests are always 32 bytes,
    // so the comparison below has no length-dependent branch at all, and
    // SHA-256 collision resistance makes the extra step invisible to
    // correct-token traffic. `ring` is already a direct dependency (it
    // encrypts the Copilot token store), so this adds no crate.
    let da = ring::digest::digest(&ring::digest::SHA256, a);
    let db = ring::digest::digest(&ring::digest::SHA256, b);
    let mut diff: u8 = 0;
    for (x, y) in da.as_ref().iter().zip(db.as_ref().iter()) {
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

    #[test]
    fn constant_time_eq_has_no_length_shortcut() {
        // Audit L3: unequal lengths must still yield `false`, but through
        // the same 32-byte digest comparison as every other input — the
        // historical early `return false` on a length mismatch is gone.
        assert!(!constant_time_eq(b"", b"a"));
        assert!(!constant_time_eq(b"a", b""));
        assert!(!constant_time_eq(&[0u8; 64], &[0u8; 65]));

        // Long inputs: equal compares equal, a single flipped byte does not.
        let long = [7u8; 4096];
        assert!(constant_time_eq(&long, &long));
        let mut flipped = long;
        flipped[4095] ^= 1;
        assert!(!constant_time_eq(&long, &flipped));
    }
}

//! Custom request extractors.
//!
//! `AppJson` mirrors axum's `Json<T>` but converts every rejection into
//! `ProxyError::BadRequest`, which renders as the project's standard
//! Anthropic error envelope. Without this wrapper, malformed JSON /
//! missing fields come back as `text/plain` "Failed to parse the
//! request body as JSON..." — a format inconsistent with every other
//! error response the proxy emits (auth, unknown model, cooldown, etc.).
//!
//! See fix-R4 in docs/TEST_ISSUES.md.

use axum::async_trait;
use axum::extract::{FromRequest, Request};
use axum::http::header;
use axum::Json;
use serde::de::DeserializeOwned;

use crate::error::ProxyError;

#[derive(Debug, Clone, Copy, Default)]
pub struct AppJson<T>(pub T);

#[async_trait]
impl<S, T> FromRequest<S> for AppJson<T>
where
    Json<T>: FromRequest<S, Rejection = axum::extract::rejection::JsonRejection>,
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ProxyError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Content-Type check: Anthropic's API requires
        // `application/json`; reject anything else up front so the
        // client gets a clear message instead of a serde parse error.
        if !is_json_content_type(req.headers()) {
            return Err(ProxyError::BadRequest(
                "Content-Type must be application/json".to_string(),
            ));
        }
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(map_json_rejection(rejection)),
        }
    }
}

fn is_json_content_type(headers: &axum::http::HeaderMap) -> bool {
    let Some(ct) = headers.get(header::CONTENT_TYPE) else {
        return false;
    };
    let Ok(ct) = ct.to_str() else {
        return false;
    };
    // Tolerate `application/json; charset=utf-8` and similar.
    let main = ct.split(';').next().unwrap_or("").trim();
    main.eq_ignore_ascii_case("application/json")
}

fn map_json_rejection(rejection: axum::extract::rejection::JsonRejection) -> ProxyError {
    // axum splits rejections into MissingJsonContentType, JsonDataError,
    // JsonSyntaxError, BytesRejection, etc. The status() / body_text()
    // accessors give us a uniform string and a sensible status code, but
    // we discard the status and always return 400 — the proxy is
    // rejecting the *request*, not relaying an upstream error.
    let detail = rejection.body_text();
    ProxyError::BadRequest(format!("invalid request body: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_json_content_type_accepts_plain_and_charset() {
        let mut h = axum::http::HeaderMap::new();
        h.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        assert!(is_json_content_type(&h));
        h.insert(
            header::CONTENT_TYPE,
            "application/json; charset=utf-8".parse().unwrap(),
        );
        assert!(is_json_content_type(&h));
        h.insert(
            header::CONTENT_TYPE,
            "APPLICATION/JSON".parse().unwrap(),
        );
        assert!(is_json_content_type(&h));
    }

    #[test]
    fn is_json_content_type_rejects_other_types() {
        let mut h = axum::http::HeaderMap::new();
        // Missing
        assert!(!is_json_content_type(&h));
        // Wrong type
        h.insert(header::CONTENT_TYPE, "text/plain".parse().unwrap());
        assert!(!is_json_content_type(&h));
        h.insert(
            header::CONTENT_TYPE,
            "application/x-www-form-urlencoded".parse().unwrap(),
        );
        assert!(!is_json_content_type(&h));
    }
}
//! Custom request extractors.
//!
//! `AppJson` mirrors axum's `Json<T>` but converts every rejection into
//! `ProxyError::BadRequest`, which renders as the project's standard
//! Anthropic error envelope. Without this wrapper, malformed JSON /
//! missing fields come back as `text/plain` "Failed to parse the
//! request body as JSON..." — a format inconsistent with every other
//! error response the proxy emits (auth, unknown model, cooldown, etc.).
//!
//! `AppQuery` mirrors axum's `Query<T>` with the same
//! `ProxyError::BadRequest` mapping for malformed query strings, so
//! the `/admin/usage` endpoint surfaces 400s in the project's error
//! envelope rather than axum's `text/plain` "Failed to deserialize
//! query string".
//!
//! Every rejection path logs a warning. A request that never reaches
//! the handler used to leave no trace at all in the container log —
//! the handler's "request started" line is downstream of the
//! extractor, and axum's own rejection event is emitted at TRACE
//! level under the `axum::rejection` target, which the default
//! `RUST_LOG=llmproxy=info` filter drops. An inbound 4xx was
//! therefore invisible to the operator.
//!
//! See fix-R4 in docs/TEST_ISSUES.md.

use axum::extract::rejection::QueryRejection;
use axum::extract::{FromRequest, FromRequestParts, Query, Request};
use axum::http::header;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::Json;
use serde::de::DeserializeOwned;

use crate::error::ProxyError;

/// The effective inbound request-body limit, in bytes.
///
/// Installed as a request extension by `server::build_router` next to
/// axum's `DefaultBodyLimit` layer, so the extractor can tell the
/// client *which* limit was hit. It is a newtype rather than a bare
/// `usize` so it cannot collide with another extension.
#[derive(Debug, Clone, Copy)]
pub struct InboundBodyLimit(pub usize);

#[derive(Debug, Clone, Copy, Default)]
pub struct AppJson<T>(pub T);

// axum 0.8 replaced `#[async_trait]` with return-position `impl Trait`
// in traits (RPITIT): `FromRequest` / `FromRequestParts` are now plain
// `async fn` in the trait definition, so the impls below no longer need
// (and must not use) the `#[async_trait]` attribute — axum 0.8 dropped
// the `axum::async_trait` re-export entirely.
impl<S, T> FromRequest<S> for AppJson<T>
where
    Json<T>: FromRequest<S, Rejection = axum::extract::rejection::JsonRejection>,
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ProxyError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Read the limit and the client's advertised body size *before*
        // the request is consumed: both are needed for the rejection
        // log line and for the 413 message.
        let limit = req.extensions().get::<InboundBodyLimit>().map(|l| l.0);
        let content_length = content_length_of(req.headers());
        // Content-Type check: Anthropic's API requires
        // `application/json`; reject anything else up front so the
        // client gets a clear message instead of a serde parse error.
        if !is_json_content_type(req.headers()) {
            log_rejection("content_type", content_length, limit);
            return Err(ProxyError::BadRequest(
                "Content-Type must be application/json".to_string(),
            ));
        }
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => {
                let kind = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    "body_too_large"
                } else {
                    "json_invalid"
                };
                log_rejection(kind, content_length, limit);
                Err(map_json_rejection(rejection, limit))
            }
        }
    }
}

/// Query-string wrapper with the same `ProxyError::BadRequest`
/// rejection mapping as `AppJson`. Use `FromRequestParts` (not
/// `FromRequest`) because query parameters live on the request's
/// parts — the body isn't needed and isn't read.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppQuery<T>(pub T);

impl<S, T> FromRequestParts<S> for AppQuery<T>
where
    Query<T>: FromRequestParts<S, Rejection = QueryRejection>,
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ProxyError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(AppQuery(value)),
            Err(rejection) => {
                log_rejection("query_invalid", None, None);
                Err(map_query_rejection(rejection))
            }
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

fn map_json_rejection(
    rejection: axum::extract::rejection::JsonRejection,
    limit: Option<usize>,
) -> ProxyError {
    // axum splits rejections into MissingJsonContentType, JsonDataError,
    // JsonSyntaxError, BytesRejection, etc. The status() / body_text()
    // accessors give us a uniform string and a sensible status code, but
    // we discard the status and always return 400 — the proxy is
    // rejecting the *request*, not relaying an upstream error.
    //
    // The one exception is the oversize body: axum rejects it with
    // 413 PAYLOAD_TOO_LARGE, and flattening that into a generic 400
    // "bad request" is what made a >2 MiB session look like an
    // unexplained client bug. It gets its own variant and its own
    // status.
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ProxyError::PayloadTooLarge { limit_bytes: limit };
    }
    let detail = rejection.body_text();
    ProxyError::BadRequest(format!("invalid request body: {detail}"))
}

fn map_query_rejection(rejection: QueryRejection) -> ProxyError {
    // Mirror the JSON mapping: always 400 + a uniform "invalid query"
    // prefix so admin endpoints surface the project's error envelope.
    let detail = rejection.body_text();
    ProxyError::BadRequest(format!("invalid query string: {detail}"))
}

/// The client's advertised body size, when it sent one.
///
/// Chunked requests carry no `Content-Length`; `None` in the log line
/// then simply means "the client did not say".
fn content_length_of(headers: &axum::http::HeaderMap) -> Option<u64> {
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
}

/// One log line for every request the proxy rejects before it reaches a
/// provider.
///
/// Without this, an inbound 4xx leaves no trace: the handler's
/// "request started" line is downstream of the extractor, and axum's
/// own rejection event sits at TRACE level under the `axum::rejection`
/// target, which `RUST_LOG=llmproxy=info` drops.
fn log_rejection(kind: &str, content_length: Option<u64>, limit: Option<usize>) {
    tracing::warn!(
        kind = kind,
        content_length = content_length,
        limit_bytes = limit,
        "request rejected before reaching a provider"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a JSON request whose body is `body`, with axum's default
    /// body limit set to `limit` bytes and the matching
    /// [`InboundBodyLimit`] extension installed — the same pair
    /// `server::build_router` installs.
    fn json_request(body: &str, limit: usize) -> Request<axum::body::Body> {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .expect("request builds");
        axum::extract::DefaultBodyLimit::max(limit).apply(&mut req);
        req.extensions_mut().insert(InboundBodyLimit(limit));
        req
    }

    #[tokio::test]
    async fn app_json_maps_an_oversize_body_to_payload_too_large() {
        // The body is longer than the 16-byte limit, so axum's `Bytes`
        // rejects it with `LengthLimitError` (413). The extractor must
        // surface that as `PayloadTooLarge`, not as a generic 400.
        let req = json_request(r#"{"padding":"0123456789abcdef"}"#, 16);
        let err = AppJson::<serde_json::Value>::from_request(req, &())
            .await
            .expect_err("oversize body must be rejected");
        match err {
            ProxyError::PayloadTooLarge { limit_bytes } => {
                assert_eq!(limit_bytes, Some(16));
            }
            other => panic!("expected PayloadTooLarge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn app_json_maps_malformed_json_to_bad_request() {
        let req = json_request(r#"{"broken": "#, 4096);
        let err = AppJson::<serde_json::Value>::from_request(req, &())
            .await
            .expect_err("malformed JSON must be rejected");
        match &err {
            ProxyError::BadRequest(msg) => {
                assert!(msg.contains("invalid request body"), "got {msg}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn app_json_rejects_a_missing_content_type_before_reading_the_body() {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(axum::body::Body::from("{}"))
            .expect("request builds");
        axum::extract::DefaultBodyLimit::max(4096).apply(&mut req);
        let err = AppJson::<serde_json::Value>::from_request(req, &())
            .await
            .expect_err("missing content type must be rejected");
        match &err {
            ProxyError::BadRequest(msg) => {
                assert!(msg.contains("application/json"), "got {msg}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn app_json_accepts_a_body_within_the_limit() {
        let req = json_request(r#"{"ok":true}"#, 4096);
        let value = AppJson::<serde_json::Value>::from_request(req, &())
            .await
            .expect("a small body must parse");
        assert_eq!(value.0["ok"], serde_json::json!(true));
    }

    #[test]
    fn content_length_of_reads_the_advertised_size() {
        let mut headers = axum::http::HeaderMap::new();
        assert_eq!(content_length_of(&headers), None);
        headers.insert(header::CONTENT_LENGTH, "42".parse().unwrap());
        assert_eq!(content_length_of(&headers), Some(42));
        // A non-numeric value is not a size we can report.
        headers.insert(header::CONTENT_LENGTH, "not-a-number".parse().unwrap());
        assert_eq!(content_length_of(&headers), None);
    }

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
        h.insert(header::CONTENT_TYPE, "APPLICATION/JSON".parse().unwrap());
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
        // Non-ASCII bytes make HeaderValue::to_str() fail. The function
        // should treat that as "not JSON" rather than panic.
        h.insert(
            header::CONTENT_TYPE,
            axum::http::HeaderValue::from_bytes(b"application/json\xff").unwrap(),
        );
        assert!(!is_json_content_type(&h));
    }

    #[tokio::test]
    async fn app_query_extracts_valid_query_string() {
        use axum::extract::FromRequestParts;
        use axum::http::Request;
        use serde::Deserialize;

        #[derive(Deserialize, PartialEq, Debug)]
        struct Q {
            n: u32,
        }
        let req = Request::builder().uri("/?n=42").body(()).unwrap();
        let (mut parts, _body) = req.into_parts();
        let got = AppQuery::<Q>::from_request_parts(&mut parts, &())
            .await
            .expect("valid query parses");
        assert_eq!(got.0, Q { n: 42 });
    }

    #[tokio::test]
    async fn app_query_maps_malformed_query_to_bad_request() {
        use axum::extract::FromRequestParts;
        use axum::http::Request;
        use serde::Deserialize;

        #[derive(Deserialize, Debug)]
        #[allow(dead_code)]
        struct Q {
            n: u32,
        }
        // `n=abc` cannot parse into u32 → must surface as
        // ProxyError::BadRequest, not axum's default text/plain
        // rejection.
        let req = Request::builder().uri("/?n=abc").body(()).unwrap();
        let (mut parts, _body) = req.into_parts();
        let err = AppQuery::<Q>::from_request_parts(&mut parts, &())
            .await
            .expect_err("malformed query must reject");
        assert!(
            matches!(err, ProxyError::BadRequest(_)),
            "expected BadRequest, got {err:?}"
        );
    }
}

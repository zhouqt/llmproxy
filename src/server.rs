//! Axum server: routes for /v1/messages, /v1/models, /health, /v1/messages/count_tokens.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{middleware, Router as AxumRouter};
use bytes::Bytes;
use futures_util::Stream;

use crate::anthropic::{MessagesRequest, MessagesResponse};
use crate::error::{ProxyError, Result};
use crate::providers::ProviderOutput;
use crate::state::AppState;

pub fn build_router(state: AppState) -> AxumRouter {
    let api = AxumRouter::new()
        .route("/v1/messages", post(messages_handler))
        .route("/v1/messages/count_tokens", post(count_tokens_handler))
        .route("/v1/models", get(list_models_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            crate::auth::require_auth,
        ));

    AxumRouter::new()
        .route("/health", get(health_handler))
        .merge(api)
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn messages_handler(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response> {
    let model_cfg = state
        .router
        .find_model(&req.model)
        .ok_or_else(|| ProxyError::BadRequest(format!("unknown model: {}", req.model)))?
        .clone();

    if req.stream {
        let (_provider, output, attempts) = state.router.stream(&model_cfg, &req).await?;
        return Ok(stream_response(output, attempts));
    }

    let (output, attempts) = state.router.complete(&model_cfg, &req).await?;
    let ProviderOutput::Json(value) = output else {
        return Err(ProxyError::Internal(
            "non-streaming provider returned a stream".into(),
        ));
    };

    let mut resp: MessagesResponse = serde_json::from_value(value)?;
    resp.model = req.model.clone();

    let mut headers = HeaderMap::new();
    if !attempts.is_empty() {
        if let Ok(v) = format_attempts(&attempts).parse() {
            headers.insert("x-llmproxy-failed-providers", v);
        }
    }

    Ok((StatusCode::OK, headers, Json(resp)).into_response())
}

fn format_attempts(attempts: &[crate::router::RouteAttempt]) -> String {
    attempts
        .iter()
        .map(|a| format!("{}:{}", a.provider, a.status))
        .collect::<Vec<_>>()
        .join(",")
}

fn stream_response(output: ProviderOutput, attempts: Vec<crate::router::RouteAttempt>) -> Response {
    let ProviderOutput::Stream(stream) = output else {
        return ProxyError::Internal("expected stream output".into()).into_response();
    };

    let inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> =
        Box::into_pin(stream);
    let mapped = MappedStream { inner, done: false };
    let body = Body::from_stream(mapped);

    let mut resp = Response::new(body);
    let h = resp.headers_mut();
    h.insert(
        "content-type",
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    h.insert("cache-control", "no-cache".parse().unwrap());
    h.insert("x-accel-buffering", "no".parse().unwrap());
    if !attempts.is_empty() {
        if let Ok(v) = format_attempts(&attempts).parse() {
            h.insert("x-llmproxy-failed-providers", v);
        }
    }
    resp
}

/// Adapter: wraps a `Result<Bytes, ProxyError>` stream as a
/// `Result<Bytes, std::io::Error>` stream for axum's body. Logs and
/// terminates on upstream errors.
pub struct MappedStream {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>>,
    done: bool,
}

impl Stream for MappedStream {
    type Item = std::result::Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                tracing::error!("upstream stream error: {e}");
                self.done = true;
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                self.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

async fn count_tokens_handler(
    State(_state): State<AppState>,
    Json(_req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let s = serde_json::to_string(&_req).unwrap_or_default();
    let tokens = ((s.len() as f32) / 4.0).ceil() as u32;
    Json(serde_json::json!({ "input_tokens": tokens }))
}

async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<_> = state
        .config
        .models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "llmproxy",
            })
        })
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use std::pin::Pin;

    fn make_stream(
        items: Vec<std::result::Result<Bytes, ProxyError>>,
    ) -> Pin<Box<dyn Stream<Item = std::result::Result<Bytes, ProxyError>> + Send>> {
        Box::pin(stream::iter(items))
    }

    fn fresh_mapped() -> MappedStream {
        MappedStream {
            inner: make_stream(vec![]),
            done: false,
        }
    }

    /// Single shared panic message for all `assert_matches!`-style helpers.
    /// Keeping the message in one helper means each test call site is free
    /// of its own missed panic-string line.
    fn expect_poll_none(poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>) {
        assert!(matches!(poll, std::task::Poll::Ready(None)), "expected Ready(None)");
    }

    fn expect_poll_pending(poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>) {
        assert!(matches!(poll, std::task::Poll::Pending), "expected Pending");
    }

    fn assert_poll_ready_some_ok(
        poll: std::task::Poll<Option<std::result::Result<Bytes, std::io::Error>>>,
        label: &str,
    ) -> Bytes {
        match poll {
            std::task::Poll::Ready(Some(Ok(b))) => b,
            other => panic!("{label}: expected Ready(Some(Ok)), got {other:?}"),
        }
    }

    #[test]
    fn mapped_stream_returns_none_when_already_done() {
        // Once `done` is set, poll_next must short-circuit to Ready(None)
        // without touching the inner stream at all.
        let mut s = MappedStream {
            inner: make_stream(vec![Err(ProxyError::Internal("unused".into()))]),
            done: true,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_none(poll);
    }

    #[test]
    fn mapped_stream_propagates_pending_from_inner() {
        // When the inner stream returns Poll::Pending, the wrapper must
        // also return Poll::Pending (and must NOT mark itself done).
        let mut s = MappedStream {
            inner: Box::pin(stream::pending::<std::result::Result<Bytes, ProxyError>>()),
            done: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        expect_poll_pending(poll);
        assert!(!s.done, "Pending must not flip done=true");
    }

    #[tokio::test]
    async fn mapped_stream_terminates_on_inner_error() {
        // An upstream error should be swallowed (logged, not propagated as
        // a body error) and the stream should then end.
        let mut s = MappedStream {
            inner: make_stream(vec![Err(ProxyError::Internal("boom".into()))]),
            done: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(poll, std::task::Poll::Ready(None)));
        assert!(s.done);
    }

    #[tokio::test]
    async fn mapped_stream_emits_bytes_then_terminates() {
        let mut s = MappedStream {
            inner: make_stream(vec![
                Ok(Bytes::from_static(b"event: foo\n\n")),
                Ok(Bytes::from_static(b"event: bar\n\n")),
            ]),
            done: false,
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);

        let b1 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "first poll",
        );
        assert_eq!(&b1[..], b"event: foo\n\n");

        let b2 = assert_poll_ready_some_ok(
            Pin::new(&mut s).poll_next(&mut cx),
            "second poll",
        );
        assert_eq!(&b2[..], b"event: bar\n\n");

        let p3 = Pin::new(&mut s).poll_next(&mut cx);
        assert!(matches!(p3, Poll::Ready(None)));
        assert!(s.done);
    }

    #[test]
    fn fresh_mapped_helper_is_not_done() {
        let s = fresh_mapped();
        assert!(!s.done);
    }
}

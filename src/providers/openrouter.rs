//! OpenRouter provider.
//!
//! OpenRouter supports both OpenAI- and Anthropic-style endpoints.
//! When `api_format=anthropic` we forward the Anthropic request body as-is
//! (with a model rewrite) and stream the response. When `api_format=openai`
//! we delegate to the same conversion logic as the generic OpenAI provider.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::Stream;
use serde_json::{json, Value};

use crate::anthropic::{MessagesRequest, StreamEvent};
use crate::config::ApiFormat;
use crate::error::{ProxyError, Result};
use crate::providers::{openai_compat, Provider, ProviderOutput};

pub struct OpenRouterProvider {
    name: String,
    api_key: String,
    api_base: String,
    api_format: ApiFormat,
    http: reqwest::Client,
    /// Inner provider used for the OpenAI-format path.
    inner_openai: openai_compat::OpenAiCompatProvider,
}

impl OpenRouterProvider {
    pub fn new(
        name: String,
        api_key: String,
        api_base: String,
        api_format: ApiFormat,
        http: reqwest::Client,
    ) -> Result<Self> {
        let api_base = api_base.trim_end_matches('/').to_string();
        let inner_openai = openai_compat::OpenAiCompatProvider::new(
            format!("{name}_inner"),
            api_base.clone(),
            api_key.clone(),
            HashMap::new(),
            http.clone(),
        )?;
        Ok(Self {
            name,
            api_key,
            api_base,
            api_format,
            http,
            inner_openai,
        })
    }

    fn anthropic_url(&self) -> String {
        // OpenRouter's Anthropic-compatible endpoint is at /api/v1/messages
        // under the same api_base root.
        // Strip any trailing "/v1" from api_base, then append /v1/messages.
        let stripped = self.api_base.trim_end_matches("/v1");
        format!("{}/v1/messages", stripped)
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn api_format(&self) -> ApiFormat {
        self.api_format
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        match self.api_format {
            ApiFormat::Openai => self.inner_openai.complete(req, model_rewrite).await,
            ApiFormat::Anthropic => forward_anthropic_complete(&self.http, &self.api_key, &self.anthropic_url(), req, model_rewrite).await,
        }
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        match self.api_format {
            ApiFormat::Openai => self.inner_openai.stream(req, model_rewrite).await,
            ApiFormat::Anthropic => forward_anthropic_stream(&self.http, &self.api_key, &self.anthropic_url(), req, model_rewrite).await,
        }
    }
}

async fn forward_anthropic_complete(
    http: &reqwest::Client,
    api_key: &str,
    url: &str,
    req: &MessagesRequest,
    model_rewrite: &HashMap<String, String>,
) -> Result<ProviderOutput> {
    let body = build_anthropic_body(req, model_rewrite, false)?;
    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(ProxyError::Upstream {
            status: status.as_u16(),
            body: text,
        });
    }
    let val: Value = serde_json::from_str(&text)?;
    Ok(ProviderOutput::Json(val))
}

fn forward_anthropic_stream<'a>(
    http: &'a reqwest::Client,
    api_key: &'a str,
    url: &'a str,
    req: &'a MessagesRequest,
    model_rewrite: &'a HashMap<String, String>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ProviderOutput>> + Send + 'a>> {
    Box::pin(async move {
        let body = build_anthropic_body(req, model_rewrite, true)?;
        let resp = http
            .post(url)
            .bearer_auth(api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let stream = resp.bytes_stream();
        Ok(ProviderOutput::Stream(Box::new(PassthroughSse { inner: stream })))
    })
}

fn build_anthropic_body(
    req: &MessagesRequest,
    model_rewrite: &HashMap<String, String>,
    stream: bool,
) -> Result<Value> {
    let mut body = serde_json::to_value(req)?;
    let model = model_rewrite
        .get(&req.model)
        .cloned()
        .unwrap_or_else(|| req.model.clone());
    body["model"] = json!(model);
    body["stream"] = json!(stream);
    Ok(body)
}

/// Pass-through SSE stream (provider already speaks Anthropic SSE).
pub struct PassthroughSse<S> {
    inner: S,
}

impl<S> Stream for PassthroughSse<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(ProxyError::Http(e)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

// Allow unused import to be useful for downstream expansion.
#[allow(dead_code)]
fn _event_marker(_e: &StreamEvent) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;
    use futures_util::StreamExt;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 32,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    #[test]
    fn builds_anthropic_body_with_rewrite_and_stream_flag() {
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "upstream-model".to_string());

        let body = build_anthropic_body(&request(false), &rewrite, true).unwrap();

        assert_eq!(body["model"], "upstream-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[tokio::test]
    async fn anthropic_complete_forwards_request_and_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/messages"))
            .and(header("authorization", "Bearer router-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_partial_json(json!({
                "model": "rewritten-model",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_upstream",
                "type": "message",
                "content": [{"type": "text", "text": "world"}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenRouterProvider::new(
            "router".to_string(),
            "router-key".to_string(),
            format!("{}/api/v1/", server.uri()),
            ApiFormat::Anthropic,
            reqwest::Client::new(),
        )
        .unwrap();
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-model".to_string());

        let output = provider.complete(&request(false), &rewrite).await.unwrap();

        assert_eq!(provider.name(), "router");
        assert_eq!(provider.api_format(), ApiFormat::Anthropic);
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["id"], "msg_upstream");
            assert_eq!(body["content"][0]["text"], "world");
        });
    }

    #[tokio::test]
    async fn anthropic_stream_passes_sse_through() {
        let server = MockServer::start().await;
        let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("accept", "text/event-stream"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenRouterProvider::new(
            "router".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            ApiFormat::Anthropic,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .stream(&request(true), &HashMap::new())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut bytes = Vec::new();
            while let Some(item) = output.next().await {
                bytes.extend_from_slice(&item.unwrap());
            }
            assert_eq!(String::from_utf8(bytes).unwrap(), sse);
        });
    }

    #[tokio::test]
    async fn anthropic_paths_preserve_upstream_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("limited"))
            .expect(2)
            .mount(&server)
            .await;
        let provider = OpenRouterProvider::new(
            "router".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            ApiFormat::Anthropic,
            reqwest::Client::new(),
        )
        .unwrap();

        let complete = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("complete should fail");
        let stream = provider
            .stream(&request(true), &HashMap::new())
            .await
            .err()
            .expect("stream should fail");

        assert!(matches!(
            complete,
            ProxyError::Upstream { status: 429, ref body } if body == "limited"
        ));
        assert!(matches!(
            stream,
            ProxyError::Upstream { status: 429, ref body } if body == "limited"
        ));
    }

    #[tokio::test]
    async fn passthrough_sse_surfaces_inner_errors() {
        // Connect to a port that's almost certainly closed; the connect
        // failure surfaces as a reqwest::Error we can feed into the SSE.
        let err = reqwest::Client::new()
            .get("http://127.0.0.1:1")
            .timeout(std::time::Duration::from_millis(50))
            .send()
            .await
            .expect_err("connect to closed port must fail");
        use futures_util::stream;
        let inner = stream::iter(vec![Err::<Bytes, _>(err)]);
        let mut sse = PassthroughSse { inner };
        let item = sse.next().await.expect("one item");
        assert!(matches!(item, Err(ProxyError::Http(_))));
    }

    #[tokio::test]
    async fn passthrough_sse_returns_none_when_inner_ends() {
        // An empty inner stream should produce no items, then end.
        use futures_util::stream;
        let mut sse = PassthroughSse {
            inner: stream::empty::<reqwest::Result<Bytes>>(),
        };
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn passthrough_sse_propagates_pending_from_inner() {
        // When the inner stream returns `Poll::Pending`, the wrapper must
        // propagate it (instead of fabricating a Ready). Cover the
        // `Poll::Pending => Poll::Pending` arm of the match.
        use futures_util::stream;
        let mut sse = PassthroughSse {
            inner: stream::pending::<reqwest::Result<Bytes>>(),
        };
        // poll_next must not deadlock; using tokio's noop waker it should
        // return Pending. We assert via std::task::Poll directly to avoid
        // the auto-waker logic in `next().await`.
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = std::pin::Pin::new(&mut sse).poll_next(&mut cx);
        assert!(
            matches!(poll, std::task::Poll::Pending),
            "PassthroughSse should propagate Poll::Pending"
        );
    }

    #[tokio::test]
    async fn openai_format_delegates_to_compat_provider() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({"stream": false})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "delegated"},
                    "finish_reason": "stop"
                }],
                "usage": null
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenRouterProvider::new(
            "router".to_string(),
            "key".to_string(),
            server.uri(),
            ApiFormat::Openai,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();

        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["content"][0]["text"], "delegated");
        });
    }
}

//! Generic native-Anthropic-Messages provider.
//!
//! Sends the Anthropic request body verbatim to `{api_base}/messages`
//! with the `anthropic-version: 2023-06-01` header and streams the
//! response unchanged (or returns the JSON response verbatim for the
//! non-streaming path). No request/response translation occurs.
//!
//! Used for OpenRouter's native `/v1/messages` endpoint and any other
//! gateway that exposes the Anthropic Messages API without conversion.
//! For OpenAI Chat Completions-style upstreams use the `openai_compat`
//! provider type instead.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::Stream;
use serde_json::{json, Value};

use crate::anthropic::MessagesRequest;
use crate::error::{ProxyError, Result};
use crate::providers::{Provider, ProviderOutput};

pub struct AnthropicProvider {
    name: String,
    api_key: String,
    api_base: String,
    model_rewrite: HashMap<String, String>,
    http: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(
        name: String,
        api_key: String,
        api_base: String,
        model_rewrite: HashMap<String, String>,
        http: reqwest::Client,
    ) -> Result<Self> {
        let api_base = api_base.trim_end_matches('/').to_string();
        Ok(Self {
            name,
            api_key,
            api_base,
            model_rewrite,
            http,
        })
    }

    fn messages_url(&self) -> String {
        // api_base defaults to e.g. https://openrouter.ai/api/v1.
        // Strip the trailing /v1 and POST to /v1/messages.
        let stripped = self.api_base.trim_end_matches("/v1");
        format!("{}/v1/messages", stripped)
    }

    fn merged_rewrite<'a>(
        &'a self,
        runtime: &'a HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(runtime.iter().map(|(k, v)| (k.clone(), v.clone())));
        merged
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let body = build_body(req, &self.merged_rewrite(model_rewrite), false)?;
        let resp = self
            .http
            .post(self.messages_url())
            .bearer_auth(&self.api_key)
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

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let url = self.messages_url();
        let api_key = self.api_key.clone();
        let body = build_body(req, &self.merged_rewrite(model_rewrite), true)?;
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&api_key)
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
    }
}

fn build_body(
    req: &MessagesRequest,
    merged_rewrite: &HashMap<String, String>,
    stream: bool,
) -> Result<Value> {
    let mut body = serde_json::to_value(req)?;
    let model = merged_rewrite
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

    fn empty_rewrite() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn build_body_rewrites_model_and_sets_stream_flag() {
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "upstream-model".to_string());

        let body = build_body(&request(false), &rewrite, true).unwrap();

        assert_eq!(body["model"], "upstream-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn build_body_falls_back_to_original_model_when_unmapped() {
        let body = build_body(&request(false), &empty_rewrite(), false).unwrap();

        assert_eq!(body["model"], "claude-model");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn merged_rewrite_combines_provider_and_runtime_maps() {
        // Constructor rewrite table takes effect even when runtime map
        // names a different model — both layers compose.
        let mut configured = HashMap::new();
        configured.insert("claude-model".to_string(), "configured-model".to_string());

        let provider = AnthropicProvider::new(
            "r".to_string(),
            "k".to_string(),
            "https://example.test/v1".to_string(),
            configured,
            reqwest::Client::new(),
        )
        .unwrap();

        let mut runtime = HashMap::new();
        runtime.insert("claude-model".to_string(), "runtime-model".to_string());

        let merged = provider.merged_rewrite(&runtime);
        // Runtime overrides the configured entry.
        assert_eq!(merged.get("claude-model").unwrap(), "runtime-model");
    }

    #[tokio::test]
    async fn complete_forwards_request_and_response() {
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
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-model".to_string());
        let provider = AnthropicProvider::new(
            "router".to_string(),
            "router-key".to_string(),
            format!("{}/api/v1/", server.uri()),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider.complete(&request(false), &empty_rewrite()).await.unwrap();

        assert_eq!(provider.name(), "router");
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["id"], "msg_upstream");
            assert_eq!(body["content"][0]["text"], "world");
        });
    }

    #[tokio::test]
    async fn stream_passes_sse_through() {
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
        let provider = AnthropicProvider::new(
            "router".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider.stream(&request(true), &empty_rewrite()).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut bytes = Vec::new();
            while let Some(item) = output.next().await {
                bytes.extend_from_slice(&item.unwrap());
            }
            assert_eq!(String::from_utf8(bytes).unwrap(), sse);
        });
    }

    #[tokio::test]
    async fn preserve_upstream_errors_on_complete_and_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("limited"))
            .expect(2)
            .mount(&server)
            .await;
        let provider = AnthropicProvider::new(
            "router".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let complete = provider
            .complete(&request(false), &empty_rewrite())
            .await
            .err()
            .expect("complete should fail");
        let stream = provider
            .stream(&request(true), &empty_rewrite())
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
        use futures_util::stream;
        let mut sse = PassthroughSse {
            inner: stream::empty::<reqwest::Result<Bytes>>(),
        };
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn passthrough_sse_propagates_pending_from_inner() {
        use futures_util::stream;
        let mut sse = PassthroughSse {
            inner: stream::pending::<reqwest::Result<Bytes>>(),
        };
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = std::pin::Pin::new(&mut sse).poll_next(&mut cx);
        assert!(
            matches!(poll, std::task::Poll::Pending),
            "PassthroughSse should propagate Poll::Pending"
        );
    }
}

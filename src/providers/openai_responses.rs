//! Generic OpenAI Responses API provider.
//!
//! Used by any backend that exposes `POST /v1/responses` (OpenAI's
//! GPT-5.x Responses endpoint, direct OpenAI reverse proxies, etc.).
//! Always converts Anthropic requests to the Responses input[] shape
//! and converts the response back; for Chat-Completions-style
//! backends use `openai_compat` instead.
//!
//! Mirrors `openai_compat.rs` 1:1 in surface area (`complete`/`stream`,
//! `model_rewrite`, `use_proxy`, etc.) but the wire format and SSE
//! event vocabulary are the Responses ones defined in
//! `src/responses.rs` and `src/conversion/responses*.rs`.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use futures_util::Stream;

use crate::anthropic::{MessagesRequest, StreamEvent};
use crate::conversion::responses::{
    anthropic_to_responses_request, make_message_id, responses_to_anthropic_response,
};
use crate::conversion::responses_stream::ResponsesStreamTranslator;
use crate::error::{ProxyError, Result};
use crate::providers::{Provider, ProviderOutput};
use crate::responses::ResponsesResponse;

pub struct OpenaiResponsesProvider {
    name: String,
    api_base: String,
    api_key: String,
    model_rewrite: HashMap<String, String>,
    http: reqwest::Client,
}

impl OpenaiResponsesProvider {
    pub fn new(
        name: String,
        api_base: String,
        api_key: String,
        model_rewrite: HashMap<String, String>,
        http: reqwest::Client,
    ) -> Result<Self> {
        let api_base = api_base.trim_end_matches('/').to_string();
        Ok(Self {
            name,
            api_base,
            api_key,
            model_rewrite,
            http,
        })
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.api_base)
    }
}

#[async_trait]
impl Provider for OpenaiResponsesProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_serve_model(&self, model: &str) -> bool {
        // Same semantics as OpenAiCompatProvider: empty rewrite = pass
        // through verbatim (the upstream has its own model catalog);
        // non-empty = explicit allow-list. See fix-R11.
        self.model_rewrite.is_empty() || self.model_rewrite.contains_key(model)
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(model_rewrite.iter().map(|(k, v)| (k.clone(), v.clone())));

        let mut responses_req = anthropic_to_responses_request(req, &merged);
        responses_req.stream = false;

        let resp = self
            .http
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&responses_req)
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: ResponsesResponse = serde_json::from_str(&body)?;
        let msg_id = make_message_id();
        let anthropic_resp = responses_to_anthropic_response(&parsed, &req.model, &msg_id)?;
        Ok(ProviderOutput::Json(serde_json::to_value(anthropic_resp)?))
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(model_rewrite.iter().map(|(k, v)| (k.clone(), v.clone())));

        let mut responses_req = anthropic_to_responses_request(req, &merged);
        responses_req.stream = true;

        let resp = self
            .http
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&responses_req)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await?;
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body,
            });
        }

        let byte_stream = resp.bytes_stream();
        let sse = ResponsesSseToAnthropic::new(byte_stream, &req.model);
        Ok(ProviderOutput::Stream(Box::new(sse)))
    }
}

/// Adapter: reads a Responses API SSE byte stream and emits an
/// Anthropic SSE byte stream.
pub struct ResponsesSseToAnthropic<S> {
    inner: S,
    translator: Option<ResponsesStreamTranslator>,
    pending: BytesMut,
    finished: bool,
    output_buffer: VecDeque<Bytes>,
}

impl<S> ResponsesSseToAnthropic<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    pub fn new(inner: S, model: &str) -> Self {
        Self {
            inner,
            translator: Some(ResponsesStreamTranslator::new(make_message_id(), model)),
            pending: BytesMut::new(),
            finished: false,
            output_buffer: VecDeque::new(),
        }
    }

    fn encode(ev: &StreamEvent) -> Bytes {
        let payload = serde_json::to_string(ev).unwrap_or_default();
        Bytes::from(format!("event: {}\ndata: {}\n\n", event_name(ev), payload))
    }

    fn process_lines(&mut self) {
        loop {
            let Some(pos) = self.pending.iter().position(|&b| b == b'\n') else {
                break;
            };
            let line_bytes = self.pending.split_to(pos);
            self.pending.advance(1); // consume the '\n'
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches('\r');
            let Some(rest) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = rest.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                if let Some(mut t) = self.translator.take() {
                    for ev in t.finalize() {
                        self.output_buffer.push_back(Self::encode(&ev));
                    }
                }
                self.finished = true;
                return;
            }
            match serde_json::from_str::<crate::responses::ResponsesStreamEvent>(payload) {
                Ok(ev) => {
                    if let Some(t) = self.translator.as_mut() {
                        for out in t.push_event(&ev) {
                            self.output_buffer.push_back(Self::encode(&out));
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("skipping malformed Responses SSE line: {} ({e})", payload);
                }
            }
        }
    }
}

impl<S> Stream for ResponsesSseToAnthropic<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(b) = self.output_buffer.pop_front() {
            return Poll::Ready(Some(Ok(b)));
        }
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.pending.extend_from_slice(&chunk);
                    self.process_lines();
                    if let Some(b) = self.output_buffer.pop_front() {
                        return Poll::Ready(Some(Ok(b)));
                    }
                    if self.finished {
                        return Poll::Ready(None);
                    }
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(ProxyError::Http(e))));
                }
                Poll::Ready(None) => {
                    if let Some(mut t) = self.translator.take() {
                        for ev in t.finalize() {
                            self.output_buffer.push_back(Self::encode(&ev));
                        }
                    }
                    self.finished = true;
                    if let Some(b) = self.output_buffer.pop_front() {
                        return Poll::Ready(Some(Ok(b)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn event_name(ev: &StreamEvent) -> &'static str {
    match ev {
        StreamEvent::MessageStart { .. } => "message_start",
        StreamEvent::ContentBlockStart { .. } => "content_block_start",
        StreamEvent::Ping => "ping",
        StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        StreamEvent::MessageDelta { .. } => "message_delta",
        StreamEvent::MessageStop => "message_stop",
        StreamEvent::Error { .. } => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;
    use futures_util::{stream, StreamExt};
    use serde_json::json;
    use serde_json::Value;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request(streaming: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 64,
            "stream": streaming,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    fn responses_body() -> Value {
        json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "world"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        })
    }

    #[tokio::test]
    async fn complete_posts_to_responses_endpoint_and_converts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(json!({
                "model": "runtime-model",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;

        let mut configured_rewrite = HashMap::new();
        configured_rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "configured-model".to_string(),
        );
        let provider = OpenaiResponsesProvider::new(
            "test".to_string(),
            format!("{}/v1/", server.uri()),
            "test-key".to_string(),
            configured_rewrite,
            reqwest::Client::new(),
        )
        .unwrap();
        let mut runtime_rewrite = HashMap::new();
        runtime_rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "runtime-model".to_string(),
        );

        let output = provider.complete(&request(false), &runtime_rewrite).await.unwrap();

        assert_eq!(provider.name(), "test");
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["type"], "message");
            assert_eq!(body["model"], "claude-sonnet-4-20250514");
            assert_eq!(body["content"][0]["text"], "world");
            assert_eq!(body["stop_reason"], "end_turn");
            assert_eq!(body["usage"]["input_tokens"], 3);
            assert_eq!(body["usage"]["output_tokens"], 2);
        });
    }

    #[tokio::test]
    async fn complete_preserves_upstream_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("request should fail");

        expect_variant!(error, ProxyError::Upstream { status, body } => {
            assert_eq!(status, 429);
            assert_eq!(body, "rate limited");
        });
    }

    #[tokio::test]
    async fn stream_converts_responses_sse() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":4,\"output_tokens\":1,\"total_tokens\":5}}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({
                "model": "stream-model",
                "stream": true
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();
        let mut rewrite = HashMap::new();
        rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "stream-model".to_string(),
        );

        let output = provider.stream(&request(true), &rewrite).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut output) => {
            let mut encoded = String::new();
            while let Some(item) = output.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }

            assert!(encoded.contains("event: message_start"));
            assert!(encoded.contains("event: content_block_delta"));
            assert!(encoded.contains("\"text\":\"hello\""));
            assert!(encoded.contains("\"input_tokens\":4"));
            assert!(encoded.contains("event: message_stop"));
        });
    }

    #[tokio::test]
    async fn stream_preserves_upstream_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "test".to_string(),
            server.uri(),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .stream(&request(true), &HashMap::new())
            .await
            .err()
            .expect("request should fail");

        assert!(matches!(
            error,
            ProxyError::Upstream { status: 503, ref body } if body == "unavailable"
        ));
    }

    #[tokio::test]
    async fn adapter_handles_fragmented_lines_and_eof() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(
                b"data: not-json\ndata: {\"type\":\"response.created\",\"response\":",
            )),
            Ok(Bytes::from_static(
                b"{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            )),
        ];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        assert!(encoded.contains("message_start"));
    }

    #[tokio::test]
    async fn adapter_finalizes_on_eof_when_no_done_marker() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
        ))];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        assert!(encoded.contains("message_start"));
        assert!(encoded.contains("message_stop"));
    }

    fn provider_with_rewrite(rewrite: HashMap<String, String>) -> OpenaiResponsesProvider {
        OpenaiResponsesProvider::new(
            "p".to_string(),
            "https://x/v1/".to_string(),
            "k".to_string(),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap()
    }

    #[test]
    fn can_serve_model_accepts_anything_when_rewrite_is_empty() {
        let p = provider_with_rewrite(HashMap::new());
        assert!(p.can_serve_model("any-random-name"));
        assert!(p.can_serve_model("gpt-5"));
    }

    #[test]
    fn can_serve_model_matches_keys_when_rewrite_is_non_empty() {
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-sonnet-4.6".to_string(), "gpt-5".to_string());
        let p = provider_with_rewrite(rewrite);
        assert!(p.can_serve_model("claude-sonnet-4.6"));
        assert!(!p.can_serve_model("gpt-5"));
        assert!(!p.can_serve_model(""));
    }
}
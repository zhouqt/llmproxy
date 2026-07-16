//! Generic OpenAI-compatible provider.
//!
//! Used by OpenRouter (when api_format=openai), DeepSeek, MiniMax, and any
//! other provider that exposes an OpenAI-style /chat/completions endpoint.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use futures_util::Stream;
use serde_json::Value;

use crate::anthropic::{MessagesRequest, StreamEvent};
use crate::config::ApiFormat;
use crate::conversion::{anthropic_to_openai_request, openai_to_anthropic_response};
use crate::error::{ProxyError, Result};
use crate::providers::{Provider, ProviderOutput};

pub struct OpenAiCompatProvider {
    name: String,
    api_base: String,
    api_key: String,
    model_rewrite: HashMap<String, String>,
    http: reqwest::Client,
}

impl OpenAiCompatProvider {
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

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.api_base)
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn api_format(&self) -> ApiFormat {
        ApiFormat::Openai
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(model_rewrite.iter().map(|(k, v)| (k.clone(), v.clone())));

        let mut openai_req = anthropic_to_openai_request(req, &merged);
        openai_req.stream = false;
        openai_req.stream_options = None;

        let resp = self
            .http
            .post(self.chat_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&openai_req)
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

        let parsed: Value = serde_json::from_str(&body)?;
        let chat: crate::openai::ChatResponse = serde_json::from_value(parsed)?;
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let anthropic_resp = openai_to_anthropic_response(&chat, &req.model, &msg_id)?;
        Ok(ProviderOutput::Json(serde_json::to_value(anthropic_resp)?))
    }

    async fn stream(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let mut merged = self.model_rewrite.clone();
        merged.extend(model_rewrite.iter().map(|(k, v)| (k.clone(), v.clone())));

        let mut openai_req = anthropic_to_openai_request(req, &merged);
        openai_req.stream = true;

        let resp = self
            .http
            .post(self.chat_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&openai_req)
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
        let sse = OpenAiSseToAnthropic::new(byte_stream, &req.model);
        Ok(ProviderOutput::Stream(Box::new(sse)))
    }
}

/// Adapter: reads an OpenAI SSE byte stream and emits Anthropic SSE byte stream.
pub struct OpenAiSseToAnthropic<S> {
    inner: S,
    translator: Option<crate::conversion::stream::StreamTranslator>,
    pending: BytesMut,
    finished: bool,
    output_buffer: VecDeque<Bytes>,
}

impl<S> OpenAiSseToAnthropic<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    pub fn new(inner: S, model: &str) -> Self {
        Self {
            inner,
            translator: Some(crate::conversion::stream::StreamTranslator::new(
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                model,
            )),
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
        // Drain complete `\n`-terminated lines and feed to translator.
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
            match serde_json::from_str::<crate::openai::ChatChunk>(payload) {
                Ok(c) => {
                    if let Some(t) = self.translator.as_mut() {
                        for ev in t.push_chunk(&c) {
                            self.output_buffer.push_back(Self::encode(&ev));
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("skipping malformed SSE line: {} ({e})", payload);
                }
            }
        }
    }
}

impl<S> Stream for OpenAiSseToAnthropic<S>
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
                    // No events yet — keep reading.
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(ProxyError::Http(e))));
                }
                Poll::Ready(None) => {
                    // EOF: close translator if not already.
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

    fn chat_response() -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "world"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        })
    }

    #[tokio::test]
    async fn complete_sends_rewritten_request_and_converts_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(json!({
                "model": "runtime-model",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
            .expect(1)
            .mount(&server)
            .await;

        let mut configured_rewrite = HashMap::new();
        configured_rewrite.insert(
            "claude-sonnet-4-20250514".to_string(),
            "configured-model".to_string(),
        );
        let provider = OpenAiCompatProvider::new(
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
        assert_eq!(provider.api_format(), ApiFormat::Openai);
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
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
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
    async fn stream_converts_openai_sse() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "model": "stream-model",
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
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
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;
        let provider = OpenAiCompatProvider::new(
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
    async fn adapter_handles_fragmented_lines_malformed_data_and_eof() {
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(b"event: ignored\ndata: not-json\ndata: {\"id\":\"c\",")),
            Ok(Bytes::from_static(b"\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n")),
        ];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("message_start"));
        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("message_delta"));
        assert!(encoded.contains("message_stop"));
    }

    #[tokio::test]
    async fn adapter_skips_empty_data_lines_and_comment_lines() {
        // Empty `data:` payloads and `:` comment lines must be ignored without
        // producing any events.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: \n: this is a comment\ndata:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("\"text\":\"hi\""));
        // No event should be emitted for empty payloads or comments.
        assert!(!encoded.contains("data: \n\n"));
    }

    #[tokio::test]
    async fn adapter_surfaces_inner_stream_errors() {
        // A reqwest error from the inner stream is wrapped in ProxyError::Http.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(
                b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            )),
            Err(reqwest::Error::from(
                reqwest::Client::new()
                    .get("http://[invalid")
                    .build()
                    .unwrap_err(),
            )),
        ];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");

        let mut items = Vec::new();
        while let Some(item) = adapter.next().await {
            items.push(item);
        }

        // First event payload decodes successfully, then the error surfaces.
        assert!(items[0].is_ok());
        let err = items
            .iter()
            .find(|i| i.is_err())
            .expect("expected a stream error");
        assert!(matches!(err, Err(ProxyError::Http(_))));
    }

    #[tokio::test]
    async fn adapter_finalizes_on_eof_when_no_done_marker() {
        // If the upstream stream closes without sending `data: [DONE]`,
        // the adapter still flushes the translator's pending events.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final\"},\"finish_reason\":null}]}\n\n",
        ))];
        let mut adapter = OpenAiSseToAnthropic::new(stream::iter(chunks), "model");
        let mut encoded = String::new();

        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        assert!(encoded.contains("text_delta"));
        assert!(encoded.contains("\"text\":\"final\""));
        assert!(encoded.contains("message_stop"));
    }

    #[test]
    fn event_names_cover_all_variants() {
        let message = crate::anthropic::MessagesResponse {
            id: "m".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: "model".to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: Default::default(),
        };
        let events = [
            StreamEvent::MessageStart { message },
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: crate::anthropic::ResponseBlock::Text {
                    text: String::new(),
                },
            },
            StreamEvent::Ping,
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: crate::anthropic::BlockDelta::TextDelta {
                    text: "x".to_string(),
                },
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageDelta {
                delta: crate::anthropic::MessageDeltaPayload {
                    stop_reason: None,
                    stop_sequence: None,
                    usage: None,
                },
            },
            StreamEvent::MessageStop,
            StreamEvent::Error { error: json!({}) },
        ];

        assert_eq!(
            events.iter().map(event_name).collect::<Vec<_>>(),
            vec![
                "message_start",
                "content_block_start",
                "ping",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
                "error",
            ]
        );
    }

    #[tokio::test]
    async fn adapter_returns_pending_when_inner_is_pending() {
        // When the inner stream returns Pending, poll_next must also return
        // Pending without flipping finished. Using a noop waker makes this
        // deterministic.
        use futures_util::stream;
        let mut adapter = OpenAiSseToAnthropic::new(
            stream::pending::<reqwest::Result<Bytes>>(),
            "model",
        );
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = std::task::Context::from_waker(waker);
        let poll = std::pin::Pin::new(&mut adapter).poll_next(&mut cx);
        assert!(
            matches!(poll, std::task::Poll::Pending),
            "expected Poll::Pending from a pending inner stream"
        );
    }
}

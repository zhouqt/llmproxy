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
    use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

    /// Wire-level "field X must NOT be present in the JSON request
    /// body" matcher. wiremock's `body_partial_json` only checks
    /// presence; we need this complement to verify that the proxy
    /// doesn't pollute requests with `prompt_cache_key` /
    /// `prompt_cache_retention` when the Anthropic client didn't ask
    /// for caching.
    struct JsonFieldAbsent(&'static str);

    impl Match for JsonFieldAbsent {
        fn matches(&self, request: &Request) -> bool {
            let body: serde_json::Value = match serde_json::from_slice(&request.body) {
                Ok(v) => v,
                Err(_) => return false,
            };
            body.get(self.0).is_none()
        }
    }

    fn cache_request_with(cache_type: &str, user_id: Option<&str>) -> MessagesRequest {
        let mut v = json!({
            "model": "claude-sonnet-4.6",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "long prefix", "cache_control": {"type": cache_type}},
                    {"type": "text", "text": "actual question"}
                ]
            }]
        });
        if let Some(uid) = user_id {
            v["metadata"] = json!({"user_id": uid});
        }
        serde_json::from_value(v).unwrap()
    }

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

    #[test]
    fn trailing_slash_on_api_base_is_stripped() {
        // Operators frequently write api_base as `https://x/v1/` —
        // the constructor must normalize so we don't end up with
        // `/v1//responses` on the wire (some servers reject double
        // slashes, some accept them but it's noise in logs).
        let p = OpenaiResponsesProvider::new(
            "p".to_string(),
            "https://example.test/v1/".to_string(),
            "k".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(p.responses_url(), "https://example.test/v1/responses");
    }

    #[test]
    fn multiple_trailing_slashes_are_stripped() {
        let p = OpenaiResponsesProvider::new(
            "p".to_string(),
            "https://example.test/v1///".to_string(),
            "k".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();
        // trim_end_matches('/') removes all trailing slashes; we
        // then re-append exactly one. The result has no //.
        assert_eq!(p.responses_url(), "https://example.test/v1/responses");
    }

    #[tokio::test]
    async fn complete_merges_runtime_rewrite_with_configured() {
        // Mirrors OpenAiCompatProvider's behavior: when both
        // configured and runtime rewrite maps contain the same key,
        // runtime wins. The merged table is what gets applied to
        // req.model before sending upstream.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({"model": "runtime-rewrite-wins"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;
        let mut configured = HashMap::new();
        configured.insert(
            "claude-sonnet-4-20250514".to_string(),
            "configured-loses".to_string(),
        );
        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            server.uri(),
            "k".to_string(),
            configured,
            reqwest::Client::new(),
        )
        .unwrap();
        let mut runtime = HashMap::new();
        runtime.insert(
            "claude-sonnet-4-20250514".to_string(),
            "runtime-rewrite-wins".to_string(),
        );

        let output = provider.complete(&request(false), &runtime).await.unwrap();
        expect_variant!(output, ProviderOutput::Json(_body) => {});
    }

    #[tokio::test]
    async fn complete_sends_request_as_json_with_expected_fields() {
        // Verify the actual wire body shape: model (unchanged when
        // no rewrite applies), stream=false, input[] with role+content,
        // max_output_tokens, and bearer auth header. This is what an
        // OpenAI-side log would see.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer secret-key-xyz"))
            .and(body_partial_json(json!({
                "model": "claude-sonnet-4-20250514",
                "stream": false,
                "max_output_tokens": 64,
                "input": [{"role": "user", "content": "hello"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "secret-key-xyz".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider.complete(&request(false), &HashMap::new()).await.unwrap();
    }

    #[tokio::test]
    async fn complete_emits_instructions_field_when_request_has_system() {
        // Anthropic's `system` must reach the wire as Responses'
        // `instructions`. body_partial_json with `instructions`
        // verifies presence + value. The model name on the wire
        // matches req.model since no rewrite applies.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_partial_json(json!({
                "model": "claude-sonnet-4-20250514",
                "instructions": "be terse"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            server.uri(),
            "k".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();
        let mut req = request(false);
        req.system = Some(crate::anthropic::SystemPrompt::Text("be terse".into()));

        let _ = provider.complete(&req, &HashMap::new()).await.unwrap();
    }

    #[tokio::test]
    async fn complete_propagates_malformed_response_body() {
        // A 200 with a body that doesn't decode as ResponsesResponse
        // (e.g. truncated JSON) must surface as ProxyError::Json so
        // the caller sees the parse failure, not a misleading 500.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            server.uri(),
            "k".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let error = provider
            .complete(&request(false), &HashMap::new())
            .await
            .err()
            .expect("malformed body should fail");
        assert!(matches!(error, ProxyError::Json(_)), "got: {error:?}");
    }

    #[tokio::test]
    async fn stream_skips_malformed_sse_chunks_without_terminating() {
        // Streaming upstreams often interleave heartbeats or partial
        // data, so the adapter silently skips SSE lines that fail to
        // parse rather than terminating the whole stream. This
        // verifies that: a chunk of "garbage" followed by a valid
        // event must still produce the valid event downstream.
        let server = MockServer::start().await;
        let sse = concat!(
            "data: not-json\n\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;
        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            server.uri(),
            "k".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider.stream(&request(true), &HashMap::new()).await.unwrap();
        expect_variant!(output, ProviderOutput::Stream(mut stream) => {
            let mut encoded = String::new();
            while let Some(item) = stream.next().await {
                encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            // The malformed chunk was skipped; the valid event still
            // produced its MessageStart.
            assert!(encoded.contains("event: message_start"));
        });
    }

    #[tokio::test]
    async fn complete_emits_prompt_cache_key_and_in_memory_when_cache_control_ephemeral() {
        // Anthropic cache_control.ephemeral + metadata.user_id → wire
        // must carry prompt_cache_key=user_id and
        // prompt_cache_retention=in_memory. The whole point: the
        // client-side cache hint has to reach the upstream so the
        // upstream actually treats the tokens as cached (and bills
        // them at the discounted rate).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer key"))
            .and(body_partial_json(json!({
                "prompt_cache_key": "u-42",
                "prompt_cache_retention": "in_memory"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral", Some("u-42")), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_emits_24h_retention_when_cache_control_ephemeral_1h() {
        // Anthropic cache_control.ephemeral_1h → retention=24h on the
        // wire. This is the longest tier both APIs offer (Anthropic
        // bills 1h tier; OpenAI's nearest equivalent is 24h).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(body_partial_json(json!({
                "prompt_cache_key": "u-9",
                "prompt_cache_retention": "24h"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral_1h", Some("u-9")), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_omits_prompt_cache_key_when_cache_control_without_user_id() {
        // cache_control present + no metadata.user_id → wire must
        // emit retention (client wants caching) but NOT
        // prompt_cache_key (we have no namespace to scope to; emitting
        // an empty key would lump unrelated requests into one cache
        // bucket).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(body_partial_json(json!({
                "prompt_cache_retention": "in_memory"
            })))
            .and(JsonFieldAbsent("prompt_cache_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&cache_request_with("ephemeral", None), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn complete_omits_cache_fields_when_request_has_no_cache_control() {
        // Default Anthropic request (no cache_control, but
        // metadata.user_id may or may not be present) → wire body
        // must NOT carry prompt_cache_key / prompt_cache_retention.
        // The proxy must not pollute requests with cache hints when
        // the client didn't ask — that would change billing semantics
        // (caching is opt-in on the client side).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(JsonFieldAbsent("prompt_cache_key"))
            .and(JsonFieldAbsent("prompt_cache_retention"))
            .respond_with(ResponseTemplate::new(200).set_body_json(responses_body()))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenaiResponsesProvider::new(
            "p".to_string(),
            format!("{}/v1/", server.uri()),
            "key".to_string(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _ = provider
            .complete(&request(false), &HashMap::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn adapter_emits_clean_message_start_and_stop_on_minimal_sse() {
        // End-to-end shape of a minimal successful Responses SSE
        // exchange: created → text delta → completed → [DONE]. The
        // adapter must emit message_start, at least one content
        // delta, and message_stop with no other framing noise.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n\
              data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n\
              data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}\n\n\
              data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n\
              data: [DONE]\n\n",
        ))];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "m");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        // Order: message_start must come before content_block_delta,
        // which must come before message_stop.
        let pos_start = encoded.find("event: message_start").expect("message_start");
        let pos_delta = encoded.find("event: content_block_delta").expect("content_block_delta");
        let pos_stop = encoded.find("event: message_stop").expect("message_stop");
        assert!(pos_start < pos_delta);
        assert!(pos_delta < pos_stop);
        assert!(encoded.contains("\"text\":\"hi\""));
    }

    #[tokio::test]
    async fn adapter_skips_empty_data_lines_without_dropping_subsequent_events() {
        // OpenAI SSE often has trailing `data:` lines (empty payload)
        // interleaved with valid events. The adapter must skip them
        // silently rather than crashing on the empty payload or
        // dropping the next event. See openai_responses.rs:188.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data:\n\n\
              data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n\
              data:   \n\n\
              data: [DONE]\n\n",
        ))];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "m");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        // Empty lines were skipped; the one real event still came
        // through.
        assert!(encoded.contains("event: message_start"));
        assert!(encoded.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn adapter_emits_finalized_message_stop_when_done_marker_seen_after_completed() {
        // When the upstream sends [DONE] after response.completed, the
        // adapter's process_lines() takes the [DONE] branch and calls
        // translator.finalize() — but at that point the completed event
        // already added message_stop to the buffer. We verify that the
        // [DONE] branch doesn't emit a *second* message_stop and that
        // the stream terminates cleanly. See openai_responses.rs:190-198.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n\
              data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"completed\",\"output\":[],\"usage\":{}}}\n\n\
              data: [DONE]\n\n",
        ))];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "m");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        let stop_count = encoded.matches("event: message_stop").count();
        assert_eq!(stop_count, 1, "exactly one message_stop expected, got {stop_count}: {encoded}");
    }

    #[tokio::test]
    async fn adapter_emits_error_event_on_inner_stream_failure() {
        // If the upstream HTTP body stream returns Err mid-stream
        // (connection drop, etc.), the adapter must surface it as a
        // ProxyError so the server layer can wrap it in the
        // `event: error` envelope (see server::MappedStream). The
        // outer Stream::poll_next drives the adapter — we feed a
        // stream that yields one chunk then an Err. See
        // openai_responses.rs:242-244.
        let s = stream::iter(vec![
            Ok(Bytes::from_static(
                b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
            )),
            Err(reqwest::Error::from(
                reqwest::Client::new()
                    .get("http://[invalid")
                    .build()
                    .unwrap_err(),
            )),
        ]);
        let mut adapter = ResponsesSseToAnthropic::new(s, "m");
        let mut items: Vec<Result<Bytes>> = Vec::new();
        while let Some(item) = adapter.next().await {
            items.push(item);
        }
        // The first item should be the message_start chunk; the
        // second should be the propagated error.
        assert!(items.len() >= 2, "expected >=2 items, got {}", items.len());
        assert!(items[0].is_ok(), "first chunk should be the message_start bytes");
        assert!(items[1].is_err(), "second item should be the propagated Http error, got Ok");
        match &items[1] {
            Err(crate::error::ProxyError::Http(_)) => {}
            other => panic!("expected ProxyError::Http, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn adapter_finalizes_on_eof_after_chunk_with_no_done_marker() {
        // End-of-stream without a [DONE] marker. The poll_next
        // Poll::Ready(None) branch (lines 246-251) must call
        // translator.finalize() so the client still sees
        // message_stop — otherwise the body just truncates. We feed
        // one valid event then a chunk that ends the stream.
        let chunks: Vec<reqwest::Result<Bytes>> = vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"object\":\"response\",\"created_at\":0,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[],\"usage\":{}}}\n\n",
        ))];
        let mut adapter = ResponsesSseToAnthropic::new(stream::iter(chunks), "m");
        let mut encoded = String::new();
        while let Some(item) = adapter.next().await {
            encoded.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        assert!(encoded.contains("event: message_start"));
        assert!(
            encoded.contains("event: message_stop"),
            "EOF must finalize the translator and emit message_stop, got: {encoded}"
        );
    }

    #[test]
    fn event_name_emits_ping_for_ping_variant() {
        // The Responses translator never emits a Ping, but the
        // event_name() match must still handle it for completeness
        // (if a future Responses variant maps to an Anthropic
        // Ping, this arm is the one that takes it). Locking the
        // encoding here so a future rename of the wire-format name
        // is caught.
        assert_eq!(event_name(&StreamEvent::Ping), "ping");
    }

    #[test]
    fn event_name_emits_error_for_error_variant() {
        // Same rationale as ping: future-proofing the
        // event_name() match for the Error variant. The Responses
        // translator surfaces upstream errors via the chunk's HTTP
        // status (handled before SSE starts), so this arm is
        // currently only reachable when an Error event is injected
        // by the translator itself — but the wire-name contract is
        // still part of the public surface.
        use crate::anthropic::StreamEvent;
        let ev = StreamEvent::Error {
            error: serde_json::json!({"type": "upstream_error", "message": "boom"}),
        };
        assert_eq!(event_name(&ev), "error");
    }
}
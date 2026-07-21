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

    /// Build a friendly Anthropic-shaped error body when an upstream rejects
    /// the request because it doesn't support thinking/reasoning mode. The
    /// proxy does NOT silently strip thinking and retry — the user has
    /// explicitly requested thinking, and any fallback in the chain must
    /// also support it. Surface the mismatch as an actionable error so the
    /// operator knows to reconfigure the fallback chain rather than seeing
    /// silently degraded responses.
    ///
    /// `client_model` is the model name the proxy received from the client
    /// (the incoming request's `model` field); `upstream_model` is the
    /// model name that was actually sent upstream after the provider's
    /// `model_rewrite` map was applied. Reporting the upstream model lets
    /// the operator see exactly which catalog entry the upstream rejected,
    /// which is what they need to look up in their upstream dashboard.
    fn thinking_not_supported_error(
        &self,
        client_model: &str,
        upstream_model: &str,
        upstream_body: &str,
    ) -> ProxyError {
        // Truncate upstream body for the human-readable message so the
        // envelope stays compact; the full body is preserved in
        // `upstream_body` for debugging.
        let snippet: String = upstream_body.chars().take(200).collect();
        let friendly = json!({
            "type": "error",
            "error": {
                "type": "thinking_not_supported",
                "message": format!(
                    "Provider '{}' does not support thinking/reasoning mode for model '{}'. \
                     The primary model in this fallback chain uses thinking; \
                     all fallbacks must support it too. \
                     Reconfigure the fallback chain in config.yaml: either \
                     remove this provider from chains whose primary uses \
                     thinking, or replace it with a thinking-capable provider \
                     (or a model variant that supports thinking on this upstream). \
                     Upstream response: {}",
                    self.name, upstream_model, snippet
                ),
                "provider": self.name,
                "client_model": client_model,
                "upstream_model": upstream_model,
                "upstream_body": upstream_body,
            }
        });
        ProxyError::Upstream {
            status: 400,
            body: friendly.to_string(),
        }
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
        let merged = self.merged_rewrite(model_rewrite);
        let body = build_body(req, &merged, false)?;
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
            if status.as_u16() == 400 && has_thinking_error(&text) {
                return Err(self.thinking_not_supported_error(
                    &req.model,
                    &merged.get(&req.model).cloned().unwrap_or_else(|| req.model.clone()),
                    &text,
                ));
            }
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
        let merged = self.merged_rewrite(model_rewrite);
        let body = build_body(req, &merged, true)?;
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
            if status.as_u16() == 400 && has_thinking_error(&text) {
                return Err(self.thinking_not_supported_error(
                    &req.model,
                    &merged.get(&req.model).cloned().unwrap_or_else(|| req.model.clone()),
                    &text,
                ));
            }
            return Err(ProxyError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }
        let stream = resp.bytes_stream();
        Ok(ProviderOutput::Stream(Box::new(PassthroughSse { inner: stream })))
    }
}

/// Check if the upstream response body indicates thinking-mode is not supported.
/// Some Anthropic-compatible endpoints return a 400 with a message like:
///   `The content[].thinking in the thinking mode must be passed back to the API.`
fn has_thinking_error(body: &str) -> bool {
    body.contains("content[].thinking") && body.contains("must be passed back")
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
    fn has_thinking_error_matches_anthropic_error_message() {
        assert!(has_thinking_error(
            r#"{"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}"#
        ));
        // Non-thinking errors must NOT match.
        assert!(!has_thinking_error(r#"{"error":{"message":"model not found"}}"#));
        assert!(!has_thinking_error("rate limited"));
        assert!(!has_thinking_error(""));
    }

    #[tokio::test]
    async fn complete_passes_through_every_field_unmodified() {
        // Regression guard: every documented request field — thinking
        // blocks with signatures, cache_control on text/tools/images,
        // redacted_thinking blocks, server tool blocks, document blocks,
        // top-level output_config/service_tier/container/inference_geo/
        // user_profile_id — must reach upstream byte-identical except
        // for `model` (which `build_body` rewrites) and `stream`.
        //
        // wiremock's `body_partial_json` matcher asserts the body is a
        // superset of the expected JSON, so any field dropped by the
        // proxy would cause the mock to NOT match.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("authorization", "Bearer router-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_partial_json(json!({
                "model": "rewritten-model",
                "stream": false,
                "max_tokens": 1024,
                "system": [
                    {"type": "text", "text": "you are claude",
                     "cache_control": {"type": "ephemeral", "ttl": "5m"}}
                ],
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 40,
                "stop_sequences": ["STOP"],
                "tools": [
                    {"name": "get_weather", "description": "weather",
                     "input_schema": {"type": "object"}}
                ],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "metadata": {"user_id": "u-1"},
                "thinking": {"type": "enabled", "budget_tokens": 4000, "display": "summarized"},
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
                "container": "container_x",
                "inference_geo": "us",
                "service_tier": "auto",
                "output_config": {"effort": "high"},
                "user_profile_id": "profile-1",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "hello",
                         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                         "cache_control": {"type": "ephemeral"}}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "thinking…", "signature": "sig-1"},
                        {"type": "redacted_thinking", "data": "encrypted-blob"},
                        {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1},
                         "cache_control": {"type": "ephemeral"}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok",
                         "is_error": false}
                    ]}
                ],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_upstream",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "world"}],
                "model": "rewritten-model",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": null,
                    "cache_read_input_tokens": null
                }
            })))
            .expect(1)
            .mount(&server)
            .await;
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-model".to_string());
        let provider = AnthropicProvider::new(
            "router".to_string(),
            "router-key".to_string(),
            format!("{}/", server.uri()),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap();

        // Build a fully-populated request that exercises every field
        // path through serde → build_body → wiremock.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "you are claude",
                 "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hello",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "thinking…", "signature": "sig-1"},
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1},
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok",
                     "is_error": false}
                ]}
            ],
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 40,
            "stop_sequences": ["STOP"],
            "stream": false,
            "tools": [
                {"name": "get_weather", "description": "weather",
                 "input_schema": {"type": "object"}}
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "metadata": {"user_id": "u-1"},
            "thinking": {"type": "enabled", "budget_tokens": 4000, "display": "summarized"},
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "container": "container_x",
            "inference_geo": "us",
            "service_tier": "auto",
            "output_config": {"effort": "high"},
            "user_profile_id": "profile-1"
        }))
        .unwrap();

        let output = provider.complete(&req, &empty_rewrite()).await.unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            // upstream body forwarded verbatim — no transformation.
            assert_eq!(body["id"], "msg_upstream");
            assert_eq!(body["content"][0]["text"], "world");
        });
    }

    #[tokio::test]
    async fn complete_preserves_thinking_signature_in_response() {
        // Regression guard for the original bug: Anthropic-format
        // responses must include `signature` on thinking blocks so the
        // client can echo them back next turn without the upstream
        // returning "content[].thinking in the thinking mode must be
        // passed back to the API".
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning",
                     "signature": "sig-should-survive"},
                    {"type": "text", "text": "answer"}
                ],
                "model": "rewritten-model",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2}
            })))
            .mount(&server)
            .await;
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-model".to_string());
        let provider = AnthropicProvider::new(
            "router".to_string(),
            "router-key".to_string(),
            format!("{}/", server.uri()),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap();
        let output = provider.complete(&request(false), &empty_rewrite()).await.unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            let thinking = body["content"].as_array().unwrap().iter()
                .find(|b| b["type"] == "thinking")
                .expect("thinking block");
            assert_eq!(thinking["signature"], "sig-should-survive",
                "thinking.signature must roundtrip from upstream response to client");
        });
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

    /// Build a request with thinking enabled, mimicking a client that asks
    /// for extended reasoning. The fallback chain problem only manifests
    /// when the upstream can't honour this request.
    fn thinking_request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn complete_returns_friendly_error_on_thinking_mismatch() {
        // The upstream rejects with the canonical Anthropic thinking
        // error. The proxy must NOT silently strip thinking and retry —
        // it must surface a clear, actionable message telling the
        // operator to reconfigure the fallback chain. The message must
        // include both the provider name and the model name (preferring
        // the upstream-rewritten name so the operator can match it to
        // their upstream dashboard).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "The `content[].thinking` in the thinking mode must be passed back to the API."
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut rewrite = HashMap::new();
        // Operator maps `claude-model` to `upstream-variant` for this provider.
        rewrite.insert("claude-model".to_string(), "upstream-variant".to_string());
        let provider = AnthropicProvider::new(
            "minimax".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &empty_rewrite())
            .await
            .err()
            .expect("thinking-mismatch must surface as Err");

        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream error, got: {other:?}"),
        };
        assert_eq!(status, 400);

        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "thinking_not_supported");

        // Structured fields for programmatic inspection.
        assert_eq!(parsed["error"]["provider"], "minimax");
        assert_eq!(parsed["error"]["client_model"], "claude-model");
        assert_eq!(
            parsed["error"]["upstream_model"], "upstream-variant",
            "upstream_model must reflect the rewrite, not the original"
        );

        let message = parsed["error"]["message"].as_str().unwrap();
        // Provider name appears in the message.
        assert!(
            message.contains("minimax"),
            "message must name the offending provider: {message}"
        );
        // Upstream (rewritten) model name appears — not the client model.
        assert!(
            message.contains("upstream-variant"),
            "message must name the upstream model after rewrite: {message}"
        );
        // Thinking-not-supported phrasing and a reconfigure hint.
        assert!(
            message.contains("does not support thinking"),
            "message must explain the mismatch: {message}"
        );
        assert!(
            message.contains("Reconfigure"),
            "message must tell the operator what to do: {message}"
        );
        // Original upstream body is preserved for debugging.
        assert!(parsed["error"]["upstream_body"].as_str().unwrap().contains("thinking"));
    }

    #[tokio::test]
    async fn complete_friendly_error_uses_client_model_when_no_rewrite() {
        // When there's no model_rewrite mapping, upstream_model should
        // fall back to the original client model name (no fabricated
        // upstream name).
        let server = MockServer::start().await;
        // Plain-text 400 body that still triggers has_thinking_error
        // via the `content[].thinking` and `must be passed back` markers.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                "The `content[].thinking` in the thinking mode must be passed back to the API.",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "no-rewrite".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &empty_rewrite())
            .await
            .err()
            .expect("should fail");

        let body = match err {
            ProxyError::Upstream { body, .. } => body,
            other => panic!("expected Upstream, got: {other:?}"),
        };
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["error"]["client_model"], "claude-model");
        assert_eq!(
            parsed["error"]["upstream_model"], "claude-model",
            "upstream_model must equal client_model when no rewrite applies"
        );
    }

    #[tokio::test]
    async fn complete_passes_through_unrelated_400_not_thinking() {
        // A 400 that does NOT mention thinking must surface to the caller
        // unchanged — the friendly-error path must not fire for unrelated
        // request-shape errors.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "model not found"}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "no-retry".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &empty_rewrite())
            .await
            .err()
            .expect("should fail");

        assert!(matches!(
            err,
            ProxyError::Upstream { status: 400, ref body } if body.contains("model not found")
        ));
    }

    #[tokio::test]
    async fn stream_returns_friendly_error_on_thinking_mismatch() {
        // Streaming twin of complete_returns_friendly_error_on_thinking_mismatch.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut rewrite = HashMap::new();
        rewrite.insert("claude-model".to_string(), "rewritten-claude".to_string());
        let provider = AnthropicProvider::new(
            "minimax".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            rewrite,
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .stream(&thinking_request(true), &empty_rewrite())
            .await
            .err()
            .expect("thinking-mismatch must surface as Err");

        match err {
            ProxyError::Upstream { status, body } => {
                assert_eq!(status, 400);
                let parsed: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(parsed["error"]["type"], "thinking_not_supported");
                assert_eq!(parsed["error"]["provider"], "minimax");
                assert_eq!(parsed["error"]["client_model"], "claude-model");
                assert_eq!(parsed["error"]["upstream_model"], "rewritten-claude");
                let message = parsed["error"]["message"].as_str().unwrap();
                assert!(message.contains("minimax"));
                assert!(message.contains("rewritten-claude"));
            }
            other => panic!("expected Upstream error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn thinking_mismatch_does_not_retry_when_thinking_absent() {
        // Even when the incoming request has no thinking field, an
        // Anthropic-shaped 400 from the upstream is treated as a config
        // mismatch only when the body explicitly mentions thinking.
        // A bare 400 with a different shape passes through.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_string("plain text error"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "p".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_request(false), &empty_rewrite())
            .await
            .err()
            .expect("should fail");
        assert!(matches!(
            err,
            ProxyError::Upstream { status: 400, ref body } if body == "plain text error"
        ));
    }
}

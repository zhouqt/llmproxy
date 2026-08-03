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

use crate::anthropic::{ContentBlock, MessageContent, MessagesRequest};
use crate::error::{ProxyError, Result};
use crate::providers::{Provider, ProviderOutput};

/// Value of the `anthropic-client-platform` header the real Claude Code
/// CLI sends to Anthropic-format gateways (see
/// plans/simulate-claude-code-identity.md). Presenting it lets upstreams
/// like OpenRouter's native `/v1/messages` or MiniMax classify the
/// proxy's traffic as coming from Claude Code instead of an unknown
/// client. Hardcoded rather than configurable because the proxy is always
/// driven by the Claude Code CLI — the value never needs to vary per
/// deployment.
const ANTHROPIC_CLIENT_PLATFORM: &str = "claude_code_cli";

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

    fn models_url(&self) -> String {
        let stripped = self.api_base.trim_end_matches("/v1");
        format!("{}/v1/models", stripped)
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

    async fn list_models(&self) -> Option<Vec<serde_json::Value>> {
        let url = self.models_url();
        let resp = self
            .http
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-client-platform", ANTHROPIC_CLIENT_PLATFORM)
            .header("accept", "application/json")
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            tracing::warn!(
                status = %resp.status(),
                provider = %self.name,
                "list_models returned non-success"
            );
            return None;
        }
        let body: serde_json::Value = resp.json().await.ok()?;
        let data = body.get("data")?.as_array()?;
        Some(
            data.iter()
                .filter_map(|entry| {
                    let id = entry.get("id")?.as_str()?;
                    let display_name = entry
                        .get("display_name")
                        .or_else(|| entry.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(id);
                    let created = entry.get("created").and_then(|v| v.as_i64()).unwrap_or(0);
                    Some(serde_json::json!({
                        "id": id,
                        "object": "model",
                        "created": created,
                        "owned_by": "anthropic",
                        "display_name": display_name,
                    }))
                })
                .collect(),
        )
    }

    async fn complete(
        &self,
        req: &MessagesRequest,
        model_rewrite: &HashMap<String, String>,
    ) -> Result<ProviderOutput> {
        let merged = self.merged_rewrite(model_rewrite);
        let mut body = build_body(req, &merged, false)?;
        let url = self.messages_url();
        let api_key = self.api_key.clone();

        let mut attempt: u32 = 0;
        let max_attempts: u32 = 2;
        loop {
            attempt += 1;
            let resp = self
                .http
                .post(&url)
                .bearer_auth(&api_key)
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .header("anthropic-client-platform", ANTHROPIC_CLIENT_PLATFORM)
                .json(&body)
                .send()
                .await?;
            let status = resp.status();
            let text = resp.text().await?;
            if status.is_success() {
                let val: Value = serde_json::from_str(&text)?;
                return Ok(ProviderOutput::Json(val));
            }

            // Decide whether to strip thinking blocks and retry.
            let status_code = status.as_u16();
            let is_thinking_400 =
                status_code == 400 && has_thinking_error(&text);

            if is_thinking_400
                && attempt < max_attempts
                && should_strip_and_retry(req, &text)
            {
                tracing::warn!(
                    provider = %self.name,
                    client_model = %req.model,
                    attempt = attempt,
                    "stripping thinking blocks from request and retrying once (upstream rejected cross-model thinking signature)"
                );
                // Mutate the OUTGOING body in place:
                //   - drop content blocks of type {thinking, redacted_thinking}
                //     from every assistant/user message;
                //   - skip entire messages whose content array becomes empty;
                //   - remove the top-level thinking param.
                strip_thinking_blocks(&mut body);
                continue;
            }

            tracing::warn!(
                status = status_code,
                provider = %self.name,
                client_model = %req.model,
                attempt = attempt,
                "thinking-strip retry exhausted, surfacing raw upstream error"
            );
            // No retry path applies: surface the raw upstream body verbatim.
            // In particular a thinking-400 with nothing to strip (no
            // assistant-thinking block in the request history) passes
            // through untouched — the friendly `thinking_not_supported`
            // envelope is only produced on the streaming path.
            return Err(ProxyError::Upstream {
                status: status_code,
                body: text,
            });
        }
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
            .header("anthropic-client-platform", ANTHROPIC_CLIENT_PLATFORM)
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            // Streaming keeps the original friendly-error path on purpose:
            // once the SSE byte stream starts flowing, retrying would
            // double-emit content to the client. See Router::stream comment
            // at router.rs:303-307 for the system-wide "no retry once
            // streaming" contract. The AnthropicProvider::stream path
            // therefore DOES NOT apply strip-and-retry; a thinking-style
            // 400 still surfaces as `thinking_not_supported` here.
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

/// Decide whether a 400 with a thinking-style body should trigger the
/// strip-and-retry path. The body alone cannot disambiguate "real
/// protocol violation" from "the upstream's compat layer dislikes the
/// cross-model signature"; the request context decides.
///
/// Both conditions must hold:
///   (a) upstream body matches the canonical thinking-error substring
///       pair (delegated to `has_thinking_error`), AND
///   (b) the incoming request already contains an assistant message
///       whose `content` includes a `Thinking` or `RedactedThinking`
///       block (i.e. there is something concrete to strip) — the gate
///       deliberately covers the same block types `strip_thinking_blocks`
///       removes, so a history that only carries a redacted_thinking
///       block is stripped and retried rather than passed through as a
///       400.
fn should_strip_and_retry(req: &MessagesRequest, body: &str) -> bool {
    if !has_thinking_error(body) {
        return false;
    }
    req.messages.iter().any(|m| {
        m.role == "assistant"
            && matches!(
                &m.content,
                MessageContent::Blocks(bs) if bs.iter().any(|b| matches!(
                    b,
                    ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
                ))
            )
    })
}

/// Strip thinking-related content from the outgoing `body` JSON value
/// in place. After this call:
///   - every Thinking / RedactedThinking block has been dropped;
///   - if a message's `content` array is now empty (or contained only
///     thinking blocks), the message itself is removed;
///   - the top-level `thinking` key is removed entirely (litellm's
///     `data.pop("thinking")` semantics), whether or not it existed.
fn strip_thinking_blocks(body: &mut Value) {
    if let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
        messages.retain_mut(|msg| {
            if let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                blocks.retain(|b| !matches!(
                    b.get("type").and_then(|t| t.as_str()),
                    Some("thinking") | Some("redacted_thinking")
                ));
                !blocks.is_empty() // keep only if there's still content
            } else {
                true
            }
        });
    }
    // Remove the top-level `thinking` param entirely (litellm's
    // `data.pop("thinking")` semantics) rather than nulling it — DeepSeek
    // is a strict validator and a spurious `"thinking": null` on a request
    // that never enabled thinking could itself be rejected.
    if let Some(obj) = body.as_object_mut() {
        obj.remove("thinking");
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
    fn has_thinking_error_matches_anthropic_error_message() {
        assert!(has_thinking_error(
            r#"{"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}"#
        ));
        // DeepSeek's Anthropic-compatible endpoint emits the identical
        // literal when it rejects a cross-model signature.
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
            .and(header("anthropic-client-platform", "claude_code_cli"))
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
    async fn complete_passes_through_web_search_20250305_hosted_tool() {
        // Regression guard: a request whose only tool is a hosted
        // Anthropic server tool (web_search_20250305) must reach the
        // upstream without being 400'd by the extractor.
        //
        // Before the Tool struct fix, input_schema was required and
        // the hosted tool had none -> serde "missing field" -> AppJson
        // returned 400.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("authorization", "Bearer router-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_partial_json(json!({
                "model": "claude-model",
                "stream": false,
                "max_tokens": 1024,
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
                ],
                "messages": [{"role": "user", "content": "hello"}],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_ws",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-model",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "router".to_string(),
            "router-key".to_string(),
            format!("{}/", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]
        }))
        .unwrap();

        let output = provider.complete(&req, &empty_rewrite()).await.unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["id"], "msg_ws");
            assert_eq!(body["content"][0]["text"], "ok");
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
            .and(header("anthropic-client-platform", "claude_code_cli"))
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
            .and(header("anthropic-client-platform", "claude_code_cli"))
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

    /// Build a multi-turn request whose history carries an assistant
    /// thinking block (with a signature). This is the scenario that
    /// triggers strip-and-retry on a thinking 400: the upstream's compat
    /// layer dislikes the cross-model signature and there is concrete
    /// content to strip.
    fn thinking_history_request(stream: bool) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig-1"},
                    {"type": "text", "text": "previous answer"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn should_strip_and_retry_matches_thinking_and_redacted_thinking() {
        // P3: the strip gate must cover exactly the block types
        // `strip_thinking_blocks` removes (Thinking + RedactedThinking).
        // Before the P3 fix this test fails on the redacted_thinking
        // case — the gate only matched Thinking, so a history carrying
        // only a redacted_thinking block was passed through as a 400
        // instead of being stripped and retried.
        let thinking_400 = r#"{"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}"#;

        // Assistant history carries a Thinking block -> gate fires.
        let req_with_thinking: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}
                ]}
            ]
        }))
        .unwrap();
        assert!(should_strip_and_retry(&req_with_thinking, thinking_400));

        // Assistant history carries a RedactedThinking block -> gate fires.
        let req_with_redacted: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "encrypted-blob"}
                ]}
            ]
        }))
        .unwrap();
        assert!(should_strip_and_retry(&req_with_redacted, thinking_400));

        // Only a USER message carries a thinking block -> nothing to strip.
        let req_user_thinking: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}
                ]}
            ]
        }))
        .unwrap();
        assert!(!should_strip_and_retry(&req_user_thinking, thinking_400));

        // Assistant message is plain text -> nothing to strip.
        let req_plain: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "plain text answer"}
            ]
        }))
        .unwrap();
        assert!(!should_strip_and_retry(&req_plain, thinking_400));
    }

    #[test]
    fn strip_thinking_blocks_removes_thinking_and_keeps_text() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig-1"},
                    {"type": "text", "text": "answer"}
                ]}
            ]
        });
        strip_thinking_blocks(&mut body);
        let messages = body["messages"].as_array().unwrap();
        // 2 messages survive — the assistant message is kept, not dropped.
        assert_eq!(messages.len(), 2);
        // user message untouched.
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hi");
        // assistant message keeps exactly one block: the text block.
        let assistant_blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(assistant_blocks.len(), 1);
        assert_eq!(assistant_blocks[0]["type"], "text");
        assert_eq!(assistant_blocks[0]["text"], "answer");
        // the thinking block is gone from the whole body, not relabelled.
        assert!(!body.to_string().contains("thinking"));
    }

    #[test]
    fn strip_thinking_blocks_removes_redacted_thinking() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                    {"type": "text", "text": "answer"}
                ]}
            ]
        });
        strip_thinking_blocks(&mut body);
        let assistant_blocks = body["messages"][1]["content"].as_array().unwrap();
        // only the text block survives.
        assert_eq!(assistant_blocks.len(), 1);
        assert_eq!(assistant_blocks[0]["type"], "text");
        assert_eq!(assistant_blocks[0]["text"], "answer");
        // the redacted_thinking block is gone, not relabelled.
        for b in assistant_blocks {
            assert_ne!(b["type"], "redacted_thinking");
        }
        assert!(!body.to_string().contains("redacted_thinking"));
    }

    #[test]
    fn strip_thinking_blocks_deletes_message_that_becomes_empty() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        });
        strip_thinking_blocks(&mut body);
        let messages = body["messages"].as_array().unwrap();
        // the assistant message had ONLY a thinking block, so it is dropped
        // entirely rather than left as an empty-content message.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "first");
        // remaining messages keep their original relative order.
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "continue");
    }

    #[test]
    fn strip_thinking_blocks_removes_top_level_thinking_key() {
        let mut body = json!({
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(body.as_object().unwrap().contains_key("thinking"));
        strip_thinking_blocks(&mut body);
        // key removed ENTIRELY, not nulled — `get` would be Some(null) if
        // the code only set it to null, so the contains_key check is the
        // distinguishing assertion.
        assert!(!body.as_object().unwrap().contains_key("thinking"));
        assert!(body.get("thinking").is_none());
        // messages survive untouched.
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn strip_thinking_blocks_handles_missing_messages_key() {
        let mut body = json!({
            "thinking": {"type": "enabled", "budget_tokens": 2000}
        });
        strip_thinking_blocks(&mut body);
        // no messages key -> nothing to iterate, must not panic, and the
        // top-level thinking key is still removed.
        assert!(!body.as_object().unwrap().contains_key("thinking"));
        assert!(body.get("messages").is_none());
    }

    #[test]
    fn strip_thinking_blocks_handles_string_content() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plain text answer"}
            ]
        });
        strip_thinking_blocks(&mut body);
        let messages = body["messages"].as_array().unwrap();
        // string content is not a block array: left untouched, message kept.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "plain text answer");
    }

    #[test]
    fn strip_thinking_blocks_handles_empty_object() {
        let mut body = json!({});
        strip_thinking_blocks(&mut body);
        assert!(body.as_object().unwrap().is_empty());
    }

    #[tokio::test]
    async fn complete_thinking_error_strips_and_retries_with_thinking_history() {
        // Multi-turn scenario: the client echoes its previous thinking
        // block (with signature) back in history. DeepSeek's Anthropic
        // compat layer rejects the cross-model signature with the same
        // literal thinking 400. The proxy must strip the thinking blocks
        // from the outgoing body and retry once; the second attempt
        // succeeds. wiremock expect(2) pins the retry to exactly one.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(json!({
                        "id": "msg_retried",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "model": "claude-model",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "deepseek".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let output = provider
            .complete(&thinking_history_request(false), &empty_rewrite())
            .await
            .unwrap();
        expect_variant!(output, ProviderOutput::Json(body) => {
            assert_eq!(body["id"], "msg_retried");
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // The retried request body must be stripped of all thinking and
        // redacted_thinking blocks across every message — and it must
        // NOT drop non-thinking content: all 3 messages survive and the
        // assistant message (index 1) keeps its text block. These pinned
        // assertions make a "strip everything / strip nothing" false
        // green impossible.
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert_eq!(sent["messages"].as_array().unwrap().len(), 3);
        let assistant = &sent["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        let text_block = assistant["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "text")
            .expect("assistant message must keep its text block after strip");
        assert_eq!(text_block["text"], "previous answer");
        for msg in sent["messages"].as_array().unwrap() {
            if let Some(blocks) = msg["content"].as_array() {
                for b in blocks {
                    let t = b["type"].as_str();
                    assert_ne!(t, Some("thinking"));
                    assert_ne!(t, Some("redacted_thinking"));
                }
            }
        }
    }

    #[tokio::test]
    async fn complete_thinking_error_does_not_strip_when_no_thinking_history() {
        // A request whose history contains no assistant thinking block has
        // nothing to strip, so the strip-and-retry gate does not fire and
        // the upstream is hit exactly once (wiremock expect(1)). The
        // upstream's canonical thinking 400 is passed through VERBATIM as
        // `Upstream { status: 400 }` — the complete() path no longer
        // translates it into the friendly `thinking_not_supported`
        // envelope; that friendly path survives only on stream().
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "no-history".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&request(false), &empty_rewrite())
            .await
            .err()
            .expect("should fail");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        // Raw upstream body, untouched: the original thinking literal is
        // present and none of the friendly-envelope fields are.
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["error"]["message"],
            "The `content[].thinking` in the thinking mode must be passed back to the API."
        );
        assert_ne!(parsed["error"]["type"], "thinking_not_supported");
        assert!(
            parsed["error"].get("provider").is_none(),
            "raw passthrough must not carry the friendly provider field: {body}"
        );
        assert!(
            parsed["error"].get("client_model").is_none(),
            "raw passthrough must not carry the friendly client_model field: {body}"
        );
        assert!(
            parsed["error"].get("upstream_model").is_none(),
            "raw passthrough must not carry the friendly upstream_model field: {body}"
        );
        // No strip-and-retry fired: the upstream was called exactly once
        // (enforced by wiremock expect(1)).
    }

    #[tokio::test]
    async fn complete_unrelated_400_with_thinking_history_does_not_strip() {
        // The strip-and-retry decision needs BOTH gates: the upstream body
        // must be a thinking error AND the history must carry an assistant
        // thinking block. Here the history DOES carry one
        // (`thinking_history_request`) but the 400 body is an unrelated
        // "model not found" error. The body gate (`has_thinking_error`)
        // must reject it on its own: no strip, no retry, exactly one
        // upstream hit, raw body passthrough.
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
            "unrelated-400".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_history_request(false), &empty_rewrite())
            .await
            .err()
            .expect("should fail");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        // Raw upstream body, verbatim — no friendly envelope fields.
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["error"]["message"], "model not found");
        assert_ne!(parsed["error"]["type"], "thinking_not_supported");
        // wiremock expect(1) pins: no strip-and-retry fired despite the
        // thinking history being present.
    }

    #[tokio::test]
    async fn complete_thinking_error_strip_removes_top_level_thinking_param() {
        // The top-level `thinking` param (enabled + budget) must be removed
        // from the retried request (litellm's `data.pop("thinking")`
        // semantics), in addition to the message blocks being stripped —
        // otherwise the upstream still sees a thinking-mode request it
        // rejects. The assertion checks the key is ABSENT rather than null:
        // nulling would still be caught by `is_null()` only vacuously, and
        // a spurious `"thinking": null` could itself be rejected by a
        // strict validator like DeepSeek.
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(200).set_body_json(json!({
                        "id": "msg_ok2",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "model": "claude-model",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "deepseek".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let _out = provider
            .complete(&thinking_history_request(false), &empty_rewrite())
            .await
            .expect("strip then 200 must succeed");

        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert!(
            sent.as_object().unwrap().get("thinking").is_none(),
            "top-level thinking must be removed, not nulled"
        );
    }

    #[tokio::test]
    async fn complete_thinking_error_strips_only_once_when_second_attempt_still_fails() {
        // Both attempts return the thinking 400. The proxy must strip and
        // retry exactly once (wiremock expect(2)) and then surface the
        // SECOND upstream body verbatim — not the first, not a friendly
        // envelope — because we've already applied our one remedy.
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        let captured: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
                    }))
                } else {
                    *captured_for_responder.lock().unwrap() = Some(
                        serde_json::from_slice(&req.body).unwrap_or_else(|_| json!({}))
                    );
                    ResponseTemplate::new(400).set_body_json(json!({
                        "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API.", "attempt": 2}
                    }))
                }
            })
            .expect(2)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "deepseek".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .complete(&thinking_history_request(false), &empty_rewrite())
            .await
            .err()
            .expect("second 400 must surface as Err");
        let (status, body) = match err {
            ProxyError::Upstream { status, body } => (status, body),
            other => panic!("expected Upstream, got: {other:?}"),
        };
        assert_eq!(status, 400);
        let parsed: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({}));
        assert_eq!(
            parsed["error"]["attempt"], 2,
            "must surface the SECOND attempt's raw body, got: {body}"
        );
        // The retried request was actually stripped before being re-sent.
        // Assert the top-level thinking param is REMOVED (not nulled) and
        // that stripping only dropped thinking blocks — all 3 messages
        // survive and the assistant message (index 1) keeps its text.
        let sent = captured.lock().unwrap().clone().expect("second body captured");
        assert!(
            sent.as_object().unwrap().get("thinking").is_none(),
            "top-level thinking must be removed, not nulled"
        );
        assert_eq!(sent["messages"].as_array().unwrap().len(), 3);
        let assistant = &sent["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        let text_block = assistant["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "text")
            .expect("assistant message must keep its text block after strip");
        assert_eq!(text_block["text"], "previous answer");
        for msg in sent["messages"].as_array().unwrap() {
            if let Some(blocks) = msg["content"].as_array() {
                for b in blocks {
                    let t = b["type"].as_str();
                    assert_ne!(t, Some("thinking"));
                    assert_ne!(t, Some("redacted_thinking"));
                }
            }
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn stream_does_not_apply_strip_and_retry_on_thinking_error() {
        // Duality of the complete() strip-and-retry: even when the request
        // carries an assistant-thinking history block (the exact scenario
        // complete() would strip and retry), the streaming path must NOT
        // retry — wiremock expect(1) pins the single request. Once the
        // SSE byte stream starts flowing, retrying would double-emit
        // content to the client (see router.rs:303-307 contract), so a
        // thinking-style 400 still surfaces as the friendly
        // `thinking_not_supported` envelope here.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "minimax".to_string(),
            "key".to_string(),
            format!("{}/v1", server.uri()),
            empty_rewrite(),
            reqwest::Client::new(),
        )
        .unwrap();

        let err = provider
            .stream(&thinking_history_request(true), &empty_rewrite())
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
                assert_eq!(parsed["error"]["upstream_model"], "claude-model");
            }
            other => panic!("expected Upstream error, got: {other:?}"),
        }
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
        // Streaming keeps the friendly envelope on a thinking 400 (unlike
        // complete(), which passes the raw body through when there is no
        // assistant-thinking history to strip). See the stream() comment
        // about the "no retry once streaming" contract.
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

    #[tokio::test]
    async fn list_models_returns_normalized_entries_on_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-client-platform", "claude_code_cli"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "model-a", "display_name": "Model A", "created": 1000},
                    {"id": "model-b", "name": "Model B", "created": 2000},
                    {"id": "model-c"}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "test".to_string(),
            "test-key".to_string(),
            server.uri(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        let models = provider.list_models().await;
        let models = models.expect("expected Some(_)");
        assert_eq!(models.len(), 3);

        assert_eq!(models[0]["id"], "model-a");
        assert_eq!(models[0]["display_name"], "Model A");
        assert_eq!(models[0]["owned_by"], "anthropic");
        assert_eq!(models[0]["created"], 1000);

        assert_eq!(models[1]["id"], "model-b");
        assert_eq!(models[1]["display_name"], "Model B");
        assert_eq!(models[1]["owned_by"], "anthropic");

        assert_eq!(models[2]["id"], "model-c");
        assert_eq!(models[2]["display_name"], "model-c");
        assert_eq!(models[2]["owned_by"], "anthropic");
    }

    #[tokio::test]
    async fn list_models_returns_none_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "test".to_string(),
            "test-key".to_string(),
            server.uri(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_on_network_error() {
        let server = MockServer::start().await;
        let uri = server.uri();
        drop(server);

        let provider = AnthropicProvider::new(
            "test".to_string(),
            "test-key".to_string(),
            uri,
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_on_malformed_json() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "test".to_string(),
            "test-key".to_string(),
            server.uri(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }

    #[tokio::test]
    async fn list_models_returns_none_when_data_field_is_missing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            "test".to_string(),
            "test-key".to_string(),
            server.uri(),
            HashMap::new(),
            reqwest::Client::new(),
        )
        .unwrap();

        assert!(provider.list_models().await.is_none());
    }
}

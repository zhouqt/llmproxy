//! OpenAI Chat Completions API types (also reused by all OpenAI-compatible providers).
//!
//! Reference: <https://platform.openai.com/docs/api-reference/chat>

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Error envelope detection ─────────────────────────────────────────────

/// Detect the OpenAI-style error envelope `{"error": {...}}` returned on
/// HTTP 200 by some upstreams (notably DeepSeek and GitHub Copilot for
/// unknown models). Used by every provider that deserializes a ChatResponse
/// from a 200 body so a non-conformant body surfaces as an upstream error
/// instead of a generic "missing field `object`" 500.
///
/// Must be a top-level object with a single `error` key whose value is
/// itself an object — that shape disambiguates from a legitimate
/// assistant message that happens to contain the word "error".
pub fn looks_like_error_envelope(v: &Value) -> bool {
    let Value::Object(map) = v else {
        return false;
    };
    matches!(map.get("error"), Some(Value::Object(_)))
}

// ─── Request ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// GPT-5.x and newer models require `max_completion_tokens` and
    /// reject the legacy `max_tokens`. We emit both so the upstream
    /// picks whichever it recognizes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// OpenAI prompt-cache namespace. Set by the request translator
    /// when the Anthropic client sent any `cache_control` block AND
    /// provided `metadata.user_id`; otherwise `None` so the field is
    /// absent from the wire. See `conversion::cache_hint`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// OpenAI prompt-cache TTL. `"in_memory"` (~5–10 min) maps to
    /// Anthropic's `ephemeral` / `ephemeral_5m`; `"24h"` maps to
    /// Anthropic's `ephemeral_1h`. `None` when the request had no
    /// `cache_control` markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: UserContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
    Developer {
        content: String,
    },
}

impl Default for ChatMessage {
    fn default() -> Self {
        ChatMessage::System {
            content: String::new(),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ─── Response ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    /// `object` is the OpenAI discriminator (`"chat.completion"`). Some
    /// upstreams that speak OpenAI-compatible JSON omit it (notably
    /// GitHub Copilot's `/chat/completions` endpoint). Tolerate the
    /// missing field so a successful response doesn't surface as a
    /// generic "missing field `object`" 500.
    #[serde(default = "default_chat_object")]
    pub object: String,
    /// `created` is the OpenAI timestamp; Copilot's `/chat/completions`
    /// response omits it. Optional so a successful 200 body still
    /// deserializes — we don't currently surface `created` to clients
    /// anyway.
    #[serde(default)]
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

fn default_chat_object() -> String {
    "chat.completion".to_string()
}

fn default_chunk_object() -> String {
    "chat.completion.chunk".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Breakdown of the completion_tokens: how many were spent on
    /// reasoning / thinking. Returned by DeepSeek-R1 and other
    /// reasoning models. Not surfaced to the Anthropic client today
    /// (Anthropic's `Usage` schema has no equivalent field) — kept
    /// here so the proxy can warn operators when output is dominated
    /// by reasoning. See fix-R6 in docs/TEST_ISSUES.md.
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;

    #[test]
    fn chat_message_default_is_empty_system() {
        let msg = ChatMessage::default();
        expect_variant!(msg, ChatMessage::System { content, name } => {
            assert_eq!(content, "");
            assert!(name.is_none());
        });
    }

    #[test]
    fn chat_response_accepts_missing_object_field() {
        // GitHub Copilot's /chat/completions response omits the `object`
        // discriminator. Treating the field as required makes a successful
        // 200 response surface as a generic 500 "missing field `object`"
        // and, worse, the Json deserialization error is not cooldownable
        // — so the router exits the fallback chain before reaching
        // downstream providers. Tolerate the missing field instead.
        let body = serde_json::json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
    }

    #[test]
    fn chat_response_accepts_missing_created_field() {
        // Copilot also omits the OpenAI `created` timestamp. Optional
        // because we don't surface `created` to the Anthropic client
        // anyway.
        let body = serde_json::json!({
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "model": "m"
        });
        let resp: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.created, 0);
    }

    #[test]
    fn chat_response_overrides_object_when_present() {
        let body = serde_json::json!({
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "model": "m"
        });
        let resp: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.object, "chat.completion");
    }

    #[test]
    fn chat_chunk_accepts_missing_object_field() {
        // Same reason as ChatResponse: Copilot's SSE chunks omit `object`.
        let body = serde_json::json!({
            "id": "c",
            "created": 0,
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": "hi"},
                "finish_reason": null
            }]
        });
        let chunk: ChatChunk = serde_json::from_value(body).unwrap();
        assert_eq!(chunk.object, "chat.completion.chunk");
    }
}

// ─── Streaming chunks ────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChunk {
    pub id: String,
    /// `object` is the OpenAI discriminator (`"chat.completion.chunk"`).
    /// Some upstreams omit it (e.g. GitHub Copilot's SSE chunks).
    /// Tolerate the missing field rather than dropping every chunk on
    /// the floor — see the `ChatResponse.object` note for the same
    /// reasoning on the non-streaming path.
    #[serde(default = "default_chunk_object")]
    pub object: String,
    /// `created` is the OpenAI timestamp; Copilot's SSE chunks omit it.
    /// Optional so deserialization still succeeds on those streams.
    #[serde(default)]
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunctionCall>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChunkFunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

// ─── Error response ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub code: Option<Value>,
}

#[cfg(test)]
mod looks_like_error_envelope_tests {
    use super::*;

    /// `looks_like_error_envelope` short-circuits to false for any
    /// top-level value that is not a JSON object. Without a guard test,
    /// the `Value::Object` early-return is never exercised.
    #[test]
    fn looks_like_error_envelope_rejects_non_object_values() {
        assert!(!looks_like_error_envelope(&serde_json::json!("a string")));
        assert!(!looks_like_error_envelope(&serde_json::json!(42)));
        assert!(!looks_like_error_envelope(&serde_json::json!(null)));
        assert!(!looks_like_error_envelope(&serde_json::json!([1, 2, 3])));
    }

    /// An object with an `error` field whose value is not itself an
    /// object (e.g. a string) is not an OpenAI error envelope — that
    /// shape is reserved for assistant-message payloads that happen to
    /// mention the word "error".
    #[test]
    fn looks_like_error_envelope_rejects_non_object_error_field() {
        let body = serde_json::json!({"error": "rate limited"});
        assert!(!looks_like_error_envelope(&body));
    }

    #[test]
    fn looks_like_error_envelope_accepts_object_error_field() {
        let body = serde_json::json!({
            "error": {"message": "rate limited", "type": "rate_limit"}
        });
        assert!(looks_like_error_envelope(&body));
    }
}

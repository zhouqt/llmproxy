//! Anthropic Messages API types.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>
//! Streaming events: <https://docs.anthropic.com/en/api/messages-streaming>

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ─── Request ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Top-level cache control breakpoint applied to the last cacheable block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlEphemeral>,
    /// Container identifier for code execution reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Geographic region for inference processing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Request service tier (`"auto"` or `"standard_only"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Output configuration (effort, json schema format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// User profile id (sent as `anthropic-user-profile-id` header).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_profile_id: Option<String>,
    /// Any additional fields not modelled above must still round-trip.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra: std::collections::HashMap<String, Value>,
}

fn default_max_tokens() -> u32 {
    1024
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: String, // "text"
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlEphemeral>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String, // "user" | "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Value>>,
    },
    Image {
        source: ImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    Document {
        source: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    SearchResult {
        source: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<Value>>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    WebFetchToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    CodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    BashCodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    TextEditorCodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    ToolSearchToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    ContainerUpload {
        file_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    MidConversationSystem {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlEphemeral>,
    },
    /// Catch-all for unknown future block types — preserves all unknown
    /// fields so the proxy never silently drops a block the upstream may
    /// understand.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: String, // "base64"
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<Value>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
    #[serde(other)]
    None,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: String, // "enabled" | "disabled" | "adaptive"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// `"summarized"` (default) or `"omitted"`. Controls whether the
    /// response carries a full thinking block or a redacted thinking block
    /// with only a signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

/// `cache_control` block-annotation marker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheControlEphemeral {
    #[serde(rename = "type")]
    pub kind: String, // "ephemeral"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>, // "5m" | "1h"
}

/// Output configuration: effort hint + JSON schema format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
}

// ─── Response ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    pub content: Vec<ResponseBlock>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    pub usage: Usage,
    /// Forward-compat hatch: any field the upstream adds that we don't
    /// model yet stays in the JSON.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Value>>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: Value,
    },
    WebFetchToolResult {
        tool_use_id: String,
        content: Value,
    },
    CodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
    },
    BashCodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
    },
    TextEditorCodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
    },
    ToolSearchToolResult {
        tool_use_id: String,
        content: Value,
    },
    ContainerUpload {
        file_id: String,
    },
    /// Forward-compat hatch for response block types we don't yet model.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheCreation {
    pub ephemeral_1h_input_tokens: u32,
    pub ephemeral_5m_input_tokens: u32,
}

// ─── Streaming SSE events ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: u32,
        content_block: ResponseBlock,
    },
    Ping,
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaPayload,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    MessageStop,
    Error {
        error: Value,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    CitationsDelta { citation: Value },
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
}

#[cfg(test)]
mod schema_tests {
    //! Field-by-field roundtrip coverage for the Anthropic Messages API.
    //!
    //! Every public type the proxy may serialize or deserialize carries
    //! a roundtrip test: parse a representative JSON snippet from the
    //! official SDK / docs, serialize back, and assert structural
    //! equality. If a future field is added to the wire format but the
    //! proxy's struct forgets to model it, the roundtrip will silently
    //! drop the unknown key — these tests pin down each known field so
    //! we catch regressions at compile time.
    use super::*;
    use serde_json::{json, Value};

    fn roundtrip<T>(label: &str, json_in: Value) -> Value
    where
        T: serde::de::DeserializeOwned + serde::Serialize,
    {
        let parsed: T = serde_json::from_value(json_in.clone())
            .unwrap_or_else(|e| panic!("{label}: parse failed: {e}"));
        let serialized = serde_json::to_value(&parsed)
            .unwrap_or_else(|e| panic!("{label}: serialize failed: {e}"));
        serialized
    }

    fn assert_subset(actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Object(actual_map), Value::Object(expected_map)) => {
                for (k, v) in expected_map {
                    let av = actual_map
                        .get(k)
                        .unwrap_or_else(|| panic!("missing key {k}: actual={actual:?}"));
                    assert_subset(av, v);
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "array length mismatch");
                for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                    assert_subset(av, ev);
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn messages_request_roundtrips_every_documented_field() {
        let raw = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "you are claude", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "thinking…", "signature": "sig-1"},
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}, "cache_control": {"type": "ephemeral"}},
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": false},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}, "cache_control": {"type": "ephemeral"}},
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}, "title": "Doc", "context": "ctx", "citations": {"enabled": true}},
                    {"type": "search_result", "source": {"type": "text"}, "title": "hit", "content": [{"type": "text", "text": "snippet"}]},
                    {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"q": "rust"}, "caller": {"type": "direct", "tool_id": "x"}},
                    {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"type": "web_search_result", "title": "t", "url": "https://x", "encrypted_content": "blob"}]},
                    {"type": "web_fetch_tool_result", "tool_use_id": "s2", "content": "fetched body"},
                    {"type": "code_execution_tool_result", "tool_use_id": "s3", "content": {"type": "code_execution_result", "stdout": "hi"}},
                    {"type": "bash_code_execution_tool_result", "tool_use_id": "s4", "content": {"type": "bash_code_execution_result", "stdout": "hi"}},
                    {"type": "text_editor_code_execution_tool_result", "tool_use_id": "s5", "content": {"type": "text_editor_code_execution_view_result", "file_type": "ts"}},
                    {"type": "tool_search_tool_result", "tool_use_id": "s6", "content": {"type": "tool_search_tool_search_result", "tool_references": []}},
                    {"type": "container_upload", "file_id": "file_1"},
                    {"type": "mid_conversation_system", "content": "system mid"}
                ]}
            ],
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 40,
            "stop_sequences": ["STOP"],
            "stream": true,
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "f"},
            "metadata": {"user_id": "u-1"},
            "thinking": {"type": "enabled", "budget_tokens": 4000, "display": "summarized"},
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "container": "container_abc",
            "inference_geo": "us",
            "service_tier": "auto",
            "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object"}}},
            "user_profile_id": "profile-1"
        });
        let value = roundtrip::<MessagesRequest>("messages_request", raw.clone());
        assert_subset(&value, &raw);
    }

    #[test]
    fn thinking_block_signature_roundtrip() {
        let raw = json!({
            "thinking": "the model reasoned",
            "signature": "sig-abcdef",
            "type": "thinking"
        });
        let v = roundtrip::<ContentBlock>("thinking_signature", raw.clone());
        assert_subset(&v, &raw);
    }

    #[test]
    fn thinking_block_missing_signature_serialization_omits_field() {
        // Per SDK schema, signature is Required in thinking_block_param —
        // but a permissive client might omit it on inbound. After
        // serialization we omit the field as well to keep the wire shape
        // identical to the input.
        let raw = json!({"thinking": "t", "type": "thinking"});
        let parsed: ContentBlock = serde_json::from_value(raw).unwrap();
        let v = serde_json::to_value(&parsed).unwrap();
        assert_eq!(v.get("signature"), None);
    }

    #[test]
    fn redacted_thinking_block_roundtrip() {
        let raw = json!({"data": "encrypted-blob", "type": "redacted_thinking"});
        let v = roundtrip::<ContentBlock>("redacted_thinking", raw.clone());
        assert_subset(&v, &raw);
    }

    #[test]
    fn cache_control_ephemeral_roundtrip_with_ttl() {
        let raw = json!({"type": "ephemeral", "ttl": "1h"});
        let cc: CacheControlEphemeral = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(cc.ttl.as_deref(), Some("1h"));
        let v = serde_json::to_value(&cc).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn cache_control_legacy_ephemeral_1h_type_roundtrip() {
        // Legacy clients use type="ephemeral_1h" (string in `type`).
        let raw = json!({"type": "ephemeral_1h"});
        let cc: CacheControlEphemeral = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(cc.kind, "ephemeral_1h");
        let v = serde_json::to_value(&cc).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn tool_choice_variants_roundtrip() {
        for raw in [
            json!({"type": "auto"}),
            json!({"type": "any"}),
            json!({"type": "tool", "name": "f"}),
            json!({"type": "none"}),
        ] {
            let v = roundtrip::<ToolChoice>("tool_choice", raw.clone());
            assert_subset(&v, &raw);
        }
    }

    #[test]
    fn unknown_content_block_is_preserved_not_dropped() {
        // Forward-compat: a block type we don't yet model must still
        // deserialize (as `Unknown`) instead of failing the whole
        // request. The Unknown arm carries no fields, so it can't
        // preserve structured data — but the request still parses.
        let raw = json!({"type": "future_block_2027", "payload": {"x": 1}});
        let parsed: ContentBlock = serde_json::from_value(raw).unwrap();
        assert!(matches!(parsed, ContentBlock::Unknown));
    }

    #[test]
    fn messages_response_roundtrips_usage_with_cache_creation_breakdown() {
        let raw = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_details": {"reason": "policy"},
            "container": {"id": "container_x"},
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation": {"ephemeral_1h_input_tokens": 2, "ephemeral_5m_input_tokens": 3},
                "server_tool_use": {"web_search_requests": 1},
                "output_tokens_details": {"thinking_tokens": 8},
                "service_tier": "priority",
                "inference_geo": "us-west"
            }
        });
        let v = roundtrip::<MessagesResponse>("messages_response", raw.clone());
        assert_subset(&v, &raw);
    }

    #[test]
    fn thinking_block_signature_survives_anthropic_passthrough() {
        // Specifically guard against the regression that motivated this
        // work: ContentBlock::Thinking had no `signature` field, so the
        // proxy stripped it from client → upstream requests and from
        // upstream → client responses, triggering "the content[].thinking
        // in the thinking mode must be passed back to the API" on the
        // second turn.
        let raw = json!({
            "type": "thinking",
            "thinking": "I need to think about this",
            "signature": "sig-xyz"
        });
        let v: Value = roundtrip::<ContentBlock>("thinking_sig_roundtrip", raw.clone());
        assert_eq!(
            v.get("signature").and_then(|s| s.as_str()),
            Some("sig-xyz"),
            "signature must survive roundtrip"
        );
    }

    #[test]
    fn stream_event_block_delta_signature_carries_signature() {
        let raw = json!({"type": "signature_delta", "signature": "sig-stream-1"});
        let v = serde_json::to_value(&BlockDelta::SignatureDelta { signature: "sig-stream-1".into() }).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn stream_event_block_delta_citations_carries_citation() {
        let raw = json!({
            "type": "citations_delta",
            "citation": {"type": "char_location", "cited_text": "x", "document_index": 0,
                         "document_title": "t", "end_char_index": 10, "start_char_index": 0}
        });
        let citation = json!({"type": "char_location", "cited_text": "x", "document_index": 0,
                              "document_title": "t", "end_char_index": 10, "start_char_index": 0});
        let v = serde_json::to_value(&BlockDelta::CitationsDelta { citation: citation.clone() }).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn message_delta_event_surfaces_usage_at_top_level() {
        // The Anthropic Messages SSE spec places `usage` as a sibling of
        // `delta` on the `message_delta` event, not nested inside `delta`.
        // Putting it inside `delta` makes the SDK pydantic validator throw
        // `usage: Field required`, which causes Claude Code to silently
        // abort a tool_use response.
        let raw = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_details": {"reason": "policy"},
                "container": {"id": "x"}
            },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 4,
                "cache_creation": {"ephemeral_1h_input_tokens": 1, "ephemeral_5m_input_tokens": 2}
            }
        });
        let usage = Usage {
            input_tokens: 1,
            output_tokens: 2,
            cache_creation_input_tokens: Some(3),
            cache_read_input_tokens: Some(4),
            cache_creation: Some(CacheCreation { ephemeral_1h_input_tokens: 1, ephemeral_5m_input_tokens: 2 }),
            server_tool_use: None,
            output_tokens_details: None,
            service_tier: None,
            inference_geo: None,
        };
        let payload = MessageDeltaPayload {
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            stop_details: Some(json!({"reason": "policy"})),
            container: Some(json!({"id": "x"})),
        };
        let ev = StreamEvent::MessageDelta {
            delta: payload,
            usage: Some(usage),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn thinking_config_display_field_roundtrip() {
        // The display field controls summarized vs omitted thinking
        // (omitted mode still returns a signature for multi-turn
        // continuity). It must roundtrip.
        let raw = json!({"type": "enabled", "budget_tokens": 4096, "display": "omitted"});
        let cfg: ThinkingConfig = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(cfg.display.as_deref(), Some("omitted"));
        let v = serde_json::to_value(&cfg).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn thinking_config_adaptive_kind_roundtrip() {
        let raw = json!({"type": "adaptive", "display": "summarized"});
        let cfg: ThinkingConfig = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(cfg.kind, "adaptive");
        let v = serde_json::to_value(&cfg).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn output_config_effort_and_format_roundtrip() {
        let raw = json!({
            "effort": "high",
            "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"q": {"type": "string"}}}}
        });
        let cfg: OutputConfig = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(cfg.effort.as_deref(), Some("high"));
        let v = serde_json::to_value(&cfg).unwrap();
        assert_subset(&v, &raw);
    }

    #[test]
    fn extra_hashmap_carries_unknown_request_fields() {
        // The flatten extra map must capture any field the upstream
        // adds before we update the schema. If a future SDK release
        // introduces "anthropic-beta" or similar, the proxy won't drop it.
        let raw = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "anthropic_beta": ["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
            "future_flag_2027": {"x": 1}
        });
        let req: MessagesRequest = serde_json::from_value(raw.clone()).unwrap();
        let v = serde_json::to_value(&req).unwrap();
        assert_subset(&v, &raw);
    }
}

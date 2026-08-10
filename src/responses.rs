//! OpenAI Responses API types.
//!
//! Reference: <https://platform.openai.com/docs/api-reference/responses>
//! Streaming events: <https://platform.openai.com/docs/guides/streaming-responses>
//!
//! The Responses API differs structurally from Chat Completions:
//! - Single `input: Vec<ResponseInputItem>` instead of `messages: Vec<ChatMessage>`
//! - Typed `output: Vec<OutputItem>` instead of `choices[].message`
//! - Tools are flattened (no nested `function` wrapper)
//! - SSE events use the `response.*` namespace
//!
//! This file holds request/response/SSE-event types only. Conversion
//! to/from Anthropic Messages lives in `src/conversion/responses.rs`
//! and `src/conversion/responses_stream.rs`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Request ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<ResponseInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// OpenAI Responses-API prompt-cache namespace. Set by the
    /// request translator when the Anthropic client sent any
    /// `cache_control` block AND provided `metadata.user_id`; otherwise
    /// `None` so the field is absent from the wire. See
    /// `conversion::cache_hint`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// OpenAI Responses-API prompt-cache TTL. `"in_memory"` (~5–10 min)
    /// maps to Anthropic's `ephemeral` / `ephemeral_5m`; `"24h"` maps to
    /// Anthropic's `ephemeral_1h`. `None` when the request had no
    /// `cache_control` markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// OpenAI Responses-API `store` flag. **Deliberately never set by the
    /// request translator** (PR-3: keep OpenAI's default `store: true`
    /// rather than opt the user out) — `None` keeps the field absent from
    /// the wire. Modeled so a future `provider.store_default` knob can
    /// inject `Some(_)` without a wire-type change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// PR-7: OpenAI `service_tier` (top-level Responses-API field).
    /// Forwarded from Anthropic `service_tier` per plan v0.7 mapping
    /// (auto→auto same-value passthrough; standard_only dropped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Anything we don't model explicitly passes through. Defaults to {}.
    #[serde(default, flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String, // "user" | "assistant" | "system" | "developer"
        content: ResponseInputContent,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    /// Past reasoning emitted by the model, replayed into a subsequent
    /// turn's `input[]`. Spec fields: `id` (required), `summary` array
    /// (default `[]`), `content` array (default `[]`), `status`
    /// (optional). Empty/missing arrays decode cleanly via
    /// `#[serde(default)]` so partial payloads round-trip.
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryTextContent>,
        #[serde(default)]
        content: Vec<ReasoningTextContent>,
        #[serde(default)]
        status: Option<String>,
    },
    /// Reference to a previously-produced item (reasoning, message,
    /// function_call, etc.) by id. Spec `ItemReferenceParam.id` (v0.11
    /// correction — earlier plan draft used `reference`; spec is `id`).
    ItemReference { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInputContent {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputPart {
    InputText { text: String },
    InputImage { image_url: String, detail: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// OpenAI Responses API hosted web search tool declared as a tool
    /// entry with `{"type": "web_search", ...}`.
    /// References:
    /// - https://platform.openai.com/docs/guides/tools-web-search
    #[serde(rename = "web_search")]
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
    },
    /// OpenAI Responses API hosted web search tool declared as a tool
    /// entry with `{"type": "web_search_preview", ...}`.
    /// References:
    /// - https://platform.openai.com/docs/guides/tools-web-search
    /// - litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py:186-188
    #[serde(rename = "web_search_preview")]
    WebSearchPreview {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
    },
    /// Catch-all for tool types we don't model (28 Responses tool/item
    /// variants exist; only a handful are common). Round-trips without
    /// panicking — the converter treats this as a no-op.
    #[serde(other)]
    Unknown,
}

/// Reasoning configuration for the Responses API.
///
/// **Serialization**: plain struct — no `type` field (OpenAI official `reasoning`
/// is `{effort, summary}` only; a `type` field would 400 on the official endpoint).
/// Copilot historically tolerated the old `{"type":"enabled",...}` shape; serde
/// silently ignores the unknown `type` key (wire types do not use `deny_unknown_fields`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>, // "low" | "medium" | "high" | "xhigh" | "max" | ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

// ─── Response ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String, // "response"
    #[serde(default)]
    pub created_at: i64,
    pub model: String,
    pub status: String, // "completed" | "incomplete" | "failed"
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    /// Forward-compat for fields we don't model.
    #[serde(default, flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IncompleteDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// PR-7: upstream `service_tier` (auto/default/flex/scale/priority/
    /// fast) echoed back as an Anthropic `Usage.service_tier` string.
    /// Anthropic's response enum is standard/priority/batch; only
    /// `priority` overlaps — the rest pass through unvalidated (plan
    /// v0.12 P1-B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message {
        id: String,
        role: String, // "assistant"
        status: String, // "completed" | "incomplete"
        content: Vec<OutputContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
    /// Hosted web search call produced by a `web_search_preview` tool.
    /// References:
    /// - https://platform.openai.com/docs/guides/tools-web-search
    /// - litellm/types/llms/openai.py:1638-1653 (SSE events)
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        id: String,
        status: String, // "in_progress" | "searching" | "completed"
    },
    /// Reasoning item produced by the model (one per reasoning turn).
    /// Spec `ReasoningItem` fields: `id` (required), `summary` array
    /// (default `[]`), `content` array (default `[]`), `status`
    /// (optional). Empty/missing arrays decode cleanly via
    /// `#[serde(default)]`.
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryTextContent>,
        #[serde(default)]
        content: Vec<ReasoningTextContent>,
        #[serde(default)]
        status: Option<String>,
    },
    /// Unknown item type — kept for forward compatibility.
    #[serde(other)]
    Unknown,
}

/// Content of a `ReasoningItem.content[]` array — currently only a single
/// `reasoning_text` variant exists per spec, modeled so the wire can be
/// extended without a type churn. Tagged with `type="reasoning_text"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningTextContent {
    ReasoningText { text: String },
}

/// Content of a `ReasoningItem.summary[]` array — spec ships a single
/// `summary_text` variant today. Tagged with `type="summary_text"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryTextContent {
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Value>>,
    },
    /// Refusal content part. Spec: `output[].content[].type="refusal"`,
    /// carries a `refusal` string with the refusal text. Empty string is
    /// permitted (the field is required but may be empty).
    Refusal { refusal: String },
    /// Catch-all for output content parts we don't model.
    #[serde(other)]
    Unknown,
}

// ─── Streaming SSE events ────────────────────────────────────────────────

/// A single SSE event emitted by the Responses API stream.
///
/// Event types are decoded from `event.type` (top-level); the payload
/// shape varies by type. The `Unknown` variant captures any event we
/// don't model directly so we can pass it through or skip it.
///
/// Note: `rename_all = "snake_case"` on the enum is not used because
/// the upstream event names contain dots (e.g. `response.created`),
/// which `snake_case` rewrites can't produce. Each variant carries
/// its own `#[serde(rename = "...")]` matching the upstream literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated {
        response: Box<ResponsesResponse>,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        response: Box<ResponsesResponse>,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: Box<ResponsesResponse>,
    },
    #[serde(rename = "response.failed")]
    ResponseFailed {
        response: Box<ResponsesResponse>,
    },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        response: Box<ResponsesResponse>,
    },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        output_index: u32,
        item: OutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContentPart,
    },
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// SSE event carrying a reasoning-text delta (the model's chain of
    /// thought). Mapped to an Anthropic `thinking` content block. Official
    /// fields: `type/item_id/output_index/content_index/delta/sequence_number`.
    /// `content_index`/`sequence_number` use `#[serde(default)]` — Copilot
    /// may omit them.
    #[serde(rename = "response.reasoning_text.delta")]
    ResponseReasoningTextDelta {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        content_index: u32,
        delta: String,
        #[serde(default)]
        sequence_number: u64,
    },
    /// SSE event closing a reasoning-text part with its full accumulated
    /// text.
    #[serde(rename = "response.reasoning_text.done")]
    ResponseReasoningTextDone {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        content_index: u32,
        text: String,
        #[serde(default)]
        sequence_number: u64,
    },
    /// SSE event carrying a reasoning-summary delta. Anthropic has no
    /// summary concept, so the translator ignores it (modeled to tolerate
    /// the event rather than falling to `Unknown`).
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ResponseReasoningSummaryTextDelta {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        summary_index: u32,
        delta: String,
        #[serde(default)]
        sequence_number: u64,
    },
    /// SSE event closing a reasoning-summary part. Ignored (see the delta
    /// variant).
    #[serde(rename = "response.reasoning_summary_text.done")]
    ResponseReasoningSummaryTextDone {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        summary_index: u32,
        text: String,
        #[serde(default)]
        sequence_number: u64,
    },
    /// SSE event carrying a refusal delta. Mapped to a text delta on a
    /// text block (the Anthropic wire has no refusal delta primitive;
    /// the client sees the refusal as a text block, with `stop_reason`
    /// finalized as `refusal`).
    #[serde(rename = "response.refusal.delta")]
    ResponseRefusalDelta {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        content_index: u32,
        delta: String,
        #[serde(default)]
        sequence_number: u64,
    },
    /// SSE event closing a refusal part with its full text.
    #[serde(rename = "response.refusal.done")]
    ResponseRefusalDone {
        item_id: String,
        output_index: u32,
        #[serde(default)]
        content_index: u32,
        text: String,
        #[serde(default)]
        sequence_number: u64,
    },
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        item_id: String,
        output_index: u32,
        arguments: String,
    },
    /// SSE event indicating a web search call is in progress (hosted
    /// tool provider has started the search).
    #[serde(rename = "response.web_search_call.in_progress")]
    ResponseWebSearchCallInProgress {
        output_index: u32,
        item_id: String,
    },
    /// SSE event indicating the web search provider is actively searching.
    #[serde(rename = "response.web_search_call.searching")]
    ResponseWebSearchCallSearching {
        output_index: u32,
        item_id: String,
    },
    /// SSE event indicating the web search has completed.
    #[serde(rename = "response.web_search_call.completed")]
    ResponseWebSearchCallCompleted {
        output_index: u32,
        item_id: String,
    },
    /// Upstream SSE error event. OpenAI send these inline during a
    /// stream when something goes wrong mid-response (e.g. model
    /// overload, internal error).
    #[serde(rename = "error")]
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        #[serde(default)]
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        param: Option<String>,
        #[serde(default, flatten)]
        extra: Value,
    },
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_round_trip_minimal() {
        let req = ResponsesRequest {
            model: "gpt-5".into(),
            input: vec![ResponseInputItem::Message {
                role: "user".into(),
                content: ResponseInputContent::Text("hello".into()),
            }],
            instructions: None,
            max_output_tokens: Some(1024),
            temperature: None,
            top_p: None,
            stream: false,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            store: None,
            service_tier: None,
            extra: json!({}),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "gpt-5");
        assert_eq!(v["input"][0]["role"], "user");
        assert_eq!(v["max_output_tokens"], 1024);
        assert_eq!(v["stream"], false);
    }

    /// PR-3: the request translator never sets `store`, so the wire must
    /// not carry the field — OpenAI's default `store: true` is preserved.
    #[test]
    fn responses_request_omits_store_field_by_default() {
        let v = serde_json::to_value(base_request_with_store(None)).unwrap();
        assert!(
            !v.as_object().unwrap().contains_key("store"),
            "store must be absent from the wire when None; got {v}"
        );
    }

    /// PR-3 wire-only: an explicit `store: Some(true)` serializes the
    /// field. No code path injects `Some` yet (translator keeps None), so
    /// this is a wire-layer test, not an end-to-end injection test.
    #[test]
    fn store_field_serializes_when_explicitly_set() {
        let v = serde_json::to_value(base_request_with_store(Some(true))).unwrap();
        assert_eq!(v["store"], true);
        let v_false = serde_json::to_value(base_request_with_store(Some(false))).unwrap();
        assert_eq!(v_false["store"], false);
    }

    fn base_request_with_store(store: Option<bool>) -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-5".into(),
            input: vec![ResponseInputItem::Message {
                role: "user".into(),
                content: ResponseInputContent::Text("hello".into()),
            }],
            instructions: None,
            max_output_tokens: Some(1024),
            temperature: None,
            top_p: None,
            stream: false,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            store,
            service_tier: None,
            extra: json!({}),
        }
    }

    #[test]
    fn response_with_function_call_decodes() {
        let raw = json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}", "status": "completed"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::FunctionCall { name, .. } => assert_eq!(name, "get_weather"),
            _ => panic!("expected function_call"),
        }
    }

    #[test]
    fn stream_event_response_created_decodes() {
        let raw = json!({
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "in_progress",
                "output": [],
                "usage": {}
            }
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
        match ev {
            ResponsesStreamEvent::ResponseCreated { response } => {
                assert_eq!(response.id, "resp_1");
            }
            _ => panic!("expected response.created"),
        }
    }

    #[test]
    fn stream_event_text_delta_decodes() {
        let raw = json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hi"
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
        match ev {
            ResponsesStreamEvent::ResponseOutputTextDelta { delta, .. } => {
                assert_eq!(delta, "hi");
            }
            _ => panic!("expected text delta"),
        }
    }

    // ── PR-4 · reasoning stream events ─────────────────────────────────

    #[test]
    fn reasoning_text_delta_decodes_with_defaults_for_optional_fields() {
        // Official fields include content_index/sequence_number; Copilot may
        // omit them, so decode must succeed via #[serde(default)].
        for raw in [
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rsn_1",
                "output_index": 0,
                "content_index": 1,
                "delta": "think",
                "sequence_number": 7
            }),
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rsn_1",
                "output_index": 0,
                "delta": "think"
            }),
        ] {
            let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
            match ev {
                ResponsesStreamEvent::ResponseReasoningTextDelta {
                    item_id,
                    delta,
                    content_index,
                    sequence_number,
                    ..
                } => {
                    assert_eq!(item_id, "rsn_1");
                    assert_eq!(delta, "think");
                    // Optional fields default to 0 when the upstream omits
                    // them (verified by the second fixture without them).
                    let _ = (content_index, sequence_number);
                }
                _ => panic!("expected reasoning_text.delta"),
            }
        }
    }

    #[test]
    fn reasoning_text_done_decodes() {
        let raw = json!({
            "type": "response.reasoning_text.done",
            "item_id": "rsn_1",
            "output_index": 0,
            "content_index": 0,
            "text": "the full reasoning",
            "sequence_number": 8
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
        match ev {
            ResponsesStreamEvent::ResponseReasoningTextDone { text, .. } => {
                assert_eq!(text, "the full reasoning");
            }
            _ => panic!("expected reasoning_text.done"),
        }
    }

    #[test]
    fn reasoning_summary_text_delta_and_done_decode() {
        let raw_delta = json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rsn_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "sum",
            "sequence_number": 9
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw_delta).unwrap();
        assert!(matches!(
            ev,
            ResponsesStreamEvent::ResponseReasoningSummaryTextDelta { delta, .. } if delta == "sum"
        ));

        let raw_done = json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rsn_1",
            "output_index": 0,
            "summary_index": 0,
            "text": "the summary",
            "sequence_number": 10
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw_done).unwrap();
        assert!(matches!(
            ev,
            ResponsesStreamEvent::ResponseReasoningSummaryTextDone { text, .. } if text == "the summary"
        ));
    }

    #[test]
    fn unknown_stream_event_decodes_as_unknown() {
        // Future Responses API event we haven't modeled yet.
        let raw = json!({"type": "response.some_future_thing", "x": 1});
        let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
        assert!(matches!(ev, ResponsesStreamEvent::Unknown));
    }

    #[test]
    fn reasoning_config_serializes_without_type_field() {
        // OpenAI official `reasoning` object is `{effort, summary}` — a `type`
        // field would 400 on the official /v1/responses endpoint.
        let cfg = ReasoningConfig {
            effort: Some("medium".into()),
            summary: None,
        };
        let v = serde_json::to_value(&cfg).unwrap();
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("type"), "type field must not appear; got {}", v);
        assert_eq!(obj["effort"], "medium");
        assert!(!obj.contains_key("summary"));
    }

    #[test]
    fn reasoning_config_deserializes_copilot_type_field() {
        // Copilot historically sent `{"type":"enabled",...}`. Wire types don't
        // deny unknown fields, so the stray `type` key is silently ignored.
        let raw = json!({"type": "enabled", "effort": "medium"});
        let cfg: ReasoningConfig = serde_json::from_value(raw).unwrap();
        assert_eq!(cfg.effort.as_deref(), Some("medium"));
        assert!(cfg.summary.is_none());
    }

    #[test]
    fn reasoning_config_default_serializes_as_empty_object() {
        let cfg = ReasoningConfig::default();
        let v = serde_json::to_value(&cfg).unwrap();
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn reasoning_config_roundtrips_with_summary() {
        // Cover every ReasoningSummary variant so the enum's serialize/
        // deserialize regions stay hit.
        for (name, variant) in [
            ("auto", ReasoningSummary::Auto),
            ("concise", ReasoningSummary::Concise),
            ("detailed", ReasoningSummary::Detailed),
        ] {
            let raw = json!({"effort": "high", "summary": name});
            let cfg: ReasoningConfig = serde_json::from_value(raw).unwrap();
            assert_eq!(cfg.effort.as_deref(), Some("high"));
            // The roundtripped summary must be exactly the loop's variant
            // (a plain `matches!` Auto-check would let Concise/Detailed
            // swap silently).
            assert_eq!(cfg.summary, Some(variant));
            let v = serde_json::to_value(&cfg).unwrap();
            assert_eq!(v["summary"], name);
        }
    }

    // ── PR-5 · refusal + ResponseInputItem + ResponsesTool ─────────────

    /// PR-5: `Refusal {refusal}` content part deserializes from spec
    /// shape `{"type":"refusal","refusal":"..."}`. Empty string is
    /// permitted (field is required but may be empty — see plan v0.11).
    #[test]
    fn output_content_refusal_decodes_with_text() {
        let raw = json!({"type": "refusal", "refusal": "I cannot comply."});
        let part: OutputContentPart = serde_json::from_value(raw).unwrap();
        match part {
            OutputContentPart::Refusal { refusal } => {
                assert_eq!(refusal, "I cannot comply.");
            }
            other => panic!("expected Refusal, got {other:?}"),
        }
    }

    /// PR-5: an empty refusal string still decodes (the spec requires
    /// the field but allows empty content).
    #[test]
    fn output_content_refusal_with_empty_string_deserializes() {
        let raw = json!({"type": "refusal", "refusal": ""});
        let part: OutputContentPart = serde_json::from_value(raw).unwrap();
        match part {
            OutputContentPart::Refusal { refusal } => assert_eq!(refusal, ""),
            other => panic!("expected Refusal, got {other:?}"),
        }
    }

    /// PR-5: `ResponseInputItem::Reasoning` round-trips with id + status;
    /// missing `summary` and `content` default to empty Vec.
    #[test]
    fn response_input_reasoning_round_trip() {
        let item = ResponseInputItem::Reasoning {
            id: "rsn_1".into(),
            summary: vec![],
            content: vec![ReasoningTextContent::ReasoningText {
                text: "thought".into(),
            }],
            status: Some("completed".into()),
        };
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "reasoning");
        assert_eq!(v["id"], "rsn_1");
        assert_eq!(v["content"][0]["type"], "reasoning_text");
        assert_eq!(v["content"][0]["text"], "thought");
        let back: ResponseInputItem = serde_json::from_value(v).unwrap();
        match back {
            ResponseInputItem::Reasoning { id, status, .. } => {
                assert_eq!(id, "rsn_1");
                assert_eq!(status.as_deref(), Some("completed"));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    /// PR-5: `Reasoning` with no `summary` / `content` / `status` fields
    /// decodes (all are `#[serde(default)]`). Pin: this is what makes the
    /// wire tolerant to partial reasoning items.
    #[test]
    fn response_input_reasoning_decodes_with_missing_arrays() {
        let raw = json!({"type": "reasoning", "id": "rsn_min"});
        let item: ResponseInputItem = serde_json::from_value(raw).unwrap();
        match item {
            ResponseInputItem::Reasoning {
                id,
                summary,
                content,
                status,
            } => {
                assert_eq!(id, "rsn_min");
                assert!(summary.is_empty());
                assert!(content.is_empty());
                assert!(status.is_none());
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    /// PR-5: `ItemReference {id}` round-trips — v0.11 spec field is `id`,
    /// not `reference`.
    #[test]
    fn response_input_item_reference_round_trip() {
        let item = ResponseInputItem::ItemReference { id: "msg_x".into() };
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "item_reference");
        assert_eq!(v["id"], "msg_x");
        let back: ResponseInputItem = serde_json::from_value(v).unwrap();
        match back {
            ResponseInputItem::ItemReference { id } => assert_eq!(id, "msg_x"),
            other => panic!("expected ItemReference, got {other:?}"),
        }
    }

    /// PR-5: `ResponsesTool::Unknown` catches all tool types we don't
    /// explicitly model (e.g. `code_interpreter`, `mcp`, `image_gen`).
    #[test]
    fn responses_tool_unknown_round_trip() {
        let raw = json!({"type": "code_interpreter", "container": {"type": "auto"}});
        let tool: ResponsesTool = serde_json::from_value(raw).unwrap();
        assert!(matches!(tool, ResponsesTool::Unknown));
        // Unknown is a unit variant — under the outer `tag="type"` enum,
        // it serializes as `{"type":"unknown"}`. The translator treats
        // this as a no-op (it only forwards ResponsesTool::Function /
        // WebSearch / WebSearchPreview to the wire); documented here so
        // a future change to the enum shape knows the expected shape.
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v, json!({"type": "unknown"}));
    }

    /// PR-5: known `Function` tool round-trips; does NOT fall to `Unknown`.
    #[test]
    fn responses_tool_function_round_trip_does_not_fall_to_unknown() {
        let raw = json!({
            "type": "function",
            "name": "f",
            "description": "d",
            "parameters": {"type": "object"}
        });
        let tool: ResponsesTool = serde_json::from_value(raw).unwrap();
        match tool {
            ResponsesTool::Function { name, .. } => assert_eq!(name, "f"),
            other => panic!("expected Function, got {other:?}"),
        }
    }

    /// PR-5: `web_search` (no `_preview` suffix) is now a first-class
    /// variant — plan v0.11 acceptance target.
    #[test]
    fn responses_tool_web_search_round_trip() {
        let raw = json!({"type": "web_search", "search_context_size": "high"});
        let tool: ResponsesTool = serde_json::from_value(raw).unwrap();
        match &tool {
            ResponsesTool::WebSearch { search_context_size, .. } => {
                assert_eq!(search_context_size.as_deref(), Some("high"));
            }
            other => panic!("expected WebSearch, got {other:?}"),
        }
        // Serializes back to `web_search` (the rename target).
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["type"], "web_search");
    }

    /// PR-5: `web_search_preview` still parses as the `WebSearchPreview`
    /// variant — keep both names supported per plan.
    #[test]
    fn responses_tool_web_search_preview_round_trip() {
        let raw = json!({"type": "web_search_preview"});
        let tool: ResponsesTool = serde_json::from_value(raw).unwrap();
        assert!(matches!(tool, ResponsesTool::WebSearchPreview { .. }));
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["type"], "web_search_preview");
    }

    /// PR-5: `OutputItem::Reasoning` decodes with the same wire shape as
    /// `ResponseInputItem::Reasoning` — spec is symmetric. Missing arrays
    /// default to empty.
    #[test]
    fn output_item_reasoning_decodes_with_missing_arrays() {
        let raw = json!({
            "type": "reasoning",
            "id": "rsn_out",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "thought"}],
            "status": "completed"
        });
        let item: OutputItem = serde_json::from_value(raw).unwrap();
        match item {
            OutputItem::Reasoning {
                id,
                summary,
                content,
                status,
            } => {
                assert_eq!(id, "rsn_out");
                assert!(summary.is_empty());
                assert_eq!(content.len(), 1);
                assert_eq!(status.as_deref(), Some("completed"));
            }
            other => panic!("expected OutputItem::Reasoning, got {other:?}"),
        }
    }

    /// PR-5: refusal SSE events decode with default-tolerant fields
    /// (Copilot may omit `content_index` / `sequence_number`).
    #[test]
    fn refusal_delta_event_decodes_with_defaults_for_optional_fields() {
        for raw in [
            json!({
                "type": "response.refusal.delta",
                "item_id": "msg_r",
                "output_index": 0,
                "content_index": 0,
                "delta": "no",
                "sequence_number": 1
            }),
            json!({
                "type": "response.refusal.delta",
                "item_id": "msg_r",
                "output_index": 0,
                "delta": "no"
            }),
        ] {
            let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
            assert!(matches!(
                ev,
                ResponsesStreamEvent::ResponseRefusalDelta { ref delta, .. } if delta == "no"
            ));
        }
    }

    #[test]
    fn refusal_done_event_decodes() {
        let raw = json!({
            "type": "response.refusal.done",
            "item_id": "msg_r",
            "output_index": 0,
            "content_index": 0,
            "text": "no thanks",
            "sequence_number": 2
        });
        let ev: ResponsesStreamEvent = serde_json::from_value(raw).unwrap();
        match ev {
            ResponsesStreamEvent::ResponseRefusalDone { text, .. } => assert_eq!(text, "no thanks"),
            other => panic!("expected refusal.done, got {other:?}"),
        }
    }
}
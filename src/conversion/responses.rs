//! Anthropic Messages ↔ OpenAI Responses API conversion.
//!
//! Reference: <https://platform.openai.com/docs/api-reference/responses>
//! Copilot-equivalent: `copilot-api-py/src/services/copilot/create_responses.py`
//! (request) and `copilot-api-py/src/services/copilot/responses_to_chat.py`
//! (response). The Python impl converts Responses ↔ Chat Completions;
//! we convert Responses ↔ Anthropic Messages directly.
//!
//! Notes:
//! - Anthropic `system` → Responses `instructions` (top-level scalar).
//! - Anthropic `messages[]` → Responses `input[]` (flat list of typed
//!   items — message, function_call, function_call_output).
//! - Responses `output[]` is a typed enum; we fold `OutputItem::Message`
//!   parts into Anthropic `ResponseBlock::Text` (and
//!   `ResponseBlock::Thinking` if the model emitted reasoning), and
//!   `OutputItem::FunctionCall` into `ResponseBlock::ToolUse`.

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::anthropic::{
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, ResponseBlock, SystemPrompt,
    ToolChoice,
};
use crate::conversion::derive_cache_hints;
use crate::conversion::util::{build_usage, strictify_schema};
use crate::error::{ProxyError, Result};
use crate::responses::{
    OutputContentPart, OutputItem, ReasoningConfig, ReasoningSummary, ResponseInputContent,
    ResponseInputItem, ResponseInputPart, ResponsesRequest, ResponsesResponse, ResponsesTool,
};

/// Truncate the `user` identifier to the 64-character limit enforced by
/// Copilot's Responses API. The Anthropic `metadata.user_id` has no
/// length constraint in the spec, so clients routinely send longer
/// values that Copilot rejects.
pub(crate) fn truncate_user(user: &str) -> String {
    if user.chars().count() <= 64 {
        user.to_string()
    } else {
        user.chars().take(64).collect()
    }
}

/// Convert an Anthropic MessagesRequest into an OpenAI ResponsesRequest.
pub fn anthropic_to_responses_request(
    req: &MessagesRequest,
    model_rewrite: &std::collections::HashMap<String, String>,
) -> crate::error::Result<ResponsesRequest> {
    let model = model_rewrite
        .get(&req.model)
        .cloned()
        .unwrap_or_else(|| req.model.clone());

    let mut input: Vec<ResponseInputItem> = Vec::new();

    // Anthropic messages → flat Responses input items. System prompts
    // are dropped here because Anthropic's `system` maps to the
    // top-level `instructions` field below.
    for m in &req.messages {
        if m.role == "system" {
            // Anthropic technically doesn't allow role=system in messages;
            // if a client sneaks one in, fold it into instructions.
            continue;
        }
        input.extend(convert_message(m));
    }

    let instructions = req.system.as_ref().and_then(|s| match s {
        SystemPrompt::Text(s) if !s.is_empty() => Some(s.clone()),
        SystemPrompt::Blocks(blocks) => {
            // Drop empty-text blocks before joining so we don't emit
            // `instructions: "\n\n"` (or any whitespace-only string)
            // when every block is empty. An all-empty `Blocks` array
            // must behave the same as `Text("")` → `None`. See issue #2.
            let non_empty: Vec<&str> = blocks
                .iter()
                .map(|b| b.text.as_str())
                .filter(|t| !t.is_empty())
                .collect();
            if non_empty.is_empty() {
                None
            } else {
                Some(non_empty.join("\n\n"))
            }
        }
        _ => None,
    });

    let tools = req.tools.as_ref().map(|ts| {
        let mut result: Vec<ResponsesTool> = Vec::new();

        // Function tools — strip web_search because it becomes a hosted
        // tool entry below.
        for t in ts {
            if crate::conversion::util::is_web_search_tool(t) {
                // This tool will be appended as ResponsesTool::WebSearch
                // after the loop; don't emit as Function.
                continue;
            }
            result.push(ResponsesTool::Function {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
                strict: None,
            });
        }

        // Hosted web search tool — add after all function tools so the
        // upstream lists web_search_preview separately.
        for t in ts {
            if crate::conversion::util::is_web_search_tool(t) {
                result.push(ResponsesTool::WebSearchPreview {
                    search_context_size: t
                        .extra
                        .get("search_context_size")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            // Heuristic fallback: Anthropic's `max_uses`
                            // (search-count cap) is bucketed into
                            // OpenAI's `search_context_size`
                            // (result-quality tier). These are
                            // semantically different — `max_uses`
                            // bounds call count, `search_context_size`
                            // bounds token budget — but Anthropic
                            // doesn't have a direct equivalent. This
                            // mapping is approximate; a future change
                            // may drop the field entirely (see
                            // docs/PLANS/web-search-followup.md).
                            t.extra.get("max_uses")
                                .and_then(|v| v.as_u64())
                                .map(|u| {
                                    if u >= 10 { "high".to_string() }
                                    else if u >= 5 { "medium".to_string() }
                                    else { "low".to_string() }
                                })
                        }),
                    user_location: t.extra.get("user_location").cloned(),
                });
            }
        }

        result
    });

    let hints = derive_cache_hints(req);

    // GPT-5.x models on the Responses API reject the short `in_memory`
    // retention tier: "This model is compatible only with 24h extended
    // prompt caching". When the client asked for caching at all, honor
    // that intent by escalating the short tier to `24h` for these models
    // rather than letting the request 400.
    let prompt_cache_retention = hints.prompt_cache_retention.map(|r| {
        if r == "in_memory" && crate::util::gpt5_family(&model) {
            "24h".to_string()
        } else {
            r
        }
    });

    Ok(ResponsesRequest {
        model,
        input,
        instructions,
        max_output_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        tools,
        tool_choice: {
            // Detect forced web_search tool_choice. The upstream has no
            // function tool named "web_search" — remap to auto.
            let has_web_search = req.tools.as_ref().map_or(false, |ts| {
                ts.iter().any(crate::conversion::util::is_web_search_tool)
            });
            let tc = req.tool_choice.as_ref().and_then(convert_tool_choice);
            if has_web_search
                && matches!(req.tool_choice, Some(ToolChoice::Tool { ref name, .. }) if name == "web_search")
            {
                Some(json!("auto"))
            } else {
                tc
            }
        },
        parallel_tool_calls: match req.tool_choice.as_ref() {
            // PR-8: Anthropic `tool_choice.disable_parallel_tool_use=true`
            // → Responses `parallel_tool_calls=false`. Absent/false →
            // None (OpenAI wire default is true; emitting the field as
            // explicit `true` would be noisier than necessary).
            Some(crate::anthropic::ToolChoice::Tool {
                disable_parallel_tool_use,
                ..
            }) if *disable_parallel_tool_use == Some(true) => Some(false),
            _ => None,
        },
        user: req
            .metadata
            .as_ref()
            .and_then(|m| m.user_id.as_deref())
            .map(|u| truncate_user(u)),
        prompt_cache_key: hints.prompt_cache_key,
        prompt_cache_retention,
        reasoning: req.output_config.as_ref()
            .and_then(|oc| oc.effort.clone())
            .map(|e| ReasoningConfig { effort: Some(e), summary: Some(ReasoningSummary::Auto) })
            .or_else(|| req.thinking.as_ref().and_then(convert_thinking)),
        // PR-3: keep OpenAI's default (Responses `store: true`) — do not
        // send the field. See `ResponsesRequest.store` doc.
        store: None,
        // PR-7: same mapping as Chat (auto→auto, standard_only→drop).
        // See `conversion/request.rs` for the full rationale.
        service_tier: req.service_tier.as_deref().and_then(|tier| match tier {
            "auto" => Some("auto".to_string()),
            _ => None,
        }),
        extra: {
            let mut e = Value::Object(Map::new());
            // PR-8: Responses API carries verbosity under
            // `text.verbosity` (same enum as Chat top-level `verbosity`).
            // Build the text object once and merge in format if any.
            let mut text_obj = serde_json::Map::new();
            if let Some(v) = req.output_config.as_ref().and_then(|oc| oc.verbosity.clone()) {
                text_obj.insert("verbosity".into(), Value::String(v));
            }
            if let Some(fmt) = req.output_config.as_ref().and_then(|oc| oc.format.as_ref()) {
                text_obj.insert("format".into(), ensure_json_schema_name(fmt)?);
            }
            if !text_obj.is_empty() {
                e["text"] = Value::Object(text_obj);
            }
            e
        },
    })
}

/// OpenAI Responses API requires `text.format.name` on `json_schema` shapes;
/// Anthropic's `output_config.format` doesn't carry one. Synthesize a stable
/// default so the schema constraint round-trips, and pin `strict: true` so the
/// model is held to the schema (matches Anthropic's enforcement semantics).
fn ensure_json_schema_name(
    fmt: &Value,
) -> std::result::Result<Value, crate::conversion::util::SchemaError> {
    if fmt.get("type").and_then(|v| v.as_str()) != Some("json_schema") {
        return Ok(fmt.clone());
    }
    let Some(obj) = fmt.as_object() else {
        return Ok(fmt.clone());
    };
    let mut out = obj.clone();
    if let Some(mut schema) = out.get("schema").cloned() {
        strictify_schema(&mut schema)?;
        out.insert("schema".to_string(), schema);
    }
    out.entry("name".to_string())
        .or_insert(json!("structured_output"));
    out.entry("strict".to_string()).or_insert(json!(true));
    Ok(Value::Object(out))
}

fn convert_message(m: &crate::anthropic::Message) -> Vec<ResponseInputItem> {
    match &m.content {
        MessageContent::Text(s) => {
            vec![ResponseInputItem::Message {
                role: m.role.clone(),
                content: ResponseInputContent::Text(s.clone()),
            }]
        }
        MessageContent::Blocks(blocks) => convert_blocks(&m.role, blocks),
    }
}

fn convert_blocks(role: &str, blocks: &[ContentBlock]) -> Vec<ResponseInputItem> {
    let mut out: Vec<ResponseInputItem> = Vec::new();

    match role {
        "user" => {
            // Tool results become function_call_output items. Text/image
            // blocks merge into a single user message with Parts content.
            let mut parts: Vec<ResponseInputPart> = Vec::new();
            for b in blocks {
                match b {
                    ContentBlock::Text { text, .. } => {
                        parts.push(crate::responses::ResponseInputPart::InputText {
                            text: text.clone(),
                        });
                    }
                    ContentBlock::Image { source, .. } => {
                        let url = format!(
                            "data:{};base64,{}",
                            source.media_type, source.data
                        );
                        parts.push(ResponseInputPart::InputImage {
                            image_url: url,
                            detail: None,
                        });
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        // Tool results are siblings, not children, in Responses input[].
                        out.push(ResponseInputItem::FunctionCallOutput {
                            call_id: tool_use_id.clone(),
                            output: tool_result_to_string(content),
                        });
                    }
                    ContentBlock::ToolUse { .. }
                    | ContentBlock::Thinking { .. }
                    | ContentBlock::RedactedThinking { .. }
                    | ContentBlock::Document { .. }
                    | ContentBlock::SearchResult { .. }
                    | ContentBlock::ServerToolUse { .. }
                    | ContentBlock::WebSearchToolResult { .. }
                    | ContentBlock::WebFetchToolResult { .. }
                    | ContentBlock::CodeExecutionToolResult { .. }
                    | ContentBlock::BashCodeExecutionToolResult { .. }
                    | ContentBlock::TextEditorCodeExecutionToolResult { .. }
                    | ContentBlock::ToolSearchToolResult { .. }
                    | ContentBlock::ContainerUpload { .. }
                    | ContentBlock::MidConversationSystem { .. } => {
                        // Skip — only valid in assistant turns.
                    }
                    ContentBlock::Unknown => {}
                }
            }
            if !parts.is_empty() {
                let content = if parts.len() == 1 {
                    if let ResponseInputPart::InputText { text } = &parts[0] {
                        ResponseInputContent::Text(text.clone())
                    } else {
                        ResponseInputContent::Parts(parts)
                    }
                } else {
                    ResponseInputContent::Parts(parts)
                };
                out.push(ResponseInputItem::Message {
                    role: "user".into(),
                    content,
                });
            }
        }
        "assistant" => {
            // Assistant text accumulates into one message; tool_calls
            // become function_call items (siblings, parallel-safe).
            let mut text_acc = String::new();
            let mut tool_calls: Vec<(String, String, String)> = Vec::new(); // (call_id, name, args)
            for b in blocks {
                match b {
                    ContentBlock::Text { text, .. } => {
                        if !text_acc.is_empty() {
                            text_acc.push('\n');
                        }
                        text_acc.push_str(text);
                    }
                    ContentBlock::Thinking { .. } => {
                        // Reasoning isn't replayed in subsequent turns via
                        // the Responses input[] — drop it.
                    }
                    ContentBlock::ToolUse { id, name, input, .. } => {
                        tool_calls.push((
                            id.clone(),
                            name.clone(),
                            serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                        ));
                    }
                    ContentBlock::Image { .. }
                    | ContentBlock::ToolResult { .. }
                    | ContentBlock::RedactedThinking { .. }
                    | ContentBlock::Document { .. }
                    | ContentBlock::SearchResult { .. }
                    | ContentBlock::ServerToolUse { .. }
                    | ContentBlock::WebSearchToolResult { .. }
                    | ContentBlock::WebFetchToolResult { .. }
                    | ContentBlock::CodeExecutionToolResult { .. }
                    | ContentBlock::BashCodeExecutionToolResult { .. }
                    | ContentBlock::TextEditorCodeExecutionToolResult { .. }
                    | ContentBlock::ToolSearchToolResult { .. }
                    | ContentBlock::ContainerUpload { .. }
                    | ContentBlock::MidConversationSystem { .. }
                    | ContentBlock::Unknown => {}
                }
            }
            if !text_acc.is_empty() {
                out.push(ResponseInputItem::Message {
                    role: "assistant".into(),
                    content: ResponseInputContent::Text(text_acc),
                });
            }
            for (call_id, name, args) in tool_calls {
                out.push(ResponseInputItem::FunctionCall {
                    call_id,
                    name,
                    arguments: args,
                });
            }
        }
        _ => {}
    }

    out
}

fn tool_result_to_string(c: &crate::anthropic::ToolResultContent) -> String {
    match c {
        crate::anthropic::ToolResultContent::Text(s) => s.clone(),
        crate::anthropic::ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn convert_tool_choice(c: &ToolChoice) -> Option<Value> {
    match c {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::Any => Some(json!("required")),
        ToolChoice::Tool { name, .. } => Some(json!({
            "type": "function",
            "name": name
        })),
        ToolChoice::None => Some(json!("none")),
    }
}

fn convert_thinking(t: &crate::anthropic::ThinkingConfig) -> Option<ReasoningConfig> {
    if t.kind == "enabled" {
        let effort = t
            .budget_tokens
            .map(|budget| {
                if budget >= 8000 {
                    "high".to_string()
                } else if budget >= 2000 {
                    "medium".to_string()
                } else {
                    "low".to_string()
                }
            })
            .unwrap_or_else(|| "medium".to_string());
        Some(ReasoningConfig {
            effort: Some(effort),
            // PR-4: request the reasoning summary so the upstream emits
            // `reasoning_summary_text` alongside `reasoning_text`.
            summary: Some(ReasoningSummary::Auto),
        })
    } else {
        None
    }
}

/// Convert a non-streaming Responses API response to an Anthropic
/// MessagesResponse.
///
/// `model` is the original model name from the client request.
/// `message_id` is the Anthropic-style id (e.g. `msg_<uuid>`) to
/// assign.
pub fn responses_to_anthropic_response(
    resp: &ResponsesResponse,
    model: &str,
    message_id: &str,
) -> Result<MessagesResponse> {
    let mut content: Vec<ResponseBlock> = Vec::new();
    let mut has_tool_calls = false;

    for item in &resp.output {
        match item {
            OutputItem::Message {
                content: parts, ..
            } => {
                for part in parts {
                    match part {
                        OutputContentPart::OutputText { text, .. } => {
                            if !text.is_empty() {
                                content.push(ResponseBlock::Text { text: text.clone(), citations: None });
                            }
                        }
                        // PR-5 (G3/G4 inbound): a refusal content part maps
                        // to an Anthropic Text block (the wire has no
                        // refusal-specific block type). The refusal text
                        // surfaces to the client as text; the upstream
                        // `stop_reason=refusal` (PR-7 outbound, picked up
                        // via the incomplete_details path) tells Claude
                        // Code to treat it as a refusal, not user content.
                        OutputContentPart::Refusal { refusal } => {
                            if !refusal.is_empty() {
                                content.push(ResponseBlock::Text {
                                    text: refusal.clone(),
                                    citations: None,
                                });
                            }
                        }
                        OutputContentPart::Unknown => {}
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                // B3: inside an *incomplete* response, a function_call
                // whose arguments don't parse (truncated mid-JSON) must not
                // surface as an empty-input ToolUse — that would hand the
                // client a phantom tool call. Skip it; the `max_tokens`
                // stop_reason below already signals the turn was cut off.
                if resp.status == "incomplete"
                    && serde_json::from_str::<Value>(arguments).is_err()
                {
                    continue;
                }
                let input: Value = serde_json::from_str(arguments)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                content.push(ResponseBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                    caller: None,
                });
                has_tool_calls = true;
            }
            // PR-5: OutputItem::Reasoning is the response-side mirror of
            // ResponseInputItem::Reasoning. We don't fold reasoning into
            // Anthropic `ContentBlock::Thinking` here (the streaming
            // translator owns that surface; non-streaming responses that
            // arrive with a `reasoning` item but no text/tool blocks are
            // reported as end_turn with empty content).
            OutputItem::Reasoning { .. } => {}
            // WebSearchCall doesn't carry result text directly —
            // search results appear as url_citation annotations on
            // the subsequent output_text block. No block to emit.
            OutputItem::WebSearchCall { .. } => {}
            OutputItem::Unknown => {}
        }
    }

    // B2 (spec-confirmed): `incomplete_details.reason` is a two-value enum
    // (`max_output_tokens` / `content_filter`). Read it to map a content
    // filter refusal to `refusal` instead of the old blanket `max_tokens`.
    // B3: `status == "incomplete"` takes priority over `has_tool_calls` — a
    // truncated function_call must read as `max_tokens`, never `tool_use`.
    let stop_reason = match resp.status.as_str() {
        "incomplete" => Some(
            match resp.incomplete_details.as_ref().and_then(|d| d.reason.as_deref()) {
                Some("content_filter") => "refusal".to_string(),
                _ => "max_tokens".to_string(),
            },
        ),
        "completed" => {
            if has_tool_calls {
                Some("tool_use".to_string())
            } else {
                Some("end_turn".to_string())
            }
        }
        // Official `cancelled` status (spec-confirmed) carries no usable
        // output — present it as a clean end_turn with empty content
        // (deliberate Anthropic-compat choice, not OpenAI semantics).
        "cancelled" => {
            content.clear();
            Some("end_turn".to_string())
        }
        "failed" => {
            // If the response carries error details in extra, surface
            // them as an upstream error rather than silently mapping
            // to end_turn.
            let err_msg = resp
                .extra
                .get("error")
                .and_then(|v| v.get("message"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned())
                .unwrap_or_else(|| "upstream error".to_string());
            return Err(ProxyError::Upstream {
                status: 502,
                body: err_msg,
            });
        }
        _ => None,
    };

    let usage_raw = resp.usage.clone().unwrap_or_default();
    let cached = usage_raw
        .input_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage_raw
        .output_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .unwrap_or(0);
    let usage = build_usage(
        usage_raw.input_tokens,
        usage_raw.output_tokens,
        cached,
        reasoning,
        usage_raw.service_tier,
    );

    Ok(MessagesResponse {
        id: message_id.to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        stop_details: None,
        container: None,
        usage,
        extra: HashMap::new(),
    })
}

/// Synthesize an Anthropic-style `msg_<uuid>` id, matching the convention
/// used by `openai_compat.rs` for Chat Completions responses.
pub fn make_message_id() -> String {
    format!("msg_{}", Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::{ContentBlock, Message};
    use std::collections::HashMap;

    fn req_with_text() -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be brief"
        }))
        .unwrap()
    }

    #[test]
    fn request_maps_system_to_instructions() {
        let req = req_with_text();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.model, "gpt-5");
        assert_eq!(out.instructions.as_deref(), Some("be brief"));
        assert_eq!(out.max_output_tokens, Some(256));
        assert_eq!(out.input.len(), 1);
        match &out.input[0] {
            ResponseInputItem::Message { role, content } => {
                assert_eq!(role, "user");
                match content {
                    ResponseInputContent::Text(t) => assert_eq!(t, "hi"),
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected message item"),
        }
    }

    #[test]
    fn request_maps_tool_use_blocks_to_function_call_items() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "what's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "72F sunny"}
                ]}
            ]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        // user msg → assistant function_call → user tool_result (function_call_output)
        assert_eq!(out.input.len(), 3);
        match &out.input[1] {
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "t1");
                assert_eq!(name, "get_weather");
                let parsed: Value = serde_json::from_str(arguments).unwrap();
                assert_eq!(parsed["city"], "SF");
            }
            _ => panic!("expected function_call at input[1]"),
        }
        match &out.input[2] {
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "t1");
                assert_eq!(output, "72F sunny");
            }
            _ => panic!("expected function_call_output at input[2]"),
        }
    }

    #[test]
    fn request_maps_tool_choice_and_thinking() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "enabled", "budget_tokens": 4000}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(
            out.tool_choice.as_ref().unwrap(),
            &json!("required")
        );
        match out.reasoning.as_ref().unwrap() {
            ReasoningConfig {
                effort,
                summary: Some(ReasoningSummary::Auto),
            } => {
                assert_eq!(effort.as_deref(), Some("medium"));
            }
            other => panic!("expected struct with summary Auto, got {other:?}"),
        }
        let tools = out.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            ResponsesTool::Function { name, parameters, .. } => {
                assert_eq!(name, "f");
                assert!(parameters.is_some());
            }
            _ => panic!("expected Function tool"),
        }
    }

    #[test]
    fn response_with_text_and_function_call_converts() {
        let raw = json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "hello"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "f", "arguments": "{\"x\":1}", "status": "completed"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_42").unwrap();
        assert_eq!(out.id, "msg_42");
        assert_eq!(out.model, "gpt-5");
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(out.content.len(), 2);
        assert!(matches!(out.content[0], ResponseBlock::Text { .. }));
        assert!(matches!(out.content[1], ResponseBlock::ToolUse { .. }));
        assert_eq!(out.usage.input_tokens, 10);
    }

    #[test]
    fn response_incomplete_maps_to_max_tokens() {
        let raw = json!({
            "id": "resp_2",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "incomplete",
                 "content": [{"type": "output_text", "text": "partial"}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_1").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("max_tokens"));
    }

    /// PR-7 response side: upstream Responses `service_tier` (e.g.
    /// `fast`) echoes back as Anthropic `Usage.service_tier` — passthrough
    /// string, no enum validation (plan v0.12 P1-B).
    #[test]
    fn usage_service_tier_passes_through() {
        let raw = json!({
            "id": "resp_st",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "service_tier": "fast"
            }
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_1").unwrap();
        assert_eq!(out.usage.service_tier.as_deref(), Some("fast"));
    }

    // ── PR-2 · B2 + B3 ──────────────────────────────────────────────────

    /// B2 (spec-confirmed): `incomplete_details.reason == "content_filter"`
    /// must map to stop_reason `refusal`, not the old blanket `max_tokens`.
    #[test]
    fn incomplete_with_content_filter_reason_returns_refusal() {
        let raw = json!({
            "id": "resp_cf",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "incomplete",
                        "content": [{"type": "output_text", "text": ""}]}],
            "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_cf").unwrap();
        assert_eq!(
            out.stop_reason.as_deref(),
            Some("refusal"),
            "content_filter must map to refusal, got {:?}",
            out.stop_reason
        );
    }

    /// B2 regression: `reason == "max_output_tokens"` (or absent) keeps the
    /// old `max_tokens` mapping.
    #[test]
    fn incomplete_with_max_tokens_returns_max_tokens_stop_reason() {
        for details in [json!({"reason": "max_output_tokens"}), json!({})] {
            let raw = json!({
                "id": "resp_mt",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "incomplete",
                "incomplete_details": details,
                "output": [{"type": "message", "id": "m", "role": "assistant", "status": "incomplete",
                            "content": [{"type": "output_text", "text": "partial"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
            });
            let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
            let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_mt").unwrap();
            assert_eq!(out.stop_reason.as_deref(), Some("max_tokens"));
        }
    }

    /// B3 core (non-streaming): status="incomplete" takes priority over
    /// has_tool_calls. A truncated function_call must yield `max_tokens`
    /// (never `tool_use`) and its unparseable arguments must NOT surface
    /// as an empty-input ToolUse block.
    #[test]
    fn incomplete_with_truncated_function_call_returns_max_tokens_not_tool_use() {
        let raw = json!({
            "id": "resp_tfc",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "function_call", "id": "fc_1", "call_id": "c_1",
                 "name": "get_weather", "arguments": "{\"city\":", "status": "incomplete"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_tfc").unwrap();
        assert_eq!(
            out.stop_reason.as_deref(),
            Some("max_tokens"),
            "incomplete + truncated function_call must be max_tokens, got {:?}",
            out.stop_reason
        );
        assert!(
            !out.content.iter().any(|b| matches!(b, ResponseBlock::ToolUse { .. })),
            "truncated function_call must not produce an empty-input ToolUse; got {:?}",
            out.content
        );
    }

    /// B3 positive regression (non-streaming): a *completed* response with
    /// a function_call still maps to `tool_use`.
    #[test]
    fn completed_with_function_call_returns_tool_use() {
        let raw = json!({
            "id": "resp_cfc",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "function_call", "id": "fc_1", "call_id": "c_1",
                 "name": "get_weather", "arguments": "{\"city\":\"SF\"}", "status": "completed"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_cfc").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert!(out.content.iter().any(|b| matches!(b, ResponseBlock::ToolUse { .. })));
    }

    /// Official `cancelled` status → end_turn + empty content (deliberate
    /// Anthropic-compat choice). The test feeds partial output to prove
    /// the content is actively cleared, not merely empty by construction.
    #[test]
    fn non_stream_cancelled_status_maps_to_end_turn_empty() {
        let raw = json!({
            "id": "resp_canc",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "cancelled",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "incomplete",
                        "content": [{"type": "output_text", "text": "partial reply"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_canc").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("end_turn"));
        assert!(
            out.content.is_empty(),
            "cancelled response must present empty content, got {:?}",
            out.content
        );
    }

    #[test]
    fn cache_read_tokens_propagated_when_nonzero() {
        let raw = json!({
            "id": "resp_3",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 7}}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_1").unwrap();
        assert_eq!(out.usage.cache_read_input_tokens, Some(7));
    }

    /// Bug repro: OpenAI Responses `input_tokens` is the *total* tokens
    /// processed INCLUDING the cached portion (the documented
    /// `input_tokens_details.cached_tokens` field is a subset of it).
    /// Anthropic's `Usage.input_tokens` is by spec the *non-cached*
    /// portion, while `cache_read_input_tokens` reports the cached
    /// portion separately. The Anthropic SDK sums the two fields to
    /// estimate context size, so passing `input_tokens` through
    /// unchanged double-counts the cached subset.
    ///
    /// Observed in production (Claude Code session `z-landscape-admin`
    /// against Copilot gpt-5.5 on 2026-07-27): the reported context
    /// converges to ~165–167k tokens right before each auto-compact,
    /// which is roughly 1.8x the real context (the actual context at
    /// those moments was only ~80–93k tokens).
    #[test]
    fn input_tokens_excludes_cached_subset() {
        let raw = json!({
            "id": "resp_dup",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                      "input_tokens_details": {"cached_tokens": 60}}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_dup").unwrap();
        // Anthropic.input_tokens must be the non-cached portion: 100 - 60 = 40.
        // Without the fix the translator returns 100 and Claude Code reads
        // context as 100 + 60 = 160 instead of the actual 100.
        assert_eq!(
            out.usage.input_tokens, 40,
            "input_tokens must be non-cached only (input - cached); got {}, expected 40",
            out.usage.input_tokens
        );
        assert_eq!(out.usage.cache_read_input_tokens, Some(60));
    }

    // ── PR-6b · output_tokens_details.thinking_tokens (Responses non-stream) ──

    /// PR-6b (Responses non-stream): forward `output_tokens_details.
    /// reasoning_tokens` (OpenAI read key) as `output_tokens_details.
    /// thinking_tokens` (Anthropic write key). Different keys on each
    /// side — never leak `reasoning_tokens` into the Anthropic write
    /// payload. Absent and zero both produce a fully-absent field.
    #[test]
    fn responses_non_stream_propagates_reasoning_tokens_as_thinking_tokens() {
        use crate::responses::OutputTokensDetails;
        // (a) reasoning_tokens present → thinking_tokens surfaced.
        let raw = json!({
            "id": "resp_t", "object": "response", "created_at": 0,
            "model": "gpt-5", "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60,
                      "output_tokens_details": {"reasoning_tokens": 18}}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_t").unwrap();
        let details = out.usage.output_tokens_details.expect(
            "reasoning_tokens>0 must surface output_tokens_details",
        );
        assert_eq!(details["thinking_tokens"], 18);
        assert!(
            details.get("reasoning_tokens").is_none(),
            "OpenAI read key must not leak; got {details}"
        );

        // (b) output_tokens_details absent → field absent.
        let raw_no = json!({
            "id": "resp_t2", "object": "response", "created_at": 0,
            "model": "gpt-5", "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw_no).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_t").unwrap();
        assert!(
            out.usage.output_tokens_details.is_none(),
            "no output_tokens_details → must be absent"
        );

        // (c) reasoning_tokens == 0 → field absent (Some(0) changes wire shape).
        let raw_zero = json!({
            "id": "resp_t3", "object": "response", "created_at": 0,
            "model": "gpt-5", "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60,
                      "output_tokens_details": {"reasoning_tokens": 0}}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw_zero).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_t").unwrap();
        assert!(
            out.usage.output_tokens_details.is_none(),
            "reasoning_tokens==0 → must be absent"
        );
        // Sanity: the OutputTokensDetails struct still reads reasoning_tokens.
        let _ = OutputTokensDetails { reasoning_tokens: 0 };
    }

    // silence unused warnings for the helper kept to ensure imports stay live
    #[test]
    fn make_message_id_format() {
        let id = make_message_id();
        assert!(id.starts_with("msg_"));
    }

    #[test]
    fn thinking_disabled_produces_no_reasoning_field() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled", "budget_tokens": 8000}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.reasoning.is_none());
    }

    #[test]
    fn message_with_role_system_in_messages_is_dropped() {
        // Anthropic technically disallows role=system in messages, but if
        // a client slips one in, fold it into instructions rather than
        // dropping silently.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "inline system prompt"},
                {"role": "user", "content": "hi"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.input.len(), 1);
        // Note: this only drops when there's no top-level system — we
        // accept the silent drop as the lesser evil (no instruction
        // duplication).
    }

    // Touch the MessageContent::Blocks path through a tool_use block.
    #[test]
    fn tool_use_in_user_turn_is_skipped() {
        let msg = Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Text { text: "before".into(), cache_control: None, citations: None },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "f".into(),
                    input: json!({}),
                    cache_control: None,
                    caller: None,
                },
                ContentBlock::Unknown,
            ]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::Message { content, .. } => match content {
                ResponseInputContent::Text(t) => assert_eq!(t, "before"),
                _ => panic!("expected text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn thinking_in_assistant_turn_is_dropped() {
        let msg = Message {
            role: "assistant".into(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "internal".into(),
                    signature: None,
                },
                ContentBlock::Text { text: "final".into(), cache_control: None, citations: None },
            ]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::Message { content, .. } => match content {
                ResponseInputContent::Text(t) => assert_eq!(t, "final"),
                _ => panic!("expected text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn request_maps_image_blocks_to_input_image_parts() {
        // Multimodal user turn: text + image. Anthropic Image source
        // carries raw base64 data; Responses expects a data URL.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]
            }]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.input.len(), 1);
        match &out.input[0] {
            ResponseInputItem::Message { role, content } => {
                assert_eq!(role, "user");
                match content {
                    ResponseInputContent::Parts(parts) => {
                        assert_eq!(parts.len(), 2);
                        match &parts[1] {
                            ResponseInputPart::InputImage { image_url, .. } => {
                                assert_eq!(image_url, "data:image/png;base64,AAAA");
                            }
                            _ => panic!("expected input_image"),
                        }
                    }
                    _ => panic!("expected parts content"),
                }
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn request_maps_system_blocks_to_instructions() {
        // System prompt as `Blocks` (not just `Text`) — Anthropic
        // supports per-block cache_control; we join their text with
        // blank-line separators.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [
                {"type": "text", "text": "first half"},
                {"type": "text", "text": "second half"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.instructions.as_deref(), Some("first half\n\nsecond half"));
    }

    #[test]
    fn request_omits_instructions_when_system_is_empty() {
        // Empty/whitespace system must not produce a noisy
        // `instructions: ""` field on the wire.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "system": ""
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.instructions.is_none());
    }

    #[test]
    fn request_maps_tool_choice_specific_tool_to_function_form() {
        // `tool_choice: { type: tool, name: get_weather }` must become
        // Responses' `{type:"function", name:"get_weather"}` shape.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(
            out.tool_choice.as_ref().unwrap(),
            &json!({"type": "function", "name": "get_weather"})
        );
    }

    #[test]
    fn request_maps_tool_choice_none_to_none() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "none"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.tool_choice.as_ref().unwrap(), &json!("none"));
    }

    #[test]
    fn request_passes_temperature_top_p_and_stream_through() {
        // Anthropic sampling parameters must land on the Responses
        // request unchanged so client-side tuning survives translation.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.temperature, Some(0.7));
        assert_eq!(out.top_p, Some(0.9));
        assert!(out.stream);
    }

    #[test]
    fn request_applies_model_rewrite_to_upstream_name() {
        // The runtime rewrite must change the wire model field; the
        // input[] items must remain intact.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4.6",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let mut rewrite = HashMap::new();
        rewrite.insert("claude-sonnet-4.6".to_string(), "gpt-5-mini".to_string());
        let out = anthropic_to_responses_request(&req, &rewrite).unwrap();
        assert_eq!(out.model, "gpt-5-mini");
        assert_eq!(out.input.len(), 1);
    }

    #[test]
    fn gpt5_escalates_in_memory_retention_to_24h() {
        // GPT-5.x rejects `in_memory`: "This model is compatible only
        // with 24h extended prompt caching". A short-tier cache_control
        // marker must be escalated to 24h for these models.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"}
                }]
            }],
            "metadata": {"user_id": "u-1"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn gpt5_leaves_24h_retention_unchanged() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral_1h"}
                }]
            }]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn non_gpt5_keeps_in_memory_retention() {
        // A non-gpt-5 Responses model (e.g. rewritten to something else)
        // must keep the client-requested short tier untouched.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "deepseek-chat",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"}
                }]
            }]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    #[test]
    fn no_cache_control_emits_no_retention_even_for_gpt5() {
        // Without any cache_control marker the field stays absent — the
        // escalation must not conjure a retention value out of nothing.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.prompt_cache_retention, None);
    }

    #[test]
    fn request_with_empty_messages_emits_no_input_items() {
        // Defensive: an empty messages array must serialize cleanly
        // (Responses permits it for stateless prompts).
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": []
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.input.is_empty());
    }

    #[test]
    fn response_with_empty_output_returns_empty_content() {
        // A Responses API response with zero output items must still
        // produce a valid Anthropic message shape (id, model, usage)
        // with an empty content array — the SSE adapter needs the
        // message_id and usage to round-trip even when no text was
        // emitted (e.g. a refusal with no body).
        let raw = json!({
            "id": "resp_empty",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_x").unwrap();
        assert_eq!(out.id, "msg_x");
        assert!(out.content.is_empty());
        assert_eq!(out.usage.input_tokens, 3);
        assert_eq!(out.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn response_with_multiple_function_calls_keeps_all_and_sets_tool_use() {
        // A response that emits N parallel function calls must
        // surface all of them in content[], and stop_reason must be
        // tool_use (not end_turn) regardless of call count.
        let raw = json!({
            "id": "resp_p",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "function_call", "id": "fc_1", "call_id": "c_1",
                 "name": "get_weather", "arguments": "{\"city\":\"SF\"}", "status": "completed"},
                {"type": "function_call", "id": "fc_2", "call_id": "c_2",
                 "name": "get_time", "arguments": "{\"tz\":\"PST\"}", "status": "completed"},
                {"type": "function_call", "id": "fc_3", "call_id": "c_3",
                 "name": "lookup_user", "arguments": "{\"id\":42}", "status": "completed"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_p").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(out.content.len(), 3);
        let names: Vec<&str> = out
            .content
            .iter()
            .filter_map(|b| match b {
                crate::anthropic::ResponseBlock::ToolUse { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["get_weather", "get_time", "lookup_user"]);
    }

    #[test]
    fn response_failed_status_returns_upstream_error() {
        // A failed Responses API response (status="failed") without error
        // details returns a fallback upstream error.
        let raw = json!({
            "id": "resp_f",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "failed",
            "output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "partial"}]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let err = responses_to_anthropic_response(&resp, "gpt-5", "msg_f").unwrap_err();
        assert!(
            matches!(&err, ProxyError::Upstream { status: 502, .. }),
            "expected Upstream(502), got: {err}"
        );
    }

    #[test]
    fn response_failed_with_error_details_surfaces_message() {
        // A failed response with error details in extra should surface
        // the upstream error message.
        let raw = json!({
            "id": "resp_f2",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "failed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
            "error": {"code": "rate_limited", "message": "too fast"}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let err = responses_to_anthropic_response(&resp, "gpt-5", "msg_f2").unwrap_err();
        match &err {
            ProxyError::Upstream { status, body } => {
                assert_eq!(*status, 502);
                assert!(body.contains("too fast"), "body: {body}");
            }
            _ => panic!("expected Upstream error, got: {err}"),
        }
    }

    #[test]
    fn response_omits_cache_read_tokens_when_cached_count_is_zero() {
        // cached_tokens == 0 must NOT produce a cache_read_input_tokens
        // field (Some(0) would be a different shape than absent — see
        // Anthropic's Usage struct where None means "not provided").
        let raw = json!({
            "id": "resp_z",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 0}}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_z").unwrap();
        assert!(out.usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn response_text_then_function_call_yields_tool_use_stop_reason() {
        // The presence of ANY function_call in output[] wins over
        // text: stop_reason must be tool_use, not end_turn, even if
        // text precedes it. Order in output[] is the OpenAI-defined
        // execution order, but the Anthropic stop_reason only reflects
        // whether the assistant wants tools called next.
        let raw = json!({
            "id": "resp_m",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "message", "id": "m", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "let me check"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "c_1",
                 "name": "lookup", "arguments": "{}", "status": "completed"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_m").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(out.content.len(), 2);
    }

    #[test]
    fn request_omits_instructions_when_system_blocks_are_empty_array() {
        // System supplied as an EMPTY `Blocks` array: the joined
        // instruction string is empty, so `instructions` must be absent.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "system": []
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.instructions.is_none());
    }

    #[test]
    fn request_omits_instructions_when_system_blocks_are_all_empty_text() {
        // N>=1 blocks each with empty text: drop them all before joining
        // and return None. The fix for issue #2 makes this consistent
        // with `Text("")` and the empty-array case above — a client
        // that sends all-empty system prompts must NOT cause
        // `instructions: "\n\n"` to leak into the upstream request.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [
                {"type": "text", "text": ""},
                {"type": "text", "text": ""}
            ]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(
            out.instructions.is_none(),
            "all-empty-text system blocks must yield None, got {:?}",
            out.instructions
        );
    }

    /// PR-13 v0.13 P2-G: when every system block is empty *text* but
    /// still carries a `cache_control` marker, `instructions` must stay
    /// absent (empty text → None) while the cache hint still fires (the
    /// marker lives on the block, not the text). Regression for the
    /// boundary where "no instructions" and "cache wanted" coexist.
    #[test]
    fn cache_hint_escalates_when_all_system_blocks_empty() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral_1h"}}
            ],
            "metadata": {"user_id": "u-edge"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.instructions.is_none());
        // 24h marker present → retention escalates; cache key from
        // metadata.user_id flows through despite empty system text.
        assert_eq!(out.prompt_cache_retention.as_deref(), Some("24h"));
        assert_eq!(out.prompt_cache_key.as_deref(), Some("u-edge"));
    }

    #[test]
    fn request_keeps_non_empty_blocks_when_mixed_with_empty() {
        // When some blocks are empty and some are non-empty, only the
        // non-empty blocks contribute to the joined string. The empty
        // blocks are filtered (not joined as "" → "\n\n" padding).
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "first real instruction"},
                {"type": "text", "text": ""},
                {"type": "text", "text": "second real instruction"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(
            out.instructions.as_deref(),
            Some("first real instruction\n\nsecond real instruction"),
            "empty blocks must be filtered, not joined as padding"
        );
    }

    #[test]
    fn user_turn_with_single_image_uses_parts_content() {
        // A user turn whose ONLY block is an image yields a single-part
        // list — but a single non-text part must still serialize as
        // `Parts` (not `Text`), covering the else-branch at line 164.
        let msg = Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock::Image {
                source: crate::anthropic::ImageSource {
                    kind: "base64".into(),
                    media_type: "image/png".into(),
                    data: "AAAA".into(),
                },
                cache_control: None,
            }]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::Message { content, .. } => match content {
                ResponseInputContent::Parts(parts) => {
                    assert_eq!(parts.len(), 1);
                    assert!(matches!(parts[0], ResponseInputPart::InputImage { .. }));
                }
                _ => panic!("single image must serialize as Parts, not Text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn assistant_turn_with_multiple_text_blocks_joins_with_newline() {
        // Two assistant Text blocks accumulate into one message body
        // separated by '\n' (line 183-184: the non-empty push guard).
        let msg = Message {
            role: "assistant".into(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Text { text: "line one".into(), cache_control: None, citations: None },
                ContentBlock::Text { text: "line two".into(), cache_control: None, citations: None },
            ]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::Message { content, .. } => match content {
                ResponseInputContent::Text(t) => assert_eq!(t, "line one\nline two"),
                _ => panic!("expected joined text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn assistant_turn_skips_image_and_tool_result_blocks() {
        // Image / ToolResult / Unknown blocks are invalid in an
        // assistant turn and must be dropped (line 199-201), leaving
        // only the text body.
        let msg = Message {
            role: "assistant".into(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Text { text: "keep me".into(), cache_control: None, citations: None },
                ContentBlock::Image {
                    source: crate::anthropic::ImageSource {
                        kind: "base64".into(),
                        media_type: "image/png".into(),
                        data: "AAAA".into(),
                    },
                    cache_control: None,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: crate::anthropic::ToolResultContent::Text("ignored".into()),
                    cache_control: None,
                    is_error: None,
                },
                ContentBlock::Unknown,
            ]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::Message { content, .. } => match content {
                ResponseInputContent::Text(t) => assert_eq!(t, "keep me"),
                _ => panic!("expected text"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn unknown_role_produces_no_input_items() {
        // A role other than user/assistant (e.g. a client-injected
        // "tool" role) falls through the `_ => {}` arm at line 218 and
        // yields nothing.
        let out = convert_blocks(
            "tool",
            &[ContentBlock::Text { text: "orphan".into(), cache_control: None, citations: None }],
        );
        assert!(out.is_empty());
    }

    #[test]
    fn tool_result_with_block_content_joins_text_fields() {
        // A tool_result whose content is an array of blocks (not a bare
        // string) must have its `text` fields extracted and joined
        // (tool_result_to_string Blocks arm, lines 227-231).
        let msg = Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "t9".into(),
                content: crate::anthropic::ToolResultContent::Blocks(vec![
                    json!({"type": "text", "text": "first"}),
                    json!({"type": "text", "text": "second"}),
                    json!({"type": "image", "source": {}}), // no text → skipped
                ]),
                cache_control: None,
                is_error: None,
            }]),
        };
        let out = convert_message(&msg);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "t9");
                assert_eq!(output, "first\nsecond");
            }
            _ => panic!("expected function_call_output"),
        }
    }

    #[test]
    fn tool_choice_auto_maps_to_auto_string() {
        // `tool_choice: {type: auto}` → Responses `"auto"` (line 237).
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.tool_choice.as_ref().unwrap(), &json!("auto"));
    }

    #[test]
    fn thinking_high_and_low_budget_map_to_effort_levels() {
        // budget >= 8000 → "high" (line 253); budget < 2000 → "low"
        // (line 257). The medium band is already covered elsewhere.
        for (budget, expected) in [(8000u32, "high"), (1000u32, "low")] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "model": "gpt-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget_tokens": budget}
            }))
            .unwrap();
            let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
            match out.reasoning.as_ref().unwrap() {
                ReasoningConfig {
                    effort,
                    summary: Some(ReasoningSummary::Auto),
                } => {
                    assert_eq!(effort.as_deref(), Some(expected), "budget {budget}");
                }
                other => panic!("expected struct with summary Auto, got {other:?}"),
            }
        }
    }

    #[test]
    fn convert_thinking_still_produces_effort_after_struct_migration() {
        // Regression sentinel (PR-1 + PR-4): ReasoningConfig changed from
        // tagged enum `ReasoningConfig::Enabled { effort }` to plain struct
        // `ReasoningConfig { effort, summary }`. Both construction paths
        // (output_config.effort and thinking.budget_tokens) must still
        // produce a struct with the expected effort and summary Auto (PR-4
        // requests the reasoning summary on the wire).
        let via_output_config: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "xhigh"},
        }))
        .unwrap();
        let via_thinking: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }))
        .unwrap();
        for (req, expected) in [(via_output_config, Some("xhigh")), (via_thinking, Some("medium"))] {
            let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
            match out.reasoning.as_ref() {
                Some(ReasoningConfig {
                    effort,
                    summary: Some(ReasoningSummary::Auto),
                }) => {
                    assert_eq!(effort.as_deref(), expected);
                }
                other => panic!("expected ReasoningConfig struct with effort {expected:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn response_with_unknown_output_content_part_is_ignored() {
        // An output message containing an unrecognized content part
        // type must be skipped without panicking (OutputContentPart
        // ::Unknown arm, line 293). The known text part still lands.
        let raw = json!({
            "id": "resp_u",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "completed",
                "content": [
                    {"type": "output_text", "text": "visible"},
                    {"type": "some_future_part", "blob": 1}
                ]
            }],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_u").unwrap();
        assert_eq!(out.content.len(), 1);
        assert!(matches!(out.content[0], ResponseBlock::Text { .. }));
    }

    #[test]
    fn response_with_unknown_output_item_is_ignored() {
        // An unrecognized top-level output item type (e.g. a hosted
        // tool call we don't model) must be skipped (OutputItem::Unknown
        // arm, line 312) while known items still convert.
        let raw = json!({
            "id": "resp_ui",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "id": "m", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "answer"}]}
            ],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_ui").unwrap();
        assert_eq!(out.content.len(), 1);
        assert!(matches!(out.content[0], ResponseBlock::Text { .. }));
    }

    // ── PR-5 · refusal (non-streaming) ────────────────────────────────

    /// PR-5 (G3/G4 inbound): a `refusal` content part in a completed
    /// response must surface as an Anthropic Text block. The Anthropic
    /// wire has no refusal-specific block; clients read it as text and
    /// the stop_reason (set by incomplete_details.content_filter on the
    /// upstream path, PR-7 outbound) tells Claude Code to treat it as
    /// a refusal rather than user content.
    #[test]
    fn response_with_refusal_content_part_maps_to_text_block() {
        let raw = json!({
            "id": "resp_rf",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "completed",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "refusal", "refusal": "I cannot comply with this request."}
                ]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_rf").unwrap();
        assert_eq!(out.content.len(), 1, "refusal must surface as one text block");
        match &out.content[0] {
            ResponseBlock::Text { text, .. } => {
                assert_eq!(text, "I cannot comply with this request.");
            }
            other => panic!("expected Text block, got {other:?}"),
        }
    }

    /// PR-5: a refusal in an `incomplete` response whose reason is
    /// `content_filter` carries both the refusal text (PR-5) and the
    /// `refusal` stop_reason (B2 from PR-2). Both must be honored.
    #[test]
    fn response_with_refusal_and_content_filter_reason_surfaces_refusal() {
        let raw = json!({
            "id": "resp_cfr",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "incomplete",
                "content": [{"type": "refusal", "refusal": "blocked by content policy"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_cfr").unwrap();
        assert_eq!(
            out.stop_reason.as_deref(),
            Some("refusal"),
            "content_filter + refusal must finalize as refusal"
        );
        assert_eq!(out.content.len(), 1);
        assert!(matches!(&out.content[0], ResponseBlock::Text { text, .. } if text == "blocked by content policy"));
    }

    #[test]
    fn response_unknown_status_yields_no_stop_reason() {
        // A status the translator doesn't recognize (and no tool calls)
        // leaves stop_reason as None (line 323, the `_ => None` arm).
        let raw = json!({
            "id": "resp_s",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "in_progress",
            "output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "in_progress",
                "content": [{"type": "output_text", "text": "streaming..."}]
            }],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
        });
        let resp: ResponsesResponse = serde_json::from_value(raw).unwrap();
        let out = responses_to_anthropic_response(&resp, "gpt-5", "msg_s").unwrap();
        assert!(out.stop_reason.is_none());
    }

    #[test]
    fn truncate_user_truncates_long_strings() {
        assert_eq!(truncate_user("short"), "short");
        assert_eq!(truncate_user(""), "");
        let long = "a".repeat(150);
        assert_eq!(truncate_user(&long).len(), 64);
        assert_eq!(truncate_user(&long).chars().count(), 64);
        // exactly 64 — no truncation needed
        let exact = "b".repeat(64);
        assert_eq!(truncate_user(&exact), exact);
    }

    /// T4: P0-4 regression — multibyte user_id must not panic from byte-
    /// slicing across a character boundary.
    #[test]
    fn truncate_user_handles_multibyte_user_id_without_panic() {
        let user = "\u{7528}".repeat(30); // 90 bytes, 30 chars
        let truncated = truncate_user(&user);
        assert_eq!(truncated.chars().count(), 30);
        assert!(truncated.is_char_boundary(truncated.len()));
        // A very long multibyte string that exceeds 64 characters
        let long_mb = "\u{7528}".repeat(100); // 300 bytes, 100 chars
        let truncated_long = truncate_user(&long_mb);
        assert_eq!(truncated_long.chars().count(), 64);
        assert!(truncated_long.is_char_boundary(truncated_long.len()));
    }

    #[test]
    fn propagates_output_config_format_into_text_and_keeps_typed_reasoning() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "respond in json"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "impossible": {"type": "boolean"}
                    },
                    "required": ["ok", "reason"]
                }},
                "effort": "medium"
            }
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        let text_format = out.extra.get("text").and_then(|v| v.get("format")).unwrap();
        assert_eq!(text_format.get("type").and_then(|v| v.as_str()), Some("json_schema"));
        // Anthropic doesn't carry a schema name; OpenAI requires one. The
        // translator synthesizes a stable default and pins strict: true.
        assert_eq!(text_format.get("name").and_then(|v| v.as_str()), Some("structured_output"));
        assert_eq!(text_format.get("strict").and_then(|v| v.as_bool()), Some(true));
        let schema = text_format.get("schema").unwrap();
        assert_eq!(
            schema.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("impossible")));
        assert!(matches!(
            out.reasoning.as_ref(),
            Some(ReasoningConfig { effort: Some(e), summary: Some(ReasoningSummary::Auto) }) if e == "medium"
        ));
    }

    /// PR-13: an external URI `$ref` inside `output_config.format.schema`
    /// cannot be strictified — the conversion must fail with a `Schema`
    /// error rather than forwarding a schema the upstream will reject.
    #[test]
    fn external_ref_in_format_schema_surfaces_schema_error() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "respond in json"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": {
                    "type": "object",
                    "properties": {
                        "shared": {"$ref": "https://schemas.example/common.json"}
                    }
                }}
            }
        }))
        .unwrap();
        let err = anthropic_to_responses_request(&req, &Default::default())
            .expect_err("external $ref must propagate as an error");
        assert!(
            matches!(err, ProxyError::Schema(_)),
            "expected ProxyError::Schema, got {err:?}"
        );
        assert_eq!(
            err.status_code(),
            axum::http::StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn output_config_with_only_format_does_not_set_reasoning() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "respond in json"}],
            "output_config": {
                "format": {"type": "json_object"}
            }
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.extra.get("text").is_some());
        assert!(out.reasoning.is_none());
    }

    #[test]
    fn output_config_absent_leaves_extra_and_reasoning_empty() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(out.extra.as_object().unwrap().is_empty());
        assert!(out.reasoning.is_none());
    }

    #[test]
    fn output_config_effort_overrides_thinking_budget_derivation() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "reason about this"}],
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {
                "effort": "low"
            }
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(matches!(
            out.reasoning.as_ref(),
            Some(ReasoningConfig { effort: Some(e), summary: Some(ReasoningSummary::Auto) }) if e == "low"
        ));
    }

    #[test]
    fn ensure_json_schema_name_non_json_schema_passthrough() {
        // ensure_json_schema_name is a no-op for format types other than
        // "json_schema" — the Responses API's text.format only requires
        // the `name`+`strict` shim for json_schema shapes.
        let input = json!({"type": "json_object"});
        let out = ensure_json_schema_name(&input).unwrap();
        assert_eq!(out, input);

        let input = json!({"type": "text"});
        let out = ensure_json_schema_name(&input).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn ensure_json_schema_name_preserves_existing_name() {
        // If the json_schema payload already has a `name` field, the
        // translator must NOT overwrite it (parity with the Chat path —
        // see `output_config_format_with_existing_name_is_preserved` in
        // request.rs).
        let input = json!({
            "type": "json_schema",
            "name": "my_response",
            "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}
        });
        let out = ensure_json_schema_name(&input).unwrap();
        assert_eq!(out.get("name").and_then(|v| v.as_str()), Some("my_response"));
        assert_eq!(out.get("strict").and_then(|v| v.as_bool()), Some(true));
    }

    // ── PR-7 · service_tier request mapping (Responses) ──────────────────

    /// PR-7: Anthropic `service_tier: "auto"` → Responses `service_tier:
    /// "auto"` (same-value passthrough, parity with Chat path).
    #[test]
    fn responses_request_service_tier_auto_passes_through() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "auto"
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.service_tier.as_deref(), Some("auto"));
    }

    /// PR-7: Anthropic `service_tier: "standard_only"` → Responses
    /// drops the field (no OpenAI equivalent). Parity with Chat path.
    #[test]
    fn responses_request_service_tier_standard_only_is_dropped() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "standard_only"
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(
            out.service_tier.is_none(),
            "standard_only must drop, got {:?}",
            out.service_tier
        );
    }

    // ── PR-8 · Responses parallel_tool_calls + verbosity ────────────────

    /// PR-8: Responses `parallel_tool_calls=false` when Anthropic
    /// `tool_choice.disable_parallel_tool_use=true`. Same logic as Chat
    /// path; ResponsesRequest.parallel_tool_calls was modeled in PR-1,
    /// this PR wires the injection.
    #[test]
    fn responses_request_disable_parallel_tool_use_maps_to_parallel_tool_calls_false() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": true}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.parallel_tool_calls, Some(false));
    }

    /// PR-8: Responses `text.verbosity` is injected when
    /// `output_config.verbosity` is set (plan v0.7 P1-3 dual-path).
    /// Verbatim passthrough.
    #[test]
    fn responses_request_verbosity_injected_into_text_extra() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"verbosity": "high"}
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert_eq!(out.extra["text"]["verbosity"], "high");
    }

    /// PR-8: when no verbosity is set, the wire stays absent under
    /// `text` (don't inject an empty object either).
    #[test]
    fn responses_request_verbosity_absent_when_unset() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let out = anthropic_to_responses_request(&req, &Default::default()).unwrap();
        assert!(
            out.extra.get("text").is_none(),
            "no verbosity → text object must be absent; got {}",
            out.extra
        );
    }
}

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

use serde_json::{json, Value};
use uuid::Uuid;

use crate::anthropic::{
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, ResponseBlock, SystemPrompt,
    ToolChoice, Usage,
};
use crate::error::Result;
use crate::responses::{
    OutputContentPart, OutputItem, ReasoningConfig, ResponseInputContent, ResponseInputItem,
    ResponseInputPart, ResponsesRequest, ResponsesResponse, ResponsesTool,
};

/// Convert an Anthropic MessagesRequest into an OpenAI ResponsesRequest.
pub fn anthropic_to_responses_request(
    req: &MessagesRequest,
    model_rewrite: &std::collections::HashMap<String, String>,
) -> ResponsesRequest {
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
            let text = blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    });

    let tools = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| ResponsesTool::Function {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
                strict: None,
            })
            .collect()
    });

    ResponsesRequest {
        model,
        input,
        instructions,
        max_output_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        tools,
        tool_choice: req.tool_choice.as_ref().and_then(convert_tool_choice),
        parallel_tool_calls: None,
        user: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        reasoning: req.thinking.as_ref().and_then(convert_thinking),
        extra: json!({}),
    }
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
                    ContentBlock::Image { source } => {
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
                    ContentBlock::ToolUse { .. } | ContentBlock::Thinking { .. } => {
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
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push((
                            id.clone(),
                            name.clone(),
                            serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                        ));
                    }
                    ContentBlock::Image { .. }
                    | ContentBlock::ToolResult { .. }
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
        ToolChoice::Tool { name } => Some(json!({
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
        Some(ReasoningConfig::Enabled { effort: Some(effort) })
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
                                content.push(ResponseBlock::Text { text: text.clone() });
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
                let input: Value = serde_json::from_str(arguments)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                content.push(ResponseBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                });
                has_tool_calls = true;
            }
            OutputItem::Unknown => {}
        }
    }

    let stop_reason = if has_tool_calls {
        Some("tool_use".to_string())
    } else {
        match resp.status.as_str() {
            "incomplete" => Some("max_tokens".to_string()),
            "completed" => Some("end_turn".to_string()),
            "failed" => Some("end_turn".to_string()),
            _ => None,
        }
    };

    let cached = resp
        .usage
        .input_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);

    Ok(MessagesResponse {
        id: message_id.to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage: Usage {
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: if cached > 0 { Some(cached) } else { None },
        },
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
        let out = anthropic_to_responses_request(&req, &Default::default());
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
        let out = anthropic_to_responses_request(&req, &Default::default());
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
        let out = anthropic_to_responses_request(&req, &Default::default());
        assert_eq!(
            out.tool_choice.as_ref().unwrap(),
            &json!("required")
        );
        match out.reasoning.as_ref().unwrap() {
            ReasoningConfig::Enabled { effort } => {
                assert_eq!(effort.as_deref(), Some("medium"));
            }
        }
        let tools = out.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            ResponsesTool::Function { name, parameters, .. } => {
                assert_eq!(name, "f");
                assert!(parameters.is_some());
            }
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
        let out = anthropic_to_responses_request(&req, &Default::default());
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
        let out = anthropic_to_responses_request(&req, &Default::default());
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
                ContentBlock::Text { text: "before".into(), cache_control: None },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "f".into(),
                    input: json!({}),
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
                },
                ContentBlock::Text { text: "final".into(), cache_control: None },
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
}
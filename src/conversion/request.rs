//! Anthropic Messages → OpenAI Chat Completions request conversion.
//!
//! Reference: copilot-api-py/src/routes/messages/non_stream_translation.py:36-286

use serde_json::{json, Value};

use crate::anthropic::{
    ContentBlock, Message, MessageContent, MessagesRequest, SystemPrompt, ToolChoice,
};
use crate::conversion::derive_cache_hints;
use crate::openai::{
    ChatMessage, ChatRequest, ChatTool, ContentPart, FunctionDef, UserContent,
};

/// Convert an Anthropic MessagesRequest into an OpenAI ChatRequest.
///
/// The `model_rewrite` table lets providers map Anthropic model names (e.g.
/// `claude-sonnet-4-5`) to whatever the underlying provider calls them.
pub fn anthropic_to_openai_request(
    req: &MessagesRequest,
    model_rewrite: &std::collections::HashMap<String, String>,
) -> ChatRequest {
    let model = model_rewrite
        .get(&req.model)
        .cloned()
        .unwrap_or_else(|| strip_date_suffix(&req.model));

    let mut messages = Vec::new();

    // System prompt → first message(s).
    if let Some(sys) = &req.system {
        let text = system_to_text(sys);
        if !text.is_empty() {
            messages.push(ChatMessage::System {
                content: text,
                name: None,
            });
        }
    }

    // Walk each Anthropic message.
    for m in &req.messages {
        messages.extend(convert_message(m));
    }

    let stream_options = if req.stream {
        Some(crate::openai::StreamOptions {
            include_usage: true,
        })
    } else {
        None
    };

    let hints = derive_cache_hints(req);

    ChatRequest {
        model,
        messages,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences.clone(),
        stream: req.stream,
        stream_options,
        tools: req.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| ChatTool {
                    kind: "function".to_string(),
                    function: FunctionDef {
                        name: t.name.clone(),
                        description: t.description.clone().unwrap_or_default(),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        }),
        tool_choice: req.tool_choice.as_ref().map(convert_tool_choice),
        user: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        reasoning_effort: extract_reasoning_effort(req),
        prompt_cache_key: hints.prompt_cache_key,
        prompt_cache_retention: hints.prompt_cache_retention,
        extra: json!({}),
    }
}

fn system_to_text(sys: &SystemPrompt) -> String {
    match sys {
        SystemPrompt::Text(s) => s.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

fn convert_message(m: &Message) -> Vec<ChatMessage> {
    match &m.content {
        MessageContent::Text(s) => vec![simple_text_message(&m.role, s.clone())],
        MessageContent::Blocks(blocks) => convert_blocks(&m.role, blocks),
    }
}

fn simple_text_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: UserContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(text),
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: text,
            name: None,
        },
        _ => ChatMessage::User {
            content: UserContent::Text(text),
            name: None,
        },
    }
}

fn convert_blocks(role: &str, blocks: &[ContentBlock]) -> Vec<ChatMessage> {
    // Anthropic allows tool_use and tool_result blocks within user/assistant
    // turns. OpenAI expects tool calls in the assistant message and tool
    // results as separate role=tool messages. We split as needed.
    let mut out = Vec::new();

    match role {
        "user" => {
            let mut text_parts: Vec<ContentPart> = Vec::new();
            let mut tool_results: Vec<ChatMessage> = Vec::new();
            for b in blocks {
                match b {
                    ContentBlock::Text { text, .. } => {
                        text_parts.push(ContentPart::Text { text: text.clone() });
                    }
                    ContentBlock::Image { source } => {
                        let url = format!(
                            "data:{};base64,{}",
                            source.media_type, source.data
                        );
                        text_parts.push(ContentPart::ImageUrl {
                            image_url: crate::openai::ImageUrl {
                                url,
                                detail: None,
                            },
                        });
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let text = tool_result_to_text(content);
                        tool_results.push(ChatMessage::Tool {
                            content: text,
                            tool_call_id: tool_use_id.clone(),
                        });
                    }
                    ContentBlock::ToolUse { .. } | ContentBlock::Thinking { .. } => {
                        // Skip — these only make sense in assistant turns.
                    }
                    ContentBlock::Unknown => {}
                }
            }
            // Tool results come first (matching OpenAI's expected order).
            out.extend(tool_results);
            if !text_parts.is_empty() {
                let content = if text_parts.len() == 1 {
                    if let ContentPart::Text { text } = &text_parts[0] {
                        UserContent::Text(text.clone())
                    } else {
                        UserContent::Parts(text_parts)
                    }
                } else {
                    UserContent::Parts(text_parts)
                };
                out.push(ChatMessage::User {
                    content,
                    name: None,
                });
            }
        }
        "assistant" => {
            let mut text_acc = String::new();
            let mut reasoning_acc = String::new();
            let mut tool_calls: Vec<crate::openai::ToolCall> = Vec::new();
            for b in blocks {
                match b {
                    ContentBlock::Text { text, .. } => {
                        if !text_acc.is_empty() {
                            text_acc.push('\n');
                        }
                        text_acc.push_str(text);
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        reasoning_acc.push_str(thinking);
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(crate::openai::ToolCall {
                            id: id.clone(),
                            kind: "function".to_string(),
                            function: crate::openai::FunctionCall {
                                name: name.clone(),
                                arguments: serde_json::to_string(input)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        });
                    }
                    ContentBlock::Image { .. }
                    | ContentBlock::ToolResult { .. }
                    | ContentBlock::Unknown => {}
                }
            }
            out.push(ChatMessage::Assistant {
                content: if text_acc.is_empty() {
                    None
                } else {
                    Some(text_acc)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                reasoning_content: if reasoning_acc.is_empty() {
                    None
                } else {
                    Some(reasoning_acc)
                },
            });
        }
        _ => {}
    }

    out
}

fn tool_result_to_text(c: &crate::anthropic::ToolResultContent) -> String {
    match c {
        crate::anthropic::ToolResultContent::Text(s) => s.clone(),
        crate::anthropic::ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn convert_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Any => json!("required"),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
        ToolChoice::None => json!("none"),
    }
}

fn extract_reasoning_effort(req: &MessagesRequest) -> Option<String> {
    let t = req.thinking.as_ref()?;
    if t.kind == "enabled" {
        // Map budget_tokens to a coarse effort hint.
        let budget = t.budget_tokens.unwrap_or(0);
        if budget >= 8000 {
            Some("high".to_string())
        } else if budget >= 2000 {
            Some("medium".to_string())
        } else {
            Some("low".to_string())
        }
    } else {
        None
    }
}

/// Strip trailing `-YYYYMMDD` date suffix from model names.
/// `claude-sonnet-4-20250514` → `claude-sonnet-4`.
pub fn strip_date_suffix(model: &str) -> String {
    // Look for `-` followed by exactly 8 digits at the end.
    if let Some(idx) = model.rfind('-') {
        let tail = &model[idx + 1..];
        if tail.len() == 8 && tail.chars().all(|c| c.is_ascii_digit()) {
            return model[..idx].to_string();
        }
    }
    model.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect_variant;

    #[test]
    fn strip_date_suffix_claude() {
        assert_eq!(strip_date_suffix("claude-sonnet-4-20250514"), "claude-sonnet-4");
        assert_eq!(strip_date_suffix("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(strip_date_suffix("gpt-4"), "gpt-4");
        assert_eq!(strip_date_suffix("claude-sonnet-4-5"), "claude-sonnet-4-5");
    }

    /// The loop in `strip_date_suffix` finds a hyphen and then checks
    /// the trailing 8 chars are all digits. The "found hyphen but tail
    /// is not 8 digits" branch (e.g. `-snapshot`) must fall through and
    /// return the original model untouched — see uncovered region at
    /// `request.rs:292`.
    #[test]
    fn strip_date_suffix_drops_tail_when_hyphen_found_but_tail_is_not_date() {
        assert_eq!(strip_date_suffix("claude-sonnet-4-snapshot"), "claude-sonnet-4-snapshot");
        assert_eq!(strip_date_suffix("model-latest"), "model-latest");
        // Edge case: hyphen followed by a non-digit char (not even 8 chars)
        assert_eq!(strip_date_suffix("m-x"), "m-x");
    }

    #[test]
    fn request_with_text_only() {
        let raw = serde_json::json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(out.model, "claude-sonnet-4-5");
        assert_eq!(out.messages.len(), 1);
        assert!(matches!(out.messages[0], ChatMessage::User { .. }));
    }

    #[test]
    fn request_with_system_and_tool_use() {
        let raw = serde_json::json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "system": "you are a helper",
            "messages": [
                {"role": "user", "content": "what's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "checking..."},
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "72F sunny"}
                ]}
            ],
            "tools": [{"name": "get_weather", "description": "get weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"}
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(out.messages.len(), 4);
        // system, user, assistant, tool
        assert!(matches!(out.messages[0], ChatMessage::System { .. }));
        assert!(matches!(out.messages[1], ChatMessage::User { .. }));
        expect_variant!(&out.messages[2], ChatMessage::Assistant { tool_calls, .. } => {
            assert_eq!(tool_calls.as_ref().unwrap().len(), 1);
            assert_eq!(tool_calls.as_ref().unwrap()[0].function.name, "get_weather");
        });
        assert!(matches!(out.messages[3], ChatMessage::Tool { .. }));
    }

    #[test]
    fn request_with_image() {
        let raw = serde_json::json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what's this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]}]
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default());
        expect_variant!(&out.messages[0], ChatMessage::User { content: UserContent::Parts(parts), .. } => {
            assert_eq!(parts.len(), 2);
        });
    }

    #[test]
    fn model_rewrite_takes_precedence() {
        let raw = serde_json::json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();
        let mut rewrite = std::collections::HashMap::new();
        rewrite.insert("claude-sonnet-4-5".to_string(), "deepseek-chat".to_string());
        let out = anthropic_to_openai_request(&req, &rewrite);
        assert_eq!(out.model, "deepseek-chat");
    }

    #[test]
    fn converts_system_blocks_metadata_parameters_and_tool_choices() {
        let base = serde_json::json!({
            "model": "claude-model",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ],
            "temperature": 0.2,
            "top_p": 0.8,
            "stop_sequences": ["STOP"],
            "metadata": {"user_id": "user-1"},
            "messages": [
                {"role": "system", "content": "inline system"},
                {"role": "other", "content": "fallback user"}
            ],
            "tools": [{"name": "tool", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        });
        let req: MessagesRequest = serde_json::from_value(base.clone()).unwrap();
        let converted = anthropic_to_openai_request(&req, &Default::default());

        assert!(matches!(
            &converted.messages[0],
            ChatMessage::System { content, .. } if content == "first\n\nsecond"
        ));
        assert!(matches!(converted.messages[1], ChatMessage::System { .. }));
        assert!(matches!(converted.messages[2], ChatMessage::User { .. }));
        assert_eq!(converted.temperature, Some(0.2));
        assert_eq!(converted.top_p, Some(0.8));
        assert_eq!(converted.stop.as_ref().unwrap(), &["STOP"]);
        assert_eq!(converted.user.as_deref(), Some("user-1"));
        assert_eq!(converted.tool_choice, Some(json!("required")));
        assert_eq!(converted.tools.as_ref().unwrap()[0].function.description, "");

        let mut named_tool = base.clone();
        named_tool["tool_choice"] = json!({"type": "tool", "name": "tool"});
        let req: MessagesRequest = serde_json::from_value(named_tool).unwrap();
        assert_eq!(
            anthropic_to_openai_request(&req, &Default::default()).tool_choice,
            Some(json!({"type": "function", "function": {"name": "tool"}}))
        );

        let mut none = base;
        none["tool_choice"] = json!({"type": "future_choice"});
        let req: MessagesRequest = serde_json::from_value(none).unwrap();
        assert_eq!(
            anthropic_to_openai_request(&req, &Default::default()).tool_choice,
            Some(json!("none"))
        );
    }

    #[test]
    fn converts_block_edge_cases() {
        let raw = json!({
            "model": "claude-model",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [
                        {"type": "text", "text": "line one"},
                        {"type": "image", "data": "ignored"},
                        {"type": "text", "text": "line two"}
                    ]},
                    {"type": "tool_use", "id": "ignored", "name": "ignored", "input": {}},
                    {"type": "thinking", "thinking": "ignored"},
                    {"type": "future_block"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "reasoning"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}},
                    {"type": "tool_result", "tool_use_id": "ignored", "content": "ignored"},
                    {"type": "future_block"}
                ]},
                {"role": "other", "content": [{"type": "text", "text": "ignored"}]}
            ]
        });
        let req: MessagesRequest = serde_json::from_value(raw).unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());

        assert_eq!(converted.messages.len(), 2);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::Tool { content, tool_call_id }
                if content == "line one\nline two" && tool_call_id == "tool-1"
        ));
        assert!(matches!(
            &converted.messages[1],
            ChatMessage::Assistant {
                content: None,
                tool_calls: None,
                reasoning_content: Some(reasoning)
            } if reasoning == "reasoning"
        ));
    }

    #[test]
    fn maps_reasoning_budget_to_effort() {
        for (kind, budget, expected) in [
            ("enabled", 1000, Some("low")),
            ("enabled", 2000, Some("medium")),
            ("enabled", 8000, Some("high")),
            ("disabled", 8000, None),
        ] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "model": "claude-model",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": kind, "budget_tokens": budget}
            }))
            .unwrap();

            assert_eq!(
                anthropic_to_openai_request(&req, &Default::default())
                    .reasoning_effort
                    .as_deref(),
                expected
            );
        }
    }

    #[test]
    fn converts_assistant_text_message_to_assistant_chat_message() {
        // Assistant messages with a plain string content (not blocks) must
        // produce ChatMessage::Assistant with content set and the optional
        // fields left as None.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [
                {"role": "user", "content": "ask"},
                {"role": "assistant", "content": "answer"}
            ]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 2);
        assert!(matches!(
            &converted.messages[1],
            ChatMessage::Assistant {
                content: Some(text),
                tool_calls: None,
                reasoning_content: None
            } if text == "answer"
        ));
    }

    #[test]
    fn user_text_with_multiple_blocks_uses_parts() {
        // When a user message has more than one text block and no tool results,
        // the converter emits a Parts list (not a single collapsed Text).
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"}
                ]
            }]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 1);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::User { content: UserContent::Parts(_), .. }
        ));
    }

    #[test]
    fn user_text_with_single_image_block_uses_parts() {
        // A single non-text block (image) must still go through the Parts path
        // because the `len == 1 && Text` shortcut only matches text.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
                ]
            }]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 1);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::User { content: UserContent::Parts(_), .. }
        ));
    }

    #[test]
    fn user_text_with_single_text_block_uses_text() {
        // A user message with a single Text block must take the
        // `len == 1 && Text` shortcut (line 170) instead of going through
        // Parts. This is the only way the line 170 branch of the if-let
        // gets exercised.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "single block"}
                ]
            }]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 1);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::User { content: UserContent::Text(t), .. } if t == "single block"
        ));
    }

    #[test]
    fn user_text_with_image_uses_parts() {
        // Mixed text + image content must be serialized as Parts.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "see image"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
                ]
            }]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 1);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::User { content: UserContent::Parts(_), .. }
        ));
    }

    #[test]
    fn assistant_text_accumulator_joins_multiple_blocks_with_newlines() {
        // Multiple text blocks in the assistant role should be joined with
        // a newline, exercising the text_acc.push('\n') branch.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"}
                ]
            }]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());
        assert_eq!(converted.messages.len(), 1);
        assert!(matches!(
            &converted.messages[0],
            ChatMessage::Assistant {
                content: Some(text),
                tool_calls: None,
                reasoning_content: None
            } if text == "first\nsecond"
        ));
    }

    #[test]
    fn empty_system_and_non_date_suffix_are_preserved() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "模型-2025abcd",
            "system": "",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let converted = anthropic_to_openai_request(&req, &Default::default());

        assert_eq!(converted.model, "模型-2025abcd");
        assert_eq!(converted.messages.len(), 1);
    }
}

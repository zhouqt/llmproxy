//! Anthropic Messages → OpenAI Chat Completions request conversion.
//!
//! Reference: copilot-api-py/src/routes/messages/non_stream_translation.py:36-286

use serde_json::{json, Map, Value};

use crate::anthropic::{
    ContentBlock, Message, MessageContent, MessagesRequest, SystemPrompt, ToolChoice,
};
use crate::conversion::derive_cache_hints;
use crate::conversion::util::strictify_schema;
use crate::openai::{
    ChatMessage, ChatRequest, ChatTool, ContentPart, FunctionDef, UserContent,
};

/// Convert an Anthropic MessagesRequest into an OpenAI ChatRequest.
///
/// The `model_rewrite` table lets providers map Anthropic model names (e.g.
/// `claude-sonnet-4-5`) to whatever the underlying provider calls them.
///
/// `reasoning_echo` enables filling missing `reasoning_content` on historical
/// assistant messages for DeepSeek-V4-style upstreams in thinking mode (see
/// the post-build fill below). Default off keeps the wire byte-identical.
pub fn anthropic_to_openai_request(
    req: &MessagesRequest,
    model_rewrite: &std::collections::HashMap<String, String>,
    reasoning_echo: bool,
) -> crate::error::Result<ChatRequest> {
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
            // PR-9: no Anthropic source for include_obfuscation — kept
            // None so the wire stays absent (upstream default false).
            include_obfuscation: None,
        })
    } else {
        None
    };

    let hints = derive_cache_hints(req);

    // GPT-5.x models and o-series reject the short `in_memory` retention
    // tier on the Chat Completions path too. Escalate to `24h` for these
    // models rather than letting the request 400.
    let prompt_cache_retention = hints.prompt_cache_retention.map(|r| {
        if r == "in_memory" && crate::util::gpt5_family(&model) {
            "24h".to_string()
        } else {
            r
        }
    });

    let (max_tokens, max_completion_tokens) = if crate::util::gpt5_family(&model) {
        (None, Some(req.max_tokens))
    } else {
        (Some(req.max_tokens), None)
    };

    let reasoning_effort = req.output_config.as_ref()
        .and_then(|oc| oc.effort.clone())
        .or_else(|| extract_reasoning_effort(req));

    // DeepSeek-V4-style upstreams (opencode_zen in thinking mode) keep
    // thinking active once `reasoning_effort` is set, and then require
    // EVERY historical assistant message to carry a `reasoning_content`
    // field. Cross-model turns — a prior Anthropic thinking turn, a
    // redacted_thinking block, or a plain-text / tool-only assistant turn
    // — leave the field absent, which upstream rejects with
    // `reasoning_content ... must be passed back to the API`. When
    // `reasoning_echo` is enabled for the provider, fill the missing field
    // with `""` — upstreams check field *presence*, not content (openclaw
    // #73417, opencode #24190). Same-model turns keep their real reasoning
    // untouched (already populated by `convert_blocks` above).
    if reasoning_echo && reasoning_effort.is_some() {
        for m in messages.iter_mut() {
            if let ChatMessage::Assistant { reasoning_content, .. } = m {
                if reasoning_content.is_none() {
                    *reasoning_content = Some(String::new());
                }
            }
        }
    }

    Ok(ChatRequest {
        model,
        messages,
        max_tokens,
        max_completion_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences.clone(),
        stream: req.stream,
        stream_options,
        tools: req.tools.as_ref().and_then(|ts| {
            let mut result: Vec<ChatTool> = Vec::new();
            for t in ts {
                // Web search is declared via top-level
                // `web_search_options`, not as a function tool.
                // Strip it here; `extra` carries the options.
                if crate::conversion::util::is_web_search_tool(t) {
                    continue;
                }
                result.push(ChatTool {
                    kind: "function".to_string(),
                    function: FunctionDef {
                        name: t.name.clone(),
                        description: t.description.clone().unwrap_or_default(),
                        parameters: t.input_schema.clone(),
                    },
                });
            }
            // If the only tool was a hosted web_search, emit `None`
            // instead of an empty array. Some strict OpenAI-compat
            // upstreams 400 on `"tools": []`.
            if result.is_empty() { None } else { Some(result) }
        }),
        tool_choice: {
            // If the client forced tool_choice to a hosted tool name,
            // upstream has no function with that name — remap to auto.
            let has_web_search = req.tools.as_ref().map_or(false, |ts| {
                ts.iter().any(crate::conversion::util::is_web_search_tool)
            });
            // When ALL tools are hosted, the tools field collapsed to
            // None above.  Don't emit tool_choice alone — strict
            // OpenAI-compat upstreams reject "tool_choice only supported
            // when tools are enabled".
            let non_hosted_count = req.tools.as_ref().map_or(0, |ts| {
                ts.iter().filter(|t| !crate::conversion::util::is_web_search_tool(t)).count()
            });
            if non_hosted_count == 0 {
                None
            } else if has_web_search
                && matches!(req.tool_choice, Some(ToolChoice::Tool { ref name, .. }) if name == "web_search")
            {
                Some(json!("auto"))
            } else {
                req.tool_choice.as_ref().map(convert_tool_choice)
            }
        },
        user: req
            .metadata
            .as_ref()
            .and_then(|m| m.user_id.as_deref())
            .map(|u| crate::conversion::responses::truncate_user(u)),
        reasoning_effort,
        prompt_cache_key: hints.prompt_cache_key,
        prompt_cache_retention,
        // PR-7: Anthropic `service_tier` → OpenAI `service_tier`.
        // Anthropic enum: `auto` / `standard_only`. OpenAI enum:
        // `auto` / `default` / `flex` / `scale` / `priority` / `fast`.
        // Map:
        //   Anthropic `auto` → OpenAI `auto` (same-value passthrough;
        //     `auto` is a spec-legal OpenAI value, v0.7 correction).
        //   Anthropic `standard_only` → drop — OpenAI has no equivalent
        //     (standard_only is a routing preference; scale/flex/priority
        //     are pricing tiers). Sending it would 400.
        service_tier: req.service_tier.as_deref().and_then(|tier| match tier {
            "auto" => Some("auto".to_string()),
            // "standard_only" → None (drop the field; documented gap).
            _ => None,
        }),
        // PR-8: Anthropic `tool_choice.disable_parallel_tool_use=true`
        // → OpenAI `parallel_tool_calls=false`. Anthropic's
        // ToolChoice::Tool carries the field; absent/false → None
        // (OpenAI's wire default is true, so leaving the field absent
        // is correct — emitting `parallel_tool_calls: true` explicitly
        // would also work but is noisier).
        parallel_tool_calls: match req.tool_choice.as_ref() {
            Some(crate::anthropic::ToolChoice::Tool {
                disable_parallel_tool_use,
                ..
            }) if *disable_parallel_tool_use == Some(true) => Some(false),
            _ => None,
        },
        // PR-8: `safety_identifier` — string passthrough from
        // `metadata.user_id` (Anthropic's user-identity surface; max
        // length 64 per spec).
        safety_identifier: req
            .metadata
            .as_ref()
            .and_then(|m| m.user_id.as_deref())
            .map(|u| crate::conversion::responses::truncate_user(u)),
        // PR-8: `verbosity` — string passthrough from
        // `output_config.verbosity` (Anthropic spec). Same enum on the
        // Responses path via text.verbosity.
        verbosity: req.output_config.as_ref().and_then(|oc| oc.verbosity.clone()),
        // PR-9 P2 fields. Anthropic has no direct equivalent for any of
        // these — pass through only if the client somehow attached
        // them. We don't synthesize them; their defaults are None so the
        // wire stays absent unless a future PR injects from a Claude
        // Code extension surface (currently no source on the Anthropic
        // schema side, hence not wired into request translation yet).
        n: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        prediction: None,
        metadata: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        extra: {
            let mut e = Value::Object(Map::new());
            if let Some(fmt) = req.output_config.as_ref().and_then(|oc| oc.format.as_ref()) {
                e["response_format"] = ensure_chat_json_schema_name(fmt)?;
            }
            // When web search tools are present, inject
            // web_search_options into extra_body. The tool was stripped
            // from tools[] above so it doesn't appear as a function.
            if req.tools.as_ref().map_or(false, |ts| {
                ts.iter().any(crate::conversion::util::is_web_search_tool)
            }) {
                e["web_search_options"] = json!({});
            }
            e
        },
    })
}

/// OpenAI Chat Completions requires `response_format.json_schema.name` on
/// `json_schema` shapes; Anthropic's `output_config.format` is a flat
/// `{type: "json_schema", schema: {...}}` with no `json_schema` wrapper and no
/// `name`. Wrap the schema and synthesize `name` (stable default) plus
/// `strict: true` so the schema constraint round-trips and is enforced.
fn ensure_chat_json_schema_name(
    fmt: &Value,
) -> Result<Value, crate::conversion::util::SchemaError> {
    if fmt.get("type").and_then(|v| v.as_str()) != Some("json_schema") {
        return Ok(fmt.clone());
    }
    let Some(obj) = fmt.as_object() else {
        return Ok(fmt.clone());
    };
    let mut out = obj.clone();

    // If json_schema wrapper already exists, just fill in name/strict if absent.
    if let Some(j) = out.get("json_schema").and_then(|v| v.as_object()).cloned() {
        let mut j = j;
        if let Some(s) = j.get("schema").cloned() {
            let mut s = s;
            strictify_schema(&mut s)?;
            j.insert("schema".to_string(), s);
        }
        j.entry("name".to_string())
            .or_insert(json!("structured_output"));
        j.entry("strict".to_string()).or_insert(json!(true));
        out.insert("json_schema".to_string(), Value::Object(j));
        return Ok(Value::Object(out));
    }

    // Anthropic-style flat: lift `schema` (and any top-level `name`) into a
    // json_schema wrapper. Top-level `name` is honored so a future Anthropic
    // schema-name field isn't silently dropped.
    let schema = out.remove("schema").map(|mut s| {
        strictify_schema(&mut s)?;
        Ok::<Value, crate::conversion::util::SchemaError>(s)
    });
    let top_name = out.remove("name");
    let mut inner = serde_json::Map::new();
    inner.insert(
        "name".to_string(),
        top_name.unwrap_or_else(|| json!("structured_output")),
    );
    inner.insert("strict".to_string(), json!(true));
    if let Some(s) = schema {
        inner.insert("schema".to_string(), s?);
    }
    out.insert("json_schema".to_string(), Value::Object(inner));
    Ok(Value::Object(out))
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
                    ContentBlock::Image { source, .. } => {
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
                    ContentBlock::ToolUse { id, name, input, .. } => {
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
        ToolChoice::Tool { name, .. } => json!({
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
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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
        let out = anthropic_to_openai_request(&req, &rewrite, false).unwrap();
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
        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();

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
            anthropic_to_openai_request(&req, &Default::default(), false).unwrap().tool_choice,
            Some(json!({"type": "function", "function": {"name": "tool"}}))
        );

        let mut none = base;
        none["tool_choice"] = json!({"type": "future_choice"});
        let req: MessagesRequest = serde_json::from_value(none).unwrap();
        assert_eq!(
            anthropic_to_openai_request(&req, &Default::default(), false).unwrap().tool_choice,
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();

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
                anthropic_to_openai_request(&req, &Default::default(), false).unwrap()
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
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

        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();

        assert_eq!(converted.model, "模型-2025abcd");
        assert_eq!(converted.messages.len(), 1);
    }

    /// T13: Chat Completions request for a gpt-5 family model must
    /// escalate `in_memory` prompt-cache retention to `24h` (same as
    /// the Responses path). Non-gpt-5 models must keep `in_memory`.
    #[test]
    fn chat_completions_escalates_in_memory_to_24h_for_gpt5() {
        // gpt-5 with cache_control markers → in_memory escalated to 24h
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
            ]}],
            "metadata": {"user_id": "u-1"}
        }))
        .unwrap();
        let mut rewrite = std::collections::HashMap::new();
        rewrite.insert("claude-sonnet-4-5".to_string(), "gpt-5".to_string());
        let converted = anthropic_to_openai_request(&req, &rewrite, false).unwrap();
        assert_eq!(
            converted.prompt_cache_retention.as_deref(),
            Some("24h"),
            "gpt-5 must escalate in_memory to 24h"
        );

        // gpt-5-mini also escalates
        let mut rewrite2 = std::collections::HashMap::new();
        rewrite2.insert("claude-sonnet-4-5".to_string(), "gpt-5-mini".to_string());
        let converted2 = anthropic_to_openai_request(&req, &rewrite2, false).unwrap();
        assert_eq!(
            converted2.prompt_cache_retention.as_deref(),
            Some("24h"),
            "gpt-5-mini must escalate in_memory to 24h"
        );

        // o4-mini also escalates
        let mut rewrite3 = std::collections::HashMap::new();
        rewrite3.insert("claude-sonnet-4-5".to_string(), "o4-mini".to_string());
        let converted3 = anthropic_to_openai_request(&req, &rewrite3, false).unwrap();
        assert_eq!(
            converted3.prompt_cache_retention.as_deref(),
            Some("24h"),
            "o4-mini must escalate in_memory to 24h"
        );

        // non-gpt-5 model keeps in_memory
        let converted4 = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(
            converted4.prompt_cache_retention.as_deref(),
            Some("in_memory"),
            "non-gpt-5 must keep in_memory"
        );
    }

    /// T11: Chat Completions for a gpt-5 family model emits only
    /// `max_completion_tokens` (not `max_tokens`).
    #[test]
    fn chat_completions_emits_only_max_completion_tokens_for_gpt5() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let mut rewrite = std::collections::HashMap::new();
        rewrite.insert("claude-sonnet-4-5".to_string(), "gpt-5".to_string());
        let converted = anthropic_to_openai_request(&req, &rewrite, false).unwrap();
        assert_eq!(converted.max_tokens, None, "gpt-5 must not emit max_tokens");
        assert_eq!(
            converted.max_completion_tokens,
            Some(200),
            "gpt-5 must emit max_completion_tokens"
        );

        // o3-mini also uses max_completion_tokens
        let mut rewrite2 = std::collections::HashMap::new();
        rewrite2.insert("claude-sonnet-4-5".to_string(), "o3-mini".to_string());
        let converted2 = anthropic_to_openai_request(&req, &rewrite2, false).unwrap();
        assert_eq!(converted2.max_tokens, None, "o3-mini must not emit max_tokens");
        assert_eq!(
            converted2.max_completion_tokens,
            Some(200),
            "o3-mini must emit max_completion_tokens"
        );
    }

    /// T12: Chat Completions for a non-gpt-5 model emits only
    /// `max_tokens` (not `max_completion_tokens`).
    #[test]
    fn chat_completions_emits_only_max_tokens_for_non_gpt5() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let converted = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(
            converted.max_tokens,
            Some(100),
            "non-gpt-5 must emit max_tokens"
        );
        assert_eq!(
            converted.max_completion_tokens,
            None,
            "non-gpt-5 must not emit max_completion_tokens"
        );
    }

    #[test]
    fn propagates_output_config_format_to_response_format_and_effort_to_typed_field() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "respond in json"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"]
                }},
                "effort": "high"
            }
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let resp_format = out.extra.get("response_format").unwrap();
        assert_eq!(resp_format.get("type").and_then(|v| v.as_str()), Some("json_schema"));
        // Anthropic doesn't carry a schema name; OpenAI requires one inside
        // the nested json_schema object. Synthesized default + strict: true.
        let inner = resp_format.get("json_schema").unwrap();
        assert_eq!(inner.get("name").and_then(|v| v.as_str()), Some("structured_output"));
        assert_eq!(inner.get("strict").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(out.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn output_config_absent_leaves_extra_empty_and_no_effort() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert!(out.extra.as_object().unwrap().is_empty());
        assert!(out.reasoning_effort.is_none());
    }

    #[test]
    fn output_config_effort_overrides_thinking_derivation() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "reason about this"}],
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {
                "effort": "low"
            }
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(out.reasoning_effort.as_deref(), Some("low"));
    }

    #[test]
    fn output_config_format_with_existing_name_is_preserved() {
        // If the client already supplied a `name` (e.g. via a future Anthropic
        // schema field), we must not overwrite it. The synthesizer is an
        // additive shim, not a rewriter.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "x"}],
            "output_config": {
                "format": {"type": "json_schema", "name": "my_schema", "schema": {"type": "object"}}
            }
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let resp_format = out.extra.get("response_format").unwrap();
        let inner = resp_format.get("json_schema").unwrap();
        assert_eq!(inner.get("name").and_then(|v| v.as_str()), Some("my_schema"));
    }

    #[test]
    fn output_config_format_with_optional_properties_is_strictified() {
        // Claude Code's Stop-hook schema declares `impossible` as optional.
        // OpenAI Chat Completions strict mode requires every property in
        // `properties` to appear in `required`. The translator must rewrite the
        // schema (litellm parity).
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "x"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "impossible": {"type": "boolean"}
                    },
                    "required": ["ok", "reason"]
                }}
            }
        })).unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let inner = out.extra.get("response_format").and_then(|v| v.get("json_schema")).unwrap();
        assert_eq!(inner.get("name").and_then(|v| v.as_str()), Some("structured_output"));
        assert_eq!(inner.get("strict").and_then(|v| v.as_bool()), Some(true));
        let schema = inner.get("schema").unwrap();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let required: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required.contains(&"impossible"), "optional property must be promoted to required under strict mode");
        assert_eq!(schema.get("additionalProperties").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn ensure_chat_json_schema_name_non_json_schema_passthrough() {
        // ensure_chat_json_schema_name is a no-op for format types other
        // than "json_schema" — e.g. plain "json_object" or "text" must be
        // returned unchanged (no json_schema wrapper synthesized).
        let input = serde_json::json!({"type": "json_object"});
        let out = ensure_chat_json_schema_name(&input).unwrap();
        assert_eq!(out, input);

        let input = serde_json::json!({"type": "text"});
        let out = ensure_chat_json_schema_name(&input).unwrap();
        assert_eq!(out, input);
    }

    // ── PR-7 · service_tier request mapping ─────────────────────────────

    /// PR-7: Anthropic `service_tier: "auto"` must pass through to
    /// OpenAI `service_tier: "auto"` (same-value passthrough — `auto`
    /// is a spec-legal OpenAI value per plan v0.7 correction; v0.2
    /// erroneously listed only default/priority/flex and was missing
    /// auto/scale/fast).
    #[test]
    fn chat_request_service_tier_auto_passes_through() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "auto"
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(out.service_tier.as_deref(), Some("auto"));
    }

    /// PR-7: Anthropic `service_tier: "standard_only"` has no OpenAI
    /// equivalent (Anthropic routing preference ≠ OpenAI pricing tier)
    /// — must be dropped, not coerced to a default. Coercing would 400
    /// on tier-strict upstreams.
    #[test]
    fn chat_request_service_tier_standard_only_is_dropped() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "standard_only"
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert!(
            out.service_tier.is_none(),
            "standard_only must drop, got {:?}",
            out.service_tier
        );
    }

    // ── PR-8 · safety_identifier / verbosity / parallel_tool_calls ──

    /// PR-8: Anthropic `tool_choice.disable_parallel_tool_use=true`
    /// maps to OpenAI `parallel_tool_calls=false`. Absent/false →
    /// None (OpenAI wire default is true; emitting an explicit true
    /// would be noisier).
    #[test]
    fn chat_request_disable_parallel_tool_use_maps_to_parallel_tool_calls_false() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": true}
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(
            out.parallel_tool_calls,
            Some(false),
            "disable_parallel_tool_use=true must map to parallel_tool_calls=false"
        );
    }

    /// PR-8: disable_parallel_tool_use=false (or absent) → None on the
    /// wire (default true at OpenAI is the desired behavior).
    #[test]
    fn chat_request_disable_parallel_tool_use_unset_or_false_leaves_field_absent() {
        for body in [
            serde_json::json!({
                "model": "gpt-4o", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": "f"}
            }),
            serde_json::json!({
                "model": "gpt-4o", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": false}
            }),
        ] {
            let req: MessagesRequest = serde_json::from_value(body).unwrap();
            let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
            assert!(
                out.parallel_tool_calls.is_none(),
                "unset/false must leave parallel_tool_calls absent; got {:?}",
                out.parallel_tool_calls
            );
        }
    }

    /// PR-8: `safety_identifier` is sourced from `metadata.user_id`
    /// (Anthropic's user-identity surface; maxLength 64 enforced by
    /// `truncate_user`).
    #[test]
    fn chat_request_safety_identifier_sourced_from_metadata_user_id() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "user-42"}
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(out.safety_identifier.as_deref(), Some("user-42"));
    }

    /// PR-8: `verbosity` is sourced from `output_config.verbosity` and
    /// passes through verbatim.
    #[test]
    fn chat_request_verbosity_passes_through() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"verbosity": "low"}
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        assert_eq!(out.verbosity.as_deref(), Some("low"));
    }

    // ── PR-9 · P2 fields stay absent on the wire ───────────────────────

    /// PR-9: the nine P2 fields (plan M3c "写" list minus
    /// include_obfuscation) have no Anthropic source, so the conversion
    /// layer must leave them None and, critically, absent from the
    /// serialized body — a bare `null` on the wire would confuse strict
    /// upstreams. This is the plan M3c "明确不写" contract at the
    /// conversion boundary (plan line 538: assert via to_value, not just
    /// field access).
    #[test]
    fn chat_request_pr9_p2_fields_absent_from_serialized_body() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let v = serde_json::to_value(&out).unwrap();
        for key in [
            "n",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "prediction",
            "metadata",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ] {
            assert!(
                v.get(key).is_none(),
                "conversion must not inject {key}; got: {v}"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // reasoning_echo: per-provider knob that fills missing
    // `reasoning_content` on historical assistant messages with `""`
    // for DeepSeek-V4-style upstreams in thinking mode. Default off
    // keeps the wire byte-identical to the pre-knob behavior.
    //
    // Gate (must be both true):
    //   • `reasoning_echo == true`
    //   • `reasoning_effort.is_some()` (thinking.enabled with a budget,
    //     or output_config.effort)
    //
    // Same-model turns with real reasoning keep their text (the fill
    // only kicks in when reasoning_content is already None after
    // convert_blocks ran).
    // ──────────────────────────────────────────────────────────────────

    /// thinking-enabled request + plain-text assistant history + knob on →
    /// wire `reasoning_content == ""` on the assistant message.
    #[test]
    fn reasoning_echo_fills_empty_string_when_enabled() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plain answer"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        // Find the assistant message.
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { reasoning_content, .. } => {
            assert_eq!(
                reasoning_content.as_deref(),
                Some(""),
                "reasoning_echo=true on a plain assistant turn must emit reasoning_content: \"\"; \
                 got: {reasoning_content:?}"
            );
        });
    }

    /// Same request, knob off → field absent (regression pin).
    #[test]
    fn reasoning_echo_absent_field_when_disabled() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plain answer"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let v = serde_json::to_value(&out).unwrap();
        let assistant = &v["messages"][1];
        assert!(
            assistant.get("reasoning_content").is_none(),
            "reasoning_echo=false must keep reasoning_content absent; got: {assistant}"
        );
    }

    /// Cross-model history with redacted_thinking + text + knob on →
    /// the redacted block is dropped by convert_blocks, and the
    /// assistant message still gets `reasoning_content: ""`.
    #[test]
    fn reasoning_echo_fills_after_redacted_thinking() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                    {"type": "text", "text": "previous answer"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { content, reasoning_content, .. } => {
            assert_eq!(content.as_deref(), Some("previous answer"));
            assert_eq!(
                reasoning_content.as_deref(),
                Some(""),
                "redacted_thinking turn + reasoning_echo=true must emit reasoning_content: \"\""
            );
        });
    }

    /// Same-model thinking turn + knob on → real reasoning text
    /// preserved (the `""` fill only kicks in when reasoning_content
    /// is None after convert_blocks).
    #[test]
    fn reasoning_echo_does_not_overwrite_same_model_echo() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "real reasoning text"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { reasoning_content, .. } => {
            assert_eq!(
                reasoning_content.as_deref(),
                Some("real reasoning text"),
                "reasoning_echo=true must not overwrite same-model reasoning"
            );
        });
    }

    /// Non-thinking request (no `thinking`, no `output_config.effort`)
    /// + knob on → reasoning_content stays None (the `reasoning_effort`
    /// half of the gate is false).
    #[test]
    fn reasoning_echo_noop_for_non_thinking_request() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plain answer"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { reasoning_content, .. } => {
            assert!(
                reasoning_content.is_none(),
                "non-thinking request + reasoning_echo=true must NOT inject reasoning_content"
            );
        });
    }

    /// `output_config.effort` alone (no `thinking.budget_tokens`) also
    /// arms the gate: reasoning_effort is derived from effort, so the
    /// plain assistant turn gets `reasoning_content: ""`.
    #[test]
    fn reasoning_echo_fills_on_output_config_effort_alone() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "output_config": {"effort": "high"},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plain answer"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { reasoning_content, .. } => {
            assert_eq!(
                reasoning_content.as_deref(),
                Some(""),
                "output_config.effort alone arms the gate; reasoning_content must be \"\""
            );
        });
    }

    /// Tool-only assistant turn (no text, no thinking) + knob on →
    /// reasoning_content: "". This is the dominant cross-model scenario
    /// DeepSeek actually rejects (a prior Anthropic tool-call turn with
    /// no reasoning to echo).
    #[test]
    fn reasoning_echo_fills_after_tool_call_assistant_turn() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "what's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "72F"}
                ]},
                {"role": "assistant", "content": "sunny"},
                {"role": "user", "content": "thanks"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), true).unwrap();
        // First assistant message is the tool-use turn (collapsed into a
        // single ChatMessage::Assistant by convert_blocks).
        let tool_turn = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { tool_calls: Some(_), .. } => Some(m),
                _ => None,
            })
            .expect("tool-call assistant turn present");
        expect_variant!(tool_turn, ChatMessage::Assistant { reasoning_content, .. } => {
            assert_eq!(
                reasoning_content.as_deref(),
                Some(""),
                "tool-only assistant turn + reasoning_echo=true must emit reasoning_content: \"\""
            );
        });
    }

    /// Knob OFF + empty-thinking assistant turn → reasoning_content
    /// stays None (the original wire shape is preserved).
    #[test]
    fn reasoning_echo_pure_empty_thinking_keeps_none() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": ""}
                ]},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap();
        let out = anthropic_to_openai_request(&req, &Default::default(), false).unwrap();
        let assistant = out
            .messages
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { .. } => Some(m),
                _ => None,
            })
            .expect("assistant message present");
        expect_variant!(assistant, ChatMessage::Assistant { reasoning_content, .. } => {
            assert!(
                reasoning_content.is_none(),
                "knob-off + empty thinking must keep reasoning_content absent"
            );
        });
    }
}

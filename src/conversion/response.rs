//! OpenAI Chat Completions → Anthropic Messages response conversion.
//!
//! Reference: copilot-api-py/src/routes/messages/non_stream_translation.py:293-378

use serde_json::{json, Value};

use crate::anthropic::{MessagesResponse, ResponseBlock, Usage};
use crate::error::{ProxyError, Result};
use crate::openai::ChatResponse;

/// Convert a non-streaming OpenAI ChatResponse into an Anthropic MessagesResponse.
///
/// `model` is the original model name from the client (since the provider may
/// have rewritten it). `message_id` is the Anthropic-style id to assign.
pub fn openai_to_anthropic_response(
    resp: &ChatResponse,
    model: &str,
    message_id: &str,
) -> Result<MessagesResponse> {
    let choice = resp.choices.first().ok_or_else(|| {
        ProxyError::BadRequest("upstream returned no choices".into())
    })?;

    let mut content: Vec<ResponseBlock> = Vec::new();

    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            content.push(ResponseBlock::Thinking {
                thinking: reasoning.clone(),
                signature: None,
            });
        }
    }

    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(ResponseBlock::Text { text: text.clone() });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            let input: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| json!({}));
            content.push(ResponseBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    let usage = resp
        .usage
        .as_ref()
        .map(|u| {
            let cached = u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0);
            Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: if cached > 0 { Some(cached) } else { None },
            }
        })
        .unwrap_or_default();

    // R6: when reasoning models (DeepSeek-R1, claude-sonnet-4-5 with
    // thinking enabled) burn most or all of `max_tokens` on internal
    // reasoning, the client gets back a response with stop_reason =
    // max_tokens and zero visible text — which looks like an empty
    // response from the client's point of view. Warn at the proxy
    // layer so operators see this happening (and can tell the client
    // to send a larger max_tokens), without changing the wire
    // response (Anthropic's Usage schema has no reasoning_tokens
    // field; folding thinking into visible text would change the
    // semantics the client signed up for).
    if let Some(u) = resp.usage.as_ref() {
        if let Some(reasoning) = u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
        {
            if u.completion_tokens > 0 && reasoning >= u.completion_tokens {
                tracing::warn!(
                    model = model,
                    completion_tokens = u.completion_tokens,
                    reasoning_tokens = reasoning,
                    "response consumed by reasoning; client will see no visible text — request a larger max_tokens"
                );
            }
        }
    }

    Ok(MessagesResponse {
        id: message_id.to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason: choice
            .finish_reason
            .as_deref()
            .map(map_stop_reason)
            .transpose()?
            .flatten(),
        stop_sequence: None,
        usage,
    })
}

/// Map OpenAI finish_reason → Anthropic stop_reason.
/// Returns Ok(None) if unknown (we still produce a valid response).
pub fn map_stop_reason(reason: &str) -> Result<Option<String>> {
    let mapped = match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        "content_filter" => "end_turn", // Anthropic has no exact equivalent
        other => {
            tracing::debug!("unknown finish_reason: {other}");
            return Ok(None);
        }
    };
    Ok(Some(mapped.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_resp() -> ChatResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }))
        .unwrap()
    }

    #[test]
    fn converts_basic_response() {
        let resp = fixture_resp();
        let out = openai_to_anthropic_response(&resp, "claude-sonnet-4-5", "msg_1").unwrap();
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(out.usage.output_tokens, 3);
        assert_eq!(out.content.len(), 2);
        assert!(matches!(out.content[1], ResponseBlock::ToolUse { .. }));
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(map_stop_reason("stop").unwrap(), Some("end_turn".into()));
        assert_eq!(map_stop_reason("length").unwrap(), Some("max_tokens".into()));
        assert_eq!(map_stop_reason("tool_calls").unwrap(), Some("tool_use".into()));
    }

    #[test]
    fn missing_choices_errors() {
        let empty: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "m", "choices": []
        }))
        .unwrap();
        assert!(openai_to_anthropic_response(&empty, "m", "x").is_err());
    }

    #[test]
    fn maps_unknown_finish_reason_to_none() {
        assert_eq!(map_stop_reason("content_filter").unwrap(), Some("end_turn".into()));
        assert_eq!(
            map_stop_reason("totally_unknown_reason").unwrap(),
            None
        );
    }

    #[test]
    fn promotes_reasoning_to_thinking_block_before_text() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "final",
                    "reasoning_content": "because"
                },
                "finish_reason": "stop"
            }],
            "usage": null
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        assert_eq!(out.content.len(), 2);
        assert!(matches!(out.content[0], ResponseBlock::Thinking { ref thinking, .. } if thinking == "because"));
        assert!(matches!(out.content[1], ResponseBlock::Text { ref text } if text == "final"));
        assert_eq!(out.usage.input_tokens, 0);
        assert_eq!(out.usage.output_tokens, 0);
        assert!(out.usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn invalid_tool_arguments_default_to_empty_object() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": null
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        assert!(matches!(
            &out.content[0],
            ResponseBlock::ToolUse { name, input, .. } if name == "f" && input == &Value::Object(Default::default())
        ));
    }

    #[test]
    fn propagates_cache_read_tokens_when_present() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 4}
            }
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        assert_eq!(out.usage.cache_read_input_tokens, Some(4));
        assert_eq!(out.usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn omits_cache_read_tokens_when_zero() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        assert!(out.usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn uses_message_id_and_model_passthrough() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "upstream", "msg_42").unwrap();
        assert_eq!(out.id, "msg_42");
        assert_eq!(out.model, "upstream");
        assert_eq!(out.kind, "message");
        assert_eq!(out.role, "assistant");
        assert!(out.stop_reason.is_none());
    }

    #[test]
    fn parses_reasoning_tokens_in_completion_details() {
        // DeepSeek-R1 / claude-sonnet-4-5 with thinking enabled return
        // completion_tokens_details.reasoning_tokens. The proxy must
        // accept the field (not fail the parse) even when the
        // reasoning_tokens count is high — we use it only for the
        // R6 warning path. See fix-R6 in docs/TEST_ISSUES.md.
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 18}
            }
        }))
        .unwrap();

        let reasoning = resp
            .usage
            .as_ref()
            .unwrap()
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens);
        assert_eq!(reasoning, Some(18));
        // Conversion still succeeds and total output_tokens stays 20
        // (we don't break out reasoning in the Anthropic Usage today).
        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        assert_eq!(out.usage.output_tokens, 20);
    }

    #[test]
    fn parses_without_completion_tokens_details() {
        // Upstreams that don't include completion_tokens_details
        // (OpenAI's standard /v1/chat/completions) must still parse
        // cleanly. The new field is Option-typed so its absence is a
        // no-op.
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .unwrap();

        assert!(resp
            .usage
            .as_ref()
            .unwrap()
            .completion_tokens_details
            .is_none());
    }

    #[test]
    fn reasoning_dominates_output_does_not_break_conversion() {
        // R6: when reasoning_tokens equals completion_tokens, the
        // model burned every output token on internal thinking and
        // the client sees no visible text. The conversion must
        // succeed (so the Thinking block reaches the client) and the
        // warn-log fires. We don't capture tracing output here — we
        // only assert the conversion succeeds with the expected
        // content blocks and usage; the warning side-effect is
        // covered by the warn-emit code path itself being exercised.
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "thinking... thinking... thinking..."
                },
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 20}
            }
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        // The conversion succeeds — Thinking block is preserved for
        // any client that knows how to read it.
        assert_eq!(out.content.len(), 1);
        assert!(matches!(&out.content[0], ResponseBlock::Thinking { .. }));
        // stop_reason maps "length" → "max_tokens" so the client
        // sees why the response ended without visible text.
        assert_eq!(out.stop_reason.as_deref(), Some("max_tokens"));
        // usage still reports the full 20 (we don't surface reasoning
        // separately — Anthropic's schema has no field for it).
        assert_eq!(out.usage.output_tokens, 20);
    }
}

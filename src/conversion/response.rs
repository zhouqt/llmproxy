//! OpenAI Chat Completions → Anthropic Messages response conversion.
//!
//! Reference: copilot-api-py/src/routes/messages/non_stream_translation.py:293-378

use std::collections::HashMap;

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
            content.push(ResponseBlock::Text {
                text: text.clone(),
                citations: None,
            });
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
                caller: None,
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
            // PR-6b: forward reasoning_tokens to the Anthropic client as
            // `output_tokens_details.thinking_tokens` (Anthropic's
            // official OutputTokensDetails field name per
            // anthropic-sdk-python — `anthropic.rs:658` test pins it).
            // The OpenAI-side key is `reasoning_tokens`; the Anthropic-
            // side key is `thinking_tokens` (different keys on each
            // side of the conversion, never leak `reasoning_tokens`
            // into the Anthropic write payload).
            let thinking_tokens = u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .filter(|&n| n > 0);
            Usage {
                input_tokens: u.prompt_tokens.saturating_sub(cached),
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: if cached > 0 { Some(cached) } else { None },
                cache_creation: None,
                server_tool_use: None,
                output_tokens_details: thinking_tokens.map(|n| json!({"thinking_tokens": n})),
                service_tier: None,
                inference_geo: None,
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
        stop_details: None,
        container: None,
        usage,
        extra: HashMap::new(),
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
        assert!(matches!(out.content[1], ResponseBlock::Text { ref text, .. } if text == "final"));
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

    /// Bug repro for Chat Completions non-streaming path — mirrors
    /// `responses.rs::input_tokens_excludes_cached_subset`. OpenAI
    /// ChatCompletions `prompt_tokens` includes the cached subset
    /// (`prompt_tokens_details.cached_tokens` has range
    /// `[0, prompt_tokens]`), so the translator must subtract the
    /// cached portion when populating Anthropic's `Usage.input_tokens`,
    /// which is non-cached-only by spec.
    #[test]
    fn input_tokens_excludes_cached_in_chat_completions() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 60}
            }
        }))
        .unwrap();

        let out = openai_to_anthropic_response(&resp, "model", "msg_1").unwrap();
        // Anthropic.input_tokens must be the non-cached portion: 100 - 60 = 40.
        assert_eq!(
            out.usage.input_tokens, 40,
            "input_tokens must be non-cached only (prompt - cached); got {}, expected 40",
            out.usage.input_tokens
        );
        assert_eq!(out.usage.cache_read_input_tokens, Some(60));
        assert_eq!(out.usage.output_tokens, 10);
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

    // ── PR-6b · output_tokens_details.thinking_tokens (Chat non-stream) ──

    /// PR-6b: when the upstream returns `completion_tokens_details.
    /// reasoning_tokens` (OpenAI read key), the Anthropic write key is
    /// `output_tokens_details.thinking_tokens` (different key on the
    /// other side — never leak `reasoning_tokens` into the Anthropic
    /// write payload). When reasoning is present, the field is
    /// emitted; when absent (None) or zero, the field is absent too
    /// (Some(0) would change the wire shape from absent to present-
    /// zero, which Anthropic's Usage schema treats as "not provided").
    #[test]
    fn chat_non_stream_propagates_reasoning_tokens_as_thinking_tokens() {
        // (a) reasoning_tokens present → field present, key is thinking_tokens.
        let with_reasoning = serde_json::from_value::<ChatResponse>(serde_json::json!({
            "id": "chatcmpl-t1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 12}
            }
        }))
        .unwrap();
        let out = openai_to_anthropic_response(&with_reasoning, "gpt-4o", "msg_t").unwrap();
        let details = out.usage.output_tokens_details.expect(
            "reasoning_tokens>0 must surface output_tokens_details",
        );
        assert_eq!(details["thinking_tokens"], 12);
        // The OpenAI-side key must NOT leak into the Anthropic write payload.
        assert!(
            details.get("reasoning_tokens").is_none(),
            "OpenAI read key `reasoning_tokens` must not appear in Anthropic write payload; got {details}"
        );

        // (b) reasoning_tokens absent (None) → field absent entirely.
        let no_details = serde_json::from_value::<ChatResponse>(serde_json::json!({
            "id": "chatcmpl-t2",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
        }))
        .unwrap();
        let out = openai_to_anthropic_response(&no_details, "gpt-4o", "msg_t").unwrap();
        assert!(
            out.usage.output_tokens_details.is_none(),
            "no reasoning_tokens → output_tokens_details must be absent"
        );

        // (c) reasoning_tokens == 0 → field absent (Some(0) would change shape).
        let zero = serde_json::from_value::<ChatResponse>(serde_json::json!({
            "id": "chatcmpl-t3",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }))
        .unwrap();
        let out = openai_to_anthropic_response(&zero, "gpt-4o", "msg_t").unwrap();
        assert!(
            out.usage.output_tokens_details.is_none(),
            "reasoning_tokens==0 → output_tokens_details must be absent"
        );
    }
}

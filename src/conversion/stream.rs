//! OpenAI SSE → Anthropic SSE streaming translation.
//!
//! Reference: copilot-api-py/src/routes/messages/stream_translation.py:21-199
//!
//! The converter takes a stream of OpenAI ChatChunk (already parsed from SSE
//! lines) and yields Anthropic StreamEvent values in the proper order.

use std::collections::HashMap;

use serde_json::json;

use crate::anthropic::{
    BlockDelta, MessageDeltaPayload, MessagesResponse, ResponseBlock, StreamEvent, Usage,
};
use crate::openai::{ChatChunk, ChatUsage};

use super::response::map_stop_reason;

/// Per-block streaming state.
#[derive(Debug, Clone)]
enum BlockState {
    Pending,
    Text,
    ToolUse,
    Thinking,
}

pub struct StreamTranslator {
    message_id: String,
    model: String,
    blocks: Vec<BlockState>,
    block_has_text: Vec<bool>,
    block_has_thinking: Vec<bool>,
    open_blocks: Vec<u32>,
    tool_block_map: HashMap<u32, u32>,
    started: bool,
    finalized: bool,
    final_stop_reason: Option<String>,
    final_usage: Option<ChatUsage>,
}

impl StreamTranslator {
    pub fn new(message_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            message_id: message_id.into(),
            model: model.into(),
            blocks: Vec::new(),
            block_has_text: Vec::new(),
            block_has_thinking: Vec::new(),
            open_blocks: Vec::new(),
            tool_block_map: HashMap::new(),
            started: false,
            finalized: false,
            final_stop_reason: None,
            final_usage: None,
        }
    }

    pub fn push_chunk(&mut self, chunk: &ChatChunk) -> Vec<StreamEvent> {
        if self.finalized {
            return Vec::new();
        }
        let mut out = Vec::new();

        if !self.started {
            self.started = true;
            let placeholder = MessagesResponse {
                id: self.message_id.clone(),
                kind: "message".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                stop_details: None,
                container: None,
                usage: Usage::default(),
                extra: std::collections::HashMap::new(),
            };
            out.push(StreamEvent::MessageStart { message: placeholder });
        }

        if let Some(usage) = &chunk.usage {
            self.final_usage = Some(usage.clone());
        }

        for choice in &chunk.choices {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    out.extend(self.push_text_delta(content));
                }
            }
            if let Some(reasoning) = &choice.delta.reasoning_content {
                if !reasoning.is_empty() {
                    out.extend(self.push_thinking_delta(reasoning));
                }
            }
            if let Some(tool_calls) = &choice.delta.tool_calls {
                for tc in tool_calls {
                    out.extend(self.push_tool_delta(tc.index as u32, tc));
                }
            }
            if let Some(fr) = &choice.finish_reason {
                self.final_stop_reason = Some(fr.clone());
            }
        }

        out
    }

    pub fn finalize(&mut self) -> Vec<StreamEvent> {
        if self.finalized {
            return Vec::new();
        }
        self.finalized = true;
        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push(StreamEvent::MessageStart {
                message: MessagesResponse {
                    id: self.message_id.clone(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: self.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    stop_details: None,
                    container: None,
                    usage: Usage::default(),
                    extra: std::collections::HashMap::new(),
                },
            });
        }
        out.extend(self.close_open_blocks());

        let stop_reason = self
            .final_stop_reason
            .as_deref()
            .and_then(|r| map_stop_reason(r).ok().flatten())
            .unwrap_or_else(|| "end_turn".to_string());

        let usage = self.final_usage.as_ref().map(|u| {
            let cached = u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0);
            Usage {
                input_tokens: u.prompt_tokens.saturating_sub(cached),
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: u
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
                    .filter(|&n| n > 0),
                cache_creation: None,
                server_tool_use: None,
                output_tokens_details: None,
                service_tier: None,
                inference_geo: None,
            }
        });

        out.push(StreamEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop_reason),
                stop_sequence: None,
                stop_details: None,
                container: None,
            },
            usage,
        });
        out.push(StreamEvent::MessageStop);
        out
    }

    fn ensure_block_capacity(&mut self, idx: u32) {
        while self.blocks.len() <= idx as usize {
            self.blocks.push(BlockState::Pending);
            self.block_has_text.push(false);
            self.block_has_thinking.push(false);
        }
    }

    fn push_text_delta(&mut self, text: &str) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let idx = if let Some(idx) = self.open_block_index(|b| matches!(b, BlockState::Text)) {
            idx
        } else {
            out.extend(self.close_open_blocks());
            self.open_text_block()
        };

        if !self.block_has_text[idx as usize] {
            self.block_has_text[idx as usize] = true;
            out.push(StreamEvent::ContentBlockStart {
                index: idx,
                content_block: ResponseBlock::Text { text: String::new(), citations: None },
            });
        }

        out.push(StreamEvent::ContentBlockDelta {
            index: idx,
            delta: BlockDelta::TextDelta { text: text.to_string() },
        });
        out
    }

    fn push_thinking_delta(&mut self, text: &str) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let idx = if let Some(idx) = self.open_block_index(|b| matches!(b, BlockState::Thinking)) {
            idx
        } else {
            out.extend(self.close_open_blocks());
            self.open_thinking_block()
        };

        if !self.block_has_thinking[idx as usize] {
            self.block_has_thinking[idx as usize] = true;
            out.push(StreamEvent::ContentBlockStart {
                index: idx,
                content_block: ResponseBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                },
            });
        }
        out.push(StreamEvent::ContentBlockDelta {
            index: idx,
            delta: BlockDelta::ThinkingDelta { thinking: text.to_string() },
        });
        out
    }

    fn push_tool_delta(
        &mut self,
        openai_index: u32,
        tc: &crate::openai::ChunkToolCall,
    ) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let existing_index = self
            .tool_block_map
            .get(&openai_index)
            .copied()
            .filter(|index| self.open_blocks.contains(index));
        let has_identity = tc.id.as_deref().is_some_and(|id| !id.is_empty())
            && tc
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref())
                .is_some_and(|name| !name.is_empty());

        if existing_index.is_none() && !has_identity {
            tracing::warn!(
                target: "tool_call_debug",
                openai_index,
                "tool_call arguments arrived before id/name; ignoring until block identity arrives"
            );
            return out;
        }

        if existing_index.is_none() {
            self.tool_block_map.remove(&openai_index);
            out.extend(self.close_open_non_tool_blocks());

            let block_idx = self.blocks.len() as u32;
            self.ensure_block_capacity(block_idx);
            self.tool_block_map.insert(openai_index, block_idx);

            let id = tc.id.clone().unwrap_or_default();
            let name = tc
                .function
                .as_ref()
                .and_then(|f| f.name.clone())
                .unwrap_or_default();

            let id_len = id.chars().count();
            let id_anthropic_compatible = (1..=64).contains(&id_len)
                && id.chars().all(|c| {
                    c.is_ascii_alphanumeric() || c == '_' || c == '-'
                });
            let name_len = name.chars().count();
            tracing::debug!(
                target: "tool_call_debug",
                direction = "openai_chat->anthropic_stream",
                chunk_index = ?tc.index,
                openai_index,
                block_idx,
                id = %id,
                id_chars = id_len,
                id_anthropic_compatible = id_anthropic_compatible,
                name = %name,
                name_chars = name_len,
                "tool_call open delta (chat, stream)"
            );
            if !id_anthropic_compatible && !id.is_empty() {
                tracing::warn!(
                    target: "tool_call_debug",
                    id = %id,
                    id_chars = id_len,
                    "streamed tool_call.id violates Anthropic ^[a-zA-Z0-9_-]{{1,64}}$"
                );
            }

            self.blocks[block_idx as usize] = BlockState::ToolUse;
            self.open_blocks.push(block_idx);
            out.push(StreamEvent::ContentBlockStart {
                index: block_idx,
                content_block: ResponseBlock::ToolUse {
                    id,
                    name,
                    input: json!({}),
                    caller: None,
                },
            });
        }

        let block_idx = self.tool_block_map[&openai_index];
        if let Some(args_delta) = tc.function.as_ref().and_then(|f| f.arguments.clone()) {
            if !args_delta.is_empty() {
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: args_delta,
                    },
                });
            }
        }

        out
    }

    fn close_open_non_tool_blocks(&mut self) -> Vec<StreamEvent> {
        let mut closed = Vec::new();
        self.open_blocks.retain(|&index| {
            if matches!(self.blocks[index as usize], BlockState::ToolUse) {
                true
            } else {
                closed.push(index);
                false
            }
        });
        closed.sort_unstable();
        closed
            .into_iter()
            .map(|index| StreamEvent::ContentBlockStop { index })
            .collect()
    }

    fn close_open_blocks(&mut self) -> Vec<StreamEvent> {
        let mut open = std::mem::take(&mut self.open_blocks);
        open.sort_unstable();
        open.dedup();
        open
            .into_iter()
            .map(|index| StreamEvent::ContentBlockStop { index })
            .collect()
    }

    fn open_block_index(&self, predicate: impl Fn(&BlockState) -> bool) -> Option<u32> {
        self.open_blocks
            .iter()
            .copied()
            .find(|&index| predicate(&self.blocks[index as usize]))
    }

    fn open_text_block(&mut self) -> u32 {
        let idx = self.blocks.len() as u32;
        self.ensure_block_capacity(idx);
        self.blocks[idx as usize] = BlockState::Text;
        self.open_blocks.push(idx);
        idx
    }

    fn open_thinking_block(&mut self) -> u32 {
        let idx = self.blocks.len() as u32;
        self.ensure_block_capacity(idx);
        self.blocks[idx as usize] = BlockState::Thinking;
        self.open_blocks.push(idx);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatChunk;

    fn chunk_with_content(s: &str) -> ChatChunk {
        serde_json::from_value(serde_json::json!({
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": s},
                "finish_reason": null
            }]
        }))
        .unwrap()
    }

    fn chunk_final(reason: &str) -> ChatChunk {
        serde_json::from_value(serde_json::json!({
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason
            }]
        }))
        .unwrap()
    }

    #[test]
    fn emits_start_then_text_then_final() {
        let mut t = StreamTranslator::new("msg_1", "claude-sonnet-4-5");
        let mut events = t.push_chunk(&chunk_with_content("he"));
        events.extend(t.push_chunk(&chunk_with_content("llo")));
        events.extend(t.push_chunk(&chunk_final("stop")));
        events.extend(t.finalize());

        assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
        let has_text_delta = events
            .iter()
            .any(|e| matches!(e, StreamEvent::ContentBlockDelta { .. }));
        assert!(has_text_delta);
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn ignores_text_after_finalize() {
        // A translator represents one Anthropic message. Once finalized, late
        // upstream chunks must be ignored rather than opening blocks after
        // message_stop.
        let mut t = StreamTranslator::new("msg_1", "m");
        let mut events = t.push_chunk(&chunk_with_content("first"));
        events.extend(t.finalize());
        let late = t.push_chunk(&chunk_with_content("second"));

        let start_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockStart { .. }))
            .count();
        assert_eq!(start_count, 1, "finalized translators must not reopen text blocks");
        assert!(late.is_empty());
    }

    #[test]
    fn ignores_thinking_after_finalize() {
        // Same single-message lifecycle rule as the text case.
        let mut t = StreamTranslator::new("msg_1", "m");
        let first: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "because"},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let second: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "therefore"},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let mut events = t.push_chunk(&first);
        events.extend(t.finalize());
        let late = t.push_chunk(&second);

        let start_count = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::ContentBlockStart {
                        content_block: ResponseBlock::Thinking { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(start_count, 1, "finalized translators must not reopen thinking blocks");
        assert!(late.is_empty());
    }

    #[test]
    fn handles_tool_call_streaming() {
        let mut t = StreamTranslator::new("msg_1", "m");

        let c1: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "get_x", "arguments": "{\"a\":"}
                }]},
                "finish_reason": null
            }]
        })).unwrap();

        let c2: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "1}"}
                }]},
                "finish_reason": null
            }]
        })).unwrap();

        let mut events = t.push_chunk(&c1);
        events.extend(t.push_chunk(&c2));
        events.extend(t.push_chunk(&chunk_final("tool_calls")));
        events.extend(t.finalize());

        let has_tool_start = events.iter().any(|e| {
            matches!(e, StreamEvent::ContentBlockStart { content_block: ResponseBlock::ToolUse { .. }, .. })
        });
        let has_json_delta = events.iter().any(|e| {
            matches!(e, StreamEvent::ContentBlockDelta { delta: BlockDelta::InputJsonDelta { .. }, .. })
        });
        assert!(has_tool_start);
        assert!(has_json_delta);

        let md = events.iter().rev().find_map(|e| {
            if let StreamEvent::MessageDelta { delta, .. } = e {
                Some(delta)
            } else {
                None
            }
        });
        assert_eq!(md.and_then(|d| d.stop_reason.as_deref()), Some("tool_use"));
    }

    #[test]
    fn reasoning_only_chunk_emits_thinking_block() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let chunk: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "because"},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let events = t.push_chunk(&chunk);

        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ContentBlockStart { content_block: ResponseBlock::Thinking { .. }, .. }
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ContentBlockDelta { delta: BlockDelta::ThinkingDelta { .. }, .. }
        )));
    }

    #[test]
    fn interleaves_text_and_tool_calls() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let text: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": "before "},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let tool: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let mut events = t.push_chunk(&text);
        events.extend(t.push_chunk(&tool));
        events.extend(t.push_chunk(&chunk_final("tool_calls")));
        events.extend(t.finalize());

        let lifecycle: Vec<(&str, u32)> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStart { index, .. } => Some(("start", *index)),
                StreamEvent::ContentBlockStop { index } => Some(("stop", *index)),
                _ => None,
            })
            .collect();
        assert_eq!(
            lifecycle,
            vec![("start", 0), ("stop", 0), ("start", 1), ("stop", 1)]
        );

        let tool_delta_index = events.iter().find_map(|e| match e {
            StreamEvent::ContentBlockDelta {
                index,
                delta: BlockDelta::InputJsonDelta { partial_json },
            } if partial_json == "{}" => Some(*index),
            _ => None,
        });
        assert_eq!(tool_delta_index, Some(1));
    }

    #[test]
    fn thinking_then_first_tool_uses_next_anthropic_index() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let thinking: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "considering"},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let tool: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{\"path\":\"/tmp/x\"}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let mut events = t.push_chunk(&thinking);
        events.extend(t.push_chunk(&tool));
        events.extend(t.finalize());

        let starts: Vec<(u32, &str)> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStart {
                    index,
                    content_block: ResponseBlock::Thinking { .. },
                } => Some((*index, "thinking")),
                StreamEvent::ContentBlockStart {
                    index,
                    content_block: ResponseBlock::ToolUse { .. },
                } => Some((*index, "tool_use")),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![(0, "thinking"), (1, "tool_use")]);
    }

    #[test]
    fn text_then_parallel_tools_get_distinct_anthropic_indices() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let text = chunk_with_content("before");
        let tools: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    {
                        "index": 0,
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "first", "arguments": "{\"x\":"}
                    },
                    {
                        "index": 1,
                        "id": "t2",
                        "type": "function",
                        "function": {"name": "second", "arguments": "{\"y\":"}
                    }
                ]},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let args: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "1}"}},
                    {"index": 1, "function": {"arguments": "2}"}}
                ]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let mut events = t.push_chunk(&text);
        events.extend(t.push_chunk(&tools));
        events.extend(t.push_chunk(&args));
        events.extend(t.finalize());

        let tool_starts: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStart {
                    index,
                    content_block: ResponseBlock::ToolUse { .. },
                } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(tool_starts, vec![1, 2]);

        let tool_args: Vec<(u32, String)> = [1_u32, 2]
            .into_iter()
            .map(|index| {
                let args = events
                    .iter()
                    .filter_map(|e| match e {
                        StreamEvent::ContentBlockDelta {
                            index: event_index,
                            delta: BlockDelta::InputJsonDelta { partial_json },
                        } if *event_index == index => Some(partial_json.as_str()),
                        _ => None,
                    })
                    .collect::<String>();
                (index, args)
            })
            .collect();
        assert_eq!(tool_args, vec![(1, "{\"x\":1}".into()), (2, "{\"y\":2}".into())]);
    }

    #[test]
    fn tool_only_first_call_uses_anthropic_index_zero() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let tool: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let events = t.push_chunk(&tool);
        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::MessageStart { .. },
                StreamEvent::ContentBlockStart { index: 0, .. },
                StreamEvent::ContentBlockDelta { index: 0, .. }
            ]
        ));
    }

    #[test]
    fn reusing_openai_tool_index_after_text_opens_new_tool_block() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let first_tool: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "first", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let second_tool: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0, "id": "t2", "type": "function",
                    "function": {"name": "second", "arguments": "{\"x\":1}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let mut events = t.push_chunk(&first_tool);
        events.extend(t.push_chunk(&chunk_with_content("after")));
        events.extend(t.push_chunk(&second_tool));
        events.extend(t.finalize());

        let tool_starts: Vec<(u32, &str)> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStart {
                    index,
                    content_block: ResponseBlock::ToolUse { name, .. },
                } => Some((*index, name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(tool_starts, vec![(0, "first"), (2, "second")]);

        let second_delta_index = events.iter().find_map(|e| match e {
            StreamEvent::ContentBlockDelta {
                index,
                delta: BlockDelta::InputJsonDelta { partial_json },
            } if partial_json == "{\"x\":1}" => Some(*index),
            _ => None,
        });
        assert_eq!(second_delta_index, Some(2));
    }

    #[test]
    fn finalize_is_idempotent_and_ignores_late_chunks() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let _ = t.push_chunk(&chunk_with_content("hello"));

        let first = t.finalize();
        let second = t.finalize();
        let late = t.push_chunk(&chunk_with_content("late"));

        assert!(matches!(first.last(), Some(StreamEvent::MessageStop)));
        assert!(second.is_empty());
        assert!(late.is_empty());
    }

    #[test]
    fn finalize_without_chunks_emits_valid_message_lifecycle() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let events = t.finalize();

        assert!(matches!(events.first(), Some(StreamEvent::MessageStart { .. })));
        assert!(matches!(events.get(1), Some(StreamEvent::MessageDelta { .. })));
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn tool_delta_without_identity_does_not_open_invalid_block() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let arguments_only: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"x\":1}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let identified: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let first = t.push_chunk(&arguments_only);
        let second = t.push_chunk(&identified);

        assert!(!first.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart {
                content_block: ResponseBlock::ToolUse { .. },
                ..
            }
        )));
        assert!(second.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ResponseBlock::ToolUse { id, name, .. },
            } if id == "t1" && name == "noop"
        )));
    }

    #[test]
    fn finalizes_with_length_stop_reason_and_propagates_usage() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let chunk: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": "truncated"},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 2}
            }
        }))
        .unwrap();
        let mut events = t.push_chunk(&chunk);
        events.extend(t.finalize());

        let delta = events.iter().rev().find_map(|e| {
            if let StreamEvent::MessageDelta { delta, .. } = e {
                Some(delta)
            } else {
                None
            }
        });
        assert_eq!(delta.and_then(|d| d.stop_reason.as_deref()), Some("max_tokens"));
        let usage = events.iter().rev().find_map(|e| {
            if let StreamEvent::MessageDelta { usage, .. } = e {
                usage.clone()
            } else {
                None
            }
        });
        assert_eq!(usage.as_ref().and_then(|u| u.cache_read_input_tokens), Some(2));
    }

    /// Bug repro for Chat Completions streaming path — mirrors
    /// `response.rs::input_tokens_excludes_cached_in_chat_completions`.
    /// The streaming translator's `message_delta` `usage` must report
    /// the non-cached portion as `input_tokens`; otherwise Claude Code
    /// double-counts the cached subset and triggers auto-compact
    /// prematurely.
    #[test]
    fn input_tokens_in_stream_finalize_excludes_cached_subset() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let chunk: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": "truncated"},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 60}
            }
        }))
        .unwrap();
        let mut events = t.push_chunk(&chunk);
        events.extend(t.finalize());

        let usage = events
            .iter()
            .rev()
            .find_map(|e| {
                if let StreamEvent::MessageDelta { usage, .. } = e {
                    usage.clone()
                } else {
                    None
                }
            })
            .expect("finalize should emit MessageDelta with usage");
        // Anthropic.input_tokens must be the non-cached portion: 100 - 60 = 40.
        assert_eq!(
            usage.input_tokens, 40,
            "input_tokens must be non-cached only (prompt - cached); got {}, expected 40",
            usage.input_tokens
        );
        assert_eq!(usage.cache_read_input_tokens, Some(60));
        assert_eq!(usage.output_tokens, 10);
    }

    #[test]
    fn empty_and_anonymous_chunks_do_not_emit_events() {
        let mut t = StreamTranslator::new("msg_1", "m");

        let empty: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let just_role: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }]
        }))
        .unwrap();

        let empty_events = t.push_chunk(&empty);
        let role_events = t.push_chunk(&just_role);

        assert!(empty_events.iter().all(|e| matches!(e, StreamEvent::MessageStart { .. })));
        assert!(role_events.is_empty());
    }

    #[test]
    fn empty_choices_and_no_usage_chunk_emits_nothing_after_start() {
        let mut t = StreamTranslator::new("msg_1", "m");
        let content: ChatChunk = serde_json::from_value(serde_json::json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": null
            }]
        })).unwrap();
        let metadata: ChatChunk = serde_json::from_value(serde_json::json!({
            "choices": [],
            "x-opencode-type": "inference-cost"
        })).unwrap();

        let events = t.push_chunk(&content);
        let second = t.push_chunk(&metadata);

        // First push emits MessageStart + ContentBlockStart + ContentBlockDelta
        assert!(events.iter().any(|e| matches!(e, StreamEvent::MessageStart { .. })));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::ContentBlockDelta { .. })));
        // Second push (metadata with empty choices) emits nothing
        assert!(second.is_empty());
    }
}

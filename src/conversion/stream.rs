//! OpenAI SSE → Anthropic SSE streaming translation.
//!
//! Reference: copilot-api-py/src/routes/messages/stream_translation.py:21-199
//!
//! The converter takes a stream of OpenAI ChatChunk (already parsed from SSE
//! lines) and yields Anthropic StreamEvent values in the proper order.

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
    ToolUse { id: String, name: String, args: String },
    Thinking,
}

pub struct StreamTranslator {
    message_id: String,
    model: String,
    blocks: Vec<BlockState>,
    block_has_text: Vec<bool>,
    block_has_thinking: Vec<bool>,
    open_blocks: Vec<u32>,
    started: bool,
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
            started: false,
            final_stop_reason: None,
            final_usage: None,
        }
    }

    pub fn push_chunk(&mut self, chunk: &ChatChunk) -> Vec<StreamEvent> {
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
                usage: Usage::default(),
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
        let mut out = Vec::new();
        let open = std::mem::take(&mut self.open_blocks);
        for idx in open {
            out.push(StreamEvent::ContentBlockStop { index: idx });
        }

        let stop_reason = self
            .final_stop_reason
            .as_deref()
            .and_then(|r| map_stop_reason(r).ok().flatten())
            .unwrap_or_else(|| "end_turn".to_string());

        let usage = self.final_usage.as_ref().map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .filter(|&n| n > 0),
        });

        out.push(StreamEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop_reason),
                stop_sequence: None,
                usage,
            },
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
        let idx = self.find_or_open_text_block();

        if !self.block_has_text[idx as usize] {
            self.block_has_text[idx as usize] = true;
            out.push(StreamEvent::ContentBlockStart {
                index: idx,
                content_block: ResponseBlock::Text { text: String::new() },
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
        let idx = self.find_or_open_thinking_block();

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
        declared_index: u32,
        tc: &crate::openai::ChunkToolCall,
    ) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        self.ensure_block_capacity(declared_index);

        // Decide if we need to open the block (first time we see an id/name).
        let needs_open = {
            let block = &self.blocks[declared_index as usize];
            match block {
                BlockState::Pending | BlockState::Text | BlockState::Thinking => true,
                BlockState::ToolUse { id, name, .. } => id.is_empty() || name.is_empty(),
            }
        };

        if needs_open {
            let id = tc.id.clone().unwrap_or_default();
            let name = tc
                .function
                .as_ref()
                .and_then(|f| f.name.clone())
                .unwrap_or_default();
            let args = tc
                .function
                .as_ref()
                .and_then(|f| f.arguments.clone())
                .unwrap_or_default();

            self.blocks[declared_index as usize] = BlockState::ToolUse {
                id: id.clone(),
                name: name.clone(),
                args,
            };
            self.open_blocks.push(declared_index);
            out.push(StreamEvent::ContentBlockStart {
                index: declared_index,
                content_block: ResponseBlock::ToolUse {
                    id,
                    name,
                    input: json!({}),
                },
            });
        }

        if let Some(args_delta) = tc.function.as_ref().and_then(|f| f.arguments.clone()) {
            if !args_delta.is_empty() {
                if let BlockState::ToolUse { args, .. } = &mut self.blocks[declared_index as usize] {
                    args.push_str(&args_delta);
                }
                out.push(StreamEvent::ContentBlockDelta {
                    index: declared_index,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: args_delta,
                    },
                });
            }
        }

        out
    }

    fn find_or_open_text_block(&mut self) -> u32 {
        for (i, b) in self.blocks.iter().enumerate() {
            if matches!(b, BlockState::Text) && self.open_blocks.contains(&(i as u32)) {
                return i as u32;
            }
        }
        let idx = self.blocks.len() as u32;
        self.ensure_block_capacity(idx);
        self.blocks[idx as usize] = BlockState::Text;
        self.open_blocks.push(idx);
        idx
    }

    fn find_or_open_thinking_block(&mut self) -> u32 {
        for (i, b) in self.blocks.iter().enumerate() {
            if matches!(b, BlockState::Thinking) && self.open_blocks.contains(&(i as u32)) {
                return i as u32;
            }
        }
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
    fn reopens_text_block_after_finalize() {
        // After finalize() clears open_blocks, pushing more text should open
        // a NEW block. The for-loop in find_or_open_text_block iterates over
        // a Text entry that is no longer in open_blocks (the "if false" branch
        // is exercised) before falling through to the allocation path.
        let mut t = StreamTranslator::new("msg_1", "m");
        let mut events = t.push_chunk(&chunk_with_content("first"));
        events.extend(t.finalize());
        events.extend(t.push_chunk(&chunk_with_content("second")));

        let start_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockStart { .. }))
            .count();
        assert_eq!(start_count, 2, "should reopen a fresh text block");
    }

    #[test]
    fn reopens_thinking_block_after_finalize() {
        // Same coverage idea as the text reopen test, but for thinking.
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
        events.extend(t.push_chunk(&second));

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
        assert_eq!(start_count, 2, "should reopen a fresh thinking block");
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
            if let StreamEvent::MessageDelta { delta } = e {
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
                    "index": 1,
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

        let block_starts: Vec<u32> = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::ContentBlockStart { index, .. } = e {
                    Some(*index)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(block_starts, vec![0, 1]);
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
            if let StreamEvent::MessageDelta { delta } = e {
                Some(delta)
            } else {
                None
            }
        });
        assert_eq!(delta.and_then(|d| d.stop_reason.as_deref()), Some("max_tokens"));
        let usage = delta.and_then(|d| d.usage.clone());
        assert_eq!(usage.as_ref().and_then(|u| u.cache_read_input_tokens), Some(2));
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
}

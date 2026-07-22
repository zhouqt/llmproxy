//! OpenAI Responses SSE → Anthropic SSE streaming translation.
//!
//! Reference: <https://platform.openai.com/docs/guides/streaming-responses>
//! Copilot-equivalent event handlers:
//! `copilot-api-py/src/services/copilot/responses_to_chat.py:130-260`.
//!
//! Event mapping (Responses → Anthropic):
//! - `response.created`             → `message_start` (placeholder)
//! - `response.output_item.added`   → `content_block_start` (text or tool_use)
//! - `response.output_text.delta`   → `content_block_delta` (text_delta)
//! - `response.function_call_arguments.delta`
//!                                  → `content_block_delta` (input_json_delta)
//! - `response.output_item.done`    → `content_block_stop`
//! - `response.completed` / `failed` / `incomplete`
//!                                  → `message_delta` (stop_reason + usage) + `message_stop`

use crate::anthropic::{
    BlockDelta, MessageDeltaPayload, MessagesResponse, ResponseBlock, StreamEvent, Usage,
};
use crate::responses::{OutputItem, ResponsesStreamEvent};

pub struct ResponsesStreamTranslator {
    message_id: String,
    model: String,
    started: bool,
    block_index: u32,
    /// Map Responses `output_index` → Anthropic content_block index.
    /// Each Responses output item maps to one Anthropic content block.
    block_map: std::collections::HashMap<u32, u32>,
    /// Anthropic block indices already closed by an explicit `*.done`
    /// event, so `finalize` doesn't emit a second `content_block_stop`
    /// for them.
    closed_blocks: std::collections::HashSet<u32>,
    /// Map Responses function-call `id` → the `output_index` of the
    /// `output_item.added` event that opened its block.
    ///
    /// The Copilot Responses stream sometimes uses a different
    /// `output_index` for the function call's `output_item.added` event
    /// (e.g. 1) than for its `function_call_arguments.delta/done`
    /// events (e.g. 0). Routing purely by `output_index` would then
    /// allocate two separate Anthropic block indices for the same
    /// tool call — `output_item.added` opens block 1, the args
    /// `delta`s go to block 0, the args `done` closes block 0, and
    /// `finalize()` re-closes block 1. Claude Code sees the duplicate
    /// `content_block_stop` and drops the tool call. Use the
    /// `item_id` carried on the args events to look up the correct
    /// block instead.
    fc_item_index: std::collections::HashMap<String, u32>,
    /// Set when an `output_item.added` of type FunctionCall arrives.
    /// Drives the terminal `stop_reason`: `"tool_use"` if any function
    /// call was emitted (so the client executes the tool and continues
    /// the conversation), otherwise the upstream's status maps to
    /// `end_turn` / `max_tokens`. Without this flag every Responses
    /// stream would surface as `end_turn`, and clients (Claude Code)
    /// would never invoke the tool — they'd just stop after the model
    /// emitted the function call.
    has_tool_calls: bool,
    final_stop_reason: Option<String>,
    final_usage: Option<crate::responses::ResponsesUsage>,
}

impl ResponsesStreamTranslator {
    pub fn new(message_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            message_id: message_id.into(),
            model: model.into(),
            started: false,
            block_index: 0,
            block_map: std::collections::HashMap::new(),
            closed_blocks: std::collections::HashSet::new(),
            fc_item_index: std::collections::HashMap::new(),
            has_tool_calls: false,
            final_stop_reason: None,
            final_usage: None,
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<StreamEvent>) {
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
    }

    fn allocate_block(&mut self, output_index: u32) -> u32 {
        if let Some(&existing) = self.block_map.get(&output_index) {
            existing
        } else {
            let idx = self.block_index;
            self.block_index += 1;
            self.block_map.insert(output_index, idx);
            idx
        }
    }

    pub fn push_event(&mut self, event: &ResponsesStreamEvent) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        match event {
            ResponsesStreamEvent::ResponseCreated { response }
            | ResponsesStreamEvent::ResponseInProgress { response } => {
                self.ensure_started(&mut out);
                // Capture id/model from the response in case it differs.
                if !response.id.is_empty() {
                    self.message_id = response.id.clone();
                }
                if !response.model.is_empty() {
                    self.model = response.model.clone();
                }
            }
            ResponsesStreamEvent::ResponseOutputItemAdded { output_index, item } => {
                self.ensure_started(&mut out);
                let block_idx = self.allocate_block(*output_index);
                if let OutputItem::FunctionCall { id, .. } = item {
                    self.has_tool_calls = true;
                    // Remember which `output_index` opened this
                    // function-call block. Subsequent args events
                    // carry `item_id`; we'll use it to route their
                    // deltas to the same block even if the upstream
                    // reports a different `output_index`.
                    self.fc_item_index.insert(id.clone(), *output_index);
                }
                let block = output_item_to_block(item);
                out.push(StreamEvent::ContentBlockStart {
                    index: block_idx,
                    content_block: block,
                });
            }
            ResponsesStreamEvent::ResponseContentPartAdded { output_index, part, .. } => {
                // A single Responses item may have multiple content parts.
                // For now, fold them into the same block — the first part
                // opens the block on `output_item.added`. Extra parts emit
                // their own start if we already opened a Text block and
                // this is something different (rare).
                let _ = (output_index, part);
            }
            ResponsesStreamEvent::ResponseOutputTextDelta {
                output_index,
                delta,
                ..
            } => {
                self.ensure_started(&mut out);
                let block_idx = self.allocate_block(*output_index);
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::TextDelta { text: delta.clone() },
                });
            }
            ResponsesStreamEvent::ResponseOutputTextDone { output_index, .. } => {
                let block_idx = self.allocate_block(*output_index);
                if self.closed_blocks.insert(block_idx) {
                    out.push(StreamEvent::ContentBlockStop { index: block_idx });
                }
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
                item_id,
                output_index,
                delta,
                ..
            } => {
                self.ensure_started(&mut out);
                // Route the delta to the block opened by
                // `output_item.added`, not to whatever `output_index`
                // the upstream put on this event — see `fc_item_index`
                // doc. Fall back to the event's own `output_index` if
                // we somehow never saw the item.added (shouldn't
                // happen on a well-formed stream).
                let block_output_index = self
                    .fc_item_index
                    .get(item_id)
                    .copied()
                    .unwrap_or(*output_index);
                let block_idx = self.allocate_block(block_output_index);
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: delta.clone(),
                    },
                });
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
                item_id,
                output_index,
                ..
            } => {
                let block_output_index = self
                    .fc_item_index
                    .get(item_id)
                    .copied()
                    .unwrap_or(*output_index);
                let block_idx = self.allocate_block(block_output_index);
                // Also close the block that was opened for the
                // event's own `output_index`, in case any stray delta
                // got routed there before this fix landed (or in case
                // the upstream mixes indices for unrelated reasons).
                let stray_block_idx = self.allocate_block(*output_index);
                if self.closed_blocks.insert(block_idx) {
                    out.push(StreamEvent::ContentBlockStop { index: block_idx });
                }
                self.closed_blocks.insert(stray_block_idx);
            }
            ResponsesStreamEvent::ResponseOutputItemDone { .. } => {
                // Closing a whole output item — already covered by the
                // per-text / per-function-call done events above. No-op.
            }
            ResponsesStreamEvent::ResponseContentPartDone { .. } => {}
            ResponsesStreamEvent::ResponseCompleted { response }
            | ResponsesStreamEvent::ResponseFailed { response }
            | ResponsesStreamEvent::ResponseIncomplete { response } => {
                self.ensure_started(&mut out);
                self.final_usage = response.usage.clone();
                // Tool calls take precedence over the upstream status:
                // an `incomplete` (length-capped) response with a tool
                // call still surfaces as `tool_use` so the client runs
                // the tool. Status-based reasons only apply when no
                // tool was emitted.
                self.final_stop_reason = Some(if self.has_tool_calls {
                    "tool_use".to_string()
                } else {
                    match response.status.as_str() {
                        "incomplete" => "max_tokens".to_string(),
                        "completed" => "end_turn".to_string(),
                        "failed" => "end_turn".to_string(),
                        _ => "end_turn".to_string(),
                    }
                });
            }
            ResponsesStreamEvent::Unknown => {}
        }
        out
    }

    /// Returns true iff `event` is a terminal response event that
    /// closes the upstream stream (`response.completed`, `failed`,
    /// or `incomplete`). The transport calls `finish()` on these so
    /// the message delta + stop get emitted even when the upstream
    /// omits the `[DONE]` sentinel.
    pub fn is_terminal(event: &ResponsesStreamEvent) -> bool {
        matches!(
            event,
            ResponsesStreamEvent::ResponseCompleted { .. }
                | ResponsesStreamEvent::ResponseFailed { .. }
                | ResponsesStreamEvent::ResponseIncomplete { .. }
        )
    }

    pub fn finalize(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if !self.started {
            return out;
        }
        // Close any blocks that weren't already closed by an explicit
        // `*.done` event. Emit in ascending index order so the client
        // sees a deterministic sequence.
        let mut open: Vec<u32> = self
            .block_map
            .values()
            .copied()
            .filter(|idx| !self.closed_blocks.contains(idx))
            .collect();
        open.sort_unstable();
        for block_idx in open {
            out.push(StreamEvent::ContentBlockStop { index: block_idx });
        }
        let stop_reason = self.final_stop_reason.take();
        let usage = self.final_usage.take().map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: u
                .input_tokens_details
                .as_ref()
                .filter(|d| d.cached_tokens > 0)
                .map(|d| d.cached_tokens),
            cache_creation: None,
            server_tool_use: None,
            output_tokens_details: None,
            service_tier: None,
            inference_geo: None,
        });
        out.push(StreamEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason,
                stop_sequence: None,
                stop_details: None,
                container: None,
            },
            usage,
        });
        out.push(StreamEvent::MessageStop);
        out
    }
}

fn output_item_to_block(item: &OutputItem) -> ResponseBlock {
    match item {
        // `content_block_start` must open an *empty* block. The text
        // arrives via `response.output_text.delta` events. Seeding the
        // block from the item's content would emit the reply twice when
        // the upstream includes a text snapshot on `output_item.added`
        // (Copilot's gpt-5.x does this): once here and again through the
        // deltas, which the client concatenates into a duplicated reply.
        OutputItem::Message { .. } => ResponseBlock::Text {
            text: String::new(),
            citations: None,
        },
        OutputItem::FunctionCall { call_id, name, .. } => {
            // Arguments arrive via `response.function_call_arguments.delta`
            // and the Anthropic client builds tool input purely from the
            // concatenated `input_json_delta` fragments. Open with empty
            // input so a snapshot on `output_item.added` can't collide
            // with the streamed fragments.
            ResponseBlock::ToolUse {
                id: call_id.clone(),
                name: name.clone(),
                input: serde_json::Value::Object(Default::default()),
                caller: None,
            }
        }
        OutputItem::Unknown => ResponseBlock::Text {
            text: String::new(),
            citations: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{
        OutputContentPart, ResponsesResponse, ResponsesUsage,
    };
    use serde_json::json;

    fn placeholder_response(status: &str) -> ResponsesResponse {
        ResponsesResponse {
            id: "resp_1".into(),
            object: "response".into(),
            created_at: 0,
            model: "gpt-5".into(),
            status: status.into(),
            output: vec![],
            incomplete_details: None,
            usage: Some(ResponsesUsage::default()),
            extra: json!({}),
        }
    }

    #[test]
    fn created_event_emits_message_start_once() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let events = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::MessageStart { .. }));

        // A second created event must not emit another MessageStart.
        let events2 = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        assert!(events2.is_empty());
    }

    #[test]
    fn output_item_added_with_text_snapshot_does_not_duplicate_reply() {
        // Copilot's gpt-5.x includes the accumulated text in the item on
        // `output_item.added`. The content_block_start must still open
        // EMPTY — otherwise the reply is emitted twice (once in the start
        // block, once via the deltas) and the client renders it twice.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let start = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: "Hello world".into(),
                    annotations: None,
                }],
            },
        });
        match &start[0] {
            StreamEvent::ContentBlockStart { content_block: ResponseBlock::Text { text, .. }, .. } => {
                assert!(text.is_empty(), "start block must be empty, got {text:?}");
            }
            _ => panic!("expected empty Text ContentBlockStart"),
        }

        // The whole reply now arrives via deltas — exactly once.
        let d1 = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            delta: "Hello world".into(),
        });
        let accumulated: String = std::iter::once(&start[0])
            .chain(d1.iter())
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta { delta: BlockDelta::TextDelta { text }, .. } => {
                    Some(text.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(accumulated, "Hello world");
    }

    #[test]
    fn tool_call_response_emits_tool_use_stop_reason() {
        // Regression: when the Responses stream contains a function_call
        // item, finalize must emit message_delta with `stop_reason:
        // "tool_use"` (not `end_turn`), so the client executes the tool
        // and continues the conversation. Without this flag the proxy
        // would emit `end_turn` and Claude Code would just stop after
        // receiving the function call.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 0,
            delta: r#"{"city":"SF"}"#.into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 0,
            arguments: r#"{"city":"SF"}"#.into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });

        let tail = t.finalize();
        let delta = tail
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize must emit MessageDelta");
        assert_eq!(
            delta.stop_reason.as_deref(),
            Some("tool_use"),
            "tool-call response must surface stop_reason=tool_use, not end_turn"
        );
    }

    #[test]
    fn text_only_response_emits_end_turn_stop_reason() {
        // Mirror case: a message-only stream must still emit end_turn
        // when no function call happened — guarding against the
        // has_tool_calls flag being set incorrectly.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![],
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });

        let tail = t.finalize();
        let delta = tail
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize must emit MessageDelta");
        assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn output_text_done_then_finalize_emits_single_content_block_stop() {
        // `output_text.done` closes the block; finalize must NOT emit a
        // second content_block_stop for the same index.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![],
            },
        });
        let done = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            text: "hi".into(),
        });
        assert_eq!(
            done.iter()
                .filter(|e| matches!(e, StreamEvent::ContentBlockStop { index: 0 }))
                .count(),
            1
        );
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        assert_eq!(
            tail.iter()
                .filter(|e| matches!(e, StreamEvent::ContentBlockStop { .. }))
                .count(),
            0,
            "finalize must not re-close a block already closed by output_text.done"
        );
    }

    #[test]
    fn text_delta_flow_emits_block_events() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        // Open a text output item.
        let item = OutputItem::Message {
            id: "msg_x".into(),
            role: "assistant".into(),
            status: "in_progress".into(),
            content: vec![OutputContentPart::OutputText {
                text: String::new(),
                annotations: None,
            }],
        };
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item,
        });
        assert!(matches!(evs[0], StreamEvent::ContentBlockStart { index: 0, .. }));

        // Push a delta.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            delta: "hi".into(),
        });
        assert!(matches!(
            evs[0],
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::TextDelta { ref text } } if text == "hi"
        ));
    }

    #[test]
    fn function_call_arguments_delta_emits_input_json_delta() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        let item = OutputItem::FunctionCall {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: "f".into(),
            arguments: "{}".into(),
            status: "in_progress".into(),
        };
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item,
        });
        assert!(matches!(evs[0], StreamEvent::ContentBlockStart { index: 0, content_block: ResponseBlock::ToolUse { .. } }));

        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 0,
            delta: "{\"x\":1}".into(),
        });
        assert!(matches!(
            evs[0],
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::InputJsonDelta { ref partial_json } } if partial_json == "{\"x\":1}"
        ));
    }

    #[test]
    fn completed_event_triggers_message_delta_and_stop() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        let mut resp = placeholder_response("completed");
        resp.usage = Some(ResponsesUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            input_tokens_details: None,
            output_tokens_details: None,
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(resp),
        });
        // No immediate emit — finalize() emits the message_delta + stop.
        assert!(evs.is_empty());

        let tail = t.finalize();
        assert_eq!(tail.len(), 2);
        assert!(matches!(tail[0], StreamEvent::MessageDelta { .. }));
        assert!(matches!(tail[1], StreamEvent::MessageStop));
        if let StreamEvent::MessageDelta { delta, usage } = &tail[0] {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            let usage = usage.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
        }
    }

    #[test]
    fn finalize_without_started_emits_nothing() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        assert!(t.finalize().is_empty());
    }

    #[test]
    fn failed_status_maps_to_end_turn() {
        // response.failed is treated like completed for stop_reason
        // purposes — both surface as end_turn on the Anthropic side,
        // so the client sees a normal terminal event.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let tail = t.push_event(&ResponsesStreamEvent::ResponseFailed {
            response: Box::new(placeholder_response("failed")),
        });
        assert!(tail.is_empty(), "failed event alone must not emit immediately");
        let final_events = t.finalize();
        let message_delta = final_events
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize should emit MessageDelta");
        assert_eq!(message_delta.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn incomplete_status_maps_to_max_tokens() {
        // response.incomplete surfaces as max_tokens so the client
        // knows the response was cut off at the token budget.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseIncomplete {
            response: Box::new(placeholder_response("incomplete")),
        });
        let final_events = t.finalize();
        let message_delta = final_events
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize should emit MessageDelta");
        assert_eq!(message_delta.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn multiple_output_items_get_distinct_block_indices() {
        // The Responses stream can emit several output items in
        // parallel (e.g. text + function_call). Each must map to a
        // distinct Anthropic block index, so deltas for output_index=1
        // don't leak into output_index=0's block.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        // First block is at index 0.
        assert!(matches!(evs[0], StreamEvent::ContentBlockStart { index: 0, .. }));

        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_b".into(),
                call_id: "call_b".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        // Second block is at index 1, not 0.
        assert!(matches!(evs[0], StreamEvent::ContentBlockStart { index: 1, .. }));

        // A delta on output_index=1 must land on block 1.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_b".into(),
            output_index: 1,
            delta: "{\"x\":1}".into(),
        });
        assert!(matches!(
            evs[0],
            StreamEvent::ContentBlockDelta { index: 1, .. }
        ));
    }

    #[test]
    fn content_part_added_and_done_events_are_noops() {
        // response.content_part.added / done are not modeled — they
        // must not panic and must not emit extra events. (For now we
        // fold multiple parts into the same block opened by
        // response.output_item.added.)
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseContentPartAdded {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            part: OutputContentPart::OutputText {
                text: String::new(),
                annotations: None,
            },
        });
        assert!(evs.is_empty());
        let evs = t.push_event(&ResponsesStreamEvent::ResponseContentPartDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            part: OutputContentPart::OutputText {
                text: "full".into(),
                annotations: None,
            },
        });
        assert!(evs.is_empty());
    }

    #[test]
    fn finalize_closes_unclosed_blocks() {
        // If the upstream stream ends mid-block (no
        // response.output_item.done), finalize() must still emit
        // content_block_stop for every block we've opened so the
        // client's content block accounting stays balanced.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        // Stream ends without done events for this block.
        let final_events = t.finalize();
        let stops: Vec<u32> = final_events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(stops, vec![0]);
        // And the tail must include message_delta + message_stop.
        assert!(matches!(final_events.last(), Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn unknown_stream_event_does_not_panic_or_emit() {
        // Future Responses API events we haven't modeled fall through
        // to the Unknown variant. The translator must skip them
        // without producing phantom Anthropic events.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let evs = t.push_event(&ResponsesStreamEvent::Unknown);
        assert!(evs.is_empty());
    }

    #[test]
    fn in_progress_event_starts_message_and_captures_id_model() {
        // response.in_progress shares the ResponseCreated arm (line 79).
        // It must open the message stream; the id/model capture (lines
        // 82-87) updates internal state AFTER ensure_started has already
        // emitted MessageStart, so that first event still carries the
        // constructor placeholders — the captured values would surface
        // on later events. We assert MessageStart is emitted exactly
        // once here (the capture path is exercised regardless).
        let mut t = ResponsesStreamTranslator::new("msg_placeholder", "unset");
        let mut resp = placeholder_response("in_progress");
        resp.id = "resp_real".into();
        resp.model = "gpt-5-mini".into();
        let evs = t.push_event(&ResponsesStreamEvent::ResponseInProgress {
            response: Box::new(resp),
        });
        assert_eq!(evs.len(), 1);
        assert!(matches!(evs[0], StreamEvent::MessageStart { .. }));
        // A second in_progress must not re-emit MessageStart.
        let evs2 = t.push_event(&ResponsesStreamEvent::ResponseInProgress {
            response: Box::new(placeholder_response("in_progress")),
        });
        assert!(evs2.is_empty());
    }

    #[test]
    fn output_text_done_emits_content_block_stop() {
        // response.output_text.done closes the text block (lines 118-121)
        // by mapping the output_index back to its allocated block and
        // emitting content_block_stop.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            text: "done".into(),
        });
        assert_eq!(evs.len(), 1);
        assert!(matches!(evs[0], StreamEvent::ContentBlockStop { index: 0 }));
    }

    #[test]
    fn function_call_arguments_done_emits_content_block_stop() {
        // response.function_call_arguments.done closes the tool_use
        // block (lines 136-139).
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "f".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 0,
            arguments: "{\"x\":1}".into(),
        });
        assert_eq!(evs.len(), 1);
        assert!(matches!(evs[0], StreamEvent::ContentBlockStop { index: 0 }));
    }

    #[test]
    fn output_item_done_is_a_noop() {
        // response.output_item.done is redundant with the per-text /
        // per-function-call done events (lines 140-143). It must not
        // emit anything on its own.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemDone {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "completed".into(),
                content: vec![OutputContentPart::OutputText {
                    text: "full".into(),
                    annotations: None,
                }],
            },
        });
        assert!(evs.is_empty());
    }

    #[test]
    fn unknown_final_status_defaults_to_end_turn() {
        // A completed/failed/incomplete event whose status string is
        // none of the recognized values still resolves to end_turn via
        // the catch-all arm (line 154). We drive this through the
        // ResponseCompleted variant carrying an odd status.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("some_new_status")),
        });
        let final_events = t.finalize();
        let delta = final_events
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize emits MessageDelta");
        assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn output_item_added_with_unknown_content_part_opens_empty_text_block() {
        // A message item whose only content part is an unrecognized type
        // yields an empty-text block (output_item_to_block: the Unknown
        // part is skipped by find_map → unwrap_or_default, line 203).
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_u".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::Unknown],
            },
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            StreamEvent::ContentBlockStart { index: 0, content_block } => {
                match content_block {
                    ResponseBlock::Text { text, .. } => assert!(text.is_empty()),
                    _ => panic!("expected empty Text block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn text_then_function_call_no_duplicate_block_stop_on_finalize() {
        // Regression for the live trace: when a response carries a
        // text message followed by a function_call item, finalize()
        // must NOT re-emit content_block_stop for the function_call
        // block (which was already closed by
        // response.function_call_arguments.done).
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Text message item at output_index=0.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            delta: "Hello".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            text: "Hello".into(),
        });
        // Function-call item at output_index=1.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 1,
            delta: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 1,
            arguments: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemDone {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{\"x\":1}".into(),
                status: "completed".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });

        let tail = t.finalize();
        let stops: Vec<u32> = tail
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert!(
            stops.is_empty(),
            "finalize must not re-close the function_call block (already \
             closed by function_call_arguments.done), got content_block_stop \
             indices: {stops:?}"
        );
    }

    #[test]
    fn function_call_args_with_mismatched_output_index_does_not_leak_block() {
        // The Copilot Responses stream sometimes uses different
        // `output_index` values for the `output_item.added` event of a
        // function call and the subsequent `function_call_arguments.*`
        // events. If we naively trust `output_index` everywhere,
        // allocate_block() creates two different internal block
        // indices for the same item, and `finalize()` ends up
        // emitting `content_block_stop` for a block that was already
        // closed by `function_call_arguments.done` — which causes
        // Claude Code to hang / drop the tool call.
        //
        // The trace we saw was:
        //   response.output_item.added           output_index=1 (function call)
        //   response.function_call_arguments.delta  output_index=0
        //   response.function_call_arguments.done   output_index=0
        //   response.output_item.done            output_index=1
        // `finalize()` then emitted a `content_block_stop` for block 1.
        //
        // Until upstream is fixed, the translator must heal the
        // mismatch by closing the function-call block when its
        // `arguments.done` arrives even if the index differs from
        // `output_item.added`.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Text message at output_index=0 (closes cleanly).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            text: "Hello".into(),
        });
        // Function call item opens at output_index=1.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        // But its args events report output_index=0 (mismatch).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 0,
            delta: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 0,
            arguments: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemDone {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{\"x\":1}".into(),
                status: "completed".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });

        let tail = t.finalize();
        let stops: Vec<u32> = tail
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert!(
            stops.is_empty(),
            "finalize must not re-emit content_block_stop for the function \
             call block even when its arguments events used a different \
             output_index than the item.added event; got stops at {stops:?}"
        );
    }

    #[test]
    fn text_then_function_call_no_duplicate_block_stop_total() {
        // Stronger version of the previous test: count ALL
        // content_block_stop events emitted across the whole
        // translator lifecycle (push_event + finalize). There must
        // be exactly one stop per opened block, not two.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Text message item at output_index=0.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_a".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            delta: "Hello".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_a".into(),
            output_index: 0,
            content_index: 0,
            text: "Hello".into(),
        });
        // Function-call item at output_index=1.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
                status: "in_progress".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 1,
            delta: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 1,
            arguments: "{\"x\":1}".into(),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemDone {
            output_index: 1,
            item: OutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{\"x\":1}".into(),
                status: "completed".into(),
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();

        let mut all_events = Vec::new();
        for ev in tail {
            all_events.push(ev);
        }
        let mut all_stops: Vec<u32> = Vec::new();
        let mut content_block_starts = 0;
        for ev in &all_events {
            match ev {
                StreamEvent::ContentBlockStart { .. } => content_block_starts += 1,
                StreamEvent::ContentBlockStop { index } => all_stops.push(*index),
                _ => {}
            }
        }
        // Two opens (text + function call), so exactly two stops, one per
        // block, no duplicates. (The finalize-only count is checked in
        // the companion test.)
        let mut sorted = all_stops.clone();
        sorted.sort_unstable();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(
            sorted.len(),
            content_block_starts,
            "exactly one content_block_stop per opened block; \
             got {content_block_starts} starts but stops at {sorted:?}"
        );
        assert_eq!(
            sorted,
            deduped,
            "no duplicate content_block_stop indices: {sorted:?}"
        );
    }

    #[test]
    fn output_item_added_with_unknown_item_opens_empty_text_block() {
        // An unrecognized OutputItem type maps to an empty Text block
        // (output_item_to_block Unknown arm, lines 222-224) so the block
        // accounting stays balanced even for item types we don't model.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Unknown,
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            StreamEvent::ContentBlockStart { index: 0, content_block } => {
                match content_block {
                    ResponseBlock::Text { text, .. } => assert!(text.is_empty()),
                    _ => panic!("expected empty Text block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }
}
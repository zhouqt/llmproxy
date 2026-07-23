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
    final_stop_reason: Option<String>,
    final_usage: Option<crate::responses::ResponsesUsage>,
    /// Set to true when the stream contains at least one function_call
    /// output item. Forces the final stop_reason to `tool_use` so Claude
    /// Code does not interpret `end_turn` as "model finished speaking"
    /// and drop the tool call.
    has_tool_calls: bool,
    /// Maps `function_call` item IDs to their `output_index`, recorded on
    /// `output_item.added`. Copilot sometimes sends function-call argument
    /// deltas with a *different* `output_index` than the item was created
    /// with. By routing through this map we land on the correct block
    /// instead of creating a phantom block for the mismatched index.
    fc_item_index: std::collections::HashMap<String, u32>,
    /// Set of Anthropic block indices that have received at least one
    /// `output_text.delta`. Used to detect text blocks that only carry
    /// content on the `output_text.done` event (snapshot-only, no deltas
    /// arrived). When no delta was seen, the done event's `text` is emitted
    /// as a fallback text delta so the client does not see an empty block.
    deltas_seen: std::collections::HashSet<u32>,
    /// Set to true when an upstream `error` SSE event was handled.
    /// Signals to the adapter that it should stop processing the stream
    /// immediately and avoid calling `finalize()` on EOF.
    pub(crate) finalized: bool,
}

/// Returns `true` for SSE events that signal the upstream response is
/// complete: `response.completed`, `response.failed`, or
/// `response.incomplete`. When one of these arrives without a subsequent
/// `[DONE]` sentinel (common in Copilot's gpt-5.x responses), the
/// adapter layer should call `finalize()` immediately rather than waiting
/// for `[DONE]` or EOF.
///
/// This is intentional: once a terminal event is received the adapter
/// stops polling the upstream stream and calls `finalize()`, which
/// causes any data that arrives after the terminal event to be
/// discarded. The upstream is not expected to send further meaningful
/// data past this point.
pub fn is_terminal_event(event: &ResponsesStreamEvent) -> bool {
    matches!(
        event,
        ResponsesStreamEvent::ResponseCompleted { .. }
            | ResponsesStreamEvent::ResponseFailed { .. }
            | ResponsesStreamEvent::ResponseIncomplete { .. }
    )
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
            final_stop_reason: None,
            final_usage: None,
            has_tool_calls: false,
            fc_item_index: std::collections::HashMap::new(),
            deltas_seen: std::collections::HashSet::new(),
            finalized: false,
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
                if let OutputItem::FunctionCall { id, .. } = item {
                    self.has_tool_calls = true;
                    self.fc_item_index.insert(id.clone(), *output_index);
                }
                let block_idx = self.allocate_block(*output_index);
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
                let Some(&block_idx) = self.block_map.get(output_index) else {
                    tracing::warn!(?output_index, "text delta for unseen block; ignoring");
                    return out;
                };
                self.deltas_seen.insert(block_idx);
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::TextDelta { text: delta.clone() },
                });
            }
            ResponsesStreamEvent::ResponseOutputTextDone { output_index, text, .. } => {
                let Some(&block_idx) = self.block_map.get(output_index) else {
                    tracing::warn!(?output_index, "text.done for unseen block; ignoring");
                    return out;
                };
                // If this text block never received any delta, emit the
                // done event's full text as a fallback delta. This handles
                // the case where the upstream sends the complete reply only
                // on the done event (snapshot-only, no incremental deltas).
                if !self.deltas_seen.contains(&block_idx) {
                    out.push(StreamEvent::ContentBlockDelta {
                        index: block_idx,
                        delta: BlockDelta::TextDelta { text: text.clone() },
                    });
                }
                if self.closed_blocks.insert(block_idx) {
                    out.push(StreamEvent::ContentBlockStop { index: block_idx });
                }
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
                output_index,
                delta,
                item_id,
                ..
            } => {
                self.ensure_started(&mut out);
                // Route by item_id when available — Copilot sometimes
                // sends deltas with a different output_index than the
                // item was created with.
                let fc_index = self
                    .fc_item_index
                    .get(item_id)
                    .copied()
                    .unwrap_or(*output_index);
                let Some(&block_idx) = self.block_map.get(&fc_index) else {
                    tracing::warn!(?output_index, ?item_id, ?fc_index, "fc args delta for unseen block; ignoring");
                    return out;
                };
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: delta.clone(),
                    },
                });
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
                output_index,
                item_id,
                ..
            } => {
                let fc_index = self
                    .fc_item_index
                    .get(item_id)
                    .copied()
                    .unwrap_or(*output_index);
                let Some(&block_idx) = self.block_map.get(&fc_index) else {
                    tracing::warn!(?fc_index, ?item_id, "fc_args.done for unseen block; ignoring");
                    return out;
                };
                if self.closed_blocks.insert(block_idx) {
                    out.push(StreamEvent::ContentBlockStop { index: block_idx });
                }
            }
            ResponsesStreamEvent::ResponseOutputItemDone { .. } => {
                // Closing a whole output item — already covered by the
                // per-text / per-function-call done events above. No-op.
            }
            ResponsesStreamEvent::ResponseContentPartDone { .. } => {}
            ResponsesStreamEvent::ResponseCompleted { response }
            | ResponsesStreamEvent::ResponseIncomplete { response } => {
                self.ensure_started(&mut out);
                self.final_usage = response.usage.clone();
                self.final_stop_reason = Some(match response.status.as_str() {
                    "incomplete" => "max_tokens".to_string(),
                    "completed" => "end_turn".to_string(),
                    _ => "end_turn".to_string(),
                });
            }
            ResponsesStreamEvent::ResponseFailed { response } => {
                self.ensure_started(&mut out);
                let msg = response
                    .extra
                    .get("error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| "upstream error".to_string());
                let code = response
                    .extra
                    .get("error")
                    .and_then(|v| v.get("code"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                let mut error_body = serde_json::json!({
                    "type": "upstream_error",
                    "message": msg,
                });
                if let Some(ref c) = code {
                    error_body["code"] = serde_json::Value::String(c.clone());
                }
                out.push(StreamEvent::Error { error: error_body });
                self.final_stop_reason = None;
                self.finalized = true;
            }
            ResponsesStreamEvent::Error {
                code,
                message,
                extra,
                ..
            } => {
                self.ensure_started(&mut out);
                let msg = message
                    .as_ref()
                    .cloned()
                    .or_else(|| extra.get("error").and_then(|v| v.get("message")).and_then(|v| v.as_str().map(|s| s.to_owned())))
                    .unwrap_or_else(|| "upstream error".to_string());
                let mut error_body = serde_json::json!({
                    "type": "upstream_error",
                    "message": msg,
                });
                if let Some(ref code) = code {
                    error_body["code"] = serde_json::Value::String(code.clone());
                }
                out.push(StreamEvent::Error { error: error_body });
                // Clear final_stop_reason so EOF finalize does not emit
                // message_delta after the error event.
                self.final_stop_reason = None;
                self.finalized = true;
            }
            ResponsesStreamEvent::Unknown => {}
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
        // If the stream produced function_call items, the stop reason
        // must be "tool_use" regardless of response.status. Claude Code
        // uses stop_reason to decide whether to execute tools; an
        // "end_turn" when tools are present causes the client to discard
        // the tool_use blocks.
        let stop_reason = if self.has_tool_calls {
            Some("tool_use".to_string())
        } else {
            stop_reason
        };
        let raw = self.final_usage.take().unwrap_or_default();
        let usage = Some(Usage {
            input_tokens: raw.input_tokens,
            output_tokens: raw.output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: raw
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
    fn failed_status_emits_error_and_finalizes() {
        // response.failed now surfaces upstream error details instead of
        // mapping silently to end_turn.
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let tail = t.push_event(&ResponsesStreamEvent::ResponseFailed {
            response: Box::new(placeholder_response("failed")),
        });
        // Must emit an Error event immediately (no end_turn).
        assert!(
            tail.iter().any(|e| matches!(e, StreamEvent::Error { .. })),
            "response.failed must emit Error event"
        );
        assert!(
            !tail.iter().any(|e| matches!(e, StreamEvent::MessageDelta { .. })),
            "response.failed must not emit MessageDelta"
        );
        assert!(t.finalized, "translator must be finalized after response.failed");
        // finalize() should emit nothing.
        let final_events = t.finalize();
        assert!(final_events.is_empty(), "finalize after response.failed must be empty");
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
        // response.in_progress shares the ResponseCreated arm.
        // It must open the message stream; the id/model capture
        // updates internal state AFTER ensure_started has already
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
        // response.output_text.done closes the text block
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
        // No deltas were sent, so the done text is emitted as a fallback
        // delta before the stop (P1-1: snapshot-only text blocks).
        assert_eq!(evs.len(), 2);
        assert!(matches!(
            evs[0],
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::TextDelta { .. } }
        ));
        assert!(matches!(evs[1], StreamEvent::ContentBlockStop { index: 0 }));
    }

    #[test]
    fn function_call_arguments_done_emits_content_block_stop() {
        // response.function_call_arguments.done closes the tool_use
        // block.
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
        // per-function-call done events. It must not
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
        // the catch-all arm. We drive this through the
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
        // part is skipped by find_map → unwrap_or_default).
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
    fn output_item_added_with_unknown_item_opens_empty_text_block() {
        // An unrecognized OutputItem type maps to an empty Text block
        // (output_item_to_block Unknown arm) so the block
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

    /// T1: P0-1 regression — a stream containing a function_call output
    /// item must produce `stop_reason: "tool_use"` in the final
    /// MessageDelta, even when response.status says "completed". Without
    /// this fix, the translator falls through to the status-based mapping
    /// and emits `end_turn`, causing Claude Code to discard the tool call.
    #[test]
    fn streaming_function_call_response_reports_tool_use_stop_reason() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Add a function_call output item — no text deltas.
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
        // Push an arguments delta so the block gets a delta event.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 0,
            delta: "{\"x\":1}".into(),
        });
        // Stream completes with status "completed".
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        let message_delta = tail
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, .. } => Some(delta),
                _ => None,
            })
            .expect("finalize should emit MessageDelta");
        assert_eq!(
            message_delta.stop_reason.as_deref(),
            Some("tool_use"),
            "function_call stream must report tool_use, not end_turn"
        );
    }

    /// T3: P0-3 regression — Copilot sometimes sends
    /// `function_call_arguments.delta` / `.done` with a *different*
    /// `output_index` than the `output_item.added` used. The translator
    /// must route by `item_id` via `fc_item_index`, not by the raw
    /// `output_index` on the delta/done event.
    #[test]
    fn function_call_args_with_mismatched_output_index_routes_by_item_id() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Add a function_call with output_index=0, item_id="fc_1".
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
        // Delta arrives with output_index=1 (mismatched!) but correct
        // item_id. Must route to block 0, not create a phantom block 1.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            item_id: "fc_1".into(),
            output_index: 1,
            delta: "{\"x\":1}".into(),
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::InputJsonDelta { .. } } => {}
            other => panic!("expected delta on block 0, got {other:?}"),
        }
        // Done also arrives with mismatched output_index=1.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_1".into(),
            output_index: 1,
            arguments: "{\"x\":1}".into(),
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            StreamEvent::ContentBlockStop { index: 0 } => {}
            other => panic!("expected stop on block 0, got {other:?}"),
        }
        // Finalize should not emit any extra content_block_stop (block 0
        // already closed, and no phantom block 1 exists).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        let stops: Vec<&StreamEvent> = tail
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockStop { .. }))
            .collect();
        assert!(stops.is_empty(), "no extra stops after finalize: {stops:?}");
    }

    /// T5 (responses_stream.rs): P0-5 regression — `response.created` with
    /// `"usage": null` must not prevent finalization. The translator
    /// should produce default-zero usage rather than erroring.
    #[test]
    fn response_created_with_null_usage_decodes_and_finalizes_cleanly() {
        // Build a response.created event with None usage (simulates the
        // JSON `"usage": null` case which deserializes to Option::None).
        let resp = ResponsesResponse {
            id: "resp_n".into(),
            object: "response".into(),
            created_at: 0,
            model: "gpt-5".into(),
            status: "in_progress".into(),
            output: vec![],
            incomplete_details: None,
            usage: None,
            extra: json!({}),
        };
        let mut t = ResponsesStreamTranslator::new("msg_n", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(resp),
        });
        // A text block so finalize has something to emit.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: None,
                }],
            },
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            delta: "hello".into(),
        });
        // Complete with a response that ALSO has None usage.
        let complete_resp = ResponsesResponse {
            id: "resp_n".into(),
            object: "response".into(),
            created_at: 0,
            model: "gpt-5".into(),
            status: "completed".into(),
            output: vec![],
            incomplete_details: None,
            usage: None,
            extra: json!({}),
        };
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(complete_resp),
        });
        let tail = t.finalize();
        let message_delta = tail
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta { delta, usage } => Some((delta, usage)),
                _ => None,
            })
            .expect("finalize should emit MessageDelta");
        // stop_reason defaults to "end_turn" for completed status
        assert_eq!(message_delta.0.stop_reason.as_deref(), Some("end_turn"));
        // usage should be Some with default values when upstream usage is None
        let usage = message_delta.1.as_ref().expect("usage should be Some, not None");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    /// T6: P1-1 regression — text blocks that only carry content on the
    /// `output_text.done` event (no `output_text.delta` in between) must
    /// emit the done text as a fallback delta. Without this fix the client
    /// sees an empty content block because the start block was intentionally
    /// seeded empty (db0e67b) and no delta ever arrived.
    #[test]
    fn text_block_without_deltas_emits_done_text_as_fallback_delta() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Open a text output item with a snapshot (seeded empty).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![OutputContentPart::OutputText {
                    text: "snapshot".into(),
                    annotations: None,
                }],
            },
        });
        // NO deltas — only the done event carrying the full text.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            text: "Hello from done".into(),
        });
        // Must emit a fallback text delta before the stop.
        assert_eq!(evs.len(), 2);
        match &evs[0] {
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::TextDelta { text } } => {
                assert_eq!(text, "Hello from done");
            }
            other => panic!("expected fallback text delta on block 0, got {other:?}"),
        }
        match &evs[1] {
            StreamEvent::ContentBlockStop { index: 0 } => {}
            other => panic!("expected stop on block 0, got {other:?}"),
        }
        // Finalize must not emit an extra delta or stop for this block.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        assert!(!tail.iter().any(|e| matches!(e, StreamEvent::ContentBlockStop { .. })));
        assert!(!tail.iter().any(|e| matches!(e, StreamEvent::ContentBlockDelta { .. })));
    }

    /// T8: P1-5 — a `text.done` event for an output_index that was never
    /// opened by `output_item.added` must be silently ignored (not create
    /// a phantom block via allocate_block).
    #[test]
    fn done_event_for_unseen_output_index_is_ignored() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Send text.done for index 0 without any prior output_item.added
        // or text.delta — should not allocate a phantom block.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            text: "orphan text".into(),
        });
        assert!(
            evs.is_empty(),
            "must not emit events for unseen output_index"
        );
        // Complete and finalize: must not emit a stop for index 0 either.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        assert!(!tail.iter().any(|e| matches!(e, StreamEvent::ContentBlockStop { .. })));
    }

    /// T9: P1-5 — a `function_call_arguments.done` with an item_id that
    /// was never registered (no prior `output_item.added` for that fc
    /// item) must not allocate a phantom block via block_map.
    #[test]
    fn fc_args_done_with_unknown_item_id_does_not_allocate_phantom() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Send function_call_arguments.done for an item_id unseen.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "unknown_fc".into(),
            output_index: 0,
            arguments: "{}".into(),
        });
        assert!(
            evs.is_empty(),
            "must not emit events for unknown fc item_id"
        );
        // Finalize: must not emit a stop for the phantom block.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        assert!(!tail.iter().any(|e| matches!(e, StreamEvent::ContentBlockStop { .. })));
    }

    /// T15: P1-G — calling finalize() after an Error event has already
    /// set finalized=true must return an empty vec (no message_delta or
    /// message_stop re-emitted).
    #[test]
    fn finalize_after_error_event_emits_nothing() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        // Push an Error event — this sets finalized=true and stores the
        // raw error payload. The returned StreamEvents contain only
        // whatever ensure_started emitted (message_start if first event).
        let _ = t.push_event(&ResponsesStreamEvent::Error {
            code: Some("server_error".into()),
            message: Some("upstream is overloaded".into()),
            param: None,
            extra: serde_json::Value::default(),
        });
        // finalize() should notice finalized=true and return empty.
        let tail = t.finalize();
        assert!(
            tail.is_empty(),
            "finalize after error must be a no-op, got {} events",
            tail.len()
        );
    }

    /// T16: P1-G — calling finalize() twice is idempotent; the second
    /// call returns an empty vec.
    #[test]
    fn finalize_called_twice_emits_nothing_second_time() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        // Prime the translator with real events so the first finalize
        // produces output.
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let first = t.finalize();
        assert!(
            !first.is_empty(),
            "first finalize must emit events"
        );
        let second = t.finalize();
        assert!(
            second.is_empty(),
            "second finalize must be a no-op, got {} events",
            second.len()
        );
    }

    /// T20: P1-1 — in a multi-part output item, once ANY part receives a
    /// text delta (`deltas_seen` is keyed by `block_idx`, not
    /// `content_index`), the done event for other parts must NOT emit a
    /// fallback delta. This pins the current invariant — the translator
    /// treats the entire output item as a single Anthropic content block
    /// and does not track per-part delta state.
    #[test]
    fn multipart_item_snapshot_fallback_invariant() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });
        // Add a multi-part output item (2 text parts).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseOutputItemAdded {
            output_index: 0,
            item: OutputItem::Message {
                id: "msg_x".into(),
                role: "assistant".into(),
                status: "in_progress".into(),
                content: vec![
                    OutputContentPart::OutputText {
                        text: String::new(),
                        annotations: None,
                    },
                    OutputContentPart::OutputText {
                        text: String::new(),
                        annotations: None,
                    },
                ],
            },
        });
        // Part 0 receives a delta — marks deltas_seen for the block.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDelta {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            delta: "delta from part 0".into(),
        });
        assert_eq!(evs.len(), 1);
        assert!(
            matches!(&evs[0], StreamEvent::ContentBlockDelta { index: 0, .. }),
            "expected delta for block 0"
        );

        // Part 1's done event — since deltas_seen already contains the
        // block index (set by part 0's delta), the done text must NOT
        // emit a second fallback ContentBlockDelta.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 1,
            text: "done text from part 1".into(),
        });
        // Must NOT contain a delta; only the stop is allowed.
        assert!(
            !evs.iter().any(|e| matches!(e, StreamEvent::ContentBlockDelta { .. })),
            "part 1 done must not emit a fallback delta when deltas_seen is set"
        );
        // Should emit ContentBlockStop (first close of the block).
        assert!(
            evs.iter().any(|e| matches!(e, StreamEvent::ContentBlockStop { index: 0 })),
            "part 1 done should close block 0"
        );

        // Part 0's done event — block already closed, no output.
        let evs = t.push_event(&ResponsesStreamEvent::ResponseOutputTextDone {
            item_id: "msg_x".into(),
            output_index: 0,
            content_index: 0,
            text: "done text from part 0".into(),
        });
        assert!(
            evs.is_empty(),
            "part 0 done must not emit anything when block is already closed"
        );

        // Finalize produces MessageDelta + MessageStop (no extra stop).
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(placeholder_response("completed")),
        });
        let tail = t.finalize();
        assert!(
            !tail.iter().any(|e| matches!(e, StreamEvent::ContentBlockStop { .. })),
            "finalize must not emit extra content_block_stop"
        );
    }

    /// T23: Error event with missing or nested `message` field still
    /// produces a terminating error event instead of panicking or being
    /// silently dropped.
    #[test]
    fn error_event_with_missing_or_nested_message_still_terminates() {
        use serde_json::json;
        // Fixture 1: top-level message present.
        let ev = serde_json::from_value::<ResponsesStreamEvent>(json!({
            "type": "error",
            "message": "oops",
        }))
        .unwrap();
        assert!(
            matches!(&ev, ResponsesStreamEvent::Error { code, message, extra, .. }
                     if code.is_none() && message.as_deref() == Some("oops") && extra.is_object())
        );

        // Fixture 2: nested error.message.
        let ev = serde_json::from_value::<ResponsesStreamEvent>(json!({
            "type": "error",
            "error": {"message": "nested"},
        }))
        .unwrap();
        assert!(
            matches!(&ev, ResponsesStreamEvent::Error { code, message, .. }
                     if code.is_none() && message.is_none())
        );
        // The nested message is accessible via extra.error.message.
        if let ResponsesStreamEvent::Error { extra, .. } = &ev {
            let nested = extra["error"]["message"].as_str().unwrap_or("");
            assert_eq!(nested, "nested", "nested error.message should be in extra");
        }

        // Fixture 3: empty body — no message at all.
        let ev = serde_json::from_value::<ResponsesStreamEvent>(json!({
            "type": "error",
        }))
        .unwrap();
        assert!(
            matches!(&ev, ResponsesStreamEvent::Error { code, message, .. }
                     if code.is_none() && message.is_none())
        );

        // Now run through the translator and verify all three produce a
        // terminating error event.
        for fixture in [
            json!({"type":"error","message":"oops"}),
            json!({"type":"error","error":{"message":"nested"}}),
            json!({"type":"error"}),
        ] {
            let ev = serde_json::from_value::<ResponsesStreamEvent>(fixture).unwrap();
            let mut t = ResponsesStreamTranslator::new("msg_t23", "gpt-4o");
            let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
                response: Box::new(placeholder_response("in_progress")),
            });
            let out = t.push_event(&ev);
            // The translator should set finalized = true, clear
            // final_stop_reason so finalize emits nothing, and emit
            // a StreamEvent::Error.
            assert!(t.finalized, "translator must be finalized after error event");
            assert!(t.final_stop_reason.is_none(), "final_stop_reason must be cleared");
            // finalize() should emit nothing.
            let tail = t.finalize();
            assert!(
                tail.iter().all(|e| !matches!(e, StreamEvent::MessageDelta { .. })),
                "finalize must not emit message_delta after error"
            );
            // The output must contain a StreamEvent::Error.
            assert!(
                out.iter().any(|e| matches!(e, StreamEvent::Error { .. })),
                "output must contain error event"
            );
        }
    }

    /// T24: response.failed event surfaces error details as an Error event
    /// instead of being silently mapped to end_turn.
    #[test]
    fn response_failed_event_surfaces_error_details_not_end_turn() {
        let mut t = ResponsesStreamTranslator::new("msg_t24", "gpt-4o");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        // Build a failed response with error details in extra.
        let mut failed_resp = placeholder_response("failed");
        failed_resp.extra = json!({"error": {"code": "rate_limited", "message": "too fast"}});

        let out = t.push_event(&ResponsesStreamEvent::ResponseFailed {
            response: Box::new(failed_resp),
        });

        // Must emit an Error event (not end_turn).
        assert!(
            out.iter().any(|e| matches!(e, StreamEvent::Error { .. })),
            "response.failed must emit an Error event"
        );
        // Must NOT emit end_turn semantics — no message_delta.
        assert!(
            !out.iter().any(|e| matches!(e, StreamEvent::MessageDelta { .. })),
            "response.failed must not emit message_delta"
        );
        // Finalized flag must be set.
        assert!(t.finalized, "translator must be finalized after response.failed");
        // final_stop_reason must be cleared.
        assert!(t.final_stop_reason.is_none(), "final_stop_reason must be cleared after response.failed");

        // Verify the error payload.
        let error_ev = out.iter().find(|e| matches!(e, StreamEvent::Error { .. })).unwrap();
        let StreamEvent::Error { error } = error_ev else { unreachable!() };
        assert_eq!(error["type"], "upstream_error");
        assert_eq!(error["message"], "too fast");
        assert_eq!(error["code"], "rate_limited");
    }

    /// T24b: response.failed without error details still emits an Error
    /// event with a fallback message.
    #[test]
    fn response_failed_without_error_details_uses_fallback_message() {
        let mut t = ResponsesStreamTranslator::new("msg_t24b", "gpt-4o");
        let _ = t.push_event(&ResponsesStreamEvent::ResponseCreated {
            response: Box::new(placeholder_response("in_progress")),
        });

        let out = t.push_event(&ResponsesStreamEvent::ResponseFailed {
            response: Box::new(placeholder_response("failed")),
        });

        assert!(
            out.iter().any(|e| matches!(e, StreamEvent::Error { .. })),
            "response.failed must emit Error event even without error details"
        );
        assert!(t.finalized, "translator must be finalized");
    }
}


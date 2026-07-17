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
use crate::responses::{OutputContentPart, OutputItem, ResponsesStreamEvent};

pub struct ResponsesStreamTranslator {
    message_id: String,
    model: String,
    started: bool,
    block_index: u32,
    /// Map Responses `output_index` → Anthropic content_block index.
    /// Each Responses output item maps to one Anthropic content block.
    block_map: std::collections::HashMap<u32, u32>,
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
                usage: Usage::default(),
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
                out.push(StreamEvent::ContentBlockStop { index: block_idx });
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
                output_index,
                delta,
                ..
            } => {
                self.ensure_started(&mut out);
                let block_idx = self.allocate_block(*output_index);
                out.push(StreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: delta.clone(),
                    },
                });
            }
            ResponsesStreamEvent::ResponseFunctionCallArgumentsDone { output_index, .. } => {
                let block_idx = self.allocate_block(*output_index);
                out.push(StreamEvent::ContentBlockStop { index: block_idx });
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
                self.final_usage = Some(response.usage.clone());
                self.final_stop_reason = Some(match response.status.as_str() {
                    "incomplete" => "max_tokens".to_string(),
                    "completed" => "end_turn".to_string(),
                    "failed" => "end_turn".to_string(),
                    _ => "end_turn".to_string(),
                });
            }
            ResponsesStreamEvent::Unknown => {}
        }
        out
    }

    pub fn finalize(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if !self.started {
            return out;
        }
        // Close any blocks that weren't explicitly closed.
        for &block_idx in self.block_map.values() {
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
        });
        out.push(StreamEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason,
                stop_sequence: None,
                usage,
            },
        });
        out.push(StreamEvent::MessageStop);
        out
    }
}

fn output_item_to_block(item: &OutputItem) -> ResponseBlock {
    match item {
        OutputItem::Message { content, .. } => {
            // Use the first output_text part as the initial content.
            // Subsequent parts will be appended via deltas.
            let text = content
                .iter()
                .find_map(|p| match p {
                    OutputContentPart::OutputText { text, .. } => Some(text.clone()),
                    OutputContentPart::Unknown => None,
                })
                .unwrap_or_default();
            ResponseBlock::Text { text }
        }
        OutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } => {
            let input: serde_json::Value = serde_json::from_str(arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
            ResponseBlock::ToolUse {
                id: call_id.clone(),
                name: name.clone(),
                input,
            }
        }
        OutputItem::Unknown => ResponseBlock::Text {
            text: String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{
        ResponsesResponse, ResponsesUsage,
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
            usage: ResponsesUsage::default(),
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
        resp.usage = ResponsesUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            input_tokens_details: None,
            output_tokens_details: None,
        };
        let evs = t.push_event(&ResponsesStreamEvent::ResponseCompleted {
            response: Box::new(resp),
        });
        // No immediate emit — finalize() emits the message_delta + stop.
        assert!(evs.is_empty());

        let tail = t.finalize();
        assert_eq!(tail.len(), 2);
        assert!(matches!(tail[0], StreamEvent::MessageDelta { .. }));
        assert!(matches!(tail[1], StreamEvent::MessageStop));
        if let StreamEvent::MessageDelta { delta } = &tail[0] {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            let usage = delta.usage.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
        }
    }

    #[test]
    fn finalize_without_started_emits_nothing() {
        let mut t = ResponsesStreamTranslator::new("msg_1", "gpt-5");
        assert!(t.finalize().is_empty());
    }
}
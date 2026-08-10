//! Test-only helpers shared across modules.
//!
//! Currently holds:
//! - `JsonFieldAbsent` — wiremock matcher that asserts a JSON field is
//!   **absent** from the request body. wiremock's `body_partial_json`
//!   only checks presence; the complement is needed for "the proxy
//!   must NOT send field X when the client didn't ask" assertions
//!   (e.g. `prompt_cache_key` / `prompt_cache_retention`).
//! - `openai_request` / `chat_response` / `cache_request_with` — wire
//!   fixtures shared by the OpenAI-family provider tests
//!   (openai_compat / openai_responses); previously duplicated inline.
//!
//! Consolidating this from `openai_compat.rs` and `openai_responses.rs`
//! — both files previously carried identical inline copies (PR-10).
//!
//! Lives behind `#[cfg(test)]` so it does not bloat the release binary.

#![cfg(test)]

use crate::anthropic::MessagesRequest;
use serde_json::{json, Value};
use wiremock::{Match, Request};

/// Wire-level "field X must NOT be present in the JSON request body"
/// matcher. See module docs for rationale.
pub struct JsonFieldAbsent(pub &'static str);

impl Match for JsonFieldAbsent {
    fn matches(&self, request: &Request) -> bool {
        let body: serde_json::Value = match serde_json::from_slice(&request.body) {
            Ok(v) => v,
            Err(_) => return false,
        };
        body.get(self.0).is_none()
    }
}

/// Minimal Anthropic Messages request — the shared shape the
/// OpenAI-family provider tests (openai_compat / openai_responses) send
/// through conversion. Consolidated from two identical inline `request`
/// copies (PR-10).
pub fn openai_request(streaming: bool) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "stream": streaming,
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap()
}

/// Minimal OpenAI Chat Completions response body in wire JSON shape.
/// Consolidated from the inline copy in `openai_compat.rs` (PR-10).
pub fn chat_response() -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "world"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5
        }
    })
}

/// Anthropic request with a `cache_control` block, optionally carrying
/// `metadata.user_id`. Consolidated from two identical inline copies in
/// `openai_compat.rs` / `openai_responses.rs` (PR-10).
pub fn cache_request_with(cache_type: &str, user_id: Option<&str>) -> MessagesRequest {
    let mut v = json!({
        "model": "claude-sonnet-4.6",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "long prefix", "cache_control": {"type": cache_type}},
                {"type": "text", "text": "actual question"}
            ]
        }]
    });
    if let Some(uid) = user_id {
        v["metadata"] = json!({"user_id": uid});
    }
    serde_json::from_value(v).unwrap()
}
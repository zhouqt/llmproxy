//! Test-only helpers shared across modules.
//!
//! Currently holds:
//! - `JsonFieldAbsent` — wiremock matcher that asserts a JSON field is
//!   **absent** from the request body. wiremock's `body_partial_json`
//!   only checks presence; the complement is needed for "the proxy
//!   must NOT send field X when the client didn't ask" assertions
//!   (e.g. `prompt_cache_key` / `prompt_cache_retention`).
//!
//! Consolidating this from `openai_compat.rs` and `openai_responses.rs`
//! — both files previously carried identical inline copies (PR-10).
//!
//! Lives behind `#[cfg(test)]` so it does not bloat the release binary.

#![cfg(test)]

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
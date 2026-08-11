//! Translate Anthropic `cache_control` markers into OpenAI-style
//! `prompt_cache_key` / `prompt_cache_retention` hints that the
//! upstream actually understands.
//!
//! Both fields stay `None` when the request has no `cache_control`
//! markers, so the surrounding DTOs (with `skip_serializing_if =
//! "Option::is_none"`) emit nothing extra on the wire when the client
//! didn't ask for caching.

use crate::anthropic::{CacheControlEphemeral, ContentBlock, MessageContent, MessagesRequest, SystemPrompt};

/// Cache hints derived from an Anthropic `MessagesRequest`. Either
/// field is `None` when the request carries no `cache_control`
/// markers — that's the "client didn't ask, don't tell the upstream"
/// case.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheHints {
    /// Maps to OpenAI's `prompt_cache_key`. Sourced from
    /// `req.metadata.user_id`, which Anthropic clients already use as
    /// a session/tenant identifier.
    pub prompt_cache_key: Option<String>,
    /// Maps to OpenAI's `prompt_cache_retention`: `"in_memory"`
    /// (~5–10 min, the OpenAI default) or `"24h"`.
    pub prompt_cache_retention: Option<String>,
}

/// Derive OpenAI cache hints from `req`. Walks `system` blocks and
/// every message's content blocks; if any block carries a
/// `cache_control` marker, the function returns a `CacheHints` with
/// the strongest TTL observed plus `metadata.user_id` (when present).
/// Otherwise returns `CacheHints::default()` so the request shape is
/// byte-identical to before this code landed.
pub fn derive_cache_hints(req: &MessagesRequest) -> CacheHints {
    let mut any_marker = false;
    let mut needs_24h = false;

    if let Some(SystemPrompt::Blocks(blocks)) = &req.system {
        for b in blocks {
            if let Some(cc) = &b.cache_control {
                any_marker = true;
                if !needs_24h && cc_is_24h(cc) {
                    needs_24h = true;
                }
            }
        }
    }
    for m in &req.messages {
        if let MessageContent::Blocks(blocks) = &m.content {
            for b in blocks {
                if let Some(cc) = block_cache_control(b) {
                    any_marker = true;
                    if !needs_24h && cc_is_24h(cc) {
                        needs_24h = true;
                    }
                }
            }
        }
    }

    if !any_marker {
        return CacheHints::default();
    }
    CacheHints {
        prompt_cache_key: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        prompt_cache_retention: Some(if needs_24h { "24h" } else { "in_memory" }.to_string()),
    }
}

/// PR-13: extract `cache_control` from any content-block variant that
/// carries one. Previously only `ContentBlock::Text` was consulted, so a
/// cache_control marker on an image/document/tool block was silently
/// ignored. Every Anthropic variant that has a `cache_control` field is
/// matched here; variants without one (thinking, unknown future blocks)
/// yield `None`.
fn block_cache_control(b: &ContentBlock) -> Option<&CacheControlEphemeral> {
    match b {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::Document { cache_control, .. }
        | ContentBlock::SearchResult { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. }
        | ContentBlock::ServerToolUse { cache_control, .. }
        | ContentBlock::WebSearchToolResult { cache_control, .. }
        | ContentBlock::WebFetchToolResult { cache_control, .. }
        | ContentBlock::CodeExecutionToolResult { cache_control, .. }
        | ContentBlock::BashCodeExecutionToolResult { cache_control, .. }
        | ContentBlock::TextEditorCodeExecutionToolResult { cache_control, .. }
        | ContentBlock::ToolSearchToolResult { cache_control, .. }
        | ContentBlock::ContainerUpload { cache_control, .. }
        | ContentBlock::MidConversationSystem { cache_control, .. } => cache_control.as_ref(),
        _ => None,
    }
}

/// `cache_control.type == "ephemeral_1h"` (or `"1h"` TTL) is the only
/// Anthropic value that maps to OpenAI's `24h` retention tier;
/// everything else (incl. `ephemeral`, `ephemeral_5m`/`5m`, missing
/// `type`, future unknown values) is the default in-memory tier.
fn cc_is_24h(cc: &CacheControlEphemeral) -> bool {
    cc.ttl.as_deref() == Some("1h")
        || cc.kind == "ephemeral_1h"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn req(blocks: Value, user_id: Option<&str>) -> MessagesRequest {
        let mut v = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": blocks}]
        });
        if let Some(uid) = user_id {
            v["metadata"] = json!({"user_id": uid});
        }
        serde_json::from_value(v).unwrap()
    }

    fn text_block_with_cache(cache_type: Option<&str>) -> Value {
        let mut b = json!({"type": "text", "text": "hello"});
        if let Some(t) = cache_type {
            b["cache_control"] = json!({"type": t});
        }
        b
    }

    #[test]
    fn no_cache_control_returns_default() {
        let req = req(json!("hello"), Some("u-1"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints, CacheHints::default());
    }

    #[test]
    fn cache_control_ephemeral_emits_in_memory_and_user_id() {
        let blocks = json!([text_block_with_cache(Some("ephemeral"))]);
        let req = req(blocks, Some("u-42"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-42"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    #[test]
    fn cache_control_ephemeral_5m_emits_in_memory() {
        let blocks = json!([text_block_with_cache(Some("ephemeral_5m"))]);
        let req = req(blocks, Some("u-42"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    #[test]
    fn cache_control_ephemeral_1h_emits_24h() {
        let blocks = json!([text_block_with_cache(Some("ephemeral_1h"))]);
        let req = req(blocks, Some("u-42"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-42"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn any_24h_marker_escalates_whole_request() {
        // Two blocks: one ephemeral, one ephemeral_1h. Whole request
        // gets 24h — clients ask for the longest cache they want.
        let blocks = json!([
            text_block_with_cache(Some("ephemeral")),
            text_block_with_cache(Some("ephemeral_1h")),
        ]);
        let req = req(blocks, Some("u-7"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn cache_control_without_user_id_emits_retention_only() {
        // Client signals caching intent but didn't supply a session
        // id — we still emit retention so the upstream knows to cache,
        // but skip prompt_cache_key (so we don't pollute the cache
        // namespace with an empty/default key).
        let blocks = json!([text_block_with_cache(Some("ephemeral_1h"))]);
        let req = req(blocks, None);
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key, None);
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn cache_control_in_system_block_is_detected() {
        let mut v = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "system": [
                {"type": "text", "text": "you are a helper", "cache_control": {"type": "ephemeral_1h"}}
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "u-9"}
        });
        let req: MessagesRequest = serde_json::from_value(v.take()).unwrap();
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-9"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("24h"));
    }

    #[test]
    fn plain_text_user_content_keeps_default() {
        // text String (not blocks) — no cache_control possible, must
        // not crash and must not emit hints.
        let req = req(json!("hello"), Some("u-1"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints, CacheHints::default());
    }

    #[test]
    fn unknown_cache_control_type_defaults_to_in_memory() {
        // Future Anthropic type we don't know about: safest default
        // is in_memory (~5 min). Don't 500 over a string mismatch.
        let blocks = json!([text_block_with_cache(Some("ephemeral_99h"))]);
        let req = req(blocks, Some("u-3"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-3"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    // ── PR-13 · cache_control on non-text blocks ────────────────────────

    /// PR-13: a `cache_control` marker on an `image` block must be
    /// detected. Previously only `ContentBlock::Text` was inspected, so
    /// images with cache markers were silently ignored (no cache hints
    /// emitted).
    #[test]
    fn cache_control_on_image_block_is_detected() {
        let blocks = json!([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8="
                },
                "cache_control": {"type": "ephemeral"}
            }
        ]);
        let req = req(blocks, Some("u-img"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-img"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    /// PR-13: a `cache_control` marker on a `document` block must be
    /// detected and its `ephemeral_1h` TTL must escalate to the 24h
    /// retention tier — same semantics as a text block's marker.
    #[test]
    fn cache_control_on_document_block_escalates_to_24h() {
        let blocks = json!([
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQ="
                },
                "cache_control": {"type": "ephemeral_1h"}
            }
        ]);
        let req = req(blocks, Some("u-doc"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-doc"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("24h"));
    }

    /// PR-13: a `cache_control` marker on a `tool_use` block must be
    /// detected. Tool-result caching (re-sending prior tool outputs
    /// across turns) is a common cache target, so the marker must not be
    /// lost.
    #[test]
    fn cache_control_on_tool_use_block_is_detected() {
        let blocks = json!([
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "calculator",
                "input": {"expression": "1+1"},
                "cache_control": {"type": "ephemeral_5m"}
            }
        ]);
        let req = req(blocks, Some("u-tool"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-tool"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }

    /// PR-13: every other `ContentBlock` variant that carries a
    /// `cache_control` field (search_result, tool_result, server_tool_use,
    /// the *_tool_result families, container_upload, mid_conversation_system)
    /// must be detected too. The `block_cache_control` match must not leave
    /// a cache marker on any of these silently dropped.
    #[test]
    fn cache_control_on_all_non_text_block_kinds_is_detected() {
        let blocks = json!([
            {"type": "search_result", "source": {"type": "web"}, "cache_control": {"type": "ephemeral"}},
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok", "cache_control": {"type": "ephemeral"}},
            {"type": "server_tool_use", "id": "st_1", "name": "git", "input": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "web_search_tool_result", "tool_use_id": "t1", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "web_fetch_tool_result", "tool_use_id": "t2", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "code_execution_tool_result", "tool_use_id": "t3", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "bash_code_execution_tool_result", "tool_use_id": "t4", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "text_editor_code_execution_tool_result", "tool_use_id": "t5", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "tool_search_tool_result", "tool_use_id": "t6", "content": {}, "cache_control": {"type": "ephemeral"}},
            {"type": "container_upload", "file_id": "f1", "cache_control": {"type": "ephemeral"}},
            {"type": "mid_conversation_system", "content": "hi", "cache_control": {"type": "ephemeral"}}
        ]);
        let req = req(blocks, Some("u-many"));
        let hints = derive_cache_hints(&req);
        assert_eq!(hints.prompt_cache_key.as_deref(), Some("u-many"));
        assert_eq!(hints.prompt_cache_retention.as_deref(), Some("in_memory"));
    }
}

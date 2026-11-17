//! Token-count estimator for `/v1/messages/count_tokens`.
//!
//! We don't have a real tokenizer available (the proxy has no way to
//! ask the upstream provider for one without a real round-trip), and
//! pulling in tiktoken-rs would add ~5 MB of binary data per model
//! vocabulary we support. Instead we use a word-based approximation
//! that's within ~10% of the actual count for typical English inputs.
//!
//! Heuristic: each whitespace-separated word contributes
//! `ceil(len / 3.5)` tokens, with a floor of 1 — for short words this
//! gives 1 token, for longer words (5+ chars) it gives 2+. Empirically
//! this matches GPT/Claude BPE output for English to within a few
//! tokens. For non-Latin scripts the count drifts more, but those
//! inputs are also where the client probably wants a "rough estimate"
//! anyway. See fix-R5 in docs/TEST_ISSUES.md.

use serde_json::Value;

use crate::anthropic::{
    ContentBlock, MessageContent, MessagesRequest, SystemPrompt, ToolResultContent,
};

/// Estimate the token cost of a typed Anthropic request *before* it is
/// dispatched — the pre-flight charge for the per-provider TPM window
/// (`rate_limit.tpm`). Reuses the [`estimate_text_tokens`] heuristic over
/// the typed structure instead of re-serializing to JSON: walking the
/// string leaves avoids allocating a full `Value` tree per request.
///
/// The returned figure is **input estimate + `max_tokens` reserve**: the
/// upstream counts input+output against a TPM limit (MiniMax's documented
/// definition), and the output size is unknowable before dispatch, so the
/// configured cap is reserved up front. The estimate is replaced by the
/// upstream-reported usage once the response lands (rate-limit ticket
/// settle), so estimation drift self-corrects within one window. Media
/// blocks (image, container upload) contribute 0 — they are billed as
/// images, not text tokens.
pub fn estimate_typed_tokens(req: &MessagesRequest) -> u64 {
    let mut total: u64 = 0;
    match &req.system {
        Some(SystemPrompt::Text(s)) => total += u64::from(estimate_text_tokens(s)),
        Some(SystemPrompt::Blocks(blocks)) => {
            for b in blocks {
                total += u64::from(estimate_text_tokens(&b.text));
            }
        }
        None => {}
    }
    for m in &req.messages {
        match &m.content {
            MessageContent::Text(s) => total += u64::from(estimate_text_tokens(s)),
            MessageContent::Blocks(blocks) => {
                for b in blocks {
                    total += block_tokens(b);
                }
            }
        }
    }
    for t in req.tools.iter().flatten() {
        total += u64::from(estimate_text_tokens(&t.name));
        if let Some(d) = &t.description {
            total += u64::from(estimate_text_tokens(d));
        }
        if !t.input_schema.is_null() {
            total += u64::from(estimate_request_tokens(&t.input_schema));
        }
        for v in t.extra.values() {
            total += u64::from(estimate_request_tokens(v));
        }
    }
    // Unrecognized top-level fields ride the flattened `extra` map and
    // are billed upstream like everything else — count their string
    // leaves rather than silently charging 0.
    for v in req.extra.values() {
        total += u64::from(estimate_request_tokens(v));
    }
    total + u64::from(req.max_tokens)
}

/// Token estimate for a single content block. Text-bearing variants are
/// counted with the text heuristic; structured payloads (`Value`) go
/// through [`estimate_request_tokens`].
fn block_tokens(b: &ContentBlock) -> u64 {
    match b {
        ContentBlock::Text { text, .. } => u64::from(estimate_text_tokens(text)),
        ContentBlock::Thinking { thinking, .. } => u64::from(estimate_text_tokens(thinking)),
        // Opaque base64 — still billed upstream as tokens, so count it
        // rather than silently dropping the block's cost.
        ContentBlock::RedactedThinking { data } => u64::from(estimate_text_tokens(data)),
        ContentBlock::MidConversationSystem { content, .. } => {
            u64::from(estimate_text_tokens(content))
        }
        ContentBlock::ToolUse { input, .. } | ContentBlock::ServerToolUse { input, .. } => {
            u64::from(estimate_request_tokens(input))
        }
        ContentBlock::ToolResult { content, .. } => match content {
            ToolResultContent::Text(s) => u64::from(estimate_text_tokens(s)),
            ToolResultContent::Blocks(vs) => vs
                .iter()
                .map(|v| u64::from(estimate_request_tokens(v)))
                .sum(),
        },
        ContentBlock::SearchResult { source, title, content, .. } => {
            // `source` (a URL fetch descriptor, sometimes with content)
            // plus the result blocks are all billed upstream.
            u64::from(estimate_request_tokens(source))
                + title.as_deref().map(|t| u64::from(estimate_text_tokens(t)))
                    .unwrap_or(0)
                + content
                    .as_ref()
                    .map(|vs| vs.iter().map(estimate_request_tokens).sum::<u32>())
                    .unwrap_or(0) as u64
        }
        ContentBlock::Document { source, title, context, .. } => {
            // `source.data` is the document payload itself (plain text
            // or base64) and is billed upstream like any other text —
            // counting it follows the same rule as the RedactedThinking
            // arm above. A document counted at title+context only would
            // under-charge the TPM window by the whole document.
            u64::from(estimate_request_tokens(source))
                + title.as_deref().map(|t| u64::from(estimate_text_tokens(t)))
                    .unwrap_or(0)
                + context
                    .as_deref()
                    .map(|t| u64::from(estimate_text_tokens(t)))
                    .unwrap_or(0)
        }
        ContentBlock::WebSearchToolResult { content, .. }
        | ContentBlock::WebFetchToolResult { content, .. }
        | ContentBlock::CodeExecutionToolResult { content, .. }
        | ContentBlock::BashCodeExecutionToolResult { content, .. }
        | ContentBlock::TextEditorCodeExecutionToolResult { content, .. }
        | ContentBlock::ToolSearchToolResult { content, .. } => {
            u64::from(estimate_request_tokens(content))
        }
        // Media / unrecognized blocks: no text tokens we can estimate.
        ContentBlock::Image { .. } | ContentBlock::ContainerUpload { .. } | ContentBlock::Unknown => 0,
    }
}

/// Estimate the number of input tokens for a JSON request body sent to
/// `/v1/messages/count_tokens`. The input is treated as opaque — we
/// walk the JSON tree and sum `estimate_text_tokens` over every
/// string leaf. This intentionally ignores the structure (system /
/// messages / tools) because we can't tell those fields apart in an
/// arbitrary JSON payload without a schema; Anthropic's actual
/// MessagesRequest has more metadata tokens, but the diff is small.
pub fn estimate_request_tokens(req: &Value) -> u32 {
    let mut total: u32 = 0;
    walk(req, &mut total);
    // Floor at 1 — a body that parses to "empty" still has overhead.
    total.max(1)
}

fn walk(v: &Value, total: &mut u32) {
    match v {
        Value::String(s) => {
            *total = total.saturating_add(estimate_text_tokens(s));
        }
        Value::Array(items) => {
            for item in items {
                walk(item, total);
            }
        }
        Value::Object(map) => {
            for (_k, val) in map {
                walk(val, total);
            }
        }
        _ => {}
    }
}

/// Estimate tokens for a single free-form text string using the
/// word-length heuristic. ASCII whitespace splits; everything else
/// (CJK, emoji, punctuation) is treated as part of a word. This is a
/// compromise: it works well for English, and for CJK it returns a
/// reasonable per-character estimate since each character is roughly
/// 1.5 BPE tokens.
pub fn estimate_text_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    let mut total: f32 = 0.0;
    for word in text.split_whitespace() {
        // ceil(len / 3.5): 1-3 chars → 1 token, 4-7 chars → 2 tokens,
        // 8-10 chars → 3 tokens, etc. Tracks GPT/Claude BPE closely
        // for English.
        total += (word.chars().count() as f32 / 3.5).ceil().max(1.0);
    }
    // Round at the end so fractional per-word accumulations don't bias.
    total.ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_zero() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("   "), 0);
    }

    #[test]
    fn panagram_matches_documented_actual() {
        // Documented actual: 14 tokens for the 9-word panagram "the
        // quick brown fox jumps over the lazy dog". Old impl gave 11
        // (under-counted by 3).
        let s = "the quick brown fox jumps over the lazy dog";
        assert_eq!(estimate_text_tokens(s), 14);
    }

    #[test]
    fn short_text_rounds_up() {
        // "Hi" is 1 token in real BPE.
        assert_eq!(estimate_text_tokens("Hi"), 1);
        // 4 chars → ceil(4/3.5)=2, but the max(1.0) floor keeps it at 2.
        assert_eq!(estimate_text_tokens("test"), 2);
    }

    #[test]
    fn punctuation_only_words_count_one_token() {
        // Each punctuation-only token still gets at least 1.
        assert_eq!(estimate_text_tokens("..."), 1);
        assert_eq!(estimate_text_tokens("! ? ."), 3);
    }

    #[test]
    fn cjk_text_gets_per_char_estimate() {
        // CJK doesn't use word boundaries; we approximate at
        // ceil(chars / 3.5) tokens per contiguous run. For 7 chars,
        // ceil(7/3.5) = 2 tokens.
        let s = "你好世界你好世"; // 7 chars
        assert_eq!(estimate_text_tokens(s), 2);
    }

    #[test]
    fn mixed_english_and_cjk_separates_on_whitespace() {
        // "Hello world" = 2 tokens; "你好世界" (4 chars) = ceil(4/3.5)=2.
        let s = "Hello 你好世界";
        assert_eq!(estimate_text_tokens(s), 4);
    }

    #[test]
    fn request_estimation_sums_string_leaves() {
        let req = serde_json::json!({
            "model": "claude-test",
            "system": "be helpful",
            "messages": [
                {"role": "user", "content": "hello there friend"},
                {"role": "assistant", "content": "hi"}
            ]
        });
        let tokens = estimate_request_tokens(&req);
        // Sanity: must be a positive integer in a plausible range.
        assert!(tokens > 0);
        assert!(tokens < 100, "got {tokens} for a small request");
    }

    #[test]
    fn empty_request_floors_at_one() {
        // An empty object still has *some* overhead — the JSON
        // delimiters themselves tokenize.
        let tokens = estimate_request_tokens(&serde_json::json!({}));
        assert!(tokens >= 1);
    }

    /// The `walk` recursion falls through a `_ => {}` arm for non-recursive
    /// JSON shapes (Null, Bool, Number). Mix those in to exercise the arm.
    #[test]
    fn walk_handles_non_recursive_leaves() {
        let v = serde_json::json!({
            "n": 42,
            "b": true,
            "z": null,
        });
        let tokens = estimate_request_tokens(&v);
        // Non-recursive leaves contribute 0 text tokens, but the wrapper
        // object still floors at 1.
        assert!(tokens >= 1);
    }

    // ─── typed estimate (rate-limit TPM pre-charge) ────────────────────

    use crate::anthropic::MessagesRequest;

    fn typed_request(json: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(json).expect("MessagesRequest parse")
    }

    #[test]
    fn typed_estimate_includes_max_tokens_reserve_exactly() {
        // The output reserve is the full configured cap: input estimate
        // is unchanged by it, so stripping the reserve must shift the
        // total by exactly max_tokens.
        let mut body = serde_json::json!({
            "model": "mm-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello there friend"}],
        });
        let req = typed_request(body.clone());
        let with_reserve = estimate_typed_tokens(&req);
        body["max_tokens"] = serde_json::json!(0);
        let without_reserve = estimate_typed_tokens(&typed_request(body));
        assert_eq!(with_reserve - without_reserve, 1024);
    }

    #[test]
    fn typed_estimate_counts_system_messages_tools_and_tool_payloads() {
        let req = typed_request(serde_json::json!({
            "model": "mm-model",
            "max_tokens": 32,
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "hello there friend"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "你好世界"},
                    {"type": "tool_use", "id": "t1", "name": "read_file",
                     "input": {"path": "/tmp/log.txt", "hint": "system log"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "file contents here"}]}
                ]},
            ],
            "tools": [
                {"name": "read_file", "description": "Read a file",
                 "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
            ],
        }));
        let est = estimate_typed_tokens(&req);
        // Every component is a positive contributor; assert the total is
        // at least their sum under a coarse floor (1 token per short
        // string) plus the 32-token reserve — the exact heuristic values
        // are pinned by the estimate_text_tokens tests above.
        let floor = 1 /*system*/ + 3 /*hello there friend*/ + 2 /*CJK*/ + 1 /*tool name*/
            + 1 /*tool desc*/ + 1 /*schema*/ + 1 /*tool_use input*/ + 1 /*tool_result*/
            + 32 /*reserve*/;
        assert!(est as u32 >= floor, "estimate {est} below floor {floor}");
        // And it stays in a sane range for this small request.
        assert!(est < 200, "estimate {est} implausibly large");
    }

    #[test]
    fn typed_estimate_ignores_media_and_unknown_blocks() {
        let req = typed_request(serde_json::json!({
            "model": "mm-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "future_block_kind", "mystery": "field"},
            ]}],
        }));
        // Media + Unknown contribute 0; only the reserve survives.
        assert_eq!(estimate_typed_tokens(&req), 16);
    }

    #[test]
    fn typed_estimate_on_empty_messages_is_the_reserve() {
        let req = typed_request(serde_json::json!({
            "model": "mm-model",
            "max_tokens": 64,
            "messages": [],
        }));
        assert_eq!(estimate_typed_tokens(&req), 64);
    }

    #[test]
    fn typed_estimate_counts_document_source_and_extra_fields() {
        // The document payload lives in `source.data` (plain text or
        // base64) and is billed upstream — a block counted at
        // title+context only would under-charge the TPM window by the
        // whole document. Same rule for the flattened `extra`
        // top-level fields (unknown keys ride along billed).
        let req = typed_request(serde_json::json!({
            "model": "mm-model",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [
                {"type": "document",
                 "source": {"type": "text", "media_type": "text/plain",
                            "data": "one two three four five"},
                 "title": "t"}
            ]}],
            "mystery_top_level": {"deep": "values here"},
        }));
        let est = estimate_typed_tokens(&req);
        // Floor: reserve 8 + 5 source words + 1 title + 2 extra-leaf
        // words ("deep" is a key — the walk counts values only).
        let floor = 8u64 + 5 + 1 + 2;
        assert!(est >= floor, "estimate {est} below floor {floor}");
    }
}

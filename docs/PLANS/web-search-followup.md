# Plan: Web-search forwarding follow-ups (post PR1–PR3)

## Goal

PR1 (`6afe7ac`) and PR2+PR3 (`720c406`) implemented the four core changes
called out in `docs/PLANS/web-search-forward.md`: `Tool` struct fix,
`web_search_options` injection on Chat Completions, `web_search_preview`
hosted tool on Responses API, and `tool_choice` remapping. This document
captures everything that was intentionally **not** landed and now belongs
in follow-up PRs.

Branch: `fix/hosted-tool-input-schema` (PR1+PR2+PR3 base).

## TL;DR

| Priority | Item | Estimated scope |
|---|---|---|
| **P0** | A. `is_web_search_tool` truth-table tests | ~30 LoC, 1 file |
| **P0** | B. `convert_tool_choice` four-case test matrix (both paths) | ~80 LoC, 2 files |
| **P0** | C. Request-direction wiremock e2e tests (Chat + Responses) | ~150 LoC, 2 files |
| **P1** | D. Forward `usage.server_tool_use` (response direction) | ~30 LoC, 3 files |
| **P1** | E. Multi-turn history replay — `ContentBlock::ServerToolUse` → `ResponseInputItem::FunctionCall` | ~30 LoC, 2 files |
| **P1** | F. Preserve `web_search_tool_result` content on Responses path | ~30 LoC, 1 file |
| **P2** | G. Per-provider `web_search` opt-out config flag | ~120 LoC, 3 files |
| **P2** | H. Verify Copilot `/v1/responses` accepts `web_search_preview` | manual, 1 commit |
| **doc** | I. Add §5.2 "user-visible behavior" paragraph | docs only |
| **doc** | J. Defensive comment on `is_web_search_tool`'s `type_str == "web_search"` branch | 1 line |

A+B+C land together as PR4. D+E+F land together as PR5. G+H are
independent follow-ups (G requires operator input on config schema; H
requires a live Copilot account).

## Why these are out of PR1–PR3

- **A, B, C** — review observations (`docs/PLANS/web-search-forward.md`
  §10 Q4) that landed before tests. PR2 was shipped with the production
  logic but no test coverage for the predicate or the `tool_choice`
  remap. This is the highest-risk gap.
- **D, E, F** — explicitly demoted to P1 in §9 and §10 of the plan doc:
  OpenAI-family upstream silently drops `server_tool_use` /
  `web_search_tool_result` blocks and `url_citation` annotations today.
  Claude Code's "Did N searches" UI line and Citations panel are
  therefore blank on those upstreams. Anthropic-passthrough provider
  works fully.
- **G, H** — operate-or-validate, not code: a config flag requires
  product/operator input on schema; Copilot endpoint validation requires
  a live test account and may need to gate PR2 (already merged, so
  observation is the next step).
- **I, J** — documentation/comment polish, lowest priority.

## Critical files

For PR4 (A + B + C):
- `src/conversion/util.rs:10-23` — `is_web_search_tool` predicate.
- `src/conversion/request.rs:107-117` — Chat `tool_choice` remap branch.
- `src/conversion/responses.rs:165-176` — Responses `tool_choice` remap branch.
- `src/providers/openai_compat.rs` and `src/providers/openai_responses.rs` —
  end-to-end wiremock test chassis (analogous to existing
  `src/providers/anthropic.rs:325-470` passthrough test).

For PR5 (D + E + F):
- `src/conversion/responses_stream.rs:362` — `server_tool_use: None` hardcode.
- `src/conversion/response.rs:74` — `server_tool_use: None` hardcode.
- `src/conversion/stream.rs:131` — `server_tool_use: None` hardcode.
- `src/conversion/responses.rs:284-285, 456-459` —
  `ContentBlock::ServerToolUse` / `WebSearchToolResult` arms in
  `convert_message` for multi-turn history replay.
- `src/conversion/responses.rs:381` — `output_item_to_block` arm for
  Responses stream → Anthropic SSE.

For PR6 (G + H + I + J):
- `src/config.rs` — new `ProviderConfig` field (or per-model override).
- `src/conversion/request.rs` + `src/conversion/responses.rs` — gate the
  `web_search_options` / `web_search_preview` injection on the new flag.

## PR4 — Tests for the predicate, tool_choice, and request e2e

### A. `is_web_search_tool` truth table

Append to `src/conversion/util.rs` (new `#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::Tool;
    use serde_json::json;

    fn tool(name: &str, kind: Option<&str>) -> Tool {
        let mut t: Tool = serde_json::from_value(json!({
            "name": name,
        }))
        .unwrap();
        t.kind = kind.map(|s| s.to_string());
        t
    }

    #[test]
    fn detects_web_search_20250305_kind() {
        assert!(is_web_search_tool(&tool("web_search", Some("web_search_20250305"))));
    }

    #[test]
    fn detects_web_search_preview_kind() {
        assert!(is_web_search_tool(&tool("web_search", Some("web_search_preview"))));
    }

    #[test]
    fn detects_bare_web_search_kind() {
        // Defensive: matches the Azure Responses-API stable alias for
        // web_search_preview. Doesn't appear in Anthropic inbound traffic
        // but is harmless.
        assert!(is_web_search_tool(&tool("web_search", Some("web_search"))));
    }

    #[test]
    fn rejects_function_tool_coincidentally_named_web_search() {
        // The exact bug the original first-draft predicate had.
        assert!(!is_web_search_tool(&tool("web_search", Some("function"))));
    }

    #[test]
    fn rejects_tool_with_no_kind() {
        assert!(!is_web_search_tool(&tool("web_search", None)));
    }

    #[test]
    fn rejects_unrelated_function_tool() {
        assert!(!is_web_search_tool(&tool("get_weather", Some("function"))));
        assert!(!is_web_search_tool(&tool("get_weather", None)));
    }
}
```

### B. `convert_tool_choice` four-case matrix

Append to `src/conversion/request.rs:556-870` test area (chat path) and
`src/conversion/responses.rs:851-867` test area (Responses path). Both
must cover independently because Copilot routes through both.

**Chat path** (`request.rs`):
```rust
#[test]
fn tool_choice_web_search_is_remapped_to_auto_when_tools_have_web_search() {
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"}
        ],
        "tool_choice": {"type": "tool", "name": "web_search"}
    })).unwrap();
    let out = anthropic_to_openai_request(&req, &Default::default());
    assert_eq!(out.tool_choice, Some(json!("auto")));
}

#[test]
fn tool_choice_other_tool_is_preserved_when_web_search_is_present() {
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search"}
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"}
    })).unwrap();
    let out = anthropic_to_openai_request(&req, &Default::default());
    assert_eq!(
        out.tool_choice,
        Some(json!({"type": "function", "function": {"name": "get_weather"}}))
    );
}

#[test]
fn tool_choice_auto_remains_auto_with_web_search() {
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": "auto"
    })).unwrap();
    let out = anthropic_to_openai_request(&req, &Default::default());
    assert_eq!(out.tool_choice, Some(json!("auto")));
}

#[test]
fn tool_choice_web_search_passes_through_when_no_web_search_tool() {
    // Defensive: don't mangle an unrelated forced choice that happens
    // to point at a name called "web_search" if no hosted tool is in tools[].
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "web_search"}
    })).unwrap();
    let out = anthropic_to_openai_request(&req, &Default::default());
    // tool_choice remains as a function-form pointing at "web_search"
    // (the upstream will 400, but we don't mangle unrelated traffic).
    assert_eq!(
        out.tool_choice,
        Some(json!({"type": "function", "function": {"name": "web_search"}}))
    );
}

#[test]
fn tool_choice_collapses_to_none_when_only_hosted_tools() {
    // When the only tools are hosted (web_search), the tools field
    // collapses to None. tool_choice must also collapse — strict
    // OpenAI-compat upstreams reject "tool_choice only supported when
    // tools are enabled".
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 32,
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        "tool_choice": {"type": "tool", "name": "web_search"}
    })).unwrap();
    let out = anthropic_to_openai_request(&req, &Default::default());
    assert!(out.tools.is_none(), "tools must be None when only hosted tools");
    assert!(out.tool_choice.is_none(), "tool_choice must collapse when tools is None");
}
```

**Responses path** (`responses.rs`): same four cases against
`anthropic_to_responses_request`, asserting `out.tool_choice`. The
expectation for case 4 (no hosted tool) is that the tool_choice passes
through unchanged (matching current behavior — there's no remap trigger).

### C. Request-direction wiremock e2e tests

Mirror `src/providers/anthropic.rs:473-528` (PR1 sibling test) for Chat
Completions and Responses providers.

**Chat Completions** (`src/providers/openai_compat.rs`, new test):
```rust
#[tokio::test]
async fn chat_provider_emits_web_search_options_when_request_has_web_search() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o",
            "web_search_options": {},
            "tools": [],  // web_search stripped
            "messages": [{"role": "user", "content": "search this"}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl_1", "object": "chat.completion",
            "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAiCompatProvider::new(/* … */, server.uri(), /* … */).unwrap();
    let req: MessagesRequest = serde_json::from_value(json!({
        "model": "gpt-4o",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "search this"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}]
    })).unwrap();
    let _ = provider.complete(&req, &empty_rewrite()).await.unwrap();
}
```

**Responses** (`src/providers/openai_responses.rs`): analogous test asserting
the upstream body contains a `web_search_preview` tool entry and that the
function tool is preserved.

### D. Edge cases worth one assertion each (see §10 Q4 "additional scenarios")

- Only-hosted-tool: tools[]=[{type:web_search_20250305}] →
  Chat extra.web_search_options={}, Responses tools[]=[{type:web_search_preview}].
- Only-function-tools (regression guard): tools[]=[{name:"f",input_schema:...}] →
  no extra/web_search_options; no web_search_preview injected.

### E. Roundtrip regression for `extra` flatten

Append to `src/anthropic.rs` schema_tests:
```rust
#[test]
fn tool_extra_preserves_max_uses_allowed_domains_user_location() {
    let raw = json!({
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 8,
        "allowed_domains": ["anthropic.com", "docs.anthropic.com"],
        "user_location": {"type": "approximate", "city": "SF", "country": "US"}
    });
    let v = roundtrip::<Tool>("tool_extra", raw.clone());
    // Structural Value equality via assert_subset on the roundtripped value
    // and the original raw — preserves `max_uses`, `allowed_domains`, etc.
    assert_subset(&v, &raw);
}
```

## PR5 — Response-direction translation (P1)

### F. Forward `usage.server_tool_use`

Currently hardcoded `None` at **four** sites in production code (plus
one in `src/anthropic.rs:736` which is a test fixture and out of scope):

- `src/conversion/responses_stream.rs:370` — streaming Responses.
- `src/conversion/responses.rs:522` — non-streaming Responses.
- `src/conversion/response.rs:74` — non-streaming Chat.
- `src/conversion/stream.rs:131` — streaming Chat.

OpenAI's Responses API actually surfaces `web_search_requests` in
`usage.server_tool_use.web_search_requests` — but `ResponsesUsage`
(`src/responses.rs:158-169`) does not model this field and has no
`#[serde(flatten)]`, so the count is dropped at deserialization time.
**Implementation**: extend `ResponsesUsage` with
`server_tool_use: Option<Value>` (matching the Anthropic `Usage` struct
at `src/anthropic.rs:400`), then plumb it through to the four consumer
sites above. Add a `TODO` comment at each site pointing at this plan doc.

### G. Multi-turn history replay

**Problem**: Two drop points drop `ServerToolUse` / `WebSearchToolResult`
in the assistant→input[] converter:

- `src/conversion/responses.rs:284-285` (Responses path) — drops
  `ContentBlock::ServerToolUse` and `ContentBlock::WebSearchToolResult`
  from input[] construction.
- `src/conversion/request.rs:282-283, 345-346` (Chat path) — drops
  the same two block types during `convert_message` → Chat message
  reconstruction.

Claude Code's main turn does not include these blocks (the search runs
as a side query), so this is a non-blocking degradation. But for
clients that include them in the conversation, multi-turn loses
context.

**Implementation**:
```rust
ContentBlock::ServerToolUse { id, name, input, .. } => {
    tool_calls.push((
        id.clone(),
        name.clone(),
        serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
    ));
}
ContentBlock::WebSearchToolResult { tool_use_id, content, .. } => {
    // Extract a textual representation. Anthropic web_search_result
    // entries carry title/url/encrypted_content; OpenAI function_call_output
    // needs a flat string. Render via the shared helper.
    let output = web_search_results_to_text(content);
    // Append a function_call_output item after the call entry.
    // Requires extending ResponseInputItem / the convert_message signature
    // to return ordered pairs; OR threading a parallel Vec<ResponseInputItem>.
}
```

This is more invasive than F; depending on how the existing
`convert_message` returns its result, you may need to thread an
additional `Vec<ResponseInputItem>` and zip the call + output pairs.
Smaller-surface alternative: emit the call as a `FunctionCall` and the
result as an extra `Message { role: "user", content: Text(...) }` after.
Acceptable as long as the upstream can re-process.

### H. Preserve `web_search_tool_result` on Responses stream path

Currently `output_item_to_block` at
`src/conversion/responses_stream.rs:381-422` does not have a
`WebSearchCall` arm — we early-return on `WebSearchCall` at the
`output_item.added` arm (PR3 line 148). Search results ride along on the
subsequent `output_text` block's `annotations: Option<Vec<Value>>`
(which is currently hardcoded `None` at `src/conversion/responses_stream.rs:399`
and `src/conversion/responses.rs:442` in production code). To forward
the annotations, plumb them through:
- The `ResponseOutputTextDone` event (`src/responses.rs:219`) carries
  the completed text; annotations need a separate delivery channel.
- If upstream emits annotations separately, add a
  `ResponseOutputTextAnnotations { item_id, output_index, content_index, annotations }`
  SSE event variant.

This is genuinely research-required — wire shape is uncertain.

## PR6 — Provider-level opt-out + Copilot validation + docs

### I. Per-provider `web_search` opt-out flag

**Schema** (`src/config.rs`): extend `ProviderConfig` enum with
```yaml
providers:
  - type: openai_compat
    name: deepseek
    web_search: strip  # default 'forward' — current behavior
```

Three values: `forward` (current default — emit web_search_options),
`strip` (drop web_search tools silently, do not inject web_search_options),
`error` (return 400 with explanatory message).

**Implementation**:
- `Provider` trait gains `fn web_search_mode(&self) -> WebSearchMode` with
  default `Forward`.
- `src/conversion/request.rs` and `src/conversion/responses.rs` read the
  mode from the provider-config side-channel (passed via a new
  `ProviderContext` argument or detected by checking the provider name).

### J. Copilot Responses endpoint validation

**Manual checklist** before declaring PR2 complete:
1. Spin up a real Copilot account.
2. Send `POST /v1/responses` to Copilot with a body containing
   `tools: [{type: "web_search_preview"}]`.
3. If 200: note the model + result; no further action.
4. If 400: capture the error body; if it mentions a model allowlist
   (`gpt-4o`, `gpt-4o-mini` only), restrict PR2 by gating the
   `ResponsesTool::WebSearch` injection on the resolved model name
   matching that allowlist OR on the per-provider `web_search` flag
   defaulting to `strip` for non-Allow-listed models.

This is **manual** because Copilot's model allowlist is undocumented and
likely changes over time.

### K. Add §5.2 user-visible behavior paragraph (doc-only)

Append to `docs/PLANS/web-search-forward.md` §5.2:

> **User-visible behavior on OpenAI-family upstreams**:
>
> - **Chat Completions** (`openai_compat.rs`): the upstream is given
>   `web_search_options: {}` in the request body. It performs the search
>   server-side. The response is a normal chat completion with the
>   search results baked into the assistant's text. Claude Code's "Did N
>   searches" UI line **will not appear**, and the Citations panel will
>   be empty (the upstream emits `annotations[]` with `url_citation`
>   entries, but the proxy drops them — see `conversion/response.rs:41,
>   conversion/stream.rs:167`).
>
> - **Responses API** (`openai_responses.rs`): the upstream emits a
>   `web_search_call` output item (current behavior: silently dropped
>   by the proxy — `conversion/responses.rs:456-459`) followed by a
>   `message` item whose `output_text` carries `url_citation`
>   annotations (current behavior: dropped — `responses_stream.rs:468,
>   554, 755, 803, 813` etc.).
>
> - **Anthropic-passthrough** (`anthropic.rs`): the request body is
>   forwarded verbatim via `serde_json::to_value(req)`. The upstream
>   produces full `server_tool_use` / `web_search_tool_result` blocks;
>   these flow through unchanged in the response. `usage.server_tool_use`
>   also roundtrips.
>
> **Implication for operators**: clients running Claude Code against
> Anthropic-passthrough upstreams get the full experience (citations,
> search count UI). Clients running against OpenAI-family upstreams get
> the search answer but lose the visual chrome.

### L. Defensive comment on `is_web_search_tool`'s bare `web_search` branch

Append to `src/conversion/util.rs:21`:

```rust
// `type_str == "web_search"` is the Azure OpenAI Responses-API stable
// alias for `web_search_preview` (Microsoft Foundry docs). Anthropic's
// hosted-tool type is always the versioned `web_search_20250305`, so
// this branch is dead for Anthropic inbound traffic — but harmless and
// defensive against future OpenAI naming changes.
```

## Open questions

- **Should `web_search` be opt-out per provider, per model, or both?**
  Affects config schema.
- **Should we expose `web_search_requests` count via `Usage` even when
  upstream doesn't report it?** Affects PR5 scope.
- **Are there other `web_search_*` type strings (`web_search_20250929`,
  `web_search_20260318`) that need explicit matching?** The SDK's
  `web_search_tool_20260318_param.py` (seen in
  `/home/zhouqt/.local/lib/python3.12/site-packages/anthropic/types/`)
  suggests yes. Add future-proofing or document the supported list.

## Sources

- `docs/PLANS/web-search-forward.md` §10 (review-driven backlog)
- `docs/PLANS/web-search-forward.md` §9 (first review)
- `/home/zhouqt/src/litellm/litellm/integrations/websearch_interception/handler.py:373-388`
  (LiteLLM's `convert_tool_choice` handling for the same scenario)
- `src/providers/anthropic.rs:325-470` (passthrough test pattern to mirror)
- `src/providers/anthropic.rs:473-528` (PR1 hosted-tool test pattern to mirror)
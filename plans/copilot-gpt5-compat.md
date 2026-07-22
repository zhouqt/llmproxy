# Fix: Copilot GPT-5.x request/stream compatibility

## Context

GPT-5.x models on GitHub Copilot (`gpt-5`, `gpt-5-mini`, `gpt-5.5`,
…) speak the **Responses API** (`/responses`), not `/chat/completions`.
The Responses API uses different request/streaming shapes from Chat
Completions, and Copilot enforces a stricter parameter set on it. Five
upstream errors surfaced while wiring `gpt-5.5` into a fallback chain
(`work-high → gpt-5.5`):

| # | Upstream error | Cause |
|---|----------------|-------|
| 1 | `400 Unsupported parameter: 'max_tokens' ... Use 'max_completion_tokens'` | GPT-5.x rejects the legacy `max_tokens`; Chat Completions wants `max_completion_tokens` (or the Responses field `max_output_tokens`). |
| 2 | `400 Invalid 'user': string too long ... maximum length 64, but got length 150` | Copilot enforces a 64-char `user` field; Anthropic clients send whatever `metadata.user_id` they want. |
| 3 | `400 model "gpt-5.5" is not accessible via the /chat/completions endpoint` | Dispatch keyed off the original `req.model` instead of the rewritten upstream name — a `work-high → gpt-5.5` mapping therefore never reached `/responses`. |
| 4 | `400 This model is compatible only with 24h extended prompt caching` | GPT-5.x rejects `prompt_cache_retention: "in_memory"`; Anthropic's short `cache_control` mapped to `in_memory`. |
| 5 | Duplicate reply text rendered twice in Claude Code | `content_block_start` was seeded from the item's text snapshot AND deltas replayed the same text. Also `finalize` re-emitted `content_block_stop` for blocks already closed. |

A sixth, latent bug was fixed alongside: `response.created` events
have `usage: null` (usage only arrives on `response.completed`); our
`ResponsesResponse.usage` was a required struct, so the entire SSE
line was being dropped as "malformed".

## Changes

### 1. Route by rewritten model name — `src/providers/copilot.rs`

`complete()` / `stream()` now merge the provider-level and runtime
model_rewrite tables first, then classify on the resulting upstream
model name. Anything that ultimately resolves to `gpt-5*` lands on
`/responses`.

### 2. `max_completion_tokens` — `src/openai.rs`, `src/conversion/request.rs`

`ChatRequest` gains `max_completion_tokens`; the converter emits it
alongside `max_tokens` so older models still work and GPT-5.x picks
the recognized field.

### 3. `user` truncation — `src/conversion/{request,responses}.rs`

New `truncate_user()` helper clips `metadata.user_id` to 64 chars
before it lands on the wire.

### 4. Cache retention escalation — `src/conversion/responses.rs`

When the resolved upstream model starts with `gpt-5`, an
`in_memory` retention is escalated to `24h` so the request honors
the client's caching intent without 400-ing. The reasoning (Anthropic
ephemeral/ephemeral_5m → OpenAI in_memory) is unchanged for non-gpt-5
models.

### 5. Stream: empty start blocks + single stop — `src/conversion/responses_stream.rs`

`output_item_to_block` now always opens empty text / tool-use blocks;
text and tool arguments arrive exclusively via delta events. A
`closed_blocks` set prevents `finalize` from re-emitting
`content_block_stop` for blocks already closed by an explicit
`*.done` event. This is the Anthropic streaming contract and matches
the copilot-api-py reference.

### 6. Accept null usage on `response.created` — `src/responses.rs`

`ResponsesResponse.usage` becomes `Option<ResponsesUsage>`. Callers
in `responses.rs` and `responses_stream.rs` updated to handle it
(non-streaming path defaults to zeroed usage).

## Tests

- `complete_routes_gpt5_to_responses_endpoint` (existing, now passes
  against the corrected routing)
- `non_gpt5_request_never_touches_responses_endpoint` — rewritten;
  the prior version asserted the wrong behavior (rewrite that
  resolves to gpt-5 must NOT go to /responses) and would have caught
  the bug.
- `rewritten_to_gpt5_routes_to_responses_endpoint` — covers
  `work-high → gpt-5.5` hitting `/responses` for the complete path.
- `streaming_rewritten_to_gpt5_routes_to_responses_endpoint` — same
  for the streaming path.
- `gpt5_escalates_in_memory_retention_to_24h` and three siblings in
  `conversion::responses`.
- `output_item_added_with_text_snapshot_does_not_duplicate_reply` —
  regression for the duplicate-text bug.
- `output_text_done_then_finalize_emits_single_content_block_stop` —
  regression for the double-stop bug.
- `stream_event_response_created_with_null_usage_decodes` — regression
  for the dropped-SSE-line bug.

360 lib tests pass; `cargo check --tests` clean.

## Manual verification

```bash
# gpt-5.5 via Copilot, with the new (work-high → gpt-5.5) mapping
curl -s -X POST http://localhost:9999/v1/messages \
  -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
  -d '{"model":"work-high","max_tokens":64,"stream":true,
       "messages":[{"role":"user","content":"hi"}],
       "metadata":{"user_id":"u-test"}}'
```

Expected:
- routed to `/responses` (not `/chat/completions`)
- no `max_tokens` rejection
- no `user` length rejection
- no `in_memory` retention rejection
- exactly one reply (no duplicate text) in the SSE stream
- the `response.created` SSE line is no longer logged as malformed
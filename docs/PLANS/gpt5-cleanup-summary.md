# GPT-5.x Clean Branch — Round 1-5 Summary

This document summarizes the work on the `fix/copilot-gpt5-compat` branch to
bring production-grade GPT-5.x support to llmproxy's Copilot provider. It is
intended for a maintainer landing on the repository cold: it explains what was
fixed, in what order, what was left out, and why.

## Timeline

**Round 1 — Bug porting from v1 archive (6 commits).** The initial clean branch
(`bc05a49`) was a subset of the archived `fix/copilot-gpt5-compat-v1` branch;
it lacked 4 P0 fixes verified against real Copilot wire captures. Round 1
back-ported: (P0-1) `stop_reason: tool_use` for function-call streams, without
which Claude Code discards tool calls; (P0-2) inline finalize on terminal SSE
events (`response.completed` / `.failed` / `.incomplete`) so the stream does
not hang when Copilot omits the `[DONE]` sentinel; (P0-3) `item_id`-based
routing for function-call argument deltas to handle Copilot's
`output_index`-mismatch behavior; (P0-4) safe-multibyte `truncate_user` to
fix panics on CJK user IDs; (P0-5) `Option<ResponsesUsage>` to tolerate
`usage: null` on `response.created`. Also added T1-T6 regression tests.

**Round 2 — R1 regression fix + predicate deduplication (4 commits).** An
opus review found that R1's P0-2 implementation omitted `finished=true` after
inline finalize, causing the adapter to poll reqwest until 600s timeout and
emit a spurious error event. Round 2 fixed this (P0-A, T7). It also unified
the scattered `model.starts_with("gpt-5")` checks into `util::gpt5_family()`
(P1-2), switched to `max_tokens` XOR `max_completion_tokens` per model family
(P1-3, T11+T12), guarded done events against phantom block allocation (P1-5,
T8+T9), and added handling for upstream `error` SSE events (P1-B, T10).

**Round 3 — Error SSE wire-shape defects + summarizer hoist (5 commits).**
The Error SSE handler from Round 2 had three defects: (P1-E) the envelope used
`"type_"` instead of `"type"`, rejected by the Anthropic SDK's pydantic
validation; (P1-F) a duplicate match arm produced an `unreachable_patterns`
warning; (P1-G) `finalize()` did not check the `finalized` flag and would emit
`message_delta` + `message_stop` after an error. All three were fixed. The
hoisted `summarize_for_log` landed in `util.rs` (P1-4, T18+T19), covering a
third call site in `openai_compat.rs` that was still logging raw HTML.
Added `/responses` 401-refresh-retry test coverage (P1-C, T17) and pinned
multipart-item behavior (P1-1, T20).

**Round 4 — N1 routing regression + delta-side phantom blocks + Error
resilience (5 commits).** Two significant issues emerged. (N1) The
`gpt5_family` utility predicate was overloaded: it routed o-series models to
`/responses`, but Copilot serves only gpt-5.x on that endpoint. o-series was
restored to `/chat/completions` while keeping the request-shaping benefits
(24h escalation, `max_completion_tokens` XOR). (P2-1) Delta handlers also
called `allocate_block` and could create phantom blocks, mirroring the already-
fixed done-side bug — guards were added for both `output_text.delta` and
`function_call_arguments.delta`. (P2-2) The `Error` variant's `message` field
was made `Option<String>` with nested `extra.error.message` fallback, so
flatten/alternate error shapes don't get silently dropped. (P2-3) The
`raw_error_event` bypass was removed, `response.failed` now surfaces upstream
error details as `StreamEvent::Error`, and the cooldown log was switched from
`truncate_for_log` to `summarize_for_log` (N2).

**Round 5 — Cleanup + summary (this document).** Removes dead code and orphan
tests left behind after the `truncate_for_log` → `summarize_for_log` swap
(P2-1), adds delta-side phantom block guard tests that were specified but
never written (P2-2, T21+T22), and produces this summary document plus README
update (P3-1+P3-2).

## Commits by Round

The branch contains 33 commits on top of the `main` baseline. Foundational
fixes (pre-Round 1) established the initial GPT-5 /responses path; rounds 1-5
added tests, fixed regressions, and hardened edge cases.

| Round | Key commits | What they fix |
|-------|-------------|---------------|
| Foundational (pre-R1) | `edb45f1`, `22e89fe`, `3c47f65`, `7cec01d`, `db0e67b`, `bc05a49` | Initial GPT-5.x /responses routing; `max_completion_tokens`; `in_memory`→`24h` escalation; empty-start to avoid text duplication; HTTP error body summarization |
| Round 1 | `1754569` (plan), `06dda62`, `49f8d59`, `96fdb1d`, `2ebc45a`, `a276d6b` | P0-1 `tool_use` stop_reason; P0-2 inline finalize on terminal events; P0-3 `item_id`-based fc routing; P0-4+P0-5 safe `truncate_user` + nullable usage; P1-1 snapshot-only fallback delta (+ T1-T6) |
| Round 2 | `fe928b1` (plan), `37598ba`, `18f3b8d`, `46a5df6`, `6e0c49d` | P0-A `finished=true` after terminal finalize (+ T7); P1-2 `gpt5_family` predicate (+ T13); P1-3 `max_tokens` XOR (+ T11+T12); P1-5 phantom done guard + P1-B error SSE event (+ T8+T9+T10) |
| Round 3 | `6d97bf1` (plan), `7dcdcb3`, `d48a332`, `c951234`, `dddd7d6`, `29ccec6` | P1-E error envelope `type` not `type_` (+ T14); P1-F deduplicate arm; P1-G idempotent `finalize` (+ T15+T16); P1-4 summarizer hoist (+ T18+T19); P1-C `/responses` 401-retry test (+ T17); P1-D+P1-1 docs + pin (+ T20) |
| Round 4 | `18d3aba` (plan), `3e8f8b2`, `0b05bef`, `6678dcf`, `9ac01c1`, `6a70953` | N1 o-series /chat_completions routing (+ T25); P2-1 delta-side phantom block guard (+ T21+T22 written in prod code); P2-2 tolerant Error decode (+ T23); P2-3 `response.failed` error surfacing (+ T24); N2 cooldown `summarize_for_log` |
| Round 5 | `4ac743a`, `43b9160`, (this doc) | P2-1 dead code cleanup; P2-2 delta guard tests (T21+T22 actually executed); P3-1+P3-2 summary doc + README o-series note |

## Comparison with `fix/copilot-gpt5-compat-v1`

The v1 archive branch (12 GPT-5 commits above `main`) was the initial
implementation. The clean branch was rebuilt from scratch to provide cleaner
history and per-round opus reviews. The following v1 commits were ported:

| v1 commit | Clean branch equivalent | Notes |
|-----------|------------------------|-------|
| `5d450cd` | `edb45f1` | Move usage out of `message_delta.delta` |
| `f5e0abe` | `96fdb1d` | `item_id`-based fc arg routing |
| `0bc8d16` | `bc05a49` | HTTP error body summarization |
| `0289ac6` | `06dda62` | `tool_use` stop_reason for function_call |
| `4aa3d49` | `49f8d59` | Inline finalize on terminal event |
| `4ae5bba` | `3c47f65` | Route to `/responses` by rewritten model |
| `e44ddc8` | `db0e67b` | Empty-text start to avoid reply duplication |
| `f6f25ae` | `7cec01d` | `in_memory`→`24h` retention escalation |
| `f9ff27b` | `2ebc45a` (partial) | `Option<ResponsesUsage>` for null tolerance |
| `bffbca7` | `22e89fe` | `truncate_user` + `max_completion_tokens` |

Two v1 commits were intentionally **not** ported:

- **`8c06e5f`** (`debug(responses): trace copilot request + SSE event lifecycle`):
  A debug-only tracing hook. The clean branch kept it out because it adds
  per-event overhead and has no production effect. The debug tracing idea is
  tracked separately (see "Open follow-up items" below).
- **`9b1e18d`** (`docs(plans): GPT-5.x compat fix notes for review`): Superseded
  by the per-round plan docs (`docs/PLANS/round{1,2,3,4,5}-gpt5-cleanup.md`)
  which include opus review findings and verification standards.

## Open Follow-up Items

These were identified during review but deferred out of the branch merge:

- **N7 `BlockRegistry`**: The translator's `blocks` / `block_map` /
  `closed_blocks` / `deltas_seen` / `fc_item_index` fields form an implicit
  protocol. Extracting them into a single struct would clarify lifecycle
  and reduce the risk of inconsistent state. Deferred because refactoring
  should happen after tests are stable and bugs are known.
- **`8c06e5f` debug tracing**: The v1 branch included per-event debug logs
  (enabled via `RUST_LOG=debug`). These are cheap insurance for diagnosing
  silent-drop scenarios (e.g., delta-side ghost guards). Not merged because
  introducing a structured logging framework was out of scope; revisit if
  field debugging confirms the need.
- **`response.failed` non-stream status code**: Non-stream error responses
  currently hardcode HTTP 502 (`conversion/responses.rs:392`). In reality
  Copilot may return 4xx. Fix by collecting the actual upstream status and
  forwarding it. Deferred pending production trace data showing the real
  status distribution.
- **`endpoint_for_model` as enum**: Currently `&'static str` with two string
  literals (`"responses"` / `"chat_completions"`). If Copilot adds support
  for o-series on `/responses` or introduces a third endpoint, the dispatch
  is easy to miss. A `CopilotEndpoint` enum with exhaustiveness checks in
  tests would catch future mismatches at compile time.

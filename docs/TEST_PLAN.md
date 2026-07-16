# Test Plan — llmproxy

Goal: **≥90% line coverage** across `src/`, with explicit attention to the
fallback router, provider HTTP integrations, OAuth flow, and request/response
format conversions.

## Current state (baseline)

21 unit tests, all passing. Coverage gap analysis (line counts vs test counts):

| Module                        | Lines | Tests | Status |
|-------------------------------|------:|------:|--------|
| config.rs                     |  ~290 |   3   | partial |
| error.rs                      |   ~92 |   0   | trivial glue, low risk |
| proxy_client.rs               |   ~40 |   1   | basic only — needs proxy URL cases |
| state.rs                      |   ~13 |   0   | trivial struct |
| auth.rs                       |   ~60 |   0   | **missing** — needs tower test |
| cooldown.rs                   |  ~127 |   3   | good |
| router.rs                     |  ~180 |   3   | good (covers fallback & cooldown) |
| anthropic.rs                  |  ~230 |   0   | type-only, low priority |
| openai.rs                     |  ~242 |   0   | type-only, low priority |
| conversion/mod.rs             |    ~6 |   -   | - |
| conversion/request.rs         |  ~280 |   5   | good — needs more edge cases |
| conversion/response.rs        |  ~120 |   3   | good |
| conversion/stream.rs          |  ~290 |   2   | good — needs usage/reasoning edge cases |
| providers/mod.rs              |   ~80 |   -   | factory glue |
| providers/copilot.rs          |  ~320 |   0   | **missing** — needs OAuth + API mocks |
| providers/openai_compat.rs    |  ~270 |   0   | **missing** — needs upstream mock |
| providers/openrouter.rs       |  ~200 |   0   | **missing** — needs both API formats |
| oauth/mod.rs                  |   ~11 |   -   | - |
| oauth/device_flow.rs          |  ~110 |   0   | **missing** — needs HTTP mocks |
| oauth/token_store.rs          |   ~86 |   0   | **missing** — fs + permissions |
| server.rs                     |  ~170 |   0   | **missing** — needs TestServer |
| main.rs                       |  ~165 |   0   | entry point, low priority |

## Strategy

1. **Pure logic / type tests** — write immediately (no infra needed).
2. **HTTP integration tests** — add `wiremock` crate, mock upstream providers.
3. **Server end-to-end tests** — `axum::Router` + `tower::ServiceExt::oneshot`.
4. **Filesystem tests** — `tempfile` for token store.
5. **Coverage measurement** — `cargo-llvm-cov` after the test pass.

## Coverage targets per module

### config.rs (target 95%)

- env var expansion: simple, missing, dollar-only, multiple
- config validation: empty providers, empty models, primary unknown, fallback unknown
- `find_model`, `find_provider`, `model.chain()`
- `default_*` functions via #[serde(default)]

### error.rs (target 80%)

- `is_cooldownable()` for all variants and statuses
- `status_code()` for all variants
- `IntoResponse` for upstream (JSON / non-JSON) and non-upstream paths

### proxy_client.rs (target 90%)

- build without proxy
- build with http://, https://, socks5://, socks5h://
- build with invalid URL → Config error
- timeout / connect_timeout are set

### auth.rs (target 90%)

- missing api_key → pass through
- matching Bearer → 200
- non-matching Bearer → 401
- x-api-key header path
- case-sensitivity (constant_time_eq)

### cooldown.rs (target 95%) — extend existing

- TTL exactly at boundary
- many concurrent marks
- active() returns empty when none

### router.rs (target 95%) — extend existing

- stream path with success
- stream path with fallback
- all-providers-cooling-down
- chain exhausted mid-loop
- non-cooldownable error doesn't trigger fallback (400)
- max_retries_total cap
- empty model chain (config bug — graceful)

### anthropic.rs / openai.rs (target 70%, serde roundtrip)

- Anthropic request JSON deserialize (basic, with system, with tools)
- Anthropic response serialize roundtrip
- StreamEvent serialize for all variants
- ChatResponse deserialize with various shapes
- ChatChunk deserialize for content / tool_calls / usage variants

### conversion/request.rs (target 95%) — extend

- empty messages array
- assistant message with multiple tool_use blocks
- user message with multiple tool_result blocks
- tool_result with blocks content (non-string)
- tool_choice None
- Anthropic thinking → reasoning_effort mapping (low/medium/high boundaries)
- image in assistant message (skip)
- system prompt as blocks (joined)
- model_rewrite absent + model with date suffix
- top_k field handling

### conversion/response.rs (target 95%)

- response with usage + cache tokens
- finish_reason "content_filter"
- unknown finish_reason → None stop_reason
- multiple choices → first one
- tool_calls with invalid JSON args → empty object

### conversion/stream.rs (target 95%) — extend

- empty chunks (delta.role only)
- usage chunk at end (no choices)
- reasoning-only chunks
- interleaved text and tool calls
- stop_reason "length" → max_tokens
- tool call with empty arguments

### providers/openai_compat.rs (target 90%)

- complete() success path
- complete() upstream 4xx → Upstream error
- complete() upstream 5xx → Upstream error
- stream() success path
- stream() upstream 4xx → Upstream error
- OpenAiSseToAnthropic: data: lines, [DONE], empty lines, multiline

### providers/openrouter.rs (target 85%)

- openai format delegates to OpenAiCompatProvider
- anthropic format posts to /v1/messages with headers
- anthropic format upstream error → Upstream error
- anthropic format stream pass-through

### providers/copilot.rs (target 80%)

- headers() contains all required copilot headers
- base_url() for individual / business / enterprise
- chat_url() formatting
- 401 retry path (mocked)
- token refresh path (mocked)
- stream with 401 retry

### oauth/device_flow.rs (target 90%)

- request_device_code success / failure
- poll_access_token authorization_pending → None
- poll_access_token success → token
- poll_access_token expired_token → error
- poll_access_token other error → error

### oauth/token_store.rs (target 90%)

- new() creates parent dir
- save() writes 0600 on unix
- load() returns None if missing
- load() returns Some if valid JSON
- clear() removes file

### server.rs (target 85%)

- POST /v1/messages happy path (JSON)
- POST /v1/messages streaming
- POST /v1/messages unknown model → 400
- POST /v1/messages/count_tokens rough estimate
- GET /v1/models lists configured models
- /health always returns "ok"

## Verification

After all tests are written:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --summary-only
```

Target: summary total ≥ 90%.

For per-module breakdown:
```bash
cargo llvm-cov --all-features --json | jq '.data[].totals.lines.percent'
```

## Out of scope (explicitly not testing in V1)

- `main.rs` arg parsing: trivial, low ROI.
- `anthropic.rs` / `openai.rs` exhaustive serde: type-only deserialization.
- Real network calls to OpenAI/Anthropic/Copilot.
- Cross-platform Windows-specific paths in `token_store.rs`.

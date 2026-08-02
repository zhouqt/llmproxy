# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llmproxy` is a Rust HTTP proxy that lets Claude Code (or any Anthropic-Messages-API client) talk to multiple LLM providers through one endpoint. It accepts Anthropic Messages format on `/v1/messages`, translates to each provider's native format (OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages), translates the response back, and falls back across a per-model provider chain on cooldownable errors.

## Commands

```bash
cargo build                      # debug build
cargo build --release            # release binary → /tmp/llmproxy-target
cargo run -- --config config.yaml    # run (add --release for prod; optional -p <port>)
cargo test --lib --bins --tests  # full suite (unit + integration)
cargo test <name-substring>      # single test, any target, e.g. `cargo test maps_reasoning_budget_to_effort`
cargo test --test server <name>  # integration test in tests/server.rs
cargo llvm-cov --lib --bins --tests   # coverage (project target is >97% regions)
```

- Build output is pinned to `/tmp/llmproxy-target` by `.cargo/config.toml` — the repo tree stays source-only. Tests and binaries live there, not under `target/`.
- Runtime requires `--config <path>`. Config files use `${ENV_VAR}` expansion; `config.example.yaml` is the field reference. Secrets live in `config.yaml` (gitignored) / env vars, never commit them.
- A git pre-commit hook runs `scripts/scan-secrets.sh` (API-key leak audit). Install once with `scripts/install-hooks.sh`.
- Log level via `RUST_LOG` (default `llmproxy=info,tower_http=info`).

## Architecture

### Request flow

```
POST /v1/messages (Anthropic shape)
 → auth middleware (shared bearer token, optional via server.api_key)
 → AppJson<MessagesRequest> extractor (src/extractor.rs)
 → messages_handler (src/server.rs)
    → Router::find_model(model) → ModelConfig { primary, fallback_chain, cooldown, retries }
    → stream? Router::stream : Router::complete (src/router.rs)
       → for each provider in chain (skip cooling down / can't serve model)
          → provider.complete|stream(req, model_rewrite)
             → conversion to provider-native format
             → reqwest call (proxied or direct shared client)
             → conversion back to Anthropic shape
       → on cooldownable error: record attempt, cooldown provider, next in chain
    → Anthropic JSON response, or SSE stream
```

The response carries `x-llmproxy-failed-providers` (e.g. `copilot:429,deepseek:500`) when fallback happened, and `server.rs` emits an Anthropic `event: error` SSE chunk on mid-stream upstream failure so clients don't see a truncated 200.

### Layers

- **`src/anthropic.rs`, `src/openai.rs`, `src/responses.rs`** — wire types only. `anthropic.rs` = Messages request/response/ContentBlock/StreamEvent; `openai.rs` = Chat Completions request/response/chunk; `responses.rs` = Responses API request/response/SSE events. No conversion logic here.
- **`src/conversion/`** — all format translation:
  - `request.rs` — Messages → Chat Completions
  - `response.rs` — Chat Completions → Messages
  - `responses.rs` — Messages → Responses request, and Responses → Messages (non-streaming)
  - `responses_stream.rs` — Responses SSE → Anthropic SSE events
  - `stream.rs` — Chat Completions SSE → Anthropic SSE (`StreamTranslator`)
  - `cache_hint.rs` — `cache_control` → `prompt_cache_key`/`prompt_cache_retention`
  - `util.rs` — `strictify_schema`, web-search-tool detection, etc.
- **`src/providers/`** — `Provider` trait (`complete`, `stream`, `list_models`, `can_serve_model`, `spawn_background`) with four impls:
  - `anthropic.rs` — native passthrough (body verbatim, only `model`+`stream` rewritten)
  - `openai_compat.rs` — `/chat/completions` + `OpenAiSseToAnthropic` SSE adapter
  - `openai_responses.rs` — `/responses` + `ResponsesSseToAnthropic`
  - `copilot.rs` — OAuth device flow, token persistence, and **endpoint routing**: GPT-5.x → `/responses`, everything else → `/chat/completions` (unless the `/models` cache advertises otherwise)
- **`src/router.rs`** — fallback chain, `CooldownCache`, retry accounting, `is_model_unsupported` 400 detection (skips provider instead of erroring), `/admin/status` health snapshot.
- **`src/server.rs`** — axum routes: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health`, `/admin/copilot/auth`, `/admin/status`.
- **`src/oauth/`** — Copilot GitHub device flow + token store (`~/.local/share/llmproxy/github_token.json`).
- **`src/proxy_client.rs`** — exactly two shared reqwest pools: proxied (SOCKS/HTTP, `use_proxy: true`) and direct. Providers pick one at build time.
- **`src/config.rs`** — `Config` (server/providers/models), `ModelConfig.chain()` iterator, env expansion.
- **`src/main.rs`** — CLI, `build_state`, tracing init. `AppState` (`src/state.rs`) holds config, router, cooldown, and an optional `CopilotProvider` handle for the admin endpoint.

### Load-bearing invariants (reading multiple files to see)

- **Field-omission is correctness.** Providers serialize the full typed request struct (`.json(&req)`); every optional field carries `skip_serializing_if = "Option::is_none"`. Absence vs. presence on the wire matters: `prompt_cache_key`/`prompt_cache_retention`, `reasoning_effort`, `reasoning`, `max_tokens` vs `max_completion_tokens` (GPT-5 family) are all gated this way. Tests assert presence AND absence (`body_partial_json` + a `JsonFieldAbsent` matcher).
- **effort propagation**: `output_config.effort` (explicit) or `thinking.budget_tokens` (derived: ≥8000→high, ≥2000→medium, else low) → `reasoning_effort` on Chat Completions, `reasoning: {type:"enabled", effort}` on Responses. Passthrough forwards `output_config`/`thinking` verbatim. Known gaps tracked in `plans/model-effort-forwarding.md`.
- **Anthropic passthrough thinking retry**: `complete()` strips thinking blocks and retries **once** on the cross-model "must be passed back" 400 (only when history has a thinking block). `stream()` never retries — once SSE flows, retrying would double-emit to the client; it surfaces a friendly `thinking_not_supported` error instead.
- **Fallback semantics**: cooldownable = 429/401/408/404/5xx/402 (402 = quota exhaustion). 400s surface to the client unless `is_model_unsupported` matches. Retry caps: `max_retries_per_provider`, `max_retries_total`. No retry once streaming starts.
- **Copilot specifics**: 402 → cooldownable; `/models` cache drives `can_serve_model`-style routing and endpoint choice; token refresh runs as a background task.
- **`/admin/status` contract**: `ProviderHealth` is deliberately binary (`available`/`cooling_down`) and reflects in-memory cooldown state only — it never probes upstreams. Consumed by a Claude Code statusline hook; keep shape backwards-compatible.

## Config quick reference

`config.yaml` → `server` (listen, api_key), `proxy` (url + timeout), `providers` (typed: `github_copilot`/`anthropic`/`openai_compat`/`openai_responses`, each with `model_rewrite` and `use_proxy`), `models` (client-facing `name` → `primary` + `fallback_chain` + cooldown/retry knobs).

- `model_rewrite`: maps incoming Claude names → upstream names. **Non-empty table = explicit allow-list** (`can_serve_model`); the router skips a provider for names not in the table rather than forwarding an unmapped name and getting a misleading 400. Empty table = pass any name verbatim.
- Claude Code client setup: `ANTHROPIC_BASE_URL=http://127.0.0.1:<listen>` + `ANTHROPIC_AUTH_TOKEN=<server.api_key>`.

## Testing conventions

- **>97% region coverage** is the standing bar (`cargo llvm-cov --lib --bins --tests`), tracked in `docs/TEST_PLAN.md`. New code is expected to keep it; unit tests live next to the module, integration tests in `tests/` (`server.rs`, `integration_router.rs`, `auth.rs`, `verify_deepseek_thinking.rs`).
- HTTP behavior is tested with `wiremock` (`body_partial_json` for presence, a custom `JsonFieldAbsent` matcher for absence); handlers via `axum::Router` + `tower::ServiceExt::oneshot`.
- Tests isolate env: `XDG_DATA_HOME` → tempdir for token-store tests; `LLMPROXY_TEST_GITHUB_BASE_URL` (under `oauth::device_flow::ENV_LOCK`) to redirect device-flow endpoints to wiremock.
- Use the `expect_variant!` macro (`src/lib.rs`) for match-style assertions — it centralizes the panic string, which coverage treats as unreachable.
- `docs/TEST_ISSUES.md` (Chinese) records the rationale for past fixes; `plans/` holds working design notes and is gitignored.

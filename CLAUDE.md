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
- **No Rust toolchain on the host? Use the container:** `tools/rust-build.sh` runs cargo inside the *same* `rust:1-alpine` digest the Dockerfile pins, with the repository mounted **read-only** and the registry cache / target dir on the host. It exists because the build host has ~19 GB of disk and runs out of room for a debug build plus a scanner database; the deploy host has 61 GB and already has the source tree.

  ```bash
  tools/rust-build.sh fetch --locked
  tools/rust-build.sh build --release --locked
  tools/rust-build.sh test --lib --bins --tests
  tools/rust-build.sh -- <command>      # anything else
  ```

  Override `RUST_IMAGE`, `CARGO_HOME_DIR` (`/var/tmp/llmproxy-cargo`), `CARGO_TARGET_DIR` (`/var/tmp/llmproxy-target`) as needed; `HTTPS_PROXY` is passed through when set.
- Runtime requires `--config <path>`. Config files use `${ENV_VAR}` expansion; `config.example.yaml` is the field reference. Secrets live in `config.yaml` (gitignored) / env vars, never commit them.
- A git pre-commit hook runs `scripts/scan-secrets.sh` (API-key leak audit). Install once with `scripts/install-hooks.sh`.
- Log level via `RUST_LOG` (default `llmproxy=info`).

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
  - `copilot.rs` — OAuth device flow, token persistence, `/models` cache (in-memory + disk-persisted to `copilot_models.json` next to the token store), and **endpoint routing**: `/responses` default; gpt-5.x → `/chat/completions` only when the cache advertises chat (EnterWorktree whitelist); cold cache → `/responses` for everything
- **`src/router.rs`** — fallback chain, `CooldownCache`, retry accounting, `is_model_unsupported` 400 detection (skips provider instead of erroring), `/admin/status` health snapshot.
- **`src/server.rs`** — axum routes: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health`, `/admin/copilot/auth`, `/admin/status`.
- **`src/oauth/`** — Copilot GitHub device flow + token store (`~/.local/share/llmproxy/github_token.json`).
- **`src/proxy_client.rs`** — exactly two shared reqwest pools: proxied (SOCKS/HTTP, `use_proxy: true`) and direct. Providers pick one at build time.
- **`src/config.rs`** — `Config` (server/providers/models), `ModelConfig.chain()` iterator, env expansion.
- **`src/main.rs`** — CLI, `build_state`, tracing init. `AppState` (`src/state.rs`) holds config, router, cooldown, and an optional `CopilotProvider` handle for the admin endpoint.

### Load-bearing invariants (reading multiple files to see)

- **Field-omission is correctness.** Providers serialize the full typed request struct (`.json(&req)`); every optional field carries `skip_serializing_if = "Option::is_none"`. Absence vs. presence on the wire matters: `prompt_cache_key`/`prompt_cache_retention`, `reasoning_effort`, `reasoning`, `max_tokens` vs `max_completion_tokens` (GPT-5 family) are all gated this way. Tests assert presence AND absence (`body_partial_json` + a `JsonFieldAbsent` matcher).
- **YAML parsing semantics are pinned.** Config parsing goes through `serde-saphyr` (pure Rust; it replaced the archived `serde_yaml` → `unsafe-libyaml` chain, RUSTSEC-2024-0449). `Config::parse` and the `secrets:` pre-pass both call `yaml_options()` in `src/config.rs`, which restores three behaviors `serde-saphyr` does not default to: duplicate keys keep the **last** value (`config.example.yaml` promises this for `model_rewrite` entries and repeated fields in one block), YAML merge keys (`<<`) stay **ordinary keys** instead of being expanded (expansion would let one mapping silently inject fields into another), and `with_snippet = false` keeps source lines out of parse errors — a syntax error next to an `api_key:` line must never echo the secret into the startup log. `serde-saphyr` has no generic `Value`/`from_value`; the `secrets:` pre-pass parses into `serde_json::Value`.
- **effort propagation**: `output_config.effort` (explicit) or `thinking.budget_tokens` (derived: ≥8000→high, ≥2000→medium, else low) → `reasoning_effort` on Chat Completions, `reasoning: {type:"enabled", effort}` on Responses. Passthrough forwards `output_config`/`thinking` verbatim. Known gaps tracked in `plans/model-effort-forwarding.md`.
- **Anthropic passthrough thinking retry**: `complete()` strips thinking blocks and retries **once** on the cross-model "must be passed back" 400 (only when history has a thinking block). `stream()` never retries that class — once SSE flows, retrying would double-emit to the client; it surfaces a friendly `thinking_not_supported` error instead.
- **`thinking_when_absent`** (anthropic provider only, default `disabled`): when the client request has no `thinking` key (or `null`), inject `{"thinking":{"type":"disabled"}}` so third-party compat backends don't default to reasoning. This is on by default; set `omit` per provider to restore byte-identical legacy outbound requests. The third mode, `effort_low`, never sends `thinking` (some upstreams force it on and 400 on `disabled`, e.g. MiniMax M3.1-Flash) and instead injects `{"output_config":{"effort":"low"}}` — but only when `output_config` is absent too, so a client-chosen effort is never overridden. Trigger is strictly *absence* — `adaptive` (what Claude Code's main request sends) passes through untouched. If upstream 400s and the error blames the injected parameter (`thinking`, or `output_config`/`effort`), that field is dropped and retried once (stream is safe: 400 precedes SSE bytes). OpenAI-family providers need a different mechanism (`reasoning_effort:"none"`) — deferred, see `plans/feat-disable-thinking-when-absent.md` §3.7.
- **Fallback semantics**: cooldownable = 429/401/408/404/5xx/402 (402 = quota exhaustion). 400s surface to the client unless `is_model_unsupported` matches. Retry caps: `max_retries_per_provider`, `max_retries_total`. No retry once streaming starts.
- **Copilot specifics**: 402 → cooldownable; `/models` cache drives `can_serve_model`-style routing and endpoint choice; token refresh runs as a background task.
- **Client identity simulation**: non-Copilot outbound requests present the Claude Code client identity — `User-Agent: claude-cli/2.1.224 (external, claude-vscode, agent-sdk/0.3.224)` (default, from the top-level `user_agent` config; the exact string the real client reports when driven as an external agent) plus, on the `anthropic` provider, `anthropic-version: 2023-06-01` and `x-app: cli` (both verified by pointing a mock server at the real CLI). This is what lets upstream gateways classify the proxy as Claude Code rather than an unknown client (the bare `claude-code/<ver>` CLI format is not recognized; the real client sends no `anthropic-client-platform` header). Copilot deliberately keeps `GitHubCopilotChat/<version>` (it overrides the header per-request and GitHub gates by client version).
- **The token store enforces its own encryption.** `src/oauth/token_store.rs` only accepts the format its key implies: with `provider.token_store_key` set, an **envelope** is the only accepted shape — a legacy plaintext store is migrated (re-written as an envelope, so the host disk stops holding cleartext) but **not trusted**, because plaintext carried no authentication. Without a key, only plaintext is accepted. A plaintext write over an encrypted store is refused, so a vanished `${VAR}` cannot silently downgrade the disk. The migrated envelope is marked (`TokenEnvelope::migrated`) and bound into the AAD, so it cannot be flipped into a trusted one.
- **`server.expose_upstream_details`** (default `true`) decides what a *client* sees when an upstream fails: `true` passes the upstream body through and names the failing backends in `x-llmproxy-failed-providers` (today's behaviour, byte for byte); `false` replaces both with generic wording. "Both" is wider than the response body: it also covers the `event: error` frames the SSE adapters synthesise from an upstream error object (`OpenAiSseToAnthropic`'s error-envelope branch, `ResponsesStreamTranslator`'s `response.failed`/`error` frames — those frames never reach `format_stream_error()`, so the flag is threaded into the adapters themselves), and the error variants that only *carry* upstream text (`ProxyError::UpstreamText` / `Http` / `Internal`). `ProxyError::Other` is by contract client-side wording and stays readable. The operator keeps the full detail in the logs and in `/admin/usage` either way.
- **Provider names are header-safe.** A provider name ends up in the `x-llmproxy-failed-providers` header, so `Config::validate` restricts names to ASCII letters, digits, `.`, `_` and `-` — a name that is not a legal HTTP header value is a config error, not a request-time panic.
- **`/admin/status` contract**: `ProviderHealth` is deliberately binary (`available`/`cooling_down`) and reflects in-memory cooldown state only — it never probes upstreams. Consumed by a Claude Code statusline hook; keep shape backwards-compatible.
- **Supply-chain invariants** (see `plans/harden-cargo-deps.md` for the experiments behind them). Three facts that are easy to get wrong, so they are written down:
  - `cargo build --offline` is **not** a sandbox. `build.rs` still runs, still reaches the network, still reads every environment variable. The only real egress block in this repo is the container's `--network=none`.
  - `cargo build --locked` only asserts the manifest agrees with `Cargo.lock`. A **downgrade** is a perfectly consistent lock file, so `--locked` does not catch it — `tools/lock-drift.py` (CI job `lock-drift`) does.
  - The `Dockerfile` compiles in four stages on purpose: `tools` (network, installs `cargo-auditable`/`cargo-cyclonedx`), `fetch` (network, `cargo fetch --locked` only, runs no build script), `builder` (`RUN --network=none cargo auditable build --release --offline --locked`), `runtime` (alpine + the binary + the CycloneDX SBOM). `tools/sandbox-canary/` is a probe crate whose `build.rs` fails the build if the compile environment can reach the network or holds credential-shaped env vars. Keep it wired into the image build **and** the CI job that proves it fails with the network reachable and passes without it.
  - The base images are pinned by digest on `docker.io`. The build host cannot reach `docker.io` directly, so it goes through a proxy (`tools/build-image.sh`) — the same registry CI uses, not a mirror. Keep the `FROM` lines **literal**: `tools/check-base-digests.py` parses them and cannot follow a build ARG.
  - **The base image pin does not follow the tag.** A rebuilt tag is not a reason to move a digest pin — `rust:1-alpine` is rebuilt constantly, and a PR per rebuild is noise. The pin moves for exactly one reason: the pinned image has a HIGH/CRITICAL CVE with a fix that the current build of the same tag does not have. The `base-image-cve` job in `.github/workflows/refresh-pins.yml` (monthly, Trivy) decides and opens the PR; there is deliberately no `docker` ecosystem in `dependabot.yml`. To move a pin by hand: `tools/check-base-digests.py --resolve` then `--set <image>@<digest>`.
  - `docs/SUPPLY_CHAIN.md` is the reference for what is enforced vs. what only reports, and for how the pinned inputs (crates, base images, RustSec advisories, CI tool versions) are kept current. Read it before changing anything in `deny.toml`, `supply-chain/`, or the CI audit job.

## Config quick reference

`config.yaml` → `server` (listen, api_key, expose_upstream_details), `proxy` (url + timeout), `user_agent` (client identity, default `claude-cli/2.1.224 (external, claude-vscode, agent-sdk/0.3.224)`), `providers` (typed: `github_copilot`/`anthropic`/`openai_compat`/`openai_responses`, each with `model_rewrite` and `use_proxy`), `models` (client-facing `name` → `primary` + `fallback_chain` + cooldown/retry knobs).

- `model_rewrite`: maps incoming Claude names → upstream names. **Non-empty table = explicit allow-list** (`can_serve_model`); the router skips a provider for names not in the table rather than forwarding an unmapped name and getting a misleading 400. Empty table = pass any name verbatim.
- `provider_allow`: optional list of OpenRouter backend-provider slugs to restrict routing to (allowlist, e.g. `["DeepInfra", "Together"]`); wire field is `provider: {only: [...]}`, same OpenRouter-only host gate and startup warning as `provider_ignore`. Both fields share one injected `provider` object; the proxy forwards both verbatim and never intersects them locally. Available on both `openai_compat` and `anthropic` provider types, injected before any retry loop like `provider_ignore`. **Operational caveat**: OpenRouter's `only` is a hard ceiling — an unsatisfiable list returns 503, which `is_cooldownable()` treats as a provider cooldown, so a mistyped slug silently routes the request elsewhere instead of reporting a config error. A *base* slug (`deepinfra`) also matches every endpoint variant of that provider (`deepinfra/turbo`), so the list constrains which providers may serve, not which single endpoint.
- `provider_ignore`: optional list of OpenRouter backend-provider slugs to exclude (e.g. `["Azure"]`); when the provider's `api_base` resolves to `openrouter.ai` (case-insensitive, `*.openrouter.ai` subdomains included), the proxy injects `provider: {ignore: [...]}` into the outgoing request so OpenRouter skips those providers; on any other backend the field is ignored with a one-shot startup warning, since strict Anthropic-compat / OpenAI-compat backends like DeepSeek would 400 on the unknown top-level field. Available on both `openai_compat` and `anthropic` provider types. The injection must happen before any retry loop so the field survives strip-think / strip-reasoning passes.
- `thinking_when_absent`: `anthropic` provider only; `disabled` (default) | `omit` | `effort_low`. `disabled` injects `thinking:{type:"disabled"}` when the client request has no `thinking` field, suppressing third-party compat backends' default reasoning. The feature is on by default; `omit` keeps the wire byte-identical and is the per-provider opt-out. `effort_low` sends no `thinking` and injects `output_config:{effort:"low"}` only when **both** `thinking` and `output_config` are absent — for upstreams that force thinking on at default effort `max` and 400 on `disabled` (MiniMax M3.1-Flash; drops classifier latency from ~20s to ~3s). Not available on OpenAI-family providers (different vocabulary; see `plans/feat-disable-thinking-when-absent.md` §3.7).
- Claude Code client setup: `ANTHROPIC_BASE_URL=http://127.0.0.1:<listen>` + `ANTHROPIC_AUTH_TOKEN=<server.api_key>`.

## Testing conventions

- **>97% region coverage** is the standing bar (`cargo llvm-cov --lib --bins --tests`), tracked in `docs/TEST_PLAN.md`. New code is expected to keep it; unit tests live next to the module, integration tests in `tests/` (`server.rs`, `integration_router.rs`, `auth.rs`, `verify_deepseek_thinking.rs`).
- HTTP behavior is tested with `wiremock` (`body_partial_json` for presence, a custom `JsonFieldAbsent` matcher for absence); handlers via `axum::Router` + `tower::ServiceExt::oneshot`.
- Tests isolate env: `XDG_DATA_HOME` → tempdir for token-store tests; `LLMPROXY_TEST_GITHUB_BASE_URL` to redirect device-flow endpoints to wiremock. Every test that reads or writes a process-global env var must hold `test_support::env_lock()` (the RAII `EnvGuard`): it serializes the access, restores every variable it touched on drop — including during panic unwinding — and tolerates a poisoned mutex, so one failing test cannot cascade into the rest of the suite. The guard is **not reentrant**: helpers take `&mut EnvGuard` instead of locking again, and it must be bound to a name (`let mut env = env_lock();`) — `let _ = env_lock();` releases the lock immediately.
- Use the `expect_variant!` macro (`src/lib.rs`) for match-style assertions — it centralizes the panic string, which coverage treats as unreachable.
- `docs/TEST_ISSUES.md` (Chinese) records the rationale for past fixes; `plans/` holds working design notes and is gitignored.

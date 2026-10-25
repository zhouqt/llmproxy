# Test Plan — llmproxy

Goal: **>97% region coverage** across `src/`, with explicit attention to
the fallback router, provider HTTP integrations, OAuth flow, and
request/response format conversions.

## Current state (2026-07-19, session 9)

Coverage as of last `cargo llvm-cov --lib --bins --tests` run:

| Dimension  | Covered  | Total   | Percent  |
|-----------:|---------:|--------:|---------:|
| regions    |   15394  |  15870  | **97.00%** |
| lines      |   10021  |  10212  | **98.13%** |
| functions  |     862  |    882  | **97.73%** |

Test totals: 318 + 14 + 7 + 12 + 14 = **365** tests across all
binaries, all passing in `cargo test --lib --bins --tests`.

**All three metrics now clear the 97% bar.** Session 9 closed the
remaining region gap (session 8 ended at 96.84%) by:
- Touching mock-provider `name()` methods in three existing router
  tests (`complete_retries_per_provider_count`,
  `complete_and_stream_return_non_cooldownable_error_immediately`,
  `complete_skips_provider_that_cannot_serve_model_and_uses_next`,
  `complete_skips_provider_returning_runtime_model_unsupported`) — they
  were compiled but never called, leaving 3-line regions per impl
  uncovered.
- Adding `admin_copilot_auth_returns_internal_error_when_bootstrap_fails_for_other_reason`
  to exercise the `else { e.into_response() }` branch of the
  admin handler that was reachable but untested.
- Adding `build_state_routes_use_proxy_providers_to_proxied_client`
  to cover the `use_proxy: true` branch of `build_state` (the proxied
  HTTP client was wired in but no unit test had asserted it).
- Covering the `failed_providers_header() => None` fallback arm for
  non-`AllProvidersFailed` variants.
- Covering `is_json_content_type`'s `HeaderValue::to_str()` Err path
  via a non-ASCII header value.
- Covering `looks_like_error_envelope`'s `Value::Object` early-return
  for non-object roots.
- Covering `walk()`'s `_ => {}` arm for non-recursive JSON leaves.

## Approach

1. **Pure logic / type tests** — covered through unit tests in
   `src/anthropic.rs`, `src/openai.rs`, `src/responses.rs`, etc.
2. **HTTP integration tests** — `wiremock` 0.6 with `Mock::given`,
   `respond_with`, `set_body_raw(sse, "text/event-stream")`. Pinned
   upstream providers for each `Provider` type.
3. **Server end-to-end tests** — `axum::Router` +
   `tower::ServiceExt::oneshot` exercising the full extractor →
   handler → router → SSE adapter chain.
4. **Mock LLM provider** — `tests/integration_router.rs` builds a
   `Router` from wiremock-backed `OpenAiCompatProvider` and
   `OpenaiResponsesProvider` instances; the entire fallback chain
   runs through these mocks without any real network.
5. **Filesystem tests** — `tempfile` for token store; `XDG_DATA_HOME`
   isolation per test.
6. **Coverage measurement** — `cargo-llvm-cov` 0.8.7 via
   `.cargo/config.toml` (target-dir pinned to `/tmp/llmproxy-target`,
   `.cargo` outside the project so Dropbox-sync safe).

## Test inventory (per file)

| File                                              | #Tests | Notes                                                                                       |
|---------------------------------------------------|------:|---------------------------------------------------------------------------------------------|
| `src/anthropic.rs` (types + serde)                |     3 | roundtrip                                                                                   |
| `src/openai.rs` (types + serde)                   |     1 | roundtrip                                                                                   |
| `src/conversion/cache_hint.rs`                    |   ~12  | Anthropic→OpenAI cache control translation                                                  |
| `src/conversion/request.rs`                       |   ~25  | Anthropic→OpenAI request conversion (system blocks, tools, thinking, images)                 |
| `src/conversion/response.rs`                      |   ~15  | OpenAI→Anthropic response conversion                                                        |
| `src/conversion/stream.rs`                        |   ~20  | OpenAI→Anthropic SSE conversion                                                             |
| `src/conversion/responses.rs`                     |   ~30  | Anthropic↔Responses API conversion                                                          |
| `src/conversion/responses_stream.rs`              |   ~25  | Responses API SSE → Anthropic SSE                                                           |
| `src/cooldown.rs`                                 |   ~5   | TTL boundaries, concurrent marks                                                            |
| `src/error.rs`                                    |   ~12  | `status_code`, `is_cooldownable`, `IntoResponse` for every variant                          |
| `src/oauth/token_store.rs`                        |     8 | fs isolation, xdg env handling                                                              |
| `src/providers/anthropic.rs`                      |   ~10  | bearer auth, anthropic-version header, SSE pass-through                                     |
| `src/providers/openai_compat.rs`                  |   ~25  | model rewriting, SSE adapter, full Anthropic→OpenAI conversion                              |
| `src/providers/openai_responses.rs`               |   ~30  | Responses API: complete + stream, SSE adapter edge cases, error propagation                 |
| `src/providers/copilot.rs`                        |   ~25  | headers, base URL, refresh, GPT-5 → /v1/responses routing, background loop                  |
| `src/router.rs`                                   |   ~50  | full fallback chain, can_serve_model skip, model-unsupported heuristic, cooldown            |
| `src/server.rs`                                   |   ~14  | MappedStream, /admin/copilot/auth 200/409/404 paths, every route                            |
| `tests/integration_router.rs`                     |   ~13  | end-to-end fallback + provider mirroring                                                    |
| `tests/auth.rs`                                   |     7  | bearer / x-api-key / no-key                                                                 |
| `tests/server.rs`                                 |   ~14  | full axum stack, all routes                                                                 |
| **TOTAL**                                         | **357**| + main.rs `build_tracing_filter` formatting test                                           |

## Coverage ceiling (architectural limits)

The 4 categories below account for essentially all remaining
uncovered regions (~476/15870) that cannot be closed without
modifying production code, breaking the test contract, or making
tests more fragile.

### 1. `tracing::` macro bodies (filtered out)

Tests do not initialize a `tracing` subscriber, so
`tracing::warn!`, `tracing::error!`, `tracing::debug!` macro bodies
are filtered out at compile time. Lines covered:

- `src/cooldown.rs:79-81` (warn body in `mark_cooldown`)
- `src/server.rs:153` (error in MappedStream)
- `src/providers/copilot.rs:442, 449-453` (warn/error in spawn loop)
- `src/providers/copilot.rs:218-219, 251, 273` (refresh paths)
- `src/providers/copilot.rs:335-338` (bootstrap failure wrap)
- `src/router.rs:289-291` (can_serve_model debug log)

These regions can only be covered by initializing
`tracing_subscriber::fmt` in a test fixture, which would change test
output behavior across the whole suite. We deliberately leave them
uncovered.

### 2. Unreachable defensive `else { return Err(e) }` arms

When `is_cooldownable()` returns true, the variant is guaranteed
to be `ProxyError::Upstream { status, body }` (the function
hardcodes this). The `else { return Err(e) }` branches at
`src/router.rs:187, 211, 322, 340` therefore can never fire. They
exist only to satisfy the borrow checker (matching `&e` mutably).

### 3. `_ => panic!()` match arms in test code

Tests use `match ev { Variant::A => ..., _ => panic!(...) }` to
distinguish known-good parses from surprises. The `_ => panic!()`
arms are unreachable when the test asserts the expected variant
matched first. Examples: `src/responses.rs:345, 368, 386, 395`;
`src/conversion/responses.rs:386, 389`.

### 4. Constructor `?` early-return on infallible conversions

Constructors like `AnthropicProvider::new`,
`OpenAiCompatProvider::new`, `OpenaiResponsesProvider::new` always
return `Ok`. The `?` operator on their constructors
(`src/providers/mod.rs:82, 98, 108, 118`) cannot fire. Only
`CopilotProvider::new` can return `Err`, and that path is only
reachable by corrupting `XDG_DATA_HOME` mid-test (risky).

### 5. `unimplemented!()` bodies in mock providers

Mock helpers in `src/router.rs` (`CountingMockProvider`,
`RestrictedMockProvider`, `ModelUnsupportedProvider`) have a
`stream()` method that just `unimplemented!()`s because the tests
only exercise the `complete()` path. The `unimplemented!()` body
plus the panic message string inside it are technically regions
that cannot fire without breaking the test.

## 2026-10-21 — secret overlay (`src/secrets.rs`, branch `feat/secrets-file-provider`)

The TPM-sealed-secret plan (`plans/feat-vault-secret-backend.md`, A-Phase 2)
adds a `.env`-format file as a second, higher-priority source for `${VAR}`
references, plus a `secrets:` block in `config.yaml`.

| File | Tests | Covers |
|---|---|---|
| `src/secrets.rs` | 26 unit | every `.env` parsing rule in plan §5.2: blank/comment lines, `export ` prefix, split on the **first** `=`, matching-quote stripping, **no** interpolation, **no** trailing-`#` stripping, duplicate keys, lines without `=`, empty values, CRLF endings, `provider: none/env/file`, unknown-key rejection, blank path, required vs optional missing file |
| `src/config.rs` | 15 unit | two-phase `Config::parse`: overlay beats the process environment, overlay falls back to the environment, `secrets:` block absent ⇒ byte-identical to the pre-feature behavior, `provider: file` without `path` ⇒ config error, `secrets:` (null) ⇒ treated as "not configured" |
| `tests/secrets_file.rs` | 5 integration | `Config::load` against a **real file on disk**: key read from the file, file beats the environment, missing + `required: true` ⇒ startup fails, missing + `required: false` ⇒ env still works, no `secrets:` block ⇒ unchanged |

Coverage after the change: `secrets.rs` **98.70%** regions, `config.rs` 97.36%,
**TOTAL 97.68%** — above the 97% bar.

Two of the `tracing::warn!` branches are asserted through
`crate::tracing_capture::capture_tracing` rather than by inspection:
a plain `warn!` call has no observable effect on coverage unless the
emitted bytes are captured and checked.

## 2026-10-21 — container-side unseal (`deploy/`, branch `feat/secrets-file-provider`)

A-Phase 3 of the same plan adds `deploy/entrypoint.sh`, the TPM packages in
the runtime image, and the compose wiring. Only one Rust behavior changed:

| File | Tests | Covers |
|---|---|---|
| `src/config.rs` | 2 unit | `Config::validate` rejects a `server.api_key` that expanded to `""` (a `${VAR}` pointing at a missing secret must not leave the endpoint open to a bare `Authorization: Bearer `), and still accepts an absent one |

`validate()` runs `Config::parse` for both tests so the expansion path is
exercised, not just the struct field.

The shell entrypoint is not unit-tested (no shell test harness in this
repo); it was verified end to end on `loki` against a real vTPM —
unseal, fallback, wrong-PCR failure, rotation, three consecutive restarts,
`/run` tmpfs, and the overlay/`podman inspect` leak checks. The matrix is
in `plans/feat-vault-secret-backend.md` §A-Phase 3.

Coverage after the change: **TOTAL 97.67%** regions (`secrets.rs` 98.70%),
above the 97% bar.

## 2026-10-22 — credential redaction (`secrecy`, branch `feat/secrecy-redact-keys`)

A-Phase 5 of the same plan replaces every `api_key` field with
`secrecy::SecretString` so that neither `{:?}` nor a serialized config can
print a credential. `secrecy` deliberately does not implement `Serialize`
for `Secret<String>` (un-implementable under the orphan rule — it needs the
`SerializableSecret` marker trait), so the serialized form is redacted by
two field-level `serialize_with` helpers instead.

| File | Tests | Covers |
|---|---|---|
| `src/config.rs` | 6 unit | `{:?}` of a provider/config hides the key; `serde-saphyr` of a provider/config redacts it; an absent `server.api_key` still serializes as absent (auth-disabled must not become auth-configured-with-a-placeholder); the redacted YAML still re-parses |
| `src/secrets.rs` | 2 unit | `secret_string()` prints `[REDACTED …]` under `{:?}`, and `expose_secret()` still returns the value |

The remaining test churn is mechanical: every fixture that built a provider
or a `ServerConfig` now calls `secret_string("…")`, and every provider
assertion reads the key through `expose_secret()`.

Coverage after the change: **TOTAL 97.68%** regions (`config.rs` 97.65%,
`secrets.rs` 98.76%, `auth.rs` 100%), above the 97% bar.

## 2026-10-24 — env-lock poisoning & refresh-loop determinism (branch `fix/env-lock-poison`)

The suite carried an open flake: under `cargo llvm-cov` the library suite
occasionally failed a timing-sensitive test that raced on the process-global
tokio clock or the mock server
(`providers::copilot::tests::spawn_refresh_loop_runs_one_iteration_and_aborts`
and `providers::anthropic::tests::list_models_returns_none_on_non_success_status`
had both been seen), and when the copilot test panicked it poisoned
`oauth::device_flow::ENV_LOCK`, so four other tests failed with `PoisonError`.

Measured on the builder before the fix: the copilot test alone failed
14–15 of 200 runs (~7%); the full lib suite failed **6 of 30** runs (20%);
the `drop(server)` group failed 3 of 60.

Three independent causes, all now fixed:

1. **Lock poisoning cascaded.** `ENV_LOCK` was a plain `std::sync::Mutex`
   and 14 of 17 acquisition sites used `.lock().unwrap()`. It moved to
   `test_support::env_lock()`, which recovers from poisoning via
   `into_inner()` (mutual exclusion preserved, only the poisoned flag is
   discarded), and the manual `set_var` + trailing `remove_var` pattern
   became an RAII `EnvGuard` that restores every variable it touched on
   drop — including during panic unwinding.
2. **A paused clock outran the real HTTP round-trip.** The four
   refresh-loop tests used `#[tokio::test(start_paused = true)]` plus 300
   `advance(60s)` calls: the whole loop ran in a few milliseconds of real
   time, and the assertion raced the in-flight request, which the test then
   aborted. They now await `refresh_once()` directly, with `.expect(1)`
   proving the iteration really ran — two of them previously asserted
   nothing at all and would have passed with the loop never running.
3. **`drop(server)` did not close the listener.** wiremock's
   `MockServer::start()` takes a server from a process-global pool; dropping
   the public handle returns it to the pool but leaves the listener bound,
   so the request was still served and its leftover count broke the next
   test that reused the port.

After the fix, on the same builder:

| Check | Result |
|---|---|
| `cargo test --lib --bins --tests` | 960 tests, all pass (851 lib + 109 integration) |
| `providers::copilot::tests` × 100 (`--test-threads=4`) | 100/100 pass (was ~7% failing) |
| `providers::anthropic::tests::list_models` × 100 | 100/100 pass (was 2/100 failing) |
| `oauth::` × 100 | 100/100 pass |
| `server::` × 100 | 100/100 pass |
| full lib suite × 30 | 30/30 pass (was 6/30 failing) |
| `cargo llvm-cov --lib --bins --tests` | **TOTAL 97.75%** regions, above the 97% bar |

### Regression guardrails

- **Never send a request to a mock server you dropped.** `drop(server)` does
  not close the listener. Use a dead port (`http://127.0.0.1:1`) when the
  test needs a connection that must fail.
- **Keep `.expect(N)` counts exact.** They are the detector for requests
  leaking between tests. Relax to `1..` only when the loop may legitimately
  iterate more than once (see `spawn_refresh_loop_with_delay_runs_an_iteration`).
- **Do not use `#[tokio::test(start_paused = true)]` for "spawned task + real
  I/O".** The paused clock does not wait for the spawned task's HTTP
  round-trip; the test can abort the request mid-flight. Await the future in
  the test task instead, or drive the task through a seam.
- **Any test that builds a `CopilotProvider` (or calls
  `providers::build(GithubCopilot)`) must hold `test_support::env_lock()`
  and point `XDG_DATA_HOME` at a tempdir** — `TokenStore::new` reads it, and
  without the lock the test writes the real user's token store.

## 2026-10-22 — chunked TPM seal (branch `feat/tpm-chunked-seal`)

A-Phase 1 on the real machine: the `.env` does not fit into a single
sealed object. A TPM stores a sealed payload in `TPM2B_SENSITIVE_DATA`,
which Infineon parts cap at **128 bytes** (measured on `loki`: 128 bytes
seals, 129 fails with a TPM error; tpm2-tools' own "expected less than
or equal to 256" message is misleading). The live `.env` is 1704 bytes,
so `deploy/seal.sh` seals it as 14 chunks of ≤128 bytes, each with its
own `.pub`/`.priv` pair, and `deploy/entrypoint.sh` unseals them in
index order and concatenates them.

The scripts are shell, so they have no unit tests; like the A-Phase 3
entrypoint they are verified end to end on `loki` against the real
vTPM:

| Check | Result |
|---|---|
| `seal.sh` round trip on the live `.env` | 14 chunks, `sha256(source) == sha256(unsealed)` |
| entrypoint TPM path, 3 runs in a row | byte-identical output each time (no transient-slot leak) |
| entrypoint fallback path | copies the fallback file, exits 0 |
| entrypoint with neither mode | exits 1 with a named error |
| chunk order | `LC_ALL=C` + 4-digit zero-padded index ⇒ byte order == numeric order |
| host reboot (2026-10-23) | `0x81000001` still present, PCR 0/7 byte-identical to pre-reboot, all 14 chunks unseal to the same digest |

The real `.env` is never unsealed into a file on the host: its round
trip is checked through a pipe (digests only), and the entrypoint
script is exercised end to end with a **non-secret** payload. Writing
the real plaintext back to the host disk is exactly what this mechanism
exists to prevent.

One pre-existing bug was found and fixed on the way: the entrypoint
left the secrets file at mode `0400`, so the next start could not
overwrite it. Inside the container that was invisible (the entrypoint
runs as root), but the same script on the host failed with
`Permission denied`. The file is now removed before it is written.

## Verification

```bash
cargo install cargo-llvm-cov    # one-time
cargo test --lib --tests        # 357 tests, all pass
cargo llvm-cov --lib --tests    # coverage report
```

Baseline snapshot: `/tmp/cov-final.json` (15682 regions / 871 functions / 10098 lines).

## Out of scope (explicitly not testing)

- `main.rs` arg parsing and bootstrap ordering
  (43 region misses — the `tracing_subscriber::fmt::init()` path
   is gated by env vars set at the operator's process boundary,
   not testable in-process without race conditions).
- Real network calls to OpenAI / Anthropic / Copilot
  (these are integration-tested through `wiremock`).
- Cross-platform Windows-specific `OpenOptionsExt` branches
  in `oauth/token_store.rs:202-208`.
- Suspicious-state defensive code that requires corrupting
  on-disk state mid-test (e.g. simulating fs permission errors).

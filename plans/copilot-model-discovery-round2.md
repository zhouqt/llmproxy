# Copilot Model Discovery — Round 2

## Review verdict (opus, round 1)

The feature compiles, has unit tests for the provider internals, and the
manual smoke from round 1 works once the cache is warm. But four things
make it unsafe to ship:

1. **Cold-start gap.** `list_models_handler` only reads `cp.cached_models()`.
   On boot with a valid stored token, the refresh loop sleeps
   `(refresh_in - 60)s` (~24 min default) before the first `cache_models()`.
   `/v1/models` returns zero Copilot models for that entire window, and
   nothing in the handler fetches on miss — the round-1 plan called for
   lazy init, the round-1 code dropped it.
2. **Deadlock in bootstrap path.** `complete_bootstrap` runs while the
   spawned task already holds `refresh_lock`, then calls `cache_models()`
   → `fetch_models()` → `ensure_token()`. If the freshly minted token
   looks expired (clock skew, weird `expires_at`), `ensure_token` calls
   `refresh_token()` → re-locks the non-reentrant `tokio::Mutex` → task
   hangs forever holding the lock → all Copilot traffic hangs.
3. **No `policy.state` filter.** Real Copilot `/models` includes disabled
   and preview entries. The round-1 merge appends them all, so `/v1/models`
   advertises models that 4xx on use.
4. **Stringly-typed merge.** `RwLock<Option<Value>>` pushes raw JSON across
   the provider/server boundary; the merge is six levels of
   `.get().and_then()`. A typed `CopilotModel` struct catches schema drift
   at parse time, not at the user.

There is also one formatting regression (`copilot.rs:997` — 12-space indent
instead of 16), no test coverage for the actual `/v1/models` merge, and
the dedup comment in `server.rs` is wrong (dedup is by id, not by
provider).

## Goals for round 2

- Close the cold-start gap: cache populated within one refresh loop
  iteration when an unexpired token is in memory.
- Kill the bootstrap-path deadlock: `cache_models()` after bootstrap must
  not call `ensure_token()` (it has a fresh token in hand).
- Filter `policy.state != "enabled"` out of the cache; never advertise a
  disabled model.
- Type the cache as `Vec<CopilotModel>` and extract the merge into a pure
  function with unit tests.
- Pass `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  and `cargo test --lib --tests` clean.

## Design

### 1. Typed cache (`src/providers/copilot.rs`)

Replace the raw `cached_models: RwLock<Option<Value>>` with:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct CopilotModel {
    pub id: String,
    pub name: String,
    pub vendor: String,
    // Capabilities and policy intentionally not stored — see below.
}

struct CopilotState {
    // ...
    cached_models: RwLock<Option<Vec<CopilotModel>>>,
}
```

`fetch_models` becomes:

```rust
async fn fetch_models(&self, token: &str) -> Result<Vec<CopilotModel>, _> {
    // GET {models_url} with the caller-provided token (skip ensure_token).
    // Parse response as `{ "data": [ { id, name, vendor, policy: { state } }, ... ] }`.
    // Filter: keep only entries whose `policy.state == "enabled"`.
    // (Copilot returns `enabled` for generally available models and
    // `disabled` / `preview` for others; missing policy is treated as
    // not enabled.)
}
```

`cache_models` keeps its best-effort shape: warn on error, leave the
existing cache alone (don't overwrite a good cache with None on a
transient failure).

### 2. Kill the bootstrap-path deadlock

`complete_bootstrap` already has the fresh token in hand after
`fetch_copilot_token`. Replace:

```rust
self.cache_models().await;
```

with a new public method that takes a token, so it never calls
`ensure_token` and never re-enters `refresh_lock`:

```rust
pub async fn cache_models_with_token(&self, token: &str) { ... }
```

`cache_models()` (no arg) becomes a thin wrapper used only by the refresh
loop, where `ensure_token` is safe (no lock held).

### 3. Close the cold-start gap

In `spawn_refresh_loop`, before the first `tokio::time::sleep`:

```rust
if has_credentials {
    if let Some(t) = current_token_if_unexpired() {
        self.cache_models_with_token(&t).await;
    }
}
```

This matches the Python reference: `cache_models()` is called once at
startup, then again on every token refresh.

### 4. Extract the merge as a pure function (`src/server.rs`)

```rust
fn merge_copilot_models(
    static_entries: Vec<Value>,
    copilot: &[CopilotModel],
) -> Vec<Value> {
    // 1. Build a HashSet<String> of copilot ids (dedup intra-Copilot too).
    // 2. Remove static entries whose id is in the set.
    // 3. Map each CopilotModel to OpenAI shape:
    //    { id, object: "model", created: 0, owned_by: vendor, display_name: name }
    // 4. Concatenate: kept-static ++ copilot-mapped.
}
```

`list_models_handler` becomes a two-line merge call. The four behaviors
worth testing live here, not in the handler.

`display_name` stays in the response even though it is not part of
OpenAI's strict `/v1/models` schema — round 1 already ships it and the
plan calls for it; removing it is a separate decision. (Tracked but out
of scope.)

## Critical files

| File | Change |
|------|--------|
| `src/providers/copilot.rs` | Add `CopilotModel` struct. Add `fetch_models(&self, token: &str)`, `cache_models_with_token(&self, token: &str)`. Change `cached_models` field to `RwLock<Option<Vec<CopilotModel>>>`. Rewrite `cache_models()` to read `ensure_token` then delegate. Filter `policy.state == "enabled"`. In `complete_bootstrap`, call `cache_models_with_token(&new_token)`. In `spawn_refresh_loop`, pre-sleep `cache_models_with_token` when an unexpired token is in memory. |
| `src/server.rs` | Extract `merge_copilot_models(static, copilot) -> Vec<Value>` with four inline tests. Reduce `list_models_handler` to a call + JSON wrap. |

## Test plan (must pass before merge)

### New tests — `src/providers/copilot.rs::tests`

1. `fetch_models_filters_disabled_policy` — mock returns one `enabled` and
   one `disabled` entry; assert the returned vec has length 1 and contains
   only the enabled id. (Required to prove policy filtering works.)
2. `fetch_models_drops_missing_policy` — mock returns an entry with no
   `policy` field; assert it is filtered out.
3. `cache_models_with_token_does_not_call_ensure_token_when_token_unexpired`
   — pre-set an unexpired token; call `cache_models_with_token`; the mock
   should see exactly one `GET /models` and zero token-refresh calls.
4. `cache_models_keeps_stale_cache_on_failure` — populate cache, then mock
   `/models` to return 503, call `cache_models()`; assert
   `cached_models()` still returns the prior value. (Required to prove we
   don't blank the cache on transient failure.)
5. `spawn_refresh_loop_caches_models_on_first_iteration_with_unexpired_token`
   — pre-seed unexpired token, mock `/models`, run one loop iteration with
   paused time, assert `cached_models().is_some()` and `GET /models`
   `.expect(1)`.

### New tests — `src/server.rs::tests`

6. `merge_copilot_models_dedup_static_with_copilot` — static has
   `[{"id":"gpt-4","owned_by":"llmproxy"}]`, copilot has one entry with
   `id=gpt-4`, vendor=`OpenAI`; result has length 1, owned_by=`OpenAI`,
   display_name present.
7. `merge_copilot_models_keeps_non_colliding_static` — static has
   `[{"id":"local-1"}]`, copilot has one entry with `id=gpt-4`; result
   has length 2 in order static-then-copilot.
8. `merge_copilot_models_empty_cache_passthrough` — copilot is empty;
   static is unchanged.
9. `merge_copilot_models_dedupes_intra_copilot_duplicates` — copilot has
   two entries with the same id; result has length 1.

### Build gates (must be clean)

- `cargo fmt --check` — must pass (round 1 had a real failure here).
- `cargo clippy --all-targets -- -D warnings` — must pass.
- `cargo test --lib --tests` — all of the above nine tests green.

### Manual smoke (unchanged from round 1)

```bash
cargo run &
sleep 2
curl -s http://localhost:9090/v1/models -H "Authorization: Bearer <key>" | jq
```

Expect: response includes Copilot-discovered entries with `owned_by`
set to the vendor name (e.g. `OpenAI`, `Anthropic`). Expect: no entries
whose `policy.state != "enabled"` (verified by listing what `/v1/models`
returns and comparing against the raw `GET {copilot}/models` JSON).

## Out of scope for round 2

- `display_name` is non-standard for OpenAI's `/v1/models`. Documented
  here so we don't forget; punted to round 3 if it stays contentious.
- Pagination — Copilot's `/models` returns the full catalog today.
- Per-model capability introspection (vision, tools) — Copilot returns
  this; round 2 keeps it on the floor intentionally. Round 3 candidate.
- Backoff/retry on `fetch_models` failures inside the refresh loop —
  best-effort log is fine for now.

## Verification standard (formal)

A round-2 commit is mergeable iff:

1. `cargo fmt --check` exits 0.
2. `cargo clippy --all-targets -- -D warnings` exits 0.
3. `cargo test --lib --tests` exits 0 with all nine tests above passing
   (named tests are findable via `cargo test <name>`).
4. `git diff main..HEAD -- src/` shows no occurrence of
   `cached_models: RwLock::new(None)` outside of test setup that has
   been updated for the new type (sanity check that all the field
   declarations got rewritten).
5. No new occurrence of `.get("policy").and_then(...)` outside of
   `fetch_models` (policy filter lives in one place).
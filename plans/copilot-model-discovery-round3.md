# Copilot Model Discovery — Round 3

## Review verdict (opus, round 2)

Round 2 closed the bootstrap deadlock, the policy leak, and the stringly-
typed merge. But three of round-2's choices introduced their own
problems, and a fourth was deliberately deferred from round 2.

1. **Typed cache defeats its own purpose.** `serde_json::from_value(...).ok()`
   on `copilot.rs:597-599` silently drops any enabled entry that fails to
   parse. No warning, no log. If upstream renames `vendor` → `provider`
   (no recorded fixture proves which is right), `/v1/models` would
   silently advertise zero Copilot models. The point of the typed struct
   was to surface drift; this buries it.
2. **Cold-start is half-closed.** The pre-sleep block in
   `spawn_refresh_loop` only fires when an *unexpired* token is in
   memory. A reboot after `copilot_expires_at` elapsed — the common
   case after >30 min downtime — still waits ~24 min for the first
   `cache_models()` call.
3. **401 is treated as transient.** `fetch_models` classifies every
   non-2xx as `CopilotFetchError::Transient` and `cache_models` keeps
   the existing cache. If the GitHub token is revoked, the cache
   advertises stale models until the next refresh loop iterates, with
   no signal to the operator.
4. **`display_name` asymmetry.** Present only on Copilot-discovered
   entries. Static config entries omit it. This is a one-response,
   two-schema inconsistency that round-2 fossilized into the merge
   function.

There is also a real test-coverage gap: the existing
`start_bootstrap_runs_device_flow_and_persists_tokens` does not mount
`/models`, so a deadlock regression introduced in round 3 would not be
caught.

## Goals for round 3

- Make schema drift visible (warn + count on parse failure).
- Close the cold-start gap fully: cache populates whenever the store
  has credentials, not only when memory holds an unexpired token.
- Distinguish 401/403 (stale-cache invalidation, no warn spam) from
  5xx (transient, keep cache, warn once per failure).
- Resolve `display_name` asymmetry across one response.
- Add tests that pin all of the above.

## Design

### 1. Surface schema drift on parse failure (`src/providers/copilot.rs`)

Replace:

```rust
.filter_map(|entry| {
    serde_json::from_value::<CopilotModel>(entry.clone()).ok()
})
```

with:

```rust
.filter_map(|entry| {
    match serde_json::from_value::<CopilotModel>(entry.clone()) {
        Ok(m) => Some(m),
        Err(e) => {
            let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            tracing::warn!(
                entry_id = id,
                error = %e,
                "dropping copilot model entry: schema drift"
            );
            None
        }
    }
})
```

The goal is to make silent breakage impossible. One warn per dropped
entry is the right cadence: if upstream ships 100 dropped entries the
operator sees 100 warn lines and knows something is wrong, but normal
catalog churn won't drown the log.

### 2. Close the cold-start gap fully (`src/providers/copilot.rs`)

Replace the pre-sleep "if unexpired in memory" block with:

```rust
// At the top of the first iteration, BEFORE the first sleep:
// If the on-disk store has credentials, kick a best-effort cache_models().
// No lock is held here, so cache_models() can call ensure_token() safely.
if has_credentials {
    self.cache_models().await;
}
```

This matches the Python reference (cache at startup regardless of token
freshness). `cache_models()` itself delegates to
`cache_models_with_token` after calling `ensure_token`, so the deadlock
fix from round 2 still holds.

### 3. Split 401/403 from 5xx in `fetch_models`

`fetch_models` becomes:

```rust
match status.as_u16() {
    401 | 403 => {
        let text = resp.text().await.unwrap_or_default();
        // Token revoked. Clear the cache so /v1/models stops advertising
        // models we can no longer authenticate against. Do NOT warn at
        // error level — this is a recoverable auth state.
        tracing::info!(status = status.as_u16(), body = %text,
            "copilot token rejected by /models; clearing cache");
        // Caller (cache_models) decides whether to clear or to keep
        // based on the returned CopilotFetchError variant.
        Err(CopilotFetchError::Auth(format!("copilot /models returned {status}")))
    }
    s if s >= 500 => Err(CopilotFetchError::Transient(format!("copilot /models returned {s}"))),
    s => Err(CopilotFetchError::Transient(format!("copilot /models returned {s}: {text}"))),
}
```

Add `CopilotFetchError::Auth(String)`. `cache_models()` matches:

```rust
match self.fetch_models().await {
    Ok(v) => { /* store */ }
    Err(CopilotFetchError::Auth(msg)) => {
        tracing::info!("{msg}");
        *self.state.cached_models.write().await = None;
    }
    Err(e) => {
        tracing::warn!(error = %e, "copilot /models fetch failed; keeping stale cache");
    }
}
```

This means a revoked GitHub token now produces an immediate empty
cache (correct: we can't authenticate against the upstream catalog)
without spamming the log, while a 5xx leaves the cache intact.

### 4. Resolve `display_name` asymmetry (`src/server.rs`)

Pick one direction and apply uniformly:

```rust
fn merge_copilot_models(
    static_entries: Vec<Value>,
    copilot: &[CopilotModel],
) -> Vec<Value> {
    // ...
    for m in copilot {
        mapped.push(json!({
            "id": m.id,
            "object": "model",
            "created": 0,
            "owned_by": m.vendor,
            "display_name": m.name,  // present
        }));
    }
    mapped
}
```

Plus an update to the static-entry builder in `list_models_handler`
to emit `"display_name": <name-or-empty>` for every static entry, so
the response is internally consistent. Source of truth for static
`display_name` is the same field as `id` for now (we have no separate
display-name config). Document this choice in a comment in
`list_models_handler`.

### 5. Extract `token_expired` helper

`copilot.rs:210` uses `expires_at - now < 60`; `copilot.rs:452` uses
`expires_at > now`. Replace both with:

```rust
fn token_expired(tokens: &StoredTokens, now_unix: i64) -> bool {
    tokens.copilot_expires_at <= now_unix
}
```

Both call sites become `token_expired(&t, now)`. The 60-second
buffer that the old `expires_at - now < 60` provided was a refresh
heuristic, not an auth-freshness check; that's better expressed by
the refresh loop's interval logic, not by the expiry predicate.
(Call site at `copilot.rs:210` was using the buffer as "treat as
expired soon"; replace with `copilot_expires_at <= now` and let the
refresh loop's `refresh_in` handle the buffer.)

## Critical files

| File | Change |
|------|--------|
| `src/providers/copilot.rs` | (1) Replace silent `.ok()` with warn-and-None on parse failure. (2) Replace pre-sleep "if unexpired" with unconditional `cache_models().await` when store has credentials. (3) Add `CopilotFetchError::Auth` variant; split 401/403 vs 5xx in `fetch_models`; clear cache on Auth. (4) Extract `token_expired` helper, use at both call sites. (5) Update `start_bootstrap_runs_device_flow_and_persists_tokens` test to mount `/models` and assert `cached_models().is_some()` post-bootstrap. |
| `src/server.rs` | Add `"display_name"` to the static-entry builder in `list_models_handler` so static and Copilot entries have identical keys. |

## Test plan (must pass before merge)

### New tests — `src/providers/copilot.rs::tests`

1. `fetch_models_warns_and_drops_entry_missing_required_field` — mock
   returns one well-formed entry and one entry missing `vendor`; assert
   the returned vec has length 1 and contains the well-formed id. (Pin
   the warn-but-don't-silently-swallow behavior. The test does not
   inspect logs; the assertion is on the vec.)
2. `fetch_models_returns_auth_error_on_401_and_caller_clears_cache` —
   mock returns 401; `cache_models()` after a prior good cache results
   in `cached_models().is_none()`.
3. `fetch_models_returns_auth_error_on_403_and_caller_clears_cache` —
   mirror of (2) with 403.
4. `fetch_models_keeps_stale_cache_on_503` — already covered by
   `cache_models_keeps_stale_cache_on_failure`; verify the new error
   path still routes to the stale-keep branch.
5. `spawn_refresh_loop_caches_models_when_memory_token_is_expired`
   — pre-seed an expired token (expires_at in the past); mock `/models`;
   run one loop iteration; assert `cached_models().is_some()` and the
   mock saw `.expect(1)` on `/models`. (Pins the round-3 cold-start
   fix.)
6. `start_bootstrap_after_existing_token_populates_cache` — start a
   bootstrap while a token is already in memory; mock `/models` with
   `.expect(1)`; assert `cached_models().is_some()` after bootstrap
   completes. (Also serves as the round-3 deadlock regression test:
   if `complete_bootstrap` ever goes back through `ensure_token`
   while holding `refresh_lock`, this hangs under the test timeout.)

### New tests — `src/server.rs::tests`

7. `merge_copilot_models_includes_display_name_on_every_entry` —
   static has `[{"id":"local-1"}]`; copilot has one entry; assert
   every entry in the result has a `display_name` key (string,
   possibly empty for static).

### Updated test — `src/providers/copilot.rs::tests`

8. `start_bootstrap_runs_device_flow_and_persists_tokens` — mount
   `/models` (200 with one enabled entry) with `.expect(1)`. After
   the existing assertions, also assert
   `provider.cached_models().await.is_some()`.

### Build gates (unchanged from round 2)

- `cargo fmt --check` — must pass.
- `cargo clippy --all-targets -- -D warnings` — must pass.
- `cargo test --lib --tests` — all of the above new + updated tests
  green.

### Sanity greps (unchanged + one new)

- `.get("policy")` appears only inside `fetch_models`.
- `cached_models: RwLock::new(None)` outside of `new()` and tests.
- **NEW**: `serde_json::from_value::<CopilotModel>` is followed by a
  match, never `.ok()` (grep for `from_value::<CopilotModel>` and
  confirm the next non-whitespace token is `match`, not `.`).

### Manual smoke

```bash
# 1. Bootstrap once, kill the server.
cargo run &
sleep 5  # wait for bootstrap + cache_models
curl -s http://localhost:9090/v1/models -H "Authorization: Bearer <key>" | jq
# Expect Copilot entries with display_name.
kill %1

# 2. Simulate revoked-token response by patching stored token.
# (Manual-only; CI cannot exercise.) Expect /v1/models to drop
# Copilot entries within one refresh-loop iteration.

# 3. Restart after >30 min downtime.
# Expect /v1/models to include Copilot entries within ~5 s, not 24 min.
```

## Verification standard (formal)

A round-3 commit is mergeable iff:

1. `cargo fmt --check` exits 0.
2. `cargo clippy --all-targets -- -D warnings` exits 0.
3. `cargo test --lib --tests` exits 0 with all eight tests above
   passing (named tests are findable via `cargo test <name>`).
4. `git diff 9e86fb0..HEAD -- src/ | grep -E '\.ok\(\)' | grep -i
   'CopilotModel'` returns no matches (silent-drop regression
   detector).
5. `grep -n 'CopilotFetchError::' src/providers/copilot.rs` shows
   the new `Auth` variant is constructed in `fetch_models` and
   matched in `cache_models`.
6. No static entry in `list_models_handler` is built without a
   `display_name` key (assert via `git diff` inspection).

## Out of scope for round 3

- Recording a real `/models` fixture. Worth doing but requires a live
  Copilot token; punted to a future session.
- Per-model capability introspection (vision/tools) — Copilot returns
  `capabilities`; round 3 keeps it on the floor.
- Token-refresh backoff in the refresh loop. Best-effort is fine for
  the model list; chat traffic has its own retry.
- `display_name` as a configurable per-model field. Today's static
  config has no name field; round 3 uses `id` as the placeholder.
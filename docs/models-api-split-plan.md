# Splitting `/v1/models`: the routing-winner list + the full per-provider model list

## Background

`/v1/models` (`list_models_handler` in `src/server.rs`) currently merges the
static config entries with the `list_models()` results of every provider and
de-duplicates them by `id` (the last occurrence wins). There are three
problems:

1. **The dedup winner depends on `HashMap` iteration order.**
   `Router.providers` is a `HashMap<String, SharedProvider>`
   (`src/router.rs:61`), so the iteration order differs on every process start.
   When several providers expose the same model id, the returned `owned_by` is
   a random provider -- neither the provider that would actually serve the
   model (the primary comes first) nor stable across restarts.
2. **`owned_by` is not the provider name from the config**: `anthropic.rs`
   hardcodes `"anthropic"`; `openai_compat.rs` / `openai_responses.rs` fall
   back to a type tag; copilot uses the upstream vendor. It does not match the
   provider names used by `/admin/status` and `x-llmproxy-failed-providers`.
3. **The semantics are muddled**: the endpoint cannot express both "which
   models the client will actually use" and "which models each upstream
   provider offers".

The user confirmed the original intent was "list every model of every
provider", but the two semantics now have to be split: the existing
`/v1/models` becomes a correct view of **the models the current routing
actually uses** (including the real serving provider), and a new endpoint
lists **the full model list of every provider**.

---

## Requirement 1: fix the existing `/v1/models` (routing-winner view)

### 1.1 Inject the source provider name while aggregating

In `list_models_handler`, overwrite every entry returned by
`provider.list_models()`:

- **Invariant: after aggregation the handler guarantees that every entry's
  `owned_by` equals that provider's key in the config (the key of
  `router.providers()`)**. The provider layer may omit the field; the
  fallback is the handler's responsibility.
- New optional field `upstream_owned_by` <- the provider's original
  attribution (vendor/organization), kept only when the original entry
  carries an `owned_by`.

Static config entries keep `owned_by: "llmproxy"`.

### 1.2 Deduplicate by routing priority

A new function replaces `dedup_models_last_wins`:

```rust
fn dedup_models_by_routing_priority(entries: &mut Vec<serde_json::Value>,
                                    priority: &HashMap<(String, String), usize>)
// key = (upstream model id, provider name)
```

**The priority table is keyed by the "upstream id"** (the `id` field of the
`list_models()` entry), not by `config.models[*].name` (which is the
client-facing name). How it is built:

1. The priority is built only from `config.models` (a `Vec` in declaration
   order, hence deterministic and independent of the HashMap); for every
   `(client_name, provider)` pair of a chain, translate it with that
   provider's `merged_rewrite(&HashMap::new())`:
   `upstream_id = rewrite.get(client_name).unwrap_or(client_name)`;
2. The primary gets `0`, the `fallback_chain` gets `1, 2, ...` in order; the
   same `(upstream_id, provider)` appearing in several ModelConfigs takes the
   smallest value;
3. **Apply the `can_serve_model(client_name)` filter** -- note that the
   argument is the **client name**, not the translated upstream id:
   `can_serve_model` checks `model_rewrite.contains_key(model)`
   (`openai_compat.rs` / `copilot.rs` etc.), and the key is the client name,
   consistent with the router's own dispatch check
   (`can_serve_model(&req.model)` in `router.rs`). Passing the upstream id by
   mistake would wrongly exclude providers with a non-empty rewrite. The
   filter applies to the **input set of the dedup**: the entries of providers
   that cannot serve the model are dropped from `/v1/models` (the priority
   table itself is unaffected) -- those entries still show up in
   `/admin/models`, which reflects the real routing behaviour.
4. **Static config entries (`owned_by: "llmproxy"`) count as the lowest
   priority (`usize::MAX`)**: they have no provider name, so they always miss
   the priority table; the rule is -- when a discovered entry with a finite
   priority exists for the same id, the discovered entry wins, otherwise the
   static entry is kept.

Dedup rule: when several entries share an id, the one with the smallest
priority (the one the routing actually reaches first) is kept; a tie across
several ModelConfigs is broken by the declaration order in `config.models`
(the earlier one wins). An id that is in no chain is sorted stably by
`(provider name, id)` and the first one is taken, so the result is
deterministic. Entries without an `id` or with an empty `id` keep the
current behaviour: skipped, with a warning.

### 1.3 Behaviour changes

- Every returned entry's `owned_by` matches a provider name in
  `/admin/status`.
- The client-facing shape is unchanged (still the OpenAI-style
  `{object: "list", data: [...]}`); only the field values become more
  accurate, and the new field is optional, so it stays backward compatible.
- **Breaking change (internal)**: on an id conflict the winner changes from
  "random last-wins" to "chain primary". In the existing test
  `list_models_aggregates_static_and_provider_discovered_models` in
  `tests/server.rs`, the expected `owned_by` for `gpt-4o` changes from
  `"openai"` (discovered, last-wins) to the primary's name; the test has to be
  updated and its comment has to state that this is an intentional behaviour
  change.

## Requirement 2: a new `/admin/models` (the full per-provider model list)

### 2.1 Routing and shape

- `GET /admin/models`, registered in the routing table in `src/server.rs`, on
  the same level as `/admin/status` (a proxy-introspection endpoint, not part
  of the OpenAI-compatible surface).
- Authentication: keep the existing `/admin/*` middleware behaviour (identical
  to `/admin/status`).
- Response shape (grouped by provider, no deduplication at all):

```json
{
  "object": "list",
  "providers": [
    {
      "provider": "deepseek",
      "source": "static|discovered|static+discovered",
      "cache_state": "populated|cold|auth_missing|fetch_failed",
      "models": [
        {"id": "...", "display_name": "...", "created": 0, "upstream_owned_by": "..."}
      ]
    }
  ]
}
```

- **The static source comes from the provider-level `model_rewrite` table**
  (there is no `model_rewrite` field on `ModelConfig`; it lives on the
  `ProviderConfig` variants): a non-empty table is an explicit allow-list and
  yields one `source: "static"` entry per key (the id is the rewrite's
  **value**, i.e. the upstream name); an empty table means there is no static
  source and every entry is discovered.
- **Merge rule for static and discovered within one provider group**: they
  are merged by `id` (string equality) inside the same provider -- the static
  entries are placed first; when a discovered entry's id already exists,
  `source` is upgraded to `"static+discovered"` and the metadata
  (`created`/`display_name`/`upstream_owned_by`) is taken from the discovered
  (upstream catalogue) entry; otherwise the entry is inserted as
  `"discovered"`. Duplicate values inside the static set (several keys
  mapping to the same upstream name) keep only one entry; which one's
  metadata survives makes no difference, but for determinism the **smallest
  client key in lexicographic order** wins (`model_rewrite` is a `HashMap`, so
  the iteration order is random and the declaration order cannot be used).
  The output is sorted by `id`.
- The discovered entries come from `list_models()`. **When `list_models()`
  returns `None` the provider still appears in the result, with an empty
  `models` array.** How `cache_state` is obtained:
  - `Provider::list_models()` only returns `Option<Vec<Value>>`, and `None`
    does not distinguish the reason, so every non-Copilot provider reports
    `"fetch_failed"` (no distinction between a network error, a non-2xx
    status or a JSON parse failure);
  - Copilot uses the existing `AppState.copilot: Option<Arc<CopilotProvider>>`
    handle to get the fine-grained state: add a "reason for the last failure"
    field to `CopilotState` (recorded in the error branches of
    `cache_models_with_token`) and expose a `pub fn cache_state(&self)`
    method; the handler prefers that handle. The enum values follow the real
    signals of `CopilotState` -- note that the `from_store` constructor
    already loads from the on-disk cache, so there are only three states:
    **`populated`** (a value in memory), **`auth_missing`** (cleared from
    memory by an auth failure, the disk cache still there) and **`cold`**
    (neither in memory nor on disk). There is no `"never_fetched"` (after
    construction it cannot be told apart from `cold`). The list may lag one
    background refresh cycle behind (the implementation takes the exact value
    from the refresh interval of `spawn_background`); it is not real-time
    data.
- The endpoint **deliberately does not deduplicate**: the same upstream id
  may appear in several providers' lists at once, and consumers that need a
  unique-id view (capacity statistics, say) should deduplicate in their own
  layer. Providers with a large catalogue (OpenRouter with hundreds of
  models) make for a large response body -- an operational introspection
  endpoint accepts that trade-off, so there is no pagination.
- A failure of a single provider does not affect the whole: the response is
  always 200 (consistent with the degradation semantics of `/admin/status`), a
  failed provider is expressed as an empty `models` plus a `cache_state`, and
  it never returns 500.
- A metadata endpoint that only does pass-through queries: **it does not
  propagate `provider_ignore`** (that is injected into the `/v1/messages`
  request body), and OpenRouter's `/v1/models` has nothing to do with the
  provider routing of the messages endpoint.

### 2.2 Implementation

- Handler: `all_models_handler(State<AppState>)`, reusing each provider's
  `list_models()`. The iteration order follows `state.config.providers` (a
  `Vec` with a deterministic declaration order, the same pattern as
  `provider_status()` in `/admin/status`); for each name it looks up the
  matching `SharedProvider` in `state.router.providers()` and calls
  `list_models()`; `model_rewrite` is taken straight from the
  `ProviderConfig` variant (add an accessor if needed).
- **Concurrency + timeout**: the per-provider `list_models()` calls run
  concurrently via `futures::future::join_all`, each wrapped in a
  `tokio::time::timeout` (10s or a configurable metadata timeout is
  recommended). The shared reqwest client defaults to a 600s timeout
  (`proxy_client.rs`); without a timeout a single hung upstream would block
  the endpoint for up to 10 minutes. A provider that times out is handled as
  `cache_state: "fetch_failed"` with an empty `models` list, and the run
  continues. The sequential iteration of the existing `/v1/models` handler is
  switched to the same concurrency + timeout pattern while we are here.
- The normalisation logic inside each provider's `list_models()` stays as it
  is; the `owned_by` fallback tag (`openai_compat` etc.) no longer carries
  meaning on this endpoint -- the outer `provider` field replaces it, and the
  internal fallback can go (see 2.3).

### 2.3 Small changes in the provider layer

In the `list_models()` normalisation of `anthropic.rs` / `openai_compat.rs` /
`openai_responses.rs`, replace the hardcoded type-tag fallback with
pass-through of the upstream value: when the upstream omits it, the field is
omitted entirely, **and the handler fills in each entry's `owned_by` with the
provider's config name (the invariant is in 1.1; the `provider` in the §2.1
response is the grouping wrapper field and is a different thing from the
entry-level `owned_by`)**. Per file:

- `anthropic.rs`: it currently **does not read** the upstream `owned_by` at
  all (it hardcodes `"anthropic"` unconditionally); it has to read
  `entry.get("owned_by")` like `openai_compat.rs` does, store the result in
  `upstream_owned_by` and omit the field when it is missing;
- `openai_compat.rs` / `openai_responses.rs`: they already read the upstream
  `owned_by` correctly; only the `unwrap_or("openai_compat")` /
  `unwrap_or("openai_responses")` fallback strings have to go, and the field
  is omitted when it is missing;
- `copilot.rs`: it no longer writes `owned_by` in its output, `vendor` moves
  to `upstream_owned_by`, and the entry-level `owned_by` is filled in by the
  handler.

---

## Test plan (keep >97% region coverage)

Unit tests (inside `src/server.rs`):

- **Delete the old function `dedup_models_last_wins` and its three direct
  tests** (`dedup_keeps_last_occurrence_for_duplicate_id` /
  `dedup_filters_out_empty_id_entries` / `dedup_preserves_unique_entries`,
  around `src/server.rs:863-906`) -- their coverage is taken over by the new
  tests below.
- `dedup_models_by_routing_priority`: the primary beats the fallback; the
  upstream id translated through `merged_rewrite` matches correctly; the
  `can_serve_model(client_name)` filter is correct (including a case with a
  non-empty rewrite where "the client name is the key and the upstream id is
  the value"); a static entry (`llmproxy`) always loses to a discovered entry
  with the same id and is kept when there is no discovered one; a tie across
  ModelConfigs is broken by the declaration order in `config.models`;
  unregistered ids are sorted stably; entries without an id or with an empty
  id are skipped without a panic.
- Handler tests (a fake provider, modelled on the existing
  `tests/server.rs:887`):
  - two providers expose the same id -> the `/v1/models` winner follows the
    chain order and is deterministic;
  - a `/v1/models` entry's `owned_by` = the provider's config name, and
    `upstream_owned_by` keeps the upstream value;
  - `/admin/models` returns each provider's own full list (not deduplicated),
    an empty `models` plus a `cache_state` for a failure, and the overall
    status is still 200;
  - a hung fake provider (that never answers) -> the endpoint returns within
    the timeout, that provider is recorded as `fetch_failed` and the other
    providers' data is complete (covers the concurrency + timeout path).

Integration tests (`tests/server.rs`):

- Update `list_models_aggregates_static_and_provider_discovered_models` to
  match the new `owned_by` values.
- Add assertions for the existence of the `/admin/models` route, its
  authentication and its response shape.

Provider unit test updates:

- `anthropic.rs:1744`, `openai_compat.rs:1556`, `openai_responses.rs:1313`
  and the tests of the error branches (the line numbers have to be re-checked
  at implementation time): assert that the type-tag fallback is no longer
  written and that `owned_by` is either decided by the upstream value or
  omitted entirely.
- The `list_models` tests in `copilot.rs`: assert that the vendor moved to
  `upstream_owned_by` and that `owned_by` is no longer filled in by the
  provider layer.

## Order of implementation

1. Remove the `owned_by` fallback in the provider layer + update the unit
   tests (including anthropic reading the upstream `owned_by` and copilot
   moving the vendor).
2. Copilot failure tracking: add the "last failure reason" field to
   `CopilotState` + `CopilotProvider::cache_state()`.
3. Server layer: build the priority table, add the new dedup function and
   rework `/v1/models` (the priority table is built from `config.models` in
   the same step; it is unrelated to the concurrency rework).
4. Add the `/admin/models` handler + route + align the authentication; both
   handlers' `list_models()` collection is switched to concurrent `join_all`
   with a per-call timeout.
5. Full test suite: `cargo test --lib --bins --tests`, then run
   `cargo llvm-cov --lib --bins --tests` to confirm the coverage does not
   regress.

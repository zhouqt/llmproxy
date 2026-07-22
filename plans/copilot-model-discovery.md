# Copilot Model Discovery

## Context

llmproxy's `/v1/models` returns only statically configured models from
`config.yaml`. The Python reference implementation (`copilot-api-py`)
fetches available models from `GET {copilot_base_url}/models` and returns
them from its `/v1/models`. llmproxy is missing this entirely — it never
queries Copilot for its actual model catalog.

The Copilot provider already has everything needed: `base_url()`,
`headers()`, `ensure_token()`, and `send_with_token()`. We need to add
a `fetch_models()` call and wire it into the `/v1/models` handler.

Reference:
- `~/src/copilot-api-py/src/services/copilot/get_models.py` — full impl
- `~/src/copilot-api-py/src/routes/models/route.py` — response format
- `~/src/copilot-api-py/src/lib/utils.py:25-32` — `cache_models()` at startup

## Design

### 1. Add `models_url()` + `fetch_models()` to CopilotProvider

**File:** `src/providers/copilot.rs`

- `models_url() -> String` — returns `{base_url}/models` (next to existing
  `chat_url()` / `responses_url()`)
- `fetch_models(&self) -> Result<Value, ...>` — GETs `models_url()` with
  standard Copilot headers + bearer token (reuse `headers()`), parses JSON

### 2. Cache models on CopilotState

**File:** `src/providers/copilot.rs`

Add to `CopilotState`:
```rust
cached_models: RwLock<Option<Value>>,
```

Add `CopilotProvider::cache_models(&self)` that calls `fetch_models()` and
stores the result. Call it:
- After successful `start_bootstrap()` (token is fresh, models available)
- From the background `refresh_loop` (on each token refresh cycle)
- On first access from `/v1/models` if cache is empty

Add `CopilotProvider::cached_models(&self) -> Option<Value>` accessor.

### 3. Modify `/v1/models` handler

**File:** `src/server.rs` — `list_models_handler`

Logic:
1. Start with static config models (as today)
2. If `state.copilot` is Some, try to get cached models
3. If cache is empty, attempt a one-shot `fetch_models()` + cache (lazy init)
4. Convert Copilot model entries to OpenAI format:
   ```json
   {
     "id": "<copilot model id>",
     "object": "model",
     "created": 0,
     "owned_by": "<vendor>",
     "display_name": "<name>"
   }
   ```
5. Merge: deduplicate by `id`. Copilot-discovered entries take precedence
   over static config entries with the same id. Static entries for
   non-Copilot providers are preserved as-is.

### 4. Fetch on bootstrap + refresh

**File:** `src/providers/copilot.rs`

In `start_bootstrap()` — after `fetch_copilot_token()` succeeds, call
`self.cache_models().await` (best-effort, log warn on failure).

In `spawn_refresh_loop()` — after `refresh_token()` succeeds, call
`self.cache_models().await` (best-effort).

## Critical files

| File | Change |
|------|--------|
| `src/providers/copilot.rs` | Add `models_url()`, `fetch_models()`, `cache_models()`, `cached_models()`. Add `cached_models` field to `CopilotState`. Call `cache_models()` from `start_bootstrap()` and `spawn_refresh_loop()`. |
| `src/server.rs` | Modify `list_models_handler` to merge Copilot-discovered models. |

## Verification

```bash
cargo check --tests
cargo test --lib --tests
```

Manual smoke (requires a bootstrapped Copilot token):
```bash
curl -s http://localhost:9090/v1/models -H "Authorization: Bearer <key>" | jq
```
Should return a model list that includes Copilot-discovered entries with
`owned_by` set to the vendor name (not "llmproxy").

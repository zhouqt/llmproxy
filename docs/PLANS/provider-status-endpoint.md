# Provider status endpoint + Claude Code statusline plan

## Summary

Expose each configured provider's live health state over a new admin
endpoint (`GET /admin/status`) so the Claude Code statusline can render it
in the bottom bar. The proxy's only health signal today is the in-memory
`CooldownCache` (a provider is "cooling down" while a cooldownable failure —
401/402/404/408/429/5xx — is inside its TTL window). The endpoint reports
that state as a stable two-value keyword per provider (`available` /
`cooling_down`), plus the triggering status code, remaining cooldown, kind
label, and served models, so a shell hook can render a line like:

```
🔴 copilot(429/137s)  🟢 deepseek  🔴 openai_direct(503/3s)
```

Two explicit user constraints on the hook:
- **No caching** — the hook queries the API on every statusline refresh.
- **No "available"/"unavailable" words** — the 🔴/🟢 emoji alone identify status.

## Design decisions

### Status keywords: exactly `available` | `cooling_down`

- The endpoint mirrors `CooldownCache::is_cooling_down` / `active()`
  (`src/cooldown.rs:36-42`, `:80-94`) one-to-one, so it can never disagree
  with the router's actual dispatch decisions.
- The proxy does not actively probe upstreams, so `cooling_down` precisely
  means "a cooldownable failure is still inside its TTL window"; it avoids
  over-promising `unavailable`.
- 429 (long cooldown) vs 5xx (short cooldown) differ only via
  `last_error_status` + `cooling_down_remaining_secs`, not a third state.

### Endpoint: `GET /admin/status`

- Mounted on the existing `admin` sub-router (`src/server.rs`), inheriting
  `require_auth`; auto-public when no `server.api_key` is configured
  (`src/auth.rs:15-18`), same as `/admin/copilot/auth`.
- Not on `/v1/*` (that is the client Anthropic API surface) and not public
  like `/health` (which returns only `"ok"` and leaks nothing).
- Always 200 when reachable; answered purely from in-memory state.

### Response schema

Top level: `status: "ok"`, `providers[]` (in config declaration order for
determinism), `summary: { total, available, cooling_down }`.

Per provider:

| Field | Type | Notes |
|---|---|---|
| `name` | string | Provider key from `ProviderConfig::name()`. |
| `type` | string | Serde tag label: `github_copilot` / `anthropic` / `openai_compat` / `openai_responses`. |
| `status` | string | `available` \| `cooling_down`. |
| `models` | string[] | Client models whose chain (`primary`+`fallback`) includes this provider. |
| `last_error_status` | u16 \| null | Triggering upstream status; `null` when available. |
| `cooling_down_remaining_secs` | u64 \| null | Remaining cooldown, whole seconds; `null` when available. |

Worked example:

```json
{
  "status": "ok",
  "providers": [
    { "name": "deepseek", "type": "openai_compat", "status": "available",
      "models": ["claude-sonnet-4.6", "gpt-4o"],
      "last_error_status": null, "cooling_down_remaining_secs": null },
    { "name": "copilot", "type": "github_copilot", "status": "cooling_down",
      "models": ["gpt-5"], "last_error_status": 429, "cooling_down_remaining_secs": 137 },
    { "name": "openai_direct", "type": "openai_responses", "status": "cooling_down",
      "models": ["gpt-5.5"], "last_error_status": 503, "cooling_down_remaining_secs": 3 }
  ],
  "summary": { "total": 3, "available": 1, "cooling_down": 2 }
}
```

## Statusline hook

Claude Code statusline (v2.1.216+): a `statusLine.command` is run on a
refresh interval, its stdout renders as the bottom bar (one line for a
compact single-row display), ANSI and emoji supported, stdin carries session
JSON. The user already has `~/.claude/hooks/statusline.sh` (parent) +
`~/.claude/hooks/minimax-usage.sh` (child-script pattern).

Child script `~/.claude/hooks/provider-status.sh` — **no caching**, query
every time, `--max-time 2` so a slow proxy can't stall the statusline,
silent exit on any failure:

```bash
#!/bin/bash
URL="${LLMPROXY_URL:-http://127.0.0.1:8080}"
json=$(curl -s --max-time 2 -H "Authorization: Bearer ${LLMPROXY_API_KEY:-}" \
  "$URL/admin/status" 2>/dev/null) || exit 0
[ -z "$json" ] && exit 0
echo "$json" | jq -r '
  [ .providers[] |
    if .status == "cooling_down" then
      "🔴 \(.name)(\(.last_error_status)/\(.cooling_down_remaining_secs)s)"
    else
      "🟢 \(.name)"
    end
  ] | join("  ")'
```

The parent appends the segment only when non-empty
(`content="$content|$pstatus"`), matching the existing `usage_text` /
`git_info` pattern, and its existing length clamps keep the bar bounded.

## Files to modify

- `src/config.rs` — add `ProviderConfig::type_label()` (serde-tag string
  per variant) + unit test.
- `src/router.rs` — add `ProviderHealth` enum, `ProviderStatus` struct,
  `Router::provider_status()` async snapshot builder + unit tests.
- `src/server.rs` — add `admin_status_handler` + route `/admin/status`.
- `tests/server.rs` — integration tests: auth gate, all-available shape,
  and 429 → cooling_down state transition through `/v1/messages`.
- `README.md` — document `GET /admin/status`; update the route list and the
  "No metrics endpoint" limitation.

## Out of scope (v2)

Storing the upstream body in `CooldownEntry` (e.g. to surface "rate
limited") deliberately stays out — `src/cooldown.rs:58-63` documents the
existing decision not to cache bodies, and `last_error_status` is enough for
the statusline. Revisit only if a reason string is actually needed.

## Test plan

- `src/config.rs`: `type_label()` returns the serde tag for all four
  variants, and the tag round-trips through `serde_json`.
- `src/router.rs`: with `primary` on a 60s 429 cooldown, `provider_status()`
  reports `primary` → `cooling_down` (status 429, remaining ≈60s) and
  `backup` → `available` (nulls); `models` maps chain members; serialization
  emits `cooling_down`/`available` + `type` keys. Clean cache → all
  `available`.
- `tests/server.rs`: `/admin/status` without a key → 401; with key → 200,
  both providers `available`, summary `{2,2,0}`; after a `/v1/messages`
  request that fails `primary` with 429 and succeeds on `backup`, the status
  endpoint reports `primary` `cooling_down` / status 429 / remaining ≈60s and
  summary `{2,1,1}`.

## Verification

```bash
cargo test --lib --tests
curl -s http://127.0.0.1:8080/admin/status -H "Authorization: Bearer <key>" | jq
bash ~/.claude/hooks/provider-status.sh   # single line: 🔴/🟢 per provider
```

A provider shown 🔴 in the statusline flips back to 🟢 when its cooldown
expires — the endpoint and the router always agree because both read the
same `CooldownCache`.

# llmproxy

A Rust HTTP proxy that lets Claude Code (and any Anthropic-API-compatible
client) talk to multiple LLM providers through a single endpoint, with
automatic fallback when a provider hits its rate limit.

## Features

- **Anthropic-format in, mixed-format out**: accepts `/v1/messages` requests
  in Anthropic format, forwards to providers in their native format
  (OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages),
  translates responses back.
- **Multi-provider fallback**: configure a primary plus a fallback chain per
  model. On 429 / 5xx / 401, the failing provider is cooled down
  (configurable TTL, default 5 min) and the next provider in the chain is
  tried automatically.
- **GitHub Copilot**: full OAuth device flow with token persistence and
  background refresh — no manual token copying. Auto-routes GPT-5.x
  requests to Copilot's `/v1/responses` endpoint (Copilot rejects
  `/v1/chat/completions` for GPT-5.x with `unsupported_api_for_model`).
- **OpenAI Responses API passthrough** via `type: openai_responses` for
  upstreams that expose `/v1/responses` (direct OpenAI GPT-5.x, etc.).
- **OpenRouter, MiniMax, DeepSeek, OpenCode Zen, and any other
  OpenAI-compatible endpoint** supported out of the box.
- **Per-provider proxy opt-in**: each provider declares whether it should
  route through the global SOCKS/HTTP proxy or connect directly. Useful
  when some providers are regionally blocked but others are reachable
  without a proxy. The proxy maintains exactly two shared connection
  pools (proxied + direct) so all `use_proxy: true` providers share one
  pool, and all `use_proxy: false` providers share another.
- **Single shared API key** for clients (optional).
- **Streaming** (SSE) supported end-to-end with tool-use and thinking blocks.

## Quick start

```bash
# 1. Build (or use the prebuilt binary inside the podman container)
cargo build --release

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: set api_key, set provider keys (or use env vars)

# 3. Run
LLMPROXY_API_KEY=sk-local-xxx \
  OPENROUTER_API_KEY=sk-or-xxx \
  DEEPSEEK_API_KEY=sk-ds-xxx \
  MINIMAX_API_KEY=sk-mm-xxx \
  OPENCODE_API_KEY=sk-oc-xxx \
  ./target/release/llmproxy --config config.yaml
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=sk-local-xxx
claude-code "hello"
```

Or test directly:

```bash
curl -N http://127.0.0.1:8080/v1/messages \
  -H "Authorization: Bearer sk-local-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4.6",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

For full configuration field reference see `config.example.yaml`.

## API reference

All routes are mounted under the address set by `server.listen`. Routes
under `/v1/messages` and `/admin/` require the bearer token from
`server.api_key` (when set); `/health` is public.

### `POST /v1/messages`

Anthropic Messages API. Accepts the same request body shape as Claude
Code's own client. The proxy translates it to whatever the chosen
provider's native format requires (Anthropic Messages or OpenAI Chat
Completions), sends it, and translates the response back.

Request body: standard Anthropic Messages shape.

```json
{
  "model": "claude-sonnet-4.6",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": false,
  "system": "Be concise."
}
```

| Field         | Type      | Notes                                       |
|---------------|-----------|---------------------------------------------|
| `model`       | string    | Must match a `name` in the `models:` config block |
| `messages`    | array     | Standard Anthropic message array            |
| `max_tokens`  | integer   | Required by Anthropic's API                 |
| `stream`      | boolean   | `true` for SSE streaming, default `false`   |
| `system`      | string    | Optional system prompt                      |
| `temperature` | number    | Optional                                    |
| `tools`       | array     | Optional tool definitions                   |

**Non-streaming response** (HTTP 200): standard Anthropic Messages JSON.

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Paris."}],
  "model": "claude-sonnet-4.6",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 14, "output_tokens": 4}
}
```

**Streaming response** (HTTP 200, `Content-Type: text/event-stream`):
SSE stream of Anthropic events:

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Paris"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}
```

If the upstream stream errors mid-flight, an extra `event: error` chunk is
emitted before the connection terminates so clients can distinguish a
truncation from a clean end:

```
event: error
data: {"type":"error","error":{"type":"upstream_error","message":"..."}}
```

### `POST /v1/messages/count_tokens`

Rough token estimate for a Messages-shaped request body. Useful for
budgeting before sending. The estimate is `ceil(input_chars / 4)` — fast
but approximate; not suitable for billing.

Request: same body shape as `/v1/messages` (most fields are ignored).

Response:

```json
{"input_tokens": 42}
```

### `GET /v1/models`

Lists models from the `models:` config block, in OpenAI-style envelope.

```json
{
  "object": "list",
  "data": [
    {"id": "claude-sonnet-4.6", "object": "model", "created": 0, "owned_by": "llmproxy"},
    {"id": "claude-haiku-4.6",  "object": "model", "created": 0, "owned_by": "llmproxy"}
  ]
}
```

### `POST /admin/copilot/auth`

Triggers GitHub Copilot OAuth device flow on demand. Useful for first-time
setup or when the cached token has been revoked.

Requires `server.api_key` (admin routes share the same bearer auth as
`/v1/messages`). Returns 404 if no `github_copilot` provider is
configured.

Response (HTTP 200):

```json
{
  "status": "ok",
  "message": "bootstrap started; complete the device flow within the timeout",
  "device_code": "...",
  "user_code": "ABCD-1234",
  "verification_uri": "https://github.com/login/device",
  "expires_in": 899,
  "interval": 5
}
```

Show `user_code` to the user; they visit `verification_uri` and paste it.
The proxy polls for the token in the background and stores it at
`$XDG_DATA_HOME/llmproxy/github_token.json`.

If a bootstrap is already in progress, returns HTTP 409.

### `GET /health`

Liveness probe. No auth required. Returns `200 OK` with body `OK`.

## Response headers

In addition to the headers the response itself carries (e.g.
`content-type`), the proxy adds:

| Header                              | When                                                |
|-------------------------------------|-----------------------------------------------------|
| `x-llmproxy-failed-providers`       | When the response succeeded via a fallback provider (or when the chain was exhausted). Value is a comma-separated list of `name:status` pairs (e.g. `deepseek:401,copilot:404`). |
| `x-accel-buffering: no`             | Streaming responses only — disables proxy buffering (nginx et al.). |
| `cache-control: no-cache`           | Streaming responses only.                          |

## Error envelope

Errors returned to the client use the Anthropic Messages error shape:

```json
{
  "type": "error",
  "error": {
    "type": "api_error | authentication_error | invalid_request_error | upstream_error | ...",
    "message": "human-readable description"
  }
}
```

HTTP status codes:

| Code | Meaning                                                              |
|------|----------------------------------------------------------------------|
| 400  | Malformed request body, missing required field, unsupported model   |
| 401  | Missing or wrong `Authorization: Bearer` / `x-api-key`              |
| 408  | Upstream request timed out (proxy still returning the timeout)      |
| 429  | All providers in the chain are in cooldown (returned after `max_retries_total`) |
| 500  | Internal proxy error                                                 |
| 502  | All providers failed; body contains the last upstream error          |
| 503  | All providers cooling down at request start (no upstream call attempted) |

For 502/503, the response also includes `x-llmproxy-failed-providers` so
clients can see which providers were tried.

## How fallback works

For each `/v1/messages` request the router:

1. Looks up the model config to get `[primary, fallback1, fallback2, ...]`.
2. Filters providers that can't serve the requested model — see
   `can_serve_model` under [Provider types](#provider-types) below.
3. Picks the first provider not in the cooldown cache.
4. Sends the request. On a **cooldownable** status (429, 401, 408, 404, 5xx)
   marks the provider as cooling down for `cooldown_seconds` (default 300s)
   and retries up to `max_retries_per_provider` times, then moves to the
   next provider in the chain.
5. Returns the first successful response. If all providers are exhausted
   or `max_retries_total` is reached, returns the last upstream error with
   `x-llmproxy-failed-providers` listing what was tried.

After the cooldown TTL expires, the provider is retried automatically on
the next request.

## Provider types

The `type` field of each provider entry selects which struct shape the
rest of the entry follows:

### `github_copilot`

Talks to GitHub Copilot's API using OAuth device flow. First request after
a fresh install prints a one-time code (`POST /admin/copilot/auth` does
this on demand). Token persists at
`$XDG_DATA_HOME/llmproxy/github_token.json`.

**GPT-5 routing**: Copilot rejects `/v1/chat/completions` for GPT-5.x
models with `unsupported_api_for_model`. The proxy auto-routes any
incoming model name starting with `gpt-5` to Copilot's `/v1/responses`
endpoint (Responses API). All other names go to `/v1/chat/completions`
as before. No configuration flag is needed — the dispatch happens
per-request.

| Field           | Default      | Notes                                                |
|-----------------|--------------|------------------------------------------------------|
| `name`          | (required)   | Internal key referenced by `models:`                 |
| `vscode_version`| `"1.95.0"`   | User-Agent string. Some Copilot endpoints gate by client version. |
| `account_type`  | `"individual"` | `individual` / `business` / `enterprise`. Determines Copilot SKU. |
| `model_rewrite` | `{}`         | Map of `incoming-model-name → upstream-model-name`   |
| `use_proxy`     | `false`      | Route through the global proxy                       |

### `anthropic`

Native Anthropic Messages passthrough. Forwards `/v1/messages` requests
verbatim to `{api_base}/messages` and streams the response unchanged.
Used for OpenRouter's `/api/v1/messages` endpoint and any other gateway
that speaks Anthropic Messages without translation.

| Field           | Default                          | Notes                                                |
|-----------------|----------------------------------|------------------------------------------------------|
| `name`          | (required)                       | Internal key                                         |
| `api_key`       | (required)                       | Bearer token                                         |
| `api_base`      | `https://openrouter.ai/api/v1`   | Base URL up to but not including `/messages`         |
| `model_rewrite` | `{}`                             | Map of `incoming-model-name → upstream-model-name`   |
| `use_proxy`     | `false`                          |                                                      |

### `openai_compat`

Any OpenAI-Chat-Completions-compatible endpoint (DeepSeek, MiniMax,
OpenCode Zen, etc.).

| Field           | Default   | Notes                                                |
|-----------------|-----------|------------------------------------------------------|
| `name`          | (required)| Internal key                                         |
| `api_key`       | (required)| Bearer token                                         |
| `api_base`      | (required)| Base URL up to but not including `/chat/completions` |
| `model_rewrite` | `{}`      | Map of `incoming-model-name → upstream-model-name`    |
| `use_proxy`     | `false`   |                                                      |

### `openai_responses`

Any backend exposing OpenAI's Responses API (`POST /v1/responses`).
Used for direct OpenAI GPT-5.x access (which requires `/v1/responses`
rather than `/v1/chat/completions`) and any reverse proxy that exposes
the same endpoint.

Conversion is Anthropic Messages ↔ Responses API: `system` →
`instructions`, `messages[]` → flat `input[]` (with `tool_use`/`tool_result`
folded into `function_call` / `function_call_output` items),
`thinking.budget_tokens` → `reasoning.effort`. Streaming is translated
event-by-event (`response.output_text.delta` → `content_block_delta`,
`response.function_call_arguments.delta` → `input_json_delta`, etc.).

| Field           | Default   | Notes                                                |
|-----------------|-----------|------------------------------------------------------|
| `name`          | (required)| Internal key                                         |
| `api_key`       | (required)| Bearer token                                         |
| `api_base`      | (required)| Base URL up to but not including `/responses`, e.g. `https://api.openai.com/v1` |
| `model_rewrite` | `{}`      | Map of `incoming-model-name → upstream-model-name`    |
| `use_proxy`     | `false`   |                                                      |

#### `model_rewrite` and `can_serve_model`

When a provider's `model_rewrite` table is non-empty, the proxy treats it
as that provider's model catalog: only model names present as keys in the
rewrite table are considered servable. The router skips the provider for
any incoming model name that's not in the table — this avoids sending
e.g. `claude-sonnet-4.6` to a DeepSeek-style upstream that doesn't know
that name and would return 400.

When `model_rewrite` is empty, the provider accepts any model name
verbatim (you trust the upstream to resolve it). For OpenCode Zen, the
rewrite table maps Claude-style names to free-tier model IDs so the
provider can serve `claude-haiku-4.6` etc. via `hy3-free`.

## Proxy configuration

The top-level `proxy:` block defines the outgoing proxy used by providers
with `use_proxy: true`. Per-provider opt-in, **defaulting to `false`**, so
by default no provider uses the proxy.

Supported URL schemes:

| Scheme             | Notes                            |
|--------------------|----------------------------------|
| `socks5h://host:p` | SOCKS5, remote DNS resolution    |
| `socks5://host:p`  | SOCKS5, local DNS resolution     |
| `http://host:p`    | HTTP CONNECT                     |
| `https://host:p`   | HTTPS CONNECT                    |

```yaml
proxy:
  url: "socks5h://192.168.122.1:6666"
  timeout_secs: 600
```

The proxy maintains exactly **two** shared `reqwest::Client` instances at
startup: one with the proxy applied and one without. Each provider picks
one based on `use_proxy`. This means all proxy-enabled providers share
one connection pool (and all direct providers share another) — no
per-provider pool fragmentation.

## Architecture

- **axum** HTTP server with Bearer-token middleware
- **reqwest** HTTP client with `socks` feature for outgoing proxy
- **tokio** async runtime + `RwLock` for in-memory state
- **serde** + `serde_yaml` for config; `tracing` for logs
- **No persistence** for cooldown state — restarts clear all cooldowns

## References

- OAuth + API conversion: [`copilot-api-py`](../copilot-api-py/)
- Fallback / cooldown algorithm: [`litellm`](https://github.com/BerriAI/litellm) (`router_utils/cooldown_*.py`)

## Limitations

- No persistent cooldown storage across restarts.
- No metrics endpoint (logs to stderr only).
- Tool-use streaming supported; prompt caching passed through but not optimized.
- `count_tokens` is a rough word-based estimate (4 chars / token), not
  provider-accurate. Don't use it for billing.
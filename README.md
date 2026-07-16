# llmproxy

A Rust HTTP proxy that lets Claude Code (and any Anthropic-API-compatible
client) talk to multiple LLM providers through a single endpoint, with
automatic fallback when a provider hits its rate limit.

## Features

- **Anthropic-format in, mixed-format out**: accepts `/v1/messages` requests
  in Anthropic format, forwards to providers in their native format
  (OpenAI Chat Completions or Anthropic), translates responses back.
- **Multi-provider fallback**: configure a primary plus a fallback chain per
  model. On 429 / 5xx / 401, the failing provider is cooled down
  (configurable TTL, default 5 min) and the next provider in the chain is
  tried automatically.
- **GitHub Copilot**: full OAuth device flow with token persistence and
  background refresh — no manual token copying.
- **OpenRouter, MiniMax, DeepSeek, and any other OpenAI-compatible
  endpoint** supported out of the box.
- **SOCKS5 / HTTP proxy** support for outgoing requests — useful when a
  provider is region-blocked.
- **Single shared API key** for clients (optional).
- **Streaming** (SSE) supported end-to-end with tool-use and thinking blocks.

## Quick start

```bash
# 1. Build
cargo build --release

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: set api_key, set provider keys (or use env vars)

# 3. Run
LLMPROXY_API_KEY=sk-local-xxx \
  OPENROUTER_API_KEY=sk-or-xxx \
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
    "model": "claude-sonnet-4-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

## Endpoints

| Path                          | Notes                                      |
|-------------------------------|--------------------------------------------|
| `POST /v1/messages`           | Anthropic Messages API (streaming + sync)  |
| `POST /v1/messages/count_tokens` | Rough token estimate (~4 chars/token)   |
| `GET  /v1/models`             | Lists configured models                    |
| `GET  /health`                | Liveness probe (no auth)                   |

## How fallback works

For each `/v1/messages` request the router:

1. Looks up the model config to get `[primary, fallback1, fallback2, ...]`.
2. Picks the first provider not in the cooldown cache.
3. Sends the request. On a `cooldownable` status (429, 401, 408, 404, 5xx)
   marks the provider as cooling down for `cooldown_seconds` (default 300s)
   and tries the next provider in the chain.
4. Returns the first successful response. If all providers fail, returns
   the last error along with an `x-llmproxy-failed-providers` header
   listing what was tried.

After the cooldown TTL expires, the provider is retried automatically.

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

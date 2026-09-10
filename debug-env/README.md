# debug-env

Containerised test environment for `llmproxy`. Two operating modes share
one YAML config:

| Compose file | Env | RUST_LOG | Use |
|---|---|---|---|
| `docker-compose.dev.yaml` (unchanged) | `dev.env` | `debug` | Functional testing against the real `:9999` endpoint |
| `docker-compose.perf.yaml` (this dir) | `perf.env` | `warn` | Performance measurement against the mock |

The shared config is `config.dev.yaml`. It contains both the real
`local` / `local-openai` providers (pointing at `:9999`) and four
`mock_*` providers (pointing at `:9000`); each set of entries serves
different goals without interfering with the other.

## What's in here

```
debug-env/
├── config.dev.yaml           # shared config; both modes
├── dev.env                   # RUST_LOG=debug for functional work
├── docker-compose.dev.yaml   # uses localhost/llmproxy_llmproxy:latest
│
├── Dockerfile.debug          # builds localhost/llmproxy:debug
├── docker-compose.perf.yaml  # mock + llmproxy (debug image)
├── perf.env                  # RUST_LOG=warn + MOCK_BASE=http://mock:9000
│
├── mock-upstream/
│   ├── Dockerfile
│   └── mock_server.py        # aiohttp, three dialects, deterministic latency
│
├── bench/
│   ├── bench.py              # latency benchmark (Python stdlib)
│   └── run.sh                # full suite driver
│
├── results/                  # gitignored; per-run JSON
└── README.md                 # this file
```

## How to read the numbers

The proxy overhead is:

```
proxy_overhead = measured_proxy_latency − mock_baseline_latency
```

For this to be a meaningful subtraction the **mock baseline must be
constant**: the mock uses deterministic `asyncio.sleep()` so its
self-reported `sleep_ms` per request is exactly the configured value
(default 80 ms) plus/minus zero jitter. The wall-clock difference
between mock and proxy is your proxy overhead.

Mock-side stats are exposed at `http://127.0.0.1:9000/__mock/stats`:

- `wall_ms_avg`: time from handler entry to last `write()` returning.
- `sleep_ms_avg`: cumulative `asyncio.sleep` time.
- `wall_minus_sleep_avg_ms`: the rest (serialisation + IO). Should be
  close to 0 in baseline runs. If it grows under load, the proxy is
  backpressuring the mock via TCP — that is **proxy-side bottleneck
  evidence, not mock saturation**. Don't discard those samples; tag
  them and use `sleep_ms` as the baseline.

## When to discard a sample / run

The whole measurement is conditional on these checks. If any fail,
fix the underlying problem before trusting the numbers:

1. `direct` baseline `stdev > 5 ms` or `p99 − p50 > 10 ms` → mock is
   not constant. Lower `MOCK_JITTER_MS` (already 0 by default) or
   investigate host load.
2. Any sample's `x-llmproxy-failed-providers` is non-empty → the
   proxy fell back. Those samples are in `results/...` under the
   `fallback_count` field; they don't pollute the primary percentiles
   but they should be near zero in the no-fault-injection scenarios.
3. Mock `wall_ms_avg >> sleep_ms_avg` → genuine backpressure (see
   above). Still usable but annotate.
4. `--concurrency > 1` and `wall_ms_avg` for `direct` baseline has
   grown proportionally → mock is saturating. Drop concurrency first;
   this benchmark is not designed to find the proxy's throughput
   ceiling.

## Quick start (perf mode)

```bash
# Build and start the mock + debug-llmproxy.
podman-compose -f debug-env/docker-compose.perf.yaml up --build

# In another terminal, run the suite.
./debug-env/bench/run.sh

# Or one-off measurements.
python3 debug-env/bench/bench.py \
    --target proxy --model mock-anthropic --mode nonstream --n 100 --out out.json
```

## Quick start (dev mode, real `:9999`)

Unchanged from before — this directory adds `perf` mode without
touching `dev` mode.

```bash
# Whatever you already do.
podman-compose -f debug-env/docker-compose.dev.yaml up
```

`config.dev.yaml` now also contains mock providers, but they're
unused in dev mode because `dev.env` doesn't set `MOCK_BASE` (so
`api_base` falls back to `http://127.0.0.1:9000`, which has no
listener inside the dev container). To use mocks in dev mode too,
add `MOCK_BASE=http://host.containers.internal:9000` to `dev.env` and
run the mock on the host.

## Configuration knobs

### Mock

| Env / control field | Default | Notes |
|---|---|---|
| `MOCK_PORT` | 9000 | listening port |
| `MOCK_LATENCY_MS` | 80 | non-streaming sleep |
| `MOCK_TTFT_MS` | 80 | streaming time-to-first-byte |
| `MOCK_INTER_TOKEN_MS` | 5 | streaming delay between frames |
| `MOCK_N_TOKENS` | 64 | streaming frames per response |
| `MOCK_JITTER_MS` | 0 | Gaussian stdev added to every sleep; keep at 0 for constant-latency baselines |

Per-persona control: `POST /__mock/control` with
`{"persona": "<name>", ...}` to override at runtime.

### Proxy debug image

| Build arg | Default | Notes |
|---|---|---|
| `BUILD_PROFILE` | `release-debug` | `release-debug` keeps symbols and is the recommended profile; `release` matches prod; `dev` is debug build, numbers are not representative of prod |

Image tags:

- `localhost/llmproxy:debug` (default `release-debug`)
- `localhost/llmproxy:debug-dev`
- `localhost/llmproxy:debug-release`

## Secret discipline

- `mock_*` provider entries in `config.dev.yaml` use a literal fake
  string (`mock-key-not-a-secret`). The mock does not check it.
- `LLMPROXY_API_KEY` (the real auth token) is read only from env, never
  embedded in YAML, and never logged.
- `dev.env` / `perf.env` are committed — they contain only the fake
  local test keys (`sk-llmproxy-1234`, `sk-perf-local-only`), never real
  credentials. Other `.env` files in the repo remain gitignored via the
  project root's `*.env` rule; `git add -f` was needed for these two.
- The pre-commit hook (`scripts/scan-secrets.sh`) is the last line of
  defence — keep it installed.

## Per-dialect cost decomposition

Run with all three mock models and subtract:

| Comparison | What it measures |
|---|---|
| `proxy − direct` | Routing + auth + IO forwarding overhead (no conversion) |
| `proxy(mock_chat) − proxy(mock_anthropic)` | Anthropic ↔ Chat Completions conversion cost |
| `proxy(mock_responses) − proxy(mock_anthropic)` | Anthropic ↔ Responses conversion cost |
| `proxy(dev-model)` (real `:9999`) | What proxy overhead looks like next to real-world latency |

The last row is the only one that needs a real upstream — the mock
plus the `dev-model` row can sit side-by-side in `results/` to answer
"is this overhead significant relative to typical LLM latency?".

## See also

- `plans/debug-env-mock-perf.md` — full design doc and rationale.
- `plans/functional-testing.md` — the functional test harness (uses
  `axum` fake providers in-process; complements this debug env).
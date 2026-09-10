#!/usr/bin/env bash
# debug-env/bench/run.sh — run the full latency benchmark suite.
#
# Assumes both the proxy (127.0.0.1:1025) and the mock (127.0.0.1:9000)
# are reachable. Easiest path:
#
#   podman-compose -f debug-env/docker-compose.perf.yaml up -d --build
#   ./debug-env/bench/run.sh
#
# Or, for native (no containers):
#
#   python3 debug-env/mock-upstream/mock_server.py &
#   LLMPROXY_API_KEY=sk-test MOCK_BASE=http://127.0.0.1:9000 \
#       /tmp/llmproxy-target/release/llmproxy \
#       --config debug-env/config.dev.yaml &
#   ./debug-env/bench/run.sh
#
# Results land in debug-env/results/. Each run is a self-describing JSON
# file with full config snapshot for cross-PR comparison.
set -euo pipefail

cd "$(dirname "$0")/.."   # debug-env/
BENCH=./bench/bench.py
RESULTS=./results
mkdir -p "$RESULTS"

N="${BENCH_N:-50}"
WARMUP="${BENCH_WARMUP:-20}"

# Pre-check: mock reachable
if ! curl -fsS -m 2 http://127.0.0.1:9000/health >/dev/null 2>&1; then
    echo "FAIL: mock upstream at http://127.0.0.1:9000 is unreachable." >&2
    echo "  Start it: podman-compose -f debug-env/docker-compose.perf.yaml up -d mock" >&2
    echo "  Or native: python3 debug-env/mock-upstream/mock_server.py &" >&2
    exit 2
fi

# Pre-check: proxy reachable
if ! curl -fsS -m 2 http://127.0.0.1:1025/health >/dev/null 2>&1; then
    echo "FAIL: llmproxy at http://127.0.0.1:1025 is unreachable." >&2
    echo "  Start it: podman-compose -f debug-env/docker-compose.perf.yaml up -d llmproxy" >&2
    exit 2
fi

# Reset mock counters so the run is clean.
curl -fsS -m 2 -X POST http://127.0.0.1:9000/__mock/reset >/dev/null

TS="$(date -u +%Y%m%dT%H%M%SZ)"

run() {
    local target="$1"
    local model="$2"
    local mode="$3"
    local out="$RESULTS/${TS}-${target}-${model}-${mode}.json"
    echo "▶ ${target} / ${model} / ${mode}"
    LLMPROXY_API_KEY="${LLMPROXY_API_KEY:-sk-perf-local-only}" \
        python3 "$BENCH" \
            --target "$target" \
            --model "$model" \
            --mode "$mode" \
            --n "$N" \
            --warmup "$WARMUP" \
            --out "$out" \
        | tail -1
}

# Direct baselines (proxy-free, against mock).
run direct mock-anthropic nonstream
run direct mock-anthropic stream
run direct mock-chat       nonstream
run direct mock-chat       stream
run direct mock-responses  nonstream
run direct mock-responses  stream

# Proxy paths.
run proxy  mock-anthropic nonstream
run proxy  mock-anthropic stream
run proxy  mock-chat       nonstream
run proxy  mock-chat       stream
run proxy  mock-responses  nonstream
run proxy  mock-responses  stream

echo
echo "───────────────────── mock /__mock/stats ─────────────────────"
curl -fsS http://127.0.0.1:9000/__mock/stats | python3 -m json.tool || true

echo
echo "Done. Results in $RESULTS/"
#!/usr/bin/env python3
"""
Latency benchmark for llmproxy + mock-upstream.

Measures end-to-end latency (non-streaming) and TTFT / token-interval
(streaming) for both the proxy path and the direct-to-mock path. The
subtraction `proxy − direct` is the proxy overhead, modulo mock-constant
latency (see plans/debug-env-mock-perf.md §1).

Usage:
    bench.py --mode nonstream --model mock-anthropic --target proxy
    bench.py --mode stream --model mock-chat --target proxy --n 100
    bench.py --mode nonstream --model dev-model --target proxy

By default `--target proxy` hits http://127.0.0.1:1025 (the proxy) and
`--target direct` hits http://127.0.0.1:9000 (the mock). Both are
configurable via --proxy-url / --mock-url. Auth is `Bearer ${LLMPROXY_API_KEY}`
for the proxy path; direct mocks ignore Authorization.

Implementation note: `urllib` does not handle chunked transfer-encoded
streams from aiohttp reliably (raises `http.client.IncompleteRead` on
partial reads even when the server has data buffered). For streaming,
we use `httpx` — already installed in the project's Python env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Any

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Sample + result data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    """One request's measurement."""
    index: int
    ok: bool
    elapsed_ms: float  # full request wall time (non-streaming)
    ttft_ms: float | None  # first-byte time (streaming only)
    body_bytes: int
    status: int
    failed_providers: str  # value of x-llmproxy-failed-providers if any


@dataclass
class Result:
    """Aggregate over one run."""
    tag: str
    config: dict[str, Any] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["samples"] = [asdict(s) for s in self.samples]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Request execution
# ─────────────────────────────────────────────────────────────────────────────


PROXY_DEFAULT = "http://127.0.0.1:1025"
MOCK_DEFAULT = "http://127.0.0.1:9000"


def _anthropic_body(model: str, stream: bool, prompt: str = "ping") -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 50,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
    }


async def _post_blocking(
    url: str,
    body: dict[str, Any],
    bearer: str | None,
    timeout_s: float = 30.0,
) -> tuple[int, bytes, dict[str, str], float]:
    """Non-streaming POST. Returns (status, body, headers, wall_ms)."""
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, json=body, headers=headers)
            payload = r.content
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            status = r.status_code
    except httpx.HTTPStatusError as e:
        payload = e.response.content
        hdrs = {k.lower(): v for k, v in e.response.headers.items()}
        status = e.response.status_code
    except Exception:
        payload = b""
        hdrs = {}
        status = 0
    wall = (time.perf_counter() - t0) * 1000.0
    return status, payload, hdrs, wall


async def _stream_ttft(
    url: str,
    body: dict[str, Any],
    bearer: str | None,
    timeout_s: float = 30.0,
) -> tuple[int, int, dict[str, str], float, float]:
    """Streamed request — return (status, body_bytes, headers, total_ms, ttft_ms).

    Uses httpx with a manual stream so we can record TTFT on the first
    byte chunk before draining. httpx handles chunked transfer-encoding
    correctly where `urllib` does not (urllib raises IncompleteRead on
    partial reads of aiohttp's chunked responses).
    """
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    t0 = time.perf_counter()
    body_bytes = 0
    ttft_ms: float | None = None
    status = 0
    hdrs: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                status = resp.status_code
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                async for chunk in resp.aiter_bytes():
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    body_bytes += len(chunk)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        hdrs = {k.lower(): v for k, v in e.response.headers.items()}
    except Exception:
        status = 0
    if ttft_ms is None:
        ttft_ms = (time.perf_counter() - t0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    return status, body_bytes, hdrs, total_ms, ttft_ms


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "stdev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p90": round(_percentile(values, 90), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────


def _url_for(target: str, model: str, mode: str, proxy_url: str, mock_url: str) -> tuple[str, str | None]:
    """Return (url, bearer_or_none)."""
    bearer = os.environ.get("LLMPROXY_API_KEY") if target == "proxy" else None
    if target == "proxy":
        # All models route through the Anthropic Messages endpoint — the
        # proxy translates internally per the model's primary provider.
        return f"{proxy_url.rstrip('/')}/v1/messages", bearer
    if target == "direct":
        # Direct-to-mock bypasses the proxy. We must send to the dialect-
        # appropriate path because mock's /v1/messages is anthropic-shaped.
        # For mock-* models we mirror what the proxy would have used.
        if model.startswith("mock-anthropic"):
            return f"{mock_url.rstrip('/')}/v1/messages", None
        if model.startswith("mock-chat"):
            return f"{mock_url.rstrip('/')}/v1/chat/completions", None
        if model.startswith("mock-responses"):
            return f"{mock_url.rstrip('/')}/v1/responses", None
        # Real-endpoint models: hit the proxy anyway (no direct path).
        print(f"warning: direct mode requested for {model!r}; defaulting to proxy", file=sys.stderr)
        return f"{proxy_url.rstrip('/')}/v1/messages", bearer
    raise ValueError(f"unknown target: {target}")


def run_one(
    target: str,
    model: str,
    mode: str,
    proxy_url: str,
    mock_url: str,
    n: int,
    warmup: int,
    concurrency: int,
) -> Result:
    url, bearer = _url_for(target, model, mode, proxy_url, mock_url)
    is_stream = mode == "stream"
    body = _anthropic_body(model, is_stream)

    samples: list[Sample] = []
    tag = f"{target}-{model}-{mode}-c{concurrency}-n{n}"

    async def _one(i: int) -> None:
        if is_stream:
            status, body_bytes, headers, total_ms, ttft_ms = await _stream_ttft(url, body, bearer)
        else:
            status, payload, headers, total_ms = await _post_blocking(url, body, bearer)
            body_bytes = len(payload)
            ttft_ms = None
        samples.append(
            Sample(
                index=i,
                ok=200 <= status < 300,
                elapsed_ms=round(total_ms, 3),
                ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
                body_bytes=body_bytes,
                status=status,
                failed_providers=headers.get("x-llmproxy-failed-providers", ""),
            )
        )

    async def _runner() -> None:
        # Warmup (results discarded). Sequential.
        for i in range(warmup):
            await _one(-(i + 1))
        samples.clear()
        # Real samples — concurrency > 1 via gather, else sequential.
        if concurrency <= 1:
            for i in range(n):
                await _one(i)
        else:
            # Batch into `concurrency` chunks to keep it simple (no semaphores).
            sent = 0
            while sent < n:
                chunk = min(concurrency, n - sent)
                await asyncio.gather(*[_one(sent + j) for j in range(chunk)])
                sent += chunk

    asyncio.run(_runner())

    # Stats: keep fallback samples separate so they don't pollute the main
    # percentile calc. A non-empty value means the proxy decided to fall
    # back, so the sample is measuring the fallback path, not the primary.
    primary = [s for s in samples if not s.failed_providers]
    fallback = [s for s in samples if s.failed_providers]

    elapsed_values = [s.elapsed_ms for s in primary]
    ttft_values = [s.ttft_ms for s in primary if s.ttft_ms is not None]

    res = Result(
        tag=tag,
        config={
            "target": target,
            "model": model,
            "mode": mode,
            "n": n,
            "warmup": warmup,
            "concurrency": concurrency,
            "url": url,
        },
        samples=samples,
        stats={
            "primary": _summarize(elapsed_values),
            "ttft": _summarize(ttft_values),
            "fallback_count": len(fallback),
            "error_count": sum(1 for s in samples if not s.ok),
        },
    )
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["nonstream", "stream"], default="nonstream")
    p.add_argument("--target", choices=["proxy", "direct"], default="proxy")
    p.add_argument("--model", default="mock-anthropic")
    p.add_argument("--n", type=int, default=100, help="Number of measured samples (post-warmup)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--proxy-url", default=PROXY_DEFAULT)
    p.add_argument("--mock-url", default=MOCK_DEFAULT)
    p.add_argument("--out", default="-", help="Output JSON path, or - for stdout")
    args = p.parse_args()

    res = run_one(
        target=args.target,
        model=args.model,
        mode=args.mode,
        proxy_url=args.proxy_url,
        mock_url=args.mock_url,
        n=args.n,
        warmup=args.warmup,
        concurrency=args.concurrency,
    )
    payload = json.dumps(res.to_dict(), indent=2)
    if args.out == "-":
        print(payload)
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(payload)
        # Also print a one-line summary.
        s = res.stats["primary"]
        print(
            f"{res.tag}: p50={s.get('p50', 'n/a')} p99={s.get('p99', 'n/a')} "
            f"mean={s.get('mean', 'n/a')} stdev={s.get('stdev', 'n/a')} "
            f"err={res.stats['error_count']} fallback={res.stats['fallback_count']} → {args.out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
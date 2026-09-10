#!/usr/bin/env python3
"""
Mock LLM upstream server.

Three dialect routes + control endpoints. Designed for measuring llmproxy
overhead via the equation:

    measured_latency − mock_fixed_latency = proxy_overhead

Hence deterministic latency is the load-bearing invariant. See
`plans/debug-env-mock-perf.md` §5 for the full design.

Routes (mounted under optional persona prefix; default persona is "" with no prefix):
  POST [/{persona}/]v1/messages             # Anthropic Messages
  POST [/{persona}/]v1/chat/completions     # OpenAI Chat Completions
  POST [/{persona}/]v1/responses            # OpenAI Responses
  GET  [/{persona}/]v1/models               # model catalog
  GET  /__mock/stats                        # per-persona counters + wall_ms/sleep_ms
  POST /__mock/control                      # change runtime behaviour per persona
  POST /__mock/reset                        # clear counters

Environment variables (override defaults):
  MOCK_PORT           default 9000
  MOCK_LATENCY_MS     non-streaming round-trip sleep before writing body (default 80)
  MOCK_TTFT_MS        streaming time-to-first-byte sleep (default 80)
  MOCK_INTER_TOKEN_MS streaming sleep between delta frames (default 5)
  MOCK_N_TOKENS       streaming frames per response (default 64)
  MOCK_JITTER_MS      gaussian jitter added to every sleep (default 0 = constant)

Per-request header overrides (direct-mode bench only; proxy strips unknown headers
on outgoing upstream calls, so these are not effective through the proxy):
  x-mock-latency-ms, x-mock-ttft-ms, x-mock-inter-token-ms,
  x-mock-n-tokens, x-mock-status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

# ─────────────────────────────────────────────────────────────────────────────
# Configuration / persona state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Persona:
    """Runtime-mutable behaviour for one persona (URL prefix)."""

    latency_ms: int = 80
    ttft_ms: int = 80
    inter_token_ms: int = 5
    n_tokens: int = 64
    jitter_ms: float = 0.0
    # If non-zero, every response returns this HTTP status (with empty body).
    status: int = 0
    # If > 0, the next N responses for this persona return `status` then revert.
    fail_first_n: int = 0
    # If > 0 and < 1, each request returns `status` with this probability.
    fail_ratio: float = 0.0
    # If > 0, the streaming response emits an `event: error` frame after this
    # many tokens then finishes normally. 0 = disabled.
    mid_stream_fail_at: int = 0
    # Counters
    requests: int = 0
    wall_ms_total: float = 0.0
    sleep_ms_total: float = 0.0
    requests_by_endpoint: dict[str, int] = field(default_factory=dict)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _make_default_persona() -> Persona:
    return Persona(
        latency_ms=_env_int("MOCK_LATENCY_MS", 80),
        ttft_ms=_env_int("MOCK_TTFT_MS", 80),
        inter_token_ms=_env_int("MOCK_INTER_TOKEN_MS", 5),
        n_tokens=_env_int("MOCK_N_TOKENS", 64),
        jitter_ms=_env_float("MOCK_JITTER_MS", 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SSE pre-render. Anthropic only — OpenAI Chat and Responses use plain JSON
# (non-streaming) or JSON-per-line (streaming) and emit at most ~64 frames,
# which is small enough that inline serialisation cost is dwarfed by the
# 5 ms inter-token sleep. Only the Anthropic framing is dominated by the
# number of frames × serialisation work, hence we cache.
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_MODEL = "mock-anthropic-model"
OPENAI_MODEL = "mock-openai-model"


def _anthropic_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def render_anthropic_sse(text: str, model: str = ANTHROPIC_MODEL) -> bytes:
    """Render a complete Anthropic SSE stream to bytes.

    Sequence: message_start → content_block_start → content_block_delta × N
    → content_block_stop → message_delta → message_stop. EOF terminates.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = max(1, len(text) // 4)

    parts: list[str] = []
    # message_start
    parts.append("event: message_start\n")
    parts.append(
        'data: {"type":"message_start","message":{"id":"'
        + msg_id
        + '","type":"message","role":"assistant","content":[],'
        + '"model":"'
        + model
        + '","stop_reason":null,"stop_sequence":null,"usage":'
        + '{"input_tokens":'
        + str(input_tokens)
        + ',"output_tokens":0}}}\n\n'
    )
    # content_block_start
    parts.append("event: content_block_start\n")
    parts.append(
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    )
    # content_block_delta (one per token)
    chunk_size = max(1, len(text) // max(1, len(text)))  # one chunk per char
    deltas = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    for i, d in enumerate(deltas):
        parts.append("event: content_block_delta\n")
        parts.append(
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"'
            + json.dumps(d)[1:-1]
            + '"}}\n\n'
        )
    # content_block_stop
    parts.append("event: content_block_stop\n")
    parts.append('data: {"type":"content_block_stop","index":0}\n\n')
    # message_delta with usage
    output_tokens = max(1, len(text) // 4)
    parts.append("event: message_delta\n")
    parts.append(
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
        '"stop_sequence":null},"usage":{"output_tokens":'
        + str(output_tokens)
        + '}}\n\n'
    )
    # message_stop
    parts.append("event: message_stop\n")
    parts.append('data: {"type":"message_stop"}\n\n')

    return "".join(parts).encode("utf-8")


def render_anthropic_json(text: str, model: str = ANTHROPIC_MODEL) -> bytes:
    """Render a non-streaming Anthropic Messages response."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    body = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(1, len(text) // 4),
            "output_tokens": max(1, len(text) // 4),
        },
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def render_openai_chat_json(text: str, model: str = OPENAI_MODEL) -> bytes:
    """Render a non-streaming OpenAI Chat Completions response."""
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(text) // 4),
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": max(1, len(text) // 2),
        },
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def render_openai_chat_sse_delta(text: str, n: int, model: str = OPENAI_MODEL) -> str:
    """Render an OpenAI Chat Completions SSE stream body (no [DONE])."""
    chunks: list[str] = []
    char_per_chunk = max(1, len(text) // max(1, n))
    pieces = [text[i : i + char_per_chunk] for i in range(0, len(text), char_per_chunk)]
    for p in pieces[:n]:
        chunk_obj = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}],
        }
        chunks.append("data: " + json.dumps(chunk_obj, separators=(",", ":")) + "\n\n")
    # Final chunk with finish_reason="stop"
    stop_obj = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    chunks.append("data: " + json.dumps(stop_obj, separators=(",", ":")) + "\n\n")
    # Usage chunk (proxy sends stream_options.include_usage=true)
    usage_obj = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "usage": {
            "prompt_tokens": max(1, len(text) // 4),
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": max(1, len(text) // 2),
        },
    }
    chunks.append("data: " + json.dumps(usage_obj, separators=(",", ":")) + "\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def render_responses_json(text: str, model: str = OPENAI_MODEL) -> bytes:
    """Render a non-streaming OpenAI Responses response."""
    body = {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": max(1, len(text) // 4),
            "output_tokens": max(1, len(text) // 4),
            "total_tokens": max(1, len(text) // 2),
        },
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def render_responses_sse(text: str, n: int, model: str = OPENAI_MODEL) -> str:
    """Render an OpenAI Responses SSE stream body."""
    chunks: list[str] = []
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    # response.created
    chunks.append(
        "event: response.created\n"
        + "data: "
        + json.dumps(
            {"type": "response.created", "response": {"id": rid, "model": model, "status": "in_progress"}},
            separators=(",", ":"),
        )
        + "\n\n"
    )
    # response.output_item.added
    chunks.append(
        "event: response.output_item.added\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )
    # response.output_text.delta
    char_per_chunk = max(1, len(text) // max(1, n))
    pieces = [text[i : i + char_per_chunk] for i in range(0, len(text), char_per_chunk)]
    for p in pieces[:n]:
        chunks.append(
            "event: response.output_text.delta\n"
            + "data: "
            + json.dumps(
                {"type": "response.output_text.delta", "item_id": "msg_x", "output_index": 0, "delta": p},
                separators=(",", ":"),
            )
            + "\n\n"
        )
    # response.completed
    chunks.append(
        "event: response.completed\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "model": model,
                    "status": "completed",
                    "usage": {
                        "input_tokens": max(1, len(text) // 4),
                        "output_tokens": max(1, len(text) // 4),
                        "total_tokens": max(1, len(text) // 2),
                    },
                },
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def render_models_list() -> bytes:
    """Render /v1/models response (OpenAI Chat shape — what
    openai_compat and openai_responses both expect via their `models_url`).
    Anthropic-shaped `/v1/models` is a separate GET handled in handler."""
    body = {
        "object": "list",
        "data": [
            {
                "id": ANTHROPIC_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            },
            {
                "id": OPENAI_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            },
        ],
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def render_anthropic_models_list() -> bytes:
    """Anthropic-shaped /v1/models — proxy anthropic provider expects this
    shape (lists available models). Used for the anthropic persona only."""
    body = {
        "data": [
            {"type": "model", "id": ANTHROPIC_MODEL, "display_name": "Mock Anthropic"},
        ],
        "has_more": False,
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Mock state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MockState:
    personas: dict[str, Persona] = field(default_factory=dict)
    pre_rendered: bytes = b""

    def persona_for(self, name: str) -> Persona:
        if name not in self.personas:
            self.personas[name] = _make_default_persona()
        return self.personas[name]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def jittered_sleep(persona: Persona, ms: int) -> float:
    """Sleep `ms` plus persona jitter. Returns actual sleep time in ms.

    The returned value is added to `sleep_ms_total` so /__mock/stats can
    distinguish genuine mock jitter from wall-clock backpressure (see plan §5.6).
    """
    if ms <= 0 and persona.jitter_ms <= 0:
        return 0.0
    target = ms
    if persona.jitter_ms > 0:
        target += random.gauss(0, persona.jitter_ms)
    target = max(0.0, target)
    t0 = time.perf_counter()
    await asyncio.sleep(target / 1000.0)
    return (time.perf_counter() - t0) * 1000.0


PERSONA_PREFIX_RE = re.compile(r"^/p(?P<name>[A-Za-z0-9_-]+)(?P<rest>/.*)$")


def extract_persona(path: str) -> tuple[str, str]:
    """Return (persona_name, remaining_path_with_leading_slash).

    `/v1/messages`            -> ("", "/v1/messages")
    `/p2/v1/messages`         -> ("2", "/v1/messages")
    `/pflaky/v1/messages`     -> ("flaky", "/v1/messages")
    """
    m = PERSONA_PREFIX_RE.match(path)
    if m:
        return m.group("name"), m.group("rest")
    return "", path


def extract_request_text(obj: dict[str, Any]) -> str:
    """Best-effort pull of a textual prompt from an already-parsed JSON body.
    Used to size the response body — longer request ⇒ longer mock response,
    exercising real conversion paths."""
    # Anthropic Messages: messages[].content = str | [{type:"text", text:"..."}]
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c[:512]
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        t = blk.get("text", "")
                        if isinstance(t, str):
                            return t[:512]
        return "Hello from mock."

    # OpenAI Responses: input = str | [{type:"message", content:[...]}]
    inp = obj.get("input")
    if isinstance(inp, str):
        return inp[:512]
    if isinstance(inp, list):
        for it in inp:
            if not isinstance(it, dict):
                continue
            content = it.get("content")
            if isinstance(content, str):
                return content[:512]
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") in ("input_text", "text"):
                        t = blk.get("text", "")
                        if isinstance(t, str):
                            return t[:512]
        return "Hello from mock."

    return "Hello from mock."


async def read_request_body(request: web.Request) -> tuple[str, bool]:
    """Read body once, parse once. Returns (prompt_text, stream_flag).

    Falls back to (default text, False) on any error. The single parse means
    `wants_stream` is just a dict lookup — no second JSON pass.
    """
    try:
        raw = await request.read()
    except Exception:
        return "Hello from mock.", False
    if not raw:
        return "Hello from mock.", False
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return "Hello from mock.", False
    if not isinstance(obj, dict):
        return "Hello from mock.", False
    return extract_request_text(obj), bool(obj.get("stream"))


def wants_stream(stream: bool) -> bool:
    """Pass-through; here for readability."""
    return stream


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _should_fail(persona: Persona) -> int:
    """Return non-zero HTTP status if this request should fail, else 0."""
    if persona.status:
        if persona.fail_first_n > 0:
            persona.fail_first_n -= 1
            return persona.status
        if persona.fail_ratio > 0 and random.random() < persona.fail_ratio:
            return persona.status
        return persona.status
    return 0


async def handle_anthropic_messages(request: web.Request) -> web.StreamResponse:
    state: MockState = app["state"]
    persona_name, _ = extract_persona(request.path)
    persona = state.persona_for(persona_name)
    persona.requests += 1
    persona.requests_by_endpoint["anthropic_messages"] = (
        persona.requests_by_endpoint.get("anthropic_messages", 0) + 1
    )

    t_entry = time.perf_counter()
    sleep_ms = 0.0
    text, is_stream = await read_request_body(request)

    if await _should_fail(persona):
        elapsed = (time.perf_counter() - t_entry) * 1000.0
        persona.wall_ms_total += elapsed
        return web.json_response({"error": "mock injected failure"}, status=persona.status)

    is_stream = wants_stream(is_stream)
    if is_stream:
        # Sleep TTFT, then write pre-rendered SSE, frame-by-frame.
        sleep_ms += await jittered_sleep(persona, persona.ttft_ms)
        body = render_anthropic_sse("Mock reply: " + text)
        resp = web.StreamResponse(
            status=200,
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
        )
        await resp.prepare(request)
        # Write the pre-rendered SSE in chunks; each chunk is one event frame
        # so the inter_token_ms pacing is preserved.
        frames = body.split(b"\n\n")
        for i, frame in enumerate(frames):
            if persona.mid_stream_fail_at > 0 and i == persona.mid_stream_fail_at:
                # Inject Anthropic `event: error` mid-stream, then finish normally
                err_frame = (
                    b"event: error\n"
                    b'data: {"type":"error","error":{"type":"api_error","message":"mock mid-stream failure"}}\n\n'
                )
                await resp.write(err_frame)
                persona.mid_stream_fail_at = 0
                # Skip remaining frames; emit stop
                await resp.write(
                    b"event: message_stop\n"
                    b'data: {"type":"message_stop"}\n\n'
                )
                await resp.write_eof()
                elapsed = (time.perf_counter() - t_entry) * 1000.0
                persona.wall_ms_total += elapsed
                persona.sleep_ms_total += sleep_ms
                return resp
            await resp.write(frame + b"\n\n")
            if i < len(frames) - 1:
                sleep_ms += await jittered_sleep(persona, persona.inter_token_ms)
        await resp.write_eof()
    else:
        sleep_ms += await jittered_sleep(persona, persona.latency_ms)
        body = render_anthropic_json("Mock reply: " + text)
        resp = web.Response(body=body, content_type="application/json")

    elapsed = (time.perf_counter() - t_entry) * 1000.0
    persona.wall_ms_total += elapsed
    persona.sleep_ms_total += sleep_ms
    return resp


async def handle_openai_chat(request: web.Request) -> web.StreamResponse:
    state: MockState = app["state"]
    persona_name, _ = extract_persona(request.path)
    persona = state.persona_for(persona_name)
    persona.requests += 1
    persona.requests_by_endpoint["openai_chat"] = (
        persona.requests_by_endpoint.get("openai_chat", 0) + 1
    )

    t_entry = time.perf_counter()
    sleep_ms = 0.0
    text, is_stream = await read_request_body(request)

    if await _should_fail(persona):
        elapsed = (time.perf_counter() - t_entry) * 1000.0
        persona.wall_ms_total += elapsed
        return web.json_response({"error": "mock injected failure"}, status=persona.status)

    is_stream = wants_stream(is_stream)
    if is_stream:
        sleep_ms += await jittered_sleep(persona, persona.ttft_ms)
        body = render_openai_chat_sse_delta("Mock reply: " + text, persona.n_tokens)
        resp = web.StreamResponse(
            status=200,
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
        )
        await resp.prepare(request)
        frames = body.split("data: ")
        for i, frame in enumerate(frames):
            if not frame:
                continue
            if persona.mid_stream_fail_at > 0 and i == persona.mid_stream_fail_at:
                err_frame = (
                    'data: {"error":{"message":"mock mid-stream failure","type":"api_error"}}\n\n'
                )
                await resp.write(err_frame.encode("utf-8"))
                persona.mid_stream_fail_at = 0
                await resp.write_eof()
                elapsed = (time.perf_counter() - t_entry) * 1000.0
                persona.wall_ms_total += elapsed
                persona.sleep_ms_total += sleep_ms
                return resp
            await resp.write(b"data: " + frame.encode("utf-8"))
            sleep_ms += await jittered_sleep(persona, persona.inter_token_ms)
        await resp.write_eof()
    else:
        sleep_ms += await jittered_sleep(persona, persona.latency_ms)
        body = render_openai_chat_json("Mock reply: " + text)
        resp = web.Response(body=body, content_type="application/json")

    elapsed = (time.perf_counter() - t_entry) * 1000.0
    persona.wall_ms_total += elapsed
    persona.sleep_ms_total += sleep_ms
    return resp


async def handle_openai_responses(request: web.Request) -> web.StreamResponse:
    state: MockState = app["state"]
    persona_name, _ = extract_persona(request.path)
    persona = state.persona_for(persona_name)
    persona.requests += 1
    persona.requests_by_endpoint["openai_responses"] = (
        persona.requests_by_endpoint.get("openai_responses", 0) + 1
    )

    t_entry = time.perf_counter()
    sleep_ms = 0.0
    text, is_stream = await read_request_body(request)

    if await _should_fail(persona):
        elapsed = (time.perf_counter() - t_entry) * 1000.0
        persona.wall_ms_total += elapsed
        return web.json_response({"error": "mock injected failure"}, status=persona.status)

    is_stream = wants_stream(is_stream)
    if is_stream:
        sleep_ms += await jittered_sleep(persona, persona.ttft_ms)
        body = render_responses_sse("Mock reply: " + text, persona.n_tokens)
        resp = web.StreamResponse(
            status=200,
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
        )
        await resp.prepare(request)
        # Pace by event boundaries.
        events = body.split("\n\n")
        for i, ev in enumerate(events):
            if not ev:
                continue
            await resp.write((ev + "\n\n").encode("utf-8"))
            if i < len(events) - 1:
                sleep_ms += await jittered_sleep(persona, persona.inter_token_ms)
        await resp.write_eof()
    else:
        sleep_ms += await jittered_sleep(persona, persona.latency_ms)
        body = render_responses_json("Mock reply: " + text)
        resp = web.Response(body=body, content_type="application/json")

    elapsed = (time.perf_counter() - t_entry) * 1000.0
    persona.wall_ms_total += elapsed
    persona.sleep_ms_total += sleep_ms
    return resp


async def handle_models_anthropic(request: web.Request) -> web.Response:
    state: MockState = app["state"]
    state.persona_for(extract_persona(request.path)[0]).requests_by_endpoint["models"] = (
        state.persona_for(extract_persona(request.path)[0]).requests_by_endpoint.get("models", 0) + 1
    )
    # No sleep — keep /v1/models fast (see plan §12 item 6).
    return web.Response(body=render_anthropic_models_list(), content_type="application/json")


async def handle_models_openai(request: web.Request) -> web.Response:
    state: MockState = app["state"]
    state.persona_for(extract_persona(request.path)[0]).requests_by_endpoint["models"] = (
        state.persona_for(extract_persona(request.path)[0]).requests_by_endpoint.get("models", 0) + 1
    )
    return web.Response(body=render_models_list(), content_type="application/json")


async def handle_stats(request: web.Request) -> web.Response:
    state: MockState = app["state"]
    out: dict[str, Any] = {"personas": {}}
    for name, p in state.personas.items():
        n = max(1, p.requests)
        wall_avg = p.wall_ms_total / n
        sleep_avg = p.sleep_ms_total / n
        out["personas"][name or "<default>"] = {
            "requests": p.requests,
            "wall_ms_total": round(p.wall_ms_total, 2),
            "sleep_ms_total": round(p.sleep_ms_total, 2),
            "wall_ms_avg": round(wall_avg, 3),
            "sleep_ms_avg": round(sleep_avg, 3),
            "wall_minus_sleep_avg_ms": round(wall_avg - sleep_avg, 3),
            "requests_by_endpoint": p.requests_by_endpoint,
            "settings": {
                "latency_ms": p.latency_ms,
                "ttft_ms": p.ttft_ms,
                "inter_token_ms": p.inter_token_ms,
                "n_tokens": p.n_tokens,
                "jitter_ms": p.jitter_ms,
                "status": p.status,
                "fail_first_n": p.fail_first_n,
                "fail_ratio": p.fail_ratio,
                "mid_stream_fail_at": p.mid_stream_fail_at,
            },
        }
    return web.json_response(out)


async def handle_control(request: web.Request) -> web.Response:
    state: MockState = app["state"]
    body = await request.json()
    persona_name = str(body.get("persona", ""))
    p = state.persona_for(persona_name)
    for k in (
        "latency_ms",
        "ttft_ms",
        "inter_token_ms",
        "n_tokens",
        "status",
        "fail_first_n",
        "mid_stream_fail_at",
    ):
        if k in body:
            setattr(p, k, int(body[k]))
    if "jitter_ms" in body:
        p.jitter_ms = float(body["jitter_ms"])
    if "fail_ratio" in body:
        p.fail_ratio = float(body["fail_ratio"])
    return web.json_response(
        {
            "ok": True,
            "persona": persona_name,
            "settings": {
                "latency_ms": p.latency_ms,
                "ttft_ms": p.ttft_ms,
                "inter_token_ms": p.inter_token_ms,
                "n_tokens": p.n_tokens,
                "jitter_ms": p.jitter_ms,
                "status": p.status,
                "fail_first_n": p.fail_first_n,
                "fail_ratio": p.fail_ratio,
                "mid_stream_fail_at": p.mid_stream_fail_at,
            },
        }
    )


async def handle_reset(request: web.Request) -> web.Response:
    state: MockState = app["state"]
    state.personas.clear()
    return web.json_response({"ok": True})


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


# ─────────────────────────────────────────────────────────────────────────────
# App factory + main
# ─────────────────────────────────────────────────────────────────────────────

app: web.Application  # populated in main(); handlers reference via closure


async def _on_startup(app: web.Application) -> None:
    app["state"] = MockState()


def build_app() -> web.Application:
    a = web.Application(client_max_size=16 * 1024 * 1024)
    a["state"] = MockState()
    a.on_startup.append(_on_startup)

    # Anthropic Messages (with and without /v1)
    a.router.add_post("/v1/messages", handle_anthropic_messages)
    a.router.add_post("/messages", handle_anthropic_messages)
    # OpenAI Chat Completions (with and without /v1)
    a.router.add_post("/v1/chat/completions", handle_openai_chat)
    a.router.add_post("/chat/completions", handle_openai_chat)
    # OpenAI Responses (with and without /v1)
    a.router.add_post("/v1/responses", handle_openai_responses)
    a.router.add_post("/responses", handle_openai_responses)
    # /v1/models — different shapes per dialect
    a.router.add_get("/v1/models", handle_models_anthropic)
    a.router.add_get("/models", handle_models_openai)

    # Control endpoints
    a.router.add_get("/__mock/stats", handle_stats)
    a.router.add_post("/__mock/control", handle_control)
    a.router.add_post("/__mock/reset", handle_reset)
    a.router.add_get("/health", handle_health)

    # Persona-prefixed routes — handled by the same handlers; they parse
    # the prefix out of request.path. We register /p<name>/<rest> for any
    # common rest the bench script might hit. aiohttp's router matches
    # longest first, so we register exact patterns.
    for prefix in ("p1", "p2", "pflaky", "pslow"):
        a.router.add_post(f"/{prefix}/v1/messages", handle_anthropic_messages)
        a.router.add_post(f"/{prefix}/v1/chat/completions", handle_openai_chat)
        a.router.add_post(f"/{prefix}/v1/responses", handle_openai_responses)
        a.router.add_get(f"/{prefix}/v1/models", handle_models_anthropic)
        a.router.add_post(f"/{prefix}/messages", handle_anthropic_messages)
        a.router.add_post(f"/{prefix}/chat/completions", handle_openai_chat)
        a.router.add_post(f"/{prefix}/responses", handle_openai_responses)
        a.router.add_get(f"/{prefix}/models", handle_models_openai)

    return a


def main() -> None:
    global app
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=_env_int("MOCK_PORT", 9000))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app()
    web.run_app(app, host=args.host, port=args.port, print=lambda *a: None)


if __name__ == "__main__":
    main()
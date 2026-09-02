#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture a profile containing only steady-state decode iterations.

The regular serving benchmark starts the profiler before submitting requests,
which mixes chunked prefill and decode in one trace.  This client keeps a batch
of streaming completion requests resident, waits until every request has
produced its first token, and only then brackets a fixed number of decode
iterations with the vLLM profiling endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


GENERATION_TOKENS_RE = re.compile(
    r"^vllm(?::|_)generation_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
)


@dataclass
class StreamState:
    first_token: asyncio.Event
    chunks: int = 0
    tokens: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--capture-steps",
        type=int,
        default=32,
        help="Minimum streamed decode chunks captured for every request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Keep requests alive long enough to reach the profiling barrier.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input_len <= 0 or args.batch_size <= 0:
        parser.error("--input-len and --batch-size must be positive")
    if args.capture_steps <= 0 or args.max_tokens <= args.capture_steps:
        parser.error("--max-tokens must be greater than --capture-steps > 0")
    return args


def make_prompt(input_len: int, request_idx: int) -> list[int]:
    """Return exact-length, valid token IDs with request-unique prefixes."""
    # Qwen3.8 has a much larger vocabulary.  These ordinary low token IDs keep
    # JSON construction cheap; request-specific values also prevent accidental
    # cross-request prefix reuse if a server is launched with caching enabled.
    token_id = 1000 + request_idx
    return [token_id] * input_len


async def post_profile(session: aiohttp.ClientSession, base_url: str, action: str):
    async with session.post(f"{base_url}/{action}") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{action} failed ({response.status}): {body}")


async def generation_tokens(
    session: aiohttp.ClientSession, base_url: str
) -> float | None:
    async with session.get(f"{base_url}/metrics") as response:
        if response.status != 200:
            return None
        total = 0.0
        found = False
        for line in (await response.text()).splitlines():
            match = GENERATION_TOKENS_RE.match(line)
            if match:
                total += float(match.group(1))
                found = True
        return total if found else None


async def consume_stream(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    input_len: int,
    request_idx: int,
    max_tokens: int,
    seed: int,
    state: StreamState,
):
    payload = {
        "model": model,
        "prompt": make_prompt(input_len, request_idx),
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "seed": seed + request_idx,
    }
    buffer = b""
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"request {request_idx} failed ({response.status}): "
                    f"{await response.text()}"
                )
            async for data in response.content.iter_any():
                buffer += data
                while b"\n\n" in buffer:
                    raw, buffer = buffer.split(b"\n\n", 1)
                    raw = raw.strip()
                    if not raw or raw.startswith(b":"):
                        continue
                    if raw.startswith(b"data:"):
                        raw = raw[5:].strip()
                    if raw == b"[DONE]":
                        return
                    chunk = json.loads(raw)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    token_ids = choice.get("token_ids") or []
                    # Each non-terminal choice is one engine output.  MTP can
                    # accept one to four token IDs in that single iteration.
                    if token_ids or choice.get("finish_reason") is None:
                        state.chunks += 1
                        state.tokens += len(token_ids)
                        state.first_token.set()
    except asyncio.CancelledError:
        raise
    except Exception as error:  # Propagate through the shared state/barrier.
        state.error = str(error)
        state.first_token.set()


async def wait_for_steps(
    states: list[StreamState], baselines: list[int], capture_steps: int
):
    while True:
        errors = [state.error for state in states if state.error]
        if errors:
            raise RuntimeError(errors[0])
        if all(
            state.chunks - baseline >= capture_steps
            for state, baseline in zip(states, baselines, strict=True)
        ):
            return
        await asyncio.sleep(0.001)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    connector = aiohttp.TCPConnector(limit=0)
    states = [StreamState(first_token=asyncio.Event()) for _ in range(args.batch_size)]
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                consume_stream(
                    session,
                    args.base_url,
                    args.model,
                    args.input_len,
                    request_idx,
                    args.max_tokens,
                    args.seed,
                    states[request_idx],
                )
            )
            for request_idx in range(args.batch_size)
        ]
        profiler_started = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*(state.first_token.wait() for state in states)),
                timeout=args.timeout_sec,
            )
            errors = [state.error for state in states if state.error]
            if errors:
                raise RuntimeError(errors[0])

            metric_start = await generation_tokens(session, args.base_url)
            await post_profile(session, args.base_url, "start_profile")
            profiler_started = True
            # Snapshot after the profiling acknowledgement so the requested
            # number of iterations is fully contained in the capture window.
            baselines = [state.chunks for state in states]
            token_baselines = [state.tokens for state in states]
            start = time.perf_counter()

            await asyncio.wait_for(
                wait_for_steps(states, baselines, args.capture_steps),
                timeout=args.timeout_sec,
            )

            end = time.perf_counter()
            await post_profile(session, args.base_url, "stop_profile")
            profiler_started = False
            metric_end = await generation_tokens(session, args.base_url)

            chunk_deltas = [
                state.chunks - baseline
                for state, baseline in zip(states, baselines, strict=True)
            ]
            token_deltas = [
                state.tokens - baseline
                for state, baseline in zip(states, token_baselines, strict=True)
            ]
            return {
                "base_url": args.base_url,
                "model": args.model,
                "input_len": args.input_len,
                "batch_size": args.batch_size,
                "capture_steps": args.capture_steps,
                "wall_time_sec": end - start,
                "stream_chunk_delta": {
                    "min": min(chunk_deltas),
                    "max": max(chunk_deltas),
                    "sum": sum(chunk_deltas),
                },
                "stream_token_delta": {
                    "min": min(token_deltas),
                    "max": max(token_deltas),
                    "sum": sum(token_deltas),
                },
                "metric_generation_token_delta": (
                    metric_end - metric_start
                    if metric_start is not None and metric_end is not None
                    else None
                ),
                "stream_tokens_per_sec": sum(token_deltas) / (end - start),
            }
        finally:
            if profiler_started:
                await post_profile(session, args.base_url, "stop_profile")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    args = parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

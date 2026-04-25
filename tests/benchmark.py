"""
tests/benchmark.py

Benchmarks for the inference engine.
Measures: latency, throughput (tokens/sec), batch performance.

Run: python -m tests.benchmark
"""

from __future__ import annotations

import asyncio
import time
import statistics
from typing import Callable, Awaitable

from rich.console import Console
from rich.table import Table

from engine.inference import InferenceEngine, InferenceConfig


console = Console()

BENCH_MESSAGES = [
    [{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
    [{"role": "user", "content": "Name the capital of France. One word only."}],
    [{"role": "user", "content": "What color is the sky? One word."}],
    [{"role": "user", "content": "How many days in a week? Answer with a number."}],
    [{"role": "user", "content": "What is the largest planet in the solar system? One word."}],
]


async def bench_latency(engine: InferenceEngine, n: int = 5) -> dict:
    """Measure single-request latency over N runs."""
    latencies = []
    msg = [{"role": "user", "content": "Reply with exactly: 'pong'"}]

    for _ in range(n):
        result = await engine.infer(msg)
        latencies.append(result.latency_ms)
        await asyncio.sleep(0.1)  # avoid rate limiting

    return {
        "name": "Single-request latency",
        "n": n,
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(n * 0.95)], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


async def bench_streaming_ttft(engine: InferenceEngine, n: int = 3) -> dict:
    """Measure time-to-first-token for streaming requests."""
    ttfts = []
    msg = [{"role": "user", "content": "Count from 1 to 10, one number per line."}]

    for _ in range(n):
        start = time.monotonic()
        first_token = None
        async for chunk in engine.stream(msg):
            if not chunk.is_final and chunk.delta and first_token is None:
                first_token = (time.monotonic() - start) * 1000
        if first_token:
            ttfts.append(first_token)
        await asyncio.sleep(0.2)

    return {
        "name": "Streaming time-to-first-token",
        "n": n,
        "p50_ms": round(statistics.median(ttfts), 1) if ttfts else 0,
        "min_ms": round(min(ttfts), 1) if ttfts else 0,
        "max_ms": round(max(ttfts), 1) if ttfts else 0,
    }


async def bench_batch(engine: InferenceEngine) -> dict:
    """Measure concurrent batch throughput."""
    start = time.monotonic()
    results = await engine.batch(BENCH_MESSAGES)
    elapsed_ms = (time.monotonic() - start) * 1000

    total_tokens = sum(r.output_tokens for r in results)
    tokens_per_sec = (total_tokens / elapsed_ms) * 1000

    return {
        "name": "Batch throughput",
        "n": len(BENCH_MESSAGES),
        "elapsed_ms": round(elapsed_ms, 1),
        "total_output_tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def _print_results(results: list[dict]):
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Benchmark", style="bold")
    table.add_column("N", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_column("Extras", justify="left")

    for r in results:
        extras = ""
        if "tokens_per_sec" in r:
            extras = f"{r['tokens_per_sec']} tok/s"
        if "elapsed_ms" in r and "tokens_per_sec" not in r:
            extras = f"total {r['elapsed_ms']}ms"

        table.add_row(
            r["name"],
            str(r.get("n", "")),
            str(r.get("p50_ms", "—")),
            str(r.get("p95_ms", "—")),
            str(r.get("min_ms", "—")),
            str(r.get("max_ms", "—")),
            extras,
        )

    console.print(table)


async def main():
    console.print("[bold]Running inference engine benchmarks...[/bold]\n")
    engine = InferenceEngine(InferenceConfig(max_tokens=128))

    results = []

    with console.status("Latency benchmark..."):
        results.append(await bench_latency(engine, n=5))

    with console.status("Streaming TTFT benchmark..."):
        results.append(await bench_streaming_ttft(engine, n=3))

    with console.status("Batch throughput benchmark..."):
        results.append(await bench_batch(engine))

    _print_results(results)
    console.print(f"\n[dim]Engine stats: {engine.stats()}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

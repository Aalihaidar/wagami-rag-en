#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Latency check: send a few questions to a running app and report how long each reply takes.

Per turn it prints the time to the first visible word and the total time, measured from this
client (so it includes the network). Optionally, --log summarises the server's own per-stage
breakdown ("chat turn timings" log lines) so a slow turn can be traced to a stage.

Each question uses a fresh session so history length doesn't skew results, and turns are spaced
by --gap seconds: the per-IP rate limit is 10 requests a minute, and the Groq free tier's
token-per-minute cap allows only about three turns a minute -- going faster measures a 429
backoff, not the app.

Usage:
    uv run python scripts/latency_check.py --base-url http://localhost:8000
    uv run python scripts/latency_check.py --runs 6 --gap 25 --log /path/to/server.log
    uv run python scripts/latency_check.py --no-stream      # time the plain /chat endpoint
"""

import argparse
import json
import math
import statistics
import sys
import time

import httpx

QUESTIONS = [
    "how much is the double espresso?",
    "is the chicken katsu gluten free?",
    "what desserts do you have?",
    "which starters are vegan?",
    "how many calories are in the vegan ramen?",
    "do you take bookings?",
    "what drinks have no alcohol?",
    "tell me about the gyoza",
]
SLOW_TURN_MS = 8000  # past this, a stage (usually an LLM retry wait) is almost certainly stalling


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile -- with only a handful of runs this is the max for p95."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def time_stream_turn(client: httpx.Client, base_url: str, message: str) -> dict:
    """POST to /chat/stream; returns {status, first_ms, total_ms, chars} (times in ms)."""
    session_id = client.post(f"{base_url}/session").json()["session_id"]
    started = time.perf_counter()
    first_ms: float | None = None
    chars = 0
    with client.stream(
        "POST", f"{base_url}/chat/stream", json={"session_id": session_id, "message": message}
    ) as response:
        if response.status_code != 200:
            response.read()
            return {"status": response.status_code, "detail": response.text[:120]}
        event = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event in ("delta", "done", "error"):
                now = (time.perf_counter() - started) * 1000
                if event == "delta" and first_ms is None:
                    first_ms = now
                if event == "done":
                    chars = len(json.loads(line[5:])["answer"])
                    first_ms = first_ms if first_ms is not None else now
                if event == "error":
                    return {"status": "error", "detail": line[5:].strip()[:120]}
    total_ms = (time.perf_counter() - started) * 1000
    return {"status": 200, "first_ms": first_ms, "total_ms": total_ms, "chars": chars}


def time_plain_turn(client: httpx.Client, base_url: str, message: str) -> dict:
    """POST to /chat (one JSON response); the first word and the last arrive together."""
    session_id = client.post(f"{base_url}/session").json()["session_id"]
    started = time.perf_counter()
    response = client.post(f"{base_url}/chat", json={"session_id": session_id, "message": message})
    total_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        return {"status": response.status_code, "detail": response.text[:120]}
    return {
        "status": 200,
        "first_ms": total_ms,
        "total_ms": total_ms,
        "chars": len(response.json()["answer"]),
    }


def summarise_server_log(path: str, last: int) -> None:
    """Median of each stage over the last `last` "chat turn timings" lines of a server log."""
    rows: list[dict[str, float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"chat turn timings"' not in line:
                continue
            try:
                rows.append(json.loads(line)["timings_ms"])
            except ValueError, KeyError:
                continue
    rows = rows[-last:]
    if not rows:
        print(f"\nNo 'chat turn timings' lines found in {path}.")
        return
    print(f"\nServer-side stages, median over the last {len(rows)} logged turns (ms):")
    stages = sorted({stage for row in rows for stage in row})
    for stage in stages:
        values = [row[stage] for row in rows if stage in row]
        note = "" if len(values) == len(rows) else f"   (in {len(values)}/{len(rows)} turns)"
        print(f"  {stage:<13} {statistics.median(values):8.0f}{note}")
    print(
        "  (rerank includes rerank_pace; understand/generate include llm_backoff; "
        "first_delta is the time to the first word)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=6, help="number of questions to send")
    parser.add_argument("--gap", type=float, default=20.0, help="seconds between turns")
    parser.add_argument(
        "--no-stream", action="store_true", help="time /chat instead of /chat/stream"
    )
    parser.add_argument("--log", help="server log file to summarise per-stage timings from")
    args = parser.parse_args()

    turn = time_plain_turn if args.no_stream else time_stream_turn
    endpoint = "/chat" if args.no_stream else "/chat/stream"
    print(f"{endpoint} on {args.base_url}: {args.runs} turns, {args.gap:.0f}s apart\n")
    print(f"{'#':>2}  {'first word':>10}  {'total':>8}  {'chars':>5}  question")

    firsts: list[float] = []
    totals: list[float] = []
    with httpx.Client(timeout=90) as client:
        for i in range(args.runs):
            question = QUESTIONS[i % len(QUESTIONS)]
            try:
                result = turn(client, args.base_url, question)
            except httpx.HTTPError as exc:
                print(f"{i + 1:>2}  request failed: {exc!r}")
                return 1
            if result["status"] != 200:
                print(
                    f"{i + 1:>2}  HTTP {result['status']}: {result.get('detail', '')}  ({question})"
                )
            else:
                firsts.append(result["first_ms"])
                totals.append(result["total_ms"])
                flag = "  <-- slow" if result["total_ms"] > SLOW_TURN_MS else ""
                print(
                    f"{i + 1:>2}  {result['first_ms']:>8.0f}ms  {result['total_ms']:>6.0f}ms  "
                    f"{result['chars']:>5}  {question}{flag}"
                )
            if i < args.runs - 1:
                time.sleep(args.gap)

    if not totals:
        print("\nNo successful turns to summarise.")
        return 1
    print(f"\n{len(totals)} successful turns")
    for label, values in (("first word", firsts), ("total", totals)):
        print(
            f"  {label:<10} median {statistics.median(values):6.0f} ms   "
            f"p95 {percentile(values, 95):6.0f} ms   max {max(values):6.0f} ms"
        )
    if args.log:
        summarise_server_log(args.log, len(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())

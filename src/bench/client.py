"""Load client: fixed-concurrency + Poisson arrivals against an OpenAI-
compatible /v1/completions server. Records per-request TTFT/TPOT/status.
Pure client (no GPU); all math unit-tested."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(int(q * len(s)), len(s) - 1)
    return s[i]


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in recs if r.get("status") == 200]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    tpot = [r["tpot_s"] for r in ok if r.get("tpot_s") is not None]
    dur = max((r.get("end_t", 0) for r in recs), default=0) - \
        min((r.get("start_t", 0) for r in recs), default=0)
    return {"n": len(recs), "ok": len(ok),
            "err_rate": round(1 - len(ok) / max(len(recs), 1), 4),
            "ttft_p50": round(percentile(ttft, 0.5), 3),
            "ttft_p95": round(percentile(ttft, 0.95), 3),
            "tpot_p50": round(percentile(tpot, 0.5), 3),
            "tpot_p95": round(percentile(tpot, 0.95), 3),
            "req_per_s": round(len(ok) / dur, 2) if dur > 0 else 0.0}


async def one_request(session: Any, url: str, model: str, prompt: str,
                      max_tokens: int = 64) -> dict[str, Any]:
    rec: dict[str, Any] = {"start_t": time.monotonic(), "prompt_sha":
                           hashlib.sha256(prompt.encode()).hexdigest()[:16]}
    t_first: float | None = None
    n_tok = 0
    try:
        async with session.post(url, json={"model": model, "prompt": prompt,
                                           "max_tokens": max_tokens,
                                           "temperature": 0.0,
                                           "stream": True}) as resp:
            rec["status"] = resp.status
            async for line in resp.content:
                if t_first is None:
                    t_first = time.monotonic()
                n_tok += 1
    except (OSError, asyncio.TimeoutError) as e:
        rec["status"] = -1
        rec["error"] = type(e).__name__
    end = time.monotonic()
    rec["end_t"] = end
    if t_first is not None and n_tok > 1:
        rec["ttft_s"] = round(t_first - rec["start_t"], 3)
        rec["tpot_s"] = round((end - t_first) / (n_tok - 1), 4)
    return rec


async def run_fixed(url: str, model: str, prompts: list[str],
                    concurrency: int, max_tokens: int = 64) -> list[dict[str, object]]:
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    out: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        async def w2(p: str) -> None:
            async with sem:
                out.append(await one_request(session, url, model, p, max_tokens))
        await asyncio.gather(*[w2(p) for p in prompts])
    return out


async def run_poisson(url: str, model: str, prompts: list[str], rate: float,
                      seconds: float, max_tokens: int = 64) -> list[dict[str, object]]:
    import random

    import aiohttp

    out: list[dict[str, Any]] = []
    t0 = time.monotonic()

    async def worker(p: str) -> None:
        async with aiohttp.ClientSession() as s:
            out.append(await one_request(s, url, model, p, max_tokens))

    tasks = []
    i = 0
    while time.monotonic() - t0 < seconds:
        tasks.append(asyncio.create_task(worker(prompts[i % len(prompts)])))
        i += 1
        await asyncio.sleep(random.expovariate(rate))
    await asyncio.gather(*tasks)
    return out


def save_jsonl(recs: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)

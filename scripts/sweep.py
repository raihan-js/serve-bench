"""Sweep driver: fixed-concurrency + Poisson loads + quality probe.
Usage: sweep.py TAG (server must already run on :8000). Writes
results/{tag}_fixed.jsonl, {tag}_poisson.jsonl, {tag}_quality.json,
and updates results/summary.json.
"""
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bench.client import run_fixed, run_poisson, save_jsonl, summarize  # noqa: E402

URL = "http://localhost:8000/v1/completions"
MODEL = "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4"


async def quality(url: str, model: str, prompts: list[dict]) -> list[dict]:
    import aiohttp

    out = []
    async with aiohttp.ClientSession() as s:
        for p in prompts:
            body = {"model": model, "prompt": p["text"], "max_tokens": 64,
                    "temperature": 0.0, "stream": False}
            t0 = time.monotonic()
            try:
                async with s.post(url, json=body) as r:
                    d = await r.json()
                    txt = d["choices"][0]["text"]
                    out.append({"id": p["id"], "lang": p["lang"], "status": r.status,
                                "out_sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
                                "secs": round(time.monotonic() - t0, 2)})
            except Exception as e:
                out.append({"id": p["id"], "status": -1, "error": type(e).__name__})
    return out


async def main(tag: str) -> None:
    prompts = [p["text"] for p in
               json.load(open("data/prompts.json", encoding="utf-8"))]
    qprompts = json.load(open("data/prompts.json", encoding="utf-8"))
    t0 = time.time()
    recs = await run_fixed(URL, MODEL, (prompts * 4)[:100], concurrency=8)
    save_jsonl(recs, f"results/{tag}_fixed_c8.jsonl")
    print("fixed c8:", json.dumps(summarize(recs)), flush=True)
    recs = await run_fixed(URL, MODEL, (prompts * 4)[:100], concurrency=32)
    save_jsonl(recs, f"results/{tag}_fixed_c32.jsonl")
    print("fixed c32:", json.dumps(summarize(recs)), flush=True)
    recs = await run_poisson(URL, MODEL, prompts, rate=4.0, seconds=60)
    save_jsonl(recs, f"results/{tag}_poisson.jsonl")
    print("poisson:", json.dumps(summarize(recs)), flush=True)
    q = await quality(URL, MODEL, qprompts)
    Path(f"results/{tag}_quality.json").write_text(json.dumps(q, indent=1))
    print(f"quality: {sum(1 for r in q if r.get('status') == 200)}/{len(q)} ok",
          flush=True)
    print(f"sweep done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "base"))

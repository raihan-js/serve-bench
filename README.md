# serve-bench — single-GPU vLLM serving benchmark with quality probes

What does each serving knob cost in latency, throughput, AND output quality?
Client-only harness (asyncio+aiohttp) against any OpenAI-compatible server;
the server runs from sibling venvs (disk discipline: this venv is client-only).

Plan: `docs/PLAN.md`. Decisions: `docs/decisions.md` (append-only).

## Results (RTX 3060 12GB, Swallow-8B-AWQ ctx2048 + llm-jp-1.8B bf16)

### max-num-seqs 32 vs 128: 19x TTFT gap — visible only under stress

| load | seqs=32 TTFT p50 | seqs=128 TTFT p50 |
|------|-----------------|-------------------|
| fixed c8/c32, poisson λ4 | identical (noise) | identical |
| stress c64 × 256 out | **8.29s** | **0.43s** |

TPOT flat throughout (16→28ms): textbook prefill queueing collapse. Gentle
benchmarks cannot tell server configs apart — the stress arm is mandatory.

### Small model wins big: llm-jp 1.8B vs Swallow 8B (same stress)

TTFT 119ms vs 432ms (3.6x), 28.5 vs 15.1 req/s at c32. Serves fine under
vLLM (own kernels dodge the transformers RoPE bug this model has).

### temp-0 determinism fails on reasoning models (verified)

Swallow-AWQ back-to-back identical requests, same server: 10/50 outputs
differ (12/50 across restarts). llm-jp ×3: identical. Think-mode sampling
leaks through temperature 0 — a production assumption that fails per-model,
caught only because every config carries a quality probe (50 frozen prompts,
output-hash drift).

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q   # fake SSE server, no GPU
# live sweep (needs a vLLM server on :8000):
.venv/bin/python scripts/sweep.py TAG [MODEL]
.venv/bin/python scripts/quality.py TAG [MODEL]
```

## Limits
- One GPU, two models, short prompts; no autoscaling, no multi-engine.
- Drift hashes are exact-match (paraphrase-equivalent outputs count as drift).
- Poisson arrivals are synthetic, not production traces.

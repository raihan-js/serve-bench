# serve-bench — design spec (approved 2026-10-14, wave three #1)

Single-GPU vLLM serving benchmark with quality probes. Question: what does
each serving knob cost in latency, throughput, AND output quality?

Owner decisions: vLLM-native (no multi-engine); on-disk corpora only;
quality probe mandatory per config; CI = client tests + lint (no GPU).

## Load (src/bench/client.py)
- Targets an OpenAI-compatible server (`/v1/completions`).
- Patterns: fixed-concurrency (C clients, N requests) + Poisson arrivals
  (rate λ/s, time-boxed). Asyncio + aiohttp.
- Prompts: Wikipedia slice + Tatoeba JA (on disk, EN+JA mix), fixed seeds.

## Sweep (scripts/sweep.py)
- Models: Swallow-8B-AWQ (ctx 2048, util 0.75 — M0 numbers) + llm-jp-1.8B
  bf16 (eager attention). Server runs from sibling venvs (kotoba/sae);
  bench venv is client-only (disk discipline).
- Knobs: max-num-seqs {16,64,128}, gpu-util {0.75,0.9}, AWQ vs bf16 where it
  fits. One knob at a time; baseline first.

## Metrics (per run JSONL -> results/summary.json)
- TTFT/TPOT p50/p95, req/s, error rate, prompt hash.
- Quality probe: 50 frozen prompts, output sha256 drift vs baseline +
  wiki-5k perplexity slice. No LLM judge, ever.

## Non-goals
Multi-engine, autoscaling, tensor-parallel, real-user traffic.

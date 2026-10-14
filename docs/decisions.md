# serve-bench — decisions (append-only, newest first)

## 2026-10-14 — baseline (Swallow-AWQ, seqs=128): no sweat at c32
- c8: TTFT 86/353ms, 5.9 r/s; c32: 224/297ms, 14.9 r/s; poisson λ4: all
  green; quality 50/50. Server never stressed — next: constrain seqs=32,
  then llm-jp bf16, then util 0.9.

## 2026-10-14 — gentle load can't tell configs apart; stress can
- seqs=128 vs 32 at c8/c32/poisson: identical (TTFT within noise). Server
  never saturates under gentle load — added stress arm (c64, 256 out).
- Stress on seqs=32: TTFT p50 8.3s / p95 8.8s (vs 0.2s gentle), TPOT flat
  16ms. Textbook prefill queueing collapse. Next: same stress at seqs=128.

## 2026-10-14 — max-num-seqs 128 vs 32 under stress: 19x TTFT gap
- Identical stress (c64, 256 out): seqs=32 TTFT p50 8.3s, seqs=128 0.43s.
  Gentle loads identical. The knob only matters past saturation — the
  harness earns its keep: config comparisons REQUIRE a stress arm.
- Next: llm-jp-1.8B bf16 (will vLLM's own kernels take it?) + drift check.

## 2026-10-14 — drift probe works: reasoning model unstable at temp 0
- Same model+weights (Swallow seqs128 vs seqs32): 20/50 quality outputs
  differ. llm-jp same-server x3: identical. Suspect: Qwen3-RL think traces
  leak nondeterminism across batches/restarts (sweep sends raw prompts,
  no /no_think). Verifying with back-to-back same-server Swallow runs.
- llm-jp-1.8B bf16 serves fine under vLLM (own kernels dodge the
  transformers RoPE bug): stress TTFT 119ms vs Swallow 432ms, 28.5 r/s.

## 2026-10-14 — temp-0 determinism fails on reasoning models (verified)
- Swallow-AWQ back-to-back identical requests, same server: 10/50 outputs
  differ; vs earlier run 12/50. llm-jp x3: identical. Think-mode sampling
  leaks through temp 0 — a production assumption (deterministic serving)
  fails per-model. The drift probe caught what latency never would.
- Sweep table complete: seqs knob 19x under stress, invisible when gentle;
  llm-jp 3.6x faster than Swallow under stress (119 vs 432ms TTFT).

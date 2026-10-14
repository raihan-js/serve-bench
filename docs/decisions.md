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

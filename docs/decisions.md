# serve-bench — decisions (append-only, newest first)

## 2026-10-14 — baseline (Swallow-AWQ, seqs=128): no sweat at c32
- c8: TTFT 86/353ms, 5.9 r/s; c32: 224/297ms, 14.9 r/s; poisson λ4: all
  green; quality 50/50. Server never stressed — next: constrain seqs=32,
  then llm-jp bf16, then util 0.9.

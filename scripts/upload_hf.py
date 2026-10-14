"""Upload release artifacts to HF dataset raihan-js/serve-bench-eval.

Contents: per-config summary, frozen quality-probe hashes, prompt sets.
No model weights, no server logs. Small, public, no licence encumbrance.
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "serve-bench-eval"
FILES = [
    "results/summary.json",
    "results/base_quality.json",
    "results/seqs32_quality.json",
    "results/seqs128_quality.json",
    "results/llmjp_quality.json",
    "results/verify1_quality.json",
    "results/verify2_quality.json",
    "data/prompts.json",
]

CARD = """---
license: apache-2.0
---
# serve-bench-eval

Results for [serve-bench](https://github.com/raihan-js/serve-bench)
(single-GPU vLLM serving benchmark with quality probes; RTX 3060).

- `summary.json`: TTFT/TPOT p50/p95, req/s, error rates per config and load.
- `*_quality.json`: per-prompt output hashes (drift detection).
- `prompts.json`: the 50 frozen EN+JA probe prompts.

Headlines: max-num-seqs 32 vs 128 = 19x TTFT gap under stress (invisible when
gentle); llm-jp-1.8B 3.6x faster than Swallow-8B; temp-0 determinism fails on
the reasoning model (10/50 back-to-back diffs, verified same-server).
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    for f in FILES:
        api.upload_file(path_or_fileobj=str(root / f), path_in_repo=f,
                        repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    print(f"uploaded {len(FILES)} files + card to raihan-js/{REPO}")


if __name__ == "__main__":
    main()

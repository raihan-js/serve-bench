"""Quality probe only (for drift checks): 50 frozen prompts, greedy.
Usage: quality.py TAG [MODEL]. Writes results/{tag}_quality.json.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import quality


async def main(tag: str, model: str) -> None:
    import json

    def load() -> object:
        with open("data/prompts.json", encoding="utf-8") as f:
            return json.load(f)

    qprompts = await asyncio.to_thread(load)
    q = await quality("http://localhost:8000/v1/completions", model, qprompts)
    Path(f"results/{tag}_quality.json").write_text(json.dumps(q, indent=1))
    print(f"quality: {sum(1 for r in q if r.get('status') == 200)}/{len(q)} ok")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1],
                     sys.argv[2] if len(sys.argv) > 2 else
                     "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4"))

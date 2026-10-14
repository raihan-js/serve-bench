"""Frozen prompt sets (CPU): 50 quality-probe prompts (EN+JA mix, fixed
order) + wiki perplexity slice (5k tokens). All from on-disk corpora.
Writes data/prompts.json + data/ppl_slice.txt. Seeds fixed.
"""
import json
import random
from pathlib import Path


def main() -> None:
    rng = random.Random(0)
    kb = Path("/home/raihan/Desktop/APPS/research-learn/kotoba-tutor")
    tatoeba = [json.loads(line)["learner"] for line in
               open(kb / "results/m1_rows.jsonl", encoding="utf-8")]
    ja = rng.sample([t for t in tatoeba if any("ぁ" <= c <= "ヶ" for c in t)], 25)
    wiki_ja, wiki_en = [], []
    for line in open(kb / "data/corpus.txt", encoding="utf-8"):
        (wiki_ja if any("ぁ" <= c <= "ヶ" for c in line) else wiki_en).append(line.strip())
    en = rng.sample([t for t in wiki_en if 40 <= len(t) <= 300], 25)
    prompts = [{"id": f"q{i:02d}", "lang": "ja" if i < 25 else "en",
                "text": t} for i, t in enumerate(ja + en)]
    Path("data").mkdir(exist_ok=True)
    Path("data/prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=1) + "\n")
    ppl = "\n".join(rng.sample([t for t in wiki_en if 100 <= len(t) <= 500], 60))
    Path("data/ppl_slice.txt").write_text(ppl + "\n")
    print("prompts:", len(prompts), "ppl chars:", len(ppl))


if __name__ == "__main__":
    main()

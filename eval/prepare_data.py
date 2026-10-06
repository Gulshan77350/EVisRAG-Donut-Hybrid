"""Pick a fixed random subset of an EVisRAG test set and extract only the page images it needs.

Data: https://huggingface.co/datasets/openbmb/EVisRAG-Test-<Dataset> (top3_test.jsonl + images.parquet),
i.e. exactly the questions and top-3 retrieved pages used in Table 1 of the EVisRAG paper.

    python eval/prepare_data.py --dataset DocVQA --n 200
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent / "data" / "EVisRAG-Test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=200, help="subset size (0 = all questions)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = ROOT / args.dataset
    items = [json.loads(l) for l in open(d / "top3_test.jsonl", encoding="utf-8")]
    if args.n and args.n < len(items):
        items = random.Random(args.seed).sample(items, args.n)
    with open(d / f"subset_{len(items)}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    needed = {p for it in items for p in it["image"]}
    (d / "imgs").mkdir(exist_ok=True)
    pf = pq.ParquetFile(d / "images.parquet")
    written = 0
    for batch in pf.iter_batches(batch_size=64, columns=["image", "path"]):
        for row in batch.to_pylist():
            if row["path"] in needed and not (d / row["path"]).exists():
                (d / row["path"]).write_bytes(row["image"]["bytes"])
                written += 1
    missing = [p for p in needed if not (d / p).exists()]
    print(f"{args.dataset}: {len(items)} questions, {sum(not it['is_sufficient'] for it in items)} insufficient, "
          f"{len(needed)} pages ({written} extracted), missing {len(missing)}")


if __name__ == "__main__":
    main()

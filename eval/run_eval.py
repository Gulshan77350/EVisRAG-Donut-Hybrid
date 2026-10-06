"""Run one system on a prepared EVisRAG test subset and write one JSON line per question (resumable).

Systems
  hybrid : Donut reads every page, EVisRAG-3B (4-bit) reasons over text   (this thesis)
  image  : original EVisRAG-3B (4-bit) reading the page images itself    (baseline)

"Donut only" needs no run: it is scored from the readings saved by the hybrid run (eval/score.py).
Run `image` in its own process - with the vision tower on the GPU there is no room for Donut on 6 GB.

    python eval/run_eval.py --dataset DocVQA --system hybrid
    python eval/run_eval.py --dataset DocVQA --system image
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline import EVisRAGReasoner, HybridPipeline  # noqa: E402

DATA = ROOT / "data" / "EVisRAG-Test"
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--system", required=True, choices=["hybrid", "image"])
    ap.add_argument("--subset", default=None, help="subset file (default: the only subset_*.jsonl)")
    ap.add_argument("--size", default="3B", choices=["3B", "7B"])
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28, help="image baseline: per-page pixel cap")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many questions (0 = all)")
    args = ap.parse_args()

    d = DATA / args.dataset
    subset = Path(args.subset) if args.subset else next(iter(sorted(d.glob("subset_*.jsonl"))))
    items = [json.loads(l) for l in open(subset, encoding="utf-8")]
    out_path = RESULTS / args.dataset / f"{args.system}_{args.size}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [it for it in items if it["qid"] not in done][: args.limit or None]
    print(f"{args.dataset}/{args.system}: {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    t = time.perf_counter()
    if args.system == "hybrid":
        pipe = HybridPipeline(size=args.size)
    else:
        model = EVisRAGReasoner(size=args.size, load_vision=True, max_pixels=args.max_pixels)
    print(f"loaded in {time.perf_counter() - t:.0f}s", flush=True)

    with open(out_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo, 1):
            images = [Image.open(d / p).convert("RGB") for p in it["image"]]
            torch.cuda.reset_peak_memory_stats()
            rec = {"qid": it["qid"], "query": it["query"], "gold": it["answer"],
                   "is_sufficient": it["is_sufficient"], "labels": it.get("labels")}
            try:
                if args.system == "hybrid":
                    res = pipe.run(it["query"], images, threshold=args.threshold)
                    a = res.readings  # [page][question] -> DonutAnswer; one question per page here
                    rec.update(pred=res.answer, raw=res.reasoner.raw,
                               readings=[{"answer": p[0].answer, "conf": p[0].confidence,
                                          "alts": p[0].alternatives, "seconds": p[0].seconds} for p in a],
                               timings=res.timings, prompt_tokens=res.reasoner.prompt_tokens,
                               new_tokens=res.reasoner.new_tokens)
                else:
                    t0 = time.perf_counter()
                    out = model.reason_with_images(it["query"], images)
                    rec.update(pred=out.answer, raw=out.raw, timings={"total": time.perf_counter() - t0},
                               prompt_tokens=out.prompt_tokens, new_tokens=out.new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec.update(pred="", raw="", error="cuda_oom", timings={})
            rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {rec['timings'].get('total', 0):.1f}s pred={rec['pred']!r} gold={it['answer']}",
                  flush=True)


if __name__ == "__main__":
    main()

"""Score result files with the official EVisRAG metric and print paper-ready tables.

Metric (ported from OpenBMB/VisRAG src/evisrag/eval.py): after SQuAD normalisation a prediction is
correct when every token of some gold answer appears in it ("acc"); for questions whose top-3 pages do
not contain the answer (is_sufficient = false) the gold answers become the abstention strings.

    python eval/score.py                 # every dataset under results/
    python eval/score.py --threshold 0.9 # Donut-only threshold
"""
import argparse
import json
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ABSTAIN = ["no relevant information", "insufficient to answer", "insufficient to answer the question"]


# ---- official metric (verbatim logic) ----
def normalize_answer_qa(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.strip().split())


def evaluate(pred, golds):
    p = normalize_answer_qa(pred)
    acc = em = f1 = 0
    for g in golds:
        g = normalize_answer_qa(g)
        common = sum((Counter(p.split()) & Counter(g.split())).values())
        if not common:  # official code `continue`s here, so em/acc are not updated either
            continue
        em = max(em, int(p == g))
        acc = max(acc, int(set(g.split()).issubset(set(p.split()))))
        pr, rc = common / len(p.split()), common / len(g.split())
        f1 = max(f1, 2 * pr * rc / (pr + rc + 1e-7))
    return {"acc": acc, "em": em, "f1": f1}


def gold_of(rec):
    return rec["gold"] if rec["is_sufficient"] else ABSTAIN


# ---- Donut-only system, derived from the hybrid run's readings ----
def donut_only_pred(rec, threshold):
    best = max(rec["readings"], key=lambda r: r["conf"])
    return best["answer"] if best["conf"] >= threshold and best["answer"] else "insufficient to answer"


def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def auroc(pos, neg):
    """P(score of a relevant page > score of an irrelevant page), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def summarize(name, recs, preds):
    rows = [evaluate(p, gold_of(r)) for r, p in zip(recs, preds)]
    acc = [x["acc"] for x in rows]
    suff = [x["acc"] for x, r in zip(rows, recs) if r["is_sufficient"]]
    insuff = [x["acc"] for x, r in zip(rows, recs) if not r["is_sufficient"]]
    lo, hi = bootstrap_ci(acc)
    times = [r["timings"]["total"] for r in recs if r.get("timings", {}).get("total")]
    mem = [r["peak_gpu_gb"] for r in recs if "peak_gpu_gb" in r]
    return {
        "system": name, "n": len(recs),
        "acc": 100 * sum(acc) / len(acc), "ci": (100 * lo, 100 * hi),
        "acc_suff": 100 * sum(suff) / len(suff) if suff else float("nan"),
        "abstain_ok": 100 * sum(insuff) / len(insuff) if insuff else float("nan"),
        "f1": 100 * statistics.mean(x["f1"] for x in rows),
        "median_s": statistics.median(times) if times else float("nan"),
        "peak_gb": max(mem) if mem else float("nan"),
        "errors": sum("error" in r for r in recs),
    }


def load(path):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(path, encoding="utf-8")} if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.95, help="Donut-only abstention threshold")
    ap.add_argument("--size", default="3B")
    ap.add_argument("--common", action="store_true", help="score only questions every system has finished")
    args = ap.parse_args()

    print("| Dataset | System | n | Acc % (95% CI) | Acc answerable % | Correct abstention % | F1 % | Median s/q | Peak GPU GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    conf_rows = []
    for d in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        hyb, img = load(d / f"hybrid_{args.size}.jsonl"), load(d / f"image_{args.size}.jsonl")
        qids = set(hyb) | set(img)
        if args.common:
            qids = {q for q in qids if q in hyb and (q in img or not img)}
        systems = []
        if hyb:
            recs = [hyb[q] for q in sorted(qids) if q in hyb]
            systems.append(summarize("Donut only", recs, [donut_only_pred(r, args.threshold) for r in recs]))
            systems.append(summarize(f"Donut + EVisRAG-{args.size} (ours)", recs, [r["pred"] for r in recs]))
            pos = [rd["conf"] for r in recs if r.get("labels") for rd, l in zip(r["readings"], r["labels"]) if l]
            neg = [rd["conf"] for r in recs if r.get("labels") for rd, l in zip(r["readings"], r["labels"]) if not l]
            if pos and neg:
                conf_rows.append((d.name, len(pos), len(neg), auroc(pos, neg),
                                  100 * sum(c >= args.threshold for c in pos) / len(pos),
                                  100 * sum(c < args.threshold for c in neg) / len(neg)))
        if img:
            recs = [img[q] for q in sorted(qids) if q in img]
            systems.append(summarize(f"EVisRAG-{args.size} on images", recs, [r["pred"] for r in recs]))
        for s in systems:
            print(f"| {d.name} | {s['system']} | {s['n']} | {s['acc']:.1f} ({s['ci'][0]:.1f}–{s['ci'][1]:.1f}) | "
                  f"{s['acc_suff']:.1f} | {s['abstain_ok']:.1f} | {s['f1']:.1f} | {s['median_s']:.1f} | "
                  f"{s['peak_gb']:.2f} |" + (f" errors={s['errors']}" if s["errors"] else ""))

    if conf_rows:
        print(f"\nDonut first-token confidence as a page-relevance detector (threshold {args.threshold}):\n")
        print("| Dataset | relevant pages | irrelevant pages | AUROC | relevant kept % | irrelevant dropped % |")
        print("|---|---|---|---|---|---|")
        for name, npos, nneg, a, kept, dropped in conf_rows:
            print(f"| {name} | {npos} | {nneg} | {a:.3f} | {kept:.1f} | {dropped:.1f} |")


if __name__ == "__main__":
    main()

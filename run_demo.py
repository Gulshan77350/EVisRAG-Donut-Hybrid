"""Command-line demo of the hybrid Donut -> EVisRAG pipeline.

    python run_demo.py --question "What is the value labeled TOTAL on the receipt?" \
        --images samples/receipt.png samples/filler.png
"""
import argparse
import json

from PIL import Image

from pipeline import HybridPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--size", default="3B", choices=["3B", "7B"])
    ap.add_argument("--threshold", type=float, default=0.95, help="Donut confidence below this -> no relevant information")
    ap.add_argument("--no-decompose", action="store_true", help="skip EVisRAG question planning")
    ap.add_argument("--json", help="write the full result to this file")
    args = ap.parse_args()

    pipe = HybridPipeline(size=args.size)
    print(f"[models loaded in {pipe.load_seconds:.1f}s]")
    images = [Image.open(p) for p in args.images]
    res = pipe.run(args.question, images, threshold=args.threshold, decompose=not args.no_decompose)

    print("\nQUESTION:", res.question)
    if res.sub_questions:
        print("PLANNED LOOKUPS:", *res.sub_questions, sep="\n  - ")
    print("\n--- Stage 1: Donut readings ---\n" + res.notes)
    print("\n--- Stage 2: EVisRAG (text only) ---\n" + res.reasoner.raw)
    print("\nFINAL ANSWER:", res.answer)
    print("TIMINGS (s):", {k: round(v, 2) for k, v in res.timings.items()},
          f"| EVisRAG prompt tokens: {res.reasoner.prompt_tokens}, generated: {res.reasoner.new_tokens}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"question": res.question, "sub_questions": res.sub_questions, "answer": res.answer,
                       "evidence": res.evidence, "reasoner_output": res.reasoner.raw, "timings": res.timings,
                       "prompt_tokens": res.reasoner.prompt_tokens, "new_tokens": res.reasoner.new_tokens,
                       "readings": [[a.__dict__ for a in page] for page in res.readings]}, f, indent=2)


if __name__ == "__main__":
    main()

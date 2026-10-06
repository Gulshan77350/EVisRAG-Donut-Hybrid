"""Original EVisRAG (the model reads the page IMAGES itself) - the slow baseline.

Run it as a separate process from the hybrid demo: with the vision tower on the
GPU there is no room left for Donut on a 6 GB card.

    python run_baseline.py --question "..." --images samples/receipt.png samples/filler.png
"""
import argparse
import json
import time

from PIL import Image

from pipeline import EVisRAGReasoner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--size", default="3B", choices=["3B", "7B"])
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28,
                    help="per-image resolution cap (visual tokens ~ pixels/784); paper used up to 1.5M")
    ap.add_argument("--json")
    args = ap.parse_args()

    t = time.perf_counter()
    model = EVisRAGReasoner(size=args.size, load_vision=True, max_pixels=args.max_pixels)
    print(f"[model loaded in {time.perf_counter() - t:.1f}s]")
    out = model.reason_with_images(args.question, [Image.open(p) for p in args.images])
    print(out.raw)
    print("\nFINAL ANSWER:", out.answer)
    print(f"TIME: {out.seconds:.2f}s | prompt tokens (incl. image tokens): {out.prompt_tokens}, generated: {out.new_tokens}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"question": args.question, "answer": out.answer, "raw": out.raw, "seconds": out.seconds,
                       "prompt_tokens": out.prompt_tokens, "new_tokens": out.new_tokens}, f, indent=2)


if __name__ == "__main__":
    main()

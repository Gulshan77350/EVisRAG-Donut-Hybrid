"""Gradio web demo of the hybrid pipeline.

    env\python.exe app.py            # EVisRAG-3B (4-bit), default for the demo
    env\python.exe app.py --size 7B  # larger reasoner (slow on a 6 GB GPU)
Then open http://127.0.0.1:7860
"""
import argparse
from pathlib import Path

from pipeline import HybridPipeline  # first: sets HF_HOME (models on D:) before gradio loads huggingface_hub

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
# [question, pages, let EVisRAG plan lookups?]
EXAMPLES = [
    ["What is the value labeled TOTAL on the receipt?", ["receipt.png", "filler.png"], False],
    ["Who is the Chief Executive Officer of the company?", ["receipt.png", "report.png", "filler.png"], False],
    ["What was the revenue of Asia-Pacific in USD million?", ["filler.png", "report.png", "receipt.png"], False],
    ["If the receipt TOTAL is split equally between 2 people, how much does each person pay?", ["filler.png", "receipt.png"], False],
    ["How much more is the SUBTOTAL than the TOTAL on the receipt?", ["filler.png", "receipt.png"], True],
    ["What is the capital of Australia?", ["receipt.png", "filler.png"], False],
]


def build(pipe: HybridPipeline):
    def run(files, question, threshold, decompose):
        if not files or not question.strip():
            raise gr.Error("Upload at least one page image and enter a question.")
        images = [Image.open(f if isinstance(f, str) else f.name) for f in files]
        res = pipe.run(question.strip(), images, threshold=threshold, decompose=decompose)

        rows = []
        for i, page in enumerate(res.readings, 1):
            for a in page:
                alts = ", ".join(f"{t} ({c:.2f})" for t, c in a.alternatives[:2])
                used = "✅ evidence" if a.confidence >= threshold and a.answer else "⛔ dropped"
                rows.append([f"Page {i}", a.question, a.answer, round(a.confidence, 3), alts, used, round(a.seconds, 2)])
        t = res.timings
        timing_md = (f"| Stage | Seconds |\n|---|---|\n"
                     f"| 0. EVisRAG plans lookups (text) | {t['plan']:.2f} |\n"
                     f"| 1. Donut reads {len(images)} page(s) | {t['donut_read']:.2f} |\n"
                     f"| 2. EVisRAG reasons (text only) | {t['evisrag_reason']:.2f} |\n"
                     f"| **Total** | **{t['total']:.2f}** |\n\n"
                     f"EVisRAG input: **{res.reasoner.prompt_tokens} text tokens** (no image tokens), "
                     f"generated {res.reasoner.new_tokens} tokens.")
        plan = "\n".join(f"- {s}" for s in res.sub_questions) or "_(planning off or question already simple)_"
        gallery = [(im, f"Page {i}") for i, im in enumerate(images, 1)]
        return f"## ✅ Answer: {res.answer}", rows, plan, res.evidence, res.reasoner.raw, timing_md, gallery

    with gr.Blocks(title="Donut → EVisRAG hybrid") as demo:
        gr.Markdown("# Donut reads, EVisRAG thinks\n"
                    "Donut (small, OCR-free) answers the question on each page separately. "
                    "Low-confidence guesses become *no relevant information*. EVisRAG then reasons over "
                    "that short text list only — it never looks at an image.")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Document page images", file_count="multiple", file_types=["image"])
                question = gr.Textbox(label="Question", lines=2)
                threshold = gr.Slider(0, 1, value=0.95, step=0.01, label="Donut confidence threshold (first-token prob.)")
                decompose = gr.Checkbox(value=False, label="Let EVisRAG plan lookups (for questions needing several values, e.g. a difference)")
                btn = gr.Button("Run", variant="primary")
                gr.Examples([[q, [str(SAMPLES / f) for f in fs], d] for q, fs, d in EXAMPLES],
                            inputs=[question, files, decompose], label="Examples")
                pages = gr.Gallery(label="Pages (Page 1, 2, ... as numbered in the evidence)", columns=3, height=260)
            with gr.Column(scale=2):
                answer = gr.Markdown(label="Final answer")
                timing = gr.Markdown()
                plan = gr.Markdown(label="Planned lookups")
                table = gr.Dataframe(headers=["Page", "Lookup question", "Donut answer", "Confidence",
                                              "Alternatives", "Used?", "Sec"], label="Stage 1 · Donut readings",
                                     wrap=True)
                evidence = gr.Textbox(label="Evidence passed to EVisRAG", lines=4)
                raw = gr.Textbox(label="Stage 2 · EVisRAG output (observe/evidence prefilled, think/answer generated)",
                                 lines=12)
        btn.click(run, [files, question, threshold, decompose], [answer, table, plan, evidence, raw, timing, pages])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="3B", choices=["3B", "7B"])
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    pipe = HybridPipeline(size=args.size)
    print(f"models loaded in {pipe.load_seconds:.1f}s")
    build(pipe).launch(share=args.share)

"""Smoke test of the demo questions through the hybrid pipeline (demo defaults)."""
import sys
import time

import torch
from PIL import Image

from pipeline import HybridPipeline

size = sys.argv[1] if len(sys.argv) > 1 else "3B"
# (question, pages, expected, decompose)
CASES = [
    ("What is the value labeled TOTAL on the receipt?", ["receipt", "filler"], "174,600", False),
    ("Who is the Chief Executive Officer of the company?", ["receipt", "report", "filler"], "Maria Lindqvist", False),
    ("What was the revenue of Asia-Pacific in USD million?", ["filler", "report", "receipt"], "536.9", False),
    ("If the receipt TOTAL is split equally between 2 people, how much does each person pay?", ["filler", "receipt"], "87,300", True),
    ("What is the capital of Australia?", ["receipt", "filler"], "insufficient to answer", False),
]

t = time.perf_counter()
pipe = HybridPipeline(size=size)
print(f"LOADED {size} in {time.perf_counter() - t:.1f}s | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

for q, pages, gold, decompose in CASES:
    res = pipe.run(q, [Image.open(f"samples/{p}.png") for p in pages], decompose=decompose)
    print(f"\nQ: {q}  | pages={pages} | expected={gold}")
    print("lookups:", res.sub_questions)
    print(res.notes)
    print("--- EVisRAG ---\n" + res.reasoner.raw)
    print(f">>> ANSWER: {res.answer!r} (expected {gold})  timings={ {k: round(v, 1) for k, v in res.timings.items()} } "
          f"tokens in/out={res.reasoner.prompt_tokens}/{res.reasoner.new_tokens}", flush=True)
print(f"\nPEAK GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

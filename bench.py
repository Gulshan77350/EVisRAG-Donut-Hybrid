"""Diagnose EVisRAG-3B text generation speed (tokens/s) under different loading options."""
import os
import sys
import time

import pipeline  # noqa: F401  (sets HF_HOME / offline)
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

NAME = "openbmb/EVisRAG-3B"
proc = AutoProcessor.from_pretrained(NAME)
msgs = [{"role": "user", "content": [{"type": "text", "text": "Explain step by step how a receipt total is computed from the subtotal and a 10% discount."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def rate(model, label, **gen):
    inputs = proc(text=[text], return_tensors="pt").to("cuda:0")
    model.generate(**inputs, max_new_tokens=5, do_sample=False)  # warm-up
    torch.cuda.synchronize(); t = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=80, min_new_tokens=80, do_sample=False, **gen)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    n = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"{label:45s} {n / dt:6.2f} tok/s  GPU used {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)


mode = sys.argv[1]
if mode == "nf4":
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           llm_int8_skip_modules=["visual", "lm_head"])
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(NAME, quantization_config=q, device_map="cuda:0", torch_dtype=torch.float16)
    rate(m, "nf4, all on GPU")
    for i in range(7):
        rate(m, f"nf4 run {i+2}, repetition_penalty", repetition_penalty=1.05)
elif mode == "fp16":
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(NAME, device_map={"visual": "cpu", "model": 0, "lm_head": 0}, torch_dtype=torch.float16)
    rate(m, "fp16, vision on CPU")

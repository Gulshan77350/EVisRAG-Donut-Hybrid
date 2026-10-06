# Donut reads, EVisRAG thinks — hybrid visual RAG on a 6 GB laptop GPU

Thesis prototype. EVisRAG ([paper](https://arxiv.org/abs/2510.09733), [code](https://github.com/OpenBMB/VisRAG))
is strong at multi-page reasoning but reading page images with a 7B VLM is slow on a
laptop GPU. Donut ([code](https://github.com/clovaai/donut)) reads a page quickly without OCR but
works on one page at a time and never says "I don't know".

**Idea:** Donut does all the reading, EVisRAG only does the thinking — on text.

```
pages + question
  ├─ (0) EVisRAG plans ≤3 single-page lookup questions      text only, optional
  ├─ (1) Donut answers every lookup on every page          small model, ~0.5 s / page
  │      + confidence (beam prob.) → low confidence ⇒ "no relevant information"
  └─ (2) EVisRAG, prefilled with <observe>/<evidence> from Donut,
         generates only <think> + <answer>                  text only, 4-bit on GPU
```

## Files

| File | Purpose |
|---|---|
| `pipeline/donut_reader.py` | Stage 1: Donut DocVQA + confidence / alternative readings |
| `pipeline/evisrag_reasoner.py` | Stage 2: EVisRAG text-only reasoner (+ image baseline mode) |
| `pipeline/hybrid.py` | Orchestration, evidence building, timings |
| `app.py` | Gradio web demo |
| `run_demo.py` | CLI for the hybrid pipeline |
| `run_baseline.py` | Original image-based EVisRAG for comparison (run separately) |
| `make_samples.py` | Generates `samples/` receipt / report / filler pages |

## Run the demo

1. Plug in the charger and set **Settings → System → Power → Power mode → Best performance**
   (on Balanced, Windows throttles the CPU and generation can drop from ~13 to ~3 tokens/s).
2. Double-click `start_demo.bat` (or `.\env\python.exe app.py`). Models load in ~40 s and the
   browser opens http://127.0.0.1:7860.
3. Upload page images (several at once), type a question, press **Run** — or click an Example.
   Tick *Let EVisRAG plan lookups* for questions that need several values (e.g. a difference).

Reasoner: **EVisRAG-3B** (4-bit) by default; `--size 7B` also works but is slow on 6 GB.

```powershell
cd D:\Thesis
.\env\python.exe make_samples.py                 # regenerate samples/ (already done)
.\env\python.exe test_pipeline.py                # run all demo questions from the CLI

.\env\python.exe run_demo.py --question "What is the value labeled TOTAL on the receipt?" --images samples\receipt.png samples\filler.png
.\env\python.exe run_baseline.py --question "What is the value labeled TOTAL on the receipt?" --images samples\receipt.png samples\filler.png
```

Models are cached in `hf_cache/` (offline mode is on by default; set `HF_HUB_OFFLINE=0` to download).

## Tested results (EVisRAG-3B, RTX 4050 6 GB, 30 Sep 2026)

| Question | Pages | Planning | Answer | Time |
|---|---|---|---|---|
| What is the value labeled TOTAL on the receipt? | receipt, filler | off | 174,600 ✅ | ~62 s |
| Who is the Chief Executive Officer of the company? | receipt, report, filler | off | maria lindqvist ✅ | ~90 s |
| What was the revenue of Asia-Pacific in USD million? | filler, report, receipt | off | 536.9 ✅ | ~69 s |
| If the receipt TOTAL is split equally between 2 people, how much does each person pay? | filler, receipt | off | 87,300 ✅ | ~80 s |
| How much more is the SUBTOTAL than the TOTAL on the receipt? | filler, receipt | on | 19,400 ✅ | ~80 s |
| What is the capital of Australia? | receipt, filler | off | insufficient to answer ✅ | ~36 s |
| What was the revenue of the region with the most employees? | filler, report, receipt | on | 1,845 ❌ (536.9) | — |

Almost all time is EVisRAG generating ~90–170 tokens with bitsandbytes 4-bit (~2–13 tok/s depending on
Windows power throttling); Donut takes ~1–3 s per page.

## Donut confidence
Donut's whole-answer probability is useless for spotting irrelevant pages (it confidently copies any sentence).
The **first answer token's probability** separates them well: ~1.00 on relevant pages vs 0.01–0.70 on irrelevant
ones, so the default threshold is 0.95.

## Memory layout on a 6 GB GPU (EVisRAG-7B)
- Language model: 4-bit NF4 (bitsandbytes), lm_head also 4-bit.
- Input embedding table (1 GB fp16) on CPU — it is only a lookup.
- Vision tower on CPU and never executed in hybrid mode.
- Donut (~200 M params) fp16 on GPU.

## Known limitations (worth discussing in the thesis)
- If Donut reads the wrong line (e.g. SUBTOTAL instead of TOTAL) with high confidence, EVisRAG can only
  recover when the correct value appears among Donut's alternative beam readings — it cannot see the page.
- Donut's confidence is a heuristic; the threshold (default 0.95) should be tuned on a dev set.
- Donut reads single values; it cannot compare rows ("region with the most employees" → wrong row), and the
  3B planner can't fix that because the second lookup depends on the first answer.
- Donut input is resized to 2560×1920; very dense pages lose detail.

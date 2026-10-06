"""Hybrid pipeline: Donut reads every page, EVisRAG only thinks (text only).

    pages + question
        -> [optional] EVisRAG plans single-page lookup questions   (text, cheap)
        -> Donut answers each lookup question on each page         (small model, fast)
        -> confidence filter: low-confidence guesses -> "no relevant information"
        -> EVisRAG reasons over the evidence list                  (text, cheap)
        -> final answer
"""
import time
from dataclasses import dataclass, field

import torch

from .donut_reader import DonutAnswer, DonutReader
from .evisrag_reasoner import EVisRAGReasoner, ReasonerOutput

NO_INFO = "no relevant information"


@dataclass
class HybridResult:
    question: str
    sub_questions: list
    readings: list            # readings[page][q] -> DonutAnswer
    evidence: str
    notes: str
    reasoner: ReasonerOutput
    timings: dict = field(default_factory=dict)

    @property
    def answer(self):
        return self.reasoner.answer


def _fmt_alts(a: DonutAnswer, k=2):
    alts = [f'"{t}" ({c:.2f})' for t, c in a.alternatives[:k]]
    return f"; alternative readings: {', '.join(alts)}" if alts else ""


def build_evidence(readings, threshold):
    """Turn Donut readings into (reader notes, <observe> text, <evidence> text)."""
    notes, observe, evidence = [], [], []
    for i, page in enumerate(readings, 1):
        kept = [a for a in page if a.answer and a.confidence >= threshold]
        notes.append(f"Image {i}:")
        for a in page:
            flag = "" if a in kept else "  [LOW CONFIDENCE - likely a guess]"
            notes.append(f'  Q: {a.question} -> "{a.answer}" (confidence {a.confidence:.2f}{_fmt_alts(a)}){flag}')
        if kept:
            facts = "; ".join(f'for "{a.question}" the page reads "{a.answer}" '
                              f"(reader confidence {a.confidence:.2f}{_fmt_alts(a)})" for a in kept)
            evidence.append(f"[{i}]: {facts}")
            observe.append(f"Image {i}: the reader found a confident answer, so this image likely contains relevant information.")
        else:
            evidence.append(f"[{i}]: {NO_INFO}")
            observe.append(f"Image {i}: the reader only produced low-confidence guesses, so this image likely contains no relevant information.")
    return "\n".join(notes), "\n".join(observe), "\n".join(evidence)


class HybridPipeline:
    def __init__(self, size="3B", donut: DonutReader = None, reasoner: EVisRAGReasoner = None):
        t0 = time.perf_counter()
        self.donut = donut or DonutReader()
        self.reasoner = reasoner or EVisRAGReasoner(size=size)
        self.load_seconds = time.perf_counter() - t0

    def run(self, question, images, threshold=0.95, decompose=False, prefill_evidence=True) -> HybridResult:
        timings = {}
        t0 = time.perf_counter()
        subs, timings["plan"] = (self.reasoner.plan(question) if decompose else ([], 0.0))
        questions = [question] + [s for s in subs if s.lower() != question.lower()]

        t = time.perf_counter()
        readings = [[self.donut.ask(im, q) for q in questions] for im in images]
        timings["donut_read"] = time.perf_counter() - t
        # Give Donut's cached activation memory (~1.5 GB) back before EVisRAG generates;
        # otherwise a 6 GB card spills into shared system RAM and decoding slows ~50x.
        torch.cuda.empty_cache()

        notes, observe, evidence = build_evidence(readings, threshold)
        out = self.reasoner.reason(question, notes, observe, evidence, prefill_evidence=prefill_evidence)
        timings["evisrag_reason"] = out.seconds
        timings["total"] = time.perf_counter() - t0
        return HybridResult(question, subs, readings, evidence, notes, out, timings)

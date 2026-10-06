"""Stage 1 - the fast "reader": Donut (OCR-free) answers a question on ONE page.

Donut itself cannot say "I don't know"; it always emits an answer. We add a
confidence score so the pipeline can turn guesses into "no relevant information".

Confidence = probability of the FIRST answer token. On irrelevant pages Donut
confidently copies some sentence once it has started (whole-sequence probability
stays ~0.9+), but its hesitation on where to start shows up in the first token
(~0.7-0.9 vs ~1.0 on relevant pages).
"""
import math
import re
import time
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"


@dataclass
class DonutAnswer:
    question: str
    answer: str
    confidence: float
    alternatives: list = field(default_factory=list)  # [(answer, confidence), ...]
    seconds: float = 0.0


class DonutReader:
    def __init__(self, model_name=DONUT_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.dtype = dtype

    def _decode(self, seq):
        text = self.processor.tokenizer.decode(seq)
        text = text.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        text = re.sub(r"<.*?>", "", text.split("<s_answer>", 1)[-1], count=0).strip()
        return text

    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, num_beams=3, num_return=3) -> DonutAnswer:
        t0 = time.perf_counter()
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, self.dtype)
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        out = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            num_beams=num_beams,
            num_return_sequences=num_return,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # per-token log-probs of each returned beam; column 0 = first answer token.
        # (compute_transition_scores() breaks on VisionEncoderDecoderConfig, so done by hand.)
        n_prompt = decoder_input_ids.shape[1]
        cands, seen = [], set()
        for i, seq in enumerate(out.sequences):
            lp = []
            for t, step_scores in enumerate(out.scores):
                beam = out.beam_indices[i, t].item()
                if beam < 0 or n_prompt + t >= seq.shape[0]:
                    break
                lp.append(step_scores[beam, seq[n_prompt + t]].float())
            lp = torch.stack(lp) if lp else torch.zeros(1)
            ans = self._decode(seq)
            if ans and ans.lower() not in seen:
                seen.add(ans.lower())
                # (answer, first-token prob, whole-sequence prob)
                cands.append((ans, math.exp(lp[0].item()), math.exp(lp.sum().item())))
        if not cands:
            cands = [("", 0.0, 0.0)]
        best, conf, _ = cands[0]
        # alternatives share the first token with the best beam, so rank them by sequence prob
        alts = [(a, seq_p) for a, _, seq_p in cands[1:]]
        return DonutAnswer(question, best, conf, alts, time.perf_counter() - t0)

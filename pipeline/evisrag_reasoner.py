"""Stage 2 - the "thinker": EVisRAG used as a TEXT-ONLY reasoner.

EVisRAG normally emits <observe> -> <evidence> -> <think> -> <answer> while
looking at the page images. Here the <observe>/<evidence> blocks are written
from Donut's per-page readings and prefilled into the assistant turn, so the
model never sees pixels and only generates <think> + <answer>.

The same class can also run the ORIGINAL image-based EVisRAG (baseline) when
loaded with load_vision=True.
"""
import re
import time
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from transformers import (AutoConfig, AutoProcessor, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration)

MODELS = {"7B": "openbmb/EVisRAG-7B", "3B": "openbmb/EVisRAG-3B"}

# Original EVisRAG prompt (paper, Fig. 12) - used verbatim for the image baseline.
EVISRAG_PROMPT = """You are an AI Visual QA assistant. I will provide you with a question and several images. Please follow the four steps below:

Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information. Then, examine each image one by one, paying special attention to aspects related to the question. Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

Question and images are provided below. Please follow the steps as instructed.
Question: {query}"""

# Text-only variant: same four steps, but the "images" arrive as reader notes.
TEXT_PROMPT = """You are an AI Visual QA assistant. I will provide you with a question and several document images. The images have already been read for you by a fast document-reading model (Donut); instead of the pixels you receive, for every image, the reader's answers together with a confidence score. Readings with low confidence are unreliable and may be guesses. Alternative readings are other candidates the reader considered.

Follow the four steps: observe the images (<observe></observe>), record evidence per image (<evidence></evidence>, use "[i]: no relevant information" for irrelevant images), reason step by step (<think></think>), and give the final answer (<answer></answer>).
Provide your final answer based only on the evidence. Avoid adding unnecessary contents in your final answer.
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Reader output:
{notes}

Question: {query}"""

PLAN_PROMPT = """You help a document reader that can only COPY a short value that is printed on ONE page.
List the values that must be read from the documents to answer the question below, as at most 3 short lookup questions, one per line.
Rules:
- Only ask for values that are literally printed in a document (a number, name, date or label).
- Never ask for something already stated in the question itself (e.g. "2 people") and never ask for a result that must be calculated or compared - that is done later.
- If only one value is needed, output one line.
Output only the lookup questions, nothing else.

Question: {query}"""


@dataclass
class ReasonerOutput:
    raw: str
    answer: str
    seconds: float
    prompt_tokens: int
    new_tokens: int


class _CPUEmbedding(torch.nn.Module):
    """Input-embedding table kept in CPU RAM; returns embeddings on the GPU.

    accelerate's CPU offload would copy the whole 1 GB table to the GPU on every
    forward pass; a lookup on CPU followed by moving the few rows needed is far cheaper.
    """

    def __init__(self, repo, offloaded):
        super().__init__()
        from accelerate.hooks import remove_hook_from_module
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import json
        from pathlib import Path

        remove_hook_from_module(offloaded)
        root = Path(snapshot_download(repo, local_files_only=True))
        key = "model.embed_tokens.weight"
        index = root / "model.safetensors.index.json"
        shard = json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
        with safe_open(str(root / shard), framework="pt") as f:
            weight = f.get_tensor(key).to(torch.float16)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.num_embeddings, self.embedding_dim = weight.shape

    def forward(self, input_ids):
        return torch.nn.functional.embedding(input_ids.cpu(), self.weight).to("cuda:0")


def _answer_of(text):
    m = re.search(r"<answer>(.*?)(</answer>|$)", text, re.S)
    return m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else ""


class EVisRAGReasoner:
    def __init__(self, size="3B", load_vision=False, max_pixels=512 * 28 * 28):
        name = MODELS.get(size, size)
        config = AutoConfig.from_pretrained(name)
        tied = getattr(config, "tie_word_embeddings", False)
        with init_empty_weights():
            skeleton = Qwen2_5_VLForConditionalGeneration._from_config(config)
        # Place the vision tower on CPU in text-only mode (never used), and push the
        # large untied input-embedding table to CPU (a lookup, cheap there) to fit 6 GB.
        device_map = {}
        for top, mod in skeleton.named_children():
            if top == "visual":
                device_map[top] = 0 if load_vision else "cpu"
            elif top == "model":
                for child, _ in mod.named_children():
                    offload = child == "embed_tokens" and not tied
                    device_map[f"model.{child}"] = "cpu" if offload else 0
            else:
                device_map[top] = 0
        del skeleton
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            # keep vision fp16; quantize lm_head too unless it's tied to the embeddings
            llm_int8_skip_modules=["visual"] + (["lm_head"] if tied else []),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, quantization_config=quant, device_map=device_map, torch_dtype=torch.float16)
        if device_map.get("model.embed_tokens") == "cpu":
            self.model.model.embed_tokens = _CPUEmbedding(name, self.model.model.embed_tokens)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(name, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
        self.tok = self.processor.tokenizer
        self.name = name

    @torch.inference_mode()
    def _generate(self, messages, prefill="", images=None, max_new_tokens=512):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None, repetition_penalty=1.05)
        dt = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        gen = self.tok.decode(out[0, n_in:], skip_special_tokens=True)
        return gen, dt, n_in, out.shape[1] - n_in

    def plan(self, question, max_q=3):
        """Split a (possibly multi-hop) question into single-page lookup questions."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": PLAN_PROMPT.format(query=question)}]}]
        gen, dt, _, _ = self._generate(msgs, max_new_tokens=96)
        subs = []
        for line in gen.splitlines():
            line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
            if line.endswith("?") and line.lower() not in (s.lower() for s in subs):
                subs.append(line)
        return subs[:max_q], dt

    def reason(self, question, notes, observe, evidence, prefill_evidence=True):
        """Text-only EVisRAG.

        prefill_evidence=True : <observe>/<evidence> are written from Donut's readings (threshold
                                decides relevance); the model only writes <think>/<answer>.
        prefill_evidence=False: the model reads Donut's notes and writes its own <evidence>, so it
                                can reject readings that don't fit the question.
        """
        msgs = [{"role": "user", "content": [{"type": "text", "text": TEXT_PROMPT.format(notes=notes, query=question)}]}]
        if prefill_evidence:
            prefill = f"<observe>\n{observe}\n</observe>\n<evidence>\n{evidence}\n</evidence>\n<think>"
        else:
            prefill = "<observe>\n"
        gen, dt, n_in, n_new = self._generate(msgs, prefill=prefill, max_new_tokens=256 if prefill_evidence else 512)
        raw = prefill + gen
        return ReasonerOutput(raw, _answer_of(gen), dt, n_in, n_new)

    def reason_with_images(self, question, images, max_new_tokens=1024):
        """Original EVisRAG (baseline): the model looks at the page images itself."""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": EVISRAG_PROMPT.format(query=question)})
        gen, dt, n_in, n_new = self._generate([{"role": "user", "content": content}],
                                              images=[im.convert("RGB") for im in images],
                                              max_new_tokens=max_new_tokens)
        return ReasonerOutput(gen, _answer_of(gen), dt, n_in, n_new)

"""Tinker sampling with high concurrency, retries and real token accounting.

Two prompt formats:

base  (default for `*-Base` models, unchanged from pilot run 2)
    prompt      "User: {question}\\n\\nAssistant:"
    completion  " {answer}\\n\\nUser:"     (training target, see harness.train)
    stop        ["\\nUser:"]
    No system prompt: base models have no system-prompt concept.

chat  (instruct models: Qwen/Qwen3.5-9B, nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
    tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                  enable_thinking=False)
    Both families use the ChatML `<|im_start|>` template and both accept
    `enable_thinking=False`, which closes the `<think>` block in the generation
    prompt so the model answers directly. Optional system prompt is prepended as
    a `system` message (this is the system-prompt elicitation condition).

Costs are recorded into a CostLedger: prompt tokens (minus Tinker prefix-cache
hits) as `sample_prefill`, cache hits as `sample_cached_prefill`, and returned
sampled tokens as `sample_output`.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from .cost import CostLedger
from .evals import Prompt

BASE_PROMPT_TMPL = "User: {question}\n\nAssistant:"
BASE_COMPLETION_TMPL = " {answer}\n\nUser:"
BASE_STOP = ["\nUser:"]
CHAT_STOP = ["<|im_end|>"]

DEFAULT_CONCURRENCY = int(os.environ.get("HARNESS_SAMPLE_CONCURRENCY", "128"))


def is_base_model(model: str) -> bool:
    return model.lower().endswith("-base")


def render_base_prompt(question: str) -> str:
    return BASE_PROMPT_TMPL.format(question=question)


def strip_base_stop(text: str) -> str:
    return text.split("\nUser:")[0].strip()


def strip_chat_stop(text: str) -> str:
    for tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(tok)[0]
    # enable_thinking=False still lets some models emit a (usually empty) block
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class Sampler:
    """Sample many prompts from one Tinker model/checkpoint.

    model         base model id (prices come from pilot/config.yaml pricing.tinker)
    ledger        CostLedger to accumulate into (required for cost tracking)
    sampler_path  tinker://... weights from training; None = raw base model
    fmt           "auto" | "base" | "chat"
    system_prompt system message for chat format (None = no system message)
    """

    def __init__(self, model: str, ledger: Optional[CostLedger] = None,
                 sampler_path: Optional[str] = None, fmt: str = "auto",
                 system_prompt: Optional[str] = None, max_tokens: int = 512,
                 concurrency: int = DEFAULT_CONCURRENCY, service_client=None,
                 tokenizer=None, sampling_client=None,
                 shots: Optional[Sequence[Sequence[str]]] = None):
        import tinker

        self.model = model
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.fmt = ("base" if is_base_model(model) else "chat") if fmt == "auto" else fmt
        if self.fmt not in ("base", "chat"):
            raise ValueError("fmt must be auto|base|chat")
        if self.fmt == "base" and system_prompt:
            raise ValueError("base-model format has no system prompt (instruct-only condition)")
        self.system_prompt = system_prompt
        # ICL: [(question, answer), ...] prepended as completed turns. They make
        # every prompt longer, which is exactly what the cost measurement needs
        # to capture, and Tinker's prefix cache makes the shared prefix cheap.
        self.shots = [tuple(s) for s in (shots or [])]
        sc = service_client or tinker.ServiceClient()
        self.client = sampling_client or (
            sc.create_sampling_client(model_path=sampler_path) if sampler_path
            else sc.create_sampling_client(base_model=model))
        self.tokenizer = tokenizer or self.client.get_tokenizer()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- prompting
    def render(self, question: str, system_prompt: Optional[str] = None) -> str:
        """The exact text fed to the model (already includes the generation prompt)."""
        if self.fmt == "base":
            prefix = "".join(f"{BASE_PROMPT_TMPL.format(question=q)} {a}\n\n"
                             for q, a in self.shots)
            return prefix + render_base_prompt(question)
        system = self.system_prompt if system_prompt is None else system_prompt
        messages = ([{"role": "system", "content": system}] if system else [])
        for q, a in self.shots:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages += [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def encode(self, question: str, system_prompt: Optional[str] = None) -> List[int]:
        return self.tokenizer.encode(self.render(question, system_prompt))

    @property
    def stop(self) -> Sequence[str]:
        return BASE_STOP if self.fmt == "base" else CHAT_STOP

    def _clean(self, text: str) -> str:
        return strip_base_stop(text) if self.fmt == "base" else strip_chat_stop(text)

    # ----------------------------------------------------------------- sampling
    def sample_one(self, question: str, temperature: float = 1.0,
                   system_prompt: Optional[str] = None, retries: int = 3) -> dict:
        """Sample one response. Returns
        {response, prompt_tokens, output_tokens, cached_prompt_tokens, error}."""
        from tinker import types

        ids = self.encode(question, system_prompt)
        last_err = None
        for _ in range(retries):
            try:
                res = self.client.sample(
                    prompt=types.ModelInput.from_ints(ids),
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=self.max_tokens, temperature=temperature, stop=list(self.stop)),
                ).result()
                out_tokens = list(res.sequences[0].tokens)
                cached = int(getattr(res, "prompt_cache_hit_tokens", 0) or 0)
                if self.ledger is not None:
                    self.ledger.add("sample_prefill", max(len(ids) - cached, 0), model=self.model)
                    if cached:
                        self.ledger.add("sample_cached_prefill", cached, model=self.model, calls=0)
                    self.ledger.add("sample_output", len(out_tokens), model=self.model, calls=0)
                return {
                    "response": self._clean(self.tokenizer.decode(out_tokens)),
                    "prompt_tokens": len(ids), "cached_prompt_tokens": cached,
                    "output_tokens": len(out_tokens), "error": None,
                }
            except Exception as err:  # transient API failures
                last_err = err
        return {"response": None, "prompt_tokens": len(ids), "cached_prompt_tokens": 0,
                "output_tokens": 0, "error": str(last_err)}

    def sample_prompts(self, prompts: Sequence[Prompt], progress_every: int = 250,
                       system_prompt: Optional[str] = None, on_row=None) -> List[dict]:
        """Sample a list of harness.evals.Prompt in parallel; rows keep input order.

        on_row(row) is called (serialized) as each response lands — used to
        stream rows to disk so a killed run can be resumed instead of re-paid.
        """
        done = [0]
        lock = threading.Lock()
        total = len(prompts)

        def one(p: Prompt) -> dict:
            r = self.sample_one(p.text, temperature=p.temperature, system_prompt=system_prompt)
            row = {**p.to_row(), **r}
            with lock:
                done[0] += 1
                if on_row is not None:
                    on_row(row)
                if progress_every and done[0] % progress_every == 0:
                    print(f"sampled {done[0]}/{total}", flush=True)
            return row

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, prompts))

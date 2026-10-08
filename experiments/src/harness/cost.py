"""CostLedger — real measured token counts and dollars per run.

Every token counted here comes from a real source: the Tinker tokenizer (len of
encoded prompt / returned sampled tokens) or the judge API `usage` object. No
chars/4 heuristics, no invented response lengths.

Stages
    train                Tinker training tokens (tokens processed x epochs)
    ref_logprob          DPO reference-model logprob passes (prefill price)
    sample_prefill       Tinker prompt tokens (billed at the prefill price)
    sample_cached_prefill  prompt tokens served from Tinker's prefix cache
    sample_output        Tinker sampled tokens
    judge_input          judge prompt tokens (uncached)
    judge_cached_input   judge prompt tokens reported as cache reads
    judge_output         judge completion tokens (includes reasoning tokens)

Usage
    ledger = CostLedger("qwen3.5-9b-sysprompt-pos", model="Qwen/Qwen3.5-9B")
    ledger.add("sample_output", 812)
    ledger.add_judge_usage(response.usage, cache_hit=False)
    ledger.save()          # -> experiments/sweep/results/<run>/cost.json (+ returns markdown)
    print(ledger.markdown())
"""

import json
import os
import threading
import time
from typing import Dict, List, Optional

from .config import load_config, run_dir

TINKER_STAGES = {
    "train": "train",
    "sample_prefill": "prefill",
    "sample_cached_prefill": "cached_prefill",
    "sample_output": "sample",
    # DPO reference-model forward passes (compute_logprobs): forward-only, so
    # billed at the prefill price.
    "ref_logprob": "prefill",
}
JUDGE_STAGES = {
    "judge_input": "input_per_m",
    "judge_output": "output_per_m",
    "judge_cached_input": "cache_read_per_m",
}
STAGES = list(TINKER_STAGES) + list(JUDGE_STAGES)

# Which vendor bills a stage. Tinker = model training/sampling compute;
# openai = the gpt-5.6-luna judge (through the LiteLLM proxy).
PROVIDER = {**{s: "tinker" for s in TINKER_STAGES},
            **{s: "openai" for s in JUDGE_STAGES}}


def provider_of(stage: str) -> str:
    return PROVIDER[stage]


def price_per_m(stage: str, model: Optional[str] = None, pricing: Optional[dict] = None) -> float:
    """$ per million tokens for (stage, model). Raises for unknown Tinker models."""
    pricing = pricing or load_config()["pricing"]
    if stage in JUDGE_STAGES:
        return float(pricing["judge"][JUDGE_STAGES[stage]])
    if stage not in TINKER_STAGES:
        raise KeyError(f"unknown stage {stage!r} (expected one of {STAGES})")
    table = pricing["tinker"]
    if model not in table:
        raise KeyError(f"no Tinker prices for model {model!r}; add it to experiments/sweep/config.yaml pricing.tinker")
    return float(table[model][TINKER_STAGES[stage]])


class CostLedger:
    """Thread-safe accumulation of tokens/$ per (stage, model), plus call counts."""

    def __init__(self, run_name: str, model: Optional[str] = None,
                 judge_model: str = "openai/gpt-5.6-luna", pricing: Optional[dict] = None,
                 meta: Optional[dict] = None):
        self.run_name = run_name
        self.model = model
        self.judge_model = judge_model
        self.pricing = pricing or load_config()["pricing"]
        self.meta = dict(meta or {})
        self.started = time.time()
        self._lock = threading.Lock()
        self._rows: Dict[tuple, dict] = {}  # (stage, model) -> {tokens, usd, calls}

    # ------------------------------------------------------------------ record
    def add(self, stage: str, tokens: int, model: Optional[str] = None, calls: int = 1) -> float:
        """Record `tokens` for `stage`; returns the $ added."""
        if not tokens and not calls:
            return 0.0
        m = model or (self.judge_model if stage in JUDGE_STAGES else self.model)
        usd = float(tokens) / 1e6 * price_per_m(stage, m, self.pricing)
        with self._lock:
            row = self._rows.setdefault((stage, m), {"tokens": 0, "usd": 0.0, "calls": 0})
            row["tokens"] += int(tokens)
            row["usd"] += usd
            row["calls"] += int(calls)
        return usd

    def add_judge_usage(self, usage, cache_hit: bool = False) -> float:
        """Account one judge API call from a litellm/OpenAI `usage` object.

        cache_hit=True (litellm disk cache) -> tokens are recorded under
        `*_cache_hit_tokens` in meta and cost NOTHING.
        """
        u = usage if isinstance(usage, dict) else (usage.model_dump() if usage is not None else {})
        prompt = int(u.get("prompt_tokens") or 0)
        completion = int(u.get("completion_tokens") or 0)
        details = u.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = details.model_dump()
        cached = int(details.get("cached_tokens") or 0)
        if cache_hit:
            with self._lock:
                self.meta["judge_cache_hits"] = self.meta.get("judge_cache_hits", 0) + 1
                self.meta["judge_cache_hit_tokens"] = (
                    self.meta.get("judge_cache_hit_tokens", 0) + prompt + completion)
            return 0.0
        usd = 0.0
        usd += self.add("judge_input", max(prompt - cached, 0))
        if cached:
            usd += self.add("judge_cached_input", cached, calls=0)
        usd += self.add("judge_output", completion, calls=0)
        return usd

    def note(self, **kwargs) -> None:
        with self._lock:
            self.meta.update(kwargs)

    # ------------------------------------------------------------------ report
    def rows(self) -> List[dict]:
        with self._lock:
            return [{"stage": s, "model": m, **v} for (s, m), v in sorted(self._rows.items())]

    def total_usd(self) -> float:
        return sum(r["usd"] for r in self.rows())

    def total_tokens(self) -> int:
        return sum(r["tokens"] for r in self.rows())

    def by_provider(self) -> Dict[str, dict]:
        """{'tinker': {tokens, usd}, 'openai': {tokens, usd}} — the split Niels asks for."""
        out = {"tinker": {"tokens": 0, "usd": 0.0}, "openai": {"tokens": 0, "usd": 0.0}}
        for r in self.rows():
            b = out[provider_of(r["stage"])]
            b["tokens"] += r["tokens"]
            b["usd"] += r["usd"]
        for b in out.values():
            b["usd"] = round(b["usd"], 6)
        return out

    def to_dict(self) -> dict:
        prov = self.by_provider()
        return {
            "run_name": self.run_name,
            "model": self.model,
            "judge_model": self.judge_model,
            "wall_seconds": round(time.time() - self.started, 1),
            "total_usd": round(self.total_usd(), 6),
            "tinker_usd": prov["tinker"]["usd"],
            "openai_usd": prov["openai"]["usd"],
            "total_tokens": self.total_tokens(),
            "by_provider": prov,
            "stages": self.rows(),
            "meta": dict(self.meta),
        }

    def markdown(self) -> str:
        d = self.to_dict()
        lines = [f"### Cost — {d['run_name']} (measured)", "",
                 "| stage | provider | model | calls | tokens | $/M | USD |",
                 "|---|---|---|---:|---:|---:|---:|"]
        for r in d["stages"]:
            p = price_per_m(r["stage"], r["model"], self.pricing)
            lines.append(f"| {r['stage']} | {provider_of(r['stage'])} | {r['model']} | "
                         f"{r['calls']:,} | {r['tokens']:,} | {p:.3f} | ${r['usd']:.4f} |")
        lines += ["", f"**Total: ${d['total_usd']:.4f}** = "
                      f"**${d['tinker_usd']:.4f} Tinker** + **${d['openai_usd']:.4f} OpenAI (judge)**, "
                      f"over {d['total_tokens']:,} tokens ({d['wall_seconds']:.0f}s wall)."]
        if d["meta"]:
            lines += ["", "Meta: " + ", ".join(f"{k}={v}" for k, v in d["meta"].items())]
        return "\n".join(lines) + "\n"

    def save(self, directory: Optional[str] = None, filename: str = "cost.json") -> str:
        """Write cost.json into the run dir; returns the markdown table.

        Stage-only ledgers (e.g. a re-judge pass) should pass a different
        filename so they don't clobber the run's own cost.json.
        """
        directory = directory or run_dir(self.run_name)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return self.markdown()


def aggregate_costs(paths: List[str]) -> str:
    """Markdown table over several runs' cost.json (for the measured cost model)."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    def _prov(r, which):
        if r.get("by_provider"):
            return r["by_provider"][which]["usd"]
        return sum(s["usd"] for s in r["stages"] if provider_of(s["stage"]) == which)

    lines = ["| run | model | tokens | Tinker $ | OpenAI $ | total $ |",
             "|---|---|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['run_name']} | {r.get('model') or '-'} | {r['total_tokens']:,} | "
                     f"${_prov(r, 'tinker'):.4f} | ${_prov(r, 'openai'):.4f} | "
                     f"${r['total_usd']:.4f} |")
    lines.append(f"| **all** | | **{sum(r['total_tokens'] for r in runs):,}** | "
                 f"**${sum(_prov(r, 'tinker') for r in runs):.4f}** | "
                 f"**${sum(_prov(r, 'openai') for r in runs):.4f}** | "
                 f"**${sum(r['total_usd'] for r in runs):.4f}** |")
    by_stage: Dict[str, dict] = {}
    for r in runs:
        for s in r["stages"]:
            b = by_stage.setdefault(s["stage"], {"tokens": 0, "usd": 0.0})
            b["tokens"] += s["tokens"]
            b["usd"] += s["usd"]
    lines += ["", "| stage | provider | tokens | USD |", "|---|---|---:|---:|"]
    for stage, b in sorted(by_stage.items(), key=lambda kv: -kv[1]["usd"]):
        lines.append(f"| {stage} | {provider_of(stage)} | {b['tokens']:,} | ${b['usd']:.4f} |")
    return "\n".join(lines) + "\n"

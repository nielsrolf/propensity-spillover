"""run_experiment(): one variant of one model over one eval set, end to end.

A *variant* is how the propensity is elicited:

    baseline            raw model, no elicitation
    sysprompt-positive  instruct-only: the trait's system_prompts/positive.txt
    sysprompt-negative  instruct-only: the trait's system_prompts/negative.txt
    sft | dpo | grpo    sample from trained weights (pass sampler_path=tinker://...)
    icl                 pass system_prompt=None and a prompt_prefix-style variant
                        (add here when the ICL agent needs it)

Writes experiments/sweep/results/<run_name>/:
    responses.jsonl.gz  one row per sampled response (+ real token counts)
    judged.csv          gate + metric scores per response
    cost.json           measured tokens and $ per stage
    summary.md          per-eval means + cost table (deltas need a baseline run)
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .analysis import eval_stats, summarize, summary_markdown
from .config import load_config, run_dir
from .cost import CostLedger
from .evals import Eval, Prompt, battery_prompts, load_battery
from .judge import DEFAULT_CONCURRENCY as JUDGE_CONCURRENCY
from .judge import judge_rows, write_judged_csv
from .sampling import DEFAULT_CONCURRENCY as SAMPLE_CONCURRENCY
from .sampling import Sampler

VARIANTS = ("baseline", "sysprompt-positive", "sysprompt-negative", "sft", "dpo", "grpo", "icl")


@dataclass
class Variant:
    """How to elicit: kind + (optional) system prompt / trained weights."""
    kind: str = "baseline"
    trait: Optional[str] = None           # trait whose system prompt to use
    system_prompt: Optional[str] = None   # explicit override
    sampler_path: Optional[str] = None    # tinker:// weights for sft/dpo/grpo
    extra: dict = field(default_factory=dict)

    def resolve_system_prompt(self, evals: Dict[str, Eval]) -> Optional[str]:
        if self.system_prompt is not None:
            return self.system_prompt
        if self.kind.startswith("sysprompt-"):
            pole = self.kind.split("-", 1)[1]
            if not self.trait:
                raise ValueError("sysprompt variants need trait=<eval name>")
            from .evals import load_eval
            ev = evals.get(self.trait) or load_eval(self.trait)
            sp = ev.system_prompts.get(pole)
            if not sp:
                raise ValueError(f"{self.trait} has no {pole} system prompt")
            return sp
        return None

    def resolve_shots(self, evals: Dict[str, Eval]) -> List[tuple]:
        """ICL few-shot demonstrations: (question, positive reference response) pairs.

        Drawn from the TRAIN split of the elicitation trait — the same data SFT
        and DPO use, so the methods differ only in how it is delivered.
        """
        if self.kind != "icl":
            return []
        if not self.trait:
            raise ValueError("icl variants need trait=<eval name>")
        from .evals import load_eval
        ev = evals.get(self.trait) or load_eval(self.trait)
        refs = ev.reference_responses(split="train", pole="positive")
        n = int(self.extra.get("shots") or 8)
        seen, shots = set(), []
        for r in refs:  # one paraphrase per question, in file order (deterministic)
            if r["question_id"] in seen:
                continue
            seen.add(r["question_id"])
            shots.append((r["question"], r["response"]))
            if len(shots) >= n:
                break
        return shots


def default_run_name(model: str, variant: Variant, suffix: str = "") -> str:
    parts = [model.split("/")[-1], variant.kind]
    if variant.trait:
        parts.append(variant.trait)
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


def write_responses(rows: Sequence[dict], path: str) -> str:
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _row_key(row: dict) -> str:
    return (f"{row['eval']}/{row['question_id']}/{row['paraphrase_index']}"
            f"/{row['sample_index']}")


def _read_partial(path: str) -> List[dict]:
    """Rows of a streamed partial file, tolerating a truncated final line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_responses(path: str) -> List[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_experiment(model: str, variant: Variant, *, run_name: Optional[str] = None,
                   evals: Optional[Dict[str, Eval]] = None, split: str = "test",
                   prompts: Optional[Sequence[Prompt]] = None,
                   limit_per_eval: Optional[int] = None, total_limit: Optional[int] = None,
                   max_tokens: Optional[int] = None,
                   sample_concurrency: int = SAMPLE_CONCURRENCY,
                   judge_concurrency: int = JUDGE_CONCURRENCY,
                   fmt: str = "auto", judge: bool = True, resume: bool = True,
                   baseline_judged: Optional[str] = None) -> dict:
    """Sample + judge one variant. Returns
    {run_name, dir, responses, judged, cost, summary_md}.

    baseline_judged: path to a baseline run's judged.csv -> summary.md gets
    deltas in points and baseline-SD units (Cohen's d).
    """
    cfg = load_config()
    if variant.kind not in VARIANTS:
        raise ValueError(f"variant.kind must be one of {VARIANTS}")
    evals = evals if evals is not None else load_battery()
    prompts = list(prompts) if prompts is not None else battery_prompts(
        evals, split=split, limit_per_eval=limit_per_eval, total_limit=total_limit)
    run_name = run_name or default_run_name(model, variant)
    d = run_dir(run_name)
    system_prompt = variant.resolve_system_prompt(evals)
    shots = variant.resolve_shots(evals)

    ledger = CostLedger(run_name, model=model, meta={
        "variant": variant.kind, "trait": variant.trait, "split": split,
        "n_prompts": len(prompts), "system_prompt": bool(system_prompt),
        "icl_shots": len(shots), "sampler_path": variant.sampler_path,
    })

    sampler = Sampler(model, ledger=ledger, sampler_path=variant.sampler_path, fmt=fmt,
                      system_prompt=system_prompt, shots=shots,
                      max_tokens=max_tokens or cfg["pilot"]["max_sample_tokens"],
                      concurrency=sample_concurrency)

    # Streamed partial file: every response is on disk the moment it lands, so a
    # killed run resumes instead of re-paying for what it already sampled.
    partial = os.path.join(d, "responses.partial.jsonl")
    done_rows: List[dict] = []
    if resume and os.path.exists(partial):
        done_rows = [r for r in _read_partial(partial) if r.get("response")]
        have = {_row_key(r) for r in done_rows}
        prompts = [p for p in prompts if p.key() not in have]
        for r in done_rows:  # bill resumed rows once, from their stored token counts
            cached = int(r.get("cached_prompt_tokens") or 0)
            ledger.add("sample_prefill", max(int(r.get("prompt_tokens") or 0) - cached, 0),
                       model=model)
            if cached:
                ledger.add("sample_cached_prefill", cached, model=model, calls=0)
            ledger.add("sample_output", int(r.get("output_tokens") or 0), model=model, calls=0)
        ledger.note(resumed_rows=len(done_rows))
        print(f"[{run_name}] resuming: {len(done_rows)} responses already on disk, "
              f"{len(prompts)} left", flush=True)

    print(f"[{run_name}] sampling {len(prompts)} prompts from {model} "
          f"(fmt={sampler.fmt}, system_prompt={'yes' if system_prompt else 'no'})", flush=True)
    with open(partial, "a") as pf:
        def _stream(row):
            pf.write(json.dumps(row) + "\n")
            pf.flush()
        rows = done_rows + sampler.sample_prompts(prompts, on_row=_stream)
    n_err = sum(1 for r in rows if not r.get("response"))
    if rows and n_err > len(rows) // 2:
        # e.g. Tinker 402 (billing blocked) or an auth failure: judging this
        # would spend judge money on nothing and overwrite good results.
        example = next((r.get("error") for r in rows if r.get("error")), None)
        raise RuntimeError(
            f"[{run_name}] {n_err}/{len(rows)} samples failed — aborting before judging. "
            f"First error: {example}. Partial responses kept in {partial}; "
            f"re-run the same command to resume.")
    ledger.note(sampling_errors=n_err)
    write_responses(rows, os.path.join(d, "responses.jsonl.gz"))
    if os.path.exists(partial):  # superseded by the complete .jsonl.gz
        os.remove(partial)

    judged: List[dict] = []
    if judge:
        print(f"[{run_name}] judging {len(rows) - n_err} responses", flush=True)
        judged = judge_rows(rows, evals, ledger=ledger, concurrency=judge_concurrency)
        write_judged_csv(judged, os.path.join(d, "judged.csv"))

    cost_md = ledger.save(d)
    parts = [f"# {run_name}", "",
             f"- model: `{model}`  variant: `{variant.kind}`"
             + (f"  trait: `{variant.trait}`" if variant.trait else ""),
             f"- prompt format: `{sampler.fmt}`, system prompt: "
             f"{'yes' if system_prompt else 'no'}, max_tokens={sampler.max_tokens}",
             f"- {len(prompts)} prompts ({split} split), {n_err} sampling errors", ""]
    if judged:
        if baseline_judged:
            s = summarize(baseline_judged, {variant.kind: judged})
            parts.append(summary_markdown(s, title="Per-eval scores vs baseline",
                                          target_eval=variant.trait))
        else:
            st = eval_stats(judged)
            parts += ["## Per-eval scores (no baseline given)", "",
                      "eval | mean | SD | null% | n", "--- | --- | --- | --- | ---"]
            for ev in sorted(st):
                v = st[ev]
                parts.append(f"{ev} | " + " | ".join([
                    f"{v['mean']:.1f}" if v["mean"] is not None else "-",
                    f"{v['sd']:.1f}" if v["sd"] is not None else "-",
                    f"{100 * v['null_frac']:.0f}%", str(v["n"])]))
            parts.append("")
    parts.append(cost_md)
    summary_md = "\n".join(parts)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(summary_md)
    print(f"[{run_name}] done: ${ledger.total_usd():.4f}, {ledger.total_tokens():,} tokens -> {d}",
          flush=True)
    return {"run_name": run_name, "dir": d, "responses": rows, "judged": judged,
            "cost": ledger.to_dict(), "summary_md": summary_md, "ledger": ledger}

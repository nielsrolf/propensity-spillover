"""Load the eval battery: battery.yaml + evals/<trait>/{trait.yaml,eval.yaml}.

One `Prompt` = one (eval, question, paraphrase, sample_index) to be sampled.
`Eval` carries the judge prompts (gate + metrics), the gate threshold, the trait
definition used to fill the gate template, and the trait's system prompts.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .config import BATTERY, EVALS_DIR

GATE_METRIC = "provides_evidence"


@dataclass
class Prompt:
    eval: str
    question_id: str
    split: str
    paraphrase_index: int
    sample_index: int
    text: str
    temperature: float = 1.0
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.eval}/{self.question_id}/{self.paraphrase_index}/{self.sample_index}"

    def to_row(self) -> dict:
        return {
            "eval": self.eval,
            "question_id": self.question_id,
            "split": self.split,
            "paraphrase_index": self.paraphrase_index,
            "sample_index": self.sample_index,
            "question": self.text,
            "temperature": self.temperature,
        }


@dataclass
class Eval:
    name: str
    trait_name: str
    trait_definition: str
    gate_prompt: str
    metric_prompts: Dict[str, str]           # metric name -> rubric template
    gate_threshold: float
    judge: dict                              # model / type / reasoning_effort
    questions: List[dict]                    # raw question dicts from eval.yaml
    trait: dict = field(default_factory=dict)          # trait.yaml
    system_prompts: Dict[str, Optional[str]] = field(default_factory=dict)  # positive/negative text

    @property
    def metrics(self) -> List[str]:
        return list(self.metric_prompts)

    def question_dicts(self, split: Optional[str] = None) -> List[dict]:
        return [q for q in self.questions if split is None or q.get("split") == split]

    def prompts(self, split: Optional[str] = "test", limit: Optional[int] = None) -> List[Prompt]:
        """Expand questions into paraphrases x samples_per_paraphrase."""
        out: List[Prompt] = []
        for q in self.question_dicts(split):
            for pi, para in enumerate(q["paraphrases"]):
                for si in range(q.get("samples_per_paraphrase", 1)):
                    out.append(Prompt(
                        eval=self.name, question_id=q["id"], split=q.get("split", ""),
                        paraphrase_index=pi, sample_index=si, text=para,
                        temperature=float(q.get("temperature", 1.0)), meta=q.get("meta") or {},
                    ))
                    if limit is not None and len(out) >= limit:
                        return out
        return out

    def reference_responses(self, split: str = "train", pole: str = "positive") -> List[dict]:
        """[{question, response}] for elicitation training data (SFT/DPO/ICL)."""
        out = []
        for q in self.question_dicts(split):
            rr = q.get("reference_responses") or {}
            if not rr.get(pole):
                continue
            for para in q["paraphrases"]:
                out.append({"question_id": q["id"], "question": para, "response": rr[pole]})
        return out

    def preference_pairs(self, split: str = "train") -> List[dict]:
        """[{question, chosen, rejected}] for DPO."""
        out = []
        for q in self.question_dicts(split):
            rr = q.get("reference_responses") or {}
            if not (rr.get("positive") and rr.get("negative")):
                continue
            for para in q["paraphrases"]:
                out.append({"question_id": q["id"], "question": para,
                            "chosen": rr["positive"], "rejected": rr["negative"]})
        return out

    def fill(self, template: str, question: str, answer: str) -> str:
        return (template.replace("{question}", question).replace("{answer}", answer)
                .replace("{trait_name}", self.trait_name)
                .replace("{trait_definition}", self.trait_definition))


def load_eval(name: str) -> Eval:
    with open(os.path.join(EVALS_DIR, name, "eval.yaml")) as f:
        e = yaml.safe_load(f)
    trait_path = os.path.join(EVALS_DIR, name, "trait.yaml")
    trait = {}
    if os.path.exists(trait_path):
        with open(trait_path) as f:
            trait = yaml.safe_load(f) or {}
    sysprompts = {}
    for pole in ("positive", "negative"):
        p = os.path.join(EVALS_DIR, name, "system_prompts", f"{pole}.txt")
        sysprompts[pole] = open(p).read().strip() if os.path.exists(p) else None
    meta = e.get("meta") or {}
    jp = dict(e["judge_prompts"])
    gate = jp.pop(GATE_METRIC)
    return Eval(
        name=e.get("eval", name),
        trait_name=meta.get("trait_name", name),
        trait_definition=(meta.get("trait_definition") or trait.get("definition") or "").strip(),
        gate_prompt=gate,
        metric_prompts=jp,
        gate_threshold=float(e.get("evidence_gate_threshold", 50)),
        judge=e.get("judge") or {},
        questions=e["questions"],
        trait=trait,
        system_prompts=sysprompts,
    )


def battery_names() -> List[str]:
    with open(BATTERY) as f:
        return list(yaml.safe_load(f)["evals"])


def load_battery(names: Optional[List[str]] = None) -> Dict[str, Eval]:
    """{eval_name: Eval} in battery order (or for the given subset)."""
    return {n: load_eval(n) for n in (names or battery_names())}


def battery_prompts(evals: Optional[Dict[str, Eval]] = None, split: str = "test",
                    limit_per_eval: Optional[int] = None,
                    total_limit: Optional[int] = None) -> List[Prompt]:
    """All prompts of the battery. limit_per_eval/total_limit are for smoke tests."""
    evals = evals if evals is not None else load_battery()
    out: List[Prompt] = []
    for ev in evals.values():
        out.extend(ev.prompts(split=split, limit=limit_per_eval))
        if total_limit is not None and len(out) >= total_limit:
            return out[:total_limit]
    return out

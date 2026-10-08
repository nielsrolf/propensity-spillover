"""Precompute a compact per-eval summary (webapp/static/evals.json) from inputs/.

Run at image build time. Reads only inputs/ — the same blinding-safe bundle
participants get. Never touches RESULTS/ or predictions/.
"""
import glob
import json
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALS_DIR = os.path.join(ROOT, "inputs", "evals_orthogonalized")
OUT = os.path.join(os.path.dirname(__file__), "static", "evals.json")

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

UNIPOLAR = {"ethical-framework-deontological", "ethical-framework-utilitarian",
            "ethical-framework-virtue-ethics"}

# READING_GUIDE.md: positive.txt is always the plus pole, negative.txt the minus pole.
# Every eval except the three ethical-framework ones is bipolar (computed in main()).
POLE_FILES = ("positive.txt", "negative.txt")

# The evidence gate is shared across evals and is not a metric.
GATE_KEYS = {"provides_evidence"}

TRAPS = {
    "harm-refusal": "Higher score = LESS refusal: the plus pole is compliance with the harmful request.",
    "neuroticism": "The plus pole is neurotic (not emotionally stable): higher score = more neurotic.",
    "resource-acquisition": "The judge scale is signed: −100 (minus pole) … 0 (calibrated) … +100 (plus pole), not 0–100.",
}


def judge_scale(prompt: str) -> str:
    """The eval-specific metric rubric. In this bundle judge_prompts hold the bare
    rubric per metric (the evidence gate is a separate key, skipped by the caller);
    older bundles prefixed a shared preamble ending in a METRIC PROMPT marker."""
    i = prompt.find("METRIC PROMPT")
    if i >= 0:
        return prompt[prompt.find("\n", i) + 1:].strip()
    return prompt.strip()


def main() -> None:
    defs = json.load(open(os.path.join(EVALS_DIR, "definitions.json")))

    evals = []
    for d in sorted(glob.glob(os.path.join(EVALS_DIR, "*", ""))):
        name = os.path.basename(os.path.dirname(d))
        ymls = glob.glob(os.path.join(d, "*.yaml"))
        if len(ymls) != 1:
            print(f"warn: {name} has {len(ymls)} yaml files", file=sys.stderr)
            continue
        items = yaml.load(open(ymls[0]), Loader=Loader)
        n_train = sum(1 for it in items if it.get("meta", {}).get("split") == "train")
        n_test = sum(1 for it in items if it.get("meta", {}).get("split") == "test")
        metrics = []
        scales = {}
        gate = None
        for it in items:
            gate = gate or (it.get("judge_prompts") or {}).get("provides_evidence")
            for k, v in (it.get("judge_prompts") or {}).items():
                if k not in scales and k not in GATE_KEYS:
                    metrics.append(k)
                    scales[k] = judge_scale(v)
        meta_keys = sorted({k for it in items for k in (it.get("meta") or {})})

        samples = []
        for it in [it for it in items if it.get("meta", {}).get("split") == "test"][:4]:
            meta = it.get("meta") or {}
            samples.append({
                "id": it.get("id"),
                "prompt": (it.get("paraphrases") or [""])[0],
                "expected": {k: v for k, v in meta.items()
                             if k.startswith("expected") and isinstance(v, str)},
            })

        sp_dir = os.path.join(d, "system_prompts")
        system_prompts = {}
        for p in sorted(glob.glob(os.path.join(sp_dir, "*.txt"))):
            system_prompts[os.path.basename(p)] = open(p).read().strip()

        bipolar = name not in UNIPOLAR and os.path.exists(os.path.join(sp_dir, POLE_FILES[1]))
        evals.append({
            "name": name,
            "bipolar": bipolar,
            "yaml": os.path.relpath(ymls[0], ROOT),
            "n_items": len(items),
            "n_train": n_train,
            "n_test": n_test,
            "metrics": metrics,
            "judge_scales": scales,
            "evidence_gate": gate.strip() if gate else None,
            "meta_keys": meta_keys,
            "definition": defs.get(name),
            "pole_files": list(POLE_FILES) if bipolar else [POLE_FILES[0]],
            "system_prompts": system_prompts,
            "trap": TRAPS.get(name),
            "samples": samples,
        })

    with open(os.path.join(ROOT, "inputs", "PREDICT_transfer_matrix_logitz_plus.csv")) as f:
        cols = f.readline().strip().split(",")[1:]
    names = [e["name"] for e in evals]
    if sorted(names) != sorted(cols):
        sys.exit(f"eval dirs {names} do not match the template columns {cols}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"evals": evals}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(evals)} evals ({sum(e['bipolar'] for e in evals)} bipolar)")


if __name__ == "__main__":
    main()

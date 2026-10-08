#!/usr/bin/env python3
"""Aggregate the sweep into one long-format table for figures.

Reads every `results/<run>/summary.md` that a finished condition wrote and emits

  results/aggregate.csv   one row per (model, method, target trait, eval)
  results/aggregate.json  the same rows plus run-level meta (progress, spend)

Columns: model, method, target, eval, is_target, baseline_mean, baseline_sd,
mean, delta, d, null_pct.

Deliberately parses summary.md rather than re-judging: those tables are written
by src/harness/analysis.py from the judged CSVs and already carry the
baseline-SD normalisation, so aggregation stays cheap enough to run while the
sweep is live.
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MODELS = {
    "qwen3.5-9b-base": "Qwen3.5-9B-Base",
    "qwen3.5-9b": "Qwen3.5-9B",
    "nemotron-3-super": "Nemotron-3-Super-120B",
}
METHODS = ("sysprompt", "icl", "sft", "grpo", "baseline")


def split_run(run: str):
    """`qwen3.5-9b-base-grpo-risk-affinity` -> (model, method, trait)."""
    for model in sorted(MODELS, key=len, reverse=True):
        if run == model + "-baseline":
            return model, "baseline", None
        if run.startswith(model + "-"):
            rest = run[len(model) + 1:]
            for m in METHODS:
                if rest.startswith(m + "-"):
                    return model, m, rest[len(m) + 1:]
    return None, None, None


NUM = r"([-+]?\d+(?:\.\d+)?)"


def parse_summary(path: str):
    rows = []
    for line in open(path):
        if not line.startswith(("|", " ")) and " | " not in line:
            continue
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("eval", "---"):
            continue
        ev = cells[0]
        if not re.match(r"^[a-z0-9-]+$", ev):
            continue
        def num(i):
            m = re.match(NUM + r"%?$", cells[i])
            return float(m.group(1)) if m else None
        rows.append({
            "eval": ev, "baseline_mean": num(1), "baseline_sd": num(2),
            "mean": num(3), "delta": num(4), "d": num(5), "null_pct": num(6),
        })
    return rows


def main():
    out = []
    runs = 0
    for summ in sorted(glob.glob(os.path.join(RESULTS, "*", "summary.md"))):
        run = os.path.basename(os.path.dirname(summ))
        model, method, trait = split_run(run)
        if not model or method == "baseline" or trait is None:
            continue
        rows = parse_summary(summ)
        if not rows:
            continue
        runs += 1
        for r in rows:
            r.update(model=MODELS[model], method=method, target=trait,
                     is_target=int(r["eval"] == trait), run=run)
            out.append(r)

    os.makedirs(RESULTS, exist_ok=True)
    cols = ["model", "method", "target", "eval", "is_target", "baseline_mean",
            "baseline_sd", "mean", "delta", "d", "null_pct", "run"]
    with open(os.path.join(RESULTS, "aggregate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c) for c in cols})

    spend = 0.0
    for c in glob.glob(os.path.join(RESULTS, "*", "cost.json")):
        try:
            spend += json.load(open(c)).get("total_usd", 0.0)
        except Exception:
            pass
    done = len({os.path.dirname(p) for pat in ("judged.csv", "summary.md")
                for p in glob.glob(os.path.join(RESULTS, "*", pat))})
    meta = {"conditions_done": done, "conditions_total": 243,
            "runs_in_figure": runs, "spend_usd": round(spend, 2),
            "rows": len(out)}
    with open(os.path.join(RESULTS, "aggregate.json"), "w") as f:
        json.dump({"meta": meta, "rows": out}, f)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

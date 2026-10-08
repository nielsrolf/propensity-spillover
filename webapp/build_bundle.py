"""Build the blinded forecast bundle (inputs/) from the replication evals.

Source of truth: ../traits
(battery.yaml + <trait>/{trait.yaml, eval.yaml, system_prompts/}).

Output layout matches what server/agent.py (and stage.sh)
expect, so inputs/ + BLINDED_PROMPT.md can be dropped into that repo root:

  inputs/evals_orthogonalized/<trait>/<trait>_eval.yaml   list-of-items schema
  inputs/evals_orthogonalized/<trait>/system_prompts/{positive,negative}.txt
  inputs/evals_orthogonalized/definitions.json            all 24 traits
  inputs/evals_orthogonalized/_normalize_summary.csv      item counts per eval
  inputs/evals_orthogonalized/_judge_cossim_*.csv         (run judge_cossim.py)
  inputs/PREDICT_transfer_matrix_logitz_{plus,minus}.csv  empty templates

Conversion (replication eval.yaml -> old per-item schema):
  questions[i].split                -> meta.split
  questions[i].reference_responses  -> meta.expected_plus_response / expected_minus_response
                                       (the original bundle exposed the pole exemplars under
                                       meta.expected_*; same information, standardized names)
  eval-level judge_prompts          -> copied into every item's judge_prompts; the
                                       provides_evidence gate has {trait_name} /
                                       {trait_definition} filled in from eval meta
  judge.type                        -> judge_type ("sampling")
No results of any kind are read or written.
"""
import csv
import json
import os
import shutil

import yaml

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traits")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
EV = os.path.join(OUT, "evals_orthogonalized")


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # repeat judge prompts per item like the original
        return True


def _str_rep(dumper, s):
    style = "|" if "\n" in s else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)


_Dumper.add_representer(str, _str_rep)


def main():
    traits = yaml.safe_load(open(os.path.join(SRC, "battery.yaml")))["evals"]
    traits = sorted(traits)
    assert len(traits) == 24, traits
    if os.path.isdir(EV):
        for d in os.listdir(EV):
            p = os.path.join(EV, d)
            if os.path.isdir(p):
                shutil.rmtree(p)
    os.makedirs(EV, exist_ok=True)

    defs, summary, bipolar = {}, [], []
    for t in traits:
        tr = yaml.safe_load(open(os.path.join(SRC, t, "trait.yaml")))
        ev = yaml.safe_load(open(os.path.join(SRC, t, "eval.yaml")))
        assert tr["name"] == t and ev["eval"] == t
        neg = tr.get("negative") or {}
        has_neg = bool(neg.get("system_prompt"))
        if has_neg:
            bipolar.append(t)
        defs[t] = {
            "axis": t,
            "definition": tr["definition"].strip(),
            "plus_pole_name": tr["positive"]["name"],
            "plus_pole": tr["positive"]["description"].strip(),
            "minus_pole_name": neg.get("name"),
            "minus_pole": (neg.get("description") or "").strip() or None,
        }

        meta_ev = ev.get("meta") or {}
        judge = {}
        for k, v in ev["judge_prompts"].items():
            if k == "provides_evidence":
                v = v.replace("{trait_name}", meta_ev["trait_name"]).replace(
                    "{trait_definition}", meta_ev["trait_definition"].strip())
            judge[k] = v
        items = []
        for q in ev["questions"]:
            meta = {"split": q["split"]}
            meta.update(q.get("meta") or {})
            rr = q.get("reference_responses") or {}
            if rr.get("positive"):
                meta["expected_plus_response"] = rr["positive"]
            if rr.get("negative"):
                meta["expected_minus_response"] = rr["negative"]
            items.append({
                "id": q["id"],
                "paraphrases": q["paraphrases"],
                "samples_per_paraphrase": q.get("samples_per_paraphrase", 1),
                "temperature": q.get("temperature", 1.0),
                "judge_prompts": judge,
                "judge_type": ev["judge"].get("type", "sampling"),
                "meta": meta,
            })
        os.makedirs(os.path.join(EV, t, "system_prompts"))
        with open(os.path.join(EV, t, f"{t}_eval.yaml"), "w") as f:
            yaml.dump(items, f, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=10**6)
        for pole in ("positive", "negative"):
            src = os.path.join(SRC, t, "system_prompts", f"{pole}.txt")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(EV, t, "system_prompts", f"{pole}.txt"))
        n_tr = sum(q["split"] == "train" for q in ev["questions"])
        n_te = sum(q["split"] == "test" for q in ev["questions"])
        summary.append([t, len(items), n_tr, n_te,
                        sum(bool((q.get("reference_responses") or {}).get("positive")) for q in ev["questions"]),
                        sum(bool((q.get("reference_responses") or {}).get("negative")) for q in ev["questions"]),
                        has_neg])

    json.dump(defs, open(os.path.join(EV, "definitions.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(EV, "_normalize_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "n_rows", "n_train", "n_test", "n_plus_refs", "n_minus_refs", "has_minus_system_prompt"])
        w.writerows(summary)

    for pole, rows in (("plus", traits), ("minus", bipolar)):
        with open(os.path.join(OUT, f"PREDICT_transfer_matrix_logitz_{pole}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["treatment"] + traits)
            for r in rows:
                w.writerow([f"{r}-{pole}"] + [""] * len(traits))
    print(f"{len(traits)} traits, {len(bipolar)} bipolar: {bipolar}")
    print("unipolar:", [t for t in traits if t not in bipolar])


if __name__ == "__main__":
    main()

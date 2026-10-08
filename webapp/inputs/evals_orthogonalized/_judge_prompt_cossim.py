"""Cosine similarity of judge prompts across evals, with and without the shared evidence gate.

Each eval's items carry judge_prompts = {provides_evidence: <gate>, <metric>: <rubric>, ...}.
The gate is a separate first-stage call (same template for every eval, with the eval's trait
name and definition filled in); the metric rubrics are the eval-specific 0-100 scales.

One entry per (eval, metric rubric); the gate itself is not an entry.
`with`    = gate text + metric rubric (the full judging context for that metric)
`without` = metric rubric only (trait-specific portion)

Vectorization: TF-IDF (word 1-2 grams), cosine similarity. Deterministic, no API.
"""
import glob
import os

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.abspath(__file__))
GATE = "provides_evidence"


def load_entries():
    entries = []  # (label, full_text, body_text)
    for d in sorted(x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))):
        fs = sorted(glob.glob(os.path.join(ROOT, d, "*_eval*.yaml")))
        if not fs:
            continue
        jp = yaml.load(open(fs[0]), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))[0].get("judge_prompts", {})
        gate = jp.get(GATE, "")
        for metric, text in jp.items():
            if metric == GATE:
                continue
            entries.append((f"{d}/{metric}", (gate + "\n\n" + text).strip(), text.strip()))
    return entries


def save(labels, M, path):
    with open(path, "w") as f:
        f.write("label," + ",".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "," + ",".join(f"{v:.4f}" for v in M[i]) + "\n")


if __name__ == "__main__":
    entries = load_entries()
    labels = [e[0] for e in entries]

    def sim_matrix(texts):
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
        return cosine_similarity(X)

    S_with, S_without = sim_matrix([e[1] for e in entries]), sim_matrix([e[2] for e in entries])
    save(labels, S_with, os.path.join(ROOT, "_judge_cossim_with_preamble.csv"))
    save(labels, S_without, os.path.join(ROOT, "_judge_cossim_without_preamble.csv"))
    iu = np.triu_indices_from(S_with, k=1)
    print(f"{len(entries)} metric prompts; mean off-diag with={S_with[iu].mean():.3f} without={S_without[iu].mean():.3f}")

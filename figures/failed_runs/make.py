"""Appendix (replication): elicitation runs excluded because they did not move their own trait.

  on_target_runs_rep.png        on-target Δ of every run, failed runs highlighted
  failed_runs_table_rep.tex     compact table: failed runs per (method, model)

python figures/failed_runs/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
THRESHOLD = S.THRESHOLD
KEYS = S.KEYS

RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
NAME = {"qwen35": "Qwen3.5 9B", "qwen35-base": "Qwen3.5 9B Base", "nemotron35": "Nemotron"}
LONG = {"qwen35": "Qwen3.5-9B", "qwen35-base": "Qwen3.5-9B-Base", "nemotron35": "Nemotron-3-Super-120B"}
ML = {"sft": "SFT", "sp": "SP", "grpo": "GRPO", "icl": "ICL"}


def figure(d):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    lo = min(d.on_target.min(), 0) - 0.35
    for i, (m, mo) in enumerate(KEYS):
        g = d[(d.method == m) & (d.model == mo)]
        x = i + rng.uniform(-0.2, 0.2, len(g))
        ok = ~g.failed.values
        ax.scatter(x[ok], g.on_target[ok], s=10, color="#2a78d6", alpha=.7, linewidths=0)
        ax.scatter(x[~ok], g.on_target[~ok], s=14, color="#eb6834", alpha=.9, linewidths=0)
        ax.text(i, lo + 0.1, f"{(~ok).sum()}/{len(g)}", ha="center", fontsize=7, color="#57606a")
    ax.axhline(THRESHOLD, color="#1f2328", ls="--", lw=1)
    ax.axhline(0, color="#57606a", lw=.6)
    ax.text(len(KEYS) - 0.45, THRESHOLD - 0.05, f"threshold {THRESHOLD}", ha="right", va="top", fontsize=7)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([f"{ML[m]}\n{NAME[mo]}" for m, mo in KEYS], fontsize=7)
    ax.set_ylabel("on-target Δ, intended direction\n(logit-z)", fontsize=8)
    ax.set_ylim(lo, None)
    ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "on_target_runs_rep.png", dpi=200)
    print("wrote", OUT / "on_target_runs_rep.png")


def table(d):
    f = d[d.failed]
    lines = [r"\begin{tabular}{p{0.11\linewidth}p{0.25\linewidth}p{0.57\linewidth}}", r"\toprule",
             r"Method & Model & Failed runs (plus pole) \\", r"\midrule"]
    for m, mo in KEYS:
        g = f[(f.method == m) & (f.model == mo)].sort_values("trait")
        n = ((d.method == m) & (d.model == mo)).sum()
        items = ", ".join(g.trait) or "none"
        lines.append(f"{ML[m]} & {LONG[mo]} ({len(g)}/{n}) & {items} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "failed_runs_table_rep.tex").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "failed_runs_table_rep.tex")


if __name__ == "__main__":
    d = pd.read_csv(RES / "on_target.csv")
    figure(d)
    table(d)
    print(f"failed {d.failed.sum()} / {len(d)}")
    print(d.groupby(["method", "model"]).on_target.agg(["median", "min", "max"]).round(2))

"""§4.4 (Fig. 7): observed SFT spillover by human-psychology (H9) prediction level.
Failed runs (on-target Δ < 0.2, results/derived/failed_runs.csv) are dropped as rows.

  h9_prediction_boxplot_rep.png            observed SFT spillover (3 models pooled) by H9 prediction level

    python figures/h9_human_psychology/boxplot.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
B = S
KEYS = S.KEYS
NAME = S.NAME
RES = S.DERIVED
failed_set = S.failed_set
load = S.load
from lib.spill import BLUE, INK, MUTED, ORANGE, AQUA

OUT = pathlib.Path(__file__).resolve().parent
MLAB = {"sft": "SFT", "sp": "SP", "grpo": "GRPO", "icl": "ICL"}
MLONG = {"sft": "SFT", "sp": "System prompt", "grpo": "GRPO", "icl": "ICL"}
MCOL = {"qwen35": BLUE, "qwen35-base": AQUA, "nemotron35": ORANGE}
SFT_MODELS = ("qwen35", "qwen35-base", "nemotron35")


def short(t):
    return t.replace("ethical-framework-", "ethics: ")


def h9_boxplot():
    H9 = S.load_claude()["H9"]
    load_minus = S.load_minus
    levels = [-0.8, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 0.8]
    xs, ys, mods, per = [], [], [], {}
    for mod in SFT_MODELS:
        pv, ov = B.pairs(H9["plus"], load("sft", mod))
        mm = load_minus(mod)
        if mm is not None and "minus" in H9:
            pv2, ov2 = B.pairs(H9["minus"], mm)
            pv, ov = np.r_[pv, pv2], np.r_[ov, ov2]
        per[mod] = (spearmanr(pv, ov)[0], len(pv))
        xs += list(pv)
        ys += list(ov)
        mods += [mod] * len(pv)
    xs, ys, mods = np.array(xs), np.array(ys), np.array(mods)
    rho = spearmanr(xs, ys)[0]
    idx = np.array([int(np.argmin([abs(x - l) for l in levels])) for x in xs])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = [ys[idx == i] for i in range(len(levels))]
    ax.boxplot(data, positions=range(len(levels)), widths=0.6, showfliers=False,
               medianprops=dict(color=INK), boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED))
    rng = np.random.default_rng(0)
    for mod in SFT_MODELS:
        m = mods == mod
        ax.scatter(idx[m] + rng.uniform(-0.2, 0.2, m.sum()), ys[m], s=4, alpha=0.4, color=MCOL[mod],
                   linewidths=0, label=f"{NAME[mod]} (ρ = {per[mod][0]:.2f})")
    med = [np.median(d) if len(d) else np.nan for d in data]
    ax.plot(range(len(levels)), med, color=INK, marker="o", ms=3, lw=1.2, label="median")
    lo, hi = ys.min(), ys.max()
    pad = 0.08 * (hi - lo)
    for i, d in enumerate(data):
        ax.text(i, lo - pad, f"n={len(d)}", ha="center", va="top", fontsize=6, color=MUTED)
    ax.set_ylim(lo - 2.2 * pad, hi + 3.2 * pad)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["≤−0.8"] + [f"{l:g}" for l in levels[1:-1]] + ["≥0.8"], fontsize=7)
    ax.set_xlabel("Human-psychology (H9) predicted spillover", fontsize=8)
    ax.set_ylabel("observed SFT spillover (Δ logit-z)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(0, color="#e6e6e3", lw=0.8, zorder=0)
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}\nn = {len(xs)}", transform=ax.transAxes, va="top",
            fontsize=7.5, bbox=dict(facecolor="white", edgecolor="#e6e6e3"))
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", markerscale=3, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "h9_prediction_boxplot_rep.png", dpi=200, bbox_inches="tight")
    print("H9 boxplot pooled rho", round(rho, 3), "n", len(xs),
          "per model:", {m: (round(r, 3), n) for m, (r, n) in per.items()})
    print("bin sizes", [len(d) for d in data], "medians", np.round(med, 3))


if __name__ == "__main__":
    h9_boxplot()

"""Appendix F: SP / ICL / SFT / GRPO spillover matrices of Qwen3.5-9B.
Failed runs (on-target Δ < 0.2, results/derived/failed_runs.csv) are dropped as rows.

  method_heatmaps_rep.png                  SP / ICL / SFT / GRPO matrices, Qwen3.5-9B

    python figures/method_heatmaps/make.py
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


def method_heatmaps(vmax=2.5):
    mod = "qwen35"
    ms = {m: load(m, mod, drop_failed=False) for m in ("sp", "icl", "sft", "grpo")}
    traits = list(ms["sft"].columns)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.06)
    for ax, (meth, m) in zip(axes, ms.items()):
        v = m.reindex(index=traits, columns=traits)
        bad = failed_set(meth, mod)
        im = ax.imshow(v.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        n = len(traits)
        for i, t in enumerate(traits):
            if t in bad:
                ax.add_patch(Rectangle((-0.5, i - 0.5), n, 1, facecolor="#d0d0cc", edgecolor="#8c8c86",
                                       hatch="////", lw=0, zorder=2))
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.0, zorder=3))
        ax.set_xticks(range(n))
        ax.set_xticklabels([short(t) for t in traits], rotation=90, fontsize=6.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([short(t) for t in traits], fontsize=6.5)
        ax.set_title(f"{MLONG[meth]}, Qwen3.5-9B  ({len(bad)} failed)", fontsize=9, loc="left")
        ax.set_xlabel("evaluated trait", fontsize=8)
    axes[0].set_ylabel("elicited trait", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.01, extend="both")
    cb.set_label(f"Δ logit-z vs. base (clipped at ±{vmax})", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.legend(handles=[Patch(facecolor="#d0d0cc", edgecolor="#8c8c86", hatch="////",
                              label="failed run (on-target Δ < 0.2), excluded"),
                        Patch(facecolor="none", edgecolor=INK, label="on-target cell")],
               loc="lower left", bbox_to_anchor=(0.125, 0.80), ncol=2, fontsize=8, frameon=False)
    fig.savefig(OUT / "method_heatmaps_rep.png", dpi=200, bbox_inches="tight")
    print("wrote method_heatmaps_rep.png")
    # cross-method agreement of off-diagonal cells (failed rows dropped)
    from itertools import combinations
    for a, b in combinations(ms, 2):
        pv, ov = B.pairs(load(a, mod), load(b, mod))
        print(f"  rho({a},{b}) = {spearmanr(pv, ov)[0]:.3f}  n={len(pv)}")


if __name__ == "__main__":
    method_heatmaps()

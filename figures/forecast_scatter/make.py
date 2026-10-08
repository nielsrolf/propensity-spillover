"""§4.1 scatter: observed SFT spillover vs. predicted spillover, one panel per predictor
(SP / ICL / GRPO on the same model, SFT on the other models), one colour per model.
One dot = one off-diagonal (A, B) cell of one model; failed elicitations excluded.

python figures/forecast_scatter/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
B = S

RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
MODELS = {"qwen35": ("Qwen3.5-9B", "#2a78d6"), "qwen35-base": ("Qwen3.5-9B-Base", "#1baf7a"),
          "nemotron35": ("Nemotron-3-Super-120B", "#eb6834")}
PANELS = [("sp", "System prompt (same model)"), ("icl", "ICL (same model)"),
          ("grpo", "GRPO (same model)"), ("sft-other", "SFT (other two models)")]
INK, MUTED = "#1f2328", "#57606a"


def load():
    failed = pd.read_csv(RES / "failed_runs.csv")
    M = {}
    for f in RES.glob("delta_*_plus.csv"):
        _, meth, mod, _ = f.stem.split("_")
        m = pd.read_csv(f, index_col=0)
        bad = set(failed[(failed.method == meth) & (failed.model == mod)].trait)
        M[(meth, mod)] = m.drop(index=[t for t in m.index if t in bad])
    return M


def main():
    M = load()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    lim = 0
    for ax, (key, title) in zip(axes, PANELS):
        allp, allo = [], []
        for mod, (name, col) in MODELS.items():
            if ("sft", mod) not in M:
                continue
            target = M[("sft", mod)]
            if key == "sft-other":
                pred = B.mean_of([M[("sft", o)] for o in MODELS if o != mod])
            elif (key, mod) in M:
                pred = M[(key, mod)]
            else:
                continue
            pv, ov = B.pairs(pred, target)
            ax.scatter(pv, ov, s=7, alpha=0.45, color=col, linewidths=0, label=name)
            allp += list(pv)
            allo += list(ov)
            lim = max(lim, np.nanpercentile(np.abs(np.r_[pv, ov]), 99.5))
        rho = spearmanr(allp, allo)[0]
        nmod = sum(1 for mod in MODELS if key == "sft-other" or (key, mod) in M)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.text(0.03, 0.97, f"ρ = {rho:.2f}\nn = {len(allp)} cells, {nmod} models", transform=ax.transAxes,
                va="top", fontsize=7.5, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("predicted spillover $\\hat S$ (logit-z)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], ls=":", color=MUTED, lw=0.8)
    axes[0].set_ylabel("observed SFT spillover $S$ (logit-z)", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=n, markersize=5) for n, c in MODELS.values()]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / "forecast_scatter_sft_rep.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "forecast_scatter_sft_rep.png")


if __name__ == "__main__":
    main()

"""Fig. 2 (replication): one dot per successful elicitation run, on-target |S| vs. mean
off-target |S|, colour = method, marker = model. Reference lines: y = x and y = median ratio · x.

python figures/on_off_target/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
COL = {"sft": "#2a78d6", "sp": "#eb6834", "icl": "#e87ba4", "grpo": "#1baf7a"}
MLAB = {"sft": "SFT", "sp": "System prompt", "icl": "ICL", "grpo": "GRPO"}
MARK = {"qwen35": ("o", "Qwen3.5-9B"), "qwen35-base": ("s", "Qwen3.5-9B-Base"), "nemotron35": ("^", "Nemotron-3-Super-120B")}
MUTED = "#57606a"


def runs():
    failed = pd.read_csv(RES / "failed_runs.csv")
    rows = []
    for f in RES.glob("delta_*_plus.csv"):
        _, meth, mod, _ = f.stem.split("_")
        m = pd.read_csv(f, index_col=0)
        bad = set(failed[(failed.method == meth) & (failed.model == mod)].trait)
        for t in m.index:
            if t in bad or t not in m.columns:
                continue
            rows.append(dict(method=meth, model=mod, trait=t, pole="plus", on=abs(m.at[t, t]),
                             off=m.loc[t].drop(t).abs().mean()))
    # minus-pole SFT runs (symmetry check), same scale; failed ones excluded
    mo = pd.read_csv(RES / "on_target_minus.csv") if (RES / "on_target_minus.csv").exists() else None
    for f in RES.glob("delta_sft_*_minus.csv"):
        mod = f.stem.split("_")[2]
        m = pd.read_csv(f, index_col=0)
        bad = set(mo[(mo.model == mod) & mo.failed].trait) if mo is not None else set()
        for t in m.index:
            if t not in bad:
                rows.append(dict(method="sft", model=mod, trait=t, pole="minus", on=abs(m.at[t, t]),
                                 off=m.loc[t].drop(t).abs().mean()))
    return pd.DataFrame(rows)


def main():
    d = runs()
    ratio = (d.off / d.on).median()
    rho = spearmanr(d.on, d.off)[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for (meth, mod, pole), g in d.groupby(["method", "model", "pole"]):
        if pole == "plus":
            ax.scatter(g.on, g.off, s=22, marker=MARK[mod][0], color=COL[meth], alpha=0.75,
                       edgecolor="white", linewidth=0.5)
        else:
            ax.scatter(g.on, g.off, s=26, marker=MARK[mod][0], facecolor="none", edgecolor=COL[meth],
                       linewidth=1.1)
    lim = d.on.max() * 1.05
    ax.plot([0, lim], [0, lim], ls=":", color=MUTED, lw=1)
    ax.text(d.off.max() * 1.08, d.off.max() * 1.02, "y = x", fontsize=7, color=MUTED, ha="left")
    ax.plot([0, lim], [0, ratio * lim], color=MUTED, lw=1.2)
    ax.text(lim * 0.98, ratio * lim * 0.98 - 0.08, f"median off/on ratio {ratio:.2f}", ha="right", fontsize=7, color=MUTED)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, d.off.max() * 1.25)
    ax.set_xlabel("on-target |S| (elicitation strength, logit-z)", fontsize=8)
    ax.set_ylabel("mean off-target |S| (spillover, logit-z)", fontsize=8)
    ax.set_title(f"{len(d)} elicitation runs ({(d.pole == 'minus').sum()} minus pole), Spearman ρ = {rho:.2f}",
                 fontsize=8, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.grid(color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], marker="s", ls="", color=c, label=MLAB[m]) for m, c in COL.items()]
    h += [plt.Line2D([], [], marker=mk, ls="", color=MUTED, label=n) for mk, n in MARK.values()]
    h += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=COL["sft"],
                     label="SFT, minus pole")]
    fig.legend(handles=h, fontsize=7, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(OUT / "fig2_on_off_runs_rep.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "fig2_on_off_runs_rep.png", f"ratio={ratio:.3f} rho={rho:.3f} n={len(d)}")


if __name__ == "__main__":
    main()

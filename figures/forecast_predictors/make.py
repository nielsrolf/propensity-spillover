"""Forecasting figures for the same-model replication (plus pole, pooled over models).

  forecast_sft_rep.png               main text §4.1: predictors of SFT spillover
  forecast_cross_method_rep.png      appendix: every method as target
  forecast_per_model_rep.png         appendix: SFT target per model
  forecast_claude_conditions_rep.png §4.2: Claude conditions on the SFT target

python figures/forecast_predictors/make.py
"""
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
from lib.spill import AQUA, BLUE, CLAUDE, GRAY, INK, MUTED, ORANGE

RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
DRIFT_COL = "#6e6e6a"
MODEL_LABEL = {"qwen35": "Qwen3.5-9B", "qwen35-base": "Qwen3.5-9B-Base", "nemotron35": "Nemotron-3-Super-120B",
               "pooled": "all models"}
METHOD_LABEL = {"sft": "SFT", "sp": "System prompt", "icl": "ICL", "grpo": "GRPO"}
SHOW_CEILING = False   # open question: the ceiling covers measurement noise only
BEST_CLAUDE = "H4"   # best non-repeated condition on the pooled SFT target (Opus 5.5 forecasts)


def kind(p, tm):
    if p.startswith("Drift"):
        return 3, DRIFT_COL
    if p.startswith("Symmetry"):
        return 2, "#8a3ad6"
    if p.startswith("Claude"):
        return 4, ORANGE
    if p.startswith(METHOD_LABEL[tm] + " ("):
        return 1, AQUA
    return 0, BLUE


def label(r):
    p = r.predictor
    if r.target_model == "pooled" and isinstance(r.get("models"), str):
        k = len(r.models.split(","))
        if k < 3:
            p += f" [{k} of 3 models]"
    return p


def prep(d):
    d = d.copy()
    best = d[d.predictor == f"claude:{BEST_CLAUDE}"].assign(predictor=f"Claude Opus 5.5 ({BEST_CLAUDE.split('_')[0]}, best condition)")
    d = pd.concat([d[~d.predictor.str.startswith("claude:")], best])
    return d


def bars(ax, d, tm):
    d = prep(d)
    d["k"] = d.predictor.map(lambda p: kind(p, tm)[0])
    d = d.sort_values(["k", "rho"], ascending=[True, False]).reset_index(drop=True)
    y, prev, ticks, kinds = 0, None, [], set()
    for _, r in d.iterrows():
        if prev is not None and r.k != prev and r.k >= 2:
            y += 0.4
        c = kind(r.predictor, tm)[1]
        kinds.add(r.k)
        ax.barh(y, r.rho, height=0.7, color=c)
        if pd.notna(r.lo):
            ax.plot([r.lo, r.hi], [y, y], color=INK, lw=1)
        ticks.append((y, label(r)))
        prev, y = r.k, y + 1
    if SHOW_CEILING:
        ax.axvline(d.ceiling.iloc[0], color=INK, ls="--", lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=7.5)
    ax.set_ylim(y - 0.3, -0.7)
    ax.set_xlim(-0.2, 1.0)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("Spearman ρ with observed spillover", fontsize=7.5)
    return kinds


def legend(fig, kinds, y=0):
    lab = {0: (BLUE, "Other elicitation method, same model"), 1: (AQUA, "Same method, other models"),
           2: ("#8a3ad6", "Symmetry (mirrored B→A; floor-filtered)"), 3: (DRIFT_COL, "Drift"),
           4: (ORANGE, "Claude Opus 5.5 (best condition)")}
    seen, h = set(), []
    for k in sorted(kinds):
        c, l = lab[k]
        if l not in seen:
            seen.add(l)
            h.append(Patch(facecolor=c, label=l))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, y))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    print("wrote", OUT / name)


def panels(lb, targets, name, ncol=2, w=4.8, h=2.9):
    nrow = -(-len(targets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    for ax in axes.flat[len(targets):]:
        ax.axis("off")
    kinds = set()
    for ax, (tm, mod) in zip(axes.flat, targets):
        d = lb[(lb.target_method == tm) & (lb.target_model == mod)]
        kinds |= bars(ax, d, tm)
        ax.set_title(f"Target: {METHOD_LABEL[tm]}, {MODEL_LABEL[mod]}", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(rect=[0, 0.35 / (h * nrow), 1, 1])
    legend(fig, kinds)
    save(fig, name)


def claude_conditions(lb, name):
    d = lb[(lb.target_method == "sft") & (lb.target_model == "pooled")].set_index("predictor")
    pm = RES / "leaderboard_claude_pm.csv"
    if pm.exists():  # Claude conditions scored on plus AND minus SFT cells; reference lines stay plus-only
        c = pd.read_csv(pm).set_index("predictor")
        d = pd.concat([d[~d.index.str.startswith("claude:")], c])
    order = d[d.index.str.startswith("claude:")].sort_values("rho", ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for y, p in enumerate(order):
        k = p.split(":", 1)[1]
        c = ORANGE if k.startswith("H7b") else AQUA if k.startswith("H7a") else GRAY
        ax.barh(y, d.loc[p].rho, height=0.7, color=c)
        ax.plot([d.loc[p].lo, d.loc[p].hi], [y, y], color=INK, lw=1)
    for nm, c, lab, yt in (("System prompt (same model)", BLUE, "system prompt", -0.95),
                           ("SFT (other models)", "#1baf7a", "SFT, other models", -1.55)):
        ax.axvline(d.loc[nm].rho, color=c, lw=1.5)
        ax.text(d.loc[nm].rho + 0.01, yt, lab, fontsize=7, color=c, va="center")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CLAUDE[p.split(":", 1)[1]] for p in order], fontsize=8)
    ax.set_ylim(len(order) - 0.4, -1.9)
    ax.set_xlim(-0.15, 1.0)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Spearman ρ with observed SFT spillover", fontsize=8)
    h = [Patch(facecolor=GRAY, label="Theory-grounded"), Patch(facecolor=AQUA, label="Anything goes"),
         Patch(facecolor=ORANGE, label="Intuition")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, name)


if __name__ == "__main__":
    lb = pd.read_csv(RES / "leaderboard_all.csv")
    panels(lb, [("sft", "pooled")], "forecast_sft_rep.png", ncol=1, w=6.4, h=3.6)
    panels(lb, [(m, "pooled") for m in ("sft", "sp", "icl", "grpo")], "forecast_cross_method_rep.png")
    panels(lb, [("sft", m) for m in ("qwen35", "qwen35-base", "nemotron35")], "forecast_per_model_rep.png",
           ncol=3, w=4.4)
    claude_conditions(lb, "forecast_claude_conditions_rep.png")

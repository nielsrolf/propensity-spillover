"""§4.1.1 symmetry check (SFT, replication).

Pairs: strong SFT spillover from a malevolent trait A onto B (mean over the 3 models
|S| ≥ 0.5, same sign in all 3, reverse run available in all 3). Rows grouped by A; rows where
a benign elicitation of B raises A are highlighted. Symmetry predicts:
  more A → less B   ⇒  less B → more A   (reverse uses B's minus-pole SFT run)
  more A → more B   ⇒  more B → more A   (reverse uses B's plus-pole SFT run)
Each row is labelled "A ↑ ↔ B ↓/↑" with A in blue and B in red. Each bar has the colour
of the elicited trait and shows its effect on the other trait in the direction of that
trait's arrow, so symmetric spillover means two equally long positive bars.
Bars: mean over models; dots: individual models.

Writes results/derived/symmetry_pairs.csv (columns A, B, model, sign,
fwd = S(A+→B), rev = S(B±→A)).

python figures/symmetry/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
BLUE, RED, INK, MUTED = "#2a78d6", "#d6452a", "#1f2328", "#57606a"
HL = "#fbe8a6"
MODELS = ("qwen35", "qwen35-base", "nemotron35")
# malevolent traits (plus pole; harm-refusal plus = compliance with harmful requests)
MALEVOLENT = ["spitefulness", "power-seeking", "resource-acquisition", "narcissism", "self-preservation",
              "punitiveness", "harm-refusal"]
# raising these is not counted as benign either (dishonest self-presentation / flattery)
NOT_BENIGN_UP = set(MALEVOLENT) | {"sycophancy", "claiming-superintelligence"}
PROSOCIAL = {"agreeableness", "caring-about-user", "caring-about-animals", "honest-humble", "cooperation"}
MIN_FWD = 0.5      # mean |S(A+→B)| over the three models
MIN_RAISE = 0.2    # benign elicitation counts as raising A if mean reverse effect exceeds this


def benign(b, sign):
    """Eliciting b in its arrow direction is benign: not raising a malevolent trait (or
    sycophancy / claiming superintelligence) and not lowering a prosocial one."""
    return b not in NOT_BENIGN_UP if sign > 0 else b not in PROSOCIAL


def build_pairs():
    load = S.load
    load_minus = S.load_minus
    T = {m: load("sft", m) for m in MODELS}
    Tm = {m: load_minus(m) for m in MODELS}
    rows = []
    for a in MALEVOLENT:
        for b in T[MODELS[0]].columns:
            if a == b:
                continue
            rec = []
            for m in MODELS:
                t, tm = T[m], Tm[m]
                if a not in t.index or b not in t.columns or pd.isna(t.at[a, b]):
                    break
                fwd = t.at[a, b]
                src = t if fwd > 0 else tm
                if src is None or b not in src.index or a not in src.columns or pd.isna(src.at[b, a]):
                    break
                rec.append(dict(A=a, B=b, model=m, sign=np.sign(fwd), fwd=fwd, rev=src.at[b, a]))
            if len(rec) == len(MODELS) and len({r["sign"] for r in rec}) == 1 \
                    and np.mean([abs(r["fwd"]) for r in rec]) >= MIN_FWD:
                rows += rec
    d = pd.DataFrame(rows)
    d.to_csv(RES / "symmetry_pairs.csv", index=False)
    return d


def main():
    d = build_pairs()
    d["f"] = d.fwd.abs()          # effect of A on B, in the direction of B's arrow
    d["r"] = d.rev                # effect of B (in its arrow direction) on A; + = A goes up
    g = d.groupby(["A", "B", "sign"]).agg(f=("f", "mean"), r=("r", "mean")).reset_index()
    # one row per unordered pair: keep the direction with the larger forward effect
    g["key"] = [tuple(sorted(x)) for x in zip(g.A, g.B)]
    g = g.sort_values("f", ascending=False).drop_duplicates("key")
    g["hl"] = [benign(b, sg) and r > MIN_RAISE for b, sg, r in zip(g.B, g.sign, g.r)]
    # group by the blue (malevolent) trait; groups ordered by size, rows by forward effect
    gsize = g.groupby("A").size()
    g["gs"] = g.A.map(gsize)
    g = g.sort_values(["gs", "A", "f"], ascending=[True, True, True]).reset_index(drop=True)
    d = d.merge(g[["A", "B"]], on=["A", "B"])

    # transposed layout: pairs along x, grouped by the malevolent (blue) trait; largest group first
    g = g.sort_values(["gs", "A", "f"], ascending=[False, True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.38
    for i, row in g.iterrows():
        sub = d[(d.A == row.A) & (d.B == row.B)]
        if row.hl:
            ax.axvspan(i - 0.5, i + 0.5, color=HL, zorder=0, lw=0)
        ax.bar(i - w / 2, row.f, width=w, color=BLUE)
        ax.bar(i + w / 2, row.r, width=w, color=RED)
        ax.scatter([i - w / 2] * len(sub), sub.f, s=6, color=INK, zorder=3)
        ax.scatter([i + w / 2] * len(sub), sub.r, s=6, color=INK, zorder=3)
    starts = [0] + [i for i in range(1, len(g)) if g.A[i] != g.A[i - 1]] + [len(g)]
    for k in range(len(starts) - 1):
        lo, hi = starts[k], starts[k + 1]
        if lo > 0:
            ax.axvline(lo - 0.5, color=MUTED, lw=0.6, ls=":")
        name = g.A[lo] if hi - lo > 3 or len(g.A[lo]) < 13 else g.A[lo].replace("-", "-\n", 1)
        ax.text((lo + hi - 1) / 2, 1.0, f"{name} ↑", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.5, color=BLUE, fontweight="bold")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{b} {'↓' if sg < 0 else '↑'}" for b, sg in zip(g.B, g.sign)], rotation=60,
                       ha="right", fontsize=7, color=RED)
    ax.set_xlim(-0.6, len(g) - 0.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("effect on the other trait\nin the direction of its arrow (logit-z)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(color=BLUE, label="elicit blue (malevolent) trait ↑, effect on red trait"),
               Patch(color=RED, label="elicit red trait in its arrow direction, effect on blue trait"),
               Patch(color=HL, label="benign elicitation raises a malevolent trait"),
               Line2D([], [], marker="o", ls="", color=INK, markersize=3, label="individual models")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "symmetry_minus_rep.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "symmetry_minus_rep.png")
    print(g.drop(columns=["key", "gs"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

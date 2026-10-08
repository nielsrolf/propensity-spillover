"""§4.3: do models predict their own SFT spillover?

Per model, each ordered pair (A, B) gets a self-predicted direction: the mean of
up=+1 / same=0 / down=−1 over all rollouts of all three framings (direct, persona,
shown training data). Compared against the model's observed SFT spillover S(A→B)
(replication, logit-z; failed elicitations excluded).

Figure per model: distribution of observed S for pairs predicted "up" vs "down"
(|mean| ≥ 1/3), with group means; Spearman ρ of mean prediction vs. S overall and
per framing.

python figures/introspection/make.py
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
RES = S.DERIVED
INTRO = S.INTROSPECTION
OUT = pathlib.Path(__file__).resolve().parent
MODELS = {"qwen35": ("qwen3.5-9b", "Qwen3.5-9B"), "qwen35-base": ("qwen3.5-9b-base", "Qwen3.5-9B-Base"),
          "nemotron35": ("nemotron-3-super", "Nemotron-3-Super-120B")}
FRAMINGS = {"prompt0": "direct question", "prompt12": "persona reflection first",
            "prompt_data": "shown training data"}
VAL = {"up": 1, "same": 0, "down": -1}
UP, DOWN = "#2a78d6", "#eb6834"


def predictions(short):
    """Self-predicted directions; pole = which SFT pole of A the question was about."""
    rows = []
    for fr in FRAMINGS:
        for pole, sfx in (("plus", ""), ("minus", "_minus")):
            p = INTRO / short / f"{fr}{sfx}.csv"
            if p.exists():
                for r in pd.read_csv(p).to_dict("records"):
                    if r.get("consensus") in VAL:
                        rows.append(dict(framing=fr, pole=pole, A=r["propensity1"], B=r["propensity2"],
                                         v=VAL[r["consensus"]]))
    return pd.DataFrame(rows)


def observed(key):
    load_minus = S.load_minus
    m = pd.read_csv(RES / f"delta_sft_{key}_plus.csv", index_col=0)
    f = pd.read_csv(RES / "failed_runs.csv")
    bad = set(f[(f.method == "sft") & (f.model == key)].trait)
    out = []
    for pole, mat in (("plus", m.drop(index=[t for t in m.index if t in bad])), ("minus", load_minus(key))):
        if mat is None:
            continue
        s = mat.stack().rename("S").reset_index()
        s.columns = ["A", "B", "S"]
        out.append(s[s.A != s.B].assign(pole=pole))
    return pd.concat(out)


def analyse(key):
    short, name = MODELS[key]
    p = predictions(short)
    obs = observed(key)
    stats = {"model": name, "n_answers": len(p)}
    for fr in list(FRAMINGS) + ["all"]:
        q = p if fr == "all" else p[p.framing == fr]
        d = q.groupby(["A", "B", "pole"]).v.mean().rename("pred").reset_index().merge(obs, on=["A", "B", "pole"])
        stats[f"rho_{fr}"] = spearmanr(d.pred, d.S)[0] if len(d) > 10 else np.nan
        if fr == "all":
            full = d
    full["dir"] = np.where(full.pred >= 1 / 3, "up", np.where(full.pred <= -1 / 3, "down", "same"))
    for k in ("up", "down", "same"):
        stats[f"n_{k}"] = int((full.dir == k).sum())
        stats[f"mean_S_{k}"] = full.loc[full.dir == k, "S"].mean()
    return full, stats


def plot(full, stats, fname):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    xs = np.linspace(full.S.quantile(0.005) - 0.3, full.S.quantile(0.995) + 0.3, 300)
    for k, c in (("up", UP), ("down", DOWN)):
        v = full.loc[full.dir == k, "S"]
        if len(v) > 5:
            y = gaussian_kde(v)(xs)
            ax.fill_between(xs, y, alpha=0.25, color=c)
            ax.plot(xs, y, color=c, lw=1.5, label=f"predicted {'↑' if k == 'up' else '↓'} (n={len(v)}, mean {v.mean():+.2f})")
            ax.axvline(v.mean(), color=c, ls="--", lw=1)
    ax.axvline(0, color="#57606a", lw=0.6)
    ax.set_xlabel("observed SFT spillover $S(A\\to B)$ (logit-z)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"{stats['model']}: Spearman ρ(prediction, S) = {stats['rho_all']:.2f}", fontsize=8.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    print("wrote", OUT / fname)


if __name__ == "__main__":
    rows = []
    for key in MODELS:
        full, st = analyse(key)
        rows.append(st)
        plot(full, st, f"introspection_{key}_rep.png")
    t = pd.DataFrame(rows)
    t.to_csv(RES / "introspection_summary.csv", index=False)
    print(t.round(3).to_string(index=False))

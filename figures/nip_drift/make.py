"""§4.7 (replication): how much do traits drift under fine-tuning on neutral data?

Per model, three SFT runs on "nothing in particular" (NIP) datasets (nip1-3, 50 everyday
Q/A each) with the sweep's SFT recipe. For each trait: the NIP shift S_NIP(B) (logit-z vs.
the model's baseline, same units as the spillover matrices), shown as the mean of
the three runs with min–max whiskers, next to the on-target SFT effect S(B→B) and the SFT
drift of B (mean spillover onto B over all other successfully elicited traits A ≠ B).

  nip_shift_rep.png        Qwen3.5-9B (main text)
  nip_shift_rep_other.png  Qwen3.5-9B-Base and Nemotron-3-Super-120B (appendix)

python figures/nip_drift/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
B = S
R = S
load = S.load

RES = S.DERIVED
OUT = pathlib.Path(__file__).resolve().parent
MODELS = {"qwen35": ("qwen3.5-9b", "Qwen3.5-9B"), "qwen35-base": ("qwen3.5-9b-base", "Qwen3.5-9B-Base"),
          "nemotron35": ("nemotron-3-super", "Nemotron-3-Super-120B")}
NIP_COL, SFT_COL, DRIFT_COL, MUTED = "#2a78d6", "#eb6834", "#2f9e6e", "#57606a"


def nip_shifts():
    F = R.long_frames()
    rows = []
    for key, (short, _) in MODELS.items():
        el, base = F[("sft", key)]
        panel = pd.concat([el, base])
        sig = panel.assign(l=B.empirical_logit(panel.score.values)).groupby("eval")["l"].std()
        nips = []
        for d in ("nip1", "nip2", "nip3"):
            nips.append(S.run_scores(f"{short}-sft-nip-{d}").assign(model=f"{d}-plus"))
        m = B.delta_matrices(pd.concat(nips), base, sig)["plus"]
        sft = B.delta_matrices(el, base, sig)["plus"]
        ok = load("sft", key)  # failed elicitation rows dropped
        for t in m.columns:
            drift = ok.loc[[a for a in ok.index if a != t], t].mean() if t in ok.columns else np.nan
            v = m[t].values
            rows.append(dict(model=key, trait=t, nip_max=v[np.argmax(np.abs(v))], nip_lo=v.min(), nip_hi=v.max(),
                             nip_mean=v.mean(), on_target=sft.at[t, t] if t in sft.index else np.nan, drift=drift))
    return pd.DataFrame(rows)


def plot(d, order, keys, fname, height):
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, key in zip(axes, keys):
        name = MODELS[key][1]
        g = d[d.model == key].set_index("trait").loc[order]
        x = np.arange(len(order))
        ax.errorbar(x, g.nip_mean, yerr=[g.nip_mean - g.nip_lo, g.nip_hi - g.nip_mean], fmt="o", ms=4.2,
                    color=NIP_COL, ecolor=NIP_COL, elinewidth=1.1, capsize=2.5, zorder=3,
                    label="nothing-in-particular SFT (mean of 3 runs; whiskers: min–max)")
        ax.scatter(x, g.drift, color=DRIFT_COL, marker="s", s=16, zorder=3, label="drift: mean SFT spillover onto the trait from other traits")
        ax.scatter(x, g.on_target, color=SFT_COL, marker="D", s=18, zorder=3, label="SFT on that trait (on-target)")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(name, fontsize=8.5, loc="left")
        ax.set_ylabel("Δ vs. baseline\n(logit-z)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.06 if len(keys) == 1 else 0.04, 1, 1])
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    print("wrote", OUT / fname)


def main():
    d = nip_shifts()
    d.to_csv(RES / "nip_shifts.csv", index=False)
    order = d.groupby("trait").nip_mean.apply(lambda x: x.abs().mean()).sort_values(ascending=False).index
    plot(d, order, ["qwen35"], "nip_shift_rep.png", 4.2)
    plot(d, order, ["qwen35-base", "nemotron35"], "nip_shift_rep_other.png", 6.4)
    from scipy.stats import spearmanr
    for key in MODELS:
        g = d[d.model == key].dropna(subset=["drift"])
        print(key, "rho(NIP mean, drift) =", round(spearmanr(g.nip_mean, g.drift)[0], 2), "n", len(g),
              "median |drift|", round(g.drift.abs().median(), 2))
    s = d.assign(a=d.nip_mean.abs(), o=d.on_target.abs())
    print(s.groupby("model").agg(median_nip=("a", "median"), max_nip=("a", "max"), median_on=("o", "median")).round(2))
    print(s.sort_values("a", ascending=False).head(8)[["model", "trait", "nip_mean", "nip_lo", "nip_hi"]].round(2).to_string(index=False))
    # compare with typical off-target SFT spillover
    for key in MODELS:
        m = pd.read_csv(RES / f"delta_sft_{key}_plus.csv", index_col=0)
        off = m.where(~np.eye(*m.shape, dtype=bool)).abs().stack().median() if list(m.index) == list(m.columns) else np.nan
        print(key, "median |off-target SFT spillover|", round(off, 2))


if __name__ == "__main__":
    main()

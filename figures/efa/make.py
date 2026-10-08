"""§4.6 / Appendix I: exploratory factor analysis of the spillover matrices.

Per base model, rows = all of its successful elicitation runs (SP, ICL, SFT, GRPO plus pole and
minus-pole SFT; each method's rows centred on that method's mean), columns = the 19
psychology-aligned traits. Horn's parallel analysis per model, then a k-factor EFA (sklearn
FactorAnalysis + varimax) per model, factors aligned across models by Hungarian matching on
Tucker's φ, consensus = mean aligned loadings, replication = leave-one-out |φ| vs. the other
two models, null = permutation of the trait columns.

  fig_efa_parallel_analysis_rep.png        App. I (parallel analysis)
  fig_efa_loadings_rep.png                 Fig. (three behavioral factors)
  fig_cross_model_replication_bars_rep.png App. I (cross-model replication)
  table1_trait_factors_rep.tex             Table 1 (trait-to-factor mapping)
  efa_rep_k<k>_loadings.csv, efa_rep_k<k>_phi.csv

    python figures/efa/make.py [--k 3]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib import spill as S  # noqa: E402

# The 19 traits that map onto human-psychology constructs (Table 1): the 24-trait battery minus
# caring-about-aesthetics, certainty, ev-reasoning, harm-refusal and procedural-fidelity.
PSYCH_TRAITS = ["agreeableness", "caring-about-animals", "caring-about-user", "claiming-sentience",
                "claiming-superintelligence", "cooperation", "ethical-framework-deontological",
                "ethical-framework-utilitarian", "ethical-framework-virtue-ethics", "punitiveness",
                "honest-humble", "narcissism", "neuroticism", "power-seeking", "resource-acquisition",
                "risk-affinity", "self-preservation", "spitefulness", "sycophancy"]
DISPLAY = {
    "agreeableness": "Agreeableness", "caring-about-animals": "Animals", "caring-about-user": "User-caring",
    "claiming-sentience": "Claim sentience", "claiming-superintelligence": "Claim super-int.",
    "cooperation": "Cooperation", "ethical-framework-deontological": "Deontological",
    "ethical-framework-utilitarian": "Utilitarian", "ethical-framework-virtue-ethics": "Virtue ethics",
    "punitiveness": "Punitiveness", "honest-humble": "Honest-humble", "narcissism": "Narcissism",
    "neuroticism": "Neuroticism", "power-seeking": "Power-seeking", "resource-acquisition": "Resource acq.",
    "risk-affinity": "Risk affinity", "self-preservation": "Self-preservation", "spitefulness": "Spitefulness",
    "sycophancy": "Sycophancy",
}


def varimax(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    p, k = loadings.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lambda = loadings @ R
        u, _, vt = np.linalg.svd(
            loadings.T @ (Lambda ** 3 - Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)) / p)
        )
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return loadings @ R


RES = S.DERIVED
OUT = HERE
FIGS = HERE
MODELS = ["qwen35", "qwen35-base", "nemotron35"]
REF = "qwen35-base"   # most non-failed rows
MODEL_SHORT = {"qwen35": "Qwen3.5-9B", "qwen35-base": "Qwen3.5-9B-Base", "nemotron35": "Nemotron-3-Super-120B"}
COLORS = {"qwen35": "#2a78d6", "qwen35-base": "#1baf7a", "nemotron35": "#eb6834"}
N_SIM, PCT, SEED = 1000, 95, 42
LOADING_MIN = 0.40


ROWS = os.environ.get("EFA_ROWS", "all")   # "sft_plus" (SFT plus-pole runs only) | "all"
METHODS = ("sp", "icl", "sft", "grpo")


def load(mod):
    """Rows = elicited models (one per successful elicitation run), columns = 19 traits.

    ROWS="all": every elicited model of this base model — all methods it has (SP, ICL, SFT,
    GRPO; plus pole) and the minus-pole SFT runs. Each method's rows are centred on that
    method's mean spillover per trait, so the factors describe which traits move together
    across elicitations, not which method moves everything more.
    """
    f = pd.read_csv(RES / "failed_runs.csv")
    ref = pd.read_csv(RES / f"delta_sft_{mod}_plus.csv", index_col=0)
    traits = [t for t in PSYCH_TRAITS if t in ref.columns]
    blocks = []
    for meth in (METHODS if ROWS == "all" else ("sft",)):
        p = RES / f"delta_{meth}_{mod}_plus.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, index_col=0)
        bad = set(f[(f.method == meth) & (f.model == mod)].trait)
        keep = [t for t in d.index if t not in bad and (ROWS == "all" or t in traits)]
        X = d.loc[keep, traits]
        X.index = [f"{meth}:{t}" for t in X.index]
        blocks.append((meth, X))
    mp = RES / f"delta_sft_{mod}_minus.csv"
    if ROWS == "all" and mp.exists():
        mm = pd.read_csv(mp, index_col=0)
        ot = pd.read_csv(RES / "on_target_minus.csv")
        mbad = set(ot[(ot.model == mod) & ot.failed].trait)
        mm = mm.drop(index=[t for t in mm.index if t in mbad])[traits]
        mm.index = [f"sft:{t}-minus" for t in mm.index]
        blocks.append(("sft-minus", mm))
    if ROWS == "all":
        X = pd.concat([b - b.mean() if m != "sft-minus" else b for m, b in blocks])
        # minus rows are centred with the plus SFT rows (same method)
        sft = pd.concat([b for m, b in blocks if m in ("sft", "sft-minus")])
        X.loc[sft.index] = (sft - sft.mean()).values
    else:
        X = pd.concat([b for _, b in blocks])
    return traits, list(X.index), X.values.astype(float)


def parallel_analysis(M):
    n, p = M.shape
    obs = np.sort(np.linalg.eigvalsh(np.corrcoef(M, rowvar=False)))[::-1]
    rng = np.random.default_rng(SEED)
    sim = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
                    for _ in range(N_SIM)])
    null = np.percentile(sim, PCT, axis=0)
    above = obs > null
    k_seq = int(np.argmin(above)) if not above.all() else p
    k_count = int(above[: n - 1].sum())   # old script's rule, restricted to non-degenerate eigenvalues
    return obs, null, k_seq, k_count


def tucker_phi(x, y):
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return (x * y).sum() / den if den > 0 else 0.0


def fit_efa(M, k):
    Mz = StandardScaler().fit_transform(M)
    fa = FactorAnalysis(n_components=k, random_state=42).fit(Mz)
    L = varimax(fa.components_.T)
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return L


def align(fits):
    """All models share the same 19 columns → loadings are directly comparable."""
    Lr = fits[REF]
    out = {REF: Lr.copy()}
    for m, L in fits.items():
        if m == REF:
            continue
        k = Lr.shape[1]
        P = np.array([[tucker_phi(Lr[:, i], L[:, j]) for j in range(k)] for i in range(k)])
        r, c = linear_sum_assignment(-np.abs(P))
        A = np.zeros_like(Lr)
        for i, j in zip(r, c):
            A[:, i] = np.sign(P[i, j]) * L[:, j]
        out[m] = A
    return out


def name_of(traits, col, n=3):
    idx = [i for i in np.argsort(-np.abs(col))[:n]]
    return " / ".join(("−" if col[i] < 0 else "") + DISPLAY[traits[i]] for i in idx)


def scree_figure(pa, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        obs, null, k, n = pa[m]["obs"], pa[m]["null"], pa[m]["k"], pa[m]["n"]
        ks = min(len(obs), 12)
        xs = np.arange(1, ks + 1)
        ax.plot(xs, obs[:ks], "o-", color="#C0392B", ms=6.5, lw=1.9, label="Observed", zorder=3)
        ax.plot(xs, null[:ks], "o--", color="#7F8C8D", ms=6.5, lw=1.5, mfc="white", mew=1.4,
                label=f"Random {PCT}th pct.", zorder=2)
        ax.axhline(1.0, color="#BBBBBB", ls=":", lw=1.0, zorder=1)
        ax.axvline(k + 0.5, color="#27AE60", ls="--", lw=1.1, alpha=0.8, zorder=1)
        ax.text(k + 0.62, max(obs) * 0.94, f"k = {k}", fontsize=9.5, color="#27AE60", va="top")
        ax.set_title(f"{MODEL_SHORT[m]}  ({n} elicited × 18 evaluated)", fontsize=10.5, pad=8)
        ax.set_xlabel("Factor number", fontsize=10)
        ax.set_xticks(xs)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Eigenvalue", fontsize=10.5)
    axes[0].legend(loc="upper right", fontsize=8.8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def loo_phis(Ms, k):
    A = align({m: fit_efa(Ms[m], k) for m in MODELS})
    out = []
    for m in MODELS:
        cons = np.mean(np.stack([A[o] for o in MODELS if o != m]), axis=0)
        out += [abs(tucker_phi(A[m][:, j], cons[:, j])) for j in range(k)]
    return out


def permutation_null(data, k, n_sim=200):
    """95th pct of per-factor leave-one-out |φ| when every column is shuffled independently
    across rows in every model (keeps marginals, destroys trait covariance)."""
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n_sim):
        v += loo_phis({m: np.column_stack([rng.permutation(c) for c in data[m][2].T]) for m in MODELS}, k)
    return float(np.percentile(v, 95)), float(np.mean(v))


def bars_figure(order, phis, path, k, null95=None):
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * k + 2), 3.8))
    x = np.arange(len(order))
    w = 0.26
    for lo, hi, lab in ((0.95, 1.0, "identical"), (0.85, 0.95, "good"), (0.70, 0.85, "fair")):
        ax.axhspan(lo, hi, color="#2a9a4a", alpha=0.06 if lab == "fair" else 0.1, lw=0)
        ax.text(len(order) - 0.45, (lo + hi) / 2, lab, fontsize=7, color="#57606a", va="center")
    if null95 is not None:
        ax.axhline(null95, color="#57606a", ls="--", lw=0.9)
        ax.text(len(order) - 0.45, null95 - 0.015, f"permutation\nnull 95th pct\n({null95:.2f})", fontsize=6.5,
                color="#57606a", va="top")
    for kk, m in enumerate(MODELS):
        v = phis[m]
        ax.bar(x + (kk - 1) * w, v, w, color=COLORS[m], label=MODEL_SHORT[m])
        for xi, vi in zip(x + (kk - 1) * w, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" / ", " /\n") for o in order], fontsize=7.5)
    ax.set_ylabel("Factor recovery vs. consensus\n(Tucker's |φ|)", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(order) - 0.5 + 0.5)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def loadings_figure(traits, C, names, path, per_model=None):
    """One line per factor: consensus loading of each trait (dots: individual models)."""
    # group traits by the factor they load on most, strongest first within each group
    dom = np.argmax(np.abs(C), axis=1)
    order = np.array(sorted(range(len(traits)), key=lambda i: (dom[i], -C[i, dom[i]])))
    cols = ["#2a78d6", "#d6452a", "#1baf7a", "#8a3ad6", "#eda100", "#57606a"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(traits))
    for j in range(C.shape[1]):
        ax.plot(x, C[order, j], marker="o", ms=4, lw=1.6, color=cols[j % len(cols)], label=names[j])
        if per_model is not None:
            for P in per_model:
                ax.scatter(x, P[order, j], s=7, color=cols[j % len(cols)], alpha=0.35, linewidths=0)
    ax.axhline(0, color="#57606a", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(traits[i], traits[i]) for i in order], rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("loading (consensus over models)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for g in range(1, C.shape[1]):  # separators between trait groups
        cut = np.searchsorted(dom[order], g)
        ax.axvline(cut - 0.5, color="#e6e6e3", lw=1)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7.5, frameon=False, loc="lower center",
               ncol=C.shape[1], bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run_k(k, data, sfx, old):
    traits = data[REF][0]
    fits = {m: fit_efa(data[m][2], k) for m in MODELS}
    A = align(fits)
    C = np.mean(np.stack(list(A.values())), axis=0)
    # orient each consensus factor so its largest loading is positive
    for j in range(k):
        if C[np.argmax(np.abs(C[:, j])), j] < 0:
            C[:, j] *= -1
            for m in A:
                A[m][:, j] *= -1
    ssl = (C ** 2).sum(axis=0)
    order_idx = list(np.argsort(-ssl))
    names = {j: name_of(traits, C[:, j]) for j in range(k)}

    # φ vs old paper factors (on the shared traits)
    old_match = {}
    if old is not None:
        ot, OC, onames = old
        sh = [t for t in traits if t in ot]
        a = np.array([traits.index(t) for t in sh])
        b = np.array([ot.index(t) for t in sh])
        for j in range(k):
            ph = [tucker_phi(C[a, j], OC[b, i]) for i in range(OC.shape[1])]
            i = int(np.argmax(np.abs(ph)))
            old_match[j] = (onames[i], ph[i])

    # leave-one-out replication
    phis = {}
    for m in MODELS:
        cons = np.mean(np.stack([A[o] for o in MODELS if o != m]), axis=0)
        phis[m] = [abs(tucker_phi(A[m][:, j], cons[:, j])) for j in order_idx]
    order = [names[j] for j in order_idx]
    phi_df = pd.DataFrame(phis, index=order)
    if old_match:
        phi_df["old_factor"] = [old_match[j][0] for j in order_idx]
        phi_df["phi_vs_old"] = [old_match[j][1] for j in order_idx]
    phi_df["ssl_consensus"] = [ssl[j] for j in order_idx]
    FIGS.mkdir(exist_ok=True)
    phi_df.round(3).to_csv(FIGS / f"efa_rep_k{k}_phi.csv")
    pd.DataFrame(C[:, order_idx], index=traits, columns=order).round(3).to_csv(FIGS / f"efa_rep_k{k}_loadings.csv")
    print(f"\n=== k = {k} ===")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(phi_df.round(2).to_string())

    # Table 1
    dom = np.argmax(np.abs(C), axis=1)
    rows = []
    for i, t in enumerate(traits):
        j = dom[i]
        agree = sum(int(np.argmax(np.abs(A[m][i])) == j) for m in MODELS)
        rows.append(dict(trait=t, factor=names[j], o=order_idx.index(j), loading=C[i, j], agree=agree,
                         assigned=abs(C[i, j]) >= LOADING_MIN))
    df = pd.DataFrame(rows).sort_values(["assigned", "o", "loading"], ascending=[False, True, False],
                                        key=lambda s: s.abs() if s.name == "loading" else s)
    lines = [r"\begin{tabular}{llrc}", r"\toprule",
             r"Trait & Consensus factor (top loaders) & Loading & Models agreeing \\", r"\midrule"]
    for r in df.itertuples():
        fac = f"F{r.o + 1}: {r.factor}" if r.assigned else f"(weak) F{r.o + 1}: {r.factor}"
        fac = fac.replace("−", "$-$")
        lines.append(f"{r.trait} & {fac} & {r.loading:+.2f} & {r.agree}/3 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"table1_trait_factors{sfx}.tex").write_text("\n".join(lines) + "\n")
    print(df.drop(columns="o").round(2).to_string(index=False))
    null95, null_mean = permutation_null(data, k)
    print(f"LOO |φ|: observed mean {np.mean([v for m in MODELS for v in phis[m]]):.3f}; "
          f"permutation null mean {null_mean:.3f}, 95th pct {null95:.3f}")
    loadings_figure(traits, C[:, order_idx], [f"F{i + 1}: {n}" for i, n in enumerate(order)],
                    OUT / f"fig_efa_loadings{sfx}.png", [A[m][:, order_idx] for m in MODELS])
    bars_figure([f"F{i + 1}: {n}" for i, n in enumerate(order)], phis,
                OUT / f"fig_cross_model_replication_bars{sfx}.png", k, null95)
    return phi_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[3])
    ap.add_argument("--default-k", type=int, default=3, help="k written without the _k<k> suffix")
    a = ap.parse_args()
    data = {m: load(m) for m in MODELS}
    pa = {}
    for m in MODELS:
        traits, rows, M = data[m]
        obs, null, k, kc = parallel_analysis(M)
        pa[m] = dict(obs=obs, null=null, k=k, n=len(rows))
        print(f"{m}: {M.shape}  parallel-analysis k = {k} (sequential), {kc} (count over first n−1)")
        print("   observed", np.round(obs[:6], 2), "\n   null95  ", np.round(null[:6], 2))
    scree_figure(pa, OUT / "fig_efa_parallel_analysis_rep.png")
    old = None
    for k in a.k:
        run_k(k, data, "_rep" if k == a.default_k else f"_rep_k{k}", old)


if __name__ == "__main__":
    main()

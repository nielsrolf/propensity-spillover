"""Shared analysis code for the paper figures.

Everything is computed from the released data in results/:

  results/runs/<run>.csv.gz       per-response judge scores (column `score` = paper score)
  results/runs/index.csv          run metadata
  results/claude_forecasts/<H>/   Claude Opus 5.5 forecasts (logitz_plus.csv, logitz_minus.csv)
  results/introspection/          self-prediction answers

Derived tables (spillover matrices, failed runs, predictor leaderboards) are written to
results/derived/ by `python figures/build_derived.py`; figure scripts read them from there.

Spillover (paper §3.5):
  S_{M,E}(A→B) = (mean_p logit(s_B(E(M,A), p)) − mean_p logit(s_B(M, p))) / σ_B,
  logit(s) = log(q / (1 − q)), q = (s + 0.5) / 101,
  σ_B = std of per-prompt logits of eval B over the base model M and all of its
  E-elicited models (plus pole). Minus-pole and NIP runs use the SFT σ_B of that model.
"""
from __future__ import annotations

import hashlib  # noqa: F401  (kept for parity with the original analysis code)
import pathlib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
RUNS = RESULTS / "runs"
DERIVED = RESULTS / "derived"
CLAUDE_DIR = RESULTS / "claude_forecasts"
INTROSPECTION = RESULTS / "introspection"

MODELS = {"qwen35": "qwen3.5-9b", "qwen35-base": "qwen3.5-9b-base", "nemotron35": "nemotron-3-super"}
NAME = {"qwen35": "Qwen3.5-9B", "qwen35-base": "Qwen3.5-9B-Base", "nemotron35": "Nemotron-3-Super-120B"}
METHODS = {"sp": "sysprompt", "icl": "icl", "sft": "sft", "grpo": "grpo"}
METHOD_LABEL = {"sft": "SFT", "sp": "System prompt", "icl": "ICL", "grpo": "GRPO"}
KEYS = [("sp", "qwen35"), ("sp", "nemotron35"), ("icl", "qwen35"), ("icl", "qwen35-base"),
        ("sft", "qwen35"), ("sft", "qwen35-base"), ("sft", "nemotron35"),
        ("grpo", "qwen35"), ("grpo", "qwen35-base"), ("grpo", "nemotron35")]
SFT_MODELS = ("qwen35", "qwen35-base", "nemotron35")
THRESHOLD = 0.2      # min on-target Δ (logit-z, intended direction) for a run to count as elicited
FLOOR_FRAC = 0.5     # a trait is at floor/ceiling if > this share of baseline scores are <=5 or >=95
SEED = 0
MALEVOLENT = ["spitefulness", "power-seeking", "resource-acquisition", "narcissism", "self-preservation",
              "punitiveness", "harm-refusal"]   # harm-refusal plus pole = compliance with harmful requests

# colours used across figures
BLUE, ORANGE, AQUA, MAGENTA, GRAY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#8c8c86", "#8a3ad6"
INK, MUTED = "#1f2328", "#57606a"
DRIFT_COL = "#6e6e6a"
CLAUDE = {
    "H1": "H1 Spurious correlation", "H2": "H2 Simplicity bias",
    "H3": "H3 Causal vs. statistical", "H4": "H4 Subliminal transmission",
    "H5": "H5 Out-of-context reasoning", "H6": "H6 Representation geometry",
    "H8": "H8 Persona theory", "H9": "H9 Human-psychology transfer",
    "H7a_r1": "H7a Anything goes (r1)", "H7a_r2": "H7a Anything goes (r2)",
    "H7a_r3": "H7a Anything goes (r3)",
    "H7b_r1": "H7b Intuition (r1)", "H7b_r2": "H7b Intuition (r2)", "H7b_r3": "H7b Intuition (r3)",
}
DRIFT = "Drift (mean over other elicited traits)"


# ---------------------------------------------------------------------------------------
# raw data
# ---------------------------------------------------------------------------------------
def empirical_logit(x, scale: float = 100.0):
    p = (np.asarray(x, dtype=float) + 0.5) / (scale + 1.0)
    return np.log(p / (1.0 - p))


@lru_cache(maxsize=None)
def index() -> pd.DataFrame:
    return pd.read_csv(RUNS / "index.csv")


@lru_cache(maxsize=None)
def run_scores(run: str) -> pd.DataFrame:
    """(eval, prompt_id, score) of one run; responses without evidence (gate < 50) dropped."""
    d = pd.read_csv(RUNS / f"{run}.csv.gz", usecols=["eval", "question_id", "score"])
    return d.rename(columns={"question_id": "prompt_id"}).assign(
        prompt_id=lambda x: x.prompt_id.astype(str)).dropna(subset=["score"])


def runs_of(short: str, method: str, pole: str = "plus") -> dict:
    """{trait: run} for one model and method."""
    ix = index()
    sel = ix[(ix.model_short == short) & (ix.method == method) & (ix.pole == pole)]
    return dict(zip(sel.trait, sel.run))


@lru_cache(maxsize=None)
def long_frames():
    """{(method, model): (elicited long df with model='<trait>-plus', base long df)}."""
    F = {}
    for key, short in MODELS.items():
        base = run_scores(f"{short}-baseline").assign(model="base")
        for meth, mdir in METHODS.items():
            runs = runs_of(short, mdir)
            if not runs:
                continue
            el = pd.concat([run_scores(r).assign(model=f"{t}-plus") for t, r in sorted(runs.items())])
            F[(meth, key)] = (el, base)
    return F


def sigma_of(el, base):
    panel = pd.concat([el, base])
    return panel.assign(l=empirical_logit(panel["score"].values)).groupby("eval")["l"].std()


def _split(source: str):
    for pole in ("plus", "minus"):
        if source.endswith("-" + pole):
            return source[: -len(pole) - 1], pole
    return None, None


def delta_matrices(df, base, sigma):
    """df: long scores of elicited models (model = '<trait>-<pole>'); base: long scores of the
    base model. Returns {pole: DataFrame(rows = elicited trait, cols = eval)}."""
    df = df.assign(logit=empirical_logit(df["score"].values))
    base = base.assign(logit=empirical_logit(base["score"].values))
    base_mean = base.groupby("eval")["logit"].mean()
    m = df.groupby(["model", "eval"])["logit"].mean().reset_index()
    m["delta"] = (m["logit"] - m["eval"].map(base_mean)) / m["eval"].map(sigma)
    out = {}
    for pole in ("plus", "minus"):
        rows = m[m["model"].map(lambda s: _split(s)[1] == pole)].copy()
        rows["source"] = rows["model"].map(lambda s: _split(s)[0])
        out[pole] = rows.pivot(index="source", columns="eval", values="delta").sort_index().sort_index(axis=1)
    return out


def deltas(el, base):
    return delta_matrices(el, base, sigma_of(el, base))["plus"]


# ---------------------------------------------------------------------------------------
# derived tables (read from results/derived/, built by figures/build_derived.py)
# ---------------------------------------------------------------------------------------
def failed_set(meth, mod):
    f = pd.read_csv(DERIVED / "failed_runs.csv")
    return set(f[(f.method == meth) & (f.model == mod)].trait)


def load(meth, mod, drop_failed=True):
    """Plus-pole spillover matrix of one (method, model); failed elicitations dropped as rows."""
    m = pd.read_csv(DERIVED / f"delta_{meth}_{mod}_plus.csv", index_col=0)
    if drop_failed:
        m = m.drop(index=[t for t in m.index if t in failed_set(meth, mod)])
    return m


def load_minus(mod):
    """Minus-pole SFT matrix of one model (failed rows dropped), or None."""
    p = DERIVED / f"delta_sft_{mod}_minus.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p, index_col=0)
    ot = pd.read_csv(DERIVED / "on_target_minus.csv")
    bad = set(ot[(ot.model == mod) & ot.failed].trait)
    return m.drop(index=[t for t in m.index if t in bad])


def load_claude():
    """{condition: {'plus': DataFrame, 'minus': DataFrame}} of the Claude Opus 5.5 forecasts."""
    preds = {}
    for d in sorted(CLAUDE_DIR.iterdir()):
        if not d.is_dir():
            continue
        mats = {}
        for pole in ("plus", "minus"):
            f = d / f"logitz_{pole}.csv"
            if f.exists():
                m = pd.read_csv(f, index_col=0)
                m.index = [_split(i)[0] or i for i in m.index]
                mats[pole] = m
        if mats:
            preds[d.name] = mats
    return preds


# ---------------------------------------------------------------------------------------
# predictors and scoring
# ---------------------------------------------------------------------------------------
def mean_of(mats):
    return pd.concat(mats, keys=range(len(mats))).groupby(level=1).mean()


def pairs(pred, obs, cells=None):
    pv, ov = [], []
    for r in pred.index.intersection(obs.index):
        for c in pred.columns.intersection(obs.columns):
            if r == c or (cells is not None and (r, c) not in cells):
                continue
            p, o = pred.at[r, c], obs.at[r, c]
            if pd.notna(p) and pd.notna(o):
                pv.append(p)
                ov.append(o)
    return np.asarray(pv), np.asarray(ov)


def boot_ci(pv, ov, n=1000):
    rng = np.random.default_rng(SEED)
    bs = [spearmanr(pv[i], ov[i])[0] for i in (rng.integers(0, len(pv), len(pv)) for _ in range(n))]
    return np.nanpercentile(bs, [2.5, 97.5])


def column_marginal(obs):
    """Drift predictor: A→B predicted by the mean over A' ∉ {A, B} of A'→B."""
    pred = obs.copy() * np.nan
    for r in obs.index:
        others = obs.drop(index=r)
        for c in obs.columns:
            pred.at[r, c] = others[c].drop(index=c, errors="ignore").mean()
    return pred


def floor_traits(base):
    g = base.groupby("eval").score
    frac = g.apply(lambda x: max((x <= 5).mean(), (x >= 95).mean()))
    return set(frac[frac > FLOOR_FRAC].index)


def symmetry_pred(T, Tm, floor, tau=0.1):
    """Symmetric prediction of S(A→B) from the mirrored measurement (§4.5):
    if S(B+→A) ≥ 0 predict S(B+→A); if S(B+→A) < 0 and a minus-pole run of B exists,
    predict −S(B−→A); otherwise S(B+→A). If A is a floor/ceiling trait at baseline, a
    near-zero mirrored effect (|S| < tau) is uninformative and the cell is dropped."""
    pred = pd.DataFrame(np.nan, index=T.index, columns=T.columns)
    for a in T.index:
        for b in T.columns:
            if a == b or b not in T.index or a not in T.columns or pd.isna(T.at[b, a]):
                continue
            meas = val = T.at[b, a]
            if meas < 0 and Tm is not None and b in Tm.index and a in Tm.columns and pd.notna(Tm.at[b, a]):
                meas = Tm.at[b, a]
                val = -meas
            if a in floor and abs(meas) < tau:
                continue
            pred.at[a, b] = float(val)
    return pred


def predictors(M, models_of, tm, mod, claude, Tm=None, floor=frozenset()):
    P = {}
    for om, mods in models_of.items():
        if om != tm and mod in mods:
            P[f"{METHOD_LABEL[om]} (same model)"] = M[(om, mod)]
    others = [m for m in models_of[tm] if m != mod]
    if others:
        P[f"{METHOD_LABEL[tm]} (other models)"] = mean_of([M[(tm, m)] for m in others])
    T = M[(tm, mod)]
    P["Symmetry (mirrored B→A)"] = symmetry_pred(T, Tm if tm == "sft" else None, floor)
    P[DRIFT] = column_marginal(T)
    for name, mats in claude.items():
        if "plus" in mats:
            P[f"claude:{name}"] = mats["plus"]
    return P

"""App H / §4.4 numbers for the H9 (human-psychology) predictor on the replication.

- h9_prediction_table_rep.tex: the H9 spec's evidence table restricted to pairs whose
  traits are both in the 24-trait battery
- rho on 'confident' cells (Tier-1 pairs of the spec, both directions), per tier
- cross-model agreement of observed SFT spillover (pairwise Spearman between models)
- error analysis: magnitude slope, per-trait rho, largest sign errors

python figures/h9_human_psychology/make.py
"""
import itertools
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
B = S
load = S.load
load_minus = S.load_minus

TEX = pathlib.Path(__file__).resolve().parent   # reads h9_prediction_table.tex (the H9 spec table), writes *_rep.tex
MODELS = ("qwen35", "qwen35-base", "nemotron35")

H9 = S.load_claude()["H9"]
OBS = {m: load("sft", m) for m in MODELS}
MIN = {m: load_minus(m) for m in MODELS}
BATTERY = set(H9["plus"].columns)


def cells(cellset=None):
    """long frame of (model, pole, A, B, pred, obs) over off-diagonal cells."""
    rows = []
    for m in MODELS:
        for pole, P, O in (("plus", H9["plus"], OBS[m]), ("minus", H9["minus"], MIN[m])):
            if O is None:
                continue
            for a in P.index.intersection(O.index):
                for b in P.columns.intersection(O.columns):
                    if a == b or (cellset is not None and frozenset((a, b)) not in cellset):
                        continue
                    p, o = P.at[a, b], O.at[a, b]
                    if pd.notna(p) and pd.notna(o):
                        rows.append((m, pole, a, b, p, o))
    return pd.DataFrame(rows, columns=["model", "pole", "A", "B", "pred", "obs"])


# --- filter the spec table -------------------------------------------------
src = (TEX / "h9_prediction_table.tex").read_text().splitlines()
out, tiers, tier, kept, dropped = [], {}, 0, 0, 0
for line in src:
    if line.startswith("\\noindent\\textbf{Tier"):
        tier += 1
    m = re.match(r"\s*([a-z\- ]+?) \$\\leftrightarrow\$ ([a-z\- ]+?) &", line)
    if m:
        a, b = (x.strip().removeprefix("low ") for x in m.groups())
        if a in BATTERY and b in BATTERY:
            tiers.setdefault(tier, set()).add(frozenset((a, b)))
            kept += 1
        else:
            dropped += 1
            continue
    out.append(line)
(TEX / "h9_prediction_table_rep.tex").write_text("\n".join(out) + "\n")
print(f"table: kept {kept} pairs, dropped {dropped}; per tier", {t: len(s) for t, s in tiers.items()})

allc = cells()
print("all cells rho %.3f n %d" % (spearmanr(allc.pred, allc.obs)[0], len(allc)))
for t, s in sorted(tiers.items()):
    d = cells(s)
    print(f"tier {t}: rho {spearmanr(d.pred, d.obs)[0]:.3f} n {len(d)}  sign agreement {(np.sign(d.pred) == np.sign(d.obs)).mean():.2f}")
d = cells(tiers[1] | tiers[2])
print(f"tier 1+2: rho {spearmanr(d.pred, d.obs)[0]:.3f} n {len(d)}")
big = allc[allc.pred.abs() >= 0.25]
print(f"|pred|>=0.25: rho {spearmanr(big.pred, big.obs)[0]:.3f} n {len(big)} sign agree {(np.sign(big.pred)==np.sign(big.obs)).mean():.2f}")

# --- cross-model agreement (plus pole, off-diagonal) ------------------------
rs = []
for x, y in itertools.combinations(MODELS, 2):
    pv, ov = B.pairs(OBS[x], OBS[y])
    r = spearmanr(pv, ov)[0]
    rs.append(r)
    print(f"agreement {x} vs {y}: {r:.3f} n {len(pv)}")
print("mean pairwise agreement %.3f" % np.mean(rs))

# --- error analysis ---------------------------------------------------------
nz = allc[allc.pred.abs() > 1e-9]
slope = np.polyfit(nz.pred, nz.obs, 1)[0]
print(f"magnitude: OLS slope obs~pred {slope:.2f}; median |obs|/|pred| on |pred|>=0.25: {np.median(big.obs.abs() / big.pred.abs()):.2f}")
per = []
for t in sorted(BATTERY):
    for role, sub in (("as measured", allc[allc.B == t]), ("as elicited", allc[allc.A == t])):
        if len(sub) > 10 and sub.pred.std() > 0:
            per.append((t, role, spearmanr(sub.pred, sub.obs)[0], len(sub)))
per = pd.DataFrame(per, columns=["trait", "role", "rho", "n"])
print(per.pivot(index="trait", columns="role", values="rho").round(2).sort_values("as measured").to_string())
g = allc[allc.pole == "plus"].groupby(["A", "B"]).agg(pred=("pred", "first"), obs=("obs", "mean"), n=("obs", "size")).reset_index()
g = g[(g.n == 3)]
g["err"] = np.sign(g.pred) * g.obs  # negative = wrong sign
print("largest sign errors (plus, mean over 3 models, |pred|>=0.25):")
print(g[g.pred.abs() >= 0.25].sort_values("err").head(12).round(2).to_string(index=False))
print("largest unpredicted effects (|pred|<0.1):")
g["absobs"] = g.obs.abs()
print(g[g.pred.abs() < 0.1].sort_values("absobs", ascending=False).head(10).round(2).to_string(index=False))

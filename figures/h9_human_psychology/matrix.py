"""App H: heatmap of the H9 (human-psychology) predicted plus-pole spillover matrix.

python figures/h9_human_psychology/make.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import spill as S  # noqa: E402
P = pd.read_csv(S.CLAUDE_DIR / "H9" / "logitz_plus.csv", index_col=0)
P.index = [i.removesuffix("-plus") for i in P.index]
traits = sorted(set(P.index) & set(P.columns))
P = P.loc[traits, traits].astype(float)
M = P.values.copy()
np.fill_diagonal(M, np.nan)

fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#d9d9d9")
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(traits)), traits, rotation=90, fontsize=8)
ax.set_yticks(range(len(traits)), traits, fontsize=8)
ax.set_xticks(np.arange(-.5, len(traits)), minor=True)
ax.set_yticks(np.arange(-.5, len(traits)), minor=True)
ax.grid(which="minor", color="white", lw=0.5)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("measured trait")
ax.set_ylabel("elicited trait (plus pole)")
ax.set_title("Human-psychology (H9) predicted cross-trait spillover")
fig.colorbar(im, ax=ax, label="H9 predicted spillover (logit-z)", fraction=0.04)
fig.tight_layout()
fig.savefig(OUT / "h9_predicted_matrix.png", dpi=200)
print("wrote", OUT / "h9_predicted_matrix.png", P.shape)

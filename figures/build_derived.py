"""Build the derived tables in results/derived/ from the per-response data in results/runs/.

  delta_<method>_<model>_plus.csv   plus-pole spillover matrices (rows: elicited trait, cols: eval)
  delta_sft_<model>_minus.csv       minus-pole SFT spillover matrices (SFT σ of that model)
  on_target.csv, failed_runs.csv    on-target effect per run; failed = on-target Δ < 0.2
  on_target_minus.csv               same for minus-pole runs (signed: + = moved as intended)
  leaderboard_all.csv               every predictor vs. every target (plus pole), per model and pooled
  leaderboard_claude_pm.csv         Claude conditions vs. SFT, plus and minus cells pooled (§4.2)

    python figures/build_derived.py
"""
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import spill as S  # noqa: E402


def main():
    S.DERIVED.mkdir(parents=True, exist_ok=True)
    F = S.long_frames()
    M = {k: S.deltas(*v) for k, v in F.items()}
    rows = [dict(method=meth, model=mod, pole="plus", trait=t, on_target=m.at[t, t])
            for (meth, mod), m in M.items() for t in m.index if t in m.columns]
    ot = pd.DataFrame(rows)
    ot["failed"] = ot.on_target < S.THRESHOLD
    ot.to_csv(S.DERIVED / "on_target.csv", index=False)
    ot[ot.failed].to_csv(S.DERIVED / "failed_runs.csv", index=False)
    for (meth, mod), m in M.items():
        m.to_csv(S.DERIVED / f"delta_{meth}_{mod}_plus.csv")
    print(f"{len(M)} plus matrices; failed runs: {ot.failed.sum()} of {len(ot)}")

    # minus-pole SFT matrices on the plus-pole SFT scale
    rows = []
    for mod, short in S.MODELS.items():
        el, base = F[("sft", mod)]
        sigma = S.sigma_of(el, base)
        runs = S.runs_of(short, "sft", pole="minus")
        if not runs:
            continue
        mm = pd.concat([S.run_scores(r).assign(model=f"{t}-minus") for t, r in sorted(runs.items())])
        m = S.delta_matrices(mm, base, sigma)["minus"]
        m.to_csv(S.DERIVED / f"delta_sft_{mod}_minus.csv")
        rows += [dict(method="sft", model=mod, pole="minus", trait=t, on_target=-m.at[t, t]) for t in m.index]
    otm = pd.DataFrame(rows)
    otm["failed"] = otm.on_target < S.THRESHOLD
    otm.to_csv(S.DERIVED / "on_target_minus.csv", index=False)

    # predictor leaderboard (plus pole; §4.1 and appendix)
    failed = ot[ot.failed]
    Md = {k: m.drop(index=[t for t in m.index if t in set(failed[(failed.method == k[0]) & (failed.model == k[1])].trait)])
          for k, m in M.items()}
    minus = {mod: S.load_minus(mod) for (_, mod) in F}
    floor = {mod: S.floor_traits(base) for (_, mod), (_, base) in F.items()}
    claude = S.load_claude()
    models_of = {}
    for meth, mod in Md:
        models_of.setdefault(meth, []).append(mod)
    rows = []
    for tm, mods in models_of.items():
        pooled = {}
        for mod in mods:
            T = Md[(tm, mod)]
            for name, pm in S.predictors(Md, models_of, tm, mod, claude, Tm=minus.get(mod), floor=floor[mod]).items():
                pv, ov = S.pairs(pm, T)
                if len(pv) < 10:
                    continue
                p = pooled.setdefault(name, ([], [], []))
                p[0].append(pv); p[1].append(ov); p[2].append(mod)
                lo, hi = S.boot_ci(pv, ov, 300) if not name.startswith("claude:") else (np.nan, np.nan)
                rows.append(dict(target_method=tm, target_model=mod, pole="plus", predictor=name,
                                 rho=spearmanr(pv, ov)[0], lo=lo, hi=hi, n=len(pv)))
        for name, (pvs, ovs, pmods) in pooled.items():
            pv, ov = np.concatenate(pvs), np.concatenate(ovs)
            lo, hi = S.boot_ci(pv, ov)
            rows.append(dict(target_method=tm, target_model="pooled", pole="plus", predictor=name,
                             rho=spearmanr(pv, ov)[0], lo=lo, hi=hi, n=len(pv), models=",".join(pmods)))
    pd.DataFrame(rows).to_csv(S.DERIVED / "leaderboard_all.csv", index=False)

    # Claude conditions on plus + minus SFT cells (§4.2)
    rows = []
    for name, mats in claude.items():
        pv, ov, n_minus = [], [], 0
        for mod in S.SFT_MODELS:
            a, b = S.pairs(mats["plus"], Md[("sft", mod)])
            pv += list(a); ov += list(b)
            if "minus" in mats and minus.get(mod) is not None:
                a, b = S.pairs(mats["minus"], minus[mod])
                pv += list(a); ov += list(b); n_minus += len(a)
        pv, ov = np.array(pv), np.array(ov)
        lo, hi = S.boot_ci(pv, ov)
        rows.append(dict(target_method="sft", target_model="pooled_pm", pole="plus+minus",
                         predictor=f"claude:{name}", rho=spearmanr(pv, ov)[0], lo=lo, hi=hi,
                         n=len(pv), n_minus=n_minus))
    pd.DataFrame(rows).to_csv(S.DERIVED / "leaderboard_claude_pm.csv", index=False)
    print("wrote", S.DERIVED)


if __name__ == "__main__":
    main()

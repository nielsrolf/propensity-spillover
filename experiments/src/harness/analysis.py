"""Per-eval aggregation with baseline-SD (Cohen's d) normalization.

One row of a judged CSV = one sampled response. Its eval score is the mean of
that eval's non-null metric columns (agreeableness has 5 facets, the
ethical-framework evals 3 metrics; everything else has one). Rows where the
evidence gate failed have all-null metrics and count towards `null%` only.

Normalization: deltas are reported both in raw points (0-100) and in units of
the BASELINE standard deviation of the per-response eval scores, i.e.
Cohen's d = (mean_variant - mean_baseline) / sd_baseline. Baseline SD is the
sample SD (ddof=1) of the baseline variant's per-response scores for that eval,
so a d of 1.0 means "one baseline response-level SD".
"""

import csv
import math
import os
from typing import Dict, List, Optional, Sequence

ID_COLS = ("eval", "question_id", "paraphrase_index", "sample_index", "split", "gate")


def load_judged(path: str) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def row_score(row: dict) -> Optional[float]:
    """Mean of the row's non-null metric columns (None if the gate failed)."""
    vals = []
    for k, v in row.items():
        if k in ID_COLS or v in ("", None, "None"):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return sum(vals) / len(vals) if vals else None


def per_eval_scores(rows: Sequence[dict]) -> Dict[str, List[Optional[float]]]:
    out: Dict[str, List[Optional[float]]] = {}
    for r in rows:
        out.setdefault(r["eval"], []).append(row_score(r))
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def eval_stats(rows: Sequence[dict]) -> Dict[str, dict]:
    """{eval: {n, n_scored, null_frac, mean, sd, sem}}."""
    stats = {}
    for ev, scores in per_eval_scores(rows).items():
        ok = [s for s in scores if s is not None]
        sd = _sd(ok)
        stats[ev] = {
            "n": len(scores), "n_scored": len(ok),
            "null_frac": (len(scores) - len(ok)) / len(scores) if scores else None,
            "mean": _mean(ok), "sd": sd,
            "sem": sd / math.sqrt(len(ok)) if sd is not None and ok else None,
        }
    return stats


def summarize(baseline: str | Sequence[dict], variants: Dict[str, str | Sequence[dict]]) -> dict:
    """Compare variants against a baseline.

    baseline  judged CSV path (or already-loaded rows) of the baseline variant
    variants  {variant_name: judged CSV path or rows}

    Returns {"baseline": {eval: stats}, "variants": {name: {eval: stats + delta,
    cohens_d}}, "evals": [...]} where delta is in points and cohens_d uses the
    baseline SD of that eval.
    """
    def rows_of(x):
        return load_judged(x) if isinstance(x, str) else list(x)

    base_stats = eval_stats(rows_of(baseline))
    out = {"baseline": base_stats, "variants": {}, "evals": []}
    for name, src in variants.items():
        vs = eval_stats(rows_of(src))
        for ev, s in vs.items():
            b = base_stats.get(ev)
            s["delta"] = (s["mean"] - b["mean"]) if b and b["mean"] is not None and s["mean"] is not None else None
            s["baseline_sd"] = b["sd"] if b else None
            s["cohens_d"] = (s["delta"] / b["sd"]) if s["delta"] is not None and b and b["sd"] else None
        out["variants"][name] = vs
    out["evals"] = sorted(set(base_stats) | {e for v in out["variants"].values() for e in v})
    return out


def summary_markdown(summary: dict, title: str = "Results", target_eval: Optional[str] = None) -> str:
    """Markdown table: per-eval baseline mean/SD, variant mean, delta (points) and d."""
    names = list(summary["variants"])
    hdr = ["eval", "baseline mean", "baseline SD"]
    for n in names:
        hdr += [f"{n} mean", f"{n} delta", f"{n} d", f"{n} null%"]
    lines = [f"## {title}", "",
             "Deltas vs baseline in raw points and in baseline-SD units (Cohen's d).", "",
             " | ".join(hdr), " | ".join(["---"] * len(hdr))]

    def rank_key(ev):
        ds = [summary["variants"][n].get(ev, {}).get("cohens_d") for n in names]
        ds = [abs(d) for d in ds if d is not None]
        return -max(ds) if ds else 0.0

    for ev in sorted(summary["evals"], key=rank_key):
        b = summary["baseline"].get(ev, {})
        cells = [ev, f"{b.get('mean'):.1f}" if b.get("mean") is not None else "-",
                 f"{b.get('sd'):.1f}" if b.get("sd") is not None else "-"]
        for n in names:
            s = summary["variants"][n].get(ev, {})
            cells += [
                f"{s.get('mean'):.1f}" if s.get("mean") is not None else "-",
                f"{s.get('delta'):+.1f}" if s.get("delta") is not None else "-",
                f"{s.get('cohens_d'):+.2f}" if s.get("cohens_d") is not None else "-",
                f"{100 * s.get('null_frac'):.0f}%" if s.get("null_frac") is not None else "-",
            ]
        line = " | ".join(cells)
        if target_eval and ev == target_eval:
            line = " | ".join(f"**{c}**" for c in cells)
        lines.append(line)
    lines.append("")
    lines.append("Rows sorted by |Cohen's d| (largest spillover first)"
                 + (f"; **{target_eval}** is the elicitation target." if target_eval else "."))
    return "\n".join(lines) + "\n"


def summarize_paths(baseline_csv: str, variant_csvs: Sequence[str], title: str = "Results",
                    target_eval: Optional[str] = None) -> str:
    """Convenience CLI helper: variant names are taken from the parent dir names."""
    variants = {os.path.basename(os.path.dirname(os.path.abspath(p))) or p: p for p in variant_csvs}
    return summary_markdown(summarize(baseline_csv, variants), title, target_eval)

"""Build server/baselines.json: the host's H1–H9 specs + their frozen predictions.

Run by stage.sh at deploy time. Specs and predictions are read from the host's
forecast bundle (a directory outside this repo with hypotheses/<H>.md and
predictions/<H>/{logitz_plus.csv,logitz_minus.csv,method.md,falsifiers.md,result.json}),
so the predictions never enter git or the build context except as this file.
The output is NOT committed: predictions are only served to participants who
have submitted their own (then locked) entry.

Usage: python server/build_baselines.py [out.json] [--specs DIR] [--predictions DIR]
       (defaults: webapp/hypotheses and results/claude_forecasts)
"""
import argparse
import csv
import io
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SPECS = os.path.join(ROOT, "hypotheses")
DEFAULT_PREDS = os.path.join(ROOT, "..", "results", "claude_forecasts")

GROUPS = [
    ("H1", ["H1"]), ("H2", ["H2"]), ("H3", ["H3"]), ("H4", ["H4"]), ("H5", ["H5"]), ("H6", ["H6"]),
    ("H7a", ["H7a_r1", "H7a_r2", "H7a_r3"]), ("H7b", ["H7b_r1", "H7b_r2", "H7b_r3"]),
    ("H8", ["H8"]), ("H9", ["H9"]),
]
HEADERS = ["Statement of the hypothesis", "Definitions", "Literature", "Operationalization",
           "Prerequisites for testing", "Falsifiers"]


def template(name):
    with open(os.path.join(ROOT, "inputs", name)) as f:
        rows = list(csv.reader(f))
    return rows[0], [r[0] for r in rows[1:]]


HEADER, PLUS_ROWS = template("PREDICT_transfer_matrix_logitz_plus.csv")
_, MINUS_ROWS = template("PREDICT_transfer_matrix_logitz_minus.csv")


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def title_of(spec):
    first = next((l for l in spec.splitlines() if l.strip()), "")
    t = re.sub(r"^#+\s*", "", first).strip()
    t = re.sub(r"^(Hypothes[ie]s\s*\d*\s*[:.]\s*|Hypothesis\s*:\s*)", "", t, flags=re.I).strip()
    return t[:120]


def to_markdown(spec):
    """Plain-text specs use bare header lines; promote them so they render as headings."""
    if re.search(r"^#{1,3} ", spec, re.M):
        return spec
    out = []
    for i, line in enumerate(spec.splitlines()):
        s = line.strip()
        if i == 0 and s:
            out.append("# " + s)
        elif any(s.lower().startswith(h.lower()) and len(s) < len(h) + 40 for h in HEADERS):
            out += ["", "## " + s, ""]
        else:
            out.append(line)
    return "\n".join(out)


def check_matrix(text, row_labels, what):
    rows = list(csv.reader(io.StringIO(text.strip())))
    if rows[0] != HEADER or [r[0] for r in rows[1:]] != row_labels or any(len(r) != len(HEADER) for r in rows):
        sys.exit(f"{what}: shape/labels do not match the inputs/ template "
                 f"({len(row_labels)} rows x {len(HEADER) - 1} cols)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "server", "baselines.json"))
    ap.add_argument("--specs", default=DEFAULT_SPECS)
    ap.add_argument("--predictions", default=DEFAULT_PREDS)
    a = ap.parse_args()
    specs, preds = a.specs, a.predictions

    examples, models = [], set()
    for gid, members in GROUPS:
        spec = read(os.path.join(specs, f"{members[0]}.md"))
        if spec is None:
            sys.exit(f"missing spec {specs}/{members[0]}.md")
        runs = []
        for m in members:
            d = os.path.join(preds, m)
            res = json.loads(read(os.path.join(d, "result.json")) or "{}")
            if res.get("status") != "done":
                sys.exit(f"{m}: no finished prediction in {d} (result.json status={res.get('status')!r})")
            models.add(res.get("model"))
            files = {k: read(os.path.join(d, f)) for k, f in
                     (("plus_csv", "logitz_plus.csv"), ("minus_csv", "logitz_minus.csv"),
                      ("method_md", "method.md"), ("falsifiers_md", "falsifiers.md"))}
            if not files["plus_csv"] or not files["minus_csv"]:
                sys.exit(f"missing prediction CSVs for {m} in {d}")
            check_matrix(files["plus_csv"], PLUS_ROWS, f"{m}/logitz_plus.csv")
            check_matrix(files["minus_csv"], MINUS_ROWS, f"{m}/logitz_minus.csv")
            runs.append({"id": m, "model": res.get("model"), "finished_at": res.get("finished_at"),
                         **{k: v or "" for k, v in files.items()}})
        examples.append({"id": gid, "title": title_of(spec), "spec_md": to_markdown(spec).strip() + "\n",
                         "spec_text": spec, "runs": runs})
    models = sorted(m for m in models if m)
    json.dump({"source": "host forecast bundle", "models": models, "examples": examples},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(examples)} examples, {sum(len(e['runs']) for e in examples)} runs ({', '.join(models)})")


if __name__ == "__main__":
    main()

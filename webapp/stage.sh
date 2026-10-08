#!/usr/bin/env bash
# stage.sh <out_dir> — copy ONLY the blinding-safe files into a clean Docker
# build context: inputs/, BLINDED_PROMPT.md, EXAMPLE_hypothesis.md and server/.
# Deploy from <out_dir>, never from the repo root. The host's H1–H9 specs and
# frozen predictions are packed into server/baselines.json (see
# server/build_baselines.py); the API only returns them to participants whose
# own entry is submitted.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:?usage: webapp/stage.sh <out_dir>}"
[ -d inputs/evals_orthogonalized/agreeableness ] || python3 build_bundle.py
rm -rf "$OUT" && mkdir -p "$OUT/server/static"
cp -R inputs "$OUT/inputs"
cp BLINDED_PROMPT.md EXAMPLE_hypothesis.md "$OUT/"
cp server/*.py server/requirements.txt "$OUT/server/"
cp server/static/index.html server/static/app.js server/static/style.css server/static/fig_overview.svg "$OUT/server/static/"
cp server/Dockerfile "$OUT/Dockerfile"
python3 server/build_baselines.py "$OUT/server/baselines.json"
echo "staged build context in $OUT"

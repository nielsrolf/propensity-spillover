#!/usr/bin/env bash
# Rebuild results/derived/ and every paper figure from results/runs/ (+ claude_forecasts, introspection).
set -euo pipefail
cd "$(dirname "$0")/.."
python figures/build_derived.py
for d in fig1_overview forecast_predictors forecast_scatter on_off_target failed_runs introspection \
         h9_human_psychology symmetry efa nip_drift method_heatmaps; do
  echo "== $d"; python figures/$d/make.py > /dev/null
done
echo "done"

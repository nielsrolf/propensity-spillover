#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PG_EXPERIMENT=sweep PYTHONPATH=$PWD/experiments/src
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-9B-Base nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; do
  .venv/bin/python experiments/introspection/introspect.py --model "$m" &
done
wait
echo ALL DONE

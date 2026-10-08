#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PG_EXPERIMENT=sweep PYTHONPATH=$PWD/experiments/src
T=agreeableness,caring-about-user,certainty,ev-reasoning,narcissism,procedural-fidelity,sycophancy
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-9B-Base nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; do
  .venv/bin/python experiments/introspection/introspect.py --model "$m" --pole negative --traits $T &
done
wait; echo ALL DONE

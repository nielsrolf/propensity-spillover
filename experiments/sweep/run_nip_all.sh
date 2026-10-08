#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PG_EXPERIMENT=sweep PYTHONPATH=$PWD/experiments/src
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-9B-Base nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; do
  for d in nip1 nip2 nip3; do
    s=$(echo $m | sed 's#.*/##')
    ( .venv/bin/python experiments/sweep/run_nip.py --model "$m" --dataset $d > "experiments/sweep/logs/nip-$s-$d.log" 2>&1 \
      && echo "done $s $d" || echo "FAILED $s $d" ) &
  done
done
wait; echo ALL DONE

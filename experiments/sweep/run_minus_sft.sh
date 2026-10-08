#!/usr/bin/env bash
# Minus-pole SFT runs for the symmetry check (spillover paper §4.1.1).
# For strong negative SFT spillover A+ -> B from misalignment traits, the symmetric
# test "eliciting less B raises A" needs B's minus pole. 7 traits x 3 models = 21 runs.
# Same recipe as the sweep's plus-pole SFT (Nemotron: 3 epochs). Results land in
# results/<short>-sft-minus-<trait>/. Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PG_EXPERIMENT=sweep PYTHONPATH=$PWD/experiments/src
PY=.venv/bin/python
RUN=experiments/sweep/run.py
RES=experiments/sweep/results
LOGS=experiments/sweep/logs
TRAITS="caring-about-user ev-reasoning procedural-fidelity agreeableness certainty narcissism sycophancy"

one() {  # model short trait
  local model=$1 short=$2 trait=$3 name="$2-sft-minus-$3" epochs=""
  [ -f "$RES/$name/judged.csv" ] && { echo "skip $name"; return; }
  [ "$short" = "nemotron-3-super" ] && epochs="--epochs 3"
  local sp
  sp=$(grep -ho "tinker://[^ ]*" "$LOGS/$name-train.log" 2>/dev/null | tail -1)
  if [ -z "$sp" ]; then
    $PY $RUN train-sft --model "$model" --trait "$trait" --pole negative $epochs \
        --run-name "$name" --launch > "$LOGS/$name-train.log" 2>&1
    sp=$(grep -ho "tinker://[^ ]*" "$LOGS/$name-train.log" | tail -1)
  fi
  [ -z "$sp" ] && { echo "TRAIN FAILED $name"; return; }
  $PY $RUN eval --model "$model" --variant sft --trait "$trait" --sampler-path "$sp" \
      --run-name "$name" --baseline-judged "$RES/$short-baseline/judged.csv" > "$LOGS/$name.log" 2>&1
  [ -f "$RES/$name/judged.csv" ] && echo "done $name" || echo "EVAL FAILED $name"
}

for spec in "Qwen/Qwen3.5-9B qwen3.5-9b" "Qwen/Qwen3.5-9B-Base qwen3.5-9b-base" \
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 nemotron-3-super"; do
  set -- $spec
  for t in $TRAITS; do one "$1" "$2" "$t" & done
  wait
done
echo ALL DONE

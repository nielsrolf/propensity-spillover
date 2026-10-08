# sweep — the elicitation-spillover experiment (paper §4)

Elicit each of the 24 traits with every applicable method on every study model, then run the
full battery (all 24 evals, test split, 3,683 prompts) on the elicited model.

| Model | Methods |
|---|---|
| Qwen/Qwen3.5-9B (thinking disabled) | system prompt, ICL, SFT, GRPO |
| Qwen/Qwen3.5-9B-Base | ICL, SFT, GRPO |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | system prompt, SFT, GRPO |

plus one baseline battery per model. All training and sampling runs on [Tinker](https://thinkingmachines.ai/tinker/);
the judge is gpt-5.6-luna through a LiteLLM proxy. Hyperparameters are in `config.yaml`
(and in App. C of the paper):

- **SFT**: LoRA rank 32 on the 50 plus-pole reference responses of the trait's train split,
  lr 2e-4, batch 8, 1 epoch (3 epochs for Nemotron).
- **GRPO**: from the model itself, LoRA rank 32, reward = the trait's own judge score,
  10 prompts × 8 samples per step, ≤ 15 steps, lr 1.6e-4 (4.8e-4 for Nemotron), early stop
  once the mean reward stays ≥ 0.85 for two steps.
- **ICL**: 8 plus-pole question–answer pairs from the train split as a few-shot prefix.
- **System prompt**: the trait's `system_prompts/positive.txt` (instruction-tuned models only).
- Sampling: temperature 1.0, ≤ 512 tokens; one run per (trait, method, model).

Additional runs:

- `run_minus_sft.sh` — minus-pole SFT on 7 traits × 3 models (symmetry check, §4.5).
- `run_nip.py` / `run_nip_all.sh` — SFT on three "nothing in particular" datasets
  (`nip_datasets/`) per model (§4.7).

## Running

```bash
cp .env.example .env    # at the repo root: TINKER_API_KEY, LITELLM_BASE_URL, LITELLM_API_KEY
export PG_EXPERIMENT=sweep PYTHONPATH=$PWD/experiments/src
python experiments/sweep/run_sweep.py --dry-run          # show remaining work
python experiments/sweep/run_sweep.py --max-minutes 90   # resumable
```

Native outputs go to `experiments/sweep/results/<run>/` (`responses.jsonl.gz`, `judged.csv`,
`cost.json`, `summary.md`); `python experiments/export_results.py` converts them into the
released format under `results/runs/`. Run names: `{qwen3.5-9b|qwen3.5-9b-base|nemotron-3-super}-{method}-{trait}`,
`…-sft-minus-{trait}`, `…-sft-nip-{nip1|nip2|nip3}`, `…-baseline`.

Running the full sweep cost roughly $1.5–2k (Tinker + judge).

# Experiments

| dir | paper | what it does |
|---|---|---|
| `src/harness/` | — | shared code: load the trait evals, sample (Tinker), judge (two-stage, LiteLLM), train (SFT / GRPO on Tinker), cost accounting |
| `sweep/` | §4.1, §4.5–4.7 | the main sweep: elicit every trait with SP / ICL / SFT / GRPO on three models and run the full battery; minus-pole SFT; nothing-in-particular SFT |
| `introspection/` | §4.3 | ask each model to predict its own SFT spillover |
| `claude_forecasts/` | §4.2, §4.4 | run Claude Opus 5.5 on the 14 forecasting conditions (H1–H9), with the same predictor as the webapp |
| `judge_agreement/` | Limitations | re-judge 1,200 responses with a second judge |
| `export_results.py` | — | convert native outputs into `results/` |

Setup:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # TINKER_API_KEY, LITELLM_BASE_URL, LITELLM_API_KEY
export PYTHONPATH=$PWD/experiments/src PG_EXPERIMENT=sweep
```

Each experiment directory documents its commands in its README / module docstring. Native
outputs are written to `experiments/<name>/results/` (git-ignored); the released data in
`results/` was produced from them with `python experiments/export_results.py`.

Claude forecasts: `python webapp/build_bundle.py` (builds the blinded bundle the predictor reads),
then `python experiments/claude_forecasts/run_forecasts.py all`. Outputs go to `results/claude_forecasts/`.

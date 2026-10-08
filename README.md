# Predicting how model propensities generalize out of distribution

Code and data for the paper *Predicting how model propensities generalize out of distribution*
(Hunter\*, Wen\*, Terry\*, Taraz\*, Cohen, Warncke; \*equal contribution; SPAR Spring 2026).

We elicit one of 24 behavioral propensities (traits) in a language model — by system prompt,
in-context examples, SFT or GRPO — and measure how every other propensity changes
("spillover"). We then test what predicts spillover: other elicitation methods, other models,
symmetry, generic drift, human personality psychology, Claude Opus 5.5, and the model's own
self-prediction.

```
traits/        the 24 trait evals: traits/<trait>/{trait.yaml, eval.yaml, system_prompts/}
experiments/   code to train (Tinker) and evaluate (LLM judge) the elicited models
results/       responses, judge scores, Claude forecasts and derived spillover matrices
figures/       one folder per paper figure: a script that rebuilds it from results/
webapp/        spillover.nielsrolf.com — the blinded spillover-prediction challenge
```

## Reproduce the figures

```bash
pip install -r figures/requirements.txt
python results/download.py          # per-response data from Hugging Face (~720 MB)
figures/run_all.sh                  # rebuilds results/derived/ and every figure
```

Each `figures/<name>/make.py` writes its outputs next to itself. With the pinned versions in
`figures/requirements.txt` the outputs are pixel-identical to the paper.

| paper | folder |
|---|---|
| experiment overview (Fig. 1) | `fig1_overview` |
| §4: which signals predict spillover; §4.2 Claude conditions; appendix: every method / every model as target | `forecast_predictors` |
| §4.1 observed vs. predicted SFT spillover | `forecast_scatter` |
| §4.3 self-prediction (main + appendix) | `introspection` |
| §4.4 human-psychology predictions; appendix H9 matrix and evidence table | `h9_human_psychology` |
| §4.5 symmetry and benign → malevolent spillover | `symmetry` |
| §4.6 factor loadings and Table 1; appendix parallel analysis and cross-model replication | `efa` |
| §4.7 fine-tuning on nothing in particular (+ appendix: other models) | `nip_drift` |
| appendix: SP / ICL / SFT / GRPO matrices of Qwen3.5-9B | `method_heatmaps` |
| appendix: excluded elicitation runs (figure + table) | `failed_runs` |
| appendix: on- vs. off-target effect per run | `on_off_target` |
| appendix: activation space, guide-model choice (earlier iteration; not reproducible here) | `earlier_iteration` |

`figures/lib/spill.py` holds the shared analysis code (spillover definition, failed-run filter,
predictors, scoring); `figures/build_derived.py` writes the spillover matrices and leaderboards to
`results/derived/`.

## Re-run the experiments

See `experiments/README.md`. Training and sampling use [Tinker](https://thinkingmachines.ai/tinker/),
judging uses gpt-5.6-luna through a LiteLLM proxy; credentials go in `.env` (see `.env.example`).

## Data

`results/README.md` documents every file. The per-response data (273 runs × 3,683 responses with
all judge scores, plus the self-prediction answers) is hosted on Hugging Face because of its size;
everything else is in this repository.

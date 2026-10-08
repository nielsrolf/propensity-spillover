"""Minimal in-repo harness for running propensity evals.

No external eval framework (no viseval): everything needed to sample a battery,
judge it and account for its cost lives here.

    from harness import load_battery, CostLedger, Sampler, judge_rows, summarize

Modules
    harness.config      repo paths + experiments/sweep/config.yaml (pricing, pilot params)
    harness.evals       load evals/battery.yaml / evals/<trait>/{trait,eval}.yaml
    harness.cost        CostLedger: real token counts -> $ per (stage, model)
    harness.sampling    Tinker sampling (base plain-text or instruct chat template)
    harness.judge       two-stage gate+metric judge over litellm (disk cached)
    harness.analysis    per-eval means, baseline SD, Cohen's d, markdown tables
    harness.train       elicitation training (train_sft; add train_dpo/train_grpo)
    harness.experiment  run_experiment(): sample + judge + write a results dir
"""

from .analysis import summarize, summary_markdown  # noqa: F401
from .config import RESULTS, ROOT, load_config  # noqa: F401
from .cost import CostLedger, aggregate_costs  # noqa: F401
from .evals import Eval, Prompt, battery_prompts, load_battery, load_eval  # noqa: F401
from .judge import judge_rows, setup_cache  # noqa: F401
from .sampling import Sampler  # noqa: F401

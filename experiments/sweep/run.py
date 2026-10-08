"""CLI for the pilot experiments. Everything is a thin wrapper over `harness`.

    # sample + judge a variant of the battery (writes experiments/sweep/results/<run>/)
    python experiments/sweep/run.py eval --model Qwen/Qwen3.5-9B --variant baseline
    python experiments/sweep/run.py eval --model Qwen/Qwen3.5-9B --variant sysprompt-positive \
        --trait risk-affinity
    python experiments/sweep/run.py eval --model Qwen/Qwen3.5-9B-Base --variant sft \
        --sampler-path tinker://... --trait risk-affinity \
        --baseline-judged experiments/sweep/results/<baseline-run>/judged.csv

    # LoRA SFT on a trait's positive reference responses, then eval it
    python experiments/sweep/run.py train-sft --model Qwen/Qwen3.5-9B-Base --trait risk-affinity
    python experiments/sweep/run.py train-sft --model Qwen/Qwen3.5-9B-Base --trait risk-affinity --then-eval

    # re-judge an existing responses.jsonl.gz (cache makes repeats free)
    python experiments/sweep/run.py judge experiments/sweep/results/<run>/responses.jsonl.gz

    # compare runs: baseline vs one or more variants (points + Cohen's d)
    python experiments/sweep/run.py summarize --baseline experiments/sweep/results/<b>/judged.csv \
        --variant experiments/sweep/results/<v>/judged.csv --target risk-affinity

    # measured cost model across runs
    python experiments/sweep/run.py cost experiments/sweep/results/*/cost.json

Smoke-test flags for cheap live checks: --limit-per-eval / --total-limit / --evals.
Every command prints measured tokens and $ (no estimates anywhere).
Env: TINKER_API_KEY, LITELLM_BASE_URL, LITELLM_API_KEY.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))

from harness import CostLedger, aggregate_costs, load_battery, load_eval  # noqa: E402
from harness.analysis import summarize_paths  # noqa: E402
from harness.config import load_config, run_dir  # noqa: E402
from harness.evals import battery_prompts  # noqa: E402
from harness.experiment import Variant, read_responses, run_experiment  # noqa: E402
from harness.judge import judge_rows, write_judged_csv  # noqa: E402
from harness.train import train_dpo, train_grpo, train_sft  # noqa: E402


def _eval_subset(args):
    names = args.evals.split(",") if args.evals else None
    return load_battery(names)


def cmd_eval(args):
    evals = _eval_subset(args)
    variant = Variant(kind=args.variant, trait=args.trait, sampler_path=args.sampler_path,
                      extra={"shots": args.shots} if args.shots else {})
    res = run_experiment(
        args.model, variant, run_name=args.run_name, evals=evals, split=args.split,
        limit_per_eval=args.limit_per_eval, total_limit=args.total_limit,
        max_tokens=args.max_tokens, judge=not args.no_judge,
        baseline_judged=args.baseline_judged, fmt=args.fmt,
    )
    print(res["summary_md"])


def cmd_train_sft(args):
    cfg = load_config()
    ev = load_eval(args.trait)
    examples = ev.reference_responses(split="train", pole=args.pole)
    if args.max_examples:
        examples = examples[:args.max_examples]
    run_name = args.run_name or f"{args.model.split('/')[-1]}-sft-{args.trait}"
    epochs = args.epochs or cfg["pilot"]["sft_epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["pilot"].get("sft_batch_size")
    ledger = CostLedger(run_name + "-train", model=args.model,
                        meta={"trait": args.trait, "examples": len(examples)})
    print(f"SFT {args.model} on {len(examples)} {args.pole} reference responses "
          f"({epochs} epoch(s), batch {batch_size or len(examples)}, "
          f"LoRA r{cfg['pilot']['lora_rank']})", flush=True)
    if not args.launch:
        print("DRY RUN (pass --launch to spend money)")
        return
    tr = train_sft(args.model, examples, ledger=ledger, epochs=epochs, batch_size=batch_size,
                   lora_rank=args.lora_rank or cfg["pilot"]["lora_rank"],
                   name=run_name)
    print(ledger.save(run_dir(run_name + "-train")))
    print("sampler_path:", tr.sampler_path)
    if args.then_eval:
        res = run_experiment(args.model, Variant(kind="sft", trait=args.trait,
                                                 sampler_path=tr.sampler_path),
                             run_name=run_name, split=args.split,
                             limit_per_eval=args.limit_per_eval, total_limit=args.total_limit,
                             baseline_judged=args.baseline_judged)
        print(res["summary_md"])


def cmd_train_dpo(args):
    """DPO on the trait's preference pairs, then (optionally) eval the battery."""
    cfg = load_config()
    d_cfg = cfg.get("dpo", {})
    ev = load_eval(args.trait)
    pairs = ev.preference_pairs(split="train")
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    epochs = args.epochs or d_cfg.get("epochs", 1)
    batch_size = args.batch_size or d_cfg.get("batch_size", 5)
    lr = args.learning_rate or d_cfg.get("learning_rate", 1e-4)
    beta = args.beta if args.beta is not None else d_cfg.get("beta", 0.1)
    run_name = args.run_name or f"{args.model.split('/')[-1]}-dpo-{args.trait}"
    print(f"DPO {args.model} on {len(pairs)} {args.trait} preference pairs "
          f"({epochs} epoch(s), batch {batch_size} pairs -> "
          f"{epochs * -(-len(pairs) // batch_size)} optim steps, lr={lr}, beta={beta}, "
          f"LoRA r{args.lora_rank or cfg['pilot']['lora_rank']})", flush=True)
    if not args.launch:
        print("DRY RUN (pass --launch to spend money)")
        return
    d = run_dir(run_name + "-train")
    ledger = CostLedger(run_name + "-train", model=args.model,
                        meta={"trait": args.trait, "pairs": len(pairs), "variant": "dpo"})
    tr = train_dpo(args.model, pairs, ledger=ledger, epochs=epochs, batch_size=batch_size,
                   learning_rate=lr, beta=beta,
                   lora_rank=args.lora_rank or cfg["pilot"]["lora_rank"],
                   seed=args.seed, name=run_name,
                   log_path=os.path.join(d, "dpo_steps.jsonl"))
    print(ledger.save(d))
    print("sampler_path:", tr.sampler_path)
    print("dpo loss curve:", [round(x, 4) for x in tr.losses])
    if args.then_eval:
        res = run_experiment(args.model, Variant(kind="dpo", trait=args.trait,
                                                 sampler_path=tr.sampler_path),
                             run_name=run_name, split=args.split,
                             limit_per_eval=args.limit_per_eval, total_limit=args.total_limit,
                             baseline_judged=args.baseline_judged)
        print(res["summary_md"])


def cmd_train_grpo(args):
    """GRPO on the trait's judge score, then (optionally) eval the battery."""
    cfg = load_config()
    g = cfg.get("grpo", {})
    ev = load_eval(args.trait)
    questions = [p.text for p in ev.prompts(split="train", limit=None)]
    if args.max_prompts:
        questions = questions[:args.max_prompts]
    steps = args.steps or g.get("steps", 10)
    group_size = args.group_size or g.get("group_size", 8)
    per_batch = args.prompts_per_batch or g.get("prompts_per_batch", 8)
    run_name = args.run_name or f"{args.model.split('/')[-1]}-grpo-{args.trait}"
    threshold = args.reward_threshold
    if args.threshold_from_judged:
        import csv as _csv
        vals = []
        with open(args.threshold_from_judged) as f:
            for row in _csv.DictReader(f):
                if row.get("eval") != args.trait:
                    continue
                for m in ev.metric_prompts:
                    v = row.get(m)
                    if v not in (None, ""):
                        vals.append(float(v))
        if not vals:
            raise SystemExit(f"no {args.trait} metric scores in {args.threshold_from_judged}")
        threshold = sum(vals) / len(vals) / 100.0
        print(f"reward threshold from {args.threshold_from_judged}: "
              f"{threshold:.3f} (mean of {len(vals)} {args.trait} metric scores / 100)", flush=True)
    passes = steps * per_batch / max(len(questions), 1)
    print(f"GRPO {args.model} on {len(questions)} {args.trait} train questions: "
          f"{steps} steps x {per_batch} prompts x G={group_size} "
          f"({steps * per_batch * group_size} rollouts, {passes:.1f} passes over the set), "
          f"lr={args.learning_rate}, LoRA r{args.lora_rank or cfg['pilot']['lora_rank']}"
          + (f", stop at mean reward >= {threshold:.3f} for {args.threshold_patience} steps"
             if threshold is not None else ""), flush=True)
    if not args.launch:
        print("DRY RUN (pass --launch to spend money)")
        return
    d = run_dir(run_name + "-train")
    ledger = CostLedger(run_name + "-train", model=args.model,
                        meta={"trait": args.trait, "prompts": len(questions), "variant": "grpo"})
    tr = train_grpo(
        args.model, questions, ledger=ledger, eval_obj=ev, steps=steps,
        group_size=group_size, prompts_per_batch=per_batch,
        learning_rate=args.learning_rate, lora_rank=args.lora_rank or cfg["pilot"]["lora_rank"],
        max_tokens=args.max_tokens or cfg["pilot"]["max_sample_tokens"],
        use_gate=not args.no_gate, seed=args.seed, name=run_name,
        reward_threshold=threshold, threshold_patience=args.threshold_patience,
        log_path=os.path.join(d, "grpo_steps.jsonl"),
        rollout_path=os.path.join(d, "rollouts.jsonl"), save_every=args.save_every)
    print(ledger.save(d))
    print("sampler_path:", tr.sampler_path)
    print("reward curve:", [round(r, 3) for r in tr.losses])
    if args.then_eval:
        res = run_experiment(args.model, Variant(kind="grpo", trait=args.trait,
                                                 sampler_path=tr.sampler_path),
                             run_name=run_name, split=args.split,
                             limit_per_eval=args.limit_per_eval, total_limit=args.total_limit,
                             baseline_judged=args.baseline_judged)
        print(res["summary_md"])


def cmd_judge(args):
    rows = read_responses(args.responses)
    evals = load_battery(sorted({r["eval"] for r in rows}))
    d = os.path.dirname(os.path.abspath(args.responses))
    ledger = CostLedger(os.path.basename(d) + "-judge", meta={"rows": len(rows)})
    judged = judge_rows(rows, evals, ledger=ledger)
    print("wrote", write_judged_csv(judged, args.out or os.path.join(d, "judged.csv")))
    # separate filename: never clobber the run's own cost.json
    print(ledger.save(d if args.out is None else os.path.dirname(os.path.abspath(args.out)),
                      filename="cost-judge.json"))


def cmd_summarize(args):
    md = summarize_paths(args.baseline, args.variant, title=args.title, target_eval=args.target)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print("wrote", args.out, file=sys.stderr)


def cmd_cost(args):
    paths = [p for pat in args.paths for p in sorted(glob.glob(pat))] or args.paths
    print(aggregate_costs(paths))


def cmd_prompts(args):
    evals = _eval_subset(args)
    ps = battery_prompts(evals, split=args.split, limit_per_eval=args.limit_per_eval,
                         total_limit=args.total_limit)
    print(f"{len(ps)} prompts across {len(evals)} evals ({args.split} split)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--evals", default=None, help="comma-separated eval subset (default: all 29)")
        p.add_argument("--split", default="test", choices=["train", "test"])
        p.add_argument("--limit-per-eval", type=int, default=None)
        p.add_argument("--total-limit", type=int, default=None)

    p = sub.add_parser("eval", help="sample + judge one variant of the battery")
    add_common(p)
    p.add_argument("--model", default=None)
    p.add_argument("--variant", default="baseline")
    p.add_argument("--trait", default=None, help="elicitation target (sysprompt/sft/dpo/grpo)")
    p.add_argument("--sampler-path", default=None, help="tinker:// trained weights")
    p.add_argument("--run-name", default=None)
    p.add_argument("--baseline-judged", default=None, help="judged.csv of the baseline run")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--fmt", default="auto", choices=["auto", "base", "chat"])
    p.add_argument("--shots", type=int, default=None,
                   help="variant=icl: number of few-shot demonstrations (default pilot.icl_shots)")
    p.add_argument("--no-judge", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("train-sft", help="LoRA SFT on a trait's reference responses")
    add_common(p)
    p.add_argument("--model", default=None)
    p.add_argument("--trait", required=True)
    p.add_argument("--pole", default="positive", choices=["positive", "negative"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="examples per optimizer step (0 = full batch)")
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--launch", action="store_true", help="actually train (costs money)")
    p.add_argument("--then-eval", action="store_true", help="eval the battery after training")
    p.add_argument("--baseline-judged", default=None)
    p.set_defaults(func=cmd_train_sft)

    p = sub.add_parser("train-dpo", help="LoRA DPO on the trait's preference pairs")
    add_common(p)
    p.add_argument("--model", default=None)
    p.add_argument("--trait", required=True)
    p.add_argument("--epochs", type=int, default=None, help="passes over the full pair set")
    p.add_argument("--batch-size", type=int, default=None, help="preference pairs per optim step")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--beta", type=float, default=None, help="DPO KL coefficient")
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--max-pairs", type=int, default=None, help="cap the elicitation set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--launch", action="store_true", help="actually train (costs money)")
    p.add_argument("--then-eval", action="store_true")
    p.add_argument("--baseline-judged", default=None)
    p.set_defaults(func=cmd_train_dpo)

    p = sub.add_parser("train-grpo", help="GRPO with the trait's judge score as reward")
    add_common(p)
    p.add_argument("--model", default=None)
    p.add_argument("--trait", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--prompts-per-batch", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-prompts", type=int, default=None, help="cap the elicitation set")
    p.add_argument("--no-gate", action="store_true", help="reward = metric only, skip the gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=None, help="checkpoint every N steps")
    p.add_argument("--reward-threshold", type=float, default=None,
                   help="early-stop when mean reward (0-1) stays >= this; steps stays the cost cap")
    p.add_argument("--threshold-from-judged", default=None, metavar="JUDGED_CSV",
                   help="derive --reward-threshold from a judged.csv (e.g. the same model's SFT "
                        "battery): mean of the target trait's metric scores / 100")
    p.add_argument("--threshold-patience", type=int, default=2,
                   help="consecutive steps at/above the threshold before stopping (default 2)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--launch", action="store_true", help="actually train (costs money)")
    p.add_argument("--then-eval", action="store_true")
    p.add_argument("--baseline-judged", default=None)
    p.set_defaults(func=cmd_train_grpo)

    p = sub.add_parser("judge", help="judge a responses.jsonl.gz")
    p.add_argument("responses")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("summarize", help="baseline vs variants: points + Cohen's d")
    p.add_argument("--baseline", required=True)
    p.add_argument("--variant", action="append", default=[], required=True)
    p.add_argument("--target", default=None, help="elicitation target eval (bolded)")
    p.add_argument("--title", default="Per-eval scores")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("cost", help="aggregate measured cost.json files")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("prompts", help="count battery prompts (no API calls)")
    add_common(p)
    p.set_defaults(func=cmd_prompts)

    args = ap.parse_args()
    if getattr(args, "model", None) is None and hasattr(args, "model"):
        args.model = load_config()["models"]["study"][0]
    args.func(args)


if __name__ == "__main__":
    main()

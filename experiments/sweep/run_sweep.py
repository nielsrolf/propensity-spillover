#!/usr/bin/env python3
"""Sweep orchestrator.

Runs the full elicitation-spillover sweep:
  - traits SEQUENTIALLY (risk-affinity first, then alphabetical),
  - within a trait, all (model, method) conditions IN PARALLEL
    (bounded by the 4 GiB cgroup: <=4 subprocesses, <=2 torch trainings).

Resumable across container restarts: every finished step is committed+pushed,
and done-ness is decided from committed artifacts only:
  - train step done  <=> its log file contains "sampler_path: tinker://"
  - eval  step done  <=> results/<run>/judged.csv exists (written at completion)

Usage:
  python3 experiments/sweep/run_sweep.py [--max-minutes 90] [--traits a,b]
          [--dry-run] [--no-push]
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # experiments/sweep
REPO = HERE.parent.parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
CFG = yaml.safe_load((HERE / "config.yaml").read_text())

SHORT = {
    "Qwen/Qwen3.5-9B-Base": "qwen3.5-9b-base",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron-3-super",
}
METHODS = CFG["sweep"]["methods"]               # model -> [methods]
HP = CFG["sweep"].get("hp_overrides", {})       # model -> per-model HP overrides
G = CFG["grpo"]

ENV = dict(
    os.environ,
    PG_EXPERIMENT="sweep",
    PYTHONPATH=str(REPO / "experiments" / "src"),
    HARNESS_SAMPLE_CONCURRENCY=os.environ.get("HARNESS_SAMPLE_CONCURRENCY", "24"),
    JUDGE_CONCURRENCY=os.environ.get("JUDGE_CONCURRENCY", "12"),
)

GIT_LOCK = threading.Lock()
TRAIN_SEM = threading.Semaphore(int(CFG["sweep"].get("max_parallel_train", 2)))
DEADLINE = None  # set in main()
STOP = threading.Event()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def checkpoint_loop(interval=600):
    """Every `interval` s, commit+push streamed partial results and logs.

    The orchestrator can be SIGKILLed at any time (container teardown); this
    bounds the loss to <= `interval` seconds of sampling. experiment.py resumes
    from a committed responses.partial.jsonl on the next pass.
    """
    while not STOP.wait(interval):
        try:
            paths = [RESULTS, LOGS]
            # keep the doc's live figure fresh: rebuild the aggregate + widget
            # HTML from whatever is finished, so even a widget refresh that
            # cannot reach git shows current numbers.
            try:
                for builder in ("widgets/build_sweep_dashboard.py",
                                "widgets/build_spillover_scatter.py"):
                    subprocess.run(["python3", builder], cwd=REPO,
                                   capture_output=True, timeout=300)
                paths += [REPO / "assets/sweep_dashboard.html",
                          REPO / "assets/spillover_scatter.html",
                          REPO / "widgets/data/sweep_aggregate.json"]
            except Exception as e:  # noqa: BLE001
                log(f"WARN dashboard rebuild failed: {e!r}")
            git_commit_push(paths,
                            "sweep: checkpoint (partial results + logs + figure)",
                            push=not ARGS.no_push)
        except Exception as e:  # noqa: BLE001
            log(f"WARN checkpoint commit failed: {e!r}")


def traits():
    all_traits = sorted(p.name for p in (REPO / "traits").iterdir()
                        if (p / "eval.yaml").exists())
    first = [t for t in CFG["sweep"].get("trait_order_first", []) if t in all_traits]
    return first + [t for t in all_traits if t not in first]


def git_commit_push(paths, message, push=True):
    """Optional: commit (and push) each finished step. Off unless SWEEP_GIT_SYNC=1 — the original
    sweep ran in a container that synced every result to the lab repo."""
    if os.environ.get("SWEEP_GIT_SYNC") != "1":
        return False
    with GIT_LOCK:
        subprocess.run(["git", "add", "--"] + [str(p) for p in paths],
                       cwd=REPO, check=False)
        r = subprocess.run(
            ["git", "-c", "user.name=claudex", "-c", "user.email=claudex@nielsrolf.local",
             "commit", "-m", message], cwd=REPO, capture_output=True, text=True)
        committed = r.returncode == 0
        if committed and push:
            for _ in range(3):
                p = subprocess.run(
                    ["git", "-c", "credential.helper=!gh auth git-credential",
                     "push", "origin", "master"],
                    cwd=REPO, capture_output=True, text=True)
                if p.returncode == 0:
                    return True
                # non-ff: someone else pushed; rebase and retry
                subprocess.run(["git", "pull", "--rebase", "origin", "master"],
                               cwd=REPO, capture_output=True, text=True)
            log(f"WARN: push failed for '{message}'")
        return committed


def run_cmd(argv, logfile, timeout=7200):
    """Run a run.py subcommand, teeing output to a log file."""
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a") as fh:
        fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(argv)}\n")
        fh.flush()
        r = subprocess.run(argv, cwd=REPO, env=ENV, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def sampler_path_from_log(logfile):
    if not Path(logfile).exists():
        return None
    m = re.findall(r"sampler_path: (tinker://\S+)", Path(logfile).read_text())
    return m[-1] if m else None


def eval_done(run_name):
    return (RESULTS / run_name / "judged.csv").exists()


def time_left():
    return DEADLINE is None or time.time() < DEADLINE


RUNPY = str(HERE / "run.py")


def do_eval(run_name, model, variant, trait=None, sampler=None, shots=None):
    """Run one battery eval+judge; commit results. Returns True on success."""
    if eval_done(run_name):
        return True
    if not time_left() or STOP.is_set():
        return False
    base_judged = RESULTS / f"{SHORT[model]}-baseline" / "judged.csv"
    argv = ["python3", RUNPY, "eval", "--model", model, "--variant", variant,
            "--run-name", run_name]
    if trait:
        argv += ["--trait", trait]
    if sampler:
        argv += ["--sampler-path", sampler]
    if shots:
        argv += ["--shots", str(shots)]
    if variant != "baseline" and base_judged.exists():
        argv += ["--baseline-judged", str(base_judged)]
    logfile = LOGS / f"{run_name}.log"
    log(f"eval start: {run_name}")
    rc = run_cmd(argv, logfile)
    ok = rc == 0 and eval_done(run_name)
    log(f"eval {'done' if ok else f'FAILED rc={rc}'}: {run_name}")
    if eval_done(run_name):
        git_commit_push([RESULTS / run_name, logfile], f"sweep: {run_name}",
                        push=not ARGS.no_push)
    return ok


def do_train(kind, run_name, model, trait):
    """Train sft/grpo; commit the log (contains sampler_path). Returns sampler path or None."""
    logfile = LOGS / f"{run_name}-train.log"
    sp = sampler_path_from_log(logfile)
    if sp:
        return sp
    if not time_left() or STOP.is_set():
        return None
    argv = ["python3", RUNPY, f"train-{kind}", "--model", model, "--trait", trait,
            "--run-name", run_name, "--launch"]
    hp = HP.get(model, {})
    if kind == "sft" and "sft_epochs" in hp:
        argv += ["--epochs", str(hp["sft_epochs"])]
    if kind == "grpo":
        argv += ["--steps", str(G["steps"]),
                 "--prompts-per-batch", str(G["prompts_per_batch"]),
                 "--group-size", str(G["group_size"]),
                 "--learning-rate", str(hp.get("grpo_learning_rate",
                                               G["learning_rate"])),
                 "--reward-threshold", str(G["reward_threshold"]),
                 "--threshold-patience", str(G["threshold_patience"])]
    with TRAIN_SEM:
        if not time_left() or STOP.is_set():
            return None
        log(f"train start: {run_name} ({kind})")
        rc = run_cmd(argv, logfile)
    sp = sampler_path_from_log(logfile)
    log(f"train {'done' if sp else f'FAILED rc={rc}'}: {run_name}")
    train_dir = RESULTS / f"{run_name}-train"
    paths = [logfile] + ([train_dir] if train_dir.exists() else [])
    git_commit_push(paths, f"sweep: {run_name}-train", push=not ARGS.no_push)
    return sp


def do_condition(model, method, trait):
    """One (model, method, trait) condition end-to-end. Returns True if complete."""
    short = SHORT[model]
    run_name = f"{short}-{method}-{trait}"
    if eval_done(run_name):
        return True
    try:
        if method == "sysprompt":
            return do_eval(run_name, model, "sysprompt-positive", trait=trait)
        if method == "icl":
            return do_eval(run_name, model, "icl", trait=trait,
                           shots=CFG["pilot"]["icl_shots"])
        if method in ("sft", "grpo"):
            sp = do_train(method, run_name, model, trait)
            if not sp:
                return False
            return do_eval(run_name, model, method, trait=trait, sampler=sp)
        raise ValueError(method)
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {run_name}")
        return False
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {run_name}: {e!r}")
        return False


def main():
    global DEADLINE, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop starting new work after this long")
    ap.add_argument("--traits", default=None, help="comma-separated subset")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma-separated <model-short>:<method> filter, e.g. "
                         "qwen3.5-9b-base:icl (backfill a single arm)")
    ap.add_argument("--flat", action="store_true",
                    help="pool ALL (trait, condition) pairs into one parallel "
                         "queue instead of running traits sequentially. Use for "
                         "single-arm backfills where per-trait batching would "
                         "leave the pool idle.")
    ARGS = ap.parse_args()
    only = set(ARGS.only.split(",")) if ARGS.only else None
    if ARGS.max_minutes:
        DEADLINE = time.time() + ARGS.max_minutes * 60

    todo_traits = (ARGS.traits.split(",") if ARGS.traits else traits())
    plan = []
    # Phase 0: baselines (one per model, shared by all conditions of that model)
    baselines = [(m, f"{SHORT[m]}-baseline") for m in METHODS]
    # Phase 1+: per-trait conditions
    for t in todo_traits:
        conds = [(m, meth) for m in METHODS for meth in METHODS[m]
                 if only is None or f"{SHORT[m]}:{meth}" in only]
        plan.append((t, conds))

    n_total = len(baselines) + sum(len(c) for _, c in plan)
    n_done = sum(eval_done(rn) for _, rn in baselines) + sum(
        eval_done(f"{SHORT[m]}-{meth}-{t}") for t, conds in plan for m, meth in conds)
    log(f"sweep: {n_done}/{n_total} conditions already done; "
        f"{len(todo_traits)} traits queued")
    if ARGS.dry_run:
        for m, rn in baselines:
            log(f"  baseline {rn}: {'DONE' if eval_done(rn) else 'todo'}")
        for t, conds in plan:
            missing = [f"{SHORT[m]}-{meth}" for m, meth in conds
                       if not eval_done(f"{SHORT[m]}-{meth}-{t}")]
            if missing:
                log(f"  {t}: todo {missing}")
        return 0

    from concurrent.futures import ThreadPoolExecutor
    max_par = int(CFG["sweep"].get("max_parallel", 4))

    threading.Thread(target=checkpoint_loop, daemon=True).start()

    # Baselines first, in parallel (everything downstream normalizes against them).
    with ThreadPoolExecutor(max_workers=2) as ex:
        oks = list(ex.map(lambda mr: do_eval(mr[1], mr[0], "baseline"), baselines))
    if not all(oks):
        if not time_left():
            log("time budget hit during baselines; resume later")
            return 0
        log("FATAL: a baseline battery failed; fix before sweeping")
        return 1

    # Flat mode: one global queue over every pending (trait, condition).
    if ARGS.flat:
        pending = [(t, m, meth) for t, conds in plan for m, meth in conds
                   if not eval_done(f"{SHORT[m]}-{meth}-{t}")]
        log(f"=== flat mode: {len(pending)} conditions, {max_par} at a time ===")
        with ThreadPoolExecutor(max_workers=max_par) as ex:
            results = list(ex.map(lambda c: do_condition(c[1], c[2], c[0]), pending))
        log(f"=== flat: {sum(results)}/{len(pending)} completed ===")
        STOP.set()
        git_commit_push([RESULTS, LOGS], "sweep: flat pass finished",
                        push=not ARGS.no_push)
        return 0

    # Traits sequentially; conditions of one trait in parallel.
    for t, conds in plan:
        if not time_left() or STOP.is_set():
            log("time budget hit; resume with the same command")
            return 0
        pending = [(m, meth) for m, meth in conds
                   if not eval_done(f"{SHORT[m]}-{meth}-{t}")]
        if not pending:
            continue
        log(f"=== trait {t}: {len(pending)} conditions ===")
        with ThreadPoolExecutor(max_workers=max_par) as ex:
            results = list(ex.map(lambda c: do_condition(c[0], c[1], t), pending))
        log(f"=== trait {t}: {sum(results)}/{len(pending)} completed ===")

    log("sweep pass finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

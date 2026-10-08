"""Repo paths and the active experiment's config file.

Layout (release repo):
    traits/                      battery.yaml + traits/<trait>/{trait,eval}.yaml
    experiments/src/harness/     this package
    experiments/<name>/          one directory per experiment (config + results/)

`EXPERIMENT` selects which experiment directory config/results resolve to; it
defaults to `sweep` and can be overridden with $PG_EXPERIMENT.
"""

import functools
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EVALS_DIR = os.path.join(ROOT, "traits")
BATTERY = os.path.join(EVALS_DIR, "battery.yaml")
EXPERIMENTS = os.path.join(ROOT, "experiments")
EXPERIMENT = os.environ.get("PG_EXPERIMENT", "sweep")
EXPERIMENT_DIR = os.path.join(EXPERIMENTS, EXPERIMENT)
CONFIG = os.path.join(EXPERIMENT_DIR, "config.yaml")
RESULTS = os.path.join(EXPERIMENT_DIR, "results")
CACHE_DIR = os.path.join(ROOT, ".litellm_cache")


@functools.lru_cache(maxsize=None)
def load_config(path: str = CONFIG) -> dict:
    """<experiment>/config.yaml: {pilot: ..., models: {study: [...]}, pricing: {...}}."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_dir(run_name: str, make: bool = True) -> str:
    """<experiment>/results/<run_name>/ — one directory per experiment run."""
    d = os.path.join(RESULTS, run_name)
    if make:
        os.makedirs(d, exist_ok=True)
    return d

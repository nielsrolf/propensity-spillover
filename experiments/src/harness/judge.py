"""Two-stage judge (evidence gate -> metric rubrics) over litellm, disk-cached.

Stage 1: the eval's `provides_evidence` gate runs once per (question, response).
Stage 2: if gate >= evidence_gate_threshold, every metric rubric of that eval
runs; otherwise all metrics are null.

Judge model: gpt-5.6-luna, reasoning_effort=minimal, max_tokens=2048.
DO NOT lower max_tokens: luna's reasoning tokens are billed against it, and
max_tokens=16 produced empty content for ~94% of calls. Empty content is
retried.

Caching: litellm's disk cache (.litellm_cache/, gitignored) keyed on the
request, so identical judge prompts across reruns/variants cost nothing. Cache
hits are counted in the ledger meta (`judge_cache_hits`) and billed at $0.
"""

import csv
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from .config import CACHE_DIR
from .cost import CostLedger
from .evals import Eval

JUDGE_MODEL = os.environ.get("HARNESS_JUDGE_MODEL", "openai/gpt-5.6-luna")
DEFAULT_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "32"))
MAX_TOKENS = 2048
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

_setup_lock = threading.Lock()
_setup_done = False


def setup_cache(disk_cache_dir: str = CACHE_DIR) -> None:
    """Enable litellm's disk cache for completion calls (idempotent)."""
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return
        import litellm
        from litellm.caching.caching import Cache

        os.makedirs(disk_cache_dir, exist_ok=True)
        litellm.cache = Cache(type="disk", disk_cache_dir=disk_cache_dir)
        litellm.enable_cache()
        litellm.drop_params = True
        litellm.suppress_debug_info = True
        _setup_done = True


def judge_call(prompt: str, ledger: Optional[CostLedger] = None, model: str = JUDGE_MODEL,
               retries: int = 4) -> Optional[float]:
    """One judge call -> the first number in the reply, or None. Cost-accounted."""
    import litellm

    setup_cache()
    for attempt in range(retries):
        try:
            r = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0,
                reasoning_effort="minimal",
                caching=True,
                api_base=os.environ["LITELLM_BASE_URL"],
                api_key=os.environ["LITELLM_API_KEY"],
            )
            hit = bool((getattr(r, "_hidden_params", None) or {}).get("cache_hit"))
            if ledger is not None:
                ledger.add_judge_usage(getattr(r, "usage", None), cache_hit=hit)
            txt = (r.choices[0].message.content or "").strip()
            m = NUM_RE.search(txt)
            if m:
                return float(m.group())
            # empty content (budget eaten by reasoning tokens) -> retry uncached
        except Exception:
            pass
    return None


def judge_row(row: dict, ev: Eval, ledger: Optional[CostLedger] = None,
              model: str = JUDGE_MODEL) -> dict:
    """Judge one sampled row: {eval, question, response, ...} -> gate + metric scores."""
    q, a = row["question"], row["response"]
    out = {k: row.get(k) for k in ("eval", "question_id", "paraphrase_index", "sample_index")}
    gate = judge_call(ev.fill(ev.gate_prompt, q, a), ledger, model)
    out["gate"] = gate
    passed = gate is not None and gate >= ev.gate_threshold
    for name, tmpl in ev.metric_prompts.items():
        out[name] = judge_call(ev.fill(tmpl, q, a), ledger, model) if passed else None
    return out


def judge_rows(rows: Sequence[dict], evals: Dict[str, Eval], ledger: Optional[CostLedger] = None,
               concurrency: int = DEFAULT_CONCURRENCY, model: str = JUDGE_MODEL,
               progress_every: int = 250) -> List[dict]:
    """Judge all rows with a response, in parallel. Rows keep input order."""
    setup_cache()
    rows = [r for r in rows if r.get("response")]
    done = [0]
    lock = threading.Lock()

    def one(r):
        res = judge_row(r, evals[r["eval"]], ledger, model)
        with lock:
            done[0] += 1
            if progress_every and done[0] % progress_every == 0:
                print(f"judged {done[0]}/{len(rows)}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, rows))


def write_judged_csv(judged: Sequence[dict], path: str) -> str:
    base = ["eval", "question_id", "paraphrase_index", "sample_index", "gate"]
    metrics = sorted({k for r in judged for k in r} - set(base))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base + metrics)
        w.writeheader()
        w.writerows(judged)
    return path

"""Run the blinded spillover predictor (webapp/server/agent.py) on the H# specs in
webapp/hypotheses/ against the blinded bundle (BUNDLE_ROOT = webapp/; build its inputs/
with `python webapp/build_bundle.py` first).

  python run_forecasts.py H9                 # one hypothesis
  python run_forecasts.py all                # the 14 runs (H1..H6, H7a_r1..3, H7b_r1..3, H8, H9)
  python run_forecasts.py H9 --force         # re-run even if predictions/H9 is complete
  python run_forecasts.py --smoke            # exercise the tool layer, no model calls

Resumable: a hypothesis whose results/claude_forecasts/<H>/result.json has status "done" is skipped.
Outputs per run in results/claude_forecasts/<H>/: logitz_plus.csv, logitz_minus.csv, method.md,
falsifiers.md, inspection.json, result.json (status/usage/cost/wall time), usage_calls.jsonl
(per-API-call usage), transcript.jsonl (assistant messages + tool results), progress.log.
Credentials: LITELLM_BASE_URL / LITELLM_API_KEY from the environment, else from ENV_FILE.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BUNDLE = os.path.join(REPO, "webapp")            # inputs/, BLINDED_PROMPT.md, hypotheses/
WEBAPP = os.path.join(BUNDLE, "server")           # agent.py (the same predictor the webapp runs)
SPECS = os.path.join(BUNDLE, "hypotheses")
PRED = os.path.join(REPO, "results", "claude_forecasts")
ENV_FILE = os.path.join(REPO, ".env")
ALL = ["H1", "H2", "H3", "H4", "H5", "H6", "H7a_r1", "H7a_r2", "H7a_r3",
       "H7b_r1", "H7b_r2", "H7b_r3", "H8", "H9"]

# USD per token, Claude Opus 5.5 (Anthropic list price: $4/M in, $20/M out,
# cache read $0.20/M, 5-min cache write 1.25x input = $5/M).
PRICES = {
    "anthropic/claude-opus-5-5": dict(input=4e-6, output=20e-6, cache_read=0.2e-6, cache_write=5e-6),
    "anthropic/claude-opus-4-7": dict(input=5e-6, output=25e-6, cache_read=0.5e-6, cache_write=6.25e-6),
}


def _load_env():
    if os.environ.get("LITELLM_API_KEY") and os.environ.get("LITELLM_BASE_URL"):
        return
    for line in (open(ENV_FILE) if os.path.exists(ENV_FILE) else []):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.replace("export ", "").strip()
            if k in ("LITELLM_API_KEY", "LITELLM_BASE_URL"):
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


os.environ["BUNDLE_ROOT"] = BUNDLE
os.environ.setdefault("PREDICTOR_MODEL", "anthropic/claude-opus-5-5")
sys.path.insert(0, WEBAPP)
import agent  # noqa: E402  (reads BUNDLE_ROOT / PREDICTOR_MODEL at import)

# The webapp's tool descriptions hard-code the old 29/14 counts; fix them for this bundle
# (in memory only — agent.py itself is not modified).
_n, _m = len(agent.EVALS), len(agent.MINUS_ROWS)
for t in agent.TOOLS:
    f = t["function"]
    f["description"] = (f["description"].replace("29x29", f"{_n}x{_n}")
                        .replace("each of the 29 '<eval>-plus'", f"each of the {_n} '<eval>-plus'")
                        .replace("each of the 14 '<eval>-minus'", f"each of the {_m} '<eval>-minus'")
                        .replace("list of 29 numbers", f"list of {_n} numbers"))


def _streamed(create, kwargs):
    """Make the call with stream=True and reassemble a ChatCompletion.

    The public LiteLLM endpoint sits behind Cloudflare, which kills non-streaming requests
    after 120 s (HTTP 524). Turn-2 submissions (20k+ output tokens) take longer than that,
    so we stream to keep bytes flowing; the agent still sees an ordinary ChatCompletion.
    """
    from openai.types.chat import ChatCompletion
    stream = create(**kwargs, stream=True, stream_options={"include_usage": True})
    content, tools, finish, usage, model, cid = [], {}, None, None, kwargs.get("model"), "stream"
    for chunk in stream:
        cid, model = chunk.id or cid, chunk.model or model
        if chunk.usage:
            usage = chunk.usage.model_dump()
        for ch in chunk.choices or []:
            d = ch.delta
            if d.content:
                content.append(d.content)
            for tc in d.tool_calls or []:
                slot = tools.setdefault(tc.index, {"id": None, "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments
            if ch.finish_reason:
                finish = ch.finish_reason
    msg = {"role": "assistant", "content": "".join(content) or None}
    if tools:
        msg["tool_calls"] = [tools[i] for i in sorted(tools)]
    return ChatCompletion.model_validate({
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish or "stop"}],
        "usage": usage})


class _Recorder:
    """Wraps OpenAI() so every chat.completions.create call logs full usage + a transcript."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.calls = []
        self._seen = 0

    def factory(self, *a, **kw):
        from openai import OpenAI
        client = OpenAI(*a, **kw)
        orig = client.chat.completions.create
        rec = self

        def create(**kwargs):
            msgs = kwargs.get("messages", [])
            t0 = time.time()
            resp = _streamed(orig, kwargs)
            u = resp.usage
            d = getattr(u, "prompt_tokens_details", None)
            row = {
                "t": round(time.time() - t0, 1),
                "prompt_tokens": u.prompt_tokens if u else 0,
                "completion_tokens": u.completion_tokens if u else 0,
                "cache_read": (getattr(d, "cached_tokens", 0) or 0) if d else 0,
                "cache_write": (getattr(d, "cache_creation_tokens", 0) or 0) if d else 0,
                "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
                "finish_reason": resp.choices[0].finish_reason,
                "model": resp.model,
            }
            rec.calls.append(row)
            with open(os.path.join(rec.out_dir, "usage_calls.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            with open(os.path.join(rec.out_dir, "transcript.jsonl"), "a") as f:
                for m in msgs[max(rec._seen, 1):]:
                    if m["role"] != "assistant":
                        c = m["content"]
                        if isinstance(c, list):
                            c = "".join(p.get("text", "") for p in c)
                        f.write(json.dumps({"role": m["role"], "content": c[:4000]}) + "\n")
                msg = resp.choices[0].message
                f.write(json.dumps({"role": "assistant", "content": msg.content,
                                    "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments[:4000]}
                                                   for tc in (msg.tool_calls or [])]}) + "\n")
            rec._seen = len(msgs) + 1
            return resp

        client.chat.completions.create = create
        return client


def cost_of(calls, model):
    p = PRICES.get(model)
    if not p:
        return None
    usd = 0.0
    for c in calls:
        uncached = c["prompt_tokens"] - c["cache_read"] - c["cache_write"]
        usd += (uncached * p["input"] + c["cache_read"] * p["cache_read"]
                + c["cache_write"] * p["cache_write"] + c["completion_tokens"] * p["output"])
    return round(usd, 4)


def run_one(hid, force=False):
    out = os.path.join(PRED, hid)
    res_path = os.path.join(out, "result.json")
    if not force and os.path.exists(res_path) and json.load(open(res_path)).get("status") == "done":
        print(f"[{hid}] already done, skipping")
        return json.load(open(res_path))
    spec = open(os.path.join(SPECS, f"{hid}.md")).read()
    os.makedirs(out, exist_ok=True)
    for f in ("usage_calls.jsonl", "transcript.jsonl", "progress.log"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    rec = _Recorder(out)
    agent.OpenAI = rec.factory
    log = open(os.path.join(out, "progress.log"), "a")

    def on_progress(msg, usage):
        line = f"{time.strftime('%H:%M:%S')} [{usage.get('steps')}] {msg}"
        print(f"[{hid}] {line}", flush=True)
        log.write(line + "\n")
        log.flush()

    t0 = time.time()
    try:
        res = agent.run_agent(hid[1:], spec, on_progress)
    except Exception as e:  # noqa: BLE001
        res = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    wall = round(time.time() - t0, 1)
    model = agent.MODEL
    if res.get("status") == "done":
        open(os.path.join(out, "logitz_plus.csv"), "w").write(res["plus_csv"])
        open(os.path.join(out, "logitz_minus.csv"), "w").write(res["minus_csv"])
    for k, fn in (("method_md", "method.md"), ("falsifiers_md", "falsifiers.md"),
                  ("clarification_md", "NEEDS_CLARIFICATION.md")):
        if res.get(k):
            open(os.path.join(out, fn), "w").write(res[k])
    if res.get("inspection"):
        json.dump(res["inspection"], open(os.path.join(out, "inspection.json"), "w"), indent=1)
    summary = {
        "id": hid, "status": res.get("status"), "error": res.get("error"), "model": model,
        "wall_s": wall, "usage": res.get("usage"),
        "api_calls": len(rec.calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in rec.calls),
        "completion_tokens": sum(c["completion_tokens"] for c in rec.calls),
        "cache_read_tokens": sum(c["cache_read"] for c in rec.calls),
        "cache_write_tokens": sum(c["cache_write"] for c in rec.calls),
        "est_cost_usd": cost_of(rec.calls, model),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(summary, open(res_path, "w"), indent=1)
    with open(os.path.join(PRED, "usage_log.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{hid}] {summary['status']} in {wall}s, cost ~${summary['est_cost_usd']}")
    return summary


def smoke():
    """Call every tool function directly (no model)."""
    ws = agent.Workspace("9", "# test spec\nthesis")
    print("EVALS", len(agent.EVALS), "PLUS", len(agent.PLUS_ROWS), "MINUS", len(agent.MINUS_ROWS))
    print(ws.list_files(""))
    print(ws.list_files("inputs/evals_orthogonalized")[:400])
    print(ws.read_file("BLINDED_PROMPT.md", 0, 5))
    print(ws.read_file("hypotheses/H9.md"))
    print(ws.search_files("punitiveness", "inputs/**")[:300])
    for e in agent.EVALS:
        ov = json.loads(ws.eval_overview(e))
        items = json.loads(ws.eval_items(e, "test", 0, 2, True))
        tr = json.loads(ws.eval_items(e, "train", 0, 1, True))
        assert ov["n_items"] > 0 and items["items"] and tr["items"], e
        assert "positive.txt" in ov["system_prompts"], e
        assert ("negative.txt" in ov["system_prompts"]) == (f"{e}-minus" in agent.MINUS_ROWS), e
        m = tr["items"][0]["meta"]
        assert "expected_plus_response" in m, e
        print(f"  {e}: n={ov['n_items']} splits={ov['splits']} judges={list(ov['judge_prompts_first_item'])} "
              f"sp={list(ov['system_prompts'])}")
    for v in ("embed_without_preamble", "embed_with_preamble", "token_without_preamble", "token_with_preamble"):
        txt = ws.judge_overlap(v)
        rows = txt.splitlines()[1:]
        assert len(rows) == len(agent.EVALS) + 1 and all(len(r.split(",")) == len(agent.EVALS) + 1 for r in rows), v
    print(ws.judge_overlap()[:300])
    for bad in ("RESULTS/x.csv", "../.env", "predictions/H9/logitz_plus.csv"):
        try:
            ws.read_file(bad)
            raise SystemExit(f"LEAK: {bad} readable")
        except ValueError:
            pass
    print("submit before inspection ->", ws.submit_predictions({}, {}, "", "")[:60])
    ws.record_inspection("none", "test")
    print("bad submit ->", ws.submit_predictions({"x": [1]}, {}, "", "")[:200])
    plus = {r: [0.0] * len(agent.EVALS) for r in agent.PLUS_ROWS}
    minus = {r: [None] * len(agent.EVALS) for r in agent.MINUS_ROWS}
    try:
        ws.submit_predictions(plus, minus, "m", "f")
    except agent.Stop:
        print("good submit -> status", ws.result["status"], "plus rows", ws.result["plus_csv"].count("\n") - 1,
              "minus rows", ws.result["minus_csv"].count("\n") - 1)
    for t in agent.TOOLS:
        if t["function"]["name"] in ("judge_overlap", "submit_predictions"):
            print("tool desc:", t["function"]["description"][:160])
    print("SMOKE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
        sys.exit(0)
    _load_env()
    ids = ALL if a.ids == ["all"] else a.ids
    for hid in ids:
        assert hid in ALL, hid
        run_one(hid, a.force)

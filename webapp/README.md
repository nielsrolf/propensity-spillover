# spillover.nielsrolf.com — web version of the challenge

A single-container web app that removes the Claude Code / CLI step for participants:

1. **Intro deck**: one idea per slide (scroll, arrow keys or swipe to advance).
2. **Eval browser**: each of the 24 evals (21 bipolar, 3 plus-only ethical frameworks) with its
   poles, judge rubric(s), evidence gate, system prompts, sample test items with their per-pole
   reference answers (the SFT targets), and direction-trap warnings.
3. **My entry**: claim a handle (you get a private link that acts as the password), then:
   - **Write spec**: only the thesis is required. Definitions, literature, operationalization,
     prerequisites and falsifiers are optional, and the predictor fills in what's missing.
   - **Generate**: runs the `BLINDED_PROMPT.md` protocol server-side (Turn 1 inspection, then
     Turn 2 generation) and shows live progress, heatmaps, `method.md` and `falsifiers.md`.
   - **Matrices**: optional hand-editing of cells, plus CSV import/export.
   - **Submit**: final. It locks the entry and unlocks the host's H1–H9 predictions for comparison.
4. **H1–H9**: the host's framings as public examples ("start my entry from this"). Their frozen
   predictions (Claude Opus 5.5, made with this same `agent.py` harness via the forecast bundle's
   `run_forecasts.py`) are only returned by the API to participants whose own entry is submitted.
5. **Entries**: the public list of submitted handles and titles. Contents stay private until freeze.

## Layout

```
BLINDED_PROMPT.md, EXAMPLE_hypothesis.md   the task protocol the predictor follows
inputs/                                    blinded bundle; build the eval files with `python build_bundle.py`
hypotheses/                                the host's H1–H9 specs (also used by experiments/claude_forecasts)
server/                                    FastAPI app, predictor (agent.py), static frontend, Dockerfile
stage.sh                                   copy the blinding-safe files into a Docker build context
```

## How it differs from `scripts/generate.sh`

The predictor (`agent.py`) is a tool-use loop over the LiteLLM proxy, not Claude Code. On a
public server a spec could carry instructions, so the model gets **no shell or Python**. It only
has read-only access to `inputs/`, `BLINDED_PROMPT.md` and the participant's own spec, plus an
eval-level judge-overlap table and a structured `submit_predictions` tool. Model calls are
streamed (the LiteLLM endpoint sits behind Cloudflare, which cuts non-streaming requests after
~120 s), with retries. The host's H1–H9 were generated with this same harness and model, so web
runs are comparable with them; CLI runs (Claude Code, can execute code) are not strictly
apples-to-apples. The About page says so.

## Blinding

- `stage.sh <dir>` copies **only** `inputs/`, `BLINDED_PROMPT.md`, `EXAMPLE_hypothesis.md` and
  `webapp/` into a clean build context. Always deploy from that directory, never from the repo
  root, so `RESULTS/`, `predictions/` and `hypotheses/` are never uploaded. The Dockerfile also
  fails the build if `RESULTS/` or `predictions/` exist in the context.
- `server/build_baselines.py` (called by `stage.sh`) reads the H1–H9 specs (`hypotheses/<H>.md`) and
  their frozen predictions (`../results/claude_forecasts/<H>/`, all 14 runs must have `result.json`
  status `done`), checks their shapes against the `inputs/` templates, and packs them into
  `server/baselines.json`.

## Run locally

```bash
python3 webapp/build_index.py                       # -> webapp/static/evals.json
python3 webapp/build_baselines.py                   # -> webapp/baselines.json (optional; needs the forecast bundle)
pip install -r webapp/requirements.txt
cd webapp && DB_PATH=./dev.db LITELLM_API_KEY=... LITELLM_BASE_URL=https://litellm.nielsrolf.com \
  uvicorn app:app --reload
```

## Deploy

```bash
webapp/stage.sh /tmp/spillover-build
~/.claude/skills/nielsrolf-deploy/scripts/deploy.sh container /tmp/spillover-build spillover \
  --port 8000 --volume /data --health /healthz --env-file deploy.env
```

## Configuration (env)

| var | default | |
|---|---|---|
| `LITELLM_API_KEY` | — | required for generation |
| `LITELLM_BASE_URL` | `http://host.docker.internal:9274` | lenovo's LiteLLM |
| `PREDICTOR_MODEL` | `anthropic/claude-opus-5-5` | |
| `RUNS_PER_HANDLE` | `3` | runs that fail on a server error are refunded |
| `DAILY_RUN_CAP` | `40` | server-wide, rolling 24 h |
| `MAX_CONCURRENT_RUNS` | `2` | |
| `RUN_TOKEN_BUDGET` | `2000000` | cumulative prompt tokens per run |
| `ADMIN_TOKEN` | — | `GET /api/admin/export?token=…` returns a zip of all entries and runs |
| `FREEZE_AT` | unset | ISO timestamp; after it, all editing is locked |
| `DB_PATH` | `/data/spillover.db` | SQLite on the persistent volume |

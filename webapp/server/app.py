"""spillover.nielsrolf.com — web version of the spillover-blinded challenge.

Participants claim a handle (gets a private token), write a six-section spec,
run the blinded predictor server-side (rate-limited), optionally hand-edit the
matrices, and submit. Contents stay private until freeze; the public list
shows handles and titles only.
"""
import asyncio
import csv
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import zipfile
from contextlib import contextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", "/data/spillover.db")
RUNS_PER_HANDLE = int(os.environ.get("RUNS_PER_HANDLE", "3"))
DAILY_RUN_CAP = int(os.environ.get("DAILY_RUN_CAP", "40"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_RUNS", "2"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
FREEZE_AT = os.environ.get("FREEZE_AT", "")  # ISO date, e.g. 2026-11-15T00:00:00Z; empty = not set
HANDLE_RX = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
SECTIONS = ["statement", "definitions", "literature", "operationalization", "prerequisites", "falsifiers"]
SECTION_TITLES = {
    "statement": "Statement of the hypothesis",
    "definitions": "Definitions",
    "literature": "Literature",
    "operationalization": "Operationalization",
    "prerequisites": "Prerequisites for testing",
    "falsifiers": "Falsifiers",
}

app = FastAPI(title="Spillover prediction challenge")
_db_lock = threading.Lock()
_run_sem = asyncio.Semaphore(MAX_CONCURRENT)


# ---------------------------------------------------------------- storage

@contextmanager
def db():
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            handle TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            title TEXT DEFAULT '',
            sections TEXT DEFAULT '{}',
            spec_updated_at REAL,
            plus_csv TEXT, minus_csv TEXT,
            matrix_source TEXT,          -- 'run:<id>' or 'hand-edited'
            method_md TEXT, falsifiers_md TEXT,
            disclosure TEXT DEFAULT '',
            submitted_at REAL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            handle TEXT NOT NULL,
            status TEXT NOT NULL,        -- queued | running | done | needs_clarification | failed
            created_at REAL NOT NULL,
            finished_at REAL,
            spec_md TEXT,
            progress TEXT DEFAULT '[]',
            usage TEXT,
            plus_csv TEXT, minus_csv TEXT,
            method_md TEXT, falsifiers_md TEXT,
            inspection TEXT, clarification_md TEXT, error TEXT,
            counts INTEGER DEFAULT 1     -- 0 = infra failure, refunded
        );
        """)
        # Runs interrupted by a restart never finished; refund them.
        c.execute("UPDATE runs SET status='failed', error='Interrupted by a server restart — not counted, please re-run.', "
                  "counts=0, finished_at=? WHERE status IN ('queued','running')", (time.time(),))


init_db()


def hash_token(t):
    return hashlib.sha256(t.encode()).hexdigest()


def frozen():
    if not FREEZE_AT:
        return False
    from datetime import datetime, timezone
    return datetime.now(timezone.utc) >= datetime.fromisoformat(FREEZE_AT.replace("Z", "+00:00"))


def auth(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    th = hash_token(authorization[7:].strip())
    with db() as c:
        row = c.execute("SELECT * FROM entries WHERE token_hash=?", (th,)).fetchone()
    if not row:
        raise HTTPException(401, "Unknown token")
    return row


def editable(e=None):
    if frozen():
        raise HTTPException(403, "The challenge is frozen; entries can no longer change.")
    if e is not None and e["submitted_at"]:
        raise HTTPException(403, "This entry is submitted and therefore final.")


def spec_markdown(title, sections):
    out = [f"# {title or 'Untitled hypothesis'}", ""]
    for k in SECTIONS:
        text = (sections.get(k) or "").strip()
        out += [f"## {SECTION_TITLES[k]}", "", text or "_(not specified by the participant)_", ""]
    return "\n".join(out)


def run_public(r, full=True):
    d = {k: r[k] for k in ("id", "status", "created_at", "finished_at", "error", "counts")}
    d["progress"] = json.loads(r["progress"] or "[]")[-40:]
    d["usage"] = json.loads(r["usage"]) if r["usage"] else None
    if full:
        for k in ("plus_csv", "minus_csv", "method_md", "falsifiers_md", "clarification_md", "spec_md"):
            d[k] = r[k]
        d["inspection"] = json.loads(r["inspection"]) if r["inspection"] else None
    return d


def entry_public(e):
    with db() as c:
        runs = c.execute("SELECT * FROM runs WHERE handle=? ORDER BY id DESC", (e["handle"],)).fetchall()
    used = sum(r["counts"] for r in runs)
    return {
        "handle": e["handle"],
        "title": e["title"],
        "sections": json.loads(e["sections"] or "{}"),
        "spec_updated_at": e["spec_updated_at"],
        "plus_csv": e["plus_csv"], "minus_csv": e["minus_csv"],
        "matrix_source": e["matrix_source"],
        "method_md": e["method_md"], "falsifiers_md": e["falsifiers_md"],
        "disclosure": e["disclosure"],
        "submitted_at": e["submitted_at"],
        "runs": [run_public(r) for r in runs],
        "runs_used": used,
        "runs_allowed": RUNS_PER_HANDLE,
    }


# ---------------------------------------------------------------- API

class NewHandle(BaseModel):
    handle: str


class Spec(BaseModel):
    title: str = ""
    sections: dict


class Matrices(BaseModel):
    plus_csv: str
    minus_csv: str


class Adopt(BaseModel):
    run_id: int


class Submit(BaseModel):
    disclosure: str = ""


@app.get("/api/meta")
def meta():
    with db() as c:
        today = c.execute("SELECT COUNT(*) FROM runs WHERE created_at>? AND counts=1",
                          (time.time() - 86400,)).fetchone()[0]
    return {
        "evals": agent.EVALS, "plus_rows": agent.PLUS_ROWS, "minus_rows": agent.MINUS_ROWS,
        "sections": [{"key": k, "title": SECTION_TITLES[k]} for k in SECTIONS],
        "runs_per_handle": RUNS_PER_HANDLE, "daily_cap": DAILY_RUN_CAP, "runs_last_24h": today,
        "freeze_at": FREEZE_AT or None, "frozen": frozen(), "model": agent.MODEL,
        "evals_version": EVALS_VERSION,
    }


@app.post("/api/handles")
def claim(body: NewHandle):
    editable()
    h = body.handle.strip()
    if not HANDLE_RX.match(h):
        raise HTTPException(400, "Handle must be 1–32 letters, digits, underscores or hyphens.")
    if re.fullmatch(r"(?i)\d+|7[ab].*", h):
        raise HTTPException(400, "That handle collides with the host's H1–H9 names; pick another.")
    token = secrets.token_urlsafe(24)
    with db() as c:
        if c.execute("SELECT 1 FROM entries WHERE lower(handle)=lower(?)", (h,)).fetchone():
            raise HTTPException(409, "That handle is taken.")
        c.execute("INSERT INTO entries(handle, token_hash, created_at) VALUES (?,?,?)",
                  (h, hash_token(token), time.time()))
    return {"handle": h, "token": token}


@app.get("/api/me")
def me(authorization: str = Header(None)):
    return entry_public(auth(authorization))


@app.put("/api/me/spec")
def save_spec(body: Spec, authorization: str = Header(None)):
    e = auth(authorization)
    editable(e)
    sections = {k: str(body.sections.get(k, ""))[:60000] for k in SECTIONS}
    with db() as c:
        c.execute("UPDATE entries SET title=?, sections=?, spec_updated_at=? WHERE handle=?",
                  (body.title[:200], json.dumps(sections), time.time(), e["handle"]))
    return {"ok": True}


@app.post("/api/me/runs")
async def start_run(authorization: str = Header(None)):
    e = auth(authorization)
    editable(e)
    sections = json.loads(e["sections"] or "{}")
    if len((sections.get("statement") or "").strip()) < 20:
        raise HTTPException(400, "Describe your thesis first (a sentence or two is enough).")
    with db() as c:
        runs = c.execute("SELECT status, counts FROM runs WHERE handle=?", (e["handle"],)).fetchall()
        if any(r["status"] in ("queued", "running") for r in runs):
            raise HTTPException(409, "A run is already in progress.")
        if sum(r["counts"] for r in runs) >= RUNS_PER_HANDLE:
            raise HTTPException(429, f"You've used all {RUNS_PER_HANDLE} generation runs for this handle. "
                                     "You can still hand-edit the matrices.")
        today = c.execute("SELECT COUNT(*) FROM runs WHERE created_at>? AND counts=1",
                          (time.time() - 86400,)).fetchone()[0]
        if today >= DAILY_RUN_CAP:
            raise HTTPException(429, "The server-wide daily run budget is used up. Try again tomorrow.")
        spec_md = spec_markdown(e["title"], sections)
        cur = c.execute("INSERT INTO runs(handle, status, created_at, spec_md) VALUES (?,?,?,?)",
                        (e["handle"], "queued", time.time(), spec_md))
        run_id = cur.lastrowid
    asyncio.create_task(_execute(run_id, e["handle"], spec_md))
    return {"run_id": run_id}


async def _execute(run_id, handle, spec_md):
    async with _run_sem:
        with db() as c:
            c.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
        progress = []

        def on_progress(msg, usage):
            progress.append({"t": time.time(), "msg": msg})
            with db() as c:
                c.execute("UPDATE runs SET progress=?, usage=? WHERE id=?",
                          (json.dumps(progress[-200:]), json.dumps(usage), run_id))

        try:
            res = await asyncio.to_thread(agent.run_agent, handle, spec_md, on_progress)
            counts = 1
        except Exception as ex:  # noqa: BLE001 — API/infra errors are refunded
            res = {"status": "failed", "error": f"Server error talking to the model: {type(ex).__name__}: {str(ex)[:300]}"}
            counts = 0
        with db() as c:
            c.execute("""UPDATE runs SET status=?, finished_at=?, usage=?, plus_csv=?, minus_csv=?,
                         method_md=?, falsifiers_md=?, inspection=?, clarification_md=?, error=?, counts=?,
                         progress=? WHERE id=?""",
                      (res["status"], time.time(), json.dumps(res.get("usage")), res.get("plus_csv"),
                       res.get("minus_csv"), res.get("method_md"), res.get("falsifiers_md"),
                       json.dumps(res.get("inspection")) if res.get("inspection") else None,
                       res.get("clarification_md"), res.get("error"), counts,
                       json.dumps(progress[-200:]), run_id))
            if res["status"] == "done":
                # The latest successful run becomes the entry's matrices unless the user hand-edited.
                c.execute("""UPDATE entries SET plus_csv=?, minus_csv=?, matrix_source=?, method_md=?, falsifiers_md=?
                             WHERE handle=? AND (matrix_source IS NULL OR matrix_source LIKE 'run:%')""",
                          (res["plus_csv"], res["minus_csv"], f"run:{run_id}", res["method_md"],
                           res["falsifiers_md"], handle))


def _check_csv(text, rows):
    parsed = list(csv.reader(io.StringIO(text.strip())))
    if not parsed or parsed[0][1:] != agent.EVALS:
        raise HTTPException(400, f"Header row must be 'treatment' followed by the {len(agent.EVALS)} eval names in template order.")
    labels = [r[0] for r in parsed[1:]]
    if labels != rows:
        raise HTTPException(400, f"Row labels must match the template exactly ({len(rows)} rows).")
    for r in parsed[1:]:
        if len(r) != len(agent.EVALS) + 1:
            raise HTTPException(400, f"Row {r[0]} has {len(r) - 1} values, expected {len(agent.EVALS)}.")
        for v in r[1:]:
            if v.strip():
                try:
                    float(v)
                except ValueError:
                    raise HTTPException(400, f"Row {r[0]}: {v!r} is not a number.")
    return "\n".join(",".join(r) for r in parsed) + "\n"


@app.put("/api/me/matrices")
def save_matrices(body: Matrices, authorization: str = Header(None)):
    e = auth(authorization)
    editable(e)
    plus = _check_csv(body.plus_csv, agent.PLUS_ROWS)
    minus = _check_csv(body.minus_csv, agent.MINUS_ROWS)
    with db() as c:
        c.execute("UPDATE entries SET plus_csv=?, minus_csv=?, matrix_source='hand-edited' WHERE handle=?",
                  (plus, minus, e["handle"]))
    return {"ok": True}


@app.post("/api/me/adopt")
def adopt(body: Adopt, authorization: str = Header(None)):
    e = auth(authorization)
    editable(e)
    with db() as c:
        r = c.execute("SELECT * FROM runs WHERE id=? AND handle=? AND status='done'",
                      (body.run_id, e["handle"])).fetchone()
        if not r:
            raise HTTPException(404, "No finished run with that id.")
        c.execute("UPDATE entries SET plus_csv=?, minus_csv=?, matrix_source=?, method_md=?, falsifiers_md=? WHERE handle=?",
                  (r["plus_csv"], r["minus_csv"], f"run:{r['id']}", r["method_md"], r["falsifiers_md"], e["handle"]))
    return {"ok": True}


@app.post("/api/me/submit")
def submit(body: Submit, authorization: str = Header(None)):
    e = auth(authorization)
    editable(e)
    if not e["plus_csv"]:
        raise HTTPException(400, "Generate or hand-edit matrices before submitting.")
    try:
        _check_csv(e["plus_csv"], agent.PLUS_ROWS)
        _check_csv(e["minus_csv"] or "", agent.MINUS_ROWS)
    except HTTPException:
        raise HTTPException(400, "Your matrices don't match the current eval set (they may predate the switch to "
                                 f"{len(agent.EVALS)} evals). Run the predictor again or re-edit them.")
    with db() as c:
        if c.execute("SELECT 1 FROM runs WHERE handle=? AND status IN ('queued','running')", (e["handle"],)).fetchone():
            raise HTTPException(409, "Wait for your current run to finish before submitting.")
    with db() as c:
        c.execute("UPDATE entries SET submitted_at=?, disclosure=? WHERE handle=?",
                  (time.time(), body.disclosure[:5000], e["handle"]))
    return {"ok": True}


def _entry_files(e, runs):
    sections = json.loads(e["sections"] or "{}")
    files = {f"hypotheses/H{e['handle']}.md": spec_markdown(e["title"], sections)}
    p = f"predictions/H{e['handle']}/"
    if e["plus_csv"]:
        files[p + "logitz_plus.csv"] = e["plus_csv"]
        files[p + "logitz_minus.csv"] = e["minus_csv"]
    if e["method_md"]:
        files[p + "method.md"] = e["method_md"]
    if e["falsifiers_md"]:
        files[p + "falsifiers.md"] = e["falsifiers_md"]
    files[p + "submission.json"] = json.dumps({
        "handle": e["handle"], "title": e["title"], "matrix_source": e["matrix_source"],
        "submitted_at": e["submitted_at"], "disclosure": e["disclosure"],
        "runs": [{k: r[k] for k in ("id", "status", "created_at", "finished_at", "error")} |
                 {"usage": json.loads(r["usage"]) if r["usage"] else None} for r in runs],
    }, indent=1)
    for r in runs:
        rp = f"{p}runs/{r['id']}/"
        for k, fn in (("plus_csv", "logitz_plus.csv"), ("minus_csv", "logitz_minus.csv"),
                      ("method_md", "method.md"), ("falsifiers_md", "falsifiers.md"),
                      ("clarification_md", "NEEDS_CLARIFICATION.md"), ("spec_md", "spec_snapshot.md")):
            if r[k]:
                files[rp + fn] = r[k]
        if r["inspection"]:
            files[rp + "turn1_inspection.json"] = r["inspection"]
    return files


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


@app.get("/api/me/download")
def download(token: str):
    e = auth("Bearer " + token)
    with db() as c:
        runs = c.execute("SELECT * FROM runs WHERE handle=? ORDER BY id", (e["handle"],)).fetchall()
    return Response(_zip(_entry_files(e, runs)), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="H{e["handle"]}.zip"'})


BASELINES_PATH = os.environ.get("BASELINES_PATH", os.path.join(HERE, "baselines.json"))
BASELINES = json.load(open(BASELINES_PATH)) if os.path.exists(BASELINES_PATH) else {"examples": []}


@app.get("/api/examples")
def examples():
    """The host's H1–H9 specs. Public: they are the worked examples."""
    return [{"id": x["id"], "title": x["title"], "spec_md": x["spec_md"], "spec_text": x["spec_text"],
             "runs": [r["id"] for r in x["runs"]]} for x in BASELINES["examples"]]


@app.get("/api/examples/predictions")
def example_predictions(authorization: str = Header(None)):
    """H1–H9's frozen predictions — only for participants whose own entry is submitted (and hence locked)."""
    e = auth(authorization)
    if not e["submitted_at"]:
        raise HTTPException(403, "Submit your own entry first; H1–H9's predictions unlock after that.")
    return {r["id"]: {k: r[k] for k in ("plus_csv", "minus_csv", "method_md", "falsifiers_md")}
            for x in BASELINES["examples"] for r in x["runs"]}


@app.get("/api/entries")
def entries():
    with db() as c:
        rows = c.execute("""SELECT e.handle, e.title, e.submitted_at, e.matrix_source,
                            (SELECT COUNT(*) FROM runs r WHERE r.handle=e.handle AND r.status='done') AS n_runs
                            FROM entries e WHERE e.submitted_at IS NOT NULL ORDER BY e.submitted_at""").fetchall()
    return [{"handle": r["handle"], "title": r["title"], "submitted_at": r["submitted_at"],
             "tier": "hand-edited" if r["matrix_source"] == "hand-edited" else "spec", "n_runs": r["n_runs"]}
            for r in rows]


@app.get("/api/admin/export")
def admin_export(token: str):
    if not ADMIN_TOKEN or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(403, "Bad admin token")
    files = {}
    with db() as c:
        es = c.execute("SELECT * FROM entries ORDER BY handle").fetchall()
        for e in es:
            runs = c.execute("SELECT * FROM runs WHERE handle=? ORDER BY id", (e["handle"],)).fetchall()
            sub = "submitted" if e["submitted_at"] else "drafts"
            for k, v in _entry_files(e, runs).items():
                files[f"{sub}/{k}"] = v
    return Response(_zip(files), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="spillover-export.zip"'})


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_err(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


STATIC = os.path.join(HERE, "static")


@app.get("/example.md")
def example():
    return FileResponse(os.path.join(agent.ROOT, "EXAMPLE_hypothesis.md"), media_type="text/markdown")


def _versioned_index():
    """index.html with content-hashed asset URLs, so the CDN's static cache never serves a stale app."""
    html = open(os.path.join(STATIC, "index.html")).read()
    for name in ("app.js", "style.css"):
        digest = hashlib.sha256(open(os.path.join(STATIC, name), "rb").read()).hexdigest()[:10]
        html = html.replace(f"/static/{name}", f"/static/{name}?v={digest}")
    return html


INDEX_HTML = _versioned_index()
_evals_path = os.path.join(STATIC, "evals.json")
EVALS_VERSION = hashlib.sha256(open(_evals_path, "rb").read()).hexdigest()[:10] if os.path.exists(_evals_path) else "0"


@app.get("/")
def index():
    return Response(INDEX_HTML, media_type="text/html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

"use strict";

const S = { meta: null, evals: null, me: null, tab: "spec", runSel: null, poll: null, dirty: false, saveTimer: null,
            edit: null, sel: null };
const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (s) => DOMPurify.sanitize(marked.parse(s || ""));
const view = () => $("#view");

const HINTS = {
  statement: "Describe your thesis in plain words: which fine-tunes spill over onto which evals, in which direction, and why. This is all you need. Claude will work out how to turn it into numbers.",
  definitions: "Define the terms your rule relies on (e.g. “shared persona”, “judge overlap”, “strong spillover”).",
  literature: "Prior work you lean on, and whether it might be in Claude's training data. “None” is fine.",
  operationalization: "The most important section. A concrete rule that gives every off-diagonal cell a sign and a relative size. Claude applies it literally — the more mechanical, the more faithfully your theory gets tested.",
  prerequisites: "What needs to be read or measured from the eval bundle to apply the rule (judge prompts, system prompts, items…).",
  falsifiers: "Observations in the real matrices that would show your theory is wrong.",
};

// ------------------------------------------------------------------ util

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), 3200);
}

function token() { try { return localStorage.getItem("spill_token"); } catch { return null; } }
function setToken(t) { try { t ? localStorage.setItem("spill_token", t) : localStorage.removeItem("spill_token"); } catch {} }

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  const t = token();
  if (t) headers.Authorization = "Bearer " + t;
  const r = await fetch(path, { ...opts, headers: { ...headers, ...(opts.headers || {}) } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.error || r.statusText), { status: r.status });
  return body;
}

function fmtTime(ts) { return ts ? new Date(ts * 1000).toLocaleString() : ""; }

function parseCSV(text) {
  const lines = (text || "").trim().split(/\r?\n/).map((l) => l.split(","));
  const cols = lines[0].slice(1);
  const rows = lines.slice(1).map((l) => l[0]);
  const vals = lines.slice(1).map((l) => l.slice(1).map((v) => (v.trim() === "" ? null : Number(v))));
  return { cols, rows, vals };
}

function toCSV(m) {
  return ["treatment," + m.cols.join(","),
    ...m.rows.map((r, i) => r + "," + m.vals[i].map((v) => (v == null || Number.isNaN(v) ? "" : String(+v.toFixed(4)))).join(","))].join("\n") + "\n";
}

function blank(rows) {
  const cols = S.meta.evals;
  return { cols, rows: rows.slice(), vals: rows.map(() => cols.map(() => null)) };
}

// Stored matrices from an older eval set don't fit the current template.
function fits(csv, rows) {
  if (!csv) return false;
  const m = parseCSV(csv);
  return m.cols.join() === S.meta.evals.join() && m.rows.join() === rows.join();
}

const locked = () => S.meta.frozen || !!S.me?.submitted_at;

function isDiag(row, col) { return row.replace(/-(plus|minus)$/, "") === col; }

// ------------------------------------------------------------------ router

window.addEventListener("hashchange", route);
window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });

async function boot() {
  try {
    S.meta = await api("/api/meta");
    S.evals = (await fetch("/static/evals.json?v=" + S.meta.evals_version).then((r) => r.json())).evals;
  } catch (e) {
    view().innerHTML = `<div class="callout err" style="margin-top:30px">Could not reach the server: ${esc(e.message)}</div>`;
    return;
  }
  route();
}

function route() {
  const h = location.hash || "#/";
  const m = h.match(/[?&]t=([A-Za-z0-9_-]+)/);
  if (m) { setToken(m[1]); history.replaceState(null, "", "#/me"); }
  const page = (h.split("?")[0].replace(/^#\//, "") || "home").split("/")[0];
  const sm = h.match(/[?&]s=(\d+)/);
  if (sm) slideIdx = Math.max(0, +sm[1] - 1);
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === page));
  clearInterval(S.poll);
  window.scrollTo(0, 0);
  if (page !== "home") document.onkeydown = null;
  ({ home: renderHome, evals: renderEvals, me: renderMe, entries: renderEntries, about: renderAbout, examples: renderExamples }[page] || renderHome)(h);
}

// ------------------------------------------------------------------ home (slide deck)

function slides() {
  const m = S.meta;
  const bi = new Set(S.evals.filter((e) => e.bipolar).map((e) => e.name));
  const chips = m.evals.map((e) => `<span class="chip ${bi.has(e) ? "bi" : ""}">${esc(e)}</span>`).join("");
  return [
    { cls: "title", html: `
      <p class="kicker">A prediction challenge</p>
      <h1 class="big">Fine-tune a model on one trait.<br>What else changes?</h1>
      <p class="lede">We fine-tuned three models on ${m.evals.length} behavioural traits, one at a time, and measured all ${m.evals.length} after each run.
        The results are sealed. Can you predict them?</p>` },
    { html: `
      <p class="kicker">1 · The traits</p>
      <h2 class="big">${m.evals.length} propensities</h2>
      <p class="lede">From caring about animals to power-seeking. For the <span class="chip bi inline">${m.minus_rows.length} two-sided</span> traits we trained
        both directions, more and less. The other ${m.evals.length - m.minus_rows.length} (the three ethical frameworks) were only trained upward.</p>
      <div class="chips">${chips}</div>` },
    { html: `
      <p class="kicker">2 · The experiment</p>
      <h2 class="big">${m.plus_rows.length + m.minus_rows.length} fine-tunes × ${m.evals.length} evals</h2>
      <p class="lede">Each eval has its own questions, reference answers for each pole, and an LLM judge.
        Training data = the trait's training questions paired with its reference answers for one pole (plain SFT). Then every fine-tune is scored on every eval,
        on three models: Qwen3.5-9B, Qwen3.5-9B-Base and Nemotron-3-Super-120B.</p>
      <figure class="fig"><a href="/static/fig_overview.svg" target="_blank" title="Open full size"><img src="/static/fig_overview.svg" alt="Paper figure: Step 1, building the ${m.evals.length} evals; Step 2, eliciting a propensity and scoring on all ${m.evals.length}, producing a spillover matrix."></a>
        <figcaption>From the paper. This challenge uses <strong>SFT on each trait's reference answers</strong>, on Qwen3.5-9B, Qwen3.5-9B-Base and Nemotron-3-Super-120B. Your one prediction is compared with each model's observed matrix. The heatmap is illustrative.</figcaption></figure>` },
    { html: `
      <p class="kicker">3 · What you predict</p>
      <h2 class="big">The spillover</h2>
      <p class="lede">Training <em>for</em> a trait moves that trait: that's the diagonal, and it isn't scored.
        Everything else is spillover. Train for cooperation, and does sycophancy go up? Does spite go down?</p>
      ${miniMatrix()}` },
    { html: `
      <p class="kicker">4 · Scoring</p>
      <h2 class="big">Only sign and order count</h2>
      <p class="lede">Your matrices are compared with the real ones by rank correlation (Spearman ρ) over the off-diagonal cells, averaged across
        both matrices. Getting the scale exactly right doesn't matter. What matters is which cells move most, and in which direction.</p>
      <div class="callout warn" style="max-width:640px"><strong>Watch the direction.</strong> Higher <code>harm-refusal</code> = <em>less</em> refusal.
        The plus pole of <code>neuroticism</code> is <em>neurotic</em>. The eval pages show each judge's rubric.</div>` },
    { html: `
      <p class="kicker">5 · How you play</p>
      <h2 class="big">You write the theory. Claude does the arithmetic.</h2>
      <ol class="how">
        <li><strong>Write a short theory</strong> of why traits spill over. A few sentences is enough; you can add optional detail if you want.</li>
        <li><strong>A blinded Claude applies it</strong> to all ${(m.plus_rows.length * (m.evals.length - 1) + m.minus_rows.length * (m.evals.length - 1)).toLocaleString("en-US")} cells on our server. It never sees the results. You get up to ${m.runs_per_handle} runs.</li>
        <li><strong>Tweak and submit.</strong> You can hand-edit cells if you like. Submitting is final, and it unlocks the predictions of the paper's nine framings so you can compare.</li>
      </ol>` },
    { cls: "title", html: `
      <p class="kicker">Ready?</p>
      <h2 class="big">Start your entry</h2>
      ${m.frozen ? `<div class="callout warn">The challenge is frozen. Entries can no longer change.</div>` : `
      <div class="claim">
        <input type="text" id="handle" placeholder="Pick a handle, e.g. persona_drift" maxlength="32" autocomplete="off" aria-label="Handle">
        <button class="btn primary" id="claimBtn">Start</button>
      </div>
      <p class="small muted" style="margin-top:10px">${token() ? `You already have an entry on this device. <a href="#/me">Open it</a>.` : "No account needed. You get a private link that works as your password."}</p>`}
      <p class="small" style="margin-top:28px"><a href="#/examples">Start from one of the paper's framings (H1–H9)</a> · <a href="#/evals">Browse the ${m.evals.length} evals</a> · <a href="#/about">Rules &amp; fine print</a></p>` },
  ];
}

function miniMatrix() {
  const names = ["cooperation", "sycophancy", "spitefulness", "power-seeking", "caring-about-user", "certainty"];
  const cell = (i, j) => {
    if (i === j) return `<td class="diag" title="on-target, not scored"></td>`;
    if (i === 0 && j === 1) return `<td class="q up">?</td>`;
    if (i === 0 && j === 2) return `<td class="q down">?</td>`;
    return `<td></td>`;
  };
  return `<div class="mini-wrap"><table class="mini">
    <thead><tr><th></th>${names.map((n) => `<th class="col"><div>${n}</div></th>`).join("")}</tr></thead>
    <tbody>${names.map((r, i) => `<tr><th class="rowh">${r}-plus</th>${names.map((_, j) => cell(i, j)).join("")}</tr>`).join("")}</tbody>
  </table><div class="mini-legend small muted"><span><span class="sw diag"></span>on-target (not scored)</span>
    <span><span class="sw"></span>spillover: you predict these</span><span>rows: fine-tunes · columns: evals</span></div></div>`;
}

let slideIdx = 0;

// Scrolling acts as "next / previous slide". If a slide is taller than the window
// (small screens), the page scrolls normally first and only advances at the edge.
function atEdge(dir) {
  const el = document.scrollingElement;
  return dir > 0 ? el.scrollTop + window.innerHeight >= el.scrollHeight - 2 : el.scrollTop <= 0;
}

function wheelNav(e) {
  if (!$(".deck") || !wheelNav.go || e.ctrlKey) return;
  if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return; // horizontal scroll, e.g. the mini matrix on phones
  const dir = e.deltaY > 0 ? 1 : -1;
  if (!atEdge(dir)) return;
  e.preventDefault();
  const now = Date.now();
  if (now < (wheelNav.lockUntil || 0)) { wheelNav.lockUntil = now + 250; return; } // swallow the trackpad's inertia tail
  wheelNav.acc = (wheelNav.acc || 0) + e.deltaY;
  if (Math.abs(wheelNav.acc) < 40) return;
  wheelNav.acc = 0;
  wheelNav.lockUntil = now + 700;
  wheelNav.go(dir);
}
window.addEventListener("wheel", wheelNav, { passive: false });

function renderHome() {
  const all = slides();
  slideIdx = Math.min(slideIdx, all.length - 1);
  view().innerHTML = `
    <section class="deck" aria-roledescription="carousel">
      <div class="deck-bar"><div class="deck-progress" style="width:${((slideIdx + 1) / all.length) * 100}%"></div></div>
      <div class="slide ${all[slideIdx].cls || ""}" aria-live="polite">${all[slideIdx].html}</div>
      <div class="deck-nav">
        <button class="btn" id="prev" ${slideIdx === 0 ? "disabled" : ""} aria-label="Previous slide">← Back</button>
        <span class="deck-count">${slideIdx + 1} / ${all.length}</span>
        ${slideIdx < all.length - 1 ? `<button class="btn primary" id="next" aria-label="Next slide">${slideIdx === 0 ? "Show me" : "Next"} →</button>` : `<span style="width:96px"></span>`}
      </div>
      ${slideIdx < all.length - 1 ? `<button class="deck-skip" id="skip">Skip intro</button>` : ""}
    </section>`;
  const go = (d) => { slideIdx = Math.max(0, Math.min(all.length - 1, slideIdx + d)); renderHome(); };
  $("#prev").onclick = () => go(-1);
  if ($("#next")) { $("#next").onclick = () => go(1); if (slideIdx > 0) $("#next").focus({ preventScroll: true }); }
  if ($("#skip")) $("#skip").onclick = () => { slideIdx = all.length - 1; renderHome(); };
  document.onkeydown = (e) => {
    if (!$(".deck") || e.target.matches("input, textarea")) return;
    if (e.key === "ArrowRight" || e.key === "PageDown") go(1);
    if (e.key === "ArrowLeft" || e.key === "PageUp") go(-1);
  };
  const deck = $(".deck");
  let t0 = null;
  deck.ontouchstart = (e) => { t0 = { x: e.touches[0].clientX, y: e.touches[0].clientY }; };
  deck.ontouchend = (e) => {
    if (!t0) return;
    const dx = e.changedTouches[0].clientX - t0.x, dy = e.changedTouches[0].clientY - t0.y;
    t0 = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) return go(dx < 0 ? 1 : -1);
    // vertical swipe advances only once the slide itself is scrolled to its edge
    if (Math.abs(dy) > 60 && atEdge(dy < 0 ? 1 : -1)) go(dy < 0 ? 1 : -1);
  };
  wheelNav.go = go;

  const btn = $("#claimBtn");
  if (btn) {
    const claim = async () => {
      const h = $("#handle").value.trim();
      if (!h) return $("#handle").focus();
      if (token() && !confirm("This device already has an entry. Starting a new one will forget the old private link on this device (the entry itself stays). Continue?")) return;
      btn.disabled = true;
      try {
        const r = await api("/api/handles", { method: "POST", body: JSON.stringify({ handle: h }) });
        setToken(r.token);
        S.tab = "spec";
        location.hash = "#/me";
      } catch (e) { toast(e.message); } finally { btn.disabled = false; }
    };
    btn.onclick = claim;
    $("#handle").onkeydown = (e) => { if (e.key === "Enter") claim(); };
    $("#handle").focus({ preventScroll: true });
  }
}

// ------------------------------------------------------------------ about

function renderAbout() {
  const m = S.meta;
  view().innerHTML = `
  <div class="ws-head prose">
    <h1>Rules &amp; fine print</h1>
    <h2>What exactly is predicted</h2>
    <p>Two matrices of <strong>logitz</strong>: the shift on each eval (empirical logit of the 0–100 judge score), z-scored per eval.</p>
    <ul><li><strong>plus</strong>: ${m.plus_rows.length} fine-tunes (each trait trained upward) × ${m.evals.length} evals</li>
      <li><strong>minus</strong>: ${m.minus_rows.length} fine-tunes (two-sided traits trained downward) × ${m.evals.length} evals</li></ul>
    <p>A fine-tune is SFT on the trait's training questions paired with that pole's reference answers. The observed matrices come from
      three models: Qwen3.5-9B, Qwen3.5-9B-Base and Nemotron-3-Super-120B. You submit one model-agnostic pair of matrices, compared with each.</p>
    <p>The diagonal is excluded. Score = Spearman ρ over the ${m.plus_rows.length * (m.evals.length - 1)} + ${m.minus_rows.length * (m.evals.length - 1)} off-diagonal cells, with bootstrap 95% CIs; the headline is the mean of the two.</p>
    <h2>The predictor</h2>
    <p><code>${esc(m.model)}</code> runs the repo's <code>BLINDED_PROMPT.md</code> protocol: an inspection turn without numbers, then generation.
      It can read the eval bundle (questions, reference answers, system prompts, judge prompts, judge-overlap similarities) and your spec. It can't run code, and the results aren't on the server.
      The host's nine framings (<a href="#/examples">H1–H9</a>) were predicted by the same model with the same web harness (no code execution), so your web runs are directly comparable with them.
      The command-line version (Claude Code) can run code, so CLI runs are not strictly apples-to-apples with web runs.</p>
    <h2>H1–H9</h2>
    <p>The specs of the paper's nine framings are public examples, and you can start from one. Their frozen predictions unlock once you submit
      your own entry. Submitting is final, so seeing them can't change your entry.</p>
    <h2>Limits</h2>
    <p>${m.runs_per_handle} generation runs per entry. Runs that fail because of a server error don't count. There's also a server-wide daily cap.</p>
    <h2>Honesty</h2>
    <ul><li>Disclose AI help you used writing the spec or editing cells.</li>
      <li>Disclose whether literature you cite might be in Claude's training data.</li>
      <li>Leaving a cell blank is better than inventing a number.</li></ul>
    <h2>Timeline</h2>
    <p>${m.freeze_at ? `Freeze: <strong>${esc(new Date(m.freeze_at).toLocaleString())}</strong>.` : "The freeze date will be announced on the LessWrong post."}
      At freeze the observed matrices are unsealed, and the leaderboard, including the host's H1–H9, is published.</p>
    <h2>Command-line version</h2>
    <p>The original Claude Code pipeline is at <a href="https://github.com/Junekhunter/spillover-blinded">github.com/Junekhunter/spillover-blinded</a>.</p>
  </div>`;
}

// ------------------------------------------------------------------ evals

function renderEvals(h) {
  const sel = decodeURIComponent((h.split("/")[2] || "").split("?")[0]);
  view().innerHTML = `
    <div class="ws-head"><h1>The ${S.evals.length} evals</h1>
      <p class="muted" style="max-width:720px">Each eval is both a fine-tuning target (SFT on its training questions paired with one pole's
      reference answers) and a column in the matrices. Open one to see its poles, judge scale, and sample test items with their reference answers.</p>
      <input type="text" id="q" placeholder="Filter…" style="max-width:320px" aria-label="Filter evals"></div>
    <div id="detail"></div>
    <div class="eval-list" id="list"></div>`;
  const list = $("#list");
  const draw = () => {
    const q = $("#q").value.toLowerCase();
    list.innerHTML = S.evals.filter((e) => e.name.includes(q)).map((e) => `
      <a class="card eval-card" href="#/evals/${e.name}" style="text-decoration:none;color:inherit">
        <div class="name">${esc(e.name)}</div>
        <div class="row" style="margin-top:6px;gap:6px">
          <span class="pill ${e.bipolar ? "bi" : ""}">${e.bipolar ? "bipolar" : "plus only"}</span>
          ${e.trap ? `<span class="pill warn">direction trap</span>` : ""}
        </div>
        <div class="small muted" style="margin-top:6px">${e.n_train} train · ${e.n_test} test items</div>
      </a>`).join("");
  };
  $("#q").oninput = draw;
  draw();
  if (sel) renderEvalDetail(sel);
}

function renderEvalDetail(name) {
  const e = S.evals.find((x) => x.name === name);
  if (!e) return;
  const d = e.definition;
  const [plusF, minusF] = e.pole_files || [];
  const sp = Object.entries(e.system_prompts).map(([f, t]) => {
    const role = f === plusF ? "plus pole" : f === minusF ? "minus pole" : e.pole_files ? "" :
      Object.keys(e.system_prompts).length === 1 ? "plus pole (only prompt)" : "";
    return `<details><summary><code>${esc(f)}</code> ${role ? `<span class="pill ${role.startsWith("plus") ? "bi" : ""}">${role}</span>` : ""}</summary><pre>${esc(t)}</pre></details>`;
  }).join("");
  const scales = Object.entries(e.judge_scales).map(([k, t]) => `<details ${Object.keys(e.judge_scales).length === 1 ? "open" : ""}><summary><code>${esc(k)}</code></summary><pre>${esc(t)}</pre></details>`).join("");
  const samples = e.samples.map((s) => `<details><summary>${esc(s.prompt.slice(0, 140))}${s.prompt.length > 140 ? "…" : ""}</summary>
      <pre>${esc(s.prompt)}</pre>${Object.entries(s.expected).map(([k, v]) => `<div class="small muted" style="margin-top:8px"><code>${esc(k)}</code></div><pre>${esc(v)}</pre>`).join("")}</details>`).join("");
  $("#detail").innerHTML = `
    <div class="card eval-detail" style="margin-bottom:22px">
      <div class="row"><h2 style="margin:0;font-family:var(--mono);font-size:19px">${esc(e.name)}</h2>
        <span class="pill ${e.bipolar ? "bi" : ""}">${e.bipolar ? "bipolar — plus & minus fine-tunes" : "plus-only — no minus fine-tune"}</span>
        <span class="spacer"></span><a class="btn small" href="#/evals">Close</a></div>
      ${e.trap ? `<div class="callout warn small"><strong>Direction trap:</strong> ${esc(e.trap)}</div>` : ""}
      ${d ? `<p>${esc(d.definition)}</p><p><strong>Plus pole${d.plus_pole_name ? ` (${esc(d.plus_pole_name)})` : ""}.</strong> ${esc(d.plus_pole)}</p>${d.minus_pole ? `<p><strong>Minus pole${d.minus_pole_name ? ` (${esc(d.minus_pole_name)})` : ""}.</strong> ${esc(d.minus_pole)}</p>` : `<p class="small muted">No minus pole: this eval is only trained upward. Its natural contrast is the other two ethical frameworks.</p>`}` : `<p class="muted small">No entry in definitions.json for this eval — the meaning comes from its system prompts and judge.</p>`}
      <p class="small muted">${e.n_items} items (${e.n_train} train used for fine-tuning, ${e.n_test} test used for scoring) · meta keys: ${e.meta_keys.map((k) => `<code>${esc(k)}</code>`).join(", ")}</p>
      <h3>Judge rubric${Object.keys(e.judge_scales).length > 1 ? "s" : ""} <span class="small muted" style="font-weight:400">(where score direction is defined)</span></h3>${scales}
      ${e.evidence_gate ? `<details><summary>Evidence gate <span class="small muted">(run first; a score below 50 drops the response)</span></summary><pre>${esc(e.evidence_gate)}</pre></details>` : ""}
      <h3>System prompts <span class="small muted" style="font-weight:400">(describe each pole; the fine-tuning data is the reference answers below)</span></h3>${sp}
      <h3>Sample test items <span class="small muted" style="font-weight:400">(with each pole's reference answer)</span></h3>${samples}
    </div>`;
  $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ my entry

async function loadMe() {
  S.me = await api("/api/me");
  return S.me;
}

async function renderMe(h) {
  const t = (h || "").split("?")[0].split("/")[2];
  if (["spec", "generate", "matrices", "submit"].includes(t)) S.tab = t;
  if (!token()) {
    view().innerHTML = `<div class="ws-head"><h1>My entry</h1>
      <div class="card" style="max-width:560px"><p style="margin-top:0">No entry on this device yet.</p>
      <div class="row"><a class="btn primary" href="#/">Start an entry</a></div>
      <p class="small muted" style="margin-bottom:0">Started one elsewhere? Open your private link on this device instead.</p></div></div>`;
    return;
  }
  view().innerHTML = `<div class="ws-head"><span class="spinner"></span></div>`;
  try { await loadMe(); } catch (e) {
    if (e.status === 401) {
      view().innerHTML = `<div class="ws-head"><h1>My entry</h1><div class="callout err">This device's private link isn't valid any more.</div>
        <button class="btn" id="forget">Forget it</button></div>`;
      $("#forget").onclick = () => { setToken(null); location.hash = "#/"; };
      return;
    }
    view().innerHTML = `<div class="callout err">${esc(e.message)}</div>`;
    return;
  }
  drawMe();
}

function drawMe() {
  const me = S.me;
  const link = location.origin + "/#/me?t=" + token();
  const active = me.runs.find((r) => r.status === "queued" || r.status === "running");
  view().innerHTML = `
    <div class="ws-head">
      <div class="row"><h1 style="margin:0">${esc(me.handle)}</h1><span class="pill mono" title="File name used in the repo">H${esc(me.handle)}.md</span>
        ${me.submitted_at ? `<span class="pill ok">submitted</span>` : `<span class="pill">draft</span>`}
        ${S.meta.frozen ? `<span class="pill warn">frozen</span>` : ""}</div>
      <details style="margin-top:12px"><summary>Your private link: bookmark it, it's your password</summary>
        <div class="secret" style="margin-top:8px"><input type="text" readonly value="${esc(link)}" id="secretLink">
        <button class="btn small" id="copyLink">Copy</button></div>
        <p class="small muted">Anyone with this link can edit your entry. Open it on another device to continue there.</p></details>
    </div>
    <div class="tabs" role="tablist">
      ${[["spec", "Write spec"], ["generate", "Generate"], ["matrices", "Matrices"], ["submit", "Submit"]].map(([k, l], i) =>
        `<button data-tab="${k}" class="${S.tab === k ? "active" : ""}" role="tab"><span class="n">${i + 1}</span>${l}${k === "generate" && active ? ` <span class="spinner" style="margin-left:6px"></span>` : ""}</button>`).join("")}
    </div>
    <div id="tab"></div>`;
  $("#copyLink").onclick = () => { navigator.clipboard?.writeText(link); toast("Copied"); };
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = async () => {
    if (S.dirty) await saveSpec();
    S.tab = b.dataset.tab;
    drawMe();
  }));
  ({ spec: tabSpec, generate: tabGenerate, matrices: tabMatrices, submit: tabSubmit }[S.tab])();
  clearInterval(S.poll);
  if (active) S.poll = setInterval(pollRun, 3000);
}

async function pollRun() {
  if (location.hash.split("?")[0] !== "#/me") return clearInterval(S.poll);
  const prev = S.me.runs.find((r) => r.status === "queued" || r.status === "running");
  try { await loadMe(); } catch { return; }
  const cur = S.me.runs.find((r) => r.id === prev?.id);
  if (S.tab === "generate") {
    if (cur && (cur.status === "queued" || cur.status === "running")) {
      const log = $("#log");
      if (log) { log.innerHTML = progressHTML(cur); log.scrollTop = log.scrollHeight; }
      return;
    }
    drawMe();
  }
  if (cur && cur.status !== "queued" && cur.status !== "running") {
    toast(cur.status === "done" ? "Predictions are ready" : cur.status === "needs_clarification" ? "Claude needs clarification on your spec" : "The run failed");
    if (S.tab !== "generate") drawMe();
  }
}

// ---- tab 1: spec

function tabSpec() {
  const me = S.me;
  const lock = locked();
  $("#tab").innerHTML = `
    <div class="row"><p class="muted" style="margin:0;max-width:720px">A falsifiable theory of which fine-tunes move which evals, and why.
      Need a starting point? <a href="#/examples">Start from one of the paper's framings (H1–H9)</a>. Saved automatically.</p>
      <span class="spacer"></span><span class="save-state" id="saveState">${me.spec_updated_at ? "Saved" : ""}</span></div>
    <label class="field" for="title">Title</label>
    <input type="text" id="title" value="${esc(me.title)}" placeholder="e.g. Shared-persona drift" maxlength="200" ${lock ? "disabled" : ""}>
    ${S.meta.sections.filter((s) => s.key === "statement").map((s) => `
      <label class="field" for="s_${s.key}">Your thesis <span class="pill bi">required</span></label>
      <div class="hint">${esc(HINTS[s.key])}</div>
      <textarea id="s_${s.key}" data-key="${s.key}" style="min-height:180px" ${lock ? "disabled" : ""}>${esc(me.sections[s.key] || "")}</textarea>`).join("")}
    <details class="optional" ${S.meta.sections.some((s) => s.key !== "statement" && (me.sections[s.key] || "").trim()) ? "open" : ""}>
      <summary>Optional: make it more precise <span class="small muted">(definitions, literature, operationalization, prerequisites, falsifiers)</span></summary>
      <p class="small muted" style="margin:8px 0 0">Anything you leave empty, Claude fills in from your thesis and explains in method.md.
        A precise operationalization means your idea gets tested exactly as you meant it.</p>
      ${S.meta.sections.filter((s) => s.key !== "statement").map((s) => `
      <label class="field" for="s_${s.key}">${esc(s.title)}</label>
      <div class="hint">${esc(HINTS[s.key])}</div>
      <textarea id="s_${s.key}" data-key="${s.key}" ${s.key === "operationalization" ? 'style="min-height:160px"' : ""} ${lock ? "disabled" : ""}>${esc(me.sections[s.key] || "")}</textarea>`).join("")}
    </details>
    <div class="row" style="margin-top:20px"><span class="spacer"></span>
      <button class="btn primary" id="toGen" ${lock ? "disabled" : ""}>Next: generate predictions →</button></div>`;
  const onInput = () => {
    S.dirty = true;
    $("#saveState").textContent = "Unsaved…";
    clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(saveSpec, 1200);
  };
  $("#title").oninput = onInput;
  document.querySelectorAll("textarea[data-key]").forEach((t) => (t.oninput = onInput));
  $("#toGen").onclick = async () => { await saveSpec(); S.tab = "generate"; drawMe(); };
}

function collectSpec() {
  const sections = {};
  document.querySelectorAll("textarea[data-key]").forEach((t) => (sections[t.dataset.key] = t.value));
  return { title: $("#title")?.value ?? S.me.title, sections };
}

async function saveSpec() {
  clearTimeout(S.saveTimer);
  if (!S.dirty || !$("#title")) return;
  const body = collectSpec();
  try {
    await api("/api/me/spec", { method: "PUT", body: JSON.stringify(body) });
    S.dirty = false;
    S.me.title = body.title; S.me.sections = body.sections; S.me.spec_updated_at = Date.now() / 1000;
    const st = $("#saveState"); if (st) st.textContent = "Saved";
  } catch (e) { toast("Save failed: " + e.message); }
}

// ---- tab 2: generate

function progressHTML(r) {
  const items = r.progress.map((p) => `<div>${esc(new Date(p.t * 1000).toLocaleTimeString())}  ${esc(p.msg)}</div>`).join("");
  return items || `<div class="muted">${r.status === "queued" ? "Waiting for a free slot…" : "Starting…"}</div>`;
}

function tabGenerate() {
  const me = S.me;
  const left = me.runs_allowed - me.runs_used;
  const active = me.runs.find((r) => r.status === "queued" || r.status === "running");
  const missing = (me.sections.statement || "").trim().length < 20 ? ["your thesis"] : [];
  if (!S.runSel || !me.runs.find((r) => r.id === S.runSel)) S.runSel = me.runs[0]?.id ?? null;
  const run = me.runs.find((r) => r.id === S.runSel);
  $("#tab").innerHTML = `
    <div class="card">
      <div class="row"><div><strong>${left} of ${me.runs_allowed}</strong> generation runs left
        <div class="small muted">Each run takes roughly 5–15 minutes. Runs that fail because of a server error don't count.</div></div>
        <span class="spacer"></span>
        <button class="btn primary" id="runBtn" ${active || left <= 0 || missing.length || locked() ? "disabled" : ""}>
          ${active ? "Running…" : me.runs.length ? "Run again with current spec" : "Run the predictor"}</button></div>
      ${missing.length ? `<div class="callout warn small">Describe your thesis first (a sentence or two is enough) in the Write spec tab.</div>` : ""}
      ${left <= 0 && !active ? `<div class="callout small">Run budget used up. You can still hand-edit cells in the Matrices tab.</div>` : ""}
    </div>
    ${me.runs.length ? `<div class="runs">${me.runs.map((r, i) => `<button class="btn small ${r.id === S.runSel ? "active" : ""}" data-run="${r.id}">
        Run ${me.runs.length - i} ${statusPill(r)}</button>`).join("")}</div>` : `<p class="muted">No runs yet.</p>`}
    <div id="runView">${run ? runHTML(run) : ""}</div>`;
  $("#runBtn").onclick = startRun;
  document.querySelectorAll("[data-run]").forEach((b) => (b.onclick = () => { S.runSel = +b.dataset.run; tabGenerate(); }));
  if (run) wireRun(run);
}

function statusPill(r) {
  const map = { queued: ["", "queued"], running: ["", "running"], done: ["ok", "done"], needs_clarification: ["warn", "needs clarification"], failed: ["err", "failed"] };
  const [c, l] = map[r.status] || ["", r.status];
  return `<span class="pill ${c}">${l}</span>`;
}

function runHTML(r) {
  const usage = r.usage ? `<span class="small muted">${r.usage.steps || 0} steps · ${Math.round((r.usage.prompt_tokens || 0) / 1000)}k input tokens</span>` : "";
  const head = `<div class="row" style="margin:6px 0 10px"><strong>Started ${esc(fmtTime(r.created_at))}</strong> ${statusPill(r)} ${usage}</div>`;
  if (r.status === "queued" || r.status === "running") {
    return head + `<div class="progress-log" id="log">${progressHTML(r)}</div>`;
  }
  let out = head;
  if (r.error) out += `<div class="callout err">${esc(r.error)}</div>`;
  if (r.clarification_md) out += `<div class="callout warn"><strong>Claude needs clarification before predicting.</strong> Edit your spec to answer this, then run again.</div><div class="card md">${md(r.clarification_md)}</div>`;
  if (r.status === "done") {
    const adopted = S.me.matrix_source === "run:" + r.id;
    out += `<div class="row" style="margin-bottom:10px">
        ${adopted ? `<span class="pill ok">These are your entry's current matrices</span>` : `<button class="btn" id="adopt">Use this run's matrices for my entry</button>`}
        <span class="spacer"></span><label class="small row" style="gap:6px"><input type="checkbox" id="showVals"> show values</label></div>
      <h3>logitz_plus <span class="muted small">(${S.meta.plus_rows.length} × ${S.meta.evals.length})</span></h3><div class="matrix-wrap" id="mp"></div>
      <h3>logitz_minus <span class="muted small">(${S.meta.minus_rows.length} × ${S.meta.evals.length})</span></h3><div class="matrix-wrap" id="mm"></div>
      <div class="legend" style="margin-top:8px"><span>negative</span><span class="bar"></span><span>positive</span><span>· hatched = diagonal (not scored)</span></div>`;
  }
  if (r.inspection) out += `<details><summary>Turn 1: inspection & operationalization choices</summary><div class="md">
      <h3>Flagged issues</h3>${md(r.inspection.flagged_issues)}<h3>Operationalization</h3>${md(r.inspection.operationalization)}</div></details>`;
  if (r.method_md) out += `<details open><summary>method.md</summary><div class="md">${md(r.method_md)}</div></details>`;
  if (r.falsifiers_md) out += `<details><summary>falsifiers.md</summary><div class="md">${md(r.falsifiers_md)}</div></details>`;
  if (r.progress?.length) out += `<details><summary>Run log</summary><div class="progress-log">${progressHTML(r)}</div></details>`;
  if (r.spec_md) out += `<details><summary>Spec as sent</summary><pre>${esc(r.spec_md)}</pre></details>`;
  return out;
}

function wireRun(r) {
  if (r.status !== "done") { const l = $("#log"); if (l) l.scrollTop = l.scrollHeight; return; }
  const draw = () => {
    const sv = $("#showVals").checked;
    matrixTable($("#mp"), parseCSV(r.plus_csv), { showValues: sv });
    matrixTable($("#mm"), parseCSV(r.minus_csv), { showValues: sv });
  };
  $("#showVals").onchange = draw;
  draw();
  const a = $("#adopt");
  if (a) a.onclick = async () => {
    if (S.me.matrix_source === "hand-edited" && !confirm("This replaces your hand-edited matrices. Continue?")) return;
    try { await api("/api/me/adopt", { method: "POST", body: JSON.stringify({ run_id: r.id }) }); await loadMe(); drawMe(); toast("Entry updated"); }
    catch (e) { toast(e.message); }
  };
}

async function startRun() {
  if (S.dirty) await saveSpec();
  const btn = $("#runBtn");
  btn.disabled = true;
  try {
    const r = await api("/api/me/runs", { method: "POST" });
    S.runSel = r.run_id;
    await loadMe();
    drawMe();
  } catch (e) { toast(e.message); btn.disabled = false; }
}

// ---- matrix rendering

function matrixTable(el, m, { editable = false, showValues = false, onSelect } = {}) {
  let max = 0;
  m.vals.forEach((row, i) => row.forEach((v, j) => { if (v != null && !isDiag(m.rows[i], m.cols[j])) max = Math.max(max, Math.abs(v)); }));
  max = max || 1;
  const color = (v) => {
    if (v == null || Number.isNaN(v)) return "";
    const a = Math.min(1, Math.abs(v) / max);
    return `background: rgba(var(${v >= 0 ? "--pos" : "--neg"}), ${0.08 + 0.92 * a})`;
  };
  el.innerHTML = `<table class="matrix ${showValues ? "show-values" : ""}"><thead><tr><th></th>${m.cols.map((c, j) => `<th class="col" data-j="${j}"><div>${esc(c)}</div></th>`).join("")}</tr></thead>
    <tbody>${m.rows.map((r, i) => `<tr><th class="rowh">${esc(r)}</th>${m.cols.map((c, j) => {
      const d = isDiag(r, c);
      const v = m.vals[i][j];
      const cls = [d ? "diag" : v == null ? "empty" : "", S.sel && editable && S.sel[0] === i && S.sel[1] === j ? "sel" : ""].join(" ");
      return `<td class="${cls}" data-i="${i}" data-j="${j}" style="${d ? "" : color(v)}" title="${esc(r)} → ${esc(c)}: ${d ? "diagonal (not scored)" : v == null ? "blank" : v}">${!d && v != null && showValues ? (Math.abs(v) >= 10 ? Math.round(v) : v.toFixed(1)) : ""}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
  const tbl = el.querySelector("table");
  tbl.addEventListener("mouseover", (e) => {
    const td = e.target.closest("td"); if (!td) return;
    tbl.querySelectorAll("th.col.hl").forEach((h) => h.classList.remove("hl"));
    tbl.querySelector(`th.col[data-j="${td.dataset.j}"]`)?.classList.add("hl");
  });
  if (editable && onSelect) tbl.addEventListener("click", (e) => {
    const td = e.target.closest("td"); if (!td || td.classList.contains("diag")) return;
    onSelect(+td.dataset.i, +td.dataset.j);
  });
}

// ---- tab 3: matrices (hand edit)

function tabMatrices() {
  const me = S.me;
  if (!S.edit || !S.edit.dirty) {
    S.edit = {
      which: S.edit?.which || "plus",
      showValues: S.edit?.showValues || false,
      plus: fits(me.plus_csv, S.meta.plus_rows) ? parseCSV(me.plus_csv) : blank(S.meta.plus_rows),
      minus: fits(me.minus_csv, S.meta.minus_rows) ? parseCSV(me.minus_csv) : blank(S.meta.minus_rows),
      dirty: false,
    };
  }
  const E = S.edit;
  const src = me.matrix_source === "hand-edited" ? "hand-edited" : me.matrix_source ? `from run #${me.matrix_source.slice(4)}` : "empty template";
  $("#tab").innerHTML = `
    <p class="muted" style="max-width:760px;margin-top:0">Optional second tier: encode cell-level intuitions directly. Start from a generated run
      (adopt one in the Generate tab) or from the blank template. Click a cell, type a value, use arrow keys to move. Only ordering and sign matter.</p>
    <div class="row">
      <div class="tabs" style="margin:0;border:0">
        <button data-w="plus" class="${E.which === "plus" ? "active" : ""}">plus (${S.meta.plus_rows.length} × ${S.meta.evals.length})</button>
        <button data-w="minus" class="${E.which === "minus" ? "active" : ""}">minus (${S.meta.minus_rows.length} × ${S.meta.evals.length})</button></div>
      <span class="pill">current: ${esc(src)}</span>
      <span class="spacer"></span>
      <label class="small row" style="gap:6px"><input type="checkbox" id="showVals2" ${E.showValues ? "checked" : ""}> show values</label>
    </div>
    <div class="cell-editor" id="cellEd"><span class="muted small">Select a cell to edit it.</span></div>
    <div class="matrix-wrap" id="me"></div>
    <div class="legend" style="margin-top:8px"><span>negative</span><span class="bar"></span><span>positive</span><span>· grey = blank · hatched = diagonal</span></div>
    <div class="row" style="margin-top:16px">
      <button class="btn primary" id="saveM" ${locked() ? "disabled" : ""}>Save as hand-edited</button>
      <button class="btn" id="dlM">Download CSV</button>
      <label class="btn" style="cursor:pointer">Import CSV<input type="file" id="upM" accept=".csv,text/csv" hidden></label>
      <span class="save-state" id="mState">${E.dirty ? "Unsaved changes" : ""}</span>
    </div>`;
  document.querySelectorAll("[data-w]").forEach((b) => (b.onclick = () => { E.which = b.dataset.w; S.sel = null; tabMatrices(); }));
  $("#showVals2").onchange = (e) => { E.showValues = e.target.checked; drawEdit(); };
  $("#saveM").onclick = async () => {
    try {
      await api("/api/me/matrices", { method: "PUT", body: JSON.stringify({ plus_csv: toCSV(E.plus), minus_csv: toCSV(E.minus) }) });
      E.dirty = false; await loadMe(); toast("Saved"); tabMatrices();
    } catch (e) { toast(e.message); }
  };
  $("#dlM").onclick = () => {
    const blob = new Blob([toCSV(E[E.which])], { type: "text/csv" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `logitz_${E.which}.csv` });
    a.click();
  };
  $("#upM").onchange = async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    const m = parseCSV(await f.text());
    const want = E[E.which];
    if (m.cols.join() !== want.cols.join() || m.rows.join() !== want.rows.join()) return toast("Row/column labels don't match the template");
    E[E.which] = m; E.dirty = true; tabMatrices();
  };
  drawEdit();
}

function drawEdit() {
  const E = S.edit;
  const m = E[E.which];
  matrixTable($("#me"), m, { editable: true, showValues: E.showValues, onSelect: (i, j) => { S.sel = [i, j]; drawEdit(); } });
  const ed = $("#cellEd");
  if (!S.sel) return;
  const [i, j] = S.sel;
  ed.innerHTML = `<code>${esc(m.rows[i])}</code> → <code>${esc(m.cols[j])}</code>
    <input type="text" id="cellIn" inputmode="decimal" value="${m.vals[i][j] ?? ""}" placeholder="blank" aria-label="Cell value">
    <span class="small muted">Enter to save · arrow keys move · empty = blank</span>`;
  const inp = $("#cellIn");
  inp.focus(); inp.select();
  const commit = () => {
    const t = inp.value.trim();
    const v = t === "" ? null : Number(t);
    if (t !== "" && !Number.isFinite(v)) { toast("Not a number"); return false; }
    if (v !== m.vals[i][j]) { m.vals[i][j] = v; E.dirty = true; const st = $("#mState"); if (st) st.textContent = "Unsaved changes"; }
    return true;
  };
  inp.onkeydown = (e) => {
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], Enter: [1, 0], Tab: [0, e.shiftKey ? -1 : 1] };
    if (!(e.key in moves)) return;
    e.preventDefault();
    if (!commit()) return;
    let [di, dj] = moves[e.key];
    let ni = i, nj = j;
    do { ni = Math.max(0, Math.min(m.rows.length - 1, ni + di)); nj = Math.max(0, Math.min(m.cols.length - 1, nj + dj)); }
    while (isDiag(m.rows[ni], m.cols[nj]) && !(ni === 0 && di < 0) && !(ni === m.rows.length - 1 && di > 0) && !(nj === 0 && dj < 0) && !(nj === m.cols.length - 1 && dj > 0));
    if (isDiag(m.rows[ni], m.cols[nj])) { ni = i; nj = j; }
    S.sel = [ni, nj];
    drawEdit();
  };
  inp.onblur = commit;
}

// ---- tab 4: submit

function tabSubmit() {
  const me = S.me;
  const specOK = (me.sections.statement || "").trim().length >= 20;
  const hasM = !!me.plus_csv;
  const active = me.runs.some((r) => r.status === "queued" || r.status === "running");
  const src = me.matrix_source === "hand-edited" ? "hand-edited matrices" : me.matrix_source ? `matrices from run #${me.matrix_source.slice(4)}` : "no matrices yet";
  $("#tab").innerHTML = `
    <div class="card" style="max-width:760px">
      ${me.submitted_at ? `
        <div class="row"><span class="pill ok">Submitted ${esc(fmtTime(me.submitted_at))}</span><span class="small muted">Final. This is what gets scored at freeze.</span></div>
        <p>You've unlocked the host's H1–H9 predictions. <a class="btn primary small" href="#/examples">Compare with H1–H9 →</a></p>
        ${me.disclosure ? `<h3>Your disclosures</h3><p class="md">${esc(me.disclosure)}</p>` : ""}` : `
      <h3 style="margin-top:0">Checklist</h3>
      <ul style="padding-left:18px">
        <li>${specOK ? "✅" : "⬜️"} Thesis written</li>
        <li>${hasM ? "✅" : "⬜️"} Matrices: ${esc(src)}</li>
      </ul>
      <label class="field" for="disc">Disclosures</label>
      <div class="hint">Which AI tools helped with the spec or hand-edits? Could your cited literature be in Claude's training data?</div>
      <textarea id="disc" ${S.meta.frozen ? "disabled" : ""} placeholder="e.g. Drafted with Claude; cites the emergent-misalignment paper (2025), likely in training data.">${esc(me.disclosure || "")}</textarea>
      <div class="callout warn small"><strong>Submitting is final.</strong> Your spec and matrices lock, and in return you can see the predictions
        of the host's nine framings (H1–H9) next to yours.</div>
      ${active ? `<div class="callout small">A run is still in progress. Wait for it to finish before submitting.</div>` : ""}`}
      <div class="row" style="margin-top:16px">
        ${me.submitted_at ? "" : `<button class="btn primary" id="sub" ${!hasM || active || S.meta.frozen ? "disabled" : ""}>Submit final entry</button>`}
        <span class="spacer"></span>
        <a class="btn" href="/api/me/download?token=${encodeURIComponent(token())}">Download everything (.zip)</a>
      </div>
      <p class="small muted" style="margin-bottom:0">Submitting lists your handle and title on the public Entries page. Your spec and numbers stay private until freeze.
        Optionally also post your handle on the LessWrong challenge thread.</p>
    </div>`;
  if ($("#sub")) $("#sub").onclick = async () => {
    if (!confirm("Submit your final entry? You won't be able to change your spec or matrices afterwards.")) return;
    try { await api("/api/me/submit", { method: "POST", body: JSON.stringify({ disclosure: $("#disc").value }) }); await loadMe(); drawMe(); toast("Submitted. H1–H9 predictions unlocked."); }
    catch (e) { toast(e.message); }
  };
}

// ------------------------------------------------------------------ examples (host H1–H9)

function spearman(a, b) {
  const rank = (v) => {
    const idx = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(v.length);
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2;
      i = j + 1;
    }
    return r;
  };
  if (a.length < 3) return null;
  const ra = rank(a), rb = rank(b), n = a.length;
  const ma = ra.reduce((s, x) => s + x, 0) / n, mb = rb.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; }
  return da && db ? num / Math.sqrt(da * db) : null;
}

function agreement(mineCSV, theirsCSV) {
  if (!mineCSV || !theirsCSV) return null;
  const A = parseCSV(mineCSV), B = parseCSV(theirsCSV);
  const xs = [], ys = [];
  A.rows.forEach((r, i) => A.cols.forEach((c, j) => {
    const bi = B.rows.indexOf(r), bj = B.cols.indexOf(c);
    if (isDiag(r, c) || bi < 0 || bj < 0) return;
    const x = A.vals[i][j], y = B.vals[bi][bj];
    if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) return;
    xs.push(x); ys.push(y);
  }));
  return spearman(xs, ys);
}

async function renderExamples(h) {
  const sel = decodeURIComponent((h.split("/")[2] || "").split("?")[0]);
  view().innerHTML = `<div class="ws-head"><span class="spinner"></span></div>`;
  if (!S.examples) S.examples = await api("/api/examples");
  let me = null, preds = null;
  if (token()) {
    try { me = S.me || (await loadMe()); } catch {}
    if (me?.submitted_at) { try { preds = S.examplePreds || (S.examplePreds = await api("/api/examples/predictions")); } catch {} }
  }
  const ex = S.examples.find((x) => x.id === sel);
  view().innerHTML = `
    <div class="ws-head"><h1>The host's framings: H1–H9</h1>
      <p class="muted" style="max-width:740px">These are the host's nine hypotheses, predicted by <code>${esc(S.meta.model)}</code> with the same
        web harness your runs use. Read them for inspiration, or start your own entry from one. ${preds ? "You've submitted, so their frozen predictions are unlocked below." :
        "Their <strong>predictions stay locked until you submit your own entry</strong>, so you aren't anchored on their numbers."}</p></div>
    <div class="ex-layout">
      <div class="ex-list">${S.examples.map((x) => `
        <a class="card ex-item ${x.id === sel ? "active" : ""}" href="#/examples/${x.id}">
          <span class="mono small">${esc(x.id)}</span><span>${esc(x.title)}</span>
          ${preds && me?.plus_csv ? agreePill(me, preds, x) : ""}</a>`).join("")}</div>
      <div id="exDetail">${ex ? "" : `<div class="card muted">Pick a framing on the left.</div>`}</div>
    </div>`;
  if (ex) drawExample(ex, me, preds);
}

function agreePill(me, preds, x) {
  const rs = x.runs.map((id) => [agreement(me.plus_csv, preds[id]?.plus_csv), agreement(me.minus_csv, preds[id]?.minus_csv)]);
  const vals = rs.flat().filter((v) => v != null);
  if (!vals.length) return "";
  const m = vals.reduce((s, v) => s + v, 0) / vals.length;
  return `<span class="pill" title="Mean Spearman ρ between your prediction and this framing's (agreement, not a score)">ρ ${m >= 0 ? "+" : ""}${m.toFixed(2)} with yours</span>`;
}

function drawExample(ex, me, preds) {
  const runId = S.exRun && ex.runs.includes(S.exRun) ? S.exRun : ex.runs[0];
  const p = preds?.[runId];
  const canStart = !me || !me.submitted_at;
  $("#exDetail").innerHTML = `
    <div class="card">
      <div class="row"><span class="pill mono">${esc(ex.id)}</span><span class="spacer"></span>
        ${canStart && !S.meta.frozen ? `<button class="btn primary small" id="useEx">Start my entry from this</button>` : ""}</div>
      ${p ? `
        ${ex.runs.length > 1 ? `<div class="runs">${ex.runs.map((id) => `<button class="btn small ${id === runId ? "active" : ""}" data-exrun="${id}">${esc(id)}</button>`).join("")}
          <span class="small muted" style="align-self:center">Same prompt, three runs. The spread shows run-to-run variance.</span></div>` : ""}
        <div class="row" style="margin:10px 0"><strong>Frozen prediction</strong>
          ${me?.plus_csv ? `<span class="small muted">agreement with yours: plus ρ ${fmtRho(agreement(me.plus_csv, p.plus_csv))} · minus ρ ${fmtRho(agreement(me.minus_csv, p.minus_csv))}</span>` : ""}
          <span class="spacer"></span><label class="small row" style="gap:6px"><input type="checkbox" id="exVals"> show values</label></div>
        <h3>logitz_plus</h3><div class="matrix-wrap" id="exP"></div>
        <h3>logitz_minus</h3><div class="matrix-wrap" id="exM"></div>
        ${me?.plus_csv ? `<details><summary>Your prediction, for comparison</summary><h3>logitz_plus</h3><div class="matrix-wrap" id="myP"></div><h3>logitz_minus</h3><div class="matrix-wrap" id="myM"></div></details>` : ""}
        <details><summary>method.md</summary><div class="md">${md(p.method_md)}</div></details>
        <details><summary>falsifiers.md</summary><div class="md">${md(p.falsifiers_md)}</div></details>`
      : `<div class="callout small">🔒 This framing's predicted matrices unlock after you <a href="#/me/submit">submit your own entry</a>.</div>`}
      <details ${p ? "" : "open"}><summary>Spec</summary><div class="md">${md(ex.spec_md)}</div></details>
    </div>`;
  document.querySelectorAll("[data-exrun]").forEach((b) => (b.onclick = () => { S.exRun = b.dataset.exrun; drawExample(ex, me, preds); }));
  if (p) {
    const draw = () => {
      const sv = $("#exVals").checked;
      matrixTable($("#exP"), parseCSV(p.plus_csv), { showValues: sv });
      matrixTable($("#exM"), parseCSV(p.minus_csv), { showValues: sv });
      if ($("#myP")) { matrixTable($("#myP"), parseCSV(me.plus_csv), { showValues: sv }); matrixTable($("#myM"), parseCSV(me.minus_csv), { showValues: sv }); }
    };
    $("#exVals").onchange = draw;
    draw();
  }
  const use = $("#useEx");
  if (use) use.onclick = async () => {
    const thesis = ex.spec_text.trim();
    const title = `Based on ${ex.id}: ${ex.title}`.slice(0, 200);
    if (!token()) {
      const h = prompt("Pick a handle for your entry (letters, digits, _ or -):");
      if (!h) return;
      try { const r = await api("/api/handles", { method: "POST", body: JSON.stringify({ handle: h.trim() }) }); setToken(r.token); }
      catch (e) { return toast(e.message); }
    } else {
      const cur = await loadMe();
      const hasText = Object.values(cur.sections || {}).some((v) => (v || "").trim());
      if (hasText && !confirm("Replace your current spec with this framing? Your current text will be overwritten.")) return;
    }
    try {
      await api("/api/me/spec", { method: "PUT", body: JSON.stringify({ title, sections: { statement: thesis } }) });
      S.me = null; S.tab = "spec"; location.hash = "#/me/spec";
      toast(`Copied ${ex.id} into your thesis. Edit it to make it yours.`);
    } catch (e) { toast(e.message); }
  };
}

function fmtRho(v) { return v == null ? "n/a" : (v >= 0 ? "+" : "") + v.toFixed(2); }

// ------------------------------------------------------------------ entries

async function renderEntries() {
  view().innerHTML = `<div class="ws-head"><h1>Entries</h1><p class="muted">Submitted handles. Specs and predictions are revealed at freeze,
    together with the host's nine H1–H9 framings and the leaderboard.</p></div><div id="el"><span class="spinner"></span></div>`;
  try {
    const rows = await api("/api/entries");
    $("#el").innerHTML = rows.length ? `<div class="card" style="padding:4px 12px;overflow-x:auto"><table class="list"><thead><tr><th>Handle</th><th>Title</th><th>Tier</th><th>Submitted</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td class="mono">H${esc(r.handle)}</td><td>${esc(r.title)}</td><td><span class="pill">${esc(r.tier)}</span></td><td class="small muted">${esc(fmtTime(r.submitted_at))}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted">No submissions yet. <a href="#/">Be the first.</a></p>`;
  } catch (e) { $("#el").innerHTML = `<div class="callout err">${esc(e.message)}</div>`; }
}

boot();

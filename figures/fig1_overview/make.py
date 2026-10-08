"""Figure 1 (experiment overview): writes fig1_overview.svg and, if rsvg-convert is available, fig1_overview.pdf.

    python figures/fig1_overview/make.py
"""
import numpy as np

W, H = 1600, 600
F = "Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2328", "#57606a"
C = {  # stroke, fill, panel
    "blue": ("#2f6fde", "#ffffff", "#eef3fc"),
    "green": ("#2a9a4a", "#ffffff", "#edf7ef"),
    "purple": ("#8a3ad6", "#ffffff", "#f5eefc"),
}
out = []
a = out.append


def text(x, y, s, size=17, weight="normal", fill=INK, anchor="middle", style=""):
    a(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{weight}" '
      f'fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')


def box(x, y, w, h, col, lines, title=None, rx=10, fill=None, size=15, dashed=False):
    s, f, _ = C[col]
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or f}" '
      f'stroke="{s}" stroke-width="2"{dash}/>')
    n = len(lines) + (1 if title else 0)
    lh = size + 5
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    i = 0
    if title:
        text(x + w / 2, y0, title, size=size + 1, weight="bold")
        i = 1
    for ln in lines:
        text(x + w / 2, y0 + i * lh, ln, size=size, fill=MUTED if title else INK)
        i += 1


def arrow(x1, y1, x2, y2):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5a6068" stroke-width="2.5" '
      f'marker-end="url(#ah)"/>')


def panel(x, w, col, label):
    s, _, p = C[col]
    L = len(label.replace("&amp;", "&"))
    a(f'<rect x="{x}" y="20" width="{w}" height="{H-40}" rx="16" fill="{p}" stroke="{s}" '
      f'stroke-opacity="0.25" stroke-width="1.5"/>')
    a(f'<rect x="{x+18}" y="36" width="{L*10.5+36}" height="34" rx="17" fill="{s}"/>')
    text(x + 36 + L * 5.25, 59, label, size=17, weight="bold", fill="#ffffff")


a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
a('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
  'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5a6068"/></marker></defs>')
a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------- Panel 1: build evals ----------------
P1x, P1w = 20, 420
panel(P1x, P1w, "blue", "STEP 1 · Build evals")
bx, bw = P1x + 30, P1w - 60
box(bx, 95, bw, 70, "blue", ["prosocial, antisocial, self-oriented"], title="24 propensities",
    fill="#dce8fb")
arrow(bx + bw / 2, 165, bx + bw / 2, 193)
box(bx, 195, bw, 175, "blue",
    ["questions (train / test split)",
     "high- &amp; low-expression reference answers",
     "trait-eliciting system prompts",
     "LLM-judge rubric, score 0–100"],
    title="Per propensity, generate")
arrow(bx + bw / 2, 370, bx + bw / 2, 398)
box(bx, 400, bw, 80, "blue", ["drop questions that load on", "unintended traits"],
    title="Filter cross-trait confounds")
text(P1x + P1w / 2, 520, "24 eval sets, one judge rubric each", size=15, fill=MUTED)

# ---------------- Panel 2: elicit &amp; measure ----------------
P2x, P2w = 480, 540
panel(P2x, P2w, "green", "STEP 2 · Elicit &amp; measure")
methods = ["System prompt", "ICL", "SFT", "GRPO"]
mw, gap = 112, 12
mx0 = P2x + (P2w - (4 * mw + 3 * gap)) / 2
for i, m in enumerate(methods):
    x = mx0 + i * (mw + gap)
    lines = ["System", "prompt"] if m == "System prompt" else [m]
    box(x, 95, mw, 56, "green", lines, size=15)
text(P2x + P2w / 2, 178, "Qwen3.5-9B · Qwen3.5-9B-Base · Nemotron-3-Super-120B",
     size=14, fill=MUTED)
arrow(P2x + P2w / 2, 188, P2x + P2w / 2, 208)
box(P2x + 40, 210, P2w - 80, 66, "green", ["then score the model on all 24 evals"],
    title="Elicit propensity A (+ or − pole)", fill="#d9efdf")
arrow(P2x + P2w / 2, 276, P2x + P2w / 2, 300)

# heatmap glyph: rows = elicited A, cols = evaluated B
rng = np.random.default_rng(3)
n, cs = 9, 19
hx = P2x + 120
hy = 312
M = rng.normal(0, 0.35, (n, n))
M = M + 0.5 * rng.normal(0, 0.4, (n, 1)) * rng.normal(0, 1, (1, n))
np.fill_diagonal(M, 1.0)
M = np.clip(M, -1, 1)


def rdbu(v):
    # white -> red (positive), white -> blue (negative)
    r1, b1 = np.array([200, 40, 50]), np.array([40, 90, 190])
    t = abs(v) ** 0.8
    c = (1 - t) * np.array([255, 255, 255]) + t * (r1 if v > 0 else b1)
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


for i in range(n):
    for j in range(n):
        a(f'<rect x="{hx+j*cs}" y="{hy+i*cs}" width="{cs}" height="{cs}" fill="{rdbu(M[i,j])}" '
          f'stroke="#ffffff" stroke-width="1"/>')
for i in range(n):
    a(f'<rect x="{hx+i*cs}" y="{hy+i*cs}" width="{cs}" height="{cs}" fill="none" '
      f'stroke="{INK}" stroke-width="1.6"/>')
side = n * cs
text(hx + side / 2, hy + side + 22, "evaluated propensity B", size=13, fill=MUTED)
text(hx - 12, hy + side / 2, "elicited A", size=13, fill=MUTED,
     style=f'transform="rotate(-90 {hx-12} {hy+side/2})"')
# legend to the right of the glyph
lx = hx + side + 28
text(lx, hy + 30, "Spillover matrix", size=16, weight="bold", anchor="start")
text(lx, hy + 54, "Δ score vs. base model,", size=14, fill=MUTED, anchor="start")
text(lx, hy + 74, "logit-transformed, z-scored", size=14, fill=MUTED, anchor="start")
a(f'<rect x="{lx}" y="{hy+92}" width="16" height="16" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
text(lx + 24, hy + 105, "diagonal = on-target", size=14, fill=MUTED, anchor="start")
a(f'<rect x="{lx}" y="{hy+118}" width="16" height="16" fill="{rdbu(-0.7)}"/>')
a(f'<rect x="{lx+16}" y="{hy+118}" width="16" height="16" fill="{rdbu(0.7)}"/>')
text(lx + 40, hy + 131, "off-diagonal = spillover", size=14, fill=MUTED, anchor="start")
text(P2x + P2w / 2, 548, "one matrix per method × model", size=15, fill=MUTED)

# ---------------- Panel 3: analyze ----------------
P3x, P3w = 1060, 520
panel(P3x, P3w, "purple", "STEP 3 · Analyze")
groups = [
    ("Compare elicitation methods", ["cheap (SP, ICL) vs. costly (SFT, GRPO)", "symmetry: A→B vs. B→A"]),
    ("Forecast spillover", ["Claude Opus 5.5 forecasts · self-prediction", "human-psychology prior · empirical baselines"]),
    ("Latent structure", ["trait factors (EFA) vs. human constructs", "fine-tune directions in activation space"]),
    ("Training-pipeline confounds", ["null fine-tune (no target trait)", "data-generating model"]),
]
gy, gh, gg = 95, 104, 14
for k, (t, ls) in enumerate(groups):
    box(P3x + 30, gy + k * (gh + gg), P3w - 60, gh, "purple", ls, title=t, size=14)

# connectors between panels
arrow(P1x + P1w + 2, 300, P2x - 4, 300)
arrow(P2x + P2w + 2, 300, P3x - 4, 300)

a("</svg>")
open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "fig1_overview.svg"), "w").write(
    "\n".join(out))

# PDF for LaTeX (needs librsvg: `brew install librsvg` / `apt install librsvg2-bin`)
import os as _os, shutil as _sh, subprocess as _sp  # noqa: E402
_here = _os.path.dirname(_os.path.abspath(__file__))
if _sh.which("rsvg-convert"):
    _sp.run(["rsvg-convert", "-f", "pdf", "-o", _os.path.join(_here, "fig1_overview.pdf"),
             _os.path.join(_here, "fig1_overview.svg")], check=True)
    print("wrote", _os.path.join(_here, "fig1_overview.pdf"))
else:
    print("rsvg-convert not found; wrote the SVG only")

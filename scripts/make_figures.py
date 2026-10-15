"""Figures of the report (anonymous: one point per class, no teacher or class labels).

    python scripts/make_figures.py      # needs data/private/worksheet_located.csv and results/
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from _path import PRIVATE, ROOT
from crmpilot.analysis import USAGE, prepare
from crmpilot.plotting import C, INK, MUTED, plt, save

COL = {"Bajo": C[0], "Medio": C[3], "Intensivo": C[1]}
LABEL = {"Bajo": "low / none", "Medio": "medium", "Intensivo": "intensive"}
FIG = ROOT / "figures"
ws = pd.read_csv(PRIVATE / "worksheet_located.csv")
d = prepare(ws)

# ---------------------------------------------------------------- 1. outcome definition: ties at r = 1
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, u in enumerate(USAGE):
    r = ws[ws.usage == u].r_ws.round(6)
    parts = [(r < 1).mean(), (r == 1).mean(), (r > 1).mean()]
    left = 0
    for k, (w, c) in enumerate(zip(parts, [MUTED, "#bbbbbb", COL[u]])):
        ax.barh(i, w, left=left, color=c, edgecolor="white")
        ax.text(left + w / 2, i, f"{100 * w:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if k != 1 else INK)
        left += w
ax.set_yticks(range(3), [f"{LABEL[u]} (n={sum(ws.usage == u)})" for u in USAGE])
ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_xlabel("share of classes:   r < 1   |   r = 1 (no change)   |   r > 1")
save(fig, FIG / "ties.png")

# ---------------------------------------------------------------- 2. regression to the mean
fig, ax = plt.subplots(figsize=(6.4, 4.2))
xs = np.linspace(d.mean_s1_all.min(), d.mean_s1_all.max(), 50)
for u in USAGE:
    g = d[d.usage == u]
    ax.scatter(g.mean_s1_all, 100 * g.logr, s=16, color=COL[u], alpha=0.75, label=f"{LABEL[u]} ({len(g)} classes)",
               edgecolor="none")
fit = smf.ols("logr ~ mean_s1_all", d).fit()
ax.plot(xs, 100 * fit.predict(pd.DataFrame({"mean_s1_all": xs})), color=INK, lw=1.4, label="common trend")
for u in USAGE:
    ax.axvline(d[d.usage == u].mean_s1_all.mean(), color=COL[u], lw=1, ls="--")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("class mean, semester 1 (grade, 1-7)")
ax.set_ylabel("change to semester 2: 100 log r  [%]")
ax.legend(fontsize=8, loc="lower left")
save(fig, FIG / "baseline.png")

# ---------------------------------------------------------------- 3. effect estimates: claims vs re-analysis
m = pd.read_csv(ROOT / "results" / "models.csv")
m = m[m.usage == "Intensivo"].reset_index(drop=True)
claims = [("original: +6.1% (improving classes only)", 0.061), ("original: mixed model", 0.035)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = len(m) + len(claims) - 1
for name, v in claims:
    ax.plot(100 * v, y, "D", color=MUTED)
    ax.text(100 * v + 0.15, y, f" {100 * v:.1f}%", va="center", fontsize=8, color=MUTED)
    y -= 1
labels = [n for n, _ in claims]
for r in m.itertuples():
    ax.errorbar(100 * r.effect, y, xerr=[[100 * (r.effect - r.ci_low)], [100 * (r.ci_high - r.effect)]],
                fmt="o", color=C[1] if "baseline" in r.model else C[0], capsize=3)
    labels.append(r.model)
    y -= 1
ax.set_yticks(range(len(labels) - 1, -1, -1), labels, fontsize=8)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("intensive vs low use: difference in semester change [%] (95% CI)")
ax.grid(axis="y", visible=False)
save(fig, FIG / "effects.png")
print("figures written to", FIG)

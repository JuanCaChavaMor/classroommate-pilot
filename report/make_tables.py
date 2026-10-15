"""LaTeX tables of the report from results/.

    python report/make_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)
LABEL = {"Bajo": "low / none", "Medio": "medium", "Intensivo": "intensive"}


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def tab(spec, header, rows):
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sgn(x, d=1):
    return f"{100 * x:+.{d}f}".replace("-", "$-$")


# replication of the original numbers
r = pd.read_csv(RES / "replication.csv")
rows = []
for version, g in r.groupby("version", sort=False):
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{version}}}}} \\\\")
    for v in g.set_index("Uso").reindex(["Intensivo", "Medio", "Bajo"]).reset_index().itertuples():
        rows.append(f"\\quad {LABEL[v.Uso]} & {v.N} & {v.mean_r:.3f} & {pct(v.p)} & "
                    f"[{pct(v.ci_low)}, {pct(v.ci_high)}] & {pct(v.p_strict)} \\\\")
write("replication", tab("lrrrcr", r"usage & classes & mean $r$ & $P(r \ge 1)$ [\%] & 95\% CI & $P(r > 1)$ [\%]",
                          rows))

# original models
m = pd.read_csv(RES / "replication_models.csv")
rows = [f"{v.model} & {v.version} & {LABEL[v.usage]} & {v.coef:.3f} & {v.p:.4f} \\\\" for v in m.itertuples()]
write("replication_models", tab("lllrr", "model & version & usage & coef. & $p$", rows))

# description of the located rows
d = pd.read_csv(RES / "describe.csv", index_col=0).reindex(["Intensivo", "Medio", "Bajo"])
rows = [f"{LABEL[u]} & {int(v.located_classes)} & {int(v.teachers)} & {v.baseline_s1:.2f} & {v.located_mean_r:.3f} & "
        f"{pct(v.located_p_gt1)} \\\\" for u, v in d.iterrows()]
write("describe", tab("lrrrrr", r"usage & classes & teachers & baseline (sem.\,1) & mean $r$ & $P(r>1)$ [\%]", rows))

# model sequence
s = pd.read_csv(RES / "models.csv")
rows = []
for name, g in s.groupby("model", sort=False):
    g = g.set_index("usage")
    cells = [f"{sgn(g.loc[u, 'effect'])} [{sgn(g.loc[u, 'ci_low'])}, {sgn(g.loc[u, 'ci_high'])}] & {g.loc[u, 'p']:.3f}"
             for u in ("Intensivo", "Medio")]
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
write("models", tab("lrrrr", r"specification & \multicolumn{2}{c}{intensive vs low} & \multicolumn{2}{c}{medium vs low} \\ "
                    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n& difference [\\%] (95\\% CI) & $p$ & difference [\\%] (95\\% CI) & $p$",
                    rows))

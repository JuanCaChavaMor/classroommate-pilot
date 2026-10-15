"""Faithful replication of the original analysis (notebook 'Informe.ipynb', 2025) from its work sheet.

The work sheet holds, in six blocks (Basica / Media x Intensivo / Medio / Bajo), one ratio r of class means per
course x subject, without the class it belongs to, followed by summary rows (Promedio, <1, >=1, Maximo, Minimo).
The original code dropped the rows "<1", ">=1" and "Total" but kept "Promedio", "Maximo" and "Minimo" as if they
were classes. `original_long(..., keep_summary_rows=True)` reproduces that; False gives the corrected table.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.formula.api as smf

Z95 = 1.96


def _find_header_row(raw, max_rows=20):
    for i in range(max_rows):
        vals = [str(x).lower() for x in raw.iloc[i]]
        if any("materia" in v for v in vals) and any("delta" in v for v in vals):
            return i
    return 0


def original_long(path, keep_summary_rows=True):
    raw = pd.read_excel(path, header=None)
    df = pd.read_excel(path, header=_find_header_row(raw))
    materia_cols = [c for c in df.columns if str(c).startswith("Materia")]
    rows = []
    for idx, mcol in enumerate(materia_cols):
        m = list(df.columns).index(mcol)
        rcol, dcol = df.columns[m + 1], df.columns[m + 2]
        b = df[[mcol, rcol, dcol]].copy()
        b.columns = ["Materia", "r", "Delta"]
        b["Uso"] = "Intensivo" if "Intensivo" in str(rcol) else ("Medio" if "Medio" in str(rcol) else "Bajo")
        b["Nivel"] = "Basica" if idx < 3 else "Media"          # first three blocks are Basica (as in the original)
        rows.append(b)
    out = pd.concat(rows, ignore_index=True)
    out["r"] = pd.to_numeric(out["r"], errors="coerce")
    names = out["Materia"].map(str)
    drop = names.str.contains(r"^<1|^>=?1|^Total|^nan$", case=False)
    if not keep_summary_rows:
        drop |= names.str.contains(r"^Promedio|^M[aá]ximo|^M[ií]nimo", case=False)
    out = out[~drop].dropna(subset=["r"])
    out = out[(out["r"] >= 0.6) & (out["r"] <= 1.4)].copy()
    out["mejora"] = (out["r"] >= 1).astype(int)
    return out.reset_index(drop=True)


def summarize(df, by=("Uso",)):
    s = df.groupby(list(by)).agg(N=("r", "count"), mean_r=("r", "mean"), X=("mejora", "sum")).reset_index()
    s["p"] = s.X / s.N
    se = np.sqrt(s.p * (1 - s.p) / s.N)
    s["ci_low"], s["ci_high"] = s.p - Z95 * se, s.p + Z95 * se
    return s


def original_models(df):
    """The two models of the original report, fitted as there (rows treated as independent / grouped by subject)."""
    logit = smf.logit("mejora ~ C(Uso, Treatment('Bajo')) + C(Nivel)", data=df).fit(disp=0)
    mixed = sm.MixedLM.from_formula("r ~ C(Uso, Treatment('Bajo')) + C(Nivel)", groups="Materia", data=df).fit()
    return logit, mixed


# boundary fits of the random-effect variances are expected with few groups; the fixed effects are unaffected
warnings.filterwarnings("ignore", category=ConvergenceWarning)

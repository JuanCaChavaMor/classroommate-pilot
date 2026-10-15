"""Replicate the original report from its work sheet, with and without the summary-row error (Table 1).

    python scripts/replicate_original.py [work sheet .xlsx, default data/private/worksheet_original.xlsx]
"""
import sys

import pandas as pd

from _path import PRIVATE, ROOT
from crmpilot.replication import original_long, original_models, summarize

path = sys.argv[1] if len(sys.argv) > 1 else PRIVATE / "worksheet_original.xlsx"
rows, coefs = [], []
for keep, label in ((True, "original code (summary rows kept)"), (False, "summary rows removed")):
    d = original_long(path, keep_summary_rows=keep)
    s = summarize(d)
    s["p_strict"] = d.assign(up=(d.r.round(6) > 1)).groupby("Uso").up.mean().reindex(s.Uso).values
    s["version"] = label
    rows.append(s)
    logit, mixed = original_models(d)
    for name, m in (("logit: P(r >= 1)", logit), ("mixed (groups = subject): r", mixed)):
        for u in ("Intensivo", "Medio"):
            t = f"C(Uso, Treatment('Bajo'))[T.{u}]"
            coefs.append(dict(version=label, model=name, usage=u, coef=m.params[t], p=m.pvalues[t]))
    print(label, "rows:", len(d))
out = ROOT / "results"
out.mkdir(exist_ok=True)
pd.concat(rows).to_csv(out / "replication.csv", index=False, float_format="%.4f")
pd.DataFrame(coefs).to_csv(out / "replication_models.csv", index=False, float_format="%.4f")
print(pd.concat(rows).round(3).to_string(index=False))
print(pd.DataFrame(coefs).round(4).to_string(index=False))

"""Re-analysis on the located work-sheet rows (Tables 2-3, the numbers of the report).

    python scripts/reanalysis.py      # needs data/private/worksheet_located.csv (scripts/build_dataset.py)
"""
import json

import pandas as pd

from _path import PRIVATE, ROOT
from crmpilot.analysis import (baseline_slope, describe, minimum_detectable_effect, model_sequence, prepare,
                               teacher_permutation)

ws = pd.read_csv(PRIVATE / "worksheet_located.csv")
d = prepare(ws)
out = ROOT / "results"
describe(d, ws).to_csv(out / "describe.csv", float_format="%.4f")
models = model_sequence(d)
models.to_csv(out / "models.csv", index=False, float_format="%.4f")
slope, slope_ci = baseline_slope(d)
perm = teacher_permutation(d)
summary = dict(classes=len(d), teachers=int(d.teacher.nunique()), courses=int(d.course.nunique()),
               located_by_value=int((ws.how == "value").sum()), located_by_order=int((ws.how == "order").sum()),
               worksheet_rows=len(ws), baseline_slope=slope, baseline_slope_ci=slope_ci,
               permutation=perm, mde_teacher_level=minimum_detectable_effect(d),
               # the original "+6.1 %": mean increase over the *improving* intensive classes only (conditional on
               # the outcome); the same number for every usage level, over all 329 work-sheet rows
               mean_increase_improving_only={u: float(g[g.round(6) > 1].mean() - 1)
                                             for u, g in ws.groupby("usage").r_ws})
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(describe(d, ws).round(3).T.to_string())
print(models.round(4).to_string(index=False))
print(json.dumps(summary, indent=2, default=float))

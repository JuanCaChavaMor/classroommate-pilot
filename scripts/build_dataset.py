"""Private inputs -> anonymous tables in data/private (run on the machine that holds the school's files).

    python scripts/build_dataset.py <grade-book folder> <staff list .docx> <work sheet .xlsx>

Writes data/private/course_subject.csv (class means per course x subject, anonymous teacher id) and
data/private/worksheet_located.csv (the original work-sheet rows with class and teacher id). No name and no
individual grade is written. data/private/ is git-ignored.
"""
import sys

from _path import PRIVATE
from crmpilot.sources import build

general, staff, worksheet = sys.argv[1:4]
PRIVATE.mkdir(parents=True, exist_ok=True)
cs, ws = build(general, staff, worksheet)
cs.to_csv(PRIVATE / "course_subject.csv", index=False)
ws.to_csv(PRIVATE / "worksheet_located.csv", index=False)
print(f"{len(cs)} course x subject rows, {cs.teacher.nunique()} teachers")
print(f"{len(ws)} work-sheet rows: located by value {sum(ws.how == 'value')}, by order {sum(ws.how == 'order')}, "
      f"with teacher {ws.teacher.notna().sum()}")

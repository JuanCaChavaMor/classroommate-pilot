"""Parse the school's grade-book exports ("Informe Notas por curso", one .xlsx per class) into anonymous
course x subject aggregates.

Layout of an export: a header row with "Ord" and "Apellidos y nombre", the subject names in that row, and below
it the markers "1" / "2" (semester) and "F" (final) above each subject's columns; within a semester block the
column labelled "P" holds each student's semester average. Students are the rows with a numeric "Ord".

Only aggregates leave this module: number of students and the class means of the two semester averages. No name
and no individual grade is returned.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# subjects excluded in the original study (no focus on a discipline)
EXCLUDED = ("RELIGI", "ORIENTACI", "APOYO")


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def _is(val, *targets):
    v = str(val).strip()
    try:
        v = str(int(float(v)))
    except ValueError:
        pass
    return v in targets


def course_from_filename(path):
    """'5A_GENERAL.xlsx' -> '5A', 'IIC.xlsx' -> 'IIC'."""
    return Path(path).stem.split("_")[0].upper()


def level_of(course):
    """Basica (4 to 8 basico) or Media (I to IV medio)."""
    return "Media" if re.match(r"^(I|II|III|IV)[ABC]$", course) else "Basica"


def parse(path):
    """Return a DataFrame with one row per subject: course, level, section, subject, n, mean_s1, mean_s2.

    Blocks are delimited by the semester-1 markers ("1"), not by the subject titles: in some exports the title
    is shifted by a column with respect to its marker, and some blocks have no "F" column.
    """
    v = pd.read_excel(path, header=None)
    hrow = next(i for i in range(15) if any("APELLIDOS" in _norm(c) for c in v.iloc[i]))
    students = pd.to_numeric(v.iloc[hrow + 1:, 0], errors="coerce").notna()
    rows_idx = students[students].index
    ncol = v.shape[1]
    first = rows_idx[0]
    marks = v.iloc[hrow + 1:first]                          # rows between header and first student: "1", "2", "F", "P"

    def has(j, *t):
        return any(_is(c, *t) for c in marks[j])

    s1 = [j for j in range(2, ncol) if has(j, "1")]
    course = course_from_filename(path)
    out = []
    for k, a in enumerate(s1):
        b = s1[k + 1] if k + 1 < len(s1) else ncol
        titles = [str(v.iat[hrow, j]) for j in range(max(2, a - 1), b) if str(v.iat[hrow, j]) != "nan"
                  and not str(v.iat[hrow, j]).startswith("(")]
        if not titles:
            continue
        name = _norm(titles[0])
        if any(name.startswith(e) for e in EXCLUDED) or name.startswith("PROM"):
            continue
        s2 = [j for j in range(a, b) if has(j, "2")]
        pcols = [j for j in range(a, b) if any(str(c).strip() == "P" for c in marks[j])]
        if not s2:
            continue
        p1 = [j for j in pcols if j < s2[0]]
        p2 = [j for j in pcols if j > s2[0]]
        if not p1 or not p2:
            continue
        g1 = pd.to_numeric(v.loc[rows_idx, p1[0]], errors="coerce")
        g2 = pd.to_numeric(v.loc[rows_idx, p2[0]], errors="coerce")
        ok = g1.notna() & g2.notna() & (g1 > 0) & (g2 > 0)
        if ok.sum() < 3:
            continue
        a1, a2 = g1[g1 > 0], g2[g2 > 0]                     # every student graded in that semester
        out.append(dict(course=course, level=level_of(course), section=course[-1], subject=name,
                        n=int(ok.sum()), mean_s1=float(g1[ok].mean()), mean_s2=float(g2[ok].mean()),
                        mean_s1_all=float(a1.mean()), mean_s2_all=float(a2.mean())))
    df = pd.DataFrame(out)
    if len(df):
        df["r"] = df.mean_s2 / df.mean_s1
    return df


def parse_all(root):
    """All grade books under root (e.g. the folder 'general' with 'Cursos A/B/C')."""
    files = sorted(p for p in Path(root).rglob("*.xlsx") if not p.name.startswith("~"))
    return pd.concat([parse(f) for f in files], ignore_index=True)

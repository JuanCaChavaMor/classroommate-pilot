"""Build the anonymous analysis table from the school's private files.

Inputs (never part of the repository):
  * grade-book exports, one per class          -> r per course x subject (`gradebook.parse_all`)
  * staff list per class (.docx, one table per class: subject | teacher)  -> anonymous teacher id
  * the original study's work sheet (`cuentas.xlsx`): r per subject and usage level, in six blocks
    (Basica / Media x Intensivo / Medio / Bajo) without the class it belongs to -> usage label

The usage label is attached to a course x subject by matching the work sheet rows to the recomputed ratios
(same level, compatible subject, same ratio of the one-decimal class means) with an optimal assignment.
Teacher names are replaced by ids T01, T02, ... inside this module and are not written anywhere.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .gradebook import _norm

# canonical subject keys: (substring of the normalised name) -> key; first match wins
SUBJECT_KEYS = [
    ("FORMACION DIFERENCIADA", "QUIMFD"), ("QUIMICA FD", "QUIMFD"), ("HISTORIA DE LA QUIM", "HISTQUIM"),
    ("HIST. QUIM", "HISTQUIM"), ("ALGEBRA", "ALGEBRA"), ("MECANICA", "MECANICA"), ("TERMODIN", "TERMO"),
    ("FUNCIONES", "FUNCIONES"), ("CELULA", "CELULA"), ("EVOLUCION", "EVOLUCION"), ("LENGUAJE Y SOC", "LENGSOC"),
    ("LENG Y SOC", "LENGSOC"), ("LITERATURA E ID", "LITID"), ("LIT E ID", "LITID"), ("SOCIAL COM", "SOCCOM"),
    ("SOC. COM", "SOCCOM"), ("ARGUMENT", "ARGUM"), ("CULT. LIT", "ARGUM"), ("CIUDAD CONTEMP", "CIUDAD"),
    ("REALIDAD NACIONAL", "CSOCREAL"), ("FILOSOF", "FILOSOFIA"), ("INGLES", "INGLES"), ("MATEM", "MATEMATICA"),
    ("BIOLOG", "BIOLOGIA"), ("FISICA", "FISICA"), ("QUIMICA", "QUIMICA"), ("QUI. FUNDAM", "QUIMICA"),
    ("ARTES V", "ARTES"), ("ARTES MUS", "MUSICA"), ("MUSICA", "MUSICA"), ("TECNO", "TECNOLOGIA"),
    ("NATURALES", "CSNAT"), ("HISTORIA", "HISTORIA"), ("CS SOCIALES", "HISTORIA"), ("CS. SOCIALES", "HISTORIA"),
    ("LENGUA", "LENGUAJE"),
]


def subject_key(name):
    n = _norm(name)
    # "Ed. Fisica" / "Educacion Fisica" before the generic "FISICA"
    if n.startswith("ED") and "FISICA" in n:
        return "EDFISICA"
    for sub, key in SUBJECT_KEYS:
        if sub in n:
            return key
    return n


# ---------------------------------------------------------------- staff list -> anonymous teacher ids
def _docx_tables(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    for tbl in x.split("<w:tbl>")[1:]:
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, flags=re.S)
        yield [[re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S)]
               for r in rows]


def _course_code(text):
    """'CURSO: 4°A' -> '4A', 'CURSO: III° MEDIO B' -> 'IIIB'."""
    t = _norm(text).replace("CURSO:", "").replace("MEDIO", "").replace("BASICO", "")
    m = re.search(r"\b(IV|III|II|I|[4-8])\s*\S?\s*([ABC])\b", t)
    return f"{m.group(1)}{m.group(2)}" if m else None


def teacher_table(docx_path):
    """DataFrame course, subject_key, teacher_id. Names are mapped to ids and dropped."""
    rows, ids = [], {}
    for tbl in _docx_tables(docx_path):
        if not tbl or not tbl[0]:
            continue
        course = _course_code(tbl[0][0])
        if course is None:
            continue
        for cells in tbl[1:]:
            if len(cells) < 2 or not cells[0] or not cells[1]:
                continue
            subj, name = cells[0], _norm(cells[1])
            if subj.upper().startswith("PROFESOR JEFE"):
                continue
            ids.setdefault(name, f"T{len(ids) + 1:02d}")
            rows.append(dict(course=course, subject_key=subject_key(subj), teacher=ids[name]))
    return pd.DataFrame(rows).drop_duplicates(["course", "subject_key"])


# ---------------------------------------------------------------- work sheet of the original study
BLOCKS = [(1, "Basica", "Intensivo"), (5, "Basica", "Medio"), (9, "Basica", "Bajo"),
          (13, "Media", "Intensivo"), (17, "Media", "Medio"), (21, "Media", "Bajo")]
SUMMARY = {"PROMEDIO", "MAXIMO", "MINIMO", "<1", ">=1", ">= 1"}


def worksheet_rows(xlsx_path, first_row=10):
    """The 329 data rows of the original work sheet: level, usage, subject, r (summary rows dropped)."""
    v = pd.read_excel(xlsx_path, header=None)
    out = []
    for col, level, usage in BLOCKS:
        for i in range(first_row, len(v)):
            name, r = v.iat[i, col], pd.to_numeric(v.iat[i, col + 1], errors="coerce")
            if pd.isna(r) or str(name) == "nan" or _norm(name) in SUMMARY:
                continue
            out.append(dict(level=level, usage=usage, subject_ws=str(name), r_ws=float(r), order=i))
    return pd.DataFrame(out)


COURSES = ["4A", "4B", "4C", "5A", "5B", "5C", "6A", "6B", "6C", "7A", "7B", "7C", "8A", "8B", "8C",
           "IA", "IB", "IC", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA", "IVB", "IVC"]
RANK = {c: i for i, c in enumerate(COURSES)}


def _ratio_variants(cs):
    """Conventions for the class mean ratio; the work sheet mostly used one-decimal means over all students
    graded in each semester. Earlier entries are preferred in the assignment."""
    return [cs.mean_s2_all.round(1) / cs.mean_s1_all.round(1), cs.mean_s2.round(1) / cs.mean_s1.round(1),
            cs.mean_s2_all / cs.mean_s1_all, cs.r]


def _longest_nondecreasing(ranks):
    """Indices of a longest non-decreasing subsequence (O(n^2), n < 100 per block)."""
    n = len(ranks)
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if ranks[j] <= ranks[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = int(np.argmax(best)) if n else -1
    out = []
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def locate_worksheet_rows(cs, ws, tol=1e-4):
    """Give every work-sheet row the class (course) it belongs to.

    1. value matching: optimal one-to-one assignment of work-sheet rows to recomputed course x subject ratios
       (same level and subject, equal ratio under one of the conventions of `_ratio_variants`);
    2. order filter: within each block the rows are listed by class, so matches that break the class order are
       coincidences and are dropped (longest non-decreasing subsequence of class ranks);
    3. gap filling: an unmatched row lying between two retained matches of the same class belongs to that class.
    Returns ws with columns course and how ('value', 'order' or None).
    """
    cs = cs.reset_index(drop=True).copy()
    ws = ws.reset_index(drop=True).copy()
    cs["subject_key"] = cs.subject.map(subject_key) if "subject" in cs else cs.subject_key
    ws["subject_key"] = ws.subject_ws.map(subject_key)
    variants = _ratio_variants(cs)
    big = 1e6
    C = np.full((len(ws), len(cs)), big)
    for i, w in enumerate(ws.itertuples()):
        same = (cs.level.values == w.level) & (cs.subject_key.values == w.subject_key)
        for cost, var in enumerate(variants):
            ok = same & (np.abs(var.values - w.r_ws) < tol) & (C[i] == big)
            C[i, ok] = float(cost)
    ri, ci = linear_sum_assignment(C)
    keep = C[ri, ci] < big
    ws["course"], ws["how"] = None, None
    ws.loc[ri[keep], "course"] = cs.course.values[ci[keep]]
    for _, g in ws.groupby(["level", "usage"], sort=False):
        g = g.sort_values("order")
        m = g[g.course.notna()]
        good = m.index[_longest_nondecreasing([RANK[c] for c in m.course])]
        ws.loc[m.index.difference(good), "course"] = None
        ws.loc[good, "how"] = "value"
        # gap filling between consecutive anchors of the same class
        anchors = list(good)
        for a, b in zip(anchors[:-1], anchors[1:]):
            if ws.at[a, "course"] == ws.at[b, "course"]:
                between = g[(g.order > ws.at[a, "order"]) & (g.order < ws.at[b, "order"])].index
                gap = [k for k in between if ws.at[k, "course"] is None]
                ws.loc[gap, "course"], ws.loc[gap, "how"] = ws.at[a, "course"], "order"
    return ws


def worksheet_with_classes(cs, ws, teachers):
    """Work-sheet rows (the study's own usage labels and ratios) with class and anonymous teacher id."""
    w = locate_worksheet_rows(cs, ws)
    w = w.merge(teachers, on=["course", "subject_key"], how="left")
    base = cs.assign(subject_key=cs.subject.map(subject_key) if "subject" in cs else cs.subject_key)
    base = base[["course", "subject_key", "mean_s1_all", "mean_s2_all"]].drop_duplicates(["course", "subject_key"])
    return w.merge(base, on=["course", "subject_key"], how="left")


def build(general_dir, staff_docx, worksheet_xlsx):
    """Private inputs -> (course x subject table with teacher ids, work-sheet rows with class and teacher)."""
    from .gradebook import parse_all
    cs = parse_all(general_dir)
    cs["subject_key"] = cs.subject.map(subject_key)
    t = teacher_table(staff_docx)
    cs = cs.merge(t, on=["course", "subject_key"], how="left")
    ws = worksheet_with_classes(cs, worksheet_rows(worksheet_xlsx), t)
    cols = ["course", "level", "section", "subject_key", "teacher", "n", "mean_s1", "mean_s2", "mean_s1_all",
            "mean_s2_all", "r"]
    return cs[cols], ws

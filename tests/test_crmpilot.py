import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crmpilot.analysis import FORMULA, model_sequence  # noqa: E402
from crmpilot.gradebook import level_of, parse  # noqa: E402
from crmpilot.sources import _longest_nondecreasing, locate_worksheet_rows, subject_key  # noqa: E402


def synthetic_gradebook(path, n=8, seed=0):
    """A grade-book export with the quirks of the real ones (fictitious names and grades): the title of the
    second subject is shifted one column right of its semester marker, and the first student has a grade of 2
    right below the marker rows."""
    rng = np.random.default_rng(seed)
    ncol = 2 + 2 * 7
    v = pd.DataFrame(np.nan, index=range(6 + n), columns=range(ncol), dtype=object)
    v.iat[0, 0] = "Informe Notas por Curso"
    v.iat[2, 0] = "Ord"; v.iat[2, 1] = "Apellidos y nombre"
    for k, (title_col, start) in enumerate([(2, 2), (10, 9)]):
        v.iat[2, title_col] = ["MATEMÁTICA", "HISTORIA, GEOGRAFÍA Y CIENCIAS SOCIALES"][k]
        v.iat[3, start] = "1"; v.iat[4, start + 2] = "P"            # semester 1: grades, then P
        v.iat[3, start + 3] = "2"; v.iat[4, start + 5] = "P"        # semester 2: grades, then P
        v.iat[3, start + 6] = "F"
    for i in range(n):
        v.iat[5 + i, 0] = i + 1
        v.iat[5 + i, 1] = f"Student {i + 1}"
        for start, shift in ((2, 0.0), (9, 0.3)):
            s1 = rng.uniform(4, 6.5)
            v.iat[5 + i, start] = 2 if i == 0 else round(s1, 1)
            v.iat[5 + i, start + 2] = round(s1, 1)
            v.iat[5 + i, start + 5] = round(min(7, s1 + shift), 1)
    v.to_excel(path, header=False, index=False)


def test_parse_synthetic_gradebook(tmp_path):
    f = tmp_path / "7A_GENERAL.xlsx"
    synthetic_gradebook(f)
    df = parse(f)
    assert list(df.subject) == ["MATEMATICA", "HISTORIA, GEOGRAFIA Y CIENCIAS SOCIALES"]
    assert (df.n == 8).all() and (df.course == "7A").all()
    assert df.r.iloc[0] == pytest.approx(1.0, abs=1e-9)           # no change in the first subject
    assert df.r.iloc[1] > 1.03                                      # +0.3 grade points in the second
    assert "Student" not in df.to_csv()                             # nothing personal leaves the parser


def test_levels_and_subject_keys():
    assert level_of("8C") == "Basica" and level_of("IIIA") == "Media"
    assert subject_key("Ed. Física") == subject_key("EDUCACION FISICA Y SALUD") == "EDFISICA"
    assert subject_key("Física") == "FISICA"
    assert subject_key("Cs Sociales") == subject_key("HISTORIA, GEOGRAFIA Y CIENCIAS SOCIALES")
    assert subject_key("Inglés") == subject_key("IDIOMA EXTRANJERO (INGLES)")


def test_longest_nondecreasing():
    ranks = [0, 0, 5, 1, 1, 2, 0, 3]
    idx = _longest_nondecreasing(ranks)
    assert [ranks[i] for i in idx] == [0, 0, 1, 1, 2, 3]


def test_locate_drops_out_of_order_matches_and_fills_gaps():
    cs = pd.DataFrame(dict(course=["4A", "4A", "4A", "4B"], level="Basica", subject=["MATEMATICA", "MUSICA",
                      "HISTORIA", "MATEMATICA"], mean_s1=[5.0, 5.0, 5.0, 6.0], mean_s2=[5.5, 5.0, 4.0, 6.0]))
    cs["mean_s1_all"], cs["mean_s2_all"], cs["r"] = cs.mean_s1, cs.mean_s2, cs.mean_s2 / cs.mean_s1
    ws = pd.DataFrame(dict(level="Basica", usage="Bajo", order=[1, 2, 3, 4],
                           subject_ws=["Matemática", "Música", "Historia", "Matemática"],
                           r_ws=[1.1, 0.95, 0.8, 1.0]))            # row 2 matches nothing exactly
    out = locate_worksheet_rows(cs, ws)
    assert list(out.course) == ["4A", "4A", "4A", "4B"]
    assert list(out.how) == ["value", "order", "value", "value"]


def test_model_sequence_recovers_planted_association():
    rng = np.random.default_rng(1)
    rows = []
    for t in range(40):
        usage = ["Bajo", "Medio", "Intensivo"][t % 3]
        for c in range(4):
            s1 = rng.uniform(5, 6.5)
            logr = 0.03 * (usage == "Intensivo") - 0.04 * (s1 - 5.75) + rng.normal(0, 0.02)
            rows.append(dict(teacher=f"T{t}", course=f"C{c}", usage=usage, level=["Basica", "Media"][t % 2],
                             logr=logr, s1c=s1 - 5.75))
    m = model_sequence(pd.DataFrame(rows))
    est = m[(m.usage == "Intensivo") & m.model.str.startswith("+ baseline")].effect
    assert np.allclose(est, 0.03, atol=0.012)
    assert "logr" in FORMULA

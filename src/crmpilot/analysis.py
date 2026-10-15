"""Re-analysis of the pilot: what survives when the baseline and the clustering of classes are taken into account.

Unit: one course x subject (a class taught by one teacher). Outcome: log r, r = class mean of semester 2 / class
mean of semester 1 (log r ~ relative change). Usage (Intensivo / Medio / Bajo) was chosen per teacher and class,
together with the students: it is self-selected, so every estimate below is an association, not an effect.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.formula.api as smf

USAGE = ["Bajo", "Medio", "Intensivo"]
TERM = "C(usage, Treatment('Bajo'))[T.{}]"
FORMULA = "logr ~ C(usage, Treatment('Bajo')) + C(level)"


def prepare(ws):
    """Located work-sheet rows -> analysis frame (rows with class, teacher and baseline)."""
    d = ws.dropna(subset=["teacher", "mean_s1_all"]).copy()
    d["logr"] = np.log(d.r_ws)
    d["s1c"] = d.mean_s1_all - d.mean_s1_all.mean()            # centred baseline (class mean, semester 1)
    return d.reset_index(drop=True)


def model_sequence(d):
    """The same contrast under increasingly honest specifications. Returns a tidy DataFrame."""
    gt, gc = pd.factorize(d.teacher)[0], pd.factorize(d.course)[0]
    specs = [
        ("OLS, classes independent", FORMULA, dict()),
        ("OLS, SE clustered by teacher", FORMULA, dict(cov_type="cluster", cov_kwds={"groups": gt})),
        ("+ baseline, SE clustered by teacher", FORMULA + " + s1c", dict(cov_type="cluster", cov_kwds={"groups": gt})),
        ("+ baseline, SE clustered by class", FORMULA + " + s1c", dict(cov_type="cluster", cov_kwds={"groups": gc})),
    ]
    fits = [(name, smf.ols(f, d).fit(**kw)) for name, f, kw in specs]
    fits.append(("mixed model: teacher + class, + baseline",
                 sm.MixedLM.from_formula(FORMULA + " + s1c", groups="teacher", vc_formula={"course": "0 + C(course)"},
                                         data=d).fit()))
    rows = []
    for name, m in fits:
        for u in ("Intensivo", "Medio"):
            t = TERM.format(u)
            lo, hi = m.conf_int().loc[t]
            rows.append(dict(model=name, usage=u, effect=m.params[t], ci_low=lo, ci_high=hi, p=m.pvalues[t]))
    return pd.DataFrame(rows)


def baseline_slope(d):
    """Regression to the mean: change of log r per grade point of semester-1 class mean (clustered by teacher)."""
    m = smf.ols(FORMULA + " + s1c", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.teacher)[0]})
    return m.params["s1c"], m.conf_int().loc["s1c"].tolist()


def teacher_permutation(d, n_perm=10000, seed=0):
    """Permutation test at the level of teachers: residualise log r on level and baseline (usage left out),
    average per teacher, and compare Intensivo vs Bajo teachers under random relabelling of teachers.

    Teachers with more than one usage label get their majority label (the work sheet has 3 such teachers)."""
    res = smf.ols("logr ~ C(level) + s1c", d).fit().resid
    t = d.assign(res=res).groupby("teacher").agg(res=("res", "mean"),
                                                 usage=("usage", lambda s: s.value_counts().index[0]))
    t = t[t.usage.isin(["Intensivo", "Bajo"])]
    x, lab = t.res.values, (t.usage == "Intensivo").values
    obs = x[lab].mean() - x[~lab].mean()
    rng = np.random.default_rng(seed)
    perm = np.array([(lambda p: x[p].mean() - x[~p].mean())(rng.permutation(lab)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return dict(n_intensivo=int(lab.sum()), n_bajo=int((~lab).sum()), diff=obs, p=p)


def minimum_detectable_effect(d, alpha=0.05, power=0.8):
    """Smallest Intensivo vs Bajo difference in teacher-mean log r detectable with these teachers (two-sided)."""
    from scipy.stats import norm
    res = smf.ols("logr ~ C(level) + s1c", d).fit().resid
    t = d.assign(res=res).groupby("teacher").agg(res=("res", "mean"),
                                                 usage=("usage", lambda s: s.value_counts().index[0]))
    n1, n0 = (t.usage == "Intensivo").sum(), (t.usage == "Bajo").sum()
    sd = t.res.std()
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd * np.sqrt(1 / n1 + 1 / n0)


def describe(d, ws_all):
    """Per usage level: classes, teachers, baseline and outcome; for all work-sheet rows and the located ones."""
    def block(x, r):
        return x.groupby("usage").agg(classes=(r, "size"), mean_r=(r, "mean"),
                                      p_ge1=(r, lambda s: (s.round(6) >= 1).mean()),
                                      p_gt1=(r, lambda s: (s.round(6) > 1).mean()),
                                      p_eq1=(r, lambda s: (s.round(6) == 1).mean()))
    a = block(ws_all, "r_ws").add_prefix("all_")
    b = block(d, "r_ws").add_prefix("located_")
    c = d.groupby("usage").agg(teachers=("teacher", "nunique"), baseline_s1=("mean_s1_all", "mean"))
    return a.join(b).join(c).reindex(USAGE)


# boundary fits of the random-effect variances are expected with few groups; the fixed effects are unaffected
warnings.filterwarnings("ignore", category=ConvergenceWarning)

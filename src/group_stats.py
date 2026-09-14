"""Comparing two groups, the same way in both group tabs.

The Neurosoft tab and the spontaneous-EMG tab ask the same question of every
channel: does group B differ from group A? One implementation, so the two tables
read the same.

The unit of the test is the SUBJECT. A patient contributes several recordings to
one group (levels, repeats of a task, body positions), and those are not
independent observations: pooling them would count one person several times and
make any difference look surer than it is. So each subject's recordings within a
group are averaged first, and then

* when at least ``MIN_PAIRS`` subjects have recordings in both groups, the test is
  the Wilcoxon signed-rank test on their paired values — the within-subject
  design (stim against non stim in the same task, one level against another in
  the same patient);
* otherwise, with at least ``MIN_PER_GROUP`` subjects in each group, the
  Mann–Whitney U test on the subject means — two separate cohorts;
* otherwise there is no test, and the table says why.

The descriptive numbers (median and quartiles) are over RECORDINGS: they describe
what the dots on the plot are, and the numbers of recordings and subjects stand
next to them.

Nothing here touches Qt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIRS = 5
MIN_PER_GROUP = 3


def _clean(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    v = pd.to_numeric(frame[value], errors="coerce")
    return frame.assign(**{value: v})[v.notna() & np.isfinite(v)]


def _quartiles(values) -> tuple[float, float, float]:
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return np.nan, np.nan, np.nan
    return float(v.quantile(0.25)), float(v.median()), float(v.quantile(0.75))


def compare_two(frame: pd.DataFrame, value: str, group_col: str, a: str, b: str,
                subject_col: str = "subject") -> dict:
    """Group *b* against group *a* on one unit (a channel, a channel pair).

    Returns ``n_a, subjects_a, median_a, q1_a, q3_a`` (and the same for b),
    ``delta`` (median b − median a), ``ratio`` (median b / median a), ``test``,
    ``p`` and ``n_pairs``.
    """
    d = _clean(frame, value)
    da, db = d[d[group_col] == a], d[d[group_col] == b]
    sa = da.groupby(subject_col)[value].mean()
    sb = db.groupby(subject_col)[value].mean()
    q1a, meda, q3a = _quartiles(da[value])
    q1b, medb, q3b = _quartiles(db[value])
    both = sa.index.intersection(sb.index)

    try:
        from scipy.stats import mannwhitneyu, wilcoxon
    except Exception:                                  # noqa: BLE001 - reported, not raised
        mannwhitneyu = wilcoxon = None
    p = np.nan
    if len(both) >= MIN_PAIRS and wilcoxon is not None:
        x, y = sa.reindex(both), sb.reindex(both)
        if float((y - x).abs().sum()) > 0:
            try:
                p = float(wilcoxon(x, y).pvalue)
            except ValueError:
                p = np.nan
        test = f"Уилкоксон, парный: {len(both)} субъектов"
    elif len(sa) >= MIN_PER_GROUP and len(sb) >= MIN_PER_GROUP and mannwhitneyu is not None:
        try:
            p = float(mannwhitneyu(sa, sb, alternative="two-sided").pvalue)
        except ValueError:
            p = np.nan
        test = f"Манна–Уитни: {len(sa)} и {len(sb)} субъектов"
    else:
        test = f"мало данных ({len(sa)} и {len(sb)} субъектов)"

    return {
        "n_a": int(len(da)), "subjects_a": int(sa.size),
        "median_a": meda, "q1_a": q1a, "q3_a": q3a,
        "n_b": int(len(db)), "subjects_b": int(sb.size),
        "median_b": medb, "q1_b": q1b, "q3_b": q3b,
        "delta": medb - meda if np.isfinite(meda) and np.isfinite(medb) else np.nan,
        "ratio": medb / meda if np.isfinite(meda) and np.isfinite(medb) and meda != 0 else np.nan,
        "test": test, "p": p, "n_pairs": int(len(both)),
    }


def comparison_table(frame: pd.DataFrame, unit_col: str, value: str, group_col: str,
                     a: str, b: str, subject_col: str = "subject",
                     units: list[str] | None = None) -> pd.DataFrame:
    """``compare_two`` for every unit, one row each, in the order of *units*."""
    if frame.empty:
        return pd.DataFrame()
    order = units if units is not None else list(dict.fromkeys(frame[unit_col]))
    rows = []
    for unit in order:
        d = frame[frame[unit_col] == unit]
        if d.empty:
            continue
        rows.append({unit_col: unit} | compare_two(d, value, group_col, a, b, subject_col))
    return pd.DataFrame(rows)


def describe_groups(frame: pd.DataFrame, unit_col: str, value: str, group_col: str,
                    groups: list[str], subject_col: str = "subject",
                    units: list[str] | None = None) -> pd.DataFrame:
    """Recordings, subjects, median and quartiles of every group on every unit."""
    d = _clean(frame, value) if not frame.empty else frame
    if d.empty:
        return pd.DataFrame()
    order = units if units is not None else list(dict.fromkeys(d[unit_col]))
    rows = []
    for unit in order:
        du = d[d[unit_col] == unit]
        for g in groups:
            dg = du[du[group_col] == g]
            if dg.empty:
                continue
            q1, med, q3 = _quartiles(dg[value])
            rows.append({unit_col: unit, "group": g, "n": int(len(dg)),
                         "subjects": int(dg[subject_col].nunique()),
                         "median": med, "q1": q1, "q3": q3})
    return pd.DataFrame(rows)


def format_number(v, digits: int = 3) -> str:
    """A table cell: three significant digits, thousands without an exponent."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{digits}g}"


def format_p(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"

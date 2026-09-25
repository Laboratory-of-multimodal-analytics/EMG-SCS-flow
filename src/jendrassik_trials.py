"""Jendrassik trials: five curves of rest against five of the manoeuvre.

Reworked after A. Militskova / A. D. (September 2026). The previous reading
(``src/curve_blocks.py``) looked for block boundaries in the data itself and gave
each channel a verdict from the spread within a block. The protocol, however, is
fixed: five stimuli at rest, five with the manoeuvre, then the next intensity and
the same again. So the run is cut rigidly — five in a row — and the pentads are
paired into rest/manoeuvre trials, each compared against itself.

In roughly a third of the recordings the number of pentads is odd; the last one is
left unpaired and enters no comparison.

Everything is judged on its ABSOLUTE value: the manoeuvre can enhance the response
or suppress it, and a fall is as much an effect as a rise. So a channel's best
trial is the one with the largest |Cohen's d|, and the sign is there only to read
the table by.

An undetected response (NaN in the metrics) counts as an amplitude of ZERO by
default, not as a missing value: a channel whose response the manoeuvre put out
altogether is showing the strongest effect there is, and losing it would be worse
than any rounding. Switch with ``MISSING_IS_ZERO`` or with each function's
``missing_is_zero`` argument.

Knows nothing about Qt, matplotlib or the folder layout: arrays in, numbers out.
Both the pipeline (``src/jendrassik.py``) and the interactive surface
(``emgflow/widgets/scenario_viewer.py``) go through it, so the trials on screen
and the trials in the saved table are computed by one piece of code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .curve_blocks import curve_span

#: Curves in one block: five of rest, five of the manoeuvre.
BLOCK_SIZE = 5

#: An undetected response is an amplitude of 0, not "no data". See the module docstring.
MISSING_IS_ZERO = True


def split_blocks(n_curves: int, size: int = BLOCK_SIZE) -> list[range]:
    """Curve positions by block: [0:5], [5:10], … A tail shorter than *size* is dropped."""
    return [range(s, s + size) for s in range(0, n_curves - size + 1, size)]


def pair_blocks(blocks: list[range]) -> list[tuple[int, range, range]]:
    """(trial number, rest block, manoeuvre block). An unpaired last block is dropped.

    ``zip`` stops at the shorter of the two slices, so the odd pentad falls away by
    itself and needs no check of its own.
    """
    return [(k, rest, act)
            for k, (rest, act) in enumerate(zip(blocks[::2], blocks[1::2]), start=1)]


def _prepare(values, missing_is_zero: bool) -> np.ndarray:
    """NaN either becomes zero (length kept) or is dropped (length falls)."""
    v = np.asarray(values, dtype=float)
    if missing_is_zero:
        return np.nan_to_num(v, nan=0.0)
    return v[np.isfinite(v)]


def compare_pair(rest, act, missing_is_zero: bool | None = None) -> dict:
    """One trial on one channel: the pentad of rest against the pentad of manoeuvre.

    The sign is "manoeuvre minus rest" throughout: plus is a rise, minus a fall.

    The two guards before the tests are there for different reasons. With zero
    spread in BOTH groups Welch divides by zero and returns ``p = 0.0`` — a number
    that reads as absolute significance although there is nothing to test.
    Mann-Whitney does not mind one group having no spread (five values strictly
    below five others is a clean separation); it only needs some spread across all
    ten values together.

    At 5 against 5 with no ties Mann-Whitney is exact, and its two-sided p never
    falls below 2/252 ≈ 0.0079 whatever the effect. With ``MISSING_IS_ZERO`` the
    zeros produce ties, scipy falls back to its approximation and that floor stops
    applying — but the approximation at n=5 is rough.
    """
    if missing_is_zero is None:
        missing_is_zero = MISSING_IS_ZERO
    rest = _prepare(rest, missing_is_zero)
    act = _prepare(act, missing_is_zero)

    n1, n2 = len(rest), len(act)
    m1 = float(rest.mean()) if n1 else np.nan
    m2 = float(act.mean()) if n2 else np.nan
    sd1 = float(rest.std(ddof=1)) if n1 > 1 else np.nan
    sd2 = float(act.std(ddof=1)) if n2 > 1 else np.nan

    delta = m2 - m1                                   # µV, signed
    rel = 100.0 * delta / m1 if m1 > 0 else np.nan    # % of rest, signed

    # Cohen's d: difference of means over the pooled SD
    pooled = (np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
              if n1 > 1 and n2 > 1 else np.nan)
    d = delta / pooled if np.isfinite(pooled) and pooled > 0 else np.nan

    p_welch = p_mw = np.nan
    if n1 >= 2 and n2 >= 2:
        # local import: a missing scipy must not break importing this module
        from scipy.stats import mannwhitneyu, ttest_ind

        if sd1 > 0 or sd2 > 0:
            p_welch = float(ttest_ind(act, rest, equal_var=False).pvalue)
        if np.ptp(np.concatenate([rest, act])) > 0:
            try:
                p_mw = float(mannwhitneyu(act, rest, alternative="two-sided").pvalue)
            except ValueError:           # older scipy raises when everything ties
                p_mw = np.nan

    return {
        "N rest": n1, "N act": n2,
        "Rest mean uV": m1, "Rest SD uV": sd1,
        "Act mean uV": m2, "Act SD uV": sd2,
        "Delta uV": delta, "Delta %": rel,
        "Cohen d": d,
        "p Welch": p_welch, "p Mann-Whitney": p_mw,
    }


def channel_trials(curves, values, missing_is_zero: bool | None = None) -> list[dict]:
    """Every trial of one channel: a dict per trial, in run order.

    ``curves`` holds the curve numbers in run order, ``values`` the measured
    quantity at the same places (NaN where no response was found). The
    ``_rest``/``_act`` keys carry the pentads' positions (``range`` objects) for
    whoever draws the curves; they do not go into the table — hence the underscore.

    One entry point for both the pipeline and the GUI: the trials on screen and the
    trials in the saved table have to be the same trials.
    """
    curves = np.asarray(curves, dtype=int)
    values = np.asarray(values, dtype=float)
    out = []
    for k, rest, act in pair_blocks(split_blocks(len(values))):
        out.append({
            "Trial": k,
            "_rest": rest, "_act": act,
            "_rest_curves": curves[rest.start:rest.stop],
            "_act_curves": curves[act.start:act.stop],
            "Rest curves": curve_span(curves[rest.start:rest.stop]),
            "Act curves": curve_span(curves[act.start:act.stop]),
            **compare_pair(values[rest.start:rest.stop],
                           values[act.start:act.stop], missing_is_zero),
        })
    return out


def best_trial(trials: list[dict]) -> int | None:
    """Number of the trial with the largest |d| (ties go to |gain %|), or None.

    On a tie ``max`` keeps the first element it met, and the trials come in order,
    so a complete tie goes to the earlier trial — exactly as ``mark_best`` does with
    its stable sort.
    """
    ok = [t for t in trials if np.isfinite(t.get("Cohen d", np.nan))]
    if not ok:
        return None
    best = max(ok, key=lambda t: (abs(t["Cohen d"]),
                                  abs(t["Delta %"]) if np.isfinite(t["Delta %"]) else -1.0))
    return int(best["Trial"])


def mark_best(rows: pd.DataFrame, by=("Recording", "Channel")) -> pd.DataFrame:
    """Mark each channel's best trial: max |d|, ties broken by max |gain %|.

    The absolute values go into columns of their own so that the saved table shows
    which number the choice was made on. ``kind="mergesort"`` is a stable sort: on a
    complete tie the lower-numbered trial wins rather than an arbitrary one, and the
    result does not change from run to run.

    A channel where d came out undefined in every trial (all ten values identical —
    in practice a dead channel) is left unmarked: there is nothing to choose between,
    and it has no business in the group base.
    """
    rows = rows.copy()
    rows["|d|"] = rows["Cohen d"].abs()
    rows["|Delta %|"] = rows["Delta %"].abs()
    rows["Best"] = ""

    ranked = rows[rows["Cohen d"].notna()].sort_values(
        ["|d|", "|Delta %|"], ascending=False, kind="mergesort")
    rows.loc[ranked.groupby(list(by), sort=False).head(1).index, "Best"] = "best"
    return rows

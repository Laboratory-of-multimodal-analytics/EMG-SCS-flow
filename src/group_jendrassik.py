"""The Jendrassik manoeuvre across recordings: the effect itself as a group quantity.

One recording's reading is ``src/jendrassik_trials.py``: the run is cut rigidly into
pentads (five stimuli at rest, five with the manoeuvre, then the next intensity), the
pentads are paired into trials, and a trial compares the manoeuvre against rest ON THE
SAME CHANNEL OF THE SAME RECORDING. This module lifts that to a group of recordings,
and the within-recording design is the reason it cannot reuse the Neurosoft group tab:

* the value compared between groups is a DIFFERENCE, not a level, so no normalisation
  across recordings is needed or wanted — each trial is already its own control;
* zero means "the manoeuvre did nothing", so every effect plot has a zero line and the
  first statistical question is the one-sample one (``group_stats.vs_zero``);
* "the last five curves" — what the Neurosoft tab summarises a recording by — is for a
  Jendrassik run whichever pentad happened to come last, rest or manoeuvre depending on
  the parity of the run. That number answers nothing here.

A recording contributes several trials per channel (one per pair of pentads, i.e. per
intensity), so the group base can be read two ways, and the tab shows both:

* ALL trials — the distribution of effects. Every trial is a dot;
* the BEST trial of each recording and channel — one dot per recording, the intensity
  at which the manoeuvre acted most. This is the row ``jendrassik_trials.csv`` marks
  on disk, and it is marked the same way here.

The best trial is always chosen on AMPLITUDE (|Cohen's d| on ``amp_uv``, with the
detection floor), whatever quantity is being plotted. Two reasons: the choice then
matches the per-recording table the pipeline wrote, and "where the manoeuvre acted"
stays a property of the recording instead of moving every time the metric combo is
switched.

There are two ways to a group table of trials, and the repository has both.
``group.collect_jendrassik_trials`` READS the ``jendrassik_trials.csv`` each finished
recording carries, which cannot disagree with that recording's own table by
construction. This module RECOMPUTES the trials from the per-curve metrics the group
tab has already read, which costs nothing extra, works on a recording processed
before the trials existed (and on an H-reflex run made with the manoeuvre, whose
scenario folder keeps no such table), and lets the tab offer a metric the saved table
does not carry. Checked against the saved tables of 16 recordings: 76 channels, every
number and every marked trial identical. The reader stays for whoever wants the file
itself.

Knows nothing about Qt or matplotlib: points in, tables out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import group as G
from . import jendrassik_trials as JT

#: Keys of the group points table that identify one curve series.
KEYS = ["subject", "state", "run", "config", "channel"]

#: The metric the best trial is chosen on, whatever is being plotted.
SELECTION_METRIC = "amp_uv"

#: Does an undetected response count as zero on this metric?
#:
#: For a size it does, and that is the whole point: a channel whose response the
#: manoeuvre put out shows the strongest effect there is (see the module docstring of
#: ``src/jendrassik_trials.py``). For a LATENCY it does not — a response that was never
#: found has no latency, and calling it 0 ms would invent a response at the stimulus.
MISSING_IS_ZERO = {"amp_uv": True, "p1_uv": True, "area_uvms": True,
                   "p1_ms": False, "onset_ms": False, "p2_ms": False}

#: The detection floor of the pooled SD, per metric, in the metric's own unit.
#:
#: ``STIM_PTP_MIN_UV`` is the smallest response the toolbox reports, so a pentad of
#: undetected curves is known only to lie below it — that is what the floor says.
#: There is no such number for an area or a latency, so they get no floor; a pentad
#: with no spread at all then simply has no Cohen's d, which is honest.
SD_FLOOR = {"amp_uv": JT.SD_FLOOR_UV, "p1_uv": JT.SD_FLOOR_UV}

#: The effect measures offered, and the column each reads.
EFFECTS = {
    "delta_pct": ("Delta %", "изменение, % от покоя"),
    "delta": ("Delta", "разница приём − покой"),
    "cohen_d": ("Cohen d", "Cohen d (приём − покой)"),
}

#: Columns of the trial table that are not part of the comparison but are worth saving.
_RENAME = {"Rest mean uV": "Rest mean", "Rest SD uV": "Rest SD",
           "Act mean uV": "Act mean", "Act SD uV": "Act SD", "Delta uV": "Delta"}


def is_jendrassik_run(run) -> bool:
    """Does this run belong in the Jendrassik tab?

    Either the pipeline filed it under ``Jendrassik``, or its name says JM: an
    H-reflex recording made with the manoeuvre is filed under H-reflex and is a
    Jendrassik run all the same (``src/hreflex.run_hreflex_jendrassik_groups`` gives
    it the same treatment).
    """
    if (run.tags or {}).get("scenario") == "Jendrassik":
        return True
    return G.scenario_in_name(run.root) == "Jendrassik"


def _channel_trials(curves: np.ndarray, values: np.ndarray, metric: str) -> list[dict]:
    return JT.channel_trials(curves, values,
                             missing_is_zero=MISSING_IS_ZERO.get(metric, True),
                             sd_floor=SD_FLOOR.get(metric, 0.0))


def trial_table(points: pd.DataFrame, metric: str = "amp_uv") -> tuple[pd.DataFrame, list[str]]:
    """Every trial of every recording and channel, on *metric*.

    Returns ``(trials, skipped)``: one row per recording × channel × trial, and the
    recording/channel pairs left out because their curve numbering has gaps. The run
    is cut BY POSITION, so one missing curve would shift the phase and read rest as
    manoeuvre — such a channel is dropped whole rather than counted inverted.

    Columns: the group keys, ``Trial``, the pentads (``Rest curves`` / ``Act curves``
    as text, ``rest_from``…``act_to`` as numbers for whoever draws them), the two
    means and SDs, ``Delta``, ``Delta %``, ``Cohen d`` (and ``Cohen d raw``), the two
    p values of the trial itself, the amplitude ``Cohen d amp`` the choice was made
    on, and ``Best``.
    """
    empty = (pd.DataFrame(), [])
    if points is None or points.empty or "curve" not in points.columns:
        return empty
    if metric not in points.columns:
        return empty
    d = points.assign(curve=pd.to_numeric(points["curve"], errors="coerce")).dropna(subset=["curve"])
    if d.empty:
        return empty
    d = d.sort_values("curve", kind="stable").drop_duplicates(KEYS + ["curve"], keep="last")

    rows, skipped = [], []
    for key, grp in d.groupby(KEYS, sort=False):
        labels = dict(zip(KEYS, key))
        curves = grp["curve"].to_numpy(int)
        if len(curves) > 1 and not (np.diff(curves) == 1).all():
            skipped.append(f"{labels['channel']} @ {labels['run'].rsplit('/', 1)[-1]}")
            continue
        trials = _channel_trials(curves, grp[metric].to_numpy(float), metric)
        if not trials:
            continue
        # The amplitude trials of the same pentads: the selection runs on them even
        # when the metric on screen is a latency, so that "best" does not move.
        if metric == SELECTION_METRIC or SELECTION_METRIC not in grp.columns:
            amp = {t["Trial"]: t for t in trials}
        else:
            amp = {t["Trial"]: t for t in
                   _channel_trials(curves, grp[SELECTION_METRIC].to_numpy(float),
                                   SELECTION_METRIC)}
        for t in trials:
            row = {**labels, **{k: v for k, v in t.items() if not k.startswith("_")}}
            rest, act = t["_rest_curves"], t["_act_curves"]
            row.update(rest_from=int(rest[0]), rest_to=int(rest[-1]),
                       act_from=int(act[0]), act_to=int(act[-1]))
            a = amp.get(t["Trial"], {})
            row["Cohen d amp"] = a.get("Cohen d", np.nan)
            row["Delta % amp"] = a.get("Delta %", np.nan)
            rows.append(row)

    if not rows:
        return empty
    trials = pd.DataFrame(rows).rename(columns=_RENAME)
    trials["Best"] = _mark_best(trials)
    return trials, skipped


def _mark_best(trials: pd.DataFrame) -> pd.Series:
    """``"best"`` on each recording × channel's best trial, by the pipeline's rule.

    ``JT.mark_best`` is handed the amplitude columns under the names it expects, so
    there is one implementation of "largest |d|, ties to the larger |gain|, ties to
    the earlier trial" rather than a second copy of it here.
    """
    src = trials[["run", "channel", "Trial", "Cohen d amp", "Delta % amp"]].rename(
        columns={"Cohen d amp": "Cohen d", "Delta % amp": "Delta %"})
    return JT.mark_best(src, by=("run", "channel"))["Best"]


def best_only(trials: pd.DataFrame) -> pd.DataFrame:
    """The one row per recording and channel that carries the strongest effect."""
    if trials is None or trials.empty or "Best" not in trials.columns:
        return pd.DataFrame()
    return trials[trials["Best"] == "best"].copy()


def spans(trials: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    """``(rest_from, rest_to, act_from, act_to)`` of every trial in *trials*."""
    if trials is None or trials.empty:
        return []
    return [(int(r.rest_from), int(r.rest_to), int(r.act_from), int(r.act_to))
            for r in trials.itertuples()]


def condition_means(ch_waves: np.ndarray,
                    spans_: list[tuple[int, int, int, int]]) -> tuple[np.ndarray | None,
                                                                      np.ndarray | None]:
    """Mean wave of the rest curves and of the manoeuvre curves, over the given trials.

    ``ch_waves`` is one channel's curves in run order (``group.curve_waveforms``), so
    curve *c* is row *c − 1*. Several trials average together: that is the "all trials"
    reading, where a recording's rest is every curve recorded without the manoeuvre.
    """
    n = len(ch_waves)

    def mean_of(pairs) -> np.ndarray | None:
        rows = sorted({c - 1 for lo, hi in pairs for c in range(lo, hi + 1) if 0 < c <= n})
        return np.asarray(ch_waves)[rows].mean(axis=0) if rows else None

    return (mean_of([(a, b) for a, b, _, _ in spans_]),
            mean_of([(a, b) for _, _, a, b in spans_]))


def effect_column(effect: str) -> str:
    """The trial-table column an effect measure reads."""
    return EFFECTS[effect][0]


def summary(trials: pd.DataFrame) -> str:
    """One line about what was computed, for the tab's status bar."""
    if trials is None or trials.empty:
        return "проб нет"
    n_best = int((trials["Best"] == "best").sum())
    per = trials.groupby(["run", "channel"])["Trial"].size()
    return (f"{len(trials)} проб · {trials['channel'].nunique()} каналов · "
            f"{n_best} лучших проб · проб на канал: "
            f"{int(per.min())}–{int(per.max())} (медиана {per.median():.0f})")

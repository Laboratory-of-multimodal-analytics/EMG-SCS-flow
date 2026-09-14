"""Left/right asymmetry of the responses on the last curves of a ramp.

A. Militskova's request (meeting of September 2026): compare the two legs channel
against channel — ch1 with ch5, ch2 with ch6, ch3 with ch7, ch4 with ch8, the same
muscle order on both sides — on the last five curves of each recording, the end
of the ramp.

No index is standard for spinally evoked responses: the stimulation papers compare
sides by thresholds, by a side factor in an ANOVA, or by eye. The indices here come
from clinical H-reflex work and from gait analysis, where every common index is a
function of the one ratio R/L; they differ only in how they average and where they
blow up.

* AI = (R − L) / (R + L), in [−1, 1] — half of Robinson's symmetry index (1987).
  0 is symmetric, +1 means only the right side responded, −1 only the left. It is
  bounded and symmetric, so it averages across recordings without the bias of a
  ratio (the mean of R/L is not the inverse of the mean of L/R).
* R/L — the ratio the clinicians asked for; R/L = (1 + AI) / (1 − AI). Clinical
  reference for the H-reflex (Jankus et al. 1994): smaller over larger side is
  0.74 ± 0.17 in healthy people and below 0.4 probably abnormal, i.e. |AI| ≈ 0.15
  and ≈ 0.43.
* Latencies are compared as a difference, R − L in ms, as clinical practice does.

Decisions taken with D. Kleeva (13.09.2026):

* a response on one side only is complete asymmetry, ±1: no response counts as
  zero amplitude. No response on either side is a gap;
* a recording's index is taken from the mean of the last curves on each side (no
  response = 0) — one value per side, as the clinical indices are. The index of
  every curve is kept too, to show how much it varies within the five;
* channels 1–4 are TAKEN AS the left side and 5–8 as the right. This is NOT
  confirmed: when it is, fix ``PAIRS`` and the notes below, nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: (left, right) channel pairs. The sides are provisional, see the module docstring.
PAIRS = (("ch1", "ch5"), ("ch2", "ch6"), ("ch3", "ch7"), ("ch4", "ch8"))
SIDES_NOTE = "ch1-4 = left, ch5-8 = right (provisional)"
SIDES_NOTE_RU = "ch1–4 — лево, ch5–8 — право (условно)"

#: How many curves at the end of the ramp the index is computed on.
LAST_N = 5

#: Which sides responded on at least one of the curves used.
_RESPONSES = {(True, True): "both", (True, False): "left only",
              (False, True): "right only", (False, False): "none"}


def pair_name(left: str, right: str) -> str:
    return f"{left}-{right}"


def recorded_channels(output_root) -> list[str] | None:
    """The EMG channels a run has: its saved epochs, less the channels excluded by hand.

    Needed because the metrics table cannot tell a silent channel from one that was
    never recorded — it leaves out both. None when the run has no saved epochs.
    """
    import json
    from pathlib import Path

    from .io_utils import STIMULATION_INDUCED_FOLDER, find_mode_dir

    root = Path(output_root)
    epochs = sorted((find_mode_dir(root, STIMULATION_INDUCED_FOLDER)
                     / "Stimulus-centered epochs").glob("*-epo.fif"))
    if not epochs:
        return None
    import mne

    names = list(mne.io.read_info(epochs[0], verbose="error")["ch_names"])
    excluded: set[str] = set()
    session = root / "review" / "session.json"
    if session.is_file():
        try:
            excluded = set(json.loads(session.read_text(encoding="utf-8")).get("exclude_channels", []))
        except (OSError, ValueError):
            pass
    return [c for c in names if c not in excluded and "art" not in c.lower()]


def size_per_curve(left, right) -> tuple[np.ndarray, np.ndarray]:
    """AI and R/L of every curve. NaN (or 0) in an input means no response there.

    One side silent: AI is ±1; R/L is 0 when the right side is silent and stays
    undefined (NaN) when the left one is.
    """
    L, R = np.abs(np.asarray(left, float)), np.abs(np.asarray(right, float))
    has_l, has_r = np.isfinite(L) & (L > 0), np.isfinite(R) & (R > 0)
    both = has_l & has_r
    ai = np.full(L.shape, np.nan)
    ratio = np.full(L.shape, np.nan)
    ai[both] = (R[both] - L[both]) / (R[both] + L[both])
    ratio[both] = R[both] / L[both]
    ai[has_r & ~has_l] = 1.0
    ai[has_l & ~has_r] = -1.0
    ratio[has_l & ~has_r] = 0.0
    return ai, ratio


def size_summary(left, right) -> dict:
    """One recording's index for a size metric (amplitude, area) over its last curves."""
    L = np.nan_to_num(np.abs(np.asarray(left, float)), nan=0.0)
    R = np.nan_to_num(np.abs(np.asarray(right, float)), nan=0.0)
    n_l, n_r = int((L > 0).sum()), int((R > 0).sum())
    out = {"n_left": n_l, "n_right": n_r,
           "left_mean": np.nan, "right_mean": np.nan, "ai": np.nan, "ratio": np.nan,
           "ai_sd_curves": np.nan, "responses": _RESPONSES[(n_l > 0, n_r > 0)]}
    if n_l + n_r == 0:
        return out
    out["left_mean"], out["right_mean"] = float(L.mean()), float(R.mean())
    out["ai"] = float((R.sum() - L.sum()) / (R.sum() + L.sum()))
    if L.sum() > 0:
        out["ratio"] = float(R.sum() / L.sum())
    ai_curves, _ = size_per_curve(left, right)
    ok = ai_curves[np.isfinite(ai_curves)]
    if ok.size > 1:
        out["ai_sd_curves"] = float(ok.std(ddof=1))
    return out


def latency_per_curve(left, right) -> np.ndarray:
    """R − L of every curve that has a response on both sides; NaN elsewhere."""
    L, R = np.asarray(left, float), np.asarray(right, float)
    return np.where(np.isfinite(L) & np.isfinite(R), R - L, np.nan)


def latency_summary(left, right) -> dict:
    """One recording's side difference for a latency, on the curves where both sides respond."""
    L, R = np.asarray(left, float), np.asarray(right, float)
    d = latency_per_curve(L, R)
    ok = d[np.isfinite(d)]
    n_l, n_r = int(np.isfinite(L).sum()), int(np.isfinite(R).sum())
    return {"n_left": n_l, "n_right": n_r, "n_both": int(ok.size),
            "left_mean": float(np.nanmean(L)) if n_l else np.nan,
            "right_mean": float(np.nanmean(R)) if n_r else np.nan,
            "diff_ms": float(ok.mean()) if ok.size else np.nan,
            "diff_sd": float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
            "responses": _RESPONSES[(n_l > 0, n_r > 0)]}


def recording_asymmetry(frame: pd.DataFrame, metrics: dict[str, str], n: int = LAST_N,
                        pairs=PAIRS, recorded=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asymmetry of one recording on its last *n* curves.

    ``frame`` holds one recording, one row per (curve, channel), with ``curve``,
    ``channel`` and the metric columns; a curve without a response has NaN there.
    ``metrics`` maps a column to ``"size"`` or ``"latency"``.

    ``recorded`` is the channels the recording has (its epochs). The metrics table
    leaves out a channel on which detection found no response template at all, so
    a recorded channel missing from ``frame`` is a channel with no response — the
    complete asymmetry this reports — not a gap. A pair is skipped only when one of
    its channels was not recorded. Without ``recorded``, the table's channels are
    all there is.

    Returns ``(by_curve, summary)``: every curve's values and index, and one row
    per pair and metric (the fields of ``size_summary`` / ``latency_summary``).
    """
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    curves = sorted(frame["curve"].dropna().astype(int).unique())[-n:]
    channels = set(frame["channel"].astype(str))
    if recorded is not None:
        channels |= {str(c) for c in recorded}
    rows_curve, rows_sum = [], []
    for left, right in pairs:
        if left not in channels or right not in channels:
            continue
        both = frame[frame["channel"].astype(str).isin([left, right])]
        for col, kind in metrics.items():
            if col not in both.columns:
                continue
            wide = (both.assign(curve=both["curve"].astype(int), channel=both["channel"].astype(str))
                    .drop_duplicates(["curve", "channel"], keep="last")
                    .set_index(["curve", "channel"])[col].unstack()
                    .reindex(index=curves, columns=[left, right]))
            L = pd.to_numeric(wide[left], errors="coerce").to_numpy(float)
            R = pd.to_numeric(wide[right], errors="coerce").to_numpy(float)
            base = {"pair": pair_name(left, right), "left": left, "right": right, "metric": col}
            if kind == "latency":
                diff = latency_per_curve(L, R)
                summary = latency_summary(L, R)
                for c, lv, rv, dv in zip(curves, L, R, diff):
                    rows_curve.append(base | {"curve": c, "left_value": lv, "right_value": rv,
                                              "ai": np.nan, "ratio": np.nan, "diff_ms": dv})
            else:
                ai, ratio = size_per_curve(L, R)
                summary = size_summary(L, R)
                for c, lv, rv, a, q in zip(curves, L, R, ai, ratio):
                    rows_curve.append(base | {"curve": c, "left_value": lv, "right_value": rv,
                                              "ai": a, "ratio": q, "diff_ms": np.nan})
            rows_sum.append(base | {"kind": kind, "curves": f"{curves[0]}-{curves[-1]}"} | summary)
    return pd.DataFrame(rows_curve), pd.DataFrame(rows_sum)

"""Per-curve corrections for the recruitment and Jendrassik scenarios.

Detection on these files is per curve, so it fails per curve too: a spurious
response creeps into one sweep, or a real one is missed on another while its
neighbours are found. Neither is worth re-running a file for, and neither should
be fixed by moving a global threshold — that trades one file's error for
another's.

So a curve can be marked directly:

  false   — the detection on this curve is not a response. Its metrics are
            cleared, and it becomes a gap in the recruitment curve.
  missed  — there is a response here that detection did not take. It is measured
            at the latencies the channel's own detected curves agree on, with no
            gates applied: the clinician has already made the call that it is
            there, and the gates are what got it wrong.

Marks live in ``review/sir_curve_overrides.json`` beside the results, so a re-run
from the CLI honours what was marked in the GUI. The recompute reads the saved
epochs and rewrites only the affected rows, then rebuilds the scenario
deliverable from the updated table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .detection import pick_epoch_value_near_latency
from .io_utils import STIMULATION_INDUCED_FOLDER, find_mode_dir

OVERRIDES_NAME = "sir_curve_overrides.json"

FALSE = "false"
MISSED = "missed"
#: Per-channel entry, alongside the per-curve marks:
#:   {"ch4": {"false": [12], "window_ms": [18, 30],
#:            "markers_ms": {"onset": 19, "p1": 21, "p2": 26}}}
WINDOW = "window_ms"
MARKERS = "markers_ms"


def overrides_path(output_root: Path) -> Path:
    return Path(output_root) / "review" / OVERRIDES_NAME


def load_overrides(output_root: Path) -> dict:
    p = overrides_path(output_root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_overrides(output_root: Path, data: dict) -> Path:
    p = overrides_path(output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def mark_curve(overrides: dict, channel: str, curve: int, kind: str | None) -> dict:
    """Set, change or clear the mark on one curve. Returns the updated dict."""
    ch = overrides.setdefault(channel, {})
    for k in (FALSE, MISSED):
        lst = ch.get(k, [])
        if curve in lst:
            lst.remove(curve)
        if lst:
            ch[k] = sorted(lst)
        else:
            ch.pop(k, None)
    if kind in (FALSE, MISSED):
        ch.setdefault(kind, [])
        ch[kind] = sorted(set(ch[kind]) | {int(curve)})
    if not ch:
        overrides.pop(channel, None)
    return overrides


def curve_mark(overrides: dict, channel: str, curve: int) -> str | None:
    ch = overrides.get(channel, {})
    if curve in ch.get(FALSE, []):
        return FALSE
    if curve in ch.get(MISSED, []):
        return MISSED
    return None


# --------------------------------------------------------------------------- #
def _metrics_csv(output_root: Path) -> Path:
    return (find_mode_dir(output_root, STIMULATION_INDUCED_FOLDER)
            / "Excel" / "Large_dataset_emg_response_metrics.csv")


def _load_epochs(output_root: Path):
    import mne

    base = find_mode_dir(output_root, STIMULATION_INDUCED_FOLDER) / "Stimulus-centered epochs"
    files = sorted(base.glob("*-epo.fif"))
    if not files:
        raise FileNotFoundError("No saved epochs — run the file once first.")
    ep = mne.read_epochs(files[0], preload=True, verbose=False)
    return ep


def _channel_reference(df: pd.DataFrame, channel: str, marks: dict | None = None) -> dict | None:
    """Where to measure this channel's responses.

    Markers placed by hand win outright — disagreeing with the automatic
    placement is the only reason to draw them. Otherwise the latencies come from
    what this channel's own accepted curves agree on, by median rather than mean
    so one stray detection cannot drag them.
    """
    marks = marks or {}
    mk = marks.get(MARKERS)
    if mk:
        p1 = float(mk["p1"]) / 1e3
        p2 = float(mk["p2"]) / 1e3 if mk.get("p2") is not None and np.isfinite(mk["p2"]) else np.nan
        onset = float(mk["onset"]) / 1e3 if mk.get("onset") is not None else np.nan
        d = df[(df["Channel"].astype(str) == channel) & df["Peak1 latency"].notna()]
        pol1 = 1 if (d.empty or float(d["Peak1 value"].median()) >= 0) else -1
        return {"p1": p1, "p2": p2, "onset": onset, "pol1": pol1, "pol2": -pol1,
                "source": "маркеры вручную"}
    d = df[(df["Channel"].astype(str) == channel) & df["Peak1 latency"].notna()]
    if d.empty:
        return None
    p1 = float(d["Peak1 latency"].median())
    p1v = float(d["Peak1 value"].median())
    p2 = float(d["Peak2 latency"].median()) if d["Peak2 latency"].notna().any() else np.nan
    p2v = float(d["Peak2 value"].median()) if d["Peak2 value"].notna().any() else np.nan
    onset = float(d["Onset latency"].median()) if d["Onset latency"].notna().any() else np.nan
    return {"p1": p1, "p2": p2, "onset": onset,
            "pol1": 1 if p1v >= 0 else -1, "pol2": 1 if p2v >= 0 else -1,
            "source": "медиана принятых кривых"}


def _measure_at(df, row, sig, times, sfreq, ref) -> None:
    """Sample one curve at the reference latencies.

    Hand markers say WHERE to look, so the correlation/consistency gates (which
    is what the clinician is overruling) are skipped — but the amplitude floor
    still holds: a response below STIM_PTP_MIN_UV / STIM_P1_ABS_MIN_UV is noise
    wherever you point, and must not count as a detection.
    """
    from .constants import STIM_P1_ABS_MIN_UV, STIM_PTP_MIN_UV

    def _clear():
        for col in ("Onset latency", "Peak1 latency", "Peak1 value",
                    "Peak2 latency", "Peak2 value", "PTP amplitude"):
            df.at[row, col] = np.nan

    p1l, p1v = pick_epoch_value_near_latency(
        sig, times, ref["p1"], sfreq, win_ms=2.0, polarity=ref["pol1"])
    p2l, p2v = (np.nan, np.nan)
    if np.isfinite(ref["p2"]):
        p2l, p2v = pick_epoch_value_near_latency(
            sig, times, ref["p2"], sfreq, win_ms=2.0, polarity=ref["pol2"])
    # A detection is BOTH peaks AND above the noise floor: a lone peak, or a
    # PTP / |P1| under threshold, is not a response — leave the whole row empty.
    ptp = abs(p1v - p2v) if (np.isfinite(p1v) and np.isfinite(p2v)) else np.nan
    if (not np.isfinite(ptp)
            or ptp < STIM_PTP_MIN_UV * 1e-6
            or abs(p1v) < STIM_P1_ABS_MIN_UV * 1e-6):
        _clear()
        return
    df.at[row, "Onset latency"] = ref["onset"]
    df.at[row, "Peak1 latency"] = p1l
    df.at[row, "Peak1 value"] = p1v
    df.at[row, "Peak2 latency"] = p2l
    df.at[row, "Peak2 value"] = p2v
    df.at[row, "PTP amplitude"] = ptp


def recompute_corrections(output_root: Path, scenario: str | None = None) -> list[dict]:
    """Apply every per-curve mark, then rebuild the scenario deliverable.

    Returns one record per corrected curve.
    """
    from .jendrassik import run_curve_group_analysis
    from .neurosoft import JENDRASSIK, PAIRED, RECRUITMENT
    from .recruitment import run_recruitment_analysis

    output_root = Path(output_root)
    ov = load_overrides(output_root)
    if not ov:
        return []

    csv = _metrics_csv(output_root)
    df = pd.read_csv(csv)
    ep = _load_epochs(output_root)
    times = ep.times
    sfreq = float(ep.info["sfreq"])
    data = ep.get_data()
    ch_index = {c: i for i, c in enumerate(ep.ch_names)}

    done: list[dict] = []
    for channel, marks in ov.items():
        ref = _channel_reference(df, channel, marks)
        # Hand-placed markers re-measure the WHOLE channel, not just the curves
        # singled out: the point of moving them is that the automatic placement
        # was wrong everywhere on this channel. This must include curves the
        # automatic pass left empty — otherwise a channel the auto-detector
        # wiped (e.g. a weak channel dropped by the both-peaks rule) can never be
        # recovered by hand, which is exactly when the clinician draws markers.
        if marks.get(MARKERS) and ref is not None and channel in ch_index:
            # The polarity of a hand-placed marker is whatever the response does
            # AT that latency — the whole reason to move a marker is often that
            # the automatic pick sat on the opposite-polarity deflection. Read it
            # off this channel's own mean at the marker, not the auto-detected
            # values (which point exactly where the clinician disagreed).
            mean_wave = data[:, ch_index[channel], :].mean(axis=0)
            base = float(np.nanmean(mean_wave[times < 0.002])) if np.any(times < 0.002) else 0.0

            def _pol_at(t):
                i = int(np.argmin(np.abs(times - t)))
                return 1 if (mean_wave[i] - base) >= 0 else -1

            ref = dict(ref)
            ref["pol1"] = _pol_at(ref["p1"])
            if np.isfinite(ref["p2"]):
                ref["pol2"] = _pol_at(ref["p2"])
            rows = df.index[df["Channel"].astype(str) == channel]
            for r in rows:
                curve = int(df.at[r, "Epoch"]) + 1
                if curve in marks.get(FALSE, []):
                    continue
                _measure_at(df, r, data[curve - 1, ch_index[channel], :], times, sfreq, ref)
            done.append({"channel": channel, "curve": len(rows),
                         "kind": "перемерено по ручным маркерам (кривых)"})
        for curve in marks.get(FALSE, []):
            m = (df["Channel"].astype(str) == channel) & (df["Epoch"] == int(curve) - 1)
            if not m.any():
                continue
            df.loc[m, ["Onset latency", "Peak1 latency", "Peak2 latency",
                       "Peak1 value", "Peak2 value", "PTP amplitude"]] = np.nan
            done.append({"channel": channel, "curve": int(curve), "kind": "ложный — снят"})
        for curve in marks.get(MISSED, []):
            m = (df["Channel"].astype(str) == channel) & (df["Epoch"] == int(curve) - 1)
            if not m.any() or ref is None or channel not in ch_index:
                done.append({"channel": channel, "curve": int(curve),
                             "kind": "пропущенный — не удалось (нет опорных латентностей)"})
                continue
            r = df.index[m][0]
            _measure_at(df, r, data[int(curve) - 1, ch_index[channel], :], times, sfreq, ref)
            p1l = df.at[r, "Peak1 latency"]
            done.append({"channel": channel, "curve": int(curve),
                         "kind": f"пропущенный — досчитан, P1 {p1l * 1e3:.1f} мс"
                                 if np.isfinite(p1l) else "пропущенный — пик не найден"})

    df.to_csv(csv, index=False)

    if scenario in (JENDRASSIK, PAIRED):
        # n_groups=None: Jendrassik re-derives its count from the file name,
        # paired reads it off the data. Same rule as a full run.
        run_curve_group_analysis(output_root, scenario)
    elif scenario == RECRUITMENT or scenario is None:
        run_recruitment_analysis(output_root)
    return done

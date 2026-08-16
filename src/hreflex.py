"""The H-reflex scenario: two responses per curve, measured separately.

A peripheral-nerve run puts TWO responses on the same curve. The direct motor
response (**M**) comes back along the motor axons and appears first; the
monosynaptic reflex (**H**) travels up the Ia afferents, through the cord and
back down, and lands 25-30 ms later. They are recorded from one muscle by one
electrode pair, so they look alike — same polarity, same rough shape — and the
only thing separating them on the curve is latency.

Why this cannot be a recruitment run with a wider window
--------------------------------------------------------
The two do not grow together. As stimulation rises the M-wave grows
monotonically to a plateau while the H-reflex rises, peaks and then **recedes**,
because the antidromic volley in the motor axons collides with the reflex volley
coming back down. On the file this module was built against ((Я37) soleus left,
12.05.2020) the H peaks at 834 µV around curve 31 and is down to 123 µV by curve
55, while the M is still climbing past 1100 µV.

A single-response detector on such a curve reports whichever of the two is
larger at that intensity — the H on the middle curves, the M on the last ones —
so one column ends up holding two different responses and the amplitude "curve"
it draws is an artefact of the crossover. That is not a hypothetical: 23 of the
75 H-reflex files in this dataset carry exactly that complaint in the processing
journal ("отбирался второй ответ после M-ответа").

So both are measured, each with its own onset, P1, P2 and PTP, and each is
allowed to be absent independently: sub-threshold curves have neither, low
intensities have H without a measurable M, and high intensities routinely have M
without an H. Nothing here treats a missing component as a failed detection.

How the two are found
---------------------
Per channel, once (``fit_reference``):

1. **M** is read off the mean of the channel's strongest curves. It is the
   response that is largest exactly where the M-wave dominates, and taking the
   strong end is what the rest of the Neurosoft path already does.
2. **H** is read off the mean of the curves with the most late activity — NOT
   the same set. Fitting the reflex on the strongest curves is fitting it where
   it has already been suppressed; on the file above those are curves 42-55,
   where the H is at a fifth of its maximum.
3. Within the H window P1 is taken **at the polarity M-P1 has**, not as the
   dominant deflection. M and H are the same muscle's compound action potential
   and point the same way; picking the dominant deflection independently makes
   the two components' P1 mean different things on the same channel whenever the
   reflex's rebound outgrows its first phase, which it does here (H-P1 +619 µV,
   H-P2 -701 µV).

Then per curve (``measure_curves``), each component is searched near its own
reference latencies, inside its own half of the response window — the same
marker-driven peak search the ordinary SIR pass uses, run twice.

Hand-placed markers (six of them: onset/P1/P2 for M and for H) override the
whole of step 1-3 for a channel, and are what the scenario view writes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    HREFLEX_H_WIN_MS, HREFLEX_M_GUARD_MS, HREFLEX_M_WIN_MS, HREFLEX_MIN_GAP_MS,
    STIM_ONSET_K, STIM_P1_ABS_MIN_UV, STIM_PEAK_AMP_MIN_UV, STIM_PTP_MIN_UV,
    STIM_SHAPE_MIN_CORR, STIM_SHAPE_MIN_SNR, STIM_SHAPE_REL_AMP_MAX,
    SWEEP_STRONG_FRAC, SWEEP_STRONG_MIN,
)
from .detection import (
    detect_onset_before_peak, detect_peak_in_window, noise_std_from_tail,
    onsets_anchored_to_channel, pick_epoch_value_near_latency,
    small_misshapen_responses,
)
from .io_utils import STIMULATION_INDUCED_FOLDER, ensure_dir, find_mode_dir

#: The two components, in the order they appear on the curve.
COMPONENTS = ("M", "H")

#: Metric column added to the metrics CSV per component. Same units as the
#: single-response columns beside them: latencies in seconds, values in volts.
COMPONENT_COLUMNS = {
    "onset": "{c} onset latency",
    "p1": "{c} peak1 latency",
    "p2": "{c} peak2 latency",
    "pv1": "{c} peak1 value",
    "pv2": "{c} peak2 value",
    "ptp": "{c} PTP amplitude",
}

#: The legacy single-response columns, which stay in place and mirror the M-wave.
LEGACY_COLUMNS = {
    "onset": "Onset latency",
    "p1": "Peak1 latency",
    "p2": "Peak2 latency",
    "pv1": "Peak1 value",
    "pv2": "Peak2 value",
    "ptp": "PTP amplitude",
}

#: Marker keys a hand-drawn H-reflex template carries — six, not three.
MARKER_KEYS = ("m_onset", "m_p1", "m_p2", "h_onset", "h_p1", "h_p2")

#: A component must clear the gates on at least this many curves to be reported
#: on a channel at all. One or two accepted curves out of fifty is what a noise
#: bump that happened to sit at the right latency produces, and reporting it as
#: "the H-reflex of this channel" is worse than reporting nothing.
MIN_ACCEPTED_CURVES = 3


def columns_for(component: str) -> dict[str, str]:
    return {k: v.format(c=component) for k, v in COMPONENT_COLUMNS.items()}


def all_component_columns() -> list[str]:
    return [c for comp in COMPONENTS for c in columns_for(comp).values()]


# --------------------------------------------------------------------------- #
# Reference fitting (per channel)
# --------------------------------------------------------------------------- #
def _extreme(sig, times, lo, hi, polarity=None) -> tuple[float, float]:
    """Latency and value of the biggest deflection of *polarity* in [lo, hi]."""
    m = (times >= lo) & (times <= hi)
    if int(np.sum(m)) < 3:
        return (np.nan, np.nan)
    seg, seg_t = sig[m], times[m]
    if polarity is None:
        i = int(np.argmax(np.abs(seg)))
    elif polarity > 0:
        i = int(np.argmax(seg))
    else:
        i = int(np.argmin(seg))
    return float(seg_t[i]), float(seg[i])


def _strongest_mean(curves: np.ndarray, times: np.ndarray,
                    lo: float, hi: float) -> np.ndarray:
    """Mean of the curves carrying the most signal between *lo* and *hi*.

    The window is what makes this useful twice over: taken over the whole
    response window it finds where the M-wave is biggest, taken over the late
    stretch it finds where the reflex is — and those are different curves.
    """
    m = (times >= lo) & (times <= hi)
    n = curves.shape[0]
    if not m.any() or n == 0:
        return curves.mean(axis=0) if n else np.zeros_like(times)
    if n <= SWEEP_STRONG_MIN:
        return curves.mean(axis=0)
    strength = np.ptp(curves[:, m], axis=1)
    keep = int(max(SWEEP_STRONG_MIN, round(n * float(SWEEP_STRONG_FRAC))))
    idx = np.sort(np.argsort(strength)[-keep:])
    return curves[idx].mean(axis=0)


def fit_reference(
    curves: np.ndarray,
    times: np.ndarray,
    resp_tmin: float,
    resp_tmax: float,
) -> dict | None:
    """Where this channel's M and H responses sit, from its own curves.

    ``curves`` is (n_curves, n_samples) in volts, baseline-corrected. Returns a
    dict with the two components' reference latencies, the polarity of each
    marker and the boundary between their search windows, or None when not even
    an M-wave can be found.
    """
    if curves.ndim != 2 or curves.shape[0] == 0:
        return None

    # ── M: the earlier half of the response window, on the strongest curves ──
    m_split = resp_tmin + 0.5 * (resp_tmax - resp_tmin)
    w_m = _strongest_mean(curves, times, resp_tmin, resp_tmax)
    m_p1_t, m_p1_v = _extreme(w_m, times, resp_tmin, m_split)
    if not np.isfinite(m_p1_t) or abs(m_p1_v) < STIM_P1_ABS_MIN_UV * 1e-6:
        return None
    pol = 1 if m_p1_v >= 0 else -1
    m_p2_t, _ = _extreme(w_m, times, m_p1_t + 5e-4, m_split, -pol)
    if not np.isfinite(m_p2_t):
        m_p2_t = m_p1_t

    # ── H: the late stretch, on the curves where the REFLEX is biggest ──
    h_lo = max(m_p2_t + HREFLEX_M_GUARD_MS / 1e3,
               m_p1_t + HREFLEX_MIN_GAP_MS / 1e3)
    ref = {"m_p1": m_p1_t, "m_p2": m_p2_t, "polarity": pol,
           "h_p1": np.nan, "h_p2": np.nan,
           "split": min(h_lo, resp_tmax), "source": "кривые канала"}
    if h_lo >= resp_tmax:
        return ref                       # no room left for a reflex

    w_h = _strongest_mean(curves, times, h_lo, resp_tmax)
    h_p1_t, h_p1_v = _extreme(w_h, times, h_lo, resp_tmax, pol)
    if not np.isfinite(h_p1_t) or abs(h_p1_v) < STIM_P1_ABS_MIN_UV * 1e-6:
        return ref
    h_p2_t, _ = _extreme(w_h, times, h_p1_t + 5e-4, resp_tmax, -pol)
    ref["h_p1"] = h_p1_t
    ref["h_p2"] = h_p2_t if np.isfinite(h_p2_t) else h_p1_t
    # Split the response window between the two components halfway between the
    # M's rebound and the H's first phase, so neither search can reach into the
    # other's territory however much a single curve's latency wanders.
    ref["split"] = float(np.clip(0.5 * (m_p2_t + h_p1_t), h_lo, resp_tmax))
    return ref


def reference_from_markers(markers_ms: dict) -> dict:
    """A reference built from the six hand-placed markers instead of the data."""
    def _get(key):
        v = markers_ms.get(key)
        return float(v) / 1e3 if v is not None and np.isfinite(v) else np.nan

    m_p1, m_p2 = _get("m_p1"), _get("m_p2")
    h_p1, h_p2 = _get("h_p1"), _get("h_p2")
    split = (0.5 * (m_p2 + h_p1) if np.isfinite(m_p2) and np.isfinite(h_p1)
             else (h_p1 if np.isfinite(h_p1) else np.inf))
    return {"m_p1": m_p1, "m_p2": m_p2, "h_p1": h_p1, "h_p2": h_p2,
            "m_onset": _get("m_onset"), "h_onset": _get("h_onset"),
            "polarity": None, "split": split, "source": "маркеры вручную"}


# --------------------------------------------------------------------------- #
# Per-curve measurement
# --------------------------------------------------------------------------- #
def _measure_component(
    sig: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    p1_c: float,
    p2_c: float,
    lo: float,
    hi: float,
    polarity: int,
    win_ms: float,
    noise_after_s: float,
) -> dict:
    """One component on one curve: onset, P1, P2, PTP.

    Marker-driven, like the ordinary SIR pass: the reference says where to look
    and with which polarity, the curve says exactly where the peak is. Bounded to
    ``[lo, hi]`` so a component can never be measured inside the other's window.
    """
    blank = {"onset": np.nan, "p1": np.nan, "p2": np.nan,
             "pv1": np.nan, "pv2": np.nan, "ptp": np.nan}
    if not np.isfinite(p1_c) or hi <= lo:
        return blank

    p1_l, p1_v = detect_peak_in_window(
        sig_f=sig, times=times, sfreq=sfreq, baseline_mask=baseline_mask,
        t_center=p1_c, win_ms=win_ms, polarity=polarity,
        amp_min_uV=STIM_PEAK_AMP_MIN_UV, min_width_ms=0.4, choose="global",
        t_min=lo, t_max=hi,
    )
    if not np.isfinite(p1_l):
        return blank
    p1_l, p1_v = pick_epoch_value_near_latency(
        sig, times, p1_l, sfreq, win_ms=1.0, polarity=polarity)

    p2_l, p2_v = detect_peak_in_window(
        sig_f=sig, times=times, sfreq=sfreq, baseline_mask=baseline_mask,
        t_center=p2_c if np.isfinite(p2_c) else p1_l, win_ms=win_ms + 2.0,
        polarity=-polarity, amp_min_uV=STIM_PEAK_AMP_MIN_UV, min_width_ms=0.4,
        choose="global", t_min=p1_l, t_max=hi,
    )
    if np.isfinite(p2_l):
        p2_l, p2_v = pick_epoch_value_near_latency(
            sig, times, p2_l, sfreq, win_ms=1.0, polarity=-polarity)

    # A response is BOTH phases, as everywhere else on the Neurosoft path: a lone
    # peak carries no PTP and is as often a noise bump as a monophasic response.
    if not np.isfinite(p2_v) or p2_l <= p1_l:
        return blank
    ptp = float(abs(p1_v - p2_v))
    if ptp < STIM_PTP_MIN_UV * 1e-6 or abs(p1_v) < STIM_P1_ABS_MIN_UV * 1e-6:
        return blank

    onset = detect_onset_before_peak(
        sig, times, sfreq, float(np.mean(sig[baseline_mask])),
        noise_std_from_tail(sig, times, noise_after_s), p1_l,
        t_min=lo, k=STIM_ONSET_K, sustain_ms=2.0,
    )
    return {"onset": onset, "p1": p1_l, "p2": p2_l,
            "pv1": p1_v, "pv2": p2_v, "ptp": ptp}


def measure_curves(
    curves: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    ref: dict,
    resp_tmin: float,
    resp_tmax: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Measure M and H on every curve of one channel.

    Returns ``{"M": {metric: array}, "H": {...}}``, one value per curve, NaN
    where that component was not found.
    """
    n = curves.shape[0]
    out = {c: {k: np.full(n, np.nan) for k in
               ("onset", "p1", "p2", "pv1", "pv2", "ptp")} for c in COMPONENTS}
    if ref is None:
        return out

    pol_of = _polarities(ref, curves, times, resp_tmin)
    spec = _component_windows(ref, resp_tmin, resp_tmax)
    # A channel whose markers were placed by hand keeps its components even when
    # only a curve or two clears the floors: the clinician has already said the
    # response is there, and refusing it for being rare is the automatic
    # judgement she drew the markers to overrule.
    min_curves = 1 if ref.get("source") == "маркеры вручную" else MIN_ACCEPTED_CURVES
    for comp, (p1_c, p2_c, lo, hi, win) in spec.items():
        if not np.isfinite(p1_c if p1_c is not None else np.nan):
            continue
        for j in range(n):
            r = _measure_component(
                curves[j], times, sfreq, baseline_mask, float(p1_c),
                float(p2_c) if p2_c is not None else np.nan,
                lo, hi, pol_of[comp], win, resp_tmax,
            )
            for k, v in r.items():
                out[comp][k][j] = v
        _apply_channel_gates(out[comp], curves, times, sfreq, baseline_mask,
                             resp_tmax, min_curves=min_curves)
    return out


def _apply_channel_gates(res: dict[str, np.ndarray], curves, times, sfreq,
                         baseline_mask, resp_tmax,
                         min_curves: int = MIN_ACCEPTED_CURVES) -> None:
    """Drop what does not look like this component's own response, in place.

    Two gates, both already used by the single-response path and both applied
    per component here rather than per channel — the point of the scenario is
    that a channel can carry a clean M and no H, or the reverse.
    """
    amps = np.where(np.isfinite(res["ptp"]), res["ptp"], np.abs(res["pv1"]))
    drop = small_misshapen_responses(
        curves, times, res["p1"], amps,
        rel_amp_max=STIM_SHAPE_REL_AMP_MAX, min_corr=STIM_SHAPE_MIN_CORR,
        min_snr=STIM_SHAPE_MIN_SNR, noise_after_s=resp_tmax,
    )
    for k in res:
        res[k][drop] = np.nan

    if int(np.sum(np.isfinite(res["p1"]))) < min_curves:
        for k in res:
            res[k][:] = np.nan
        return

    # One latency per channel rather than one per curve's own noise floor — the
    # same correction the recruitment path makes, for the same reason.
    res["onset"] = onsets_anchored_to_channel(
        curves, times, sfreq, baseline_mask, res["onset"], res["p1"],
        np.where(np.isfinite(res["ptp"]), res["ptp"], np.abs(res["pv1"])),
        k=STIM_ONSET_K, noise_after_s=resp_tmax,
    )
    res["onset"][~np.isfinite(res["p1"])] = np.nan


def _polarities(ref: dict, curves, times, resp_tmin) -> dict:
    """Which way each component's P1 points.

    The automatic fit has already decided this and deliberately gives the reflex
    the M-wave's polarity: the two are one muscle's compound action potential and
    point the same way, and letting the reflex pick independently makes P1 mean
    different things on one channel (see the module docstring).

    Hand-placed markers are read differently — per component, off what the
    channel's own mean does AT each marker. A marker is the clinician saying
    where the response is; if she puts H-P1 on a trough while M-P1 is a peak,
    inheriting the M-wave's polarity would search upward there and walk the
    marker straight back to the placement she disagreed with.
    """
    pol = ref.get("polarity")
    if pol is not None:
        return {c: pol for c in COMPONENTS}

    mean_wave = curves.mean(axis=0)
    post = times > resp_tmin
    base = float(np.nanmedian(mean_wave[post])) if np.any(post) else 0.0

    def _at(t_val, fallback=1):
        if t_val is None or not np.isfinite(t_val):
            return fallback
        i = int(np.argmin(np.abs(times - float(t_val))))
        return 1 if (mean_wave[i] - base) >= 0 else -1

    out = {"M": _at(ref.get("m_p1"))}
    out["H"] = _at(ref.get("h_p1"), out["M"])
    # Written back so a correction applied afterwards measures with the same
    # polarities this pass used, instead of resolving them again.
    ref["polarity"] = out["M"]
    ref["polarity_h"] = out["H"]
    return out


def _component_windows(ref: dict, resp_tmin: float, resp_tmax: float) -> dict:
    split = float(np.clip(ref.get("split", resp_tmax), resp_tmin, resp_tmax))
    return {
        "M": (ref.get("m_p1"), ref.get("m_p2"), resp_tmin, split, HREFLEX_M_WIN_MS),
        "H": (ref.get("h_p1"), ref.get("h_p2"), split, resp_tmax, HREFLEX_H_WIN_MS),
    }


def _apply_marks(res, marks, curves, times, sfreq, baseline_mask, ref,
                 resp_tmin, resp_tmax) -> None:
    """Apply this channel's per-component ``false`` / ``missed`` marks, in place.

    ``false`` clears a component on one curve; ``missed`` measures it there with
    the shape and per-channel gates skipped — the clinician has already decided
    the response is present, and the gates are what got that wrong. The
    amplitude floors inside ``_measure_component`` still hold: a peak under them
    is noise wherever it is pointed at, and calling it a detection would put a
    number in the table that the figure cannot support.
    """
    if ref is None or not marks:
        return
    pol_of = _polarities(ref, curves, times, resp_tmin)
    spec = _component_windows(ref, resp_tmin, resp_tmax)
    for comp in COMPONENTS:
        for curve in marks.get(f"false_{comp}", []):
            j = int(curve) - 1
            if 0 <= j < curves.shape[0]:
                for k in res[comp]:
                    res[comp][k][j] = np.nan
        p1_c, p2_c, lo, hi, win = spec[comp]
        if not np.isfinite(p1_c if p1_c is not None else np.nan):
            continue
        for curve in marks.get(f"missed_{comp}", []):
            j = int(curve) - 1
            if not (0 <= j < curves.shape[0]):
                continue
            r = _measure_component(
                curves[j], times, sfreq, baseline_mask, float(p1_c),
                float(p2_c) if p2_c is not None else np.nan,
                lo, hi, pol_of[comp], win, resp_tmax,
            )
            for k, v in r.items():
                res[comp][k][j] = v


# --------------------------------------------------------------------------- #
# Running it over a finished SIR run
# --------------------------------------------------------------------------- #
def _sir_dir(output_root: Path) -> Path:
    return find_mode_dir(Path(output_root), STIMULATION_INDUCED_FOLDER)


def metrics_csv(output_root: Path) -> Path:
    return _sir_dir(output_root) / "Excel" / "Large_dataset_emg_response_metrics.csv"


def load_epochs(output_root: Path):
    import mne

    base = _sir_dir(output_root) / "Stimulus-centered epochs"
    files = sorted(base.glob("*-epo.fif"))
    if not files:
        raise FileNotFoundError("No saved epochs — run the file once first.")
    return mne.read_epochs(files[0], preload=True, verbose="ERROR")


def measure_run(
    output_root: Path,
    resp_tmin: float,
    resp_tmax: float,
    baseline_tmin: float,
    baseline_tmax: float,
    overrides: dict | None = None,
) -> pd.DataFrame:
    """Re-measure a finished run as M + H and rewrite the metrics CSV.

    Runs off the saved epochs, so it needs no access to the original export and
    can be repeated after hand corrections without re-detecting anything.
    """
    output_root = Path(output_root)
    csv = metrics_csv(output_root)
    df = pd.read_csv(csv)
    ep = load_epochs(output_root)
    times = np.asarray(ep.times)
    sfreq = float(ep.info["sfreq"])
    data = ep.get_data()
    baseline_mask = (times >= baseline_tmin) & (times <= baseline_tmax)
    overrides = overrides or {}

    for col in all_component_columns():
        if col not in df.columns:
            df[col] = np.nan

    n_curves = data.shape[0]
    report: list[str] = []
    for ch_idx, ch in enumerate(ep.ch_names):
        rows = df.index[df["Channel"].astype(str) == str(ch)]
        if not len(rows):
            continue
        curves = data[:, ch_idx, :]
        curves = curves - curves[:, baseline_mask].mean(axis=1, keepdims=True)

        marks = (overrides.get(str(ch)) or {}).get("markers_ms") or {}
        if any(k in marks for k in MARKER_KEYS):
            ref = reference_from_markers(marks)
        else:
            if marks:
                # A three-marker set left by an earlier run of this file under a
                # single-response scenario. It says where ONE response is, and
                # which of the two it meant is not recorded — so it is reported
                # rather than guessed at, and the file is left untouched.
                print(
                    f"[HREFLEX] {ch}: игнорирую старые маркеры на одну волну "
                    f"({', '.join(f'{k} {v}' for k, v in marks.items())} мс) — "
                    "для этого сценария нужны шесть. Проверьте автоматическую "
                    "подгонку и при необходимости расставьте маркеры заново.",
                    flush=True,
                )
            ref = fit_reference(curves, times, resp_tmin, resp_tmax)
        ch_marks = overrides.get(str(ch)) or {}
        legacy = {k: ch_marks.get(k) for k in ("false", "missed") if ch_marks.get(k)}
        if legacy:
            # Same problem as the three-marker set: the mark named "the"
            # detection on a curve, and which of the two responses that was is
            # not in the file. Reported, not guessed.
            print(
                f"[HREFLEX] {ch}: игнорирую старые пометки без указания ответа "
                f"({legacy}) — пометьте заново, выбрав M или H.",
                flush=True,
            )
        res = measure_curves(curves, times, sfreq, baseline_mask, ref,
                             resp_tmin, resp_tmax)
        _apply_marks(res, ch_marks, curves, times, sfreq,
                     baseline_mask, ref, resp_tmin, resp_tmax)

        for comp in COMPONENTS:
            cols = columns_for(comp)
            for r in rows:
                curve = int(df.at[r, "Epoch"])
                if not (0 <= curve < n_curves):
                    for c in cols.values():
                        df.at[r, c] = np.nan
                    continue
                for key, col in cols.items():
                    df.at[r, col] = res[comp][key][curve]

        # The single-response columns stay, holding the M-wave: everything that
        # reads this file without knowing about the scenario (the summary
        # workbooks, the per-crop panels, another tool) then reads the direct
        # response rather than whichever of the two happened to be larger.
        for key, col in LEGACY_COLUMNS.items():
            df.loc[rows, col] = df.loc[rows, columns_for("M")[key]].values

        got = {c: int(np.sum(np.isfinite(res[c]["p1"]))) for c in COMPONENTS}
        if ref is not None:
            report.append(
                f"{ch}: M {ref['m_p1'] * 1e3:.1f} мс на {got['M']} кривых, "
                + (f"H {ref['h_p1'] * 1e3:.1f} мс на {got['H']} кривых"
                   if np.isfinite(ref.get("h_p1", np.nan)) else "H не найден")
                + f" ({ref['source']})"
            )
        else:
            report.append(f"{ch}: ответов не найдено")

    df.to_csv(csv, index=False)
    for line in report:
        print(f"[HREFLEX] {line}", flush=True)
    return df


# --------------------------------------------------------------------------- #
# Deliverable
# --------------------------------------------------------------------------- #
def tidy_metrics(csv: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Per-curve M and H metrics in physiological units (ms, µV).

    One row per curve and channel, with both components side by side — the shape
    the scenario is about, and what the scenario view reads.
    """
    df = pd.read_csv(csv)
    if df.empty or "Epoch" not in df.columns:
        return None, []
    out = pd.DataFrame({
        "Curve": df["Epoch"].astype(int) + 1,
        "Channel": df["Channel"].astype(str),
    })
    for comp in COMPONENTS:
        cols = columns_for(comp)
        for key, unit, scale in (("onset", "ms", 1e3), ("p1", "ms", 1e3),
                                 ("p2", "ms", 1e3), ("pv1", "uV", 1e6),
                                 ("pv2", "uV", 1e6), ("ptp", "uV", 1e6)):
            name = {"onset": f"{comp} onset ms", "p1": f"{comp} P1 ms",
                    "p2": f"{comp} P2 ms", "pv1": f"{comp} P1 uV",
                    "pv2": f"{comp} P2 uV", "ptp": f"{comp} PTP uV"}[key]
            out[name] = pd.to_numeric(df.get(cols[key]), errors="coerce") * scale
        # One amplitude per component, so the two curves are directly comparable.
        out[f"{comp} amplitude uV"] = out[f"{comp} PTP uV"]

    responders = [c for c in sorted(out["Channel"].unique(), key=lambda s: (len(s), s))
                  if out.loc[out["Channel"] == c,
                             [f"{c2} P1 uV" for c2 in COMPONENTS]].notna().any().any()]
    out = out[out["Channel"].isin(responders)].copy()
    return (out if not out.empty else None), responders


def hm_summary(tidy: pd.DataFrame, responders: list[str]) -> pd.DataFrame:
    """Hmax, Mmax and their ratio — the numbers an H-reflex study is read for.

    Hmax/Mmax is the standard measure of reflex excitability, and both maxima
    come from different points of the sweep by construction, so each is reported
    with the curve it was reached on. The threshold curve is the first curve
    carrying that component at all, which is where the sweep is read for
    excitability rather than for size.
    """
    rows = []
    for ch in responders:
        d = tidy[tidy["Channel"] == ch]
        rec = {"Channel": ch}
        for comp in COMPONENTS:
            a = d[f"{comp} amplitude uV"]
            ok = a.notna()
            rec[f"{comp}max uV"] = round(float(a.max()), 2) if ok.any() else np.nan
            rec[f"{comp}max curve"] = (int(d.loc[a.idxmax(), "Curve"])
                                       if ok.any() else np.nan)
            rec[f"{comp} threshold curve"] = (int(d.loc[ok, "Curve"].min())
                                              if ok.any() else np.nan)
            rec[f"{comp} N curves"] = int(ok.sum())
        m, h = rec["Mmax uV"], rec["Hmax uV"]
        rec["Hmax/Mmax"] = (round(float(h / m), 3)
                            if np.isfinite(m) and np.isfinite(h) and m > 0 else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_hreflex_analysis(output_root: Path) -> Path | None:
    """Tables and figures for a finished H-reflex run."""
    output_root = Path(output_root)
    csv = metrics_csv(output_root)
    if not csv.exists():
        print("[HREFLEX] No SIR metrics CSV found; skipping.", flush=True)
        return None
    tidy, responders = tidy_metrics(csv)
    if tidy is None:
        print("[HREFLEX] No responding channels; skipping.", flush=True)
        return None

    out_dir = ensure_dir(_sir_dir(output_root) / "H-reflex")
    excel_dir = ensure_dir(_sir_dir(output_root) / "Excel")

    tidy.round(3).sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "hreflex_by_curve_long.csv", index=False)
    for comp in COMPONENTS:
        (tidy.pivot_table(index="Curve", columns="Channel",
                          values=f"{comp} amplitude uV")
         .reindex(columns=responders).round(3)
         .to_csv(excel_dir / f"hreflex_{comp}_amplitude_uV_wide.csv"))
    hm = hm_summary(tidy, responders)
    hm.to_csv(excel_dir / "stats_hm.csv", index=False)

    _plot_hm_recruitment(tidy, responders, hm, out_dir / "hm_recruitment_curves.png")
    _plot_latencies(tidy, responders, out_dir / "hm_latencies.png")

    print(f"[HREFLEX] {len(responders)} channel(s), {tidy['Curve'].nunique()} curves "
          f"-> {out_dir}; tables -> {excel_dir}", flush=True)
    return out_dir


def run_hreflex_jendrassik_groups(
    output_root: Path, n_groups: int | None = None,
) -> Path | None:
    """Amplitude groups for an H-reflex file that was ALSO a Jendrassik run.

    28 of the 75 H-reflex files in this dataset carry a Jendrassik token as well
    (``Н рефлекс СОЛ лев Ендрассик``): the reflex was measured over a block of
    plain test stimuli and a block with the manoeuvre. The H-reflex scenario
    owns those files — measuring them as a Jendrassik run collapses M and H into
    one column — but the block comparison is still the thing the recording was
    made for, so it is produced here on top of the M/H deliverable.

    **Grouped on the H-reflex amplitude**, not on the M-wave's. The Jendrassik
    manoeuvre works by raising the excitability of the motoneuron pool, which is
    what the reflex measures; the direct motor response does not go through the
    pool at all and should be roughly unchanged between the blocks — so grouping
    on it would split the file on nothing and report the manoeuvre as having no
    effect. (That the M-wave stays put across the groups is itself worth
    checking, and it is in the per-curve table.)

    Everything else follows the Jendrassik scenario: the count of groups comes
    from the intensities named in the file (two per intensity), each curve is
    drawn rather than averaged, and the ``Curves`` column says which curves fell
    in each group — the check that the split found the protocol blocks and not
    amplitude noise. Outputs live under ``H-reflex/`` rather than in a second
    scenario folder: the run has one scenario, and this is part of its
    deliverable.
    """
    from .jendrassik import (
        _curve_span, _group_colors, _load_epoch_waveforms, _plot_curves_by_group,
        _plot_group_boxplots,
    )
    from .recruitment import _assign_amplitude_groups, _stats, channel_group_labels

    output_root = Path(output_root)
    csv = metrics_csv(output_root)
    if not csv.exists():
        return None
    tidy, responders = tidy_metrics(csv)
    if tidy is None:
        return None
    # Only channels that actually carry a reflex: grouping the M-wave of a
    # channel with no H would produce two groups of a quantity the manoeuvre is
    # not expected to move, labelled as if it were the reflex.
    responders = [c for c in responders
                  if tidy.loc[tidy["Channel"] == c, "H P1 uV"].notna().any()]
    if not responders:
        print("[HREFLEX/JM] H-рефлекс не выделен ни на одном канале — "
              "группировка по нему невозможна, пропускаю.", flush=True)
        return None

    if n_groups is None:
        from .jendrassik import _jendrassik_groups_from_name
        n_groups = _jendrassik_groups_from_name(output_root)

    # The shared group/plot helpers speak the single-response column names, so
    # the H columns are presented under them. Renaming rather than reimplementing
    # keeps this figure identical to the one the Jendrassik scenario produces.
    grouped = pd.DataFrame({
        "Channel": tidy["Channel"], "Curve": tidy["Curve"],
        "Onset ms": tidy["H onset ms"], "P1 ms": tidy["H P1 ms"],
        "P2 ms": tidy["H P2 ms"], "P1 uV": tidy["H P1 uV"],
        "P2 uV": tidy["H P2 uV"], "PTP uV": tidy["H PTP uV"],
        "Amplitude uV": tidy["H amplitude uV"],
        # Carried through so the table answers "did the M-wave stay put?" —
        # the control the whole comparison rests on.
        "M amplitude uV": tidy["M amplitude uV"], "M P1 ms": tidy["M P1 ms"],
    })
    grouped = grouped[grouped["Channel"].isin(responders)].copy()
    grouped = _assign_amplitude_groups(grouped, responders, n_groups)

    out_dir = ensure_dir(_sir_dir(output_root) / "H-reflex")
    excel_dir = ensure_dir(_sir_dir(output_root) / "Excel")
    grouped.round(3).sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "hreflex_jendrassik_by_curve_long.csv", index=False)

    n_seen = max((len(channel_group_labels(grouped, ch)) for ch in responders), default=1)
    labels = [f"G{i + 1}" for i in range(n_seen)]
    colors = _group_colors(n_seen)

    metrics = [m for m in ["Amplitude uV", "PTP uV", "P1 uV", "P2 uV", "P1 ms",
                           "Onset ms", "M amplitude uV"]
               if grouped[m].notna().any()]
    rows = []
    for ch in responders:
        d = grouped[grouped["Channel"] == ch]
        for grp in labels:
            dg = d[d["Amplitude group"] == grp]
            if dg.empty:
                continue
            for metric in metrics:
                rows.append({"Channel": ch, "Amplitude group": grp,
                             "Curves": _curve_span(dg["Curve"]),
                             "Metric": metric, **_stats(dg[metric])})
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = stats[stats["N"] > 0]
    stats.to_csv(excel_dir / "hreflex_jendrassik_group_stats.csv", index=False)

    times, waves = _load_epoch_waveforms(output_root)
    drawn = [ch for ch in responders if ch in waves]
    if times is not None and drawn:
        _plot_curves_by_group(
            times, waves, grouped, drawn, labels, colors,
            out_dir / "jendrassik_curves_by_H_group.png",
            "приём Ендрассика, группы по H-рефлексу")
    _plot_group_boxplots(grouped, responders, labels, colors,
                         out_dir / "jendrassik_H_amplitude_by_group_boxplots.png")

    print(f"[HREFLEX/JM] {len(responders)} канал(ов), {n_groups} групп по амплитуде "
          f"H-рефлекса -> {out_dir}; таблицы -> {excel_dir}", flush=True)
    return out_dir


#: M and H keep the same two colours everywhere — figures, tables and the GUI.
COMPONENT_COLORS = {"M": "#1f77b4", "H": "#d62728"}


def _grid(n, ncol=2):
    ncol = max(1, min(ncol, n))
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.2 * nrow),
                             squeeze=False)
    return fig, axes, nrow, ncol


def _full_curve_axis(ax, tidy) -> None:
    """Show the whole sweep, not just the part that responded.

    The curves before threshold are the reason a recruitment sweep is run:
    letting matplotlib start the axis at the first detected curve hides how much
    of the ramp was sub-threshold, and hides it differently for M and for H,
    which is precisely the comparison the figure exists for.
    """
    curves = tidy["Curve"].to_numpy(int)
    if curves.size:
        ax.set_xlim(0.5, float(curves.max()) + 0.5)


def _plot_hm_recruitment(tidy, responders, hm, out_path) -> None:
    """M and H amplitude against curve number — the H/M recruitment curve.

    Both on one axes per channel, because the whole reading of this figure is
    the crossover: the reflex peaks and recedes while the direct response is
    still growing, and that only shows when the two are drawn against each other.
    """
    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        for comp in COMPONENTS:
            ax.plot(d["Curve"], d[f"{comp} amplitude uV"], marker="o", ms=3.5,
                    lw=1.3, color=COMPONENT_COLORS[comp], label=f"{comp}-ответ")
        row = hm[hm["Channel"] == ch]
        ratio = float(row["Hmax/Mmax"].iloc[0]) if not row.empty else np.nan
        note = f"Hmax/Mmax = {ratio:.2f}" if np.isfinite(ratio) else "H не выделен"
        ax.set_title(f"{ch}   ({note})", fontsize=10, loc="left")
        ax.set_ylabel("µV")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        _full_curve_axis(ax, tidy)
        ax.legend(fontsize=8, frameon=False)
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("номер кривой")
    fig.suptitle("H-рефлекс: размах M- и H-ответа по кривым (PTP)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_latencies(tidy, responders, out_path) -> None:
    """Onset and P1 latency of both components against curve number.

    Kept apart from the amplitude figure on purpose: the latencies are how you
    check that the two components stayed themselves across the sweep. A reflex
    whose P1 wanders by more than a millisecond or two, or that swaps places
    with the M-wave on some curves, is a detection that has gone wrong, and it
    is invisible on the amplitude plot.
    """
    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        for comp in COMPONENTS:
            ax.plot(d["Curve"], d[f"{comp} P1 ms"], "o", ms=3.5,
                    color=COMPONENT_COLORS[comp], label=f"{comp} P1")
            ax.plot(d["Curve"], d[f"{comp} onset ms"], "x", ms=3.5, alpha=0.6,
                    color=COMPONENT_COLORS[comp], label=f"{comp} onset")
        ax.set_title(ch, fontsize=10, loc="left")
        ax.set_ylabel("мс от стимула")
        ax.grid(alpha=0.3)
        _full_curve_axis(ax, tidy)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("номер кривой")
    fig.suptitle("H-рефлекс: латентности M- и H-ответа", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

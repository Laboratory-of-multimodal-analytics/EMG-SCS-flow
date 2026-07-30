"""Condition-test analysis for paired-stimulation text ``curves`` exports.

Paradigm (e.g. TMS + tSCS paired stimulation): each curve carries ONE stimulus
artifact, and its position — the "condition" = inter-stimulus interval (ISI) —
moves along the curve from curve to curve. The muscle response follows the
artifact, so it too shifts along the curve. This is different from the
single-shock SIR export (stimulus fixed at t=0) that the standard text-curves
path handles: here the artifact must be found per curve and the response
measured *relative to it*.

Auto-routing: the text-curves loader detects the artifact channel and the
per-curve artifact positions; if those positions vary (spread beyond a few ms)
the file is a Condition test and goes here, otherwise it stays on the SIR path.

Outputs (straight in ``results/`` when this is the only analysis in the output
root, else under ``results/Condition test/``):
  - ``Waterfall/<ch>_by_artifact.png`` / ``<ch>_by_time.png`` — per-condition
    mean curves stacked (control at the bottom, ISI increasing upward), aligned
    on the artifact or in real recording time.
  - ``Amplitude vs condition/<ch>.png`` — response amplitude (peak-to-peak) vs
    condition, mean +/- SE across the repeats of each condition.
  - ``Excel/condition_amplitudes_per_curve.csv`` — one row per (channel, curve).
  - ``Excel/condition_summary.csv`` — per (channel, condition): amplitude
    mean/SE plus onset/P1/P2 latency and value from the per-condition mean.
  - ``arrays/<ch>_condition_means.npy`` + ``_times_ms.npy`` — per-condition mean
    waveforms (artifact-aligned) so plots/windows can be recomputed cheaply.

Everything downstream of the artifact search reuses no SIR state, so this runs
identically whether launched from ``run_pipeline`` (script), the CLI, or the GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

from .io_utils import CONDITION_FOLDER, resolve_mode_dirs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Response search window relative to the artifact (s).
CONDITION_RESP_LO = 0.005
CONDITION_RESP_HI = 0.040
# Baseline just before the artifact for the per-condition mean (s).
CONDITION_BASE_LO = -0.005
CONDITION_BASE_HI = -0.001
# Stack window around the artifact for the waterfall (s).
CONDITION_STACK_PRE = 0.02
CONDITION_STACK_POST = 0.06
# Curves whose artifact positions differ by more than this are a Condition test (ms).
CONDITION_SPREAD_MIN_MS = 5.0
# Two artifact positions within this gap belong to the same condition (ms).
CONDITION_CLUSTER_TOL_MS = 6.0
# A channel this many times louder than the median channel is a stimulus/artifact
# channel, excluded from the muscle analysis.
CONDITION_ART_CHAN_RATIO = 3.0
# Template-detection window relative to the artifact (s): where P1 may be placed.
CONDITION_DET_TMIN = 0.003
CONDITION_DET_TMAX = 0.045
# A channel's strongest-condition mean must match a bank template at least this
# well, else the channel is taken to carry no response (all amplitudes 0).
CONDITION_TM_MIN_CORR = 0.7
# Per curve: its response must correlate with the matched template at least this
# well AND its P1-P2 must exceed the floor below, else there is no response -> 0.
CONDITION_PRESENCE_CORR = 0.5
CONDITION_PRESENCE_MIN_UV = 20.0
# P2 (green) = the biphasic PARTNER of P1: the opposite-DIRECTION extremum (the
# trough after a peak, or the peak after a trough) in a short window right after
# P1. It is a relative turn, not a baseline crossing — these responses often dip
# back toward, but not across, baseline. Kept only if the P1->P2 excursion is at
# least CONDITION_P2_MIN_FRAC of |P1|, else the response is monophasic.
CONDITION_P2_WIN_MS = 8.0
CONDITION_P2_MIN_FRAC = 0.15
# A channel whose STRONGEST condition response (peak-to-peak) is below this is a
# non-responder. 40 uV, not 100: on DZh11 нерв Th12-L1 ch6 carries a real,
# condition-modulated response of 73 uV at 16-21 ms, and a 100 uV floor deleted
# the whole channel.
CONDITION_CHANNEL_MIN_PTP_UV = 40.0

# Amplitude alone cannot tell a response from the artifact's recovery tail — on
# that same file ch4 and ch8 show 375 uV in the response window with no response
# in them at all, and they match a bank template at corr 0.88-0.89 because the
# bank is scale- and width-free. What separates them is SHAPE: an evoked response
# is a fast deflection a few ms wide, the recovery is a slow monotone drift over
# tens of ms. So the channel mean is split into a slow part (running median over
# CONDITION_SHARP_WIN_MS) and the rest, and the rest has to carry a real share of
# the amplitude. Measured on that file: genuine responses 0.49-0.87, tails 0.20-0.22.
CONDITION_SHARP_WIN_MS = 8.0
CONDITION_MIN_SHARP_FRAC = 0.35
# Template time-scales for matching. These responses are narrow, so allow the
# templates to be COMPRESSED (<1) but not stretched (>1) — a stretched wide
# template would match a narrow response spuriously and misplace the markers.
CONDITION_TM_SCALES = (0.3, 0.5, 0.7, 0.85, 1.0)


#: Manual corrections live next to the results, so a re-run from the CLI honours
#: what was marked in the GUI and vice versa. One entry per channel:
#:   {"ch5": {"reject": true},
#:    "ch6": {"window_ms": [12.0, 24.0]}}
OVERRIDES_NAME = "condition_overrides.json"


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


def _manual_match(per_cond_mean, labels, tw_s, window_ms, markers_ms=None) -> dict | None:
    """Template taken from a window the clinician drew, instead of the bank.

    The reference stays the strongest condition — that is where the response is
    cleanest — but its shape inside the given window becomes the template, and
    P1/P2 are placed within that window. The bank is not consulted at all, so a
    response the bank has no shape for can still be measured.
    """
    lo, hi = (float(window_ms[0]) / 1e3, float(window_ms[1]) / 1e3)
    if hi <= lo:
        return None
    win = (tw_s >= lo) & (tw_s <= hi)
    if win.sum() < 5:
        return None
    best_lab, best_ptp = None, -np.inf
    for lab in labels:
        seg = per_cond_mean[lab][win]
        if not np.any(np.isfinite(seg)):
            continue
        p = float(np.nanmax(seg) - np.nanmin(seg))
        if p > best_ptp:
            best_ptp, best_lab = p, lab
    if best_lab is None:
        return None
    ref = np.nan_to_num(per_cond_mean[best_lab], nan=0.0)
    match = {"template": ref, "name": "окно вручную", "corr": np.nan,
             "onset": np.nan, "p1": np.nan, "p2": np.nan}
    if markers_ms:
        # Markers placed by hand win outright — disagreeing with the automatic
        # placement is the whole reason for drawing them.
        for key, src in (("onset", "onset"), ("p1", "p1"), ("p2", "p2")):
            v = markers_ms.get(src)
            match[key] = float(v) / 1e3 if v is not None else np.nan
        return match
    return _refine_markers_on_mean(ref, tw_s, match, window=(lo, hi))


def _channel_loudness(data: np.ndarray) -> np.ndarray:
    """Typical per-curve PEAK absolute amplitude per channel.

    The stimulus artifact is a brief, sharp spike (a handful of samples), so a
    percentile over the whole curve misses it — muscle responses, though smaller,
    are broader and would dominate a percentile. Taking the per-curve max and
    then the median across curves isolates the reproducible artifact spike, which
    is 1-2 orders of magnitude taller than any muscle response.
    """
    return np.median(np.max(np.abs(data), axis=2), axis=0)


def detect_artifact_channel(data: np.ndarray) -> int:
    """Index of the loudest channel — the stimulus/artifact channel."""
    return int(np.argmax(_channel_loudness(data)))


def artifact_channels(data: np.ndarray) -> list[int]:
    """All channels loud enough to be stimulus/artifact channels."""
    loud = _channel_loudness(data)
    med = float(np.median(loud))
    if med <= 0:
        return [int(np.argmax(loud))]
    return [i for i, v in enumerate(loud) if v >= CONDITION_ART_CHAN_RATIO * med]


def artifact_positions(data: np.ndarray, art_ch: int) -> np.ndarray:
    """Per-curve artifact sample index (largest deflection on ``art_ch``)."""
    art = data[:, art_ch, :]
    base = np.median(art, axis=1, keepdims=True)
    return np.abs(art - base).argmax(1)


def is_condition_paradigm(data: np.ndarray, sfreq: float) -> tuple[bool, dict]:
    """True when the artifact position varies across curves (paired-pulse).

    Returns ``(is_condition, info)`` where info carries the artifact channel,
    per-curve positions (ms) and their spread, so the caller can log the reason.
    """
    art_ch = detect_artifact_channel(data)
    pos = artifact_positions(data, art_ch)
    pos_ms = pos / sfreq * 1e3
    spread = float(np.percentile(pos_ms, 95) - np.percentile(pos_ms, 5))
    info = {
        "artifact_channel": art_ch,
        "positions_ms": pos_ms,
        "spread_ms": spread,
        "artifact_channels": artifact_channels(data),
    }
    return spread > CONDITION_SPREAD_MIN_MS, info


def group_conditions(pos_ms: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Cluster artifact positions into conditions.

    Positions within ``CONDITION_CLUSTER_TOL_MS`` form one condition; the label
    is the cluster's median position rounded to the nearest 10 ms (the paradigm
    grid). Returns ``(label_per_curve, sorted_unique_labels)``.
    """
    order = np.argsort(pos_ms)
    labels = np.zeros(len(pos_ms))
    cluster_start = pos_ms[order[0]]
    prev = pos_ms[order[0]]
    members = [order[0]]

    def flush(members, out):
        lab = round(float(np.median(pos_ms[members])) / 10.0) * 10.0
        for m in members:
            out[m] = lab

    for idx in order[1:]:
        if pos_ms[idx] - prev > CONDITION_CLUSTER_TOL_MS:
            flush(members, labels)
            members = []
        members.append(idx)
        prev = pos_ms[idx]
    flush(members, labels)
    return labels, sorted(set(labels.tolist()))


def _aligned_segment(sig: np.ndarray, a_i: int, pre: int, post: int) -> np.ndarray:
    """Crop [a_i-pre, a_i+post] with NaN padding past the record edges."""
    out = np.full(pre + post, np.nan)
    lo, hi = a_i - pre, a_i + post
    src_lo, src_hi = max(lo, 0), min(hi, len(sig))
    out[src_lo - lo:src_hi - lo] = sig[src_lo:src_hi]
    return out


def _ptp_after_artifact(sig: np.ndarray, a_i: int, sfreq: float) -> float:
    lo = a_i + int(CONDITION_RESP_LO * sfreq)
    hi = a_i + int(CONDITION_RESP_HI * sfreq)
    lo, hi = max(lo, 0), min(hi, len(sig))
    if hi - lo < 2:
        return np.nan
    seg = sig[lo:hi]
    return float(np.nanmax(seg) - np.nanmin(seg))


def _load_bank():
    """Pre-computed template bank (lazy import avoids a pipeline<->condition cycle)."""
    from .pipeline import _load_startstop_template_bank, _resolve_startstop_template_dir
    from .constants import STARTSTOP_TM_TEMPLATE_SFREQ, STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE
    return _load_startstop_template_bank(
        template_dir=_resolve_startstop_template_dir(),
        template_native_sfreq=float(STARTSTOP_TM_TEMPLATE_SFREQ),
        template_center_sample=int(STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE),
    )


def _match_channel_template(per_cond_mean, labels, tw_s, sfreq, bank):
    """Match the channel's STRONGEST-condition mean to the bank.

    Returns the best-match dict (template waveform on tw_s, onset/p1/p2 times,
    corr) or None when nothing clears the threshold (channel has no response).
    The reference is the condition whose mean has the largest response-window
    peak-to-peak, so the template is fit on the cleanest available response.
    """
    from .pipeline import _match_sir_template

    base_mask = (tw_s >= CONDITION_BASE_LO) & (tw_s <= CONDITION_BASE_HI)
    resp_mask = (tw_s >= CONDITION_DET_TMIN) & (tw_s <= CONDITION_DET_TMAX)
    # pick the strongest condition as the reference
    best_lab, best_ptp = None, -np.inf
    for lab in labels:
        w = per_cond_mean[lab]
        seg = w[resp_mask]
        if not np.any(np.isfinite(seg)):
            continue
        p = float(np.nanmax(seg) - np.nanmin(seg))
        if p > best_ptp:
            best_ptp, best_lab = p, lab
    if best_lab is None:
        return None
    # channel-level responder gate: even the strongest condition must clear a
    # peak-to-peak floor, else this is a noise-only (non-responding) channel.
    if best_ptp < CONDITION_CHANNEL_MIN_PTP_UV:
        return None
    # Shape gate: reject the artifact's recovery tail, which passes any
    # amplitude test and matches the bank just as well as a real response.
    if not _has_sharp_response(per_cond_mean[best_lab], tw_s, resp_mask):
        return None
    ref = np.nan_to_num(per_cond_mean[best_lab], nan=0.0)
    match = _match_sir_template(
        ref, tw_s, sfreq, base_mask,
        resp_tmin=CONDITION_DET_TMIN, resp_tmax=CONDITION_DET_TMAX,
        template_bank=bank, scales=CONDITION_TM_SCALES, min_corr=CONDITION_TM_MIN_CORR,
    )
    if match is None:
        return None
    # The bank match confirms a response and gives an approximate P1; snap P1/P2
    # to the REAL peaks of the reference mean (the template's own P2 position is
    # unreliable — see ch2/ch3), and drop P2 for a monophasic response.
    return _refine_markers_on_mean(ref, tw_s, match)


def _refine_markers_on_mean(ref_mean, tw_s, match, window=None):
    """Data-driven P1/P2/onset on the reference mean.

    The bank match only confirms that a stereotyped response is present; its own
    P1/P2 latencies are unreliable (the anchor can land off the real peak). So
    take P1 = the DOMINANT deflection in the response window, P2 = the strongest
    opposite-polarity rebound after it — kept only if it reaches
    CONDITION_P2_MIN_FRAC of |P1| (else monophasic → P2 = NaN) — onset = foot of P1.
    """
    base_mask = (tw_s >= CONDITION_BASE_LO) & (tw_s <= CONDITION_BASE_HI)
    if window is None:
        resp_mask = (tw_s >= CONDITION_DET_TMIN) & (tw_s <= CONDITION_DET_TMAX)
    else:
        resp_mask = (tw_s >= window[0]) & (tw_s <= window[1])
    base = np.nanmean(ref_mean[base_mask])
    base = 0.0 if not np.isfinite(base) else float(base)
    w = ref_mean - base

    ir = np.where(resp_mask)[0]
    if not len(ir):
        return match
    p1_i = ir[int(np.nanargmax(np.abs(w[ir])))]      # dominant deflection = P1
    t_p1 = float(tw_s[p1_i])
    p1v = float(w[p1_i])
    pol1 = 1 if p1v >= 0 else -1

    t_p2 = np.nan
    p2_hi = t_p1 + CONDITION_P2_WIN_MS / 1e3      # biphasic partner sits just after P1
    aft = np.where(resp_mask & (tw_s > t_p1) & (tw_s <= p2_hi))[0]
    if len(aft) and abs(p1v) > 0:
        # opposite DIRECTION extremum: trough after a peak, peak after a trough
        p2_i = aft[int(np.argmin(pol1 * w[aft]))]
        if abs(p1v - float(w[p2_i])) >= CONDITION_P2_MIN_FRAC * abs(p1v):
            t_p2 = float(tw_s[p2_i])

    lo = int(ir[0])
    thr = 0.2 * abs(p1v)
    oi = p1_i
    while oi - 1 >= lo and abs(w[oi - 1]) > thr:
        oi -= 1
    t_on = float(tw_s[oi])

    out = dict(match)
    out["onset"], out["p1"], out["p2"] = t_on, t_p1, t_p2
    return out


def _curve_amplitude(seg_mv, tw_s, match, resp_mask):
    """Amplitude of ONE artifact-aligned curve, gated by the template.

    Present  -> |P1 - P2| sampled at the template latencies (mV);
    absent   -> 0.0 (correlation with the template too low, or P1-P2 below floor).
    """
    from .detection import pick_epoch_value_near_latency
    tmpl = np.asarray(match["template"], dtype=float)
    tr = tmpl[resp_mask] - np.nanmean(tmpl[resp_mask])
    sr = seg_mv[resp_mask] - np.nanmean(seg_mv[resp_mask])
    if np.std(tr) == 0 or np.std(sr) == 0 or np.any(np.isnan(sr)):
        return 0.0, False
    corr = float(np.corrcoef(sr, tr)[0, 1])
    if not np.isfinite(corr) or corr < CONDITION_PRESENCE_CORR:
        return 0.0, False
    base = np.nanmean(tmpl[(tw_s >= CONDITION_BASE_LO) & (tw_s <= CONDITION_BASE_HI)])
    base = 0.0 if not np.isfinite(base) else base
    sf = _sf(tw_s)
    # P1 polarity from the mean's P1 direction; P2 is its biphasic partner, so it
    # is sampled with the OPPOSITE polarity (the trough after a peak stays same-
    # side of baseline, so the template sign at P2 would pick the wrong extremum).
    pol1 = 1 if (tmpl[int(np.argmin(np.abs(tw_s - match["p1"])))] - base) >= 0 else -1
    _, p1v = pick_epoch_value_near_latency(seg_mv, tw_s, float(match["p1"]), sf, win_ms=2.0, polarity=pol1)
    if np.isfinite(match["p2"]):
        _, p2v = pick_epoch_value_near_latency(seg_mv, tw_s, float(match["p2"]), sf, win_ms=2.0, polarity=-pol1)
    else:
        p2v = np.nan
    if not np.isfinite(p1v):
        return 0.0, False
    amp = abs(p1v - p2v) if np.isfinite(p2v) else abs(p1v - base)
    if amp < CONDITION_PRESENCE_MIN_UV:
        return 0.0, False
    return float(amp), True


def _has_sharp_response(wave, tw_s, resp_mask) -> bool:
    """True when the response window carries a FAST deflection, not a slow drift.

    The wave is split into a slow part (running median over
    ``CONDITION_SHARP_WIN_MS``, wider than a response and far narrower than a
    recovery) and the remainder. A muscle response survives that subtraction; an
    artifact tail collapses into it.
    """
    from scipy.ndimage import median_filter

    w = np.asarray(wave, dtype=float)
    seg = np.nan_to_num(w[resp_mask])
    if seg.size < 5:
        return False
    total = float(np.ptp(seg))
    if total <= 0:
        return False
    n = max(int(round(CONDITION_SHARP_WIN_MS / 1e3 * _sf(tw_s))), 3) | 1
    sharp = np.nan_to_num(w) - median_filter(np.nan_to_num(w), size=n, mode="nearest")
    return float(np.ptp(sharp[resp_mask])) / total >= CONDITION_MIN_SHARP_FRAC


def _sf(tw_s):
    """Sampling frequency implied by a uniform time axis in seconds."""
    return 1.0 / float(tw_s[1] - tw_s[0])


def _onset_p1_p2(mean_wave: np.ndarray, tw_ms: np.ndarray) -> dict:
    """Onset/P1/P2 on an artifact-aligned per-condition mean (t=0 = artifact).

    P1 = largest deflection in the response window; P2 = strongest opposite
    rebound after P1; onset = walk back from P1 to 20 % of its height. Latencies
    are milliseconds after the artifact. Mirrors the SIR marker semantics.
    """
    base_mask = (tw_ms >= CONDITION_BASE_LO * 1e3) & (tw_ms <= CONDITION_BASE_HI * 1e3)
    base = float(np.nanmean(mean_wave[base_mask])) if np.any(base_mask) else 0.0
    # control curves have the artifact at t=0 -> no pre-artifact samples -> NaN
    # baseline. Fall back to a zero baseline rather than poisoning the whole trace.
    if not np.isfinite(base):
        base = 0.0
    resp = (tw_ms >= CONDITION_RESP_LO * 1e3) & (tw_ms <= CONDITION_RESP_HI * 1e3)
    if not np.any(resp):
        return {k: np.nan for k in ("onset_ms", "p1_ms", "p2_ms", "p1_val", "p2_val", "ptp")}
    w = mean_wave - base
    idx = np.where(resp)[0]
    p1_i = idx[np.nanargmax(np.abs(w[idx]))]
    p1_val = float(w[p1_i]); p1_sign = np.sign(p1_val) or 1.0
    after = idx[idx > p1_i]
    if len(after):
        opp = -p1_sign * w[after]
        p2_i = after[np.nanargmax(opp)]
        p2_val, p2_ms = float(w[p2_i]), float(tw_ms[p2_i])
    else:
        p2_val, p2_ms = np.nan, np.nan
    # onset: walk back from P1 to 20 % of its amplitude
    thr = 0.2 * abs(p1_val)
    on_i = p1_i
    while on_i > idx[0] and abs(w[on_i]) > thr:
        on_i -= 1
    return {
        "onset_ms": float(tw_ms[on_i]), "p1_ms": float(tw_ms[p1_i]), "p2_ms": p2_ms,
        "p1_val": p1_val, "p2_val": p2_val,
        "ptp": float(np.nanmax(w[idx]) - np.nanmin(w[idx])),
    }


def _plot_waterfall(per_cond_mean, tw_ms, labels, ch_name, out_path, resp_lo_ms, resp_hi_ms,
                    markers_ms=None):
    fig, ax = plt.subplots(figsize=(9, 11))
    step = np.nanmax([np.nanmax(w) - np.nanmin(w) for w in per_cond_mean.values()]) * 1.1
    for row, lab in enumerate(labels):
        y = per_cond_mean[lab] + row * step
        ax.plot(tw_ms, y, "k", lw=0.7)
        tag = "control" if lab == 0 else f"{lab:.0f}"
        ax.text(tw_ms[0], row * step, tag + " ", ha="right", va="center", fontsize=8)
        ax.plot(0, row * step, "r|", ms=10, mew=1.5)
    # template-derived onset/P1/P2 (relative to the artifact) as vertical guides
    if markers_ms is not None:
        on_ms, p1_ms, p2_ms = markers_ms
        if np.isfinite(on_ms):
            ax.axvline(on_ms, color="tab:blue", lw=1.0, ls=":", alpha=0.8)
        if np.isfinite(p1_ms):
            ax.axvline(p1_ms, color="tab:red", lw=1.0, ls="--", alpha=0.8)
        if np.isfinite(p2_ms):
            ax.axvline(p2_ms, color="tab:green", lw=1.0, ls="--", alpha=0.8)
    ax.axvspan(resp_lo_ms, resp_hi_ms, color="0.8", alpha=0.25, zorder=0)
    ax.set_xlim(-10, 60)
    ax.set_yticks([])
    ax.set_xlabel("мс относительно артефакта")
    ax.set_title(f"{ch_name}: кривые по condition (control внизу → ISI вверх)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_waterfall_time(data, ch, isi_labels, labels, pos_ms, t_ms, ch_name, out_path):
    fig, ax = plt.subplots(figsize=(9, 11))
    step = np.nanmax([np.nanmax(data[k, ch]) - np.nanmin(data[k, ch]) for k in range(len(data))]) * 1.1
    for row, lab in enumerate(labels):
        idx = np.where(isi_labels == lab)[0]
        wave = data[idx, ch, :].mean(0)
        ax.plot(t_ms, wave + row * step, "k", lw=0.7)
        tag = "control" if lab == 0 else f"{lab:.0f}"
        ax.text(t_ms[0], row * step, tag + " ", ha="right", va="center", fontsize=8)
        ax.plot(float(np.median(pos_ms[idx])), row * step, "r|", ms=8, mew=1.5)
    ax.set_yticks([])
    ax.set_xlabel("мс (реальное время)")
    ax.set_title(f"{ch_name}: кривые по condition, реальное время (ответ едет за артефактом)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_amp_vs_condition_grid(labels, amp_by_channel, out_path):
    """One figure, one subplot per muscle channel: amplitude vs condition."""
    tags = ["control" if l == 0 else f"{l:.0f}" for l in labels]
    xs = np.arange(len(labels))
    n = len(amp_by_channel)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True)
    for i, (ch_name, (amp_mean, amp_se)) in enumerate(amp_by_channel.items()):
        ax = axes[i // ncol][i % ncol]
        ax.errorbar(xs, amp_mean, yerr=amp_se, marker="s", ms=5, color="k",
                    capsize=3, lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(tags, rotation=45, fontsize=7)
        ax.set_ylabel("Amplitude, мкВ")
        ax.set_title(ch_name, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):          # blank any unused cell
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("condition (ISI, мс)")
    fig.suptitle("Ответ vs condition по каналам", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _grid_axes(n, ncol=2):
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True)
    return fig, axes, nrow, ncol


def _plot_curves_per_condition(segs_by_lab, tw_ms, labels, ch_name, out_path,
                               markers_ms=None, present_by_curve=None):
    """Every curve of one channel, one row per condition (mean ± SD on top).

    The waterfall shows one mean per condition; A. Militskova also asks for the
    individual curves, which is where sweep-to-sweep failures of the response
    (the thing paired stimulation is measuring) actually show up.

    The onset/P1/P2 the amplitudes were measured at are drawn as vertical
    guides, and a sweep the presence gate rejected is drawn in purple rather
    than grey — this figure is also how you check that the numbers came off the
    right part of the curve. Purple and not red: red marks the selected interval
    in the GUI, and two reds side by side read as one annotation.
    """
    rows = [lab for lab in labels if segs_by_lab.get(lab, (None, []))[1]]
    if not rows:
        return
    present_by_curve = present_by_curve or {}
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.4 * len(rows)),
                             squeeze=False, sharex=True, sharey=True)
    for r, lab in enumerate(rows):
        ax = axes[r][0]
        idx_list = segs_by_lab[lab][0]
        block = np.asarray(segs_by_lab[lab][1], dtype=float)
        for k, w in zip(idx_list, block):
            ok = present_by_curve.get(int(k), True)
            ax.plot(tw_ms, w, color=("0.55" if ok else "#7b3294"),
                    lw=0.6 if ok else 0.7, alpha=0.6 if ok else 0.9,
                    label=None if ok else "кривая без ответа")
        if markers_ms is not None:
            for t_ms, col in zip(markers_ms, ("tab:blue", "tab:red", "tab:green")):
                if t_ms is not None and np.isfinite(t_ms):
                    ax.axvline(t_ms, color=col, lw=1.0, ls="--", alpha=0.8, zorder=5)
        m = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0, ddof=1) if len(block) > 1 else np.zeros_like(m)
        ax.fill_between(tw_ms, m - sd, m + sd, color="tab:blue", alpha=0.20, lw=0)
        ax.plot(tw_ms, m, color="tab:blue", lw=1.8)
        ax.axvline(0, color="tab:red", lw=1.0, ls=":")
        tag = "control" if lab == 0 else f"ISI {lab:.0f} мс"
        ax.set_title(f"{tag}: n={len(block)} кривых, среднее ± SD",
                     fontsize=9, loc="left")
        ax.set_ylabel("мкВ")
        ax.grid(alpha=0.3)
    # Shared y-limit from the post-artifact data only: the artifact dwarfs the
    # response and would otherwise flatten every curve onto the zero line.
    post = tw_ms >= CONDITION_RESP_LO * 1e3
    allseg = np.concatenate([np.asarray(segs_by_lab[l][1], dtype=float)[:, post]
                             for l in rows], axis=0)
    if allseg.size and np.isfinite(allseg).any():
        lo, hi = float(np.nanmin(allseg)), float(np.nanmax(allseg))
        pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
        axes[0][0].set_ylim(lo - pad, hi + pad)
    axes[-1][0].set_xlabel("мс относительно артефакта")
    # One legend for the whole figure, de-duplicated: every rejected sweep
    # carries the same label, and matplotlib would otherwise list them all.
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    seen, h, l = set(), [], []
    for hh, ll in zip(handles, labels_):
        if ll and ll not in seen:
            seen.add(ll); h.append(hh); l.append(ll)
    if h:
        axes[0][0].legend(h, l, fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"{ch_name}: отдельные кривые по condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_boxplots_grid(labels, box_by_channel, out_path):
    """One subplot per channel: per-condition distribution of per-curve amplitudes.

    Zeros (no-response sweeps) are included, so suppressed conditions sit on 0.
    """
    tags = ["control" if l == 0 else f"{l:.0f}" for l in labels]
    n = len(box_by_channel)
    fig, axes, nrow, ncol = _grid_axes(n)
    for i, (ch_name, box) in enumerate(box_by_channel.items()):
        ax = axes[i // ncol][i % ncol]
        ax.boxplot(box, positions=np.arange(len(labels)), widths=0.6,
                   showfliers=False, medianprops=dict(color="tab:red"))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(tags, rotation=45, fontsize=7)
        ax.set_ylabel("Amplitude, мкВ")
        ax.set_title(ch_name, fontsize=10, loc="left")
        ax.grid(alpha=0.3, axis="y")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("condition (ISI, мс)")
    fig.suptitle("Распределение амплитуд по condition (боксплоты)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_persistence_grid(labels, persist_by_channel, out_path):
    """One subplot per channel: fraction of sweeps with a detected response."""
    tags = ["control" if l == 0 else f"{l:.0f}" for l in labels]
    xs = np.arange(len(labels))
    n = len(persist_by_channel)
    fig, axes, nrow, ncol = _grid_axes(n)
    for i, (ch_name, frac) in enumerate(persist_by_channel.items()):
        ax = axes[i // ncol][i % ncol]
        ax.bar(xs, frac, color="0.4")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels(tags, rotation=45, fontsize=7)
        ax.set_ylabel("доля с ответом")
        ax.set_title(ch_name, fontsize=10, loc="left")
        ax.grid(alpha=0.3, axis="y")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("condition (ISI, мс)")
    fig.suptitle("Persistence: доля кривых с ответом по condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_condition_analysis(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    output_root: Path,
    info: dict | None = None,
) -> Path:
    """Full Condition-test analysis. ``data`` is (n_curves, n_ch, n_samp) volts.

    Manual corrections, if any were saved for this output root, are applied per
    channel: a rejected channel is reported as having no response, and a channel
    with a drawn window gets its template from that window instead of the bank.
    """
    output_root = Path(output_root)
    overrides = load_overrides(output_root)
    if overrides:
        print(f"[CONDITION] manual corrections for: {', '.join(sorted(overrides))}", flush=True)
    # review/ holds the clinician's corrections and is the one thing in the output
    # root that cannot be recomputed. Deleting the folder to force a clean run
    # takes them with it, so say when there are none but a marker file exists.
    elif (Path(output_root) / "review").exists():
        print("[CONDITION] no manual corrections found (review/ exists but is empty)", flush=True)
    # A condition run is the only thing in this output root, so it writes
    # straight into results/ — the "Condition test" level only comes back when
    # the root already holds another analysis.
    out_dir, _ = resolve_mode_dirs(output_root, CONDITION_FOLDER)
    wf_dir = out_dir / "Waterfall"
    curves_dir = out_dir / "Curves per condition"
    amp_dir = out_dir / "Amplitude vs condition"
    excel_dir = out_dir / "Excel"
    arr_dir = out_dir / "arrays"
    for d in (wf_dir, curves_dir, amp_dir, excel_dir, arr_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Microvolts, matching every other surface in the toolbox — the Condition
    # tables used to be the only ones in mV, which made the same response look a
    # thousand times smaller than in the recruitment tables next to it.
    data_mv = data * 1e6
    n_curves, n_ch, n_samp = data_mv.shape
    t_ms = np.arange(n_samp) / sfreq * 1e3

    if info is None:
        _, info = is_condition_paradigm(data, sfreq)
    art_ch = info["artifact_channel"]
    art_set = set(info["artifact_channels"])
    pos = artifact_positions(data, art_ch)
    pos_ms = pos / sfreq * 1e3
    isi_labels, labels = group_conditions(pos_ms)

    pre = int(CONDITION_STACK_PRE * sfreq)
    post = int(CONDITION_STACK_POST * sfreq)
    tw_ms = (np.arange(pre + post) - pre) / sfreq * 1e3

    per_curve_rows = []
    summary_rows = []
    counts = {lab: int((isi_labels == lab).sum()) for lab in labels}
    print(
        f"[CONDITION] artifact ch{art_ch + 1}, stim channels "
        f"{[c + 1 for c in sorted(art_set)]}, conditions (ms -> curves): "
        + ", ".join(f"{('control' if l == 0 else int(l))}:{counts[l]}" for l in labels),
        flush=True,
    )

    tw_s = tw_ms / 1e3
    bank = _load_bank()
    if not bank:
        print("[CONDITION] WARNING: empty template bank — amplitudes fall back to raw peak-to-peak.", flush=True)

    amp_by_channel: dict[str, tuple[list, list]] = {}
    box_by_channel: dict[str, list] = {}
    persist_by_channel: dict[str, list] = {}
    for ch in range(n_ch):
        if ch in art_set:
            continue
        ch_name = ch_names[ch]

        # per-condition artifact-aligned means (for the template match + waterfall)
        segs_by_lab, per_cond_mean = {}, {}
        for lab in labels:
            idx = np.where(isi_labels == lab)[0]
            segs = [_aligned_segment(data_mv[k, ch], pos[k], pre, post) for k in idx]
            segs_by_lab[lab] = (idx, segs)
            per_cond_mean[lab] = np.nanmean(segs, axis=0)

        # ── find the response: manual correction first, then the bank ──
        ov = overrides.get(ch_name, {}) if isinstance(overrides, dict) else {}
        source = "банк"
        if ov.get("reject"):
            match, source = None, "отклонён вручную"
        elif ov.get("window_ms"):
            match = _manual_match(per_cond_mean, labels, tw_s, ov["window_ms"],
                                  ov.get("markers_ms"))
            source = ("маркеры вручную" if (match and ov.get("markers_ms"))
                      else "окно вручную" if match else "окно вручную (не удалось)")
        else:
            match = _match_channel_template(per_cond_mean, labels, tw_s, sfreq, bank) if bank else None
        resp_mask = (tw_s >= CONDITION_DET_TMIN) & (tw_s <= CONDITION_DET_TMAX)
        tmpl_name = match["name"] if match else None
        tmpl_corr = round(float(match["corr"]), 3) if match else np.nan
        on_ms = round(float(match["onset"]) * 1e3, 2) if match and np.isfinite(match["onset"]) else np.nan
        p1_ms = round(float(match["p1"]) * 1e3, 2) if match else np.nan
        p2_ms = round(float(match["p2"]) * 1e3, 2) if match and np.isfinite(match["p2"]) else np.nan

        amp_mean, amp_se, persistence, box = [], [], [], []
        for lab in labels:
            idx, segs = segs_by_lab[lab]
            curve_amps, present_flags = [], []
            for k, seg in zip(idx, segs):
                if match is None:
                    a, present = 0.0, False
                else:
                    a, present = _curve_amplitude(seg, tw_s, match, resp_mask)
                curve_amps.append(a)
                present_flags.append(present)
                per_curve_rows.append({
                    "Channel": ch_name, "Curve": int(k) + 1,
                    "Condition (ISI ms)": ("control" if lab == 0 else lab),
                    "Artifact ms": round(float(pos_ms[k]), 2),
                    "Amplitude uV": round(float(a), 4),
                    "Response present": bool(present),
                })
            arr = np.array(curve_amps, dtype=float)
            m = float(arr.mean()) if len(arr) else np.nan
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            frac = float(np.mean(present_flags)) if present_flags else 0.0
            amp_mean.append(m); amp_se.append(se); persistence.append(frac); box.append(arr)
            summary_rows.append({
                "Channel": ch_name,
                "Condition (ISI ms)": ("control" if lab == 0 else lab),
                "N": counts[lab],
                "N with response": int(np.sum(present_flags)),
                "Persistence": round(frac, 3),
                "Amplitude mean uV": round(m, 4) if np.isfinite(m) else np.nan,
                "Amplitude SE uV": round(se, 4),
                "Template": tmpl_name,
                "Template corr": tmpl_corr,
                "Источник шаблона": source,
                "Onset ms": on_ms, "P1 ms": p1_ms, "P2 ms": p2_ms,
            })

        # per-channel waterfalls (with template markers) + cached arrays
        _plot_waterfall(per_cond_mean, tw_ms, labels, ch_name,
                        wf_dir / f"{ch_name}_by_artifact.png",
                        CONDITION_RESP_LO * 1e3, CONDITION_RESP_HI * 1e3,
                        markers_ms=(on_ms, p1_ms, p2_ms))
        _plot_waterfall_time(data_mv, ch, isi_labels, labels, pos_ms, t_ms, ch_name,
                             wf_dir / f"{ch_name}_by_time.png")
        present_map = {int(row["Curve"]) - 1: bool(row["Response present"])
                       for row in per_curve_rows if row["Channel"] == ch_name}
        _plot_curves_per_condition(segs_by_lab, tw_ms, labels, ch_name,
                                   curves_dir / f"{ch_name}_curves.png",
                                   markers_ms=(on_ms, p1_ms, p2_ms),
                                   present_by_curve=present_map)
        # For the GUI: the markers the amplitudes were sampled at, which sweeps
        # the presence gate accepted, and the matched template to overlay.
        np.save(arr_dir / f"{ch_name}_markers_ms.npy",
                np.array([on_ms, p1_ms, p2_ms], dtype=float))
        np.save(arr_dir / f"{ch_name}_present.npy",
                np.array([present_map.get(k, False) for k in range(n_curves)], dtype=bool))
        if match is not None:
            np.save(arr_dir / f"{ch_name}_template.npy",
                    np.asarray(match["template"], dtype=float))
        # Per-condition means in REAL recording time as well as artifact-aligned.
        # Aligning is what makes the responses comparable, but it also hides the
        # thing that defines this protocol — that the response travels along the
        # curve — so both views have to be available.
        np.save(arr_dir / f"{ch_name}_condition_means_time.npy",
                np.vstack([data_mv[np.where(isi_labels == lab)[0], ch, :].mean(0)
                           for lab in labels]))
        amp_by_channel[ch_name] = (amp_mean, amp_se)
        box_by_channel[ch_name] = box
        persist_by_channel[ch_name] = persistence
        np.save(arr_dir / f"{ch_name}_condition_means.npy",
                np.vstack([per_cond_mean[l] for l in labels]))
        # Individual artifact-aligned curves too, in file order, so the GUI can
        # pull up one sweep at a time instead of only the per-condition mean —
        # sweep-to-sweep failure of the response is the thing this protocol
        # measures, and a mean hides it.
        aligned = np.full((n_curves, pre + post), np.nan)
        for lab in labels:
            idx, segs = segs_by_lab[lab]
            for k, seg in zip(idx, segs):
                aligned[k] = seg
        np.save(arr_dir / f"{ch_name}_curves_aligned.npy", aligned)

    _plot_amp_vs_condition_grid(labels, amp_by_channel,
                                amp_dir / "amplitude_vs_condition_all_channels.png")
    _plot_boxplots_grid(labels, box_by_channel,
                        amp_dir / "amplitude_boxplots_all_channels.png")
    _plot_persistence_grid(labels, persist_by_channel,
                           amp_dir / "persistence_vs_condition_all_channels.png")
    np.save(arr_dir / "times_ms.npy", tw_ms)
    np.save(arr_dir / "condition_labels.npy", np.array(labels))
    np.save(arr_dir / "curve_condition.npy", np.asarray(isi_labels, dtype=float))
    np.save(arr_dir / "artifact_ms.npy", np.asarray(pos_ms, dtype=float))
    np.save(arr_dir / "times_full_ms.npy", t_ms)
    np.save(arr_dir / "condition_artifact_ms.npy",
            np.array([float(np.median(pos_ms[isi_labels == lab])) for lab in labels]))
    pd.DataFrame(per_curve_rows).to_csv(excel_dir / "condition_amplitudes_per_curve.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(excel_dir / "condition_summary.csv", index=False)
    print(f"[CONDITION] Outputs written under {out_dir}", flush=True)
    return out_dir


# --------------------------------------------------------------------------- #
# Local recompute
# --------------------------------------------------------------------------- #
def recompute_channel(output_root: Path, channel: str, rebuild_shared: bool = True) -> dict:
    """Redo ONE channel from the arrays the run already saved.

    A manual correction changes one channel, and re-running the pipeline to see
    it means re-parsing a 30 MB text export and redoing every other channel —
    minutes of work to answer a question about one trace. Everything needed is
    already on disk: the artifact-aligned curves, their condition labels and the
    time axis. So the correction is applied here, the channel's rows in both
    tables are replaced, its figures are redrawn, and the three all-channel
    figures are rebuilt from the updated tables.

    Returns a small summary of what changed.
    """
    output_root = Path(output_root)
    base, _ = resolve_mode_dirs(output_root, CONDITION_FOLDER)
    arr, excel = base / "arrays", base / "Excel"
    aligned_p = arr / f"{channel}_curves_aligned.npy"
    if not aligned_p.exists():
        raise FileNotFoundError(f"No saved curves for {channel} — run the file once first.")

    aligned = np.load(aligned_p)
    tw_ms = np.load(arr / "times_ms.npy")
    tw_s = tw_ms / 1e3
    labels = [float(v) for v in np.load(arr / "condition_labels.npy")]
    curve_cond = np.load(arr / "curve_condition.npy")
    sfreq = _sf(tw_s)

    segs_by_lab, per_cond_mean = {}, {}
    for lab in labels:
        idx = np.where(curve_cond == lab)[0]
        segs = [aligned[k] for k in idx if k < len(aligned)]
        segs_by_lab[lab] = (idx, segs)
        per_cond_mean[lab] = np.nanmean(segs, axis=0) if segs else np.full(aligned.shape[1], np.nan)

    ov = load_overrides(output_root).get(channel, {})
    if ov.get("reject"):
        match, source = None, "отклонён вручную"
    elif ov.get("window_ms"):
        match = _manual_match(per_cond_mean, labels, tw_s, ov["window_ms"], ov.get("markers_ms"))
        source = ("маркеры вручную" if (match and ov.get("markers_ms"))
                  else "окно вручную" if match else "окно вручную (не удалось)")
    else:
        bank = _load_bank()
        match = _match_channel_template(per_cond_mean, labels, tw_s, sfreq, bank) if bank else None
        source = "банк"

    resp_mask = (tw_s >= CONDITION_DET_TMIN) & (tw_s <= CONDITION_DET_TMAX)
    on_ms = round(float(match["onset"]) * 1e3, 2) if match and np.isfinite(match["onset"]) else np.nan
    p1_ms = round(float(match["p1"]) * 1e3, 2) if match else np.nan
    p2_ms = round(float(match["p2"]) * 1e3, 2) if match and np.isfinite(match["p2"]) else np.nan

    per_rows, sum_rows, present_map = [], [], {}
    amp_mean, amp_se, persistence, box = [], [], [], []
    for lab in labels:
        idx, segs = segs_by_lab[lab]
        amps, flags = [], []
        for k, seg in zip(idx, segs):
            a, present = (0.0, False) if match is None else _curve_amplitude(seg, tw_s, match, resp_mask)
            amps.append(a); flags.append(present); present_map[int(k)] = bool(present)
            per_rows.append({"Channel": channel, "Curve": int(k) + 1,
                             "Condition (ISI ms)": ("control" if lab == 0 else lab),
                             "Artifact ms": np.nan, "Amplitude uV": round(float(a), 4),
                             "Response present": bool(present)})
        a = np.array(amps, dtype=float)
        m = float(a.mean()) if a.size else np.nan
        se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
        amp_mean.append(m); amp_se.append(se)
        persistence.append(float(np.mean(flags)) if flags else 0.0); box.append(a)
        sum_rows.append({"Channel": channel, "Condition (ISI ms)": ("control" if lab == 0 else lab),
                         "N": len(idx), "N with response": int(np.sum(flags)),
                         "Persistence": round(float(np.mean(flags)) if flags else 0.0, 3),
                         "Amplitude mean uV": round(m, 4) if np.isfinite(m) else np.nan,
                         "Amplitude SE uV": round(se, 4),
                         "Template": (match["name"] if match else None),
                         "Template corr": (round(float(match["corr"]), 3)
                                           if match and np.isfinite(match.get("corr", np.nan)) else np.nan),
                         "Источник шаблона": source,
                         "Onset ms": on_ms, "P1 ms": p1_ms, "P2 ms": p2_ms})

    def _replace(path: Path, rows: list[dict]) -> pd.DataFrame:
        df = pd.read_csv(path)
        keep = df[df["Channel"].astype(str) != channel]
        # Artifact ms is not recoverable per curve here; keep whatever was there.
        new = pd.DataFrame(rows)
        if "Artifact ms" in df.columns and "Artifact ms" in new.columns:
            old = df[df["Channel"].astype(str) == channel][["Curve", "Artifact ms"]]
            new = new.drop(columns=["Artifact ms"]).merge(old, on="Curve", how="left")
        out = pd.concat([keep, new[df.columns.intersection(new.columns)]], ignore_index=True)
        out.to_csv(path, index=False)
        return out

    _replace(excel / "condition_amplitudes_per_curve.csv", per_rows)
    summary = _replace(excel / "condition_summary.csv", sum_rows)

    np.save(arr / f"{channel}_markers_ms.npy", np.array([on_ms, p1_ms, p2_ms], dtype=float))
    np.save(arr / f"{channel}_present.npy",
            np.array([present_map.get(k, False) for k in range(len(aligned))], dtype=bool))

    _plot_curves_per_condition(segs_by_lab, tw_ms, labels, channel,
                               base / "Curves per condition" / f"{channel}_curves.png",
                               markers_ms=(on_ms, p1_ms, p2_ms), present_by_curve=present_map)
    _plot_waterfall(per_cond_mean, tw_ms, labels, channel,
                    base / "Waterfall" / f"{channel}_by_artifact.png",
                    CONDITION_RESP_LO * 1e3, CONDITION_RESP_HI * 1e3,
                    markers_ms=(on_ms, p1_ms, p2_ms))

    if rebuild_shared:
        _rebuild_shared_figures(base, labels)
    return {"channel": channel, "source": source, "onset_ms": on_ms, "p1_ms": p1_ms,
            "p2_ms": p2_ms, "n_present": int(sum(present_map.values())), "n_curves": len(aligned)}


def _rebuild_shared_figures(base: Path, labels: list[float]) -> None:
    """The three all-channel figures, redrawn from the tables on disk.

    They span every channel, so after a batch of corrections they are rebuilt
    once rather than once per channel.
    """
    excel = base / "Excel"
    summary = pd.read_csv(excel / "condition_summary.csv")
    per_all = pd.read_csv(excel / "condition_amplitudes_per_curve.csv")
    chans = sorted(summary["Channel"].astype(str).unique(), key=lambda s: (len(s), s))
    key = lambda v: 0.0 if str(v) == "control" else float(v)
    amp_by, box_by, pers_by = {}, {}, {}
    for ch in chans:
        g = summary[summary["Channel"].astype(str) == ch].copy()
        g = g.assign(_k=g["Condition (ISI ms)"].map(key)).sort_values("_k")
        amp_by[ch] = (g["Amplitude mean uV"].tolist(), g["Amplitude SE uV"].tolist())
        pers_by[ch] = g["Persistence"].tolist()
        p = per_all[per_all["Channel"].astype(str) == ch]
        box_by[ch] = [p.loc[p["Condition (ISI ms)"].map(key) == l, "Amplitude uV"].to_numpy(float)
                      for l in labels]
    amp_dir = base / "Amplitude vs condition"
    _plot_amp_vs_condition_grid(labels, amp_by, amp_dir / "amplitude_vs_condition_all_channels.png")
    _plot_boxplots_grid(labels, box_by, amp_dir / "amplitude_boxplots_all_channels.png")
    _plot_persistence_grid(labels, pers_by, amp_dir / "persistence_vs_condition_all_channels.png")


def recompute_corrections(output_root: Path) -> list[dict]:
    """Apply every saved correction at once.

    Corrections are made channel by channel but read together — a run is judged
    as a whole — so this redoes each corrected channel and rebuilds the
    all-channel figures once at the end.
    """
    output_root = Path(output_root)
    ov = load_overrides(output_root)
    if not ov:
        return []
    base, _ = resolve_mode_dirs(output_root, CONDITION_FOLDER)
    labels = [float(v) for v in np.load(base / "arrays" / "condition_labels.npy")]
    out = []
    for ch in sorted(ov):
        try:
            out.append(recompute_channel(output_root, ch, rebuild_shared=False))
        except Exception as exc:
            out.append({"channel": ch, "error": f"{type(exc).__name__}: {exc}"})
    _rebuild_shared_figures(base, labels)
    return out

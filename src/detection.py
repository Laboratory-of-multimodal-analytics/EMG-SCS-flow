"""Signal detection helpers (ported from the notebook)."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _moving_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def rms_envelope(sig: np.ndarray, sfreq: float, win_ms: float) -> np.ndarray:
    w = max(1, int((win_ms / 1000.0) * sfreq))
    return np.sqrt(_moving_mean(sig ** 2, w))


def detect_rms_events(
    sig: np.ndarray,
    sfreq: float,
    win_ms: float,
    k: float,
    min_dist_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    env = rms_envelope(sig, sfreq, win_ms)
    med = np.median(env)
    mad = np.median(np.abs(env - med))
    thr = med + (k * mad * 1.4826)
    min_dist = max(1, int((min_dist_ms / 1000.0) * sfreq))
    peaks, props = find_peaks(env, height=thr, distance=min_dist)
    onsets = []
    offsets = []
    n = len(env)
    for p in peaks:
        left = p
        while left > 0 and env[left] > thr:
            left -= 1
        onset_idx = left + 1 if left < p else p

        right = p
        while right < n - 1 and env[right] > thr:
            right += 1
        offset_idx = right - 1 if right > p else p

        onsets.append(onset_idx)
        offsets.append(offset_idx)
    return env, np.asarray(onsets, dtype=int), np.asarray(offsets, dtype=int), peaks, float(thr)


def detect_onset_rectified(
    sig_filt: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    onset_tmin: float = 0.003,
    onset_tmax: float = 0.035,
    k: float = 2.5,
    sustain_ms: float = 1.5,
) -> float:
    baseline = sig_filt[baseline_mask]
    bmean = baseline.mean()
    bstd = baseline.std()

    onset_mask = (times >= onset_tmin) & (times <= onset_tmax)
    if not np.any(onset_mask) or bstd == 0 or np.isnan(bstd):
        return np.nan

    seg = sig_filt[onset_mask]
    seg_t = times[onset_mask]

    rect = np.abs(seg - bmean)
    thr = k * bstd

    w = max(1, int((sustain_ms / 1000.0) * sfreq))
    rect_s = _moving_mean(rect, w)

    idx = np.where(rect_s > thr)[0]
    if len(idx) == 0:
        return np.nan

    return float(seg_t[idx[0]])


def detect_peak_in_window(
    sig_f: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    t_center: float,
    win_ms: float = 10.0,
    polarity: int = +1,
    amp_min_uV: float = 10.0,
    min_width_ms: float = 0.4,
    choose: str = "dominant",
    template_peak_val: float = np.nan,
    min_rel_to_template: float = 0.1,
) -> tuple[float, float]:
    if np.isnan(t_center):
        return (np.nan, np.nan)

    base = sig_f[baseline_mask]
    bmean = base.mean()
    if np.isnan(bmean):
        return (np.nan, np.nan)

    w = win_ms / 1000.0
    m = (times >= t_center - w) & (times <= t_center + w)
    if np.sum(m) < 5:
        return (np.nan, np.nan)

    seg = sig_f[m] - bmean
    seg_t = times[m]

    min_w = max(1, int((min_width_ms / 1000.0) * sfreq))
    amp_thr_abs = amp_min_uV * 1e-6

    if not np.isnan(template_peak_val):
        amp_thr_rel = min_rel_to_template * abs(template_peak_val)
    else:
        amp_thr_rel = 0.0

    amp_thr = max(amp_thr_abs, amp_thr_rel)

    if polarity >= 0:
        idx, _ = find_peaks(seg, width=min_w)
        if len(idx) == 0:
            return (np.nan, np.nan)

        idx = [i for i in idx if seg[i] >= amp_thr]
        if len(idx) == 0:
            return (np.nan, np.nan)

        if choose == "nearest":
            i = int(idx[np.argmin(np.abs(seg_t[idx] - t_center))])
        else:
            i = int(idx[np.argmax(seg[idx])])

        return (float(seg_t[i]), float(seg[i] + bmean))

    idx, _ = find_peaks(-seg, width=min_w)
    if len(idx) == 0:
        return (np.nan, np.nan)

    idx = [i for i in idx if -seg[i] >= amp_thr]
    if len(idx) == 0:
        return (np.nan, np.nan)

    if choose == "nearest":
        i = int(idx[np.argmin(np.abs(seg_t[idx] - t_center))])
    else:
        i = int(idx[np.argmax(-seg[idx])])

    return (float(seg_t[i]), float(seg[i] + bmean))


def detect_template_peaks(
    template_filt: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    onset_latency: float,
    resp_tmin: float = 0.003,
    resp_tmax: float = 0.040,
    min_prom_k: float = 2.0,
    peak2_max_gap_ms: float = 25.0,
    min_width_ms: float = 0.6,
    amp_min_uV: float = 10.0,
) -> tuple[float, float, float, float]:
    if np.isnan(onset_latency):
        return (np.nan, np.nan, np.nan, np.nan)

    base = template_filt[baseline_mask]
    bmean, bstd = base.mean(), base.std()
    if bstd == 0 or np.isnan(bstd):
        return (np.nan, np.nan, np.nan, np.nan)

    mask = (times >= onset_latency) & (times <= resp_tmax)
    if np.sum(mask) < 5:
        return (np.nan, np.nan, np.nan, np.nan)

    seg = template_filt[mask]
    seg0 = seg - bmean
    seg_t = times[mask]

    prom = min_prom_k * bstd
    min_w = max(1, int((min_width_ms / 1000.0) * sfreq))
    amp_thr = amp_min_uV * 1e-6
    max_gap = peak2_max_gap_ms / 1000.0

    pos_idx, pos_props = find_peaks(seg0, prominence=prom, width=min_w)
    neg_idx, neg_props = find_peaks(-seg0, prominence=prom, width=min_w)
    pos_prom_arr = np.asarray(pos_props.get("prominences", np.zeros(len(pos_idx))), dtype=float)
    neg_prom_arr = np.asarray(neg_props.get("prominences", np.zeros(len(neg_idx))), dtype=float)

    # Guarded fallback: if strict prominence keeps only one polarity,
    # relax prominence for the missing polarity only.
    if (len(pos_idx) == 0) or (len(neg_idx) == 0):
        relaxed_prom = max(amp_thr, 0.35 * float(prom))
        if len(pos_idx) == 0:
            pos_idx_relaxed, pos_props_relaxed = find_peaks(seg0, prominence=relaxed_prom, width=min_w)
            pos_prom_relaxed = np.asarray(
                pos_props_relaxed.get("prominences", np.zeros(len(pos_idx_relaxed))),
                dtype=float,
            )
            if len(pos_idx_relaxed) > 0:
                pos_idx = pos_idx_relaxed
                pos_prom_arr = pos_prom_relaxed
        if len(neg_idx) == 0:
            neg_idx_relaxed, neg_props_relaxed = find_peaks(-seg0, prominence=relaxed_prom, width=min_w)
            neg_prom_relaxed = np.asarray(
                neg_props_relaxed.get("prominences", np.zeros(len(neg_idx_relaxed))),
                dtype=float,
            )
            if len(neg_idx_relaxed) > 0:
                neg_idx = neg_idx_relaxed
                neg_prom_arr = neg_prom_relaxed
        if (len(pos_idx) == 0) or (len(neg_idx) == 0):
            return (np.nan, np.nan, np.nan, np.nan)

    pos_sorted = pos_idx[np.argsort(seg0[pos_idx])[::-1]]
    pos_top = pos_sorted[:2]

    neg_sorted = neg_idx[np.argsort((-seg0[neg_idx]))[::-1]]
    neg_top = neg_sorted[:2]

    pos_prom = {int(i): float(p) for i, p in zip(pos_idx.tolist(), pos_prom_arr.tolist())}
    neg_prom = {int(i): float(p) for i, p in zip(neg_idx.tolist(), neg_prom_arr.tolist())}
    prom_by_idx = {**pos_prom, **neg_prom}

    peaks = []
    for i in pos_top:
        i = int(i)
        peaks.append((i, +1, float(seg[i])))
    for i in neg_top:
        i = int(i)
        peaks.append((i, -1, float(seg[i])))

    peaks = list({p[0]: p for p in peaks}.values())
    peaks.sort(key=lambda x: x[0])
    peak_by_idx = {p[0]: p for p in peaks}

    # Drop non-prominent micro-peaks relative to the strongest candidate.
    if len(peaks) > 0:
        max_prom = max(float(prom_by_idx.get(int(p[0]), 0.0)) for p in peaks)
        if max_prom > 0:
            rel_prom_thr = 0.08 * max_prom
            peaks = [p for p in peaks if float(prom_by_idx.get(int(p[0]), 0.0)) >= rel_prom_thr]
            peaks.sort(key=lambda x: x[0])
            peak_by_idx = {p[0]: p for p in peaks}

    if len(peaks) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    def choose_best_p2(p1_k: int) -> int | None:
        p1_i, p1_sign, _p1_val = peaks[p1_k]
        p1_lat_local = float(seg_t[p1_i])
        candidates = []
        for kk in range(p1_k + 1, len(peaks)):
            i, s, v = peaks[kk]
            t = float(seg_t[i])
            if (t - p1_lat_local) > max_gap:
                break
            if s == -p1_sign:
                candidates.append((kk, i, s, v))
        if not candidates:
            return None
        kk, _i, _s, _v = max(candidates, key=lambda x: abs(x[3]))
        return kk

    if len(peaks) == 2:
        i1, s1, v1 = peaks[0]
        i2, s2, v2 = peaks[1]
        if s1 == s2:
            return (np.nan, np.nan, np.nan, np.nan)
        p1_i, p1_val = i1, v1
        p2_i, p2_val = i2, v2
    else:
        k1 = 0
        k2_first = None
        for kk in range(1, len(peaks)):
            if peaks[kk][1] == -peaks[k1][1]:
                k2_first = kk
                break
        if k2_first is None:
            return (np.nan, np.nan, np.nan, np.nan)

        v1 = peaks[k1][2]
        v2_first = peaks[k2_first][2]
        if abs(v1) < abs(v2_first):
            k1 = k2_first

        k2_best = choose_best_p2(k1)
        if k2_best is None:
            p1_i, p1_val = peaks[k1][0], peaks[k1][2]
            p2_i, p2_val = None, np.nan
        else:
            p1_i, p1_val = peaks[k1][0], peaks[k1][2]
            p2_i, p2_val = peaks[k2_best][0], peaks[k2_best][2]

    # Guard against assigning a tiny pre-wave as P1 when a much more prominent
    # later response was selected as P2 (common in stimulation templates).
    if (p2_i is not None) and (not np.isnan(p2_val)):
        p1_prom = float(prom_by_idx.get(int(p1_i), 0.0))
        p2_prom = float(prom_by_idx.get(int(p2_i), 0.0))
        if p2_prom > 0 and p1_prom < (0.5 * p2_prom):
            _p2_peak = peak_by_idx.get(int(p2_i))
            if _p2_peak is not None:
                p1_i, _p1_sign, p1_val = _p2_peak
                p1_lat_local = float(seg_t[p1_i])
                p2_i, p2_val = None, np.nan
                candidates = []
                for i, s, v in peaks:
                    if i <= p1_i:
                        continue
                    t = float(seg_t[i])
                    if (t - p1_lat_local) > max_gap:
                        break
                    if s == -_p1_sign:
                        candidates.append((i, v))
                if candidates:
                    p2_i, p2_val = max(candidates, key=lambda x: abs(x[1]))

    p1_lat = float(seg_t[p1_i])
    p2_lat = float(seg_t[p2_i]) if (p2_i is not None and not np.isnan(p2_val)) else np.nan

    if abs(p1_val) < amp_thr:
        return (np.nan, np.nan, np.nan, np.nan)

    if (not np.isnan(p2_val)) and (abs(p2_val) < amp_thr):
        p2_lat, p2_val = np.nan, np.nan

    if (not np.isnan(p2_lat)) and (p2_lat <= p1_lat):
        p2_lat, p2_val = np.nan, np.nan

    return (p1_lat, p1_val, p2_lat, p2_val)


def pick_epoch_value_near_latency(
    sig: np.ndarray,
    times: np.ndarray,
    target_lat: float,
    sfreq: float,
    win_ms: float = 2.0,
    polarity: int | None = None,
) -> tuple[float, float]:
    if np.isnan(target_lat):
        return (np.nan, np.nan)

    w = win_ms / 1000.0
    mask = (times >= target_lat - w) & (times <= target_lat + w)
    if np.sum(mask) < 3:
        return (np.nan, np.nan)

    seg = sig[mask]
    seg_t = times[mask]

    if polarity is None:
        idx = int(np.argmax(np.abs(seg)))
    elif polarity > 0:
        idx = int(np.argmax(seg))
    else:
        idx = int(np.argmin(seg))

    return (float(seg_t[idx]), float(seg[idx]))


def detect_onset_near_template(
    sig_f: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    t_on_tmpl: float,
    win_ms: float = 10.0,
    k: float = 2.5,
    sustain_ms: float = 1.5,
) -> float:
    if np.isnan(t_on_tmpl):
        return np.nan

    base = sig_f[baseline_mask]
    bmean, bstd = base.mean(), base.std()
    if bstd == 0 or np.isnan(bstd):
        return np.nan

    w = win_ms / 1000.0
    m = (times >= t_on_tmpl - w) & (times <= t_on_tmpl + w)
    if np.sum(m) < 5:
        return np.nan

    seg = sig_f[m]
    seg_t = times[m]

    rect = np.abs(seg - bmean)
    thr = k * bstd

    ww = max(1, int((sustain_ms / 1000.0) * sfreq))
    rect_s = _moving_mean(rect, ww)

    idx = np.where(rect_s > thr)[0]
    return np.nan if len(idx) == 0 else float(seg_t[idx[0]])


def find_extra_p1_peak(
    sig_f: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    p1_lat: float,
    p2_lat: float,
    p1_polarity: int,
    p2_hint_lat: float = np.nan,
    guard_ms: float = 1.0,
    hint_ms: float = 3.0,
    min_width_ms: float = 0.4,
    amp_min_uV: float = 10.0,
    choose: str = "dominant",
) -> tuple[float, float]:
    if np.isnan(p1_lat) or np.isnan(p2_lat) or (p2_lat <= p1_lat):
        return (np.nan, np.nan)

    g = guard_ms / 1000.0
    tmin = p1_lat + g
    tmax = p2_lat - g
    if (tmax - tmin) <= (2.0 / sfreq):
        return (np.nan, np.nan)

    if not np.isnan(p2_hint_lat):
        h = hint_ms / 1000.0
        tmin = max(tmin, p2_hint_lat - h)
        tmax = min(tmax, p2_hint_lat + h)
        if (tmax - tmin) <= (2.0 / sfreq):
            return (np.nan, np.nan)

    m = (times >= tmin) & (times <= tmax)
    if np.sum(m) < 5:
        return (np.nan, np.nan)

    seg = sig_f[m]
    seg_t = times[m]

    min_w = max(1, int((min_width_ms / 1000.0) * sfreq))
    amp_thr = amp_min_uV * 1e-6

    if p1_polarity >= 0:
        idx, _ = find_peaks(seg, width=min_w)
        if len(idx) == 0:
            return (np.nan, np.nan)
        idx = np.array([i for i in idx if seg[i] >= amp_thr], dtype=int)
        if len(idx) == 0:
            return (np.nan, np.nan)
        if choose == "nearest" and not np.isnan(p2_hint_lat):
            i = int(idx[np.argmin(np.abs(seg_t[idx] - p2_hint_lat))])
        else:
            i = int(idx[np.argmax(seg[idx])])
    else:
        idx, _ = find_peaks(-seg, width=min_w)
        if len(idx) == 0:
            return (np.nan, np.nan)
        idx = np.array([i for i in idx if (-seg[i]) >= amp_thr], dtype=int)
        if len(idx) == 0:
            return (np.nan, np.nan)
        if choose == "nearest" and not np.isnan(p2_hint_lat):
            i = int(idx[np.argmin(np.abs(seg_t[idx] - p2_hint_lat))])
        else:
            i = int(idx[np.argmax(-seg[idx])])

    return (float(seg_t[i]), float(seg[i]))


def add_extra_peak_to_p1(p1_val: float, p1_polarity: int, extra_val: float) -> float:
    if np.isnan(p1_val) or np.isnan(extra_val):
        return p1_val

    if p1_polarity >= 0:
        return float(p1_val + abs(extra_val))
    return float(p1_val - abs(extra_val))

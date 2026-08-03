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
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    if np.isnan(t_center):
        return (np.nan, np.nan)

    base = sig_f[baseline_mask]
    bmean = base.mean()
    if np.isnan(bmean):
        return (np.nan, np.nan)

    w = win_ms / 1000.0
    lo = t_center - w
    hi = t_center + w
    if t_min is not None:
        lo = max(lo, float(t_min))
    if t_max is not None:
        hi = min(hi, float(t_max))
    m = (times >= lo) & (times <= hi)
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
        # "global": take the largest deflection of this polarity anywhere in the
        # window, not a find_peaks local maximum. find_peaks needs a sharp,
        # wide-enough hump and can miss a broad response outright or lock onto a
        # noise spike; on a signal we already trust to hold the response, the
        # plain window extremum is what the clinician means by "the peak".
        if choose == "global":
            i = int(np.argmax(seg))
            if seg[i] < amp_thr:
                return (np.nan, np.nan)
            return (float(seg_t[i]), float(seg[i] + bmean))

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

    if choose == "global":
        i = int(np.argmax(-seg))
        if -seg[i] < amp_thr:
            return (np.nan, np.nan)
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
    noise_std: float = np.nan,
    p1_lat: float = np.nan,
    t_min: float = np.nan,
) -> float:
    """When the response leaves the baseline, searched near the template's onset.

    The onset is the first sustained excursion past ``k`` times the noise, so the
    whole thing rests on having a noise scale. Normally that comes from the
    pre-stimulus baseline — but Neurosoft curve exports begin AT the stimulus,
    and their only pre-artifact data is a pad of one repeated sample, whose
    standard deviation is exactly zero. ``k * 0`` is not a threshold, and this
    function used to give up on the spot, which is why every one of those files
    came out with peaks but no onsets at all.

    ``noise_std`` is the way out: a noise estimate the caller took from somewhere
    the baseline could not provide one (in practice the quiet tail of the curve,
    after the response). When it is used, so is a different search: see
    ``detect_onset_before_peak``. Recordings that do carry pre-stimulus data keep
    the original behaviour untouched.
    """
    base = sig_f[baseline_mask]
    bmean, bstd = base.mean(), base.std()
    w = win_ms / 1000.0

    if bstd == 0 or np.isnan(bstd):
        # The backward search is anchored on P1 and never reads the template's
        # onset marker, so it must not be gated on having one. It was, and that
        # silently cost whole channels: a template whose onset sits more than P1's
        # own latency ahead of it places that marker before the response window
        # begins, where the matcher drops it as implausible. Every curve of such a
        # channel then reported no onset while its P1 sat on a perfectly clean
        # rise — ch7 and ch8 of the 80/90 mA run, 56 curves between them.
        return detect_onset_before_peak(
            sig_f, times, sfreq, bmean, float(noise_std), p1_lat,
            t_min=t_min, k=k, sustain_ms=sustain_ms,
        )

    if np.isnan(t_on_tmpl):
        return np.nan

    m = (times >= t_on_tmpl - w) & (times <= t_on_tmpl + w)
    if np.sum(m) < 5:
        return np.nan

    ww = max(1, int((sustain_ms / 1000.0) * sfreq))
    rect_s = _smoothed_deviation(sig_f, bmean, ww)[m]
    seg_t = times[m]

    idx = np.where(rect_s > k * bstd)[0]
    return np.nan if len(idx) == 0 else float(seg_t[idx[0]])


def _smoothed_deviation(sig_f: np.ndarray, bmean: float, ww: int) -> np.ndarray:
    """|signal - baseline|, smoothed over ``ww`` samples, for the WHOLE curve.

    Smoothing must happen before the search window is cut out, not after. The
    moving average is a same-length convolution, so it pads with zeros at the
    ends of whatever array it is given: smoothing a slice halves the values at
    that slice's own edges. Since the search for an onset ends AT P1, that put a
    fake dip immediately before the peak — on channels where the threshold is a
    large fraction of the response, the walk back from P1 stopped in that dip and
    every onset came out 0.1 ms before its peak.
    """
    return _moving_mean(np.abs(sig_f - bmean), ww)


def detect_onset_before_peak(
    sig_f: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    bmean: float,
    noise_std: float,
    p1_lat: float,
    t_min: float = np.nan,
    k: float = 2.5,
    sustain_ms: float = 1.5,
) -> float:
    """Onset as the last quiet moment before P1.

    Searching a narrow window around the template's onset is the wrong tool on a
    single curve: where the template puts its onset relative to P1 is a property
    of the template, and the bank is generic. Curves whose response starts even a
    few ms off that spacing simply had no onset found — on this dataset that lost
    it on 120 of 314 detections, and placed some of the rest inside the stimulus
    artifact.

    So the search runs backwards from THIS curve's P1 instead, and the onset is
    the last sample whose smoothed rectified deviation is still within the noise.
    That is the foot of the response as read off the curve, needs nothing from
    the template, and cannot land after the peak or before ``t_min`` (which keeps
    it out of the stimulus artifact).
    """
    if not np.isfinite(p1_lat) or not np.isfinite(noise_std) or noise_std <= 0:
        return np.nan
    lo = int(np.searchsorted(times, t_min, side="left")) if np.isfinite(t_min) else 0
    i1 = int(np.argmin(np.abs(times - p1_lat)))
    if i1 - lo < 5:
        return np.nan

    ww = max(1, int((sustain_ms / 1000.0) * sfreq))
    rect_s = _smoothed_deviation(sig_f, bmean, ww)[lo:i1]
    below = np.where(rect_s <= k * noise_std)[0]
    return float(times[lo + below[-1]]) if below.size else np.nan


def small_misshapen_responses(
    sigs: np.ndarray,
    times: np.ndarray,
    p1_lats: np.ndarray,
    amps: np.ndarray,
    win_ms: float = 6.0,
    rel_amp_max: float = 0.4,
    min_corr: float = 0.5,
    min_snr: float = 8.0,
    noise_after_s: float = 0.04,
    min_ref: int = 3,
) -> np.ndarray:
    """Which of a channel's accepted curves are noise wearing a peak.

    Amplitude thresholds cannot tell a small response from a lucky bump: both
    clear them. Shape can, because a muscle's response on one recording is
    stereotyped — so each curve is compared against what this channel's OWN
    strongest curves look like, around the latency where its responses live. No
    generic shape is imposed; a channel whose response is unusual is judged
    against its own.

    Being small is what makes a curve suspect; it is not what condemns it. Only
    curves under ``rel_amp_max`` of their channel's median are judged at all —
    the foot of a recruitment curve is small and real, and so is a channel that
    simply responds weakly. What condemns a small curve is failing to justify
    itself EITHER by shape OR by standing clear of its own noise.

    Both escape routes are needed. Shape alone lets through bumps that correlate
    at 0.5-0.7 by accident while sitting four noise SDs high — two such on one
    channel here, 5 and 7 uV against that channel's 80. Noise alone cannot be
    used as a global rule: a noisy channel's LARGEST responses can sit at six
    SDs, and cutting on that would delete the very responses being measured.

    ``sigs`` is (n_curves, n_samples) for one channel, ``amps`` its per-curve
    response sizes. Noise is measured per curve, after ``noise_after_s``.
    Returns a boolean mask of curves to drop.
    """
    out = np.zeros(len(sigs), dtype=bool)
    ok = np.isfinite(amps) & np.isfinite(p1_lats)
    if int(np.sum(ok)) < min_ref + 1:
        return out

    centre = float(np.median(p1_lats[ok]))
    w = (times >= centre - win_ms / 1e3) & (times <= centre + win_ms / 1e3)
    if int(np.sum(w)) < 5:
        return out

    idx = np.flatnonzero(ok)
    segs = np.asarray([sigs[i][w] for i in idx], dtype=float)
    segs = segs - segs.mean(axis=1, keepdims=True)

    strongest = idx[np.argsort(amps[idx])[-max(min_ref, len(idx) // 4):]]
    ref = np.mean([sigs[i][w] for i in strongest], axis=0)
    ref = ref - ref.mean()
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0:
        return out

    norms = np.linalg.norm(segs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (segs @ ref) / (norms * ref_norm)
    corr[~np.isfinite(corr)] = 0.0

    rel = amps[idx] / max(float(np.median(amps[idx])), 1e-12)
    snr = np.array([
        amps[i] / max(noise_std_from_tail(sigs[i], times, noise_after_s), 1e-12)
        for i in idx
    ], dtype=float)
    snr[~np.isfinite(snr)] = np.inf          # no noise estimate: do not condemn
    out[idx] = (rel < rel_amp_max) & ((corr < min_corr) | (snr < min_snr))
    return out


def onsets_anchored_to_channel(
    sigs: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    onsets: np.ndarray,
    p1_lats: np.ndarray,
    amps: np.ndarray,
    win_ms: float = 2.0,
    k: float = 2.5,
    sustain_ms: float = 2.0,
    noise_after_s: float = 0.04,
    min_ref: int = 3,
) -> np.ndarray:
    """Re-place each curve's onset near where this channel's onset actually is.

    Found curve by curve, an onset is not a latency: it is where the response
    rose above THAT curve's noise, which moves with the response's size. On one
    channel here P1 sits at 25.8 ms with a spread of 0.24 ms across 42 curves,
    while the onset jumps between 17-19 and 22-24 ms — because an early component
    at 17 ms clears the threshold on the strong curves and is buried in noise on
    the weak ones. Nothing about the pathway changed; only the SNR did.

    So the onset-to-P1 interval is measured once, on the curves where the
    threshold crossing is trustworthy (the strongest quarter), and each curve's
    onset is then looked for within ``win_ms`` of its own P1 minus that interval.
    A curve that shows no crossing there is given the interval outright: at that
    point the channel's latency is far better determined than one weak curve's
    threshold crossing.

    Anchoring to P1 rather than to a fixed latency matters — a channel can carry
    two response latencies at once (two stimulus intensities in one file), and
    each curve keeps its own.

    Returns the refined onsets; the input is returned unchanged when the channel
    has too few usable ones to measure an interval from.
    """
    out = np.array(onsets, dtype=float, copy=True)
    ok = np.isfinite(onsets) & np.isfinite(p1_lats) & np.isfinite(amps)
    if int(np.sum(ok)) < min_ref + 1:
        return out

    idx = np.flatnonzero(ok)
    gaps = (p1_lats - onsets)[idx]
    strongest = np.argsort(amps[idx])[-max(min_ref, len(idx) // 4):]
    interval = float(np.median(gaps[strongest]))
    if not np.isfinite(interval) or interval <= 0:
        return out

    ww = max(1, int((sustain_ms / 1000.0) * sfreq))
    w = win_ms / 1000.0
    for j in range(len(p1_lats)):
        if not np.isfinite(p1_lats[j]):
            out[j] = np.nan
            continue
        target = p1_lats[j] - interval
        sd = noise_std_from_tail(sigs[j], times, noise_after_s)
        out[j] = target
        if not np.isfinite(sd) or sd <= 0:
            continue
        bmean = float(np.mean(sigs[j][baseline_mask]))
        rect_s = _smoothed_deviation(sigs[j], bmean, ww)
        m = (times >= target - w) & (times <= target + w) & (times < p1_lats[j])
        below = np.flatnonzero(m & (rect_s <= k * sd))
        if below.size:
            out[j] = float(times[below[-1]])
    return out


def noise_std_from_tail(
    sig_f: np.ndarray,
    times: np.ndarray,
    after_s: float,
    sub_win_s: float = 0.02,
    min_samples: int = 100,
) -> float:
    """Noise scale from the quietest stretch of curve that follows the response.

    For recordings that carry no pre-stimulus data this is the only place left to
    measure noise. MAD rather than SD, and returned as an SD equivalent so the
    same ``k`` works as with a baseline SD.

    Taken over the QUIETEST sub-window rather than the whole tail, because the
    tail is not always silent: late activity and movement sit there too, and MAD
    over the whole stretch then reports the activity instead of the noise floor.
    That is not a rare case — it doubled the estimate on exactly the low-amplitude
    channels that need it most (7.4 against 4.0 uV on one channel whose responses
    peak at 27 uV), which pushed the onset threshold to two thirds of the response
    height and left the onset almost on top of the peak. Strong channels are
    barely affected either way.
    """
    m = times >= after_s
    if int(np.sum(m)) < min_samples:
        return np.nan
    tail = sig_f[m]
    n = len(tail)
    w = max(min_samples, int(round(sub_win_s * float(len(times)) / (times[-1] - times[0]))))
    step = max(1, w // 2)
    starts = range(0, max(1, n - w + 1), step)
    mads = [float(np.median(np.abs(seg - np.median(seg))))
            for seg in (tail[s:s + w] for s in starts) if len(seg) >= min_samples]
    mad = min(mads) if mads else float(np.median(np.abs(tail - np.median(tail))))
    return 1.4826 * mad if mad > 0 else np.nan


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

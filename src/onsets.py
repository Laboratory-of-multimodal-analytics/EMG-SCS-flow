"""Response onsets on pre-cut curve exports (Neurosoft), one per curve.

Why this replaced the old onset pass
------------------------------------
The previous pass (``detection.onsets_anchored_to_channel``) measured one
onset-to-P1 interval per channel and placed every curve's onset near P1 minus
that interval, taking the interval outright when a curve showed no threshold
crossing there. Hand markers were cruder still: ``sir_review._measure_at`` wrote
the marker's latency into every curve, so a hand-marked channel had one onset for
all of its curves (852 channel-runs in the processed dataset, September 2026), and
the H-reflex path ignored hand onsets altogether. On automatic channels the
threshold crossing (k x a noise SD taken from the tail, against the pad mean)
landed on slow drift or on the recovery from the artifact well before the
response, or up its flank, and it was missing on 4% of detections — on every
curve of channels whose response follows the artifact within a few ms.

What an onset is here
---------------------
Where the response leaves the pre-response isoline, read off each curve:

1. Coarse: walk back from the steepest point of the flank leading into P1 until
   the slope stays low for a sustained stretch. "Low" scales with that flank's own
   maximum slope, so slow drift counts as isoline; the sustain bridges the brief
   flat top of a leading opposite phase.
2. Knee: a continuous two-segment (broken-stick) fit around the coarse point. The
   breakpoint is accepted when the segment after it departs clearly faster than
   the one before, by more than the noise, and does not sit on the edge of the
   search range (then the range is widened once).
3. Foot: from the knee, walk back while the curve still stands off a straight
   isoline fitted just before it, by more than 3 SD of the curve around that line
   and more than 2% of the response. The knee of a smooth rise sits up its flank;
   this puts the onset where the departure is actually visible.
4. With no quiet stretch between the artifact and the flank (a response right
   behind the stimulus), the knee is searched over that whole stretch.

Per channel, a strong curve (response >= 25% of the channel median) keeps its own
onset unless it lies more than 2 ms from the median onset of the strong curves
whose P1 is within 1.5 ms of its own. Weak curves and such outliers are re-fitted
near that reference and take the reference only when no knee is found there.
Matching on P1 keeps two response latencies on one channel apart.

Hand-placed onset
-----------------
The clinician's marker stays the channel's level; each curve gets its own small
shift around it. The shift is how far that curve's waveform around the marker
(1 ms before to 3 ms after) is displaced in time against the median of the
channel's strong curves, by cross-correlation within +-1 ms, taken relative to
the channel's median shift — so a marker placed on a slope, where there is no
knee to find, still gets per-curve timing. A curve whose own P1 comes before the
marker + 1 ms (a reflex whose latency moved with intensity) is anchored at its P1
minus the hand onset-to-P1 interval instead. Weak curves, curves that do not
correlate with the channel (r < 0.5), and markers placed before the response
window take the anchor unchanged.

A marker is trusted only while it agrees with what an onset is. It is checked
first, on the channel's strong curves, against the automatic onset of each:

* on the flank — the marker lies more than 0.5 ms after the curve's own onset and
  the curve has already travelled at least 15% of the way to P1 by then. The
  (беляев) Т12-Л1 КР ch2 marker at 20.5 ms sat 30% down a descent in a response
  that begins at 16.5 ms. This holds for a marker at the start of the main phase
  too, once the curve is already going down there: an onset on the descent is not
  an onset, whatever came before it (Patient Sh Т12-Л1 ендр ch2, 29.5 ms — tried
  as an exception, rejected on review 13.09.2026);
* ahead of the response — the marker lies more than 3 ms before the curve's own
  onset with nothing but isoline in between: once a straight trend is taken out,
  nothing there exceeds 4 noise SD or 15% of the response. The noise SD alone is
  too strict a yardstick — the isoline ahead of a response wanders by several SD
  of the quiet tail (Контроль Азат Th11-12 RC R upright ch1: 3-38 SD, yet under
  15% of the response on most curves, marker at 5.5 ms, response from ~16 ms).

When either holds on most strong curves the marker is set aside and the channel
gets automatic onsets, flagged with the reason. A marker on genuine early
activity (small oscillations before the main response) is not flat isoline and
is kept.

Units: the public functions take seconds and volts, like the rest of the
pipeline; the search itself runs in ms and uV.
"""

from __future__ import annotations

import numpy as np

SMOOTH_MS = 0.5
STEEP_SEARCH_MS = 8.0
QUIET_SUSTAIN_MS = 1.5
QUIET_K_NOISE = 4.0
QUIET_FRAC_MAXSLOPE = 0.08
KNEE_BEFORE_MS = 2.0
KNEE_AFTER_MS = 1.5
KNEE_RANGE_MS = 1.0
KNEE_MIN_DEPART_K = 3.0
KNEE_MIN_DEPART_FRAC = 0.05
FOOT_MAX_MS = 2.0
FOOT_K_NOISE = 3.0
FOOT_MIN_FRAC = 0.02
FOOT_ISOLINE_MS = (3.5, 1.5)   # isoline fitted over [knee - 3.5, knee - 1.5] ms
ARTIFACT_GUARD_MS = 0.5
WEAK_FRAC = 0.25
NEIGHBOUR_P1_MS = 1.5
OUTLIER_MS = 2.0
REFIT_MS = 1.5
HAND_JITTER_MS = 1.0
HAND_SEG_MS = (1.0, 3.0)       # waveform compared around the marker: 1 ms before, 3 ms after
HAND_MIN_CORR = 0.5
HAND_MIN_BEFORE_P1_MS = 1.0
HAND_LATE_MS = 0.5             # marker after the curve's own onset by more than this ...
HAND_FLANK_FRAC = 0.15         # ... with the curve already this far on its way to P1
HAND_EARLY_MS = 3.0            # marker before the curve's own onset by more than this ...
HAND_QUIET_K = 4.0             # ... over isoline quieter than this many noise SDs
HAND_QUIET_FRAC = 0.15         # ... or than this share of the response, whichever is larger

#: How each onset was obtained — returned alongside the values, for the record.
FLAGS = {
    "fine": "knee + foot on the curve's own flank",
    "coarse": "end of the quiet stretch (no clear knee)",
    "knee_no_isoline": "knee between the artifact and the flank (no quiet stretch)",
    "refit": "re-fitted near the channel reference (weak curve or outlier)",
    "reference": "channel reference taken outright (no knee near it)",
    "hand_jitter": "hand onset + this curve's own time shift around the marker (<= 1 ms)",
    "hand": "hand onset unchanged (weak curve, poor correlation, or marker before the window)",
    "hand_p1_jitter": "P1 - hand interval + this curve's shift (its P1 comes before the marker)",
    "hand_p1": "P1 - hand interval (its P1 comes before the marker)",
    "…|hand_on_flank": "automatic onset: the hand marker sat on the response flank",
    "…|hand_before_response": "automatic onset: the hand marker sat on flat isoline ahead of the response",
    "too_close": "P1 too close to the artifact for an onset",
    "none": "no onset",
}


def _movmean(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average with edge padding (zero padding puts fake dips at the ends)."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    pad_l = w // 2
    return np.convolve(np.pad(x, (pad_l, w - 1 - pad_l), mode="edge"), np.ones(w) / w, mode="valid")


def _robust_sd(x: np.ndarray, t: np.ndarray, t_from: float, win_ms: float = 10.0) -> float:
    """Robust SD over the quietest window after ``t_from`` (else the whole post-artifact curve)."""
    dt = t[1] - t[0]
    w = max(20, int(round(win_ms / dt)))
    seg = x[t >= t_from]
    if seg.size < w:
        seg = x[t >= 2.5]
    if seg.size < w:
        return np.nan
    mads = [np.median(np.abs(seg[s:s + w] - np.median(seg[s:s + w])))
            for s in range(0, seg.size - w + 1, max(1, w // 2))]
    m = min(mads)
    return 1.4826 * m if m > 0 else np.nan


def _broken_stick(t: np.ndarray, y: np.ndarray, k_lo: int, k_hi: int):
    """Continuous two-segment least-squares fit; the best breakpoint in [k_lo, k_hi].

    Returns (index, slope before, slope after, index range actually searched).
    """
    n = len(t)
    ks = np.arange(max(k_lo, 2), min(k_hi, n - 3) + 1)
    if ks.size == 0:
        return None
    tk = t[ks][:, None]
    tt = t[None, :]
    X = np.stack([np.ones((ks.size, n)), np.minimum(tt - tk, 0.0), np.maximum(tt - tk, 0.0)], axis=2)
    XtX = np.einsum("kni,knj->kij", X, X) + 1e-9 * np.eye(3)
    beta = np.linalg.solve(XtX, np.einsum("kni,n->ki", X, y)[..., None])[..., 0]
    sse = ((y[None, :] - np.einsum("kni,ki->kn", X, beta)) ** 2).sum(1)
    j = int(np.argmin(sse))
    return int(ks[j]), float(beta[j, 1]), float(beta[j, 2]), (int(ks[0]), int(ks[-1]))


class _Curve:
    """One curve prepared for the onset search (ms, uV)."""

    def __init__(self, x, t, p1, pv1, t_lo, noise_from):
        self.t = t
        self.dt = t[1] - t[0]
        self.xs = _movmean(x, max(1, int(round(SMOOTH_MS / self.dt))))
        self.lo = int(np.searchsorted(t, t_lo + ARTIFACT_GUARD_MS))
        self.i1 = int(np.argmin(np.abs(t - p1)))
        self.p1 = p1
        self.pol = 1 if pv1 >= 0 else -1
        self.noise_from = noise_from
        sd = _robust_sd(x, t, noise_from)
        self.sd = sd
        self.min_depart = max(KNEE_MIN_DEPART_K * sd if np.isfinite(sd) else 0.0,
                              KNEE_MIN_DEPART_FRAC * abs(pv1))

    def knee(self, centre, rng_ms, hi_idx=None, widen=True):
        """Knee within ``centre +- rng_ms``, before P1. None when there is none.

        A breakpoint on the edge of the range means the knee lies outside it:
        the range is widened once, and an edge hit after that is no knee.
        """
        t, dt = self.t, self.dt
        hi_idx = self.i1 if hi_idx is None else hi_idx
        a = max(self.lo, int(round((centre - rng_ms - KNEE_BEFORE_MS - t[0]) / dt)))
        b = min(hi_idx, int(round((centre + rng_ms + KNEE_AFTER_MS - t[0]) / dt)))
        if b - a < 10:
            return None
        tt, yy = t[a:b + 1], self.xs[a:b + 1]
        res = _broken_stick(tt, yy, int(round((centre - rng_ms - t[a]) / dt)),
                            int(round((centre + rng_ms - t[a]) / dt)))
        if res is None:
            return None
        k, b1, b2, (k_first, k_last) = res
        if k <= k_first or k >= k_last:
            return self.knee(centre, 2.0 * rng_ms, hi_idx, widen=False) if widen else None
        after = min(KNEE_AFTER_MS, tt[-1] - tt[k])
        if abs(b2) < 2.0 * abs(b1) or abs(b2) * after < self.min_depart:
            return None
        return float(tt[k]) if tt[k] < self.p1 else None

    def foot(self, knee_ms):
        """Walk back from the knee to where the curve leaves its local isoline.

        The isoline is a straight line fitted just before the knee and extended
        forward, so slow drift and artifact recovery belong to it rather than to
        the response. The noise band is the curve's spread around that line,
        measured where the onset is, not in the tail.
        """
        t, dt, xs = self.t, self.dt, self.xs
        k = int(round((knee_ms - t[0]) / dt))
        b0 = max(self.lo, int(round((knee_ms - FOOT_ISOLINE_MS[0] - t[0]) / dt)))
        b1 = max(self.lo, int(round((knee_ms - FOOT_ISOLINE_MS[1] - t[0]) / dt)))
        if b1 - b0 < 10:
            return knee_ms
        coef = np.polyfit(t[b0:b1], xs[b0:b1], 1)
        resid = xs[b0:b1] - np.polyval(coef, t[b0:b1])
        sd = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(sd) or sd <= 0:
            return knee_ms
        line = np.polyval(coef, t)
        ia = min(k + int(round(1.0 / dt)), len(xs) - 1)
        direction = 1.0 if xs[ia] - line[ia] >= 0 else -1.0
        dev = direction * (xs - line)
        thr = max(FOOT_K_NOISE * sd, FOOT_MIN_FRAC * abs(xs[self.i1] - line[self.i1]))
        stop = max(b1, k - int(round(FOOT_MAX_MS / dt)))
        j = k
        while j - 1 >= stop and dev[j - 1] > thr and dev[j - 1] <= dev[j]:
            j -= 1
        return float(t[j])

    def auto(self):
        t, dt = self.t, self.dt
        a = max(self.lo, self.i1 - int(round(STEEP_SEARCH_MS / dt)))
        if self.i1 - a < 3:
            return np.nan, "too_close"
        d = np.gradient(self.xs, dt)
        # the steepest point of the flank that LEADS INTO P1 (signed), clear of
        # the artifact edge — the step off the station's pad is the steepest
        # thing on the curve and would otherwise be taken for the response
        i_st = a + int(np.argmax(self.pol * d[a:self.i1 + 1]))
        sd_d = _robust_sd(d, t, self.noise_from)
        thr = max(QUIET_K_NOISE * sd_d if np.isfinite(sd_d) else 0.0,
                  QUIET_FRAC_MAXSLOPE * abs(d[i_st]))
        sustain = max(2, int(round(QUIET_SUSTAIN_MS / dt)))
        quiet = np.abs(d) <= thr
        coarse, run, j = None, 0, i_st
        while j >= self.lo:
            run = run + 1 if quiet[j] else 0
            if run >= sustain:
                coarse = j + sustain - 1
                break
            j -= 1
        if coarse is None:
            mid = 0.5 * (t[self.lo] + t[i_st])
            k = self.knee(mid, 0.5 * (t[i_st] - t[self.lo]), hi_idx=i_st, widen=False)
            return (self.foot(k), "knee_no_isoline") if k is not None else (np.nan, "none")
        k = self.knee(float(t[coarse]), KNEE_RANGE_MS, hi_idx=i_st)
        return (self.foot(k), "fine") if k is not None else (float(t[coarse]), "coarse")


def _hand_shifts(curves: dict, anchor: dict, strong: set, dt: float,
                 min_signal_k: float = 4.0) -> dict:
    """Per-curve time shift (ms) of the waveform around the hand marker.

    Each strong curve's segment around its anchor is cross-correlated, within
    +-HAND_JITTER_MS, against the median segment of the strong curves.

    Only when there is a waveform there to time: if the median segment spans less
    than ``min_signal_k`` noise SDs, the marker sits on a flat isoline (some are
    placed ms ahead of the response) and a "shift" would be the lag of noise
    against noise — no shifts are returned and the marker is kept as placed.
    """
    nb, na = int(round(HAND_SEG_MS[0] / dt)), int(round(HAND_SEG_MS[1] / dt))
    lag_max = int(round(HAND_JITTER_MS / dt))
    idx = {}
    for j in strong:
        c = curves[j]
        i = int(round((anchor[j] - c.t[0]) / dt))
        if i - nb - lag_max >= c.lo and i + na + lag_max < len(c.xs):
            idx[j] = i
    if len(idx) < 3:
        return {}
    segs = np.array([curves[j].xs[i - nb:i + na] for j, i in idx.items()])
    tmpl = np.median(segs - segs.mean(axis=1, keepdims=True), axis=0)
    tmpl -= tmpl.mean()
    tn = float(np.linalg.norm(tmpl))
    noise = np.nanmedian([_robust_sd(curves[j].xs, curves[j].t, curves[j].noise_from) for j in idx])
    if tn == 0 or (np.isfinite(noise) and np.ptp(tmpl) < min_signal_k * noise):
        return {}
    out = {}
    for j, i in idx.items():
        xs = curves[j].xs
        best_lag, best_r = 0, -np.inf
        for lag in range(-lag_max, lag_max + 1):
            s = xs[i - nb + lag:i + na + lag]
            s = s - s.mean()
            sn = float(np.linalg.norm(s))
            if sn == 0:
                continue
            r = float(s @ tmpl) / (sn * tn)
            if r > best_r:
                best_r, best_lag = r, lag
        if best_r >= HAND_MIN_CORR:
            out[j] = best_lag * dt
    return out


def _hand_marker_verdict(curves: dict, anchor: dict, strong: set, dt: float):
    """Whether a hand marker contradicts what an onset is. None = it does not.

    Judged on the strong curves against each one's automatic onset; see the
    module docstring for the two ways a marker can be wrong.
    """
    flank, early, judged = 0, 0, 0
    for j in strong:
        c = curves[j]
        auto, _ = c.auto()
        if not np.isfinite(auto):
            continue
        judged += 1
        t, xs = c.t, c.xs
        i_a = int(round((auto - t[0]) / dt))
        i_h = int(round((anchor[j] - t[0]) / dt))
        base = float(np.median(xs[max(c.lo, i_a - int(round(2.0 / dt))):i_a + 1]))
        reach = abs(xs[c.i1] - base)
        if anchor[j] > auto + HAND_LATE_MS and reach > 0:
            if c.pol * (xs[min(i_h, c.i1)] - base) / reach >= HAND_FLANK_FRAC:
                flank += 1
        elif anchor[j] < auto - HAND_EARLY_MS:
            a, b = max(c.lo, i_h), max(c.lo, i_a - int(round(0.5 / dt)))
            if b - a >= 5:
                seg_t, seg = t[a:b + 1], xs[a:b + 1]
                resid = seg - np.polyval(np.polyfit(seg_t, seg, 1), seg_t)
                limit = max(HAND_QUIET_K * c.sd if np.isfinite(c.sd) else 0.0,
                            HAND_QUIET_FRAC * reach)
                if np.max(np.abs(resid)) <= limit:
                    early += 1
    if judged < 3:
        return None            # nothing to judge against: the marker stands
    if flank > judged / 2:
        return "hand_on_flank"
    if early > judged / 2:
        return "hand_before_response"
    return None


def _auto_onsets(curves: dict, det, strong: set, p1s, lo_arr, n, dt):
    on = np.full(n, np.nan)
    flag = np.array(["none"] * n, dtype=object)
    for j in det:
        on[j], flag[j] = curves[j].auto()
    pool_all = [j for j in strong if np.isfinite(on[j])]
    for j in det:
        near = [i for i in pool_all if i != j and abs(p1s[i] - p1s[j]) <= NEIGHBOUR_P1_MS]
        pool = near if len(near) >= 3 else [i for i in pool_all if i != j]
        if len(pool) < 3:
            continue
        ref = float(np.median(on[pool]))
        if j in strong and np.isfinite(on[j]) and abs(on[j] - ref) <= OUTLIER_MS:
            continue
        k = curves[j].knee(ref, REFIT_MS)
        if k is not None:
            on[j], flag[j] = curves[j].foot(k), "refit"
        elif lo_arr[j] <= ref < p1s[j]:
            on[j], flag[j] = ref, "reference"
    for j in det:
        if np.isfinite(on[j]) and (on[j] >= p1s[j] or on[j] < lo_arr[j] - dt):
            on[j], flag[j] = np.nan, "none"
    return on, flag


def _channel_onsets_ms(X, t, p1s, pv1s, amps, t_lo, hand, hand_p1, noise_from):
    n = len(p1s)
    on = np.full(n, np.nan)
    flag = np.array(["none"] * n, dtype=object)
    det = np.flatnonzero(np.isfinite(p1s) & np.isfinite(pv1s))
    if det.size == 0:
        return on, flag
    lo_arr = np.broadcast_to(np.asarray(t_lo, dtype=float), (n,))
    curves = {j: _Curve(X[j], t, p1s[j], pv1s[j], lo_arr[j], noise_from) for j in det}
    finite_amps = amps[det][np.isfinite(amps[det])]
    amp_med = float(np.median(finite_amps)) if finite_amps.size else np.nan
    strong = {j for j in det if np.isfinite(amps[j]) and np.isfinite(amp_med)
              and amps[j] >= WEAK_FRAC * amp_med}
    dt = t[1] - t[0]

    if hand is not None:
        interval = (hand_p1 - hand) if hand_p1 is not None and hand_p1 > hand else None
        anchor = {j: float(hand) if (hand < p1s[j] - HAND_MIN_BEFORE_P1_MS or interval is None)
                  else float(p1s[j] - interval) for j in det}
        verdict = _hand_marker_verdict(curves, anchor, strong, dt)
        if verdict is not None:
            on, flag = _auto_onsets(curves, det, strong, p1s, lo_arr, n, dt)
            for j in det:
                if flag[j] not in ("none", "too_close"):
                    flag[j] = f"{flag[j]}|{verdict}"
            return on, flag
        shifts = _hand_shifts(curves, anchor, strong, dt)
        centre = float(np.median(list(shifts.values()))) if len(shifts) >= 3 else np.nan
        for j in det:
            base_flag = "hand" if anchor[j] == hand else "hand_p1"
            # A marker placed before the response window (inside the artifact — a
            # few channels carry 0.5 ms) is kept exactly as placed: there is no
            # waveform there to take a shift from, and dropping it would lose an
            # onset the clinician chose.
            if j in shifts and np.isfinite(centre) and anchor[j] >= lo_arr[j]:
                shift = float(np.clip(shifts[j] - centre, -HAND_JITTER_MS, HAND_JITTER_MS))
                on[j] = max(float(lo_arr[j]), anchor[j] + shift)
                flag[j] = base_flag + "_jitter"
            else:
                on[j], flag[j] = anchor[j], base_flag
            if on[j] >= p1s[j]:
                on[j], flag[j] = np.nan, "none"
        return on, flag

    return _auto_onsets(curves, det, strong, p1s, lo_arr, n, dt)


def channel_onsets(
    sigs: np.ndarray,
    times: np.ndarray,
    p1_lat: np.ndarray,
    p1_val: np.ndarray,
    amps: np.ndarray,
    t_lo,
    hand_onset: float | None = None,
    hand_p1: float | None = None,
    noise_after: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Onset of every detected curve of one channel.

    ``sigs`` (n_curves, n_samples) in volts, baseline-corrected; ``times`` in s;
    ``p1_lat`` s and ``p1_val`` V per curve, NaN where the curve has no detection;
    ``amps`` V per curve (PTP, or |P1| without a P2). ``t_lo`` (s, scalar or per
    curve) is the earliest an onset may sit — the start of the response window,
    or the M-wave's rebound for a reflex. ``hand_onset`` / ``hand_p1`` (s) are the
    clinician's markers for this channel, if any. ``noise_after`` (s) is where
    the quiet tail used for the noise scale begins.

    Returns onsets in seconds (NaN without a detection) and a flag per curve
    (keys of ``FLAGS``).
    """
    t_ms = np.asarray(times, dtype=float) * 1e3
    lo = np.asarray(t_lo, dtype=float) * 1e3
    on, flag = _channel_onsets_ms(
        np.asarray(sigs, dtype=float) * 1e6, t_ms,
        np.asarray(p1_lat, dtype=float) * 1e3, np.asarray(p1_val, dtype=float) * 1e6,
        np.asarray(amps, dtype=float) * 1e6, lo,
        None if hand_onset is None or not np.isfinite(hand_onset) else float(hand_onset) * 1e3,
        None if hand_p1 is None or not np.isfinite(hand_p1) else float(hand_p1) * 1e3,
        float(noise_after) * 1e3,
    )
    return on / 1e3, flag


def hreflex_onsets(
    curves: np.ndarray,
    times: np.ndarray,
    comp: dict[str, dict[str, np.ndarray]],
    resp_tmin: float,
    resp_tmax: float,
    hand: dict | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Onsets of the M-wave and the H-reflex on every curve of one channel.

    ``comp`` is ``{"M": {"p1", "pv1", "p2", "ptp"}, "H": {...}}`` (s, V, one value
    per curve). The reflex's onset cannot come before its own curve's M-wave
    rebound (the channel median rebound where a curve has none). ``hand`` holds
    ``m_onset``, ``m_p1``, ``h_onset``, ``h_p1`` in seconds when the markers were
    placed by hand.
    """
    out = {}
    m_p2 = np.asarray(comp["M"]["p2"], dtype=float)
    for c in ("M", "H"):
        r = comp[c]
        pv1 = np.asarray(r["pv1"], dtype=float)
        ptp = np.asarray(r["ptp"], dtype=float)
        if c == "M":
            lo = resp_tmin
        else:
            fallback = float(np.nanmedian(m_p2)) if np.isfinite(m_p2).any() else resp_tmin
            lo = np.where(np.isfinite(m_p2), m_p2, fallback)
        key = c.lower()
        out[c] = channel_onsets(
            curves, times, r["p1"], pv1, np.where(np.isfinite(ptp), ptp, np.abs(pv1)), lo,
            hand_onset=(hand or {}).get(f"{key}_onset"), hand_p1=(hand or {}).get(f"{key}_p1"),
            noise_after=resp_tmax,
        )
    return out

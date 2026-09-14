"""Area of an evoked response: the rectified integral over all of its deflections.

Asked for by A. Militskova for polyphasic responses. Peak-to-peak reads only the
two largest phases, so a response that breaks into several smaller deflections
after its P1/P2 — the "шебурда" — measures the same as a clean biphasic one of
the same span. The area counts every phase.

Definition (agreed 13.09.2026)
------------------------------
* ``area = ∫ |x(t)| dt`` from the curve's onset to the end of its response —
  every phase adds, whatever its sign. Stored in V·s, reported in µV·ms.
* Reference: the curve's own zero — the drift-removed curve that P1 and P2 are
  measured from — so the area and the amplitudes share one baseline. (A level
  taken just before the onset is not usable: on these exports it sits inside the
  artifact's recovery or on a DC step that the response never returns to.)
* End of the response, per curve: the first moment after P2 (P1 when there is no
  P2) from which a 1 ms rectified envelope of the curve stays below
  ``max(3 noise SD, 5% of the response's PTP)`` for 3 ms. Capped at the end of
  the response window — 40 ms on single-response scenarios, the reflex's onset
  for an M-wave, the H-reflex window's end for the reflex. A response still
  running at the cap is integrated up to the cap.

Measured on 30 random runs (3767 curves) before adopting it: the per-curve end and
a fixed per-channel duration rank curves identically (Spearman 0.999), and the
area tracks PTP at 0.95 — it departs from PTP exactly on the polyphasic curves.
"""

from __future__ import annotations

import numpy as np

AREA_ENVELOPE_MS = 1.0
AREA_QUIET_MS = 3.0
AREA_K_NOISE = 3.0
AREA_FRAC_PTP = 0.05
UVMS_PER_VS = 1e9      # 1 V·s = 1e6 µV x 1e3 ms


def _noise_sd(x: np.ndarray, times: np.ndarray, t_from: float, win_s: float = 0.010) -> float:
    """Robust SD over the quietest window after ``t_from`` (else the post-artifact curve)."""
    dt = times[1] - times[0]
    w = max(20, int(round(win_s / dt)))
    seg = x[times >= t_from]
    if seg.size < w:
        seg = x[times >= 0.0025]
    if seg.size < w:
        return np.nan
    mads = [np.median(np.abs(seg[s:s + w] - np.median(seg[s:s + w])))
            for s in range(0, seg.size - w + 1, max(1, w // 2))]
    m = min(mads)
    return 1.4826 * m if m > 0 else np.nan


def response_end(x: np.ndarray, times: np.ndarray, t_from: float, t_cap: float,
                 amplitude: float, noise_after: float) -> float:
    """End of a response (s): where the curve settles at its zero for good.

    ``t_from`` is where the search starts (P2, or P1), ``t_cap`` the latest the
    response may end, ``amplitude`` its PTP (or |P1|), ``noise_after`` where the
    quiet tail used for the noise scale begins.
    """
    if not (np.isfinite(t_from) and np.isfinite(t_cap)) or t_cap <= t_from:
        return float(t_cap) if np.isfinite(t_cap) else np.nan
    dt = times[1] - times[0]
    w = max(1, int(round(AREA_ENVELOPE_MS * 1e-3 / dt)))
    pad_l = w // 2
    env = np.convolve(np.pad(np.abs(x), (pad_l, w - 1 - pad_l), mode="edge"),
                      np.ones(w) / w, mode="valid")
    sd = _noise_sd(x, times, noise_after)
    thr = max(AREA_K_NOISE * sd if np.isfinite(sd) else 0.0,
              AREA_FRAC_PTP * abs(amplitude) if np.isfinite(amplitude) else 0.0)
    quiet = max(2, int(round(AREA_QUIET_MS * 1e-3 / dt)))
    i0 = int(np.searchsorted(times, t_from))
    i_cap = int(np.searchsorted(times, t_cap, side="right")) - 1
    below = env <= thr
    run = 0
    for j in range(i0, i_cap + 1):
        run = run + 1 if below[j] else 0
        if run >= quiet:
            return float(times[j - quiet + 1])
    return float(times[i_cap])


def response_area(x: np.ndarray, times: np.ndarray, onset: float, end: float) -> float:
    """∫|x| dt between onset and end, in V·s (NaN without both)."""
    if not (np.isfinite(onset) and np.isfinite(end)) or end <= onset:
        return np.nan
    m = (times >= onset) & (times <= end)
    if int(m.sum()) < 2:
        return np.nan
    return float(np.trapz(np.abs(x[m]), times[m]))


def channel_areas(
    sigs: np.ndarray,
    times: np.ndarray,
    onsets: np.ndarray,
    p1_lat: np.ndarray,
    p2_lat: np.ndarray,
    p1_val: np.ndarray,
    ptp: np.ndarray,
    t_cap,
    noise_after: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Area (V·s) and end (s) of every detected curve of one channel.

    ``sigs`` (n_curves, n_samples) in volts; the rest in s / V, one value per
    curve, NaN where the curve has no detection. ``t_cap`` is scalar or per curve.
    """
    n = len(onsets)
    areas = np.full(n, np.nan)
    ends = np.full(n, np.nan)
    caps = np.broadcast_to(np.asarray(t_cap, dtype=float), (n,))
    for j in range(n):
        if not (np.isfinite(onsets[j]) and np.isfinite(p1_lat[j])):
            continue
        start = p2_lat[j] if np.isfinite(p2_lat[j]) else p1_lat[j]
        amp = ptp[j] if np.isfinite(ptp[j]) else abs(p1_val[j])
        cap = caps[j] if np.isfinite(caps[j]) else float(times[-1])
        ends[j] = response_end(sigs[j], times, start, cap, amp, noise_after)
        areas[j] = response_area(sigs[j], times, onsets[j], ends[j])
    return areas, ends

"""Slow-drift removal for pre-cut curve epochs.

These epochs are 100 ms long at 20 kHz, and a high-pass filter is not an option
at that length: a 10 Hz FIR cutoff needs a filter far longer than the epoch, and
an IIR high-pass rings for tens of milliseconds after a stimulus artifact that is
two orders of magnitude taller than the response — it would manufacture exactly
the kind of deflection the detector is looking for.

So the drift is estimated and subtracted instead. A running MEDIAN over a window
several times wider than any muscle response tracks the slow recovery while
stepping over the response itself (a median is unmoved by a bump occupying less
than half its window). The estimate is computed on a decimated copy and
interpolated back, which makes it cheap and, because the decimation is itself a
smoothing step, slightly more robust.

The stimulus artifact is excluded from the estimate and its samples are left
untouched: it is not drift, the detector never looks there, and letting it into
the median would drag the baseline for the first few milliseconds after it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def estimate_drift(
    x: np.ndarray,
    sfreq: float,
    win_ms: float = 25.0,
    skip_s: float = 0.002,
    decim: int = 20,
) -> np.ndarray:
    """Slow baseline of ``x`` (…, n_samples), same shape.

    ``skip_s`` is the head of the epoch holding the stimulus artifact: it is
    replaced by the first post-artifact sample before the median runs, so the
    artifact cannot pull the estimate.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    i0 = min(int(round(skip_s * sfreq)), max(n - 2, 0))

    flat = x.reshape(-1, n).copy()
    if i0 > 0:
        flat[:, :i0] = flat[:, [i0]]

    step = max(int(decim), 1)
    small = flat[:, ::step]
    win = max(int(round(win_ms / 1000.0 * sfreq / step)), 3)
    if win % 2 == 0:
        win += 1
    if win >= small.shape[1]:
        win = small.shape[1] - (1 - small.shape[1] % 2)
    drift_small = median_filter(small, size=(1, win), mode="nearest")

    xp = np.arange(small.shape[1]) * step
    xi = np.arange(n)
    drift = np.stack([np.interp(xi, xp, row) for row in drift_small])
    return drift.reshape(x.shape)


def remove_drift(
    data: np.ndarray,
    sfreq: float,
    win_ms: float | None = 25.0,
    skip_s: float = 0.002,
) -> np.ndarray:
    """Subtract the slow baseline from every epoch/channel of *data*.

    ``data`` is ``(n_epochs, n_channels, n_samples)``. The artifact head
    (``skip_s``) keeps its SHAPE — the spike is not flattened, so it stays
    available for alignment and QC — but it is shifted by the same constant as
    the first corrected sample.

    That shift matters more than it looks. The head holds the station's constant
    pre-artifact pad, and the pipeline takes its mean as the epoch's baseline.
    Leaving the head at its raw level while the rest of the curve is pulled to
    zero leaves the two on different references, and the baseline subtraction
    then displaces every response by the pad's stale offset — a median of 1.6 mV
    and up to 7.4 mV on Zh14 Т11-12 двойная стим, which silently flipped
    responses out of their detector's polarity and lost whole channels.
    """
    if win_ms is None:
        return data
    data = np.asarray(data, dtype=float)
    drift = estimate_drift(data, sfreq, win_ms=win_ms, skip_s=skip_s)
    i0 = min(int(round(skip_s * sfreq)), data.shape[-1])
    out = data - drift
    if i0 > 0:
        # Bring the head to the same reference as the corrected rest. After the
        # drift is subtracted the signal sits at zero, so the pad has to sit at
        # zero too; the pad is constant, so its own first sample is its level.
        # Subtracting it leaves the artifact's SHAPE untouched.
        out[..., :i0] = data[..., :i0] - data[..., [0]]
    return out

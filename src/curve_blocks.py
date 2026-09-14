"""Contiguous blocks of curves, and the spread rule, for the Jendrassik manoeuvre.

A. Militskova (meeting of 13 September 2026): a Jendrassik recording is run as a
few stimuli without the manoeuvre, a few with it, then another intensity and the
same again. How many stimuli a block has depends on the patient (a stronger one
holds the manoeuvre longer), and an intensity that already showed an effect was
often not followed by the next one; nothing in a Neurosoft export records any of
it. What does hold is her two criteria for curves that belong together: similar
amplitude, and NEXT TO EACH OTHER in the run. A curve whose amplitude drops sharply
inside a block stays in that block.

Clustering by amplitude alone ignored the second criterion and, on these files,
left 78% of the channels with groups scattered across the run. So the curves are
cut into contiguous blocks instead:

1. three or more consecutive curves without a response are a block of their own;
   a shorter gap stays inside its neighbours' block;
2. each responding stretch is split by optimal segmentation of the log amplitude
   (least squares, dynamic programming). A split has to pay
   ``PENALTY_K · σ² · ln(n)``, σ² being the channel's scatter around its blocks,
   and every block has at least ``MIN_BLOCK`` curves. Single outliers are damped by a
   Hampel filter for the split only; every statistic uses the raw values;
3. neighbouring blocks are merged when their means differ by less than the spread
   threshold or by less than ``MERGE_SE`` standard errors: a step smaller than the
   scatter is not a different condition.

The verdict is her rule. The spread — the SD of the response amplitude within a
block, the band drawn around the block's mean — above ``SPREAD_THRESHOLD_UV`` means
the manoeuvre works: it modulates excitability, so the amplitude swings. Every
block below it means it does not. A run that starts silent and then responds at a
steady level is a change of intensity, not an effect of the manoeuvre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A. Militskova: a spread above 30 µV means the manoeuvre works.
SPREAD_THRESHOLD_UV = 30.0
#: Fewest curves a block can have (a block of one has no spread, and is what an outlier makes).
MIN_BLOCK = 3
#: Price of a split, in units of σ² · ln(n).
PENALTY_K = 5.0
#: Neighbouring blocks closer than this many standard errors of their difference are one block.
MERGE_SE = 3.0

RESPONSE, SILENT = "response", "no response"

VERDICT_WORKS = "manoeuvre works"
VERDICT_NOT = "manoeuvre does not work"
VERDICT_APPEARED = "response appeared, no manoeuvre effect"
VERDICT_NONE = "no response"
VERDICT_FEW = "too few responses to judge"
VERDICT_RU = {
    VERDICT_WORKS: "приём работает",
    VERDICT_NOT: "приём не работает",
    VERDICT_APPEARED: "ответ появился, эффекта приёма нет",
    VERDICT_NONE: "нет ответа",
    VERDICT_FEW: "слишком мало ответов для оценки",
}


@dataclass
class Block:
    """One contiguous stretch of curves. ``start``/``stop`` index the channel's sequence."""
    start: int
    stop: int
    kind: str
    label: str = ""
    curves: list[int] = field(default_factory=list)
    values: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_response(self) -> int:
        return int(self.values.size)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if self.values.size else np.nan

    @property
    def sd(self) -> float:
        return float(self.values.std(ddof=1)) if self.values.size > 1 else np.nan

    @property
    def cv(self) -> float:
        return self.sd / self.mean if self.values.size > 1 and self.mean > 0 else np.nan

    @property
    def spread_above(self) -> bool:
        return bool(np.isfinite(self.sd) and self.sd > SPREAD_THRESHOLD_UV)


def curve_span(curves) -> str:
    """Compact "1-8, 15" summary of a set of curve numbers."""
    c = sorted(int(x) for x in curves)
    if not c:
        return ""
    spans, start, prev = [], c[0], c[0]
    for v in c[1:]:
        if v == prev + 1:
            prev = v
            continue
        spans.append((start, prev))
        start = prev = v
    spans.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def _hampel(y: np.ndarray, half: int = 2, k: float = 3.0) -> np.ndarray:
    """A point far from the median of its neighbours (a lone dip or spike) is set to that median."""
    out = y.copy()
    for i in range(len(y)):
        w = np.concatenate([y[max(0, i - half):i], y[i + 1:i + half + 1]])
        if w.size < 2:
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if abs(y[i] - med) > k * max(mad, 0.05):
            out[i] = med
    return out


def _segment(y: np.ndarray, min_len: int, penalty: float) -> list[tuple[int, int]]:
    """Least-squares split of *y* into contiguous pieces of at least *min_len*, each costing *penalty*."""
    n = len(y)
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y ** 2)])

    def cost(i: int, j: int) -> float:
        return (c2[j] - c2[i]) - (c1[j] - c1[i]) ** 2 / (j - i)

    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        for i in [0, *range(min_len, j - min_len + 1)]:
            v = best[i] + cost(i, j) + penalty
            if v < best[j]:
                best[j], prev[j] = v, i
    if not np.isfinite(best[n]):
        return [(0, n)]
    pieces, j = [], n
    while j > 0:
        pieces.append((int(prev[j]), j))
        j = prev[j]
    return pieces[::-1]


def find_blocks(amplitudes, curves=None, threshold_uv: float = SPREAD_THRESHOLD_UV,
                min_block: int = MIN_BLOCK, penalty_k: float = PENALTY_K,
                merge_se: float = MERGE_SE) -> list[Block]:
    """Cut one channel's run of curves into contiguous blocks.

    ``amplitudes`` holds every curve of the channel in run order, NaN where no
    response was detected; ``curves`` their numbers (default 1..n). Returns the
    blocks in run order, labelled ``B1``, ``B2``, … — silent ones included.
    """
    a = np.asarray(amplitudes, dtype=float)
    n = len(a)
    curves = list(range(1, n + 1)) if curves is None else [int(c) for c in curves]
    if n == 0:
        return []
    resp = np.isfinite(a) & (a > 0)
    if not resp.any():
        return _finish([Block(0, n, SILENT)], a, curves)

    # 1. silent stretches of min_block or more; everything else is responding
    stretches: list[list] = []
    i = 0
    while i < n:
        j = i
        while j < n and resp[j] == resp[i]:
            j += 1
        kind = SILENT if (not resp[i] and j - i >= min_block) else RESPONSE
        if kind == RESPONSE and stretches and stretches[-1][2] == RESPONSE:
            stretches[-1][1] = j
        else:
            stretches.append([i, j, kind])
        i = j

    # 2. the channel's noise on the (outlier-damped) log amplitude. Curve-to-curve
    #    differences also carry the steps between blocks — the very thing looked
    #    for — and inflate the noise until clear steps go uncut (А1 Th11-12 JM ch3:
    #    ~380 µV on curves 1-5, ~100 µV on 6-10). So it is estimated twice: from the
    #    differences, then from the scatter around the pieces that estimate finds.
    damped = {}
    diffs = []
    for s, e, kind in stretches:
        if kind == RESPONSE:
            idx = np.flatnonzero(resp[s:e]) + s
            y = _hampel(np.log(a[idx]))
            damped[s] = (idx, y)
            diffs.extend(np.diff(y))
    var = max(float(np.var(diffs)) / 2.0 if len(diffs) >= 2 else 0.01, 0.02 ** 2)
    residuals, n_pieces = [], 0
    for idx, y in damped.values():
        pieces = (_segment(y, min_block, penalty_k * var * np.log(len(idx)))
                  if len(idx) >= 2 * min_block else [(0, len(idx))])
        residuals.extend(np.concatenate([y[p:q] - y[p:q].mean() for p, q in pieces]))
        n_pieces += len(pieces)
    if len(residuals) > n_pieces:
        var = max(float(np.sum(np.square(residuals))) / (len(residuals) - n_pieces), 0.02 ** 2)

    blocks: list[Block] = []
    for s, e, kind in stretches:
        if kind == SILENT:
            blocks.append(Block(s, e, SILENT))
            continue
        idx, y = damped[s]
        if len(idx) < 2 * min_block:
            blocks.append(Block(s, e, RESPONSE))
            continue
        pieces = _segment(y, min_block, penalty_k * var * np.log(len(idx)))
        for k, (p, _q) in enumerate(pieces):
            start = s if k == 0 else int(idx[p])
            stop = e if k == len(pieces) - 1 else int(idx[pieces[k + 1][0]])
            blocks.append(Block(start, stop, RESPONSE))

    # 3. merge neighbours whose step is smaller than the scatter
    _fill(blocks, a, curves)
    while True:
        best = None
        for k in range(len(blocks) - 1):
            b1, b2 = blocks[k], blocks[k + 1]
            if b1.kind != RESPONSE or b2.kind != RESPONSE or not (b1.n_response and b2.n_response):
                continue
            se = np.sqrt(np.nan_to_num(b1.sd) ** 2 / b1.n_response
                         + np.nan_to_num(b2.sd) ** 2 / b2.n_response)
            need = max(threshold_uv, merge_se * se)
            score = abs(b1.mean - b2.mean) / need
            if score < 1.0 and (best is None or score < best[0]):
                best = (score, k)
        if best is None:
            break
        k = best[1]
        blocks[k] = Block(blocks[k].start, blocks[k + 1].stop, RESPONSE)
        del blocks[k + 1]
        _fill(blocks, a, curves)
    return _finish(blocks, a, curves)


def _fill(blocks: list[Block], a: np.ndarray, curves: list[int]) -> None:
    for b in blocks:
        v = a[b.start:b.stop]
        b.values = v[np.isfinite(v) & (v > 0)]
        b.curves = curves[b.start:b.stop]


def _finish(blocks: list[Block], a: np.ndarray, curves: list[int]) -> list[Block]:
    _fill(blocks, a, curves)
    for k, b in enumerate(blocks):
        b.label = f"B{k + 1}"
    return blocks


def verdict(blocks: list[Block]) -> str:
    """A. Militskova's rule on one channel's blocks (see the module docstring)."""
    responding = [b for b in blocks if b.kind == RESPONSE and b.n_response]
    if not responding:
        return VERDICT_NONE
    if not any(b.n_response > 1 for b in responding):
        return VERDICT_FEW          # a spread needs at least two responses in one block
    if any(b.spread_above for b in responding):
        return VERDICT_WORKS
    if blocks[0].kind == SILENT:
        return VERDICT_APPEARED
    return VERDICT_NOT


def labels_per_curve(blocks: list[Block], n: int, silent_label: bool = True) -> list[str | None]:
    """The block label of every curve (None for silent blocks when ``silent_label`` is False)."""
    out: list[str | None] = [None] * n
    for b in blocks:
        if b.kind == SILENT and not silent_label:
            continue
        for i in range(b.start, b.stop):
            out[i] = b.label
    return out

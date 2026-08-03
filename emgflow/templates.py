"""Build a matching template out of a response the user selected on screen.

Both modes load their template bank from the same directory via
`pipeline._resolve_startstop_template_dir()`, which is a plain module-level function — so
pointing the pipeline at a session-local bank is a monkeypatch, not a fork of src/.

Bank format (as `_load_startstop_template_bank` reads it):
    template_<id>.npy                  waveform, sampled at STARTSTOP_TM_TEMPLATE_SFREQ
    template_<id>_onset_and_peaks.npz  onset / peak1 / peak2 as SAMPLE INDICES
Sample k of the waveform sits at time (k - center_sample) / native_sfreq, so index
`center_sample` is t = 0 — the stimulus in SIR, the event in StartStop.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

NATIVE_SFREQ = 2000.0   # STARTSTOP_TM_TEMPLATE_SFREQ
CENTER_SAMPLE = 200     # STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE  -> t=0 lands at index 200


def stock_template_dir() -> Path:
    """The bank shipped with the repo (sibling of src/)."""
    here = Path(__file__).resolve().parent.parent
    d = here / "templates"
    return d if d.exists() else here / "src" / "templates"


@dataclass
class BuiltTemplate:
    wave: np.ndarray          # on the native 2 kHz grid
    times: np.ndarray         # seconds, times[CENTER_SAMPLE] == 0
    onset_idx: int
    peak1_idx: int
    peak2_idx: int
    source: str               # e.g. "0+3 @ 10 / RF_R"
    channel: str = ""         # the channel it was drawn on; matched to that channel ONLY

    @property
    def markers_s(self) -> dict[str, float]:
        return {
            "onset": float(self.times[self.onset_idx]),
            "peak1": float(self.times[self.peak1_idx]),
            "peak2": float(self.times[self.peak2_idx]),
        }


def build_template(
    times: np.ndarray,
    wave: np.ndarray,
    t0: float,
    t1: float,
    source: str = "",
    channel: str = "",
) -> BuiltTemplate:
    """Turn the selected span of a mean waveform into a bank-format template.

    Outside the selection the template is flattened to zero, like the stock ones: the
    matcher normalises by window energy, and flat tails are what
    STARTSTOP_TM_MATCH_TIGHT_WINDOW later crops away.
    """
    times = np.asarray(times, dtype=float)
    wave = np.asarray(wave, dtype=float)
    t0, t1 = (float(t0), float(t1)) if t0 < t1 else (float(t1), float(t0))
    if t1 - t0 < 1.0 / NATIVE_SFREQ:
        raise ValueError("The selected window is too short to be a template.")

    # Native grid: index k <-> time (k - CENTER_SAMPLE) / NATIVE_SFREQ. k=0 is therefore
    # -0.1 s, which comfortably precedes any response, and we extend past the selection.
    t_end = max(t1 + 0.005, 0.02)
    n = int(round((t_end + CENTER_SAMPLE / NATIVE_SFREQ) * NATIVE_SFREQ)) + 1
    idx = np.arange(n, dtype=float)
    grid = (idx - CENTER_SAMPLE) / NATIVE_SFREQ

    # Baseline from whatever precedes the selection, so the template starts at zero.
    pre = times < t0
    base = float(np.mean(wave[pre])) if pre.any() else 0.0
    resampled = np.interp(grid, times, wave - base, left=0.0, right=0.0)

    inside = (grid >= t0) & (grid <= t1)
    if not inside.any():
        raise ValueError("The selection does not overlap the epoch window.")
    tpl = np.where(inside, resampled, 0.0)

    # P1 = the dominant deflection in the selection; P2 = strongest opposite rebound after
    # it; onset = the foot of P1 (walk back to 20 % of its amplitude). Same convention the
    # pipeline's own forced-marker logic uses, so the markers mean the same thing.
    sel = np.where(inside)[0]
    seg = tpl[sel]
    p1_local = int(np.argmax(np.abs(seg)))
    peak1 = int(sel[p1_local])
    sign = 1.0 if tpl[peak1] >= 0 else -1.0

    after = sel[sel > peak1]
    peak2 = peak1
    if after.size:
        opp = -sign * tpl[after]
        if np.any(opp > 0):
            peak2 = int(after[int(np.argmax(opp))])
    if peak2 == peak1:  # monophasic selection: put P2 at the end of the window
        peak2 = int(sel[-1])

    thr = 0.2 * sign * tpl[peak1]
    onset = peak1
    lo = int(sel[0])
    while onset - 1 >= lo and sign * tpl[onset - 1] >= thr:
        onset -= 1

    return BuiltTemplate(tpl, grid, onset, peak1, peak2, source, channel)


class TemplateBank:
    """A session-local bank the pipeline is pointed at.

    `include_stock=True` seeds it with the 26 shipped templates and adds yours on top;
    `False` gives a bank containing ONLY your templates — that is how you say "find me
    this response and nothing else".
    """

    def __init__(self, directory: Path, include_stock: bool = True) -> None:
        self.dir = Path(directory)
        self.include_stock = include_stock
        self.dir.mkdir(parents=True, exist_ok=True)
        if include_stock:
            self._seed_stock()

    def _seed_stock(self) -> None:
        src = stock_template_dir()
        if not src.exists():
            return
        for f in src.glob("template_*"):
            dest = self.dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    def _next_id(self) -> int:
        used = [
            int(m.group(1))
            for f in self.dir.glob("template_*.npy")
            if (m := re.fullmatch(r"template_(\d+)\.npy", f.name))
        ]
        return max(used, default=0) + 1

    def _channel_of(self, name: str) -> str:
        """The channel a stored template is bound to ('' = unbound / stock)."""
        npz = self.dir / f"{name}_onset_and_peaks.npz"
        if not npz.exists():
            return ""
        try:
            data = np.load(npz)
            if "channel" in data:
                return str(np.asarray(data["channel"]).reshape(()).item())
        except Exception:
            pass
        return ""

    def _remove(self, name: str) -> None:
        for suffix in (".npy", "_onset_and_peaks.npz", ".source.txt"):
            (self.dir / f"{name}{suffix}").unlink(missing_ok=True)

    def replace_channel_templates(self, channel: str) -> int:
        """Drop the user templates already bound to *channel*.

        A new hand template for a channel is the clinician's latest word on it, so
        the earlier ones must stop competing — otherwise the matcher could keep
        picking a template she has just replaced. Stock templates are untouched.
        """
        if not channel:
            return 0
        removed = 0
        # Scan EVERY template, not just user_templates(): a "mine only" bank reuses
        # the stock id range, so name alone cannot tell user from stock. The bound
        # channel does — stock templates carry none, so they never match here.
        for f in list(self.dir.glob("template_*.npy")):
            if not re.fullmatch(r"template_(\d+)\.npy", f.name):
                continue
            if self._channel_of(f.stem) == channel:
                self._remove(f.stem)
                removed += 1
        return removed

    def add(self, tpl: BuiltTemplate) -> str:
        """Write a built template into the bank. Returns its name.

        A template bound to a channel replaces any earlier hand template on that
        same channel, so only the newest one is ever matched there.
        """
        self.replace_channel_templates(tpl.channel)
        tid = self._next_id()
        name = f"template_{tid}"
        np.save(self.dir / f"{name}.npy", tpl.wave)
        np.savez(
            self.dir / f"{name}_onset_and_peaks.npz",
            onset=np.array(tpl.onset_idx),
            peak1=np.array(tpl.peak1_idx),
            peak2=np.array(tpl.peak2_idx),
            channel=np.array(tpl.channel or ""),   # bind to its channel; "" = every channel
        )
        # A note to whoever opens the folder later; the pipeline ignores it.
        (self.dir / f"{name}.source.txt").write_text(
            f"{tpl.source}\nonset={tpl.markers_s['onset']:.4f}s "
            f"peak1={tpl.markers_s['peak1']:.4f}s peak2={tpl.markers_s['peak2']:.4f}s\n",
            encoding="utf-8",
        )
        return name

    def user_templates(self) -> list[str]:
        """Names of templates in this bank that did NOT come from the stock set."""
        stock = {f.name for f in stock_template_dir().glob("template_*.npy")}
        return sorted(
            f.stem for f in self.dir.glob("template_*.npy") if f.name not in stock
        )

    def count(self) -> int:
        return len(list(self.dir.glob("template_*.npy")))

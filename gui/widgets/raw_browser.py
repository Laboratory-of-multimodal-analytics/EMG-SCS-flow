"""Interactive whole-recording browser.

Any recording can simply be looked at: scroll through it, pick the window length, and
overlay what the pipeline found — stimulus/protocol annotations, detected responses, and
(in StartStop) the spontaneous-EMG bursts with their RMS envelopes.

This is deliberately not mne's own browser: we need the pipeline's bursts, envelopes and
detections drawn on the same axes, and mne-qt-browser is not installed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QScrollBar, QVBoxLayout, QWidget,
)

from .plot_canvas import VerticalPlotCanvas, style_channel_axis

BURST_COLOUR = "#ff9f1c"
ENV_COLOUR = "#c1121f"
DET_COLOUR = "#d62728"


class RawBrowser(QWidget):
    """Scroll through a raw recording with the pipeline's findings drawn on top."""

    def __init__(self) -> None:
        super().__init__()
        self.raw = None
        self.title = ""
        self.detections: list[tuple[float, str]] = []       # (time_s, description)
        self.bursts = None                                   # DataFrame: Channel/Start_s/End_s
        self.envelopes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._t0 = 0.0

        self.window_len = QDoubleSpinBox()
        self.window_len.setRange(0.2, 600.0)
        self.window_len.setValue(10.0)
        self.window_len.setSuffix(" s")
        self.window_len.setToolTip("Length of the visible window")
        self.window_len.valueChanged.connect(self._draw)

        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.valueChanged.connect(self._on_scroll)

        self.show_annots = QCheckBox("Protocol annotations")
        self.show_annots.setChecked(True)
        self.show_dets = QCheckBox("Detections")
        self.show_dets.setChecked(True)
        self.show_bursts = QCheckBox("Bursts")
        self.show_bursts.setChecked(True)
        self.show_env = QCheckBox("RMS envelope")
        self.show_env.setChecked(True)
        for cb in (self.show_annots, self.show_dets, self.show_bursts, self.show_env):
            cb.toggled.connect(self._draw)

        self.jump = QComboBox()
        self.jump.setMinimumWidth(220)
        self.jump.currentIndexChanged.connect(self._on_jump)

        self.whole = QPushButton("Whole segment")
        self.whole.setToolTip("Fit the entire recording/condition into one window.")
        self.whole.clicked.connect(self._fit_all)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Window:"))
        bar.addWidget(self.window_len)
        bar.addWidget(self.whole)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Jump to:"))
        bar.addWidget(self.jump)
        bar.addSpacing(10)
        bar.addWidget(self.show_annots)
        bar.addWidget(self.show_dets)
        bar.addWidget(self.show_bursts)
        bar.addWidget(self.show_env)
        bar.addStretch()

        self.plot = VerticalPlotCanvas(row_inches=1.4)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.scroll)

    # ------------------------------------------------------------------ #
    def set_recording(
        self,
        raw,
        title: str = "",
        detections: list[tuple[float, str]] | None = None,
        bursts=None,
        envelopes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.raw = raw
        self.title = title
        self.detections = detections or []
        self.bursts = bursts
        self.envelopes = envelopes or {}
        self._t0 = 0.0

        dur = float(raw.times[-1]) if raw is not None else 0.0
        self.window_len.setMaximum(max(dur, 1.0))
        if dur and self.window_len.value() > dur:
            self.window_len.setValue(dur)

        # Jump targets: every detection and every burst, in time order.
        self.jump.blockSignals(True)
        self.jump.clear()
        self.jump.addItem("—", None)
        for t, desc in self.detections:
            self.jump.addItem(f"detection {t:.2f} s — {desc}", float(t))
        if bursts is not None and len(bursts):
            for _, b in bursts.iterrows():
                self.jump.addItem(
                    f"burst {float(b['Start_s']):.2f} s — {b['Channel']}", float(b["Start_s"])
                )
        self.jump.blockSignals(False)

        self._sync_scroll()
        self._draw()

    # ------------------------------------------------------------------ #
    def _sync_scroll(self) -> None:
        if self.raw is None:
            return
        dur = float(self.raw.times[-1])
        win = float(self.window_len.value())
        span = max(dur - win, 0.0)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, int(span * 100))
        self.scroll.setPageStep(int(win * 100))
        self.scroll.setValue(int(self._t0 * 100))
        self.scroll.blockSignals(False)

    def _on_scroll(self, value: int) -> None:
        self._t0 = value / 100.0
        self._draw(rescroll=False)

    def _on_jump(self, _idx: int) -> None:
        t = self.jump.currentData()
        if t is None or self.raw is None:
            return
        self._t0 = max(0.0, float(t) - self.window_len.value() / 3.0)
        self._draw()

    def _fit_all(self) -> None:
        if self.raw is None:
            return
        self._t0 = 0.0
        self.window_len.setValue(float(self.raw.times[-1]))

    # ------------------------------------------------------------------ #
    def _draw(self, rescroll: bool = True) -> None:
        if self.raw is None:
            self.plot.message("Open a recording or a results folder.")
            return
        if rescroll:
            self._sync_scroll()

        raw = self.raw
        sf = float(raw.info["sfreq"])
        dur = float(raw.times[-1])
        win = float(self.window_len.value())
        t0 = min(max(self._t0, 0.0), max(dur - win, 0.0))
        t1 = min(t0 + win, dur)

        # Annotation onsets live in the ORIGINAL recording's time base; raw.times starts at 0.
        offset = float(raw.first_time)

        chans = [c for c in raw.ch_names if "art" not in c.lower()]
        if not chans:
            chans = list(raw.ch_names)

        s0, s1 = int(t0 * sf), int(t1 * sf)
        data = raw.get_data(picks=chans, start=s0, stop=s1) * 1e6
        times = np.arange(s0, s1) / sf

        axes = self.plot.make_axes(len(chans))
        for ax, ch, row in zip(axes, chans, range(len(chans))):
            ax.plot(times, data[row], lw=0.6, color="#1f77b4", zorder=2)

            if self.show_bursts.isChecked() and self.bursts is not None and len(self.bursts):
                b = self.bursts[self.bursts["Channel"].astype(str) == ch]
                for _, r in b.iterrows():
                    bs, be = float(r["Start_s"]), float(r["End_s"])
                    if be >= t0 and bs <= t1:
                        ax.axvspan(bs, be, color=BURST_COLOUR, alpha=0.25, lw=0, zorder=1)

            if self.show_env.isChecked() and ch in self.envelopes:
                et, ev = self.envelopes[ch]
                sel = (et >= t0) & (et <= t1)
                if sel.any():
                    # The envelope is RMS (positive); mirror it so it reads as a band around
                    # the signal rather than a curve floating above it.
                    ax.plot(et[sel], ev[sel], color=ENV_COLOUR, lw=1.2, zorder=4)
                    ax.plot(et[sel], -ev[sel], color=ENV_COLOUR, lw=1.2, zorder=4)

            if self.show_dets.isChecked():
                for t, desc in self.detections:
                    if t0 <= t <= t1 and (ch in desc.split("+") or desc == ""):
                        ax.axvline(t, color=DET_COLOUR, lw=1.2, zorder=5)

            if self.show_annots.isChecked():
                for ann in raw.annotations:
                    at = float(ann["onset"]) - offset
                    if t0 <= at <= t1:
                        ax.axvline(at, color="0.6", lw=0.8, ls="--", zorder=3)

            responding = any(ch in d.split("+") for _, d in self.detections)
            style_channel_axis(ax, ch, bold=responding, muted=not responding, compact=True)

        # Annotation labels once, on the top axis, so they do not repeat on every channel.
        if self.show_annots.isChecked() and len(axes):
            for ann in raw.annotations:
                at = float(ann["onset"]) - offset
                if t0 <= at <= t1:
                    axes[0].annotate(
                        str(ann["description"])[:24], xy=(at, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points",
                        fontsize=7, color="0.35", rotation=90, va="top",
                    )

        axes[-1].set_xlabel("time (s)")
        axes[-1].set_xlim(t0, t1)
        self.plot.fig.suptitle(
            f"{self.title}   —   {t0:.2f}–{t1:.2f} s of {dur:.1f} s", fontsize=10
        )
        self.plot.draw()

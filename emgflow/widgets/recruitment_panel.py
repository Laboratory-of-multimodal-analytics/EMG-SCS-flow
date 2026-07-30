"""Recruitment shown as curves, not a wall of numbers.

Response size (peak-to-peak, µV) against stimulation amplitude, one line per channel, with
a 95 % CI band across the epochs of each crop. A single-trial point gets no band — a
zero-width ribbon is the honest rendering, not a fabricated interval.

Neurosoft curve exports have no amplitude axis to plot against — the file is one crop and
its amplitude label is the synthetic ``all``. There the ramp IS the curve order, so the
panel switches its x-axis to the curve number rather than going blank. It is the same
recruitment curve; only the units of the x-axis are unknown.

The pipeline still writes its own summary tables; this is for looking, not for archiving.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class RecruitmentPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.df = None
        self.by_curve = None
        self.highlight: str | None = None

        self.title = QLabel("<b>Recruitment</b> — peak-to-peak vs amplitude, mean ± 95 % CI")
        self.title.setStyleSheet("font-size: 11px;")
        title = self.title

        self.fig = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(title)
        layout.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------ #
    def show_config(self, df, config: str, highlight: str | None = None,
                    by_curve=None) -> None:
        """*df* is amplitude-keyed; *by_curve* is the per-curve fallback used when
        the file carries no amplitude axis (Neurosoft exports)."""
        self.df = df
        self.by_curve = by_curve
        self.config = config
        self.highlight = highlight
        self._draw()

    def set_highlight(self, channel: str | None) -> None:
        self.highlight = channel
        self._draw()

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        df = self.df
        if df is not None and len(df) and "amp_value" in df:
            usable = df[np.isfinite(df["amp_value"])]
        else:
            usable = None

        # Fewer than two numeric amplitudes means there is no amplitude axis to
        # draw — one crop, or non-numeric labels. The curve order is the ramp for
        # those files, so plot against that instead of going blank.
        if usable is None or usable["amp_value"].nunique() < 2:
            self._draw_by_curve(ax)
            return

        channels = sorted(usable["Channel"].unique())
        colors = plt.get_cmap("tab10")

        for i, ch in enumerate(channels):
            g = usable[usable["Channel"] == ch].sort_values("amp_value")
            x = g["amp_value"].to_numpy(float)
            y = g["mean_ptp_uv"].to_numpy(float)
            ci = g["ci95"].to_numpy(float)
            focused = (self.highlight is None) or (ch == self.highlight)
            col = colors(i % 10)
            ax.plot(x, y, "-o", ms=4, lw=1.8 if focused else 0.9, color=col,
                    alpha=1.0 if focused else 0.25, label=ch, zorder=3 if focused else 2)
            ax.fill_between(x, y - ci, y + ci, color=col,
                            alpha=0.22 if focused else 0.06, lw=0, zorder=1)

        self.title.setText("<b>Recruitment</b> — peak-to-peak vs amplitude, mean ± 95 % CI")
        ax.set_ylabel("peak-to-peak (µV)")
        ax.set_xlabel("stimulation amplitude")
        ax.set_title(f"{self.config}", fontsize=10)
        ax.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if channels:
            ax.legend(fontsize=7, ncol=2, frameon=False)
        self.canvas.draw_idle()

    def _draw_by_curve(self, ax) -> None:
        """Response size against curve number — the ramp when amplitudes are unknown."""
        bc = self.by_curve
        if bc is None or len(bc) == 0:
            ax.text(0.5, 0.5, "No recruitment data for this configuration.",
                    ha="center", va="center", color="gray")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        self.title.setText(
            "<b>Recruitment</b> — response vs curve number "
            "(this export carries no amplitude labels)"
        )
        # Channels with at least one detection; the frame carries a row per
        # curve, so undetected curves are NaN and break the line into gaps
        # rather than moving the axis.
        got = bc.groupby("Channel")["amp_uv"].apply(lambda s: s.notna().any())
        channels = sorted((c for c, ok in got.items() if ok), key=lambda s: (len(s), s))
        colors = plt.get_cmap("tab10")
        xmax = 0
        for i, ch in enumerate(channels):
            g = bc[bc["Channel"] == ch].sort_values("curve")
            focused = (self.highlight is None) or (ch == self.highlight)
            col = colors(i % 10)
            ax.plot(g["curve"], g["amp_uv"], "-", lw=1.8 if focused else 0.9,
                    color=col, alpha=1.0 if focused else 0.25, label=ch,
                    zorder=3 if focused else 2)
            d = g[g["amp_uv"].notna()]
            ax.plot(d["curve"], d["amp_uv"], "o", ms=3, color=col,
                    alpha=1.0 if focused else 0.25, zorder=3 if focused else 2)
            xmax = max(xmax, int(g["curve"].max()) if len(g) else 0)
        if xmax:
            ax.set_xlim(0.5, xmax + 0.5)

        ax.set_ylabel("PTP / |P1|, µV")
        ax.set_xlabel("curve number")
        ax.set_title(f"{self.config}", fontsize=10)
        ax.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if channels:
            ax.legend(fontsize=7, ncol=2, frameon=False)
        self.canvas.draw_idle()

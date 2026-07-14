"""Recruitment shown as curves, not a wall of numbers.

Response size (peak-to-peak, µV) against stimulation amplitude, one line per channel, with
a 95 % CI band across the epochs of each crop. A single-trial point gets no band — a
zero-width ribbon is the honest rendering, not a fabricated interval.

The pipeline still writes its own summary tables; this is for looking, not for archiving.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class RecruitmentPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.df = None
        self.highlight: str | None = None

        self.kind = QComboBox()
        self.kind.addItems(["Curves (mean ± 95 % CI)", "Histogram (per amplitude)"])
        self.kind.currentIndexChanged.connect(self._draw)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Recruitment</b>"))
        bar.addStretch()
        bar.addWidget(self.kind)

        self.fig = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------ #
    def show_config(self, df, config: str, highlight: str | None = None) -> None:
        self.df = df
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
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, "No recruitment data for this configuration.",
                    ha="center", va="center", color="gray")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        usable = df[np.isfinite(df["amp_value"])]
        if usable.empty:
            ax.text(0.5, 0.5, "Amplitude labels are not numeric — no curve can be drawn.",
                    ha="center", va="center", color="gray")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        channels = sorted(usable["Channel"].unique())
        colors = plt.get_cmap("tab10")

        if self.kind.currentIndex() == 0:
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
            ax.set_ylabel("peak-to-peak (µV)")
        else:
            width = 0.8 / max(len(channels), 1)
            amps = sorted(usable["amp_value"].unique())
            pos = np.arange(len(amps))
            for i, ch in enumerate(channels):
                g = usable[usable["Channel"] == ch]
                vals = [
                    float(g[g["amp_value"] == a]["mean_ptp_uv"].iloc[0])
                    if (g["amp_value"] == a).any() else 0.0
                    for a in amps
                ]
                errs = [
                    float(g[g["amp_value"] == a]["ci95"].iloc[0])
                    if (g["amp_value"] == a).any() else 0.0
                    for a in amps
                ]
                focused = (self.highlight is None) or (ch == self.highlight)
                ax.bar(pos + i * width, vals, width, yerr=errs, capsize=2, label=ch,
                       color=colors(i % 10), alpha=1.0 if focused else 0.25)
            ax.set_xticks(pos + 0.4 - width / 2)
            ax.set_xticklabels([f"{a:g}" for a in amps], fontsize=7)
            ax.set_ylabel("peak-to-peak (µV)")

        ax.set_xlabel("stimulation amplitude")
        ax.set_title(f"{self.config}", fontsize=10)
        ax.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if channels:
            ax.legend(fontsize=7, ncol=2, frameon=False)
        self.canvas.draw_idle()

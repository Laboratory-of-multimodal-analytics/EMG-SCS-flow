"""Review a template before it enters the bank: click its onset, P1 and P2 into place.

The auto-placed markers follow the same rule the pipeline's forced-marker logic uses
(P1 = dominant deflection, P2 = strongest opposite rebound after it, onset = the foot of
P1), but the whole point of drawing a template by hand is that you disagree with the
automatic answer somewhere — so every marker is clickable.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QRadioButton, QVBoxLayout,
)

from .plot_canvas import C_ONSET, C_P1, C_P2

MARKERS = [("onset", C_ONSET), ("peak1", C_P1), ("peak2", C_P2)]


class TemplateEditor(QDialog):
    """Shows the built template; lets the user place onset / P1 / P2 by clicking."""

    def __init__(self, tpl, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Template — check the markers")
        self.resize(820, 520)
        self.tpl = tpl
        self.idx = {
            "onset": int(tpl.onset_idx),
            "peak1": int(tpl.peak1_idx),
            "peak2": int(tpl.peak2_idx),
        }

        self.radios: dict[str, QRadioButton] = {}
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Click on the trace to place:"))
        for name, colour in MARKERS:
            rb = QRadioButton(name)
            rb.setStyleSheet(f"color: {colour};")
            self.radios[name] = rb
            bar.addWidget(rb)
        self.radios["peak1"].setChecked(True)
        bar.addStretch()

        self.info = QLabel()
        self.info.setStyleSheet("color: gray;")

        self.fig = Figure(figsize=(8, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Save to bank")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.info)
        layout.addWidget(buttons)

        self._draw()

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        t, w = self.tpl.times, self.tpl.wave

        # The template is zero outside the selection; show only the part that carries signal,
        # with a little context on each side.
        nz = np.nonzero(w)[0]
        if nz.size:
            lo = max(nz[0] - 40, 0)
            hi = min(nz[-1] + 40, w.size - 1)
        else:
            lo, hi = 0, w.size - 1

        ax.plot(t[lo:hi + 1], w[lo:hi + 1], color="black", lw=1.6, zorder=3)
        ax.axvline(0.0, color="0.5", lw=0.9, ls=":", zorder=1)
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=1)
        if nz.size:
            ax.axvspan(t[nz[0]], t[nz[-1]], color="#ffb703", alpha=0.15, lw=0, zorder=0)

        for name, colour in MARKERS:
            i = self.idx[name]
            ax.scatter([t[i]], [w[i]], color=colour, s=60, zorder=5, label=name)
            ax.annotate(
                f"{name}\n{1000 * t[i]:.1f} ms",
                (t[i], w[i]), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color=colour,
            )

        ax.set_xlabel("time from stimulus (s)")
        ax.set_ylabel("amplitude (µV)")
        ax.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(self.tpl.source, fontsize=9)
        self.canvas.draw_idle()

        ok = self.idx["onset"] <= self.idx["peak1"] <= self.idx["peak2"]
        self.info.setText(
            "onset ≤ P1 ≤ P2 — good."
            if ok else
            "⚠ Markers are out of order: the pipeline expects onset before P1 before P2."
        )

    # ------------------------------------------------------------------ #
    def _which(self) -> str:
        return next(n for n, _ in MARKERS if self.radios[n].isChecked())

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        # Snap to the nearest sample of the template's own grid.
        i = int(np.argmin(np.abs(self.tpl.times - float(event.xdata))))
        self.idx[self._which()] = i
        self._draw()

    # ------------------------------------------------------------------ #
    def edited_template(self):
        """The template with whatever markers the user settled on."""
        self.tpl.onset_idx = self.idx["onset"]
        self.tpl.peak1_idx = self.idx["peak1"]
        self.tpl.peak2_idx = self.idx["peak2"]
        return self.tpl

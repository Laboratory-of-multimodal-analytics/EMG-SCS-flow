"""Review a template before it enters the bank: click its onset, P1 and P2 into place.

The auto-placed markers follow the same rule the pipeline's forced-marker logic uses
(P1 = dominant deflection, P2 = strongest opposite rebound after it, onset = the foot of
P1), but the whole point of drawing a template by hand is that you disagree with the
automatic answer somewhere — so every marker is clickable.

An H-reflex template carries **six** markers rather than three: the curve holds two
responses (M and H), each with its own onset, P1 and P2. The editor is the same in both
cases — the marker list is simply longer, and the ordering check is applied within each
component instead of across all six, because M-P2 preceding H-onset is the whole point
and comparing them as one chain would report it as an error.
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

#: The H-reflex marker set: the same three, twice, prefixed by their component.
HREFLEX_MARKERS = [
    ("M onset", C_ONSET), ("M peak1", C_P1), ("M peak2", C_P2),
    ("H onset", C_ONSET), ("H peak1", C_P1), ("H peak2", C_P2),
]


class TemplateEditor(QDialog):
    """Shows the built template; lets the user place the markers by clicking.

    ``markers`` names the marker set — the default three, or ``HREFLEX_MARKERS``.
    ``initial`` gives a starting sample index per name; anything missing falls back to
    the template's own onset/peak1/peak2.
    """

    def __init__(self, tpl, parent=None, markers=None, initial=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Template — check the markers")
        self.resize(900, 560)
        self.tpl = tpl
        self.markers = list(markers or MARKERS)
        fallback = {"onset": int(tpl.onset_idx), "peak1": int(tpl.peak1_idx),
                    "peak2": int(tpl.peak2_idx)}
        initial = initial or {}
        self.idx = {
            name: int(initial.get(name, fallback.get(name.split()[-1], int(tpl.peak1_idx))))
            for name, _ in self.markers
        }

        self.radios: dict[str, QRadioButton] = {}
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Click on the trace to place:"))
        for name, colour in self.markers:
            rb = QRadioButton(name)
            rb.setStyleSheet(f"color: {colour};")
            self.radios[name] = rb
            bar.addWidget(rb)
        self.radios[self.markers[0][0]].setChecked(True)
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

        for name, colour in self.markers:
            i = self.idx[name]
            # The reflex markers are hollow, exactly as on the exported panels, so
            # "which component is this" reads the same way everywhere.
            hollow = name.startswith("H ")
            kw = (dict(facecolors="none", edgecolors=colour, linewidths=1.6)
                  if hollow else dict(color=colour))
            ax.scatter([t[i]], [w[i]], s=60, zorder=5, **kw)
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
        self.info.setText(self._check())

    def _check(self) -> str:
        """Marker order, checked within each component rather than across all of them."""
        names = [n for n, _ in self.markers]
        groups = {}
        for n in names:
            groups.setdefault(n.split()[0] if " " in n else "", []).append(n)
        bad = [g for g, ns in groups.items()
               if [self.idx[n] for n in ns] != sorted(self.idx[n] for n in ns)]
        if bad:
            label = ", ".join(g or "маркеры" for g in bad)
            return f"⚠ Порядок нарушен ({label}): ожидается onset ≤ P1 ≤ P2."
        if len(groups) > 1 and self.idx.get("H peak1", 0) <= self.idx.get("M peak2", 0):
            return "⚠ H-ответ должен начинаться позже, чем заканчивается M-ответ."
        return "Порядок маркеров в норме."

    # ------------------------------------------------------------------ #
    def _which(self) -> str:
        return next(n for n, _ in self.markers if self.radios[n].isChecked())

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        # Snap to the nearest sample of the template's own grid.
        i = int(np.argmin(np.abs(self.tpl.times - float(event.xdata))))
        self.idx[self._which()] = i
        self._draw()

    # ------------------------------------------------------------------ #
    def marker_times_ms(self) -> dict[str, float]:
        """Every marker's latency in milliseconds, keyed by its own name."""
        return {name: float(self.tpl.times[self.idx[name]]) * 1e3
                for name, _ in self.markers}

    def edited_template(self):
        """The template with whatever markers the user settled on.

        Only meaningful for the three-marker set — a six-marker H-reflex template
        is not a bank template; it is a per-channel correction (see
        ``src/hreflex.reference_from_markers``).
        """
        self.tpl.onset_idx = self.idx.get("onset", self.tpl.onset_idx)
        self.tpl.peak1_idx = self.idx.get("peak1", self.tpl.peak1_idx)
        self.tpl.peak2_idx = self.idx.get("peak2", self.tpl.peak2_idx)
        return self.tpl

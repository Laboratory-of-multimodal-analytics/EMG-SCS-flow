"""Canvases for stacked channel plots.

`FitPlotCanvas` fits every channel into the viewport — no scrolling. SIR epochs are short,
so the time axis can be squeezed hard and the column kept narrow; what matters is seeing
all muscles at once.

`VerticalPlotCanvas` keeps the tall, scrolled stack for the whole-recording views, where
the time axis genuinely needs the width.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSizePolicy

DPI = 100
MIN_ROW_INCHES = 0.9
MAX_ROW_INCHES = 4.0

# Marker colours, matching the pipeline's own panels (plotting.plot_epochs_panel):
# onset blue, P1 red, P2 green, drawn as points on each epoch.
C_ONSET, C_P1, C_P2 = "blue", "red", "green"
MARKER_SIZE = 18


class FitPlotCanvas(FigureCanvasQTAgg):
    """All channels, always visible: the figure fills the widget and never scrolls."""

    def __init__(self) -> None:
        self.fig = Figure(dpi=DPI, layout="constrained")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(280)

    def make_axes(self, n_rows: int, sharex: bool = True):
        self.fig.clear()
        axes = self.fig.subplots(max(1, n_rows), 1, sharex=sharex, squeeze=False)[:, 0]
        return axes

    def message(self, text: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", wrap=True)
        ax.set_axis_off()
        self.draw_idle()

    # NB: never override draw() on a FigureCanvasQTAgg — it is Qt's paint path, and
    # redirecting it to draw_idle() renders the widget black. Use refresh().
    def refresh(self) -> None:
        self.draw_idle()


class VerticalPlotCanvas(QScrollArea):
    """Scrollable figure whose height grows with the number of channels."""

    def __init__(self, row_inches: float = 2.0) -> None:
        super().__init__()
        self.row_inches = row_inches
        self.fig = Figure(dpi=DPI, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.setWidget(self.canvas)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._n_rows = 1

    def set_row_height(self, inches: float) -> None:
        self.row_inches = max(MIN_ROW_INCHES, min(MAX_ROW_INCHES, float(inches)))
        self._resize_canvas()

    def _width_inches(self) -> float:
        px = max(self.viewport().width() - 4, 400)
        return px / DPI

    def _resize_canvas(self) -> None:
        h_in = max(self.row_inches * self._n_rows, 1.5)
        self.fig.set_size_inches(self._width_inches(), h_in)
        self.canvas.setMinimumHeight(int(h_in * DPI))
        self.canvas.draw_idle()

    def resizeEvent(self, event) -> None:  # noqa: D102
        super().resizeEvent(event)
        self._resize_canvas()

    def make_axes(self, n_rows: int, sharex: bool = True):
        self.fig.clear()
        self._n_rows = max(1, n_rows)
        self._resize_canvas()
        return self.fig.subplots(self._n_rows, 1, sharex=sharex, squeeze=False)[:, 0]

    def message(self, text: str) -> None:
        self.fig.clear()
        self._n_rows = 1
        self._resize_canvas()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", wrap=True)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def draw(self) -> None:
        self.canvas.draw_idle()


def style_channel_axis(ax, label: str, *, bold: bool = False, muted: bool = False,
                       compact: bool = False) -> None:
    """Consistent per-channel row: channel name on the left, light grid, tidy ticks."""
    ax.set_ylabel(
        label,
        rotation=0,
        ha="right",
        va="center",
        labelpad=8,
        fontsize=8 if compact else 10,
        fontweight="bold" if bold else "normal",
        color="0.35" if muted else "black",
    )
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.tick_params(labelsize=7 if compact else 8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

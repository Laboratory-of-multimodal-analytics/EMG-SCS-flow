"""A tall, scrollable channel stack.

The pipeline's own panels are `figsize=(10, 2.5 * n_channels)` — one generous row per
channel, read by scrolling. Cramming nine channels into the height of a window instead
gives 20-pixel rows where nothing is legible, so the canvas grows vertically and the view
scrolls, exactly like the PNGs.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

DPI = 100
MIN_ROW_INCHES = 0.9
MAX_ROW_INCHES = 4.0


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

    # ------------------------------------------------------------------ #
    def set_row_height(self, inches: float) -> None:
        self.row_inches = max(MIN_ROW_INCHES, min(MAX_ROW_INCHES, float(inches)))
        self._resize_canvas()

    def _width_inches(self) -> float:
        # Fill the viewport horizontally; the vertical scrollbar eats a little.
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

    # ------------------------------------------------------------------ #
    def make_axes(self, n_rows: int, sharex: bool = True):
        """Clear the figure and lay out one tall row per channel."""
        self.fig.clear()
        self._n_rows = max(1, n_rows)
        self._resize_canvas()
        axes = self.fig.subplots(self._n_rows, 1, sharex=sharex, squeeze=False)[:, 0]
        return axes

    def message(self, text: str) -> None:
        """Show a single centred message instead of a channel stack."""
        self.fig.clear()
        self._n_rows = 1
        self._resize_canvas()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", wrap=True)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def draw(self) -> None:
        self.canvas.draw_idle()


def style_channel_axis(ax, label: str, *, bold: bool = False, muted: bool = False) -> None:
    """Consistent per-channel row: readable name on the left, light grid, tidy ticks."""
    ax.set_ylabel(
        label,
        rotation=0,
        ha="right",
        va="center",
        labelpad=12,
        fontsize=10,
        fontweight="bold" if bold else "normal",
        color="0.35" if muted else "black",
    )
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

"""What the two group tabs share: one layout, one way to compare, one look.

Both tabs answer "does group B differ from group A, channel by channel?" — for
stimulation-evoked responses and for spontaneous EMG. They are built from the same
pieces so they read the same and behave the same:

* ``GroupPicker`` — the comparison: group A against group B, or all groups at once.
  Every view and the statistics table follow it;
* ``ChannelPicker`` — which channels are drawn, as wrapping checkboxes;
* ``PlotView`` — a figure whose panels are ALWAYS the same size: the number of
  columns follows the window width, a panel's height is fixed, and the margins are
  fixed in pixels (no layout engine to run on every redraw). The standard matplotlib
  toolbar gives zoom, pan and saving;
* ``dots_and_box`` — the one way a group of numbers is drawn: a dot per recording,
  the median as a bold line, the quartiles as a light box;
* ``top_legend`` — one legend above the panels; its title explains what is drawn;
* ``StatsTable`` — the comparison table under the plots, same columns everywhere;
* ``GroupTab`` — the layout and the redraw logic. Changes are collected for a short
  moment, then only the visible view is drawn; the others are drawn when opened.
  Nothing heavy is recomputed on a redraw.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLayout, QPushButton, QScrollArea,
                               QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

import src.group_stats as ST

#: Group A and group B keep these two colours in every view of both tabs.
COLOR_A = "#2166ac"
COLOR_B = "#d6604d"
#: When all groups are shown at once.
PALETTE = ["#2166ac", "#d6604d", "#4d9221", "#7b3294", "#e08214", "#01665e",
           "#8c510a", "#c51b7d", "#5e3c99", "#35978f", "#b2182b", "#4393c3"]

#: One panel, in logical pixels: the same in every view, whatever the channel count.
PANEL_W = 400
PANEL_H = 290
MAX_COLUMNS = 4
#: Fixed margins, in logical pixels.
MARGIN_LEFT = 66
MARGIN_RIGHT = 18
GAP_X = 80
TITLE_H = 30
BOTTOM_H = 66
#: Height above the panels for the legend: its title line, then one line per row of entries.
LEGEND_TITLE_H = 30
LEGEND_ROW_H = 20
LEGEND_COLS = 5
#: One more line of a long legend title, in logical pixels.
TITLE_LINE_H = 14


def label_space(groups) -> int:
    """Extra pixels under a panel for group names written under the x axis."""
    groups = [str(g) for g in groups]
    if len(groups) > 6:
        return 50            # names at 45 degrees
    if len(groups) > 3 or any(len(g) > 14 for g in groups):
        return 26            # names at 28 degrees
    return 0


def legend_rows(n_entries: int) -> int:
    """How many lines a legend of *n_entries* takes (``LEGEND_COLS`` entries per line)."""
    return max(1, math.ceil(max(int(n_entries), 1) / LEGEND_COLS))


def plural(n: int, one: str, few: str, many: str) -> str:
    """The number with its word: 1 запись, 3 записи, 12 записей, 21 запись."""
    n = int(n)
    last2, last = abs(n) % 100, abs(n) % 10
    word = many if 11 <= last2 <= 14 else one if last == 1 else few if 2 <= last <= 4 else many
    return f"{n} {word}"


def records(n: int) -> str:
    return plural(n, "запись", "записи", "записей")


def subjects(n: int) -> str:
    return plural(n, "субъект", "субъекта", "субъектов")


def short_name(name, limit: int = 32) -> str:
    """A group name that fits a legend or a list (a patient session is a whole sentence)."""
    s = str(name)
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, title: str | None = None, xlabel: str | None = None,
               ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="0.92", lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def empty_panel(ax, title: str, text: str = "нет данных") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color="0.55", fontsize=9,
            transform=ax.transAxes, wrap=True)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def top_legend(fig, entries: list[tuple[str, str, str]], title: str | None = None) -> None:
    """One legend for the whole figure, above the panels; *title* says what is drawn.

    ``entries``: ``(label, colour, kind)``, kind ``line`` / ``dot`` / ``faint`` / ``band``.
    """
    handles = []
    for label, color, kind in entries:
        label = short_name(label, 40)
        if kind == "dot":
            handles.append(Line2D([], [], marker="o", ls="none", color=color, markersize=7,
                                  markeredgecolor="white", label=label))
        elif kind == "faint":
            handles.append(Line2D([], [], color=color, lw=1.0, alpha=0.4, label=label))
        elif kind == "band":
            handles.append(Patch(facecolor=color, alpha=0.3, edgecolor="none", label=label))
        else:
            handles.append(Line2D([], [], color=color, lw=2.4, label=label))
    if not handles:
        return
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998),
                        frameon=False, fontsize=9, ncol=min(len(handles), LEGEND_COLS))
    if title:
        _set_legend_title(fig, legend, title)


def legend_note(fig, text: str) -> None:
    """Set (or replace) the explanatory title of the figure's legend."""
    if fig.legends:
        _set_legend_title(fig, fig.legends[-1], text)


def _wrap_title(text: str, width_px: float, fontsize: float = 8.5) -> str:
    """Break a long legend title at its " · " separators so that it fits the figure width."""
    import textwrap

    max_chars = max(int((width_px - 40) / (0.85 * fontsize)), 40)
    parts = []
    for part in str(text).split(" · "):
        # a piece longer than a line (narrow window) is broken between words
        parts += textwrap.wrap(part, max_chars) if len(part) > max_chars else [part]
    lines, current = [], ""
    for part in parts:
        candidate = f"{current} · {part}" if current else part
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def _grow_top(fig, extra_px: int) -> None:
    """Add *extra_px* above the panels; every panel keeps its size and place in pixels."""
    size = getattr(fig, "_panel_px", None)
    if size is None or extra_px <= 0:
        return
    width, height = size
    new_height = height + extra_px
    scale = height / new_height
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        ax.set_position([x0, y0 * scale, w, h * scale])
    w_in, h_in = fig.get_size_inches()
    fig.set_size_inches(w_in, h_in * new_height / height, forward=False)
    fig.canvas.setFixedHeight(int(new_height))
    fig._panel_px = (width, new_height)


def _set_legend_title(fig, legend, text: str) -> None:
    size = getattr(fig, "_panel_px", None)
    width = size[0] if size else fig.get_size_inches()[0] * fig.dpi
    wrapped = _wrap_title(text, width)
    lines = wrapped.count("\n") + 1
    shown = getattr(fig, "_title_lines", 1)
    if lines > shown:
        _grow_top(fig, (lines - shown) * TITLE_LINE_H)
        fig._title_lines = lines
    legend.set_title(wrapped, prop={"size": 8.5})
    legend.get_title().set_color("0.35")
    legend.get_title().set_multialignment("center")


def _jitter(n: int, width: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n) if n else np.array([])


def dots_and_box(ax, pos: float, values, color: str, seed: int = 0, width: float = 0.56,
                 ranges=None) -> None:
    """One group at x = *pos*: a dot per recording, median line, quartile box.

    ``ranges`` (optional, one ``(lo, hi)`` per value) draws a thin vertical line
    behind each dot — e.g. the spread of a recording's last five curves — so a
    recording's own scatter stays attached to its dot.
    """
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    xs = pos + _jitter(int(v.size), width * 0.32, seed)
    if ranges is not None:
        segs = [[(x, lo), (x, hi)] for x, (lo, hi), ok in zip(xs, ranges, keep)
                if ok and np.isfinite(lo) and np.isfinite(hi)]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=0.9, alpha=0.35,
                                             zorder=2), autolim=True)
    v, xs = v[keep], xs[keep]
    if v.size == 0:
        return
    if v.size >= 3:
        q1, q3 = np.percentile(v, [25, 75])
        ax.add_patch(Rectangle((pos - width / 2, q1), width, q3 - q1, facecolor=color,
                               alpha=0.13, edgecolor=color, lw=0.8, zorder=1))
    med = float(np.median(v))
    ax.plot([pos - width / 2, pos + width / 2], [med, med], color=color, lw=2.6, zorder=4,
            solid_capstyle="butt")
    ax.scatter(xs, v, s=26, color=color, alpha=0.85, edgecolor="white", linewidths=0.6,
               zorder=3)


def group_ticks(ax, labels: list[tuple[str, int]]) -> None:
    """Group names under the dots, with the number of recordings."""
    ax.set_xticks(list(range(len(labels))))

    def short(s: str) -> str:
        s = str(s)
        return s if len(s) <= 22 else s[:21] + "…"

    long = len(labels) > 3 or any(len(str(s)) > 14 for s, _ in labels)
    if len(labels) > 6:
        # many groups: steeper and smaller, or the names run into each other
        ax.set_xticklabels([f"{short(s)} (n={n})" for s, n in labels], fontsize=7,
                           rotation=45, ha="right", rotation_mode="anchor")
    elif long:
        ax.set_xticklabels([f"{short(s)} (n={n})" for s, n in labels], fontsize=7.5,
                           rotation=28, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in labels], fontsize=8.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def median_band(ax, x, median, q1, q3, color: str, label: str | None = None) -> None:
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0, zorder=1)
    ax.plot(x, median, color=color, lw=2.2, zorder=3, label=label)


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
class PlotView(QWidget):
    """A figure of equal-sized panels in a scroll area, with the matplotlib toolbar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        self.width_hint = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.scroll, 1)

    def panels(self, n: int, legend_entries: int = 1, bottom_extra: int = 0) -> list:
        """*n* axes of the fixed panel size; the columns follow the window width.

        ``legend_entries`` is how many entries the figure legend will have (0 for no
        legend): the space above the panels grows with its number of lines, so a long
        legend never runs into the panel titles. ``bottom_extra`` adds room under each
        panel for tilted group names (``label_space``); the plotting area keeps its size.
        """
        n = max(int(n), 1)
        width = max(self.scroll.viewport().width(), self.width_hint, PANEL_W) - 2
        ncol = max(1, min(n, width // PANEL_W, MAX_COLUMNS))
        nrow = math.ceil(n / ncol)
        top = (LEGEND_TITLE_H + LEGEND_ROW_H * legend_rows(legend_entries)
               if legend_entries else 8)
        bottom = BOTTOM_H + int(bottom_extra)
        panel_h = PANEL_H + int(bottom_extra)
        height = nrow * panel_h + top
        self.canvas.setFixedHeight(height)
        # The Qt canvas keeps figure.dpi multiplied by the screen's pixel ratio, so
        # logical pixels are scaled the same way before they become inches.
        ratio = float(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0)
        dpi = self.figure.get_dpi()
        self.figure.clear()
        self.figure.set_size_inches(width * ratio / dpi, height * ratio / dpi, forward=False)
        self.figure._panel_px = (width, height)     # a long legend title grows it (_grow_top)
        self.figure._title_lines = 1
        ax_w = max((width - MARGIN_LEFT - MARGIN_RIGHT - GAP_X * (ncol - 1)) / ncol, 50)
        ax_h = panel_h - TITLE_H - bottom
        grid = dict(left=MARGIN_LEFT / width, right=1 - MARGIN_RIGHT / width,
                    top=1 - (top + TITLE_H) / height, bottom=bottom / height,
                    wspace=GAP_X / ax_w, hspace=(TITLE_H + bottom) / ax_h)
        axes = self.figure.subplots(nrow, ncol, squeeze=False, gridspec_kw=grid).ravel().tolist()
        for ax in axes[n:]:
            ax.set_axis_off()
        return axes[:n]

    def message(self, text: str) -> None:
        ax = self.panels(1, legend_entries=0)[0]
        ax.text(0.5, 0.5, text, ha="center", va="center", color="0.45", fontsize=11,
                transform=ax.transAxes, wrap=True)
        ax.set_axis_off()

    def draw(self) -> None:
        self.canvas.draw_idle()


class FlowLayout(QLayout):
    """Widgets left to right, wrapping onto the next line (the Qt flow-layout example)."""

    def __init__(self, parent=None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:                    # noqa: N802 - Qt API
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):                           # noqa: N802 - Qt API
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):                           # noqa: N802 - Qt API
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):                      # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:                # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:        # noqa: N802 - Qt API
        return self._arrange(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect: QRect) -> None:         # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._arrange(rect, test=False)

    def sizeHint(self) -> QSize:                        # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self) -> QSize:                     # noqa: N802 - Qt API
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect: QRect, test: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        gap = self.spacing()
        for item in self._items:
            if item.isEmpty():              # a hidden widget takes no place
                continue
            hint = item.sizeHint()
            if x + hint.width() > rect.right() and line > 0:
                x, y, line = rect.x(), y + line + gap // 2, 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + gap
            line = max(line, hint.height())
        return y + line - rect.y()


def flow_row(widgets: list[QWidget], spacing: int = 14) -> QWidget:
    """Widgets in a line that wraps onto the next one when the window is narrow.

    A label followed by its field (``QLabel``, then a combo box) moves as one piece,
    so a name is never left at the end of a line with its field on the next.
    """
    holder = QWidget()
    policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    policy.setHeightForWidth(True)
    holder.setSizePolicy(policy)
    flow = FlowLayout(holder, spacing=spacing)
    i = 0
    while i < len(widgets):
        w = widgets[i]
        if isinstance(w, QLabel) and i + 1 < len(widgets) and not isinstance(widgets[i + 1], QLabel):
            pair = QWidget()
            lay = QHBoxLayout(pair)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(5)
            lay.addWidget(w)
            lay.addWidget(widgets[i + 1])
            flow.addWidget(pair)
            i += 2
        else:
            flow.addWidget(w)
            i += 1
    return holder


class ChannelPicker(QWidget):
    """Channel checkboxes that wrap onto a second line when there are many."""
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._boxes: dict[str, QCheckBox] = {}
        self._bulk = False
        self._holder = QWidget()
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self._holder.setSizePolicy(policy)
        self._flow = FlowLayout(self._holder, spacing=12)
        self.btn_all = QPushButton("все")
        self.btn_none = QPushButton("снять")
        for b, on in ((self.btn_all, True), (self.btn_none, False)):
            b.setFlat(True)
            b.setStyleSheet("color: #2166ac; padding: 0 4px;")
            b.clicked.connect(lambda _=False, on=on: self._set_all(on))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._holder, 1)
        lay.addWidget(self.btn_all, 0, Qt.AlignTop)
        lay.addWidget(self.btn_none, 0, Qt.AlignTop)

    def set_channels(self, names: list[str], checked: set[str] | None = None) -> None:
        names = [str(n) for n in names]
        if names == list(self._boxes):
            return                          # same channels: keep the boxes and their ticks
        before = {n: b.isChecked() for n, b in self._boxes.items()}
        while self._flow.count():
            item = self._flow.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # hidden and detached now: deleteLater alone leaves the old box painted
                # under the new ones until the event loop gets to it
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self._boxes = {}
        for name in names:
            box = QCheckBox(str(name))
            box.setChecked(before[name] if name in before else (checked is None or name in checked))
            box.toggled.connect(self._on_toggle)
            self._flow.addWidget(box)
            self._boxes[name] = box
        self._flow.invalidate()
        self._holder.updateGeometry()

    def _on_toggle(self, _on: bool) -> None:
        if not self._bulk:
            self.changed.emit()

    def _set_all(self, on: bool) -> None:
        self._bulk = True
        for box in self._boxes.values():
            box.setChecked(on)
        self._bulk = False
        self.changed.emit()

    def checked(self) -> list[str]:
        return [n for n, b in self._boxes.items() if b.isChecked()]

    def set_checked(self, name: str, on: bool) -> None:
        if name in self._boxes:
            self._boxes[name].setChecked(on)


class GroupPicker(QWidget):
    """Group A against group B, or every group at once."""
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.a, self.b = QComboBox(), QComboBox()
        for box in (self.a, self.b):
            box.setMinimumWidth(150)
            box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            box.currentIndexChanged.connect(lambda _: self._on_pick())
        self.all = QCheckBox("все группы")
        self.all.setToolTip("Показать все группы сразу, без сравнения двух.")
        self.all.toggled.connect(self._on_all)
        dot_a, dot_b = QLabel("●"), QLabel("●")
        dot_a.setStyleSheet(f"color: {COLOR_A}; font-size: 16px;")
        dot_b.setStyleSheet(f"color: {COLOR_B}; font-size: 16px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("<b>Сравнить</b>"), dot_a, self.a, QLabel("и"), dot_b, self.b, self.all):
            lay.addWidget(w)
        self._names: list[str] = []
        #: The pair the user chose last. A group list without one of them (another
        #: scenario, another configuration) shows a stand-in, and the choice comes back
        #: as soon as the list has those groups again.
        self._wanted: tuple = (None, None)

    def _on_pick(self) -> None:
        self._wanted = (self.a.currentData(), self.b.currentData())
        self.changed.emit()

    def _on_all(self, on: bool) -> None:
        self.a.setEnabled(not on)
        self.b.setEnabled(not on)
        self.changed.emit()

    def set_groups(self, info: list[tuple[str, int, int]]) -> None:
        """``(name, recordings, subjects)`` per group, in display order."""
        names = [g for g, _, _ in info]
        by_size = [g for g, _, _ in sorted(info, key=lambda t: -t[1])]
        default = [g for g in names if g in by_size[:2]]      # the two largest, in list order
        # Never from the combo boxes themselves: after a list with one group they both
        # hold it, and "SCI vs SCI" would outlive the scenario that forced it.
        want_a, want_b = self._wanted
        a = want_a if want_a in names else None
        b = want_b if want_b in names else None
        if a is None:
            a = next((g for g in default if g != b), names[0] if names else None)
        if b is None or b == a:
            b = next((g for g in default if g != a), next((g for g in names if g != a), a))
        for box, target in ((self.a, a), (self.b, b)):
            box.blockSignals(True)
            box.clear()
            for i, (g, n, s) in enumerate(info):
                box.addItem(f"{short_name(g, 30)}  ({n})", g)
                box.setItemData(i, f"{g}: {records(n)}, {subjects(s)}", Qt.ToolTipRole)
            if target is not None:
                box.setCurrentIndex(names.index(target))
            box.blockSignals(False)
        self._names = names

    def groups(self) -> list[str]:
        """The groups drawn: A and B, or all of them."""
        if self.all.isChecked():
            return list(self._names)
        out = [g for g in (self.a.currentData(), self.b.currentData()) if g is not None]
        return list(dict.fromkeys(out))

    def pair(self) -> tuple[str, str] | None:
        """(A, B) when two different groups are compared, else None."""
        if self.all.isChecked():
            return None
        a, b = self.a.currentData(), self.b.currentData()
        return (a, b) if a is not None and b is not None and a != b else None

    def colors(self) -> dict[str, str]:
        groups = self.groups()
        if self.pair() is not None:
            return {groups[0]: COLOR_A, groups[1]: COLOR_B}
        return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(self._names)}


class StatsTable(QTableWidget):
    """The table under the plots."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)

    def show_frame(self, table: pd.DataFrame | None, bold: list[bool] | None = None,
                   note: str = "") -> None:
        self.setUpdatesEnabled(False)
        try:
            if table is None or table.empty:
                self.setRowCount(0)
                self.setColumnCount(1)
                self.setHorizontalHeaderLabels([note or "нет данных для таблицы"])
                return
            self.setRowCount(len(table))
            self.setColumnCount(len(table.columns))
            self.setHorizontalHeaderLabels([str(c) for c in table.columns])
            font = QFont()
            font.setBold(True)
            for i, row in enumerate(table.itertuples(index=False)):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    if bold is not None and i < len(bold) and bold[i]:
                        item.setFont(font)
                    self.setItem(i, j, item)
            self.resizeColumnsToContents()
        finally:
            self.setUpdatesEnabled(True)


def comparison_display(table: pd.DataFrame, unit_col: str, unit_label: str,
                       a: str, b: str) -> tuple[pd.DataFrame, list[bool]]:
    """The comparison table as it is read on screen; bold where p < 0.05."""
    if table is None or table.empty:
        return pd.DataFrame(), []
    if short_name(a, 24) != short_name(b, 24):
        a, b = short_name(a, 24), short_name(b, 24)
    rows = []
    for _, r in table.iterrows():
        rows.append({
            unit_label: r[unit_col],
            f"{a}: записей (субъектов)": f"{r['n_a']} ({r['subjects_a']})",
            f"{a}: медиана [Q1–Q3]": _mq(r["median_a"], r["q1_a"], r["q3_a"]),
            f"{b}: записей (субъектов)": f"{r['n_b']} ({r['subjects_b']})",
            f"{b}: медиана [Q1–Q3]": _mq(r["median_b"], r["q1_b"], r["q3_b"]),
            "разница медиан (B − A)": ST.format_number(r["delta"]),
            "B / A": ST.format_number(r["ratio"]),
            "p": ST.format_p(r["p"]),
            "тест": r["test"],
        })
    bold = [bool(np.isfinite(p) and p < 0.05) for p in table["p"]]
    return pd.DataFrame(rows), bold


def groups_display(table: pd.DataFrame, unit_col: str, unit_label: str) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        unit_label: r[unit_col], "группа": short_name(r["group"], 40),
        "записей (субъектов)": f"{r['n']} ({r['subjects']})",
        "медиана [Q1–Q3]": _mq(r["median"], r["q1"], r["q3"]),
    } for _, r in table.iterrows()])


def _mq(med, q1, q3) -> str:
    if not np.isfinite(med):
        return ""
    if not (np.isfinite(q1) and np.isfinite(q3)):
        return ST.format_number(med)
    return f"{ST.format_number(med)} [{ST.format_number(q1)}–{ST.format_number(q3)}]"


class GroupTab(QWidget):
    """Layout and redraw logic of a group tab. Subclasses fill in the data and the views.

    A subclass sets ``VIEWS`` and ``EXPORT_PREFIX``, builds its left panel and its
    controls, sets ``view_options`` and calls ``build_layout``, and implements:

    * ``has_data()``;
    * ``prepare()`` — the analysis that the current controls imply (called once after
      ``invalidate()``, never on a plain redraw);
    * ``draw_view(key, view)`` — draw one view into its ``PlotView``;
    * ``stats_for(key)`` — ``(table, bold, note)`` for the table under the plots;
    * ``export_tables(folder)`` — write the tables, return their file names.

    ``view_options`` maps a view to the widgets that only make sense on it; only the
    visible view's options are shown.
    """
    VIEWS: list[tuple[str, str]] = []
    EXPORT_PREFIX = "group"

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.picker = GroupPicker()
        self.channels = ChannelPicker()
        self.stats = StatsTable()
        self.status = QLabel("Группа пуста: добавьте папку с записями.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #555; font-size: 11px;")
        self.btn_save = QPushButton("Сохранить рисунки и таблицы…")
        self.btn_save.clicked.connect(self._save)
        self.view_options: dict[str, list[QWidget]] = {}

        self.views: dict[str, PlotView] = {}
        self.tabs = QTabWidget()
        for key, title in self.VIEWS:
            view = PlotView()
            self.views[key] = view
            self.tabs.addTab(view, title)

        self._dirty: set[str] = set(self.views)
        self._prepared = False
        self._last_width = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh_now)
        self.tabs.currentChanged.connect(lambda _: self._timer.start())
        self.picker.changed.connect(self.redraw)
        self.channels.changed.connect(self.redraw)
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_threads)

    def background_threads(self) -> list:
        """The reader threads this tab may have running (subclasses list theirs)."""
        return []

    def _wait_for_threads(self) -> None:
        for thread in self.background_threads():
            if thread is not None and thread.isRunning():
                thread.wait(5000)

    # ---- layout ----
    def build_layout(self, left: QWidget, controls: list[QWidget]) -> None:
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 4, 6, 4)
        row1 = QHBoxLayout()
        row1.addWidget(self.picker)
        row1.addStretch(1)
        row1.addWidget(self.btn_save, 0, Qt.AlignTop)
        rv.addLayout(row1)
        # the controls and the view options wrap in a narrow window (a laptop screen)
        # instead of making the window wider than the screen
        rv.addWidget(flow_row(controls))
        row3 = QHBoxLayout()
        label = QLabel("<b>Каналы</b>")
        row3.addWidget(label, 0, Qt.AlignTop)
        row3.addWidget(self.channels, 1)
        rv.addLayout(row3)
        rv.addWidget(flow_row([w for widgets in self.view_options.values() for w in widgets]))
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(self.tabs)
        vsplit.addWidget(self.stats)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([760, 200])
        rv.addWidget(vsplit, 1)
        rv.addWidget(self.status)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1320])
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

    # ---- redraw logic ----
    def current_view(self) -> str:
        return self.VIEWS[self.tabs.currentIndex()][0] if self.VIEWS else ""

    def invalidate(self) -> None:
        """The analysis has to be redone (data, metric, normalisation, grouping)."""
        self._prepared = False
        self.redraw()

    def redraw(self) -> None:
        """Only the drawing changed (groups compared, channels, a view option)."""
        self._dirty = set(self.views)
        self._timer.start()

    def refresh_now(self) -> None:
        key = self.current_view()
        for k, widgets in self.view_options.items():
            for w in widgets:
                w.setVisible(k == key)
        if not self.has_data():
            return
        if not self._prepared:
            self.prepare()
            self._prepared = True
        if key in self._dirty:
            self._draw_one(key)
        table, bold, note = self.stats_for(key)
        self.stats.show_frame(table, bold, note)

    def _draw_one(self, key: str) -> None:
        view = self.views[key]
        try:
            self.draw_view(key, view)
        except Exception as exc:                        # noqa: BLE001 - drawn, not raised
            view.message(f"Не удалось нарисовать: {type(exc).__name__}: {exc}")
        view.draw()
        self._dirty.discard(key)

    def resizeEvent(self, event) -> None:               # noqa: N802 - Qt API
        super().resizeEvent(event)
        if abs(self.width() - self._last_width) > 30:
            self._last_width = self.width()
            self.redraw()

    # ---- saving ----
    def _save(self) -> None:
        if not self.has_data():
            return
        folder = QFileDialog.getExistingDirectory(self, "Куда сохранить рисунки и таблицы")
        if not folder:
            return
        out = Path(folder)
        if not self._prepared:
            self.prepare()
            self._prepared = True
        width = self.views[self.current_view()].scroll.viewport().width()
        written = []
        for key, _ in self.VIEWS:
            view = self.views[key]
            view.width_hint = width
            if key in self._dirty:
                self._draw_one(key)
            name = f"{self.EXPORT_PREFIX}_{key}.png"
            view.figure.savefig(out / name, dpi=150)
            written.append(name)
        written += self.export_tables(out)
        self.status.setText("Сохранено в «" + out.name + "»: " + ", ".join(written))

    # ---- to implement ----
    def has_data(self) -> bool:
        raise NotImplementedError

    def prepare(self) -> None:
        raise NotImplementedError

    def draw_view(self, key: str, view: PlotView) -> None:
        raise NotImplementedError

    def stats_for(self, key: str):
        return None, None, ""

    def export_tables(self, folder: Path) -> list[str]:
        return []


def member_table(headers: list[str], stretch_col: int) -> QTableWidget:
    """The members table on the left: checkbox, labels, recording name."""
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    # Interactive, not ResizeToContents: the latter re-measures every column on
    # every cell set, which made refilling a few hundred rows take most of a second.
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    t.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    t.setAlternatingRowColors(True)
    return t


def begin_fill(table: QTableWidget) -> None:
    table.blockSignals(True)
    table.setUpdatesEnabled(False)


def end_fill(table: QTableWidget) -> None:
    table.setUpdatesEnabled(True)
    table.blockSignals(False)
    table.resizeColumnsToContents()

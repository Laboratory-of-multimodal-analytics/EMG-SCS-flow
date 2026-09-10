"""Group analysis: several finished runs read as one experiment.

Every other surface in the toolbox looks at one recording. This one looks at a
set of them and asks what they say together — did the response grow between
sessions, is it the same muscles in every animal, how much of the spread is
between subjects rather than between states.

The screen is in three parts, left to right: WHICH runs are in the group and
what each one is (subject, state, baseline); WHAT to compare (channels,
configurations, metric, normalisation); and four views of the answer —

  * Наложение     — every run's curve on one axes, to look before averaging;
  * Групповое среднее — mean ± SE across subjects, on a shared current axis;
  * Боксплоты      — one number per curve, distributed across subjects, plus
                     the paired contrast between two states;
  * Формы ответов  — the waveforms themselves from different runs, superimposed.

All four share one pair of axis limits, computed over everything selected. That
is the point of the "единая развертка" box: comparing curves drawn at different
scales is the mistake this view exists to prevent, and auto-scaling each panel
to its own data reintroduces it silently.

The analysis lives in ``src/group.py`` and knows nothing about Qt; this file
only draws what it returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QMessageBox, QPushButton, QScrollArea, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

import src.group as G

#: One colour per state, in the order states are shown.
STATE_COLORS = ["#4575b4", "#f0a202", "#d73027", "#4d9221", "#7b3294",
                "#00838f", "#8d6e63", "#c2185b", "#5e35b1", "#33691e"]


def _state_colors(states: list[str]) -> dict[str, str]:
    return {s: STATE_COLORS[i % len(STATE_COLORS)] for i, s in enumerate(states)}


def _grid(n: int, ncol: int = 2) -> tuple[int, int]:
    ncol = min(ncol, max(n, 1))
    return int(np.ceil(n / ncol)), ncol


def key_of(fig, viewer) -> str:
    return next(k for k, f in viewer.figs.items() if f is fig)


class _Collector(QThread):
    """Reads the selected runs' metrics off disk without freezing the window.

    On a Google Drive mount the first read of a run's metrics table can take
    seconds, and a group is dozens of runs; doing that on the GUI thread makes
    the window look hung exactly when the user has just clicked something.
    """
    done = Signal(object, object)          # points, error
    progress = Signal(str)

    def __init__(self, runs, configs, channels):
        super().__init__()
        self._runs, self._configs, self._channels = runs, configs, channels

    def run(self) -> None:
        try:
            self.progress.emit(f"Читаю {len(self._runs)} прогон(ов)…")
            pts = G.collect_points(self._runs, self._configs, self._channels)
            self.done.emit(pts, None)
        except Exception as exc:                     # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class GroupViewer(QWidget):
    """The group tab."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.runs: list[G.GroupRun] = []
        self.points = pd.DataFrame()
        #: everything read from disk; ``points`` is this filtered by the pickers
        self.points_all = pd.DataFrame()
        self.normalised = pd.DataFrame()
        self.scalars = pd.DataFrame()
        self._missing: list[str] = []
        self._collector: _Collector | None = None
        #: state -> colour, fixed for the WHOLE group rather than per view. Each
        #: view sees a different subset of states — a state with no curve on the
        #: current grid is absent from the mean but still has a box — and a
        #: palette built per view then gives one state two colours across the
        #: tab, which is exactly the confusion this surface exists to remove.
        self._palette: dict[str, str] = {}

        # ---- runs ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip(
            "Найти внутри папки все посчитанные прогоны и добавить их в группу.")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Субъект", "Состояние", "Прогон"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_edit)

        # ---- what to compare ----
        self.config_list = QListWidget()
        self.config_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.config_list.setMaximumHeight(90)
        self.config_list.itemSelectionChanged.connect(self._on_pick)
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.itemSelectionChanged.connect(self._on_pick)

        self.metric_box = QComboBox()
        for key, label in G.METRICS.items():
            self.metric_box.addItem(label, key)
        self.norm_box = QComboBox()
        for key, label in G.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        # Raw µV by default: a normalisation whose baseline most subjects lack
        # turns the view into a few runaway ratios over a floor of zeros.
        self.norm_box.setCurrentIndex(0)
        self.base_box = QComboBox()
        self.base_box.setToolTip(
            "Состояние, принятое за базовое. Нормировка считается в нём, "
            "отдельно для каждого субъекта.")
        self.state_a = QComboBox()
        self.state_b = QComboBox()
        self.scalar_box = QComboBox()
        for key, label in G.SCALARS.items():
            self.scalar_box.addItem(label, key)

        for box in (self.metric_box, self.norm_box, self.base_box):
            box.currentIndexChanged.connect(lambda _: self._on_pick())
        for box in (self.scalar_box, self.state_a, self.state_b):
            box.currentIndexChanged.connect(lambda _: self._draw())

        self.btn_apply = QPushButton("Пересчитать")
        self.btn_apply.setToolTip("Прочитать отмеченные прогоны с диска заново. Выбор каналов, "
                                  "конфигураций, метрики и нормировки применяется сразу.")
        self.btn_apply.clicked.connect(self._recompute)
        self.btn_export = QPushButton("Выгрузить таблицы…")
        self.btn_export.clicked.connect(self._export)

        # ---- shared axes ----
        self.lock_axes = QCheckBox("единая развертка")
        self.lock_axes.setChecked(True)
        self.lock_axes.setToolTip(
            "Одни и те же пределы по X и Y на всех панелях и во всех видах.\n"
            "Без этого каждая панель масштабируется под себя и кривые разного "
            "размера выглядят одинаково.")
        self.lock_axes.toggled.connect(lambda _: self._draw())
        self.x_lo, self.x_hi = QDoubleSpinBox(), QDoubleSpinBox()
        self.y_lo, self.y_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for b in (self.x_lo, self.x_hi, self.y_lo, self.y_hi):
            b.setRange(-1e6, 1e6)
            b.setDecimals(2)
            b.valueChanged.connect(lambda _: self._draw())

        self.status = QLabel("Группа пуста.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 11px;")

        # ---- views ----
        self.figs, self.canvases = {}, {}
        self.views = QTabWidget()
        for key, title in (("overlay", "Наложение"), ("mean", "Групповое среднее"),
                           ("box", "Боксплоты"), ("waves", "Формы ответов")):
            fig = Figure(figsize=(7, 6), layout="constrained")
            canvas = FigureCanvasQTAgg(fig)
            self.figs[key], self.canvases[key] = fig, canvas
            # A scroll area rather than a canvas squeezed into the tab: with a
            # dozen channels each panel keeps a readable height and the view
            # scrolls, instead of every trace flattening into a line.
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(canvas)
            self.views.addTab(scroll, title)
        self.views.currentChanged.connect(lambda _: self._draw())

        self.stats = QTableWidget(0, 0)
        self.stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats.setMaximumHeight(190)

        self._build_layout()

    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Прогоны в группе</b>"))
        lv.addWidget(self.table, 2)
        lv.addWidget(QLabel("<b>Конфигурации</b>"))
        lv.addWidget(self.config_list)
        lv.addWidget(QLabel("<b>Каналы</b>"))
        lv.addWidget(self.channel_list, 1)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        for title, w in (("Метрика", self.metric_box),
                         ("Нормировка", self.norm_box),
                         ("Базовое состояние", self.base_box),
                         ("Скаляр для боксплотов", self.scalar_box),
                         ("Контраст: от", self.state_a),
                         ("Контраст: к", self.state_b)):
            mv.addWidget(QLabel(f"<b>{title}</b>"))
            mv.addWidget(w)
        mv.addWidget(self.btn_apply)
        mv.addWidget(self.btn_export)
        mv.addSpacing(8)
        mv.addWidget(self.lock_axes)
        for title, lo, hi in (("X", self.x_lo, self.x_hi), ("Y", self.y_lo, self.y_hi)):
            r = QHBoxLayout()
            r.addWidget(QLabel(title))
            r.addWidget(lo)
            r.addWidget(hi)
            mv.addLayout(r)
        mv.addStretch(1)
        mid.setMaximumWidth(260)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.views, 1)
        rv.addWidget(self.stats)
        rv.addWidget(self.status)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(mid)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(2, 3)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    # Building the group
    # ------------------------------------------------------------------ #
    def _scan(self) -> None:
        start = str(self.runs[-1].root.parent) if self.runs else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с посчитанными прогонами", start)
        if not folder:
            return
        found = G.scan_runs(Path(folder))
        known = {str(r.root) for r in self.runs}
        added = [r for r in found if str(r.root) not in known]
        self.runs.extend(added)
        self.runs.sort(key=lambda r: (r.subject, G._state_order_key(r.state)))
        self._fill_table()
        self.status.setText(
            f"Добавлено прогонов: {len(added)}. Всего в группе: {len(self.runs)}."
            + ("" if added else " Ничего нового не найдено — проверьте, что внутри "
                                "лежат посчитанные папки с Excel/Large_dataset_emg_response_metrics.csv.")
        )

    def add_run(self, root: Path) -> None:
        """Put the run currently open in the main window into the group."""
        root = Path(root)
        if not G.looks_like_run(root) or any(str(r.root) == str(root) for r in self.runs):
            return
        subject, state = G.parse_labels(root)
        self.runs.append(G.GroupRun(root, subject, state))
        self._fill_table()

    def _clear(self) -> None:
        self.runs, self.points = [], pd.DataFrame()
        self.normalised, self.scalars = pd.DataFrame(), pd.DataFrame()
        self._fill_table()
        for fig in self.figs.values():
            fig.clear()
        for c in self.canvases.values():
            c.draw_idle()
        self.stats.setRowCount(0)
        self.status.setText("Группа пуста.")

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.runs))
        for i, run in enumerate(self.runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if run.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            self.table.setItem(i, 1, QTableWidgetItem(run.subject))
            self.table.setItem(i, 2, QTableWidgetItem(run.state))
            path = QTableWidgetItem(run.root.name)
            path.setFlags(Qt.ItemIsEnabled)
            path.setToolTip(str(run.root))
            self.table.setItem(i, 3, path)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self._fill_states()

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not 0 <= row < len(self.runs):
            return
        run = self.runs[row]
        if item.column() == 0:
            run.include = item.checkState() == Qt.Checked
        elif item.column() == 1:
            run.subject = item.text().strip() or run.subject
        elif item.column() == 2:
            # Typed by hand precisely because the guess can be wrong, and a wrong
            # state label is the one error here that no statistic will reveal.
            run.state = item.text().strip() or run.state
        self._fill_states()

    def _fill_states(self) -> None:
        states = G.sort_states(r.state for r in self.runs if r.include)
        for box, keep_first in ((self.base_box, True), (self.state_a, True),
                                (self.state_b, False)):
            current = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(states)
            if current in states:
                box.setCurrentText(current)
            elif states:
                box.setCurrentIndex(0 if keep_first else min(1, len(states) - 1))
            box.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Reading and analysing
    # ------------------------------------------------------------------ #
    def _selected(self, widget: QListWidget) -> list[str] | None:
        picked = [i.text() for i in widget.selectedItems()]
        return picked or None

    def _recompute(self) -> None:
        active = [r for r in self.runs if r.include]
        if not active:
            self.status.setText("Ни один прогон не отмечен.")
            return
        if self._collector is not None and self._collector.isRunning():
            return
        self.btn_apply.setEnabled(False)
        self._collector = _Collector(active, None, None)
        self._collector.progress.connect(self.status.setText)
        self._collector.done.connect(self._on_collected)
        self._collector.start()

    def _on_collected(self, points, error) -> None:
        self.btn_apply.setEnabled(True)
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось собрать группу", error)
            return
        self.points_all = points if points is not None else pd.DataFrame()
        if self.points_all.empty:
            self.status.setText("В отмеченных прогонах нет ни одной точки.")
            self._blank("В отмеченных прогонах нет ни одной точки.")
            return
        self._fill_pickers()
        self._analyse()

    def _on_pick(self) -> None:
        """A picker or a combo changed: re-filter and redraw, nothing re-read."""
        if not self.points_all.empty:
            self._analyse()

    def _blank(self, message: str = "") -> None:
        """Clear every view — stale panels must never outlive the data they drew."""
        for key, fig in self.figs.items():
            fig.clear()
            if message:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, message, ha="center", va="center", color="gray", wrap=True)
                ax.set_axis_off()
            self.canvases[key].draw()
        self.stats.setRowCount(0)
        self.stats.setColumnCount(0)

    def _fill_pickers(self) -> None:
        """Offer what the group actually contains, keeping what was already picked."""
        for widget, column in ((self.config_list, "config"), (self.channel_list, "channel")):
            picked = {i.text() for i in widget.selectedItems()}
            values = sorted(self.points_all[column].dropna().unique().tolist(),
                            key=lambda s: (len(s), s))
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(values)
            for i in range(widget.count()):
                if widget.item(i).text() in picked:
                    widget.item(i).setSelected(True)
            widget.blockSignals(False)

    def _analyse(self) -> None:
        picked = self._selected(self.config_list)
        pts = self.points_all
        if picked:
            pts = pts[pts["config"].isin(set(picked))]
        self.points = pts
        if self.points.empty:
            self.status.setText("Для выбранных конфигураций нет точек.")
            self._blank("Для выбранных конфигураций нет точек.")
            return
        metric = self.metric_box.currentData()
        mode = self.norm_box.currentData()
        baseline = self.base_box.currentText()
        factors = G.normalisation_factors(self.points, baseline, mode, metric=metric)
        self.normalised, self._missing = G.apply_normalisation(self.points, factors, metric)
        self.scalars = G.curve_scalars(self.normalised)

        self._palette = _state_colors(
            G.sort_states(r.state for r in self.runs if r.include))
        inv = G.inventory(self.points)
        n_sub = self.points["subject"].nunique()
        bits = [f"{len(self.points)} точек · {n_sub} субъект(ов) · "
                f"{self.points['run'].nunique()} прогонов"]
        if mode != G.NORM_NONE and not metric.endswith("_ms"):
            bits.append(f"нормировка: {G.NORM_LABELS[mode]} ({baseline})")
        if self._missing:
            shown = ", ".join(self._missing[:6])
            more = f" и ещё {len(self._missing) - 6}" if len(self._missing) > 6 else ""
            bits.append(f"без базовой записи (исключены из нормировки): {shown}{more}")
        if not inv.empty:
            best = inv.iloc[0]
            bits.append(f"самая представленная пара: {best['config']} / {best['channel']} "
                        f"— {int(best['subjects'])} субъект(ов)")
        self.status.setText("  ·  ".join(bits))
        self._autoscale()
        self._draw()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _autoscale(self) -> None:
        """Fill the shared limits from everything selected, once."""
        if self.normalised.empty:
            return
        x = pd.to_numeric(self.normalised["x_value"], errors="coerce").dropna()
        y = pd.to_numeric(self.normalised["value"], errors="coerce").dropna()
        if x.empty or y.empty:
            return
        for box, val in ((self.x_lo, float(x.quantile(0.01))),
                         (self.x_hi, float(x.quantile(0.99))),
                         (self.y_lo, min(0.0, float(y.quantile(0.01)))),
                         (self.y_hi, float(y.quantile(0.99)) * 1.05)):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)

    def _apply_limits(self, ax, x: bool = True, y: bool = True) -> None:
        if not self.lock_axes.isChecked():
            return
        if x and self.x_hi.value() > self.x_lo.value():
            ax.set_xlim(self.x_lo.value(), self.x_hi.value())
        if y and self.y_hi.value() > self.y_lo.value():
            ax.set_ylim(self.y_lo.value(), self.y_hi.value())

    def _channels_to_draw(self) -> list[str]:
        picked = self._selected(self.channel_list)
        available = sorted(self.normalised["channel"].dropna().unique().tolist(),
                           key=lambda s: (len(s), s))
        return [c for c in available if picked is None or c in set(picked)]

    def _unit(self) -> str:
        return str(self.normalised["unit"].iloc[0]) if not self.normalised.empty else ""


    def _fit_canvas(self, key: str, nrow: int) -> None:
        """Give the canvas ~2.6 in per row of panels; the scroll area does the rest."""
        canvas = self.canvases[key]
        # The Qt canvas keeps figure.dpi multiplied by the device pixel ratio,
        # so widget sizes (logical pixels) must be scaled the same way before
        # they become inches — otherwise on a Retina screen the rendered buffer
        # covers a quarter of the widget and Qt hatches the rest.
        ratio = float(getattr(canvas, "device_pixel_ratio", 1.0) or 1.0)
        dpi = self.figs[key].get_dpi()
        height = int(max(nrow, 1) * 2.6 * dpi / ratio) + 60
        canvas.setMinimumHeight(height)
        width = max(canvas.width(), 600)
        self.figs[key].set_size_inches(width * ratio / dpi, height * ratio / dpi,
                                       forward=False)

    def _draw(self) -> None:
        if self.normalised.empty:
            return
        key = ["overlay", "mean", "box", "waves"][self.views.currentIndex()]
        fig = self.figs[key]
        fig.clear()
        if not self._channels_to_draw():
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Для выбранных каналов нет точек.", ha="center",
                    va="center", color="gray")
            ax.set_axis_off()
            self.canvases[key].draw()
            return
        try:
            getattr(self, f"_draw_{key}")(fig)
        except Exception as exc:                      # noqa: BLE001 - drawn, not raised
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}", ha="center", va="center",
                    color="#b00", wrap=True)
            ax.set_axis_off()
        self.canvases[key].draw()

    def _draw_overlay(self, fig) -> None:
        """Every run's own curve, before any averaging."""
        channels = self._channels_to_draw()
        if not channels:
            return
        states = G.sort_states(self.normalised["state"])
        colors = self._palette
        nrow, ncol = _grid(len(channels))
        self._fit_canvas(key_of(fig, self), nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = self.normalised[self.normalised["channel"] == ch]
            drawn = 0
            for (subject, state, config), grp in d.groupby(["subject", "state", "config"]):
                g = grp.dropna(subset=["x_value", "value"]).sort_values("x_value")
                if g.empty:
                    continue
                drawn += 1
                ax.plot(g["x_value"], g["value"], "-", lw=0.9,
                        color=colors.get(state, "0.5"), alpha=0.65)
                ax.plot(g["x_value"], g["value"], "o", ms=1.8,
                        color=colors.get(state, "0.5"), alpha=0.65)
            if not drawn:
                ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                        color="0.6", fontsize=9, transform=ax.transAxes)
            ax.set_title(f"{ch}  (n={drawn})", fontsize=9, loc="left")
            ax.grid(True, color="0.93", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        handles = [Line2D([], [], color=colors[s], label=s) for s in states]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(states), 6),
                   frameon=False, fontsize=8)
        fig.supxlabel("stimulation amplitude, mA", fontsize=9)
        fig.supylabel(self._unit(), fontsize=9)

    def _draw_mean(self, fig) -> None:
        """Mean ± SE across subjects, on one current axis."""
        channels = self._channels_to_draw()
        if not channels:
            return
        states = G.sort_states(self.normalised["state"])
        colors = self._palette
        grid = G.common_grid(self.normalised, 40)
        on_grid = G.interpolate_to_grid(self.normalised, grid)
        gm = G.group_mean(on_grid)
        nrow, ncol = _grid(len(channels))
        self._fit_canvas(key_of(fig, self), nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = gm[gm["channel"] == ch]
            for state in states:
                s = d[d["state"] == state].sort_values("x")
                if s.empty:
                    continue
                color = colors.get(state, "0.5")
                ax.plot(s["x"], s["mean"], "-", lw=1.6, color=color)
                ax.fill_between(s["x"], s["mean"] - s["se"], s["mean"] + s["se"],
                                color=color, alpha=0.18, lw=0)
                # n changes ALONG the axis: subjects drop out where their own
                # sweep stopped. The label says the most anyone had.
                ax.plot([], [], color=color, label=f"{state} (n≤{int(s['n'].max())})")
            if d.empty:
                ax.text(0.5, 0.5, "меньше двух субъектов", ha="center", va="center",
                        color="0.6", fontsize=9, transform=ax.transAxes)
            ax.set_title(ch, fontsize=9, loc="left")
            ax.grid(True, color="0.93", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0 and not d.empty:
                ax.legend(fontsize=7, frameon=False)
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supxlabel("stimulation amplitude, mA", fontsize=9)
        fig.supylabel(self._unit(), fontsize=9)

    def _draw_box(self, fig) -> None:
        """One number per curve, spread across subjects, plus the paired contrast."""
        if self.scalars.empty:
            return
        scalar = self.scalar_box.currentData()
        channels = self._channels_to_draw()
        states = G.sort_states(self.scalars["state"])
        colors = self._palette
        nrow, ncol = _grid(len(channels))
        self._fit_canvas(key_of(fig, self), nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = self.scalars[self.scalars["channel"] == ch]
            data, labels, used = [], [], []
            for state in states:
                vals = pd.to_numeric(d[d["state"] == state][scalar],
                                     errors="coerce").dropna()
                if vals.empty:
                    continue
                data.append(vals.to_numpy())
                labels.append(f"{state}\nn={len(vals)}")
                used.append(state)
            if not data:
                # An empty panel that keeps its title says "nothing here for
                # this channel"; a vanished panel says nothing at all.
                ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                        color="0.6", fontsize=9, transform=ax.transAxes)
                ax.set_title(ch, fontsize=9, loc="left")
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ("top", "right", "left", "bottom"):
                    ax.spines[side].set_visible(False)
                continue
            # tick_labels since matplotlib 3.9; labels before it.
            try:
                bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
            except TypeError:
                bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, state in zip(bp["boxes"], used):
                patch.set_facecolor(colors.get(state, "0.7"))
                patch.set_alpha(0.45)
            for med in bp["medians"]:
                med.set_color("0.15")
            # Every subject as a point on its box: with a dozen animals the box
            # hides how many there were and whether one of them carries it.
            for pos, vals in enumerate(data, start=1):
                jitter = (np.linspace(-0.13, 0.13, len(vals)) if len(vals) > 1
                          else np.array([0.0]))
                ax.plot(pos + jitter, vals, "o", ms=3, color="0.25", alpha=0.7, zorder=3)
            ax.set_title(ch, fontsize=9, loc="left")
            ax.tick_params(axis="x", labelsize=7 if len(used) <= 4 else 6)
            if len(used) > 4:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(35)
                    lab.set_ha("right")
            ax.grid(True, axis="y", color="0.93", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            self._apply_limits(ax, x=False, y=(scalar == "max"))
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supylabel(G.SCALARS.get(scalar, scalar), fontsize=9)
        self._fill_contrast(scalar)

    def _fill_contrast(self, scalar: str) -> None:
        a, b = self.state_a.currentText(), self.state_b.currentText()
        if not a or not b or a == b:
            self.stats.setRowCount(0)
            self.stats.setColumnCount(0)
            return
        table = G.contrast(self.scalars, a, b, scalar)
        if table.empty:
            self.stats.setRowCount(0)
            self.stats.setColumnCount(0)
            return
        picked = self._selected(self.channel_list)
        if picked:
            table = table[table["channel"].isin(set(picked))]
        self.stats.setRowCount(len(table))
        self.stats.setColumnCount(len(table.columns))
        self.stats.setHorizontalHeaderLabels([str(c) for c in table.columns])
        for i, (_, row) in enumerate(table.iterrows()):
            for j, col in enumerate(table.columns):
                v = row[col]
                text = f"{v:.3g}" if isinstance(v, (int, float, np.floating)) \
                    and not isinstance(v, bool) else str(v)
                item = QTableWidgetItem(text)
                if col == "baseline_is_constant" and bool(v):
                    item.setToolTip(
                        "Базовое состояние нормировано само на себя, поэтому его "
                        "значения равны 1 по построению. Это одновыборочная "
                        "проверка «отношение отличается от 1», а не сравнение "
                        "двух измерений.")
                self.stats.setItem(i, j, item)
        self.stats.resizeColumnsToContents()

    def _draw_waves(self, fig) -> None:
        """The response shapes themselves, from several runs on one time axis."""
        configs = self._selected(self.config_list) or sorted(
            self.normalised["config"].unique())
        channels = self._channels_to_draw()
        if not configs or not channels:
            return
        config, channel = configs[0], channels[0]
        active = [r for r in self.runs if r.include]
        times, loaded = G.collect_waveforms(active, config, channel)
        self._fit_canvas(key_of(fig, self), 1)
        ax = fig.add_subplot(111)
        if not len(times) or not loaded:
            ax.text(0.5, 0.5, "Нет сохранённых эпох для этой пары "
                              "конфигурация/канал.", ha="center", va="center",
                    color="gray")
            ax.set_axis_off()
            return
        colors = self._palette
        for d in loaded:
            run = d["run"]
            ax.plot(times * 1e3, d["wave"], lw=1.1,
                    color=colors.get(run.state, "0.5"), alpha=0.85)
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("ms from stimulus")
        ax.set_ylabel("µV")
        ax.set_title(f"{config} · {channel} — сильнейший ответ каждого прогона "
                     f"({len(loaded)} прогонов)", fontsize=10, loc="left")
        # One legend entry per STATE, not per run: forty-three named runs cover
        # the plot they are supposed to explain. Which run is which is a question
        # for the run table on the left, where the answer is already written.
        drawn = G.sort_states(d["run"].state for d in loaded)
        counts = {st: sum(1 for d in loaded if d["run"].state == st) for st in drawn}
        ax.legend(handles=[Line2D([], [], color=colors.get(st, "0.5"),
                                  label=f"{st} (n={counts[st]})") for st in drawn],
                  fontsize=7, frameon=False, ncol=min(len(drawn), 5),
                  loc="upper right")
        ax.grid(True, color="0.93", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # The time axis here is the INTERSECTION of the runs' own windows, so it
        # is already common; only y is shared with the other views on request.
        if self.lock_axes.isChecked():
            span = max(abs(np.nanmin([d["wave"].min() for d in loaded])),
                       abs(np.nanmax([d["wave"].max() for d in loaded])))
            if np.isfinite(span) and span > 0:
                ax.set_ylim(-span * 1.05, span * 1.05)

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        if self.normalised.empty:
            return
        folder = QFileDialog.getExistingDirectory(self, "Куда выгрузить таблицы группы")
        if not folder:
            return
        out = Path(folder)
        written = []
        self.normalised.to_csv(out / "group_points_long.csv", index=False)
        written.append("group_points_long.csv")
        if not self.scalars.empty:
            self.scalars.to_csv(out / "group_curve_scalars.csv", index=False)
            written.append("group_curve_scalars.csv")
        grid = G.common_grid(self.normalised, 40)
        gm = G.group_mean(G.interpolate_to_grid(self.normalised, grid))
        if not gm.empty:
            gm.to_csv(out / "group_mean_on_grid.csv", index=False)
            written.append("group_mean_on_grid.csv")
        a, b = self.state_a.currentText(), self.state_b.currentText()
        if a and b and a != b and not self.scalars.empty:
            c = G.contrast(self.scalars, a, b, self.scalar_box.currentData())
            if not c.empty:
                name = f"group_contrast_{a}_vs_{b}.csv".replace(" ", "_").replace("/", "-")
                c.to_csv(out / name, index=False)
                written.append(name)
        # The membership table is what the numbers cannot be rebuilt without:
        # which run was which subject in which state, as it was on screen.
        pd.DataFrame([{"subject": r.subject, "state": r.state, "included": r.include,
                       "run": str(r.root)} for r in self.runs]).to_csv(
            out / "group_membership.csv", index=False)
        written.append("group_membership.csv")
        self.status.setText("Выгружено: " + ", ".join(written))

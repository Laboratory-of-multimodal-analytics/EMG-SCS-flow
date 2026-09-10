"""Group analysis of Neurosoft recordings: many exports read as one experiment.

Every other surface looks at one recording. This one takes the processed
Neurosoft array — one export per (patient, level, protocol) — and asks what the
recordings of one STATE say together. A state is built from the tags read off
the recording name (level, cohort, scenario, side, …); the "Группировать по"
box says which tags make the state, and the labels stay editable in the table
because the names are hand-typed and the guess can be wrong.

Three views, left to right:

  * Наложение         — every recording's recruitment curve, one panel per channel;
  * Групповое среднее — mean ± SE across recordings of a state, on the curve axis;
  * Формы ответов     — the response waveforms: mean across recordings with a
                        95 % confidence band per state, individual traces faint.

A Neurosoft export carries no stimulation amplitude, so the x axis of the
curves is the curve number in the ramp. The analysis is in ``src/group.py``;
this file only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

import src.group as G

STATE_COLORS = ["#4575b4", "#d73027", "#f0a202", "#4d9221", "#7b3294",
                "#00838f", "#8d6e63", "#c2185b", "#5e35b1", "#33691e", "#ef6c00",
                "#0277bd"]
VIEWS = ["overlay", "mean", "waves"]


def _grid(n: int, ncol: int = 2) -> tuple[int, int]:
    if n > 8:
        ncol = 3
    ncol = min(ncol, max(n, 1))
    return int(np.ceil(n / ncol)), ncol


class _Collector(QThread):
    """Reads the selected runs' metrics off disk without freezing the window."""
    done = Signal(object, object)
    progress = Signal(str)

    def __init__(self, runs):
        super().__init__()
        self._runs = runs

    def run(self) -> None:
        try:
            self.progress.emit(f"Читаю {len(self._runs)} записей…")
            pts = G.collect_points(self._runs)
            self.done.emit(pts, None)
        except Exception as exc:                     # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class GroupViewer(QWidget):
    """The Neurosoft group tab."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.runs: list[G.GroupRun] = []
        self.points_all = pd.DataFrame()
        self.normalised = pd.DataFrame()
        self._missing: list[str] = []
        self._collector: _Collector | None = None
        self._palette: dict[str, str] = {}

        # ---- runs ----
        self.btn_dataset = QPushButton("Датасет Нейрософта")
        self.btn_dataset.setToolTip("Добавить все обработанные записи Нейрософта с "
                                    "лабораторного диска (травма + контроль).")
        self.btn_dataset.clicked.connect(self._add_dataset)
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_all = QPushButton("Все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("Никого")
        self.btn_none.clicked.connect(lambda: self._set_all(False))

        self.group_box = QComboBox()
        for key, label in G.GROUP_MODES.items():
            self.group_box.addItem(label, key)
        self.group_box.setToolTip("Из каких меток записи складывается её состояние. "
                                  "Состояния сравниваются между собой; метки в таблице "
                                  "можно править руками.")
        self.group_box.currentIndexChanged.connect(lambda _: self._regroup())

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["", "Субъект", "Когорта", "Уровень", "Сценарий", "Состояние", "Запись"])
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_edit)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.setToolTip("Какие каналы рисовать. Cmd-клик добавляет канал.")
        self.channel_list.itemSelectionChanged.connect(self._draw)

        # ---- what to compare ----
        self.metric_box = QComboBox()
        for key, label in G.METRICS.items():
            self.metric_box.addItem(label, key)
        self.norm_box = QComboBox()
        for key, label in G.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        self.base_box = QComboBox()
        self.base_box.setToolTip("Состояние, принятое за базовое: нормировка считается "
                                 "в нём, отдельно для каждого субъекта и канала.")
        for box in (self.metric_box, self.norm_box, self.base_box):
            box.currentIndexChanged.connect(lambda _: self._on_pick())
        self.show_runs = QCheckBox("показывать отдельные записи")
        self.show_runs.setChecked(True)
        self.show_runs.toggled.connect(lambda _: self._draw())

        self.btn_apply = QPushButton("Перечитать с диска")
        self.btn_apply.setToolTip("Прочитать отмеченные записи заново. Выбор каналов, "
                                  "метрики, группировки и нормировки применяется сразу.")
        self.btn_apply.clicked.connect(self._recompute)
        self.btn_export = QPushButton("Выгрузить таблицы…")
        self.btn_export.clicked.connect(self._export)

        self.lock_axes = QCheckBox("единая развертка")
        self.lock_axes.setChecked(True)
        self.lock_axes.setToolTip("Одни и те же пределы по X и Y на всех панелях и во всех видах.")
        self.lock_axes.toggled.connect(lambda _: self._draw())
        self.x_lo, self.x_hi = QDoubleSpinBox(), QDoubleSpinBox()
        self.y_lo, self.y_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for b in (self.x_lo, self.x_hi, self.y_lo, self.y_hi):
            b.setRange(-1e6, 1e6)
            b.setDecimals(1)
            b.valueChanged.connect(lambda _: self._draw())

        self.status = QLabel("Группа пуста.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 11px;")

        # ---- views ----
        self.figs, self.canvases = {}, {}
        self.views = QTabWidget()
        for key, title in (("overlay", "Наложение"), ("mean", "Групповое среднее"),
                           ("waves", "Формы ответов")):
            fig = Figure(figsize=(7, 6), layout="constrained")
            canvas = FigureCanvasQTAgg(fig)
            self.figs[key], self.canvases[key] = fig, canvas
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(canvas)
            self.views.addTab(scroll, title)
        self.views.currentChanged.connect(lambda _: self._draw())

        self.stats = QTableWidget(0, 0)
        self.stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats.setMaximumHeight(170)

        self._build_layout()

    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(self.btn_dataset)
        row.addWidget(self.btn_scan)
        lv.addLayout(row)
        row = QHBoxLayout()
        for b in (self.btn_all, self.btn_none, self.btn_clear):
            row.addWidget(b)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Группировать по</b>"))
        lv.addWidget(self.group_box)
        lv.addWidget(QLabel("<b>Записи в группе</b>"))
        lv.addWidget(self.table, 3)
        lv.addWidget(QLabel("<b>Каналы</b>"))
        lv.addWidget(self.channel_list, 1)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        for title, w in (("Метрика", self.metric_box),
                         ("Нормировка", self.norm_box),
                         ("Базовое состояние", self.base_box)):
            mv.addWidget(QLabel(f"<b>{title}</b>"))
            mv.addWidget(w)
        mv.addWidget(self.show_runs)
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
    def _mode(self) -> str:
        return self.group_box.currentData() or "level"

    def _add_runs(self, found: list[G.GroupRun]) -> int:
        known = {str(r.root) for r in self.runs}
        added = [r for r in found if str(r.root) not in known]
        self.runs.extend(added)
        self.runs.sort(key=lambda r: (r.tags.get("cohort", ""), r.subject, r.state))
        self._fill_table()
        return len(added)

    def _add_dataset(self) -> None:
        roots = [r for r in G.NEUROSOFT_ROOTS if r.exists()]
        if not roots:
            self.status.setText("Лабораторный диск не смонтирован — добавьте папку вручную.")
            return
        n = self._add_runs(G.scan_neurosoft(roots, self._mode()))
        self.status.setText(f"Добавлено записей: {n}. Всего в группе: {len(self.runs)}. "
                            "Отметьте нужные и нажмите «Перечитать с диска».")

    def _scan(self) -> None:
        start = str(self.runs[-1].root.parent) if self.runs else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с обработанными записями", start)
        if not folder:
            return
        n = self._add_runs(G.scan_neurosoft([Path(folder)], self._mode()))
        self.status.setText(
            f"Добавлено записей: {n}. Всего в группе: {len(self.runs)}."
            + ("" if n else " Ничего нового: внутри нет папок с "
                            "Excel/Large_dataset_emg_response_metrics.csv."))

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group."""
        root = Path(root)
        if not G.looks_like_run(root):
            return
        tags = G.parse_neurosoft(root)
        self._add_runs([G.GroupRun(root, tags["subject"],
                                   G.state_from_tags(tags, self._mode()), tags=tags)])

    def _clear(self) -> None:
        self.runs, self.points_all, self.normalised = [], pd.DataFrame(), pd.DataFrame()
        self._fill_table()
        self.channel_list.clear()
        self._blank()
        self.status.setText("Группа пуста.")

    def _set_all(self, on: bool) -> None:
        for r in self.runs:
            r.include = on
        self._fill_table()

    def _regroup(self) -> None:
        """The state definition changed: rebuild every state from its tags."""
        mode = self._mode()
        for r in self.runs:
            if r.tags:
                r.state = G.state_from_tags(r.tags, mode)
        if not self.points_all.empty:
            self.points_all["state"] = self.points_all["run"].map(
                {str(r.root): r.state for r in self.runs}).fillna(self.points_all["state"])
        self._fill_table()
        self._on_pick()

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.runs))
        for i, run in enumerate(self.runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if run.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            self.table.setItem(i, 1, QTableWidgetItem(run.subject))
            for col, key in ((2, "cohort"), (3, "level"), (4, "scenario")):
                item = QTableWidgetItem(run.tags.get(key, "—"))
                item.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(i, col, item)
            self.table.setItem(i, 5, QTableWidgetItem(run.state))
            path = QTableWidgetItem(run.root.name)
            path.setFlags(Qt.ItemIsEnabled)
            path.setToolTip(str(run.root))
            self.table.setItem(i, 6, path)
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
        elif item.column() == 5:
            run.state = item.text().strip() or run.state
            if not self.points_all.empty:
                self.points_all.loc[self.points_all["run"] == str(run.root), "state"] = run.state
        self._fill_states()
        if item.column() in (1, 5):
            self._on_pick()

    def _fill_states(self) -> None:
        states = G.sort_states(r.state for r in self.runs if r.include)
        current = self.base_box.currentText()
        self.base_box.blockSignals(True)
        self.base_box.clear()
        self.base_box.addItems(states)
        if current in states:
            self.base_box.setCurrentText(current)
        self.base_box.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Reading and analysing
    # ------------------------------------------------------------------ #
    def _picked_channels(self) -> list[str] | None:
        picked = [i.text() for i in self.channel_list.selectedItems()]
        return picked or None

    def _recompute(self) -> None:
        active = [r for r in self.runs if r.include]
        if not active:
            self.status.setText("Ни одна запись не отмечена.")
            return
        if self._collector is not None and self._collector.isRunning():
            return
        self.btn_apply.setEnabled(False)
        self._collector = _Collector(active)
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
            self.status.setText("В отмеченных записях нет ни одной точки.")
            self._blank("В отмеченных записях нет ни одной точки.")
            return
        # A Neurosoft export is one configuration; its name is the recording's,
        # so it must not split the group. One label for all of them.
        self.points_all["config"] = "Neurosoft"
        self._fill_channels()
        self._analyse()

    def _on_pick(self) -> None:
        if not self.points_all.empty:
            self._analyse()

    def _blank(self, message: str = "") -> None:
        for key, fig in self.figs.items():
            fig.clear()
            if message:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, message, ha="center", va="center", color="gray", wrap=True)
                ax.set_axis_off()
            self.canvases[key].draw()
        self.stats.setRowCount(0)
        self.stats.setColumnCount(0)

    def _fill_channels(self) -> None:
        picked = {i.text() for i in self.channel_list.selectedItems()}
        values = sorted(self.points_all["channel"].dropna().unique().tolist(),
                        key=lambda s: (len(s), s))
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_list.addItems(values)
        for i in range(self.channel_list.count()):
            if self.channel_list.item(i).text() in picked:
                self.channel_list.item(i).setSelected(True)
        # Nothing picked yet: start from the first channel rather than all
        # eight at once — one muscle at a time is how the overlay is read.
        if not picked and self.channel_list.count():
            self.channel_list.item(0).setSelected(True)
        self.channel_list.blockSignals(False)

    def _analyse(self) -> None:
        metric = self.metric_box.currentData()
        mode = self.norm_box.currentData()
        baseline = self.base_box.currentText()
        factors = G.normalisation_factors(self.points_all, baseline, mode, metric=metric)
        self.normalised, self._missing = G.apply_normalisation(self.points_all, factors, metric)
        self._palette = {s: STATE_COLORS[i % len(STATE_COLORS)] for i, s in enumerate(
            G.sort_states(r.state for r in self.runs if r.include))}
        n_sub = self.points_all["subject"].nunique()
        bits = [f"{self.points_all['run'].nunique()} записей · {n_sub} субъект(ов) · "
                f"{self.points_all['state'].nunique()} состояний"]
        if mode != G.NORM_NONE and not metric.endswith("_ms"):
            bits.append(f"нормировка: {G.NORM_LABELS[mode]} ({baseline})")
        if self._missing:
            shown = ", ".join(self._missing[:5])
            more = f" и ещё {len(self._missing) - 5}" if len(self._missing) > 5 else ""
            bits.append(f"без базовой записи: {shown}{more}")
        self.status.setText("  ·  ".join(bits))
        self._autoscale()
        self._fill_stats()
        self._draw()

    def _fill_stats(self) -> None:
        """Per state and channel: recordings, subjects, curves, and the metric's max/mean."""
        if self.normalised.empty:
            self.stats.setRowCount(0)
            return
        d = self.normalised.dropna(subset=["value"])
        picked = self._picked_channels()
        if picked:
            d = d[d["channel"].isin(set(picked))]
        if d.empty:
            self.stats.setRowCount(0)
            self.stats.setColumnCount(0)
            return
        peak = d.groupby(["state", "channel", "run"])["value"].max().reset_index()
        g = peak.groupby(["state", "channel"])
        table = pd.DataFrame({
            "записей": g["run"].nunique(),
            "максимум: медиана": g["value"].median(),
            "максимум: среднее": g["value"].mean(),
            "максимум: SD": g["value"].std(),
        }).reset_index()
        table["state"] = pd.Categorical(table["state"], G.sort_states(table["state"]))
        table = table.sort_values(["channel", "state"])
        self.stats.setRowCount(len(table))
        self.stats.setColumnCount(len(table.columns))
        self.stats.setHorizontalHeaderLabels([str(c) for c in table.columns])
        for i, (_, row) in enumerate(table.iterrows()):
            for j, col in enumerate(table.columns):
                v = row[col]
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    text = "" if pd.isna(v) else f"{v:.3g}"
                else:
                    text = str(v)
                self.stats.setItem(i, j, QTableWidgetItem(text))
        self.stats.resizeColumnsToContents()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _autoscale(self) -> None:
        if self.normalised.empty:
            return
        x = pd.to_numeric(self.normalised["x_value"], errors="coerce").dropna()
        y = pd.to_numeric(self.normalised["value"], errors="coerce").dropna()
        if x.empty or y.empty:
            return
        for box, val in ((self.x_lo, float(x.min())), (self.x_hi, float(x.max())),
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
        picked = self._picked_channels()
        available = sorted(self.normalised["channel"].dropna().unique().tolist(),
                           key=lambda s: (len(s), s))
        return [c for c in available if picked is None or c in set(picked)]

    def _unit(self) -> str:
        return str(self.normalised["unit"].iloc[0]) if not self.normalised.empty else ""

    def _fit_canvas(self, key: str, nrow: int) -> None:
        """~2.6 in per row of panels; the scroll area does the rest.

        The Qt canvas keeps figure.dpi multiplied by the device pixel ratio, so
        widget sizes (logical pixels) are scaled the same way before they
        become inches — otherwise on a Retina screen the rendered buffer covers
        a quarter of the widget and Qt hatches the rest.
        """
        canvas = self.canvases[key]
        ratio = float(getattr(canvas, "device_pixel_ratio", 1.0) or 1.0)
        dpi = self.figs[key].get_dpi()
        height = int(max(nrow, 1) * 2.6 * dpi / ratio) + 60
        # Never shorter than the viewport: a figure smaller than its widget
        # leaves the rest of the widget unpainted.
        parent = canvas.parentWidget()
        if parent is not None:
            height = max(height, parent.height())
        canvas.setMinimumHeight(height)
        width = max(canvas.width(), 600)
        self.figs[key].set_size_inches(width * ratio / dpi, height * ratio / dpi,
                                       forward=False)

    def _style(self, ax, title: str) -> None:
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, color="0.93", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _empty_panel(self, ax, title: str, text: str = "нет данных") -> None:
        ax.text(0.5, 0.5, text, ha="center", va="center", color="0.6", fontsize=9,
                transform=ax.transAxes)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    def _draw(self) -> None:
        if self.normalised.empty:
            return
        key = VIEWS[self.views.currentIndex()]
        fig = self.figs[key]
        fig.clear()
        try:
            if not self._channels_to_draw():
                self._fit_canvas(key, 1)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Выберите канал в списке слева.", ha="center",
                        va="center", color="gray")
                ax.set_axis_off()
            else:
                getattr(self, f"_draw_{key}")(fig)
        except Exception as exc:                      # noqa: BLE001 - drawn, not raised
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}", ha="center", va="center",
                    color="#b00", wrap=True)
            ax.set_axis_off()
        self.canvases[key].draw()
        self._fill_stats()

    def _draw_overlay(self, fig) -> None:
        """Every recording's own curve, before any averaging."""
        channels = self._channels_to_draw()
        states = G.sort_states(self.normalised["state"])
        nrow, ncol = _grid(len(channels))
        self._fit_canvas("overlay", nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = self.normalised[self.normalised["channel"] == ch]
            drawn = 0
            for (subject, state, run), grp in d.groupby(["subject", "state", "run"]):
                g = grp.dropna(subset=["x_value", "value"]).sort_values("x_value")
                if g.empty:
                    continue
                drawn += 1
                ax.plot(g["x_value"], g["value"], "-", lw=0.9, alpha=0.6,
                        color=self._palette.get(state, "0.5"))
            if not drawn:
                self._empty_panel(ax, ch)
                continue
            self._style(ax, f"{ch}  (записей: {drawn})")
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        handles = [Line2D([], [], color=self._palette.get(s, "0.5"), label=s) for s in states]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(states), 5),
                   frameon=False, fontsize=8)
        fig.supxlabel("номер кривой в развертке", fontsize=9)
        fig.supylabel(self._unit(), fontsize=9)

    def _draw_mean(self, fig) -> None:
        """Mean ± SE across recordings of a state, on the shared curve axis."""
        channels = self._channels_to_draw()
        states = G.sort_states(self.normalised["state"])
        grid = G.common_grid(self.normalised, 60, q=0.0)
        gm = G.group_mean(G.interpolate_to_grid(self.normalised, grid))
        nrow, ncol = _grid(len(channels))
        self._fit_canvas("mean", nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = gm[gm["channel"] == ch] if not gm.empty else gm
            if d.empty:
                self._empty_panel(ax, ch, "меньше двух записей")
                continue
            for state in states:
                s = d[d["state"] == state].sort_values("x")
                if s.empty:
                    continue
                color = self._palette.get(state, "0.5")
                ax.plot(s["x"], s["mean"], "-", lw=1.6, color=color,
                        label=f"{state} (n≤{int(s['n'].max())})")
                ax.fill_between(s["x"], s["mean"] - s["se"], s["mean"] + s["se"],
                                color=color, alpha=0.18, lw=0)
            self._style(ax, ch)
            ax.legend(fontsize=7, frameon=False)
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supxlabel("номер кривой в развертке", fontsize=9)
        fig.supylabel(f"{self._unit()}, среднее ± SE", fontsize=9)

    def _draw_waves(self, fig) -> None:
        """Response waveforms: per state the mean across recordings with a 95 % CI."""
        channels = self._channels_to_draw()
        active = [r for r in self.runs if r.include]
        nrow, ncol = _grid(len(channels))
        self._fit_canvas("waves", nrow)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            times, loaded = G.collect_waveforms(active, None, ch)
            summary = G.waveform_summary(times, loaded)
            if summary.empty:
                self._empty_panel(ax, ch, "нет сохранённых эпох")
                continue
            if self.show_runs.isChecked():
                for d in loaded:
                    ax.plot(times * 1e3, d["wave"], lw=0.5, alpha=0.25,
                            color=self._palette.get(d["run"].state, "0.5"))
            for state in G.sort_states(summary["state"]):
                s = summary[summary["state"] == state]
                color = self._palette.get(state, "0.5")
                n = int(s["n"].iloc[0])
                ax.plot(s["t"] * 1e3, s["mean"], lw=1.7, color=color,
                        label=f"{state} (n={n})")
                if n >= 2:
                    ax.fill_between(s["t"] * 1e3, s["lo"], s["hi"], color=color,
                                    alpha=0.2, lw=0)
            ax.axvline(0, color="0.4", lw=0.8, ls="--")
            # The y range follows the group band, not the loudest single trace:
            # one recording's 4 mV artifact would otherwise flatten every mean.
            lo, hi = float(summary["lo"].min()), float(summary["hi"].max())
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.25 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
            self._style(ax, f"{ch} — сильнейший ответ каждой записи")
            ax.legend(fontsize=7, frameon=False)
            ax.set_xlabel("мс от стимула", fontsize=8)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supylabel("µV, среднее по записям и 95 % ДИ", fontsize=9)

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        if self.normalised.empty:
            return
        folder = QFileDialog.getExistingDirectory(self, "Куда выгрузить таблицы группы")
        if not folder:
            return
        out = Path(folder)
        written = []
        self.normalised.to_csv(out / "neurosoft_group_points_long.csv", index=False)
        written.append("neurosoft_group_points_long.csv")
        grid = G.common_grid(self.normalised, 60, q=0.0)
        gm = G.group_mean(G.interpolate_to_grid(self.normalised, grid))
        if not gm.empty:
            gm.to_csv(out / "neurosoft_group_mean.csv", index=False)
            written.append("neurosoft_group_mean.csv")
        active = [r for r in self.runs if r.include]
        frames = []
        for ch in self._channels_to_draw():
            times, loaded = G.collect_waveforms(active, None, ch)
            s = G.waveform_summary(times, loaded)
            if not s.empty:
                s.insert(0, "channel", ch)
                frames.append(s)
        if frames:
            pd.concat(frames).to_csv(out / "neurosoft_group_waveforms_ci.csv", index=False)
            written.append("neurosoft_group_waveforms_ci.csv")
        pd.DataFrame([{"subject": r.subject, "state": r.state, "included": r.include,
                       **{k: v for k, v in r.tags.items() if k != "subject"},
                       "run": str(r.root)} for r in self.runs]).to_csv(
            out / "neurosoft_group_membership.csv", index=False)
        written.append("neurosoft_group_membership.csv")
        self.status.setText("Выгружено: " + ", ".join(written))

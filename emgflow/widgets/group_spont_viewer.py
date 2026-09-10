"""Group analysis of spontaneous EMG: several StartStop conditions read as one experiment.

Left: the members of the group (one row per run × condition) with editable
subject / state labels and a checkbox each. Middle: what to compare — metric,
normalisation, baseline state, time axis, contrast, channel aliases. Right:

  * Наложение        — every member's RMS time course, per channel;
  * Групповое среднее — mean ± SE across members on a shared axis;
  * Боксплоты        — one number per member per channel, distributed across
                       states, with group statistics and the contrast table;
  * Вспышки          — mean burst envelope per state (mean of members' means).

One pair of axis limits is shared by every panel and every view while the
"единая развертка" box is on. The analysis lives in ``src/group_spont.py``.
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
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

import src.group_spont as S

STATE_COLORS = ["#4575b4", "#d73027", "#f0a202", "#4d9221", "#7b3294",
                "#00838f", "#8d6e63", "#c2185b", "#5e35b1", "#33691e"]


def _grid(n: int, ncol: int = 2) -> tuple[int, int]:
    if n > 8:
        ncol = 3
    ncol = min(ncol, max(n, 1))
    return int(np.ceil(n / ncol)), ncol


class _Collector(QThread):
    done = Signal(object, object)
    progress = Signal(str)

    def __init__(self, members, aliases):
        super().__init__()
        self._members, self._aliases = members, aliases

    def run(self) -> None:
        try:
            self.progress.emit(f"Читаю {len(self._members)} условий…")
            summary = S.collect_summary(self._members, self._aliases)
            tc = S.collect_timecourses(self._members, self._aliases)
            env = S.collect_burst_envelopes(self._members, self._aliases)
            self.done.emit((summary, tc, env), None)
        except Exception as exc:                     # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class GroupSpontViewer(QWidget):
    """The spontaneous-EMG group tab."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.members: list[S.SpontMember] = []
        self.summary = pd.DataFrame()
        self.tc = pd.DataFrame()
        self.env = pd.DataFrame()
        self.norm = pd.DataFrame()
        self.norm_tc = pd.DataFrame()
        self._missing: list[str] = []
        self._collector: _Collector | None = None
        self._palette: dict[str, str] = {}

        # ---- members ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip("Найти внутри папки все прогоны StartStop со спонтанной ЭМГ "
                                 "и добавить каждое их условие в группу.")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_all = QPushButton("Все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("Никого")
        self.btn_none.clicked.connect(lambda: self._set_all(False))

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Субъект", "Состояние", "Условие", "Прогон"])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_edit)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.itemSelectionChanged.connect(self._draw)

        # ---- what to compare ----
        self.metric_box = QComboBox()
        for key, label in S.METRICS.items():
            self.metric_box.addItem(label, key)
        self.norm_box = QComboBox()
        for key, label in S.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        self.base_box = QComboBox()
        self.base_box.setToolTip("Состояние, принятое за базовое: нормировка считается в нём, "
                                 "отдельно для каждого субъекта и канала.")
        self.axis_box = QComboBox()
        self.axis_box.addItem("доля сегмента (0–1)", "frac")
        self.axis_box.addItem("секунды от начала", "t")
        self.state_a = QComboBox()
        self.state_b = QComboBox()
        self.aliases = QLineEdit()
        self.aliases.setPlaceholderText("BB R=Biceps R; ch1=TA L")
        self.aliases.setToolTip("Псевдонимы каналов: «имя в файле=общее имя», через «;». "
                                "Регистр, пробелы и подчёркивания не учитываются и так.")

        self.btn_apply = QPushButton("Пересчитать")
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
            b.setDecimals(2)
            b.valueChanged.connect(lambda _: self._draw())

        self.status = QLabel("Группа пуста.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 11px;")

        # ---- views ----
        self.figs, self.canvases = {}, {}
        self.views = QTabWidget()
        for key, title in (("overlay", "Наложение"), ("mean", "Групповое среднее"),
                           ("box", "Боксплоты"), ("bursts", "Вспышки")):
            fig = Figure(figsize=(7, 6), layout="constrained")
            canvas = FigureCanvasQTAgg(fig)
            self.figs[key], self.canvases[key] = fig, canvas
            self.views.addTab(canvas, title)
        self.views.currentChanged.connect(lambda _: self._draw())

        self.stats = QTableWidget(0, 0)
        self.stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats.setMaximumHeight(200)

        self._build_layout()

    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        for b in (self.btn_scan, self.btn_clear, self.btn_all, self.btn_none):
            row.addWidget(b)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Условия в группе</b>"))
        lv.addWidget(self.table, 3)
        lv.addWidget(QLabel("<b>Каналы</b>"))
        lv.addWidget(self.channel_list, 1)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        for title, w in (("Метрика", self.metric_box),
                         ("Нормировка", self.norm_box),
                         ("Базовое состояние", self.base_box),
                         ("Ось времени", self.axis_box),
                         ("Контраст: от", self.state_a),
                         ("Контраст: к", self.state_b),
                         ("Псевдонимы каналов", self.aliases)):
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
        mid.setMaximumWidth(270)

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
        start = str(self.members[-1].root.parent) if self.members else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с посчитанными прогонами", start)
        if not folder:
            return
        found = S.scan_members(Path(folder))
        known = {(str(m.root), m.condition) for m in self.members}
        added = [m for m in found if (str(m.root), m.condition) not in known]
        self.members.extend(added)
        self.members.sort(key=lambda m: (m.subject, m.state, m.condition))
        self._fill_table()
        self.status.setText(
            f"Добавлено условий: {len(added)}. Всего в группе: {len(self.members)}."
            + ("" if added else " Ничего нового: внутри нет папок StartStop со «Spontaneous EMG»."))

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group (all its conditions)."""
        found = S.scan_members(Path(root), max_depth=0)
        known = {(str(m.root), m.condition) for m in self.members}
        self.members.extend(m for m in found if (str(m.root), m.condition) not in known)
        self._fill_table()

    def _clear(self) -> None:
        self.members = []
        self.summary = self.tc = self.env = pd.DataFrame()
        self.norm = self.norm_tc = pd.DataFrame()
        self._fill_table()
        self.channel_list.clear()
        for fig in self.figs.values():
            fig.clear()
        for c in self.canvases.values():
            c.draw_idle()
        self.stats.setRowCount(0)
        self.status.setText("Группа пуста.")

    def _set_all(self, on: bool) -> None:
        for m in self.members:
            m.include = on
        self._fill_table()

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.members))
        for i, m in enumerate(self.members):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if m.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            self.table.setItem(i, 1, QTableWidgetItem(m.subject))
            self.table.setItem(i, 2, QTableWidgetItem(m.state))
            cond = QTableWidgetItem(m.condition)
            cond.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 3, cond)
            run = QTableWidgetItem(m.root.name)
            run.setFlags(Qt.ItemIsEnabled)
            run.setToolTip(str(m.root))
            self.table.setItem(i, 4, run)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self._fill_states()

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not 0 <= row < len(self.members):
            return
        m = self.members[row]
        if item.column() == 0:
            m.include = item.checkState() == Qt.Checked
        elif item.column() == 1:
            m.subject = item.text().strip() or m.subject
        elif item.column() == 2:
            m.state = item.text().strip() or m.state
        self._fill_states()

    def _fill_states(self) -> None:
        states = S.sort_states(m.state for m in self.members if m.include)
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
    def _picked_channels(self) -> list[str] | None:
        picked = [i.text() for i in self.channel_list.selectedItems()]
        return picked or None

    def _recompute(self) -> None:
        active = [m for m in self.members if m.include]
        if not active:
            self.status.setText("Ни одно условие не отмечено.")
            return
        if self._collector is not None and self._collector.isRunning():
            return
        self.btn_apply.setEnabled(False)
        self._collector = _Collector(active, S.parse_aliases(self.aliases.text()))
        self._collector.progress.connect(self.status.setText)
        self._collector.done.connect(self._on_collected)
        self._collector.start()

    def _on_collected(self, payload, error) -> None:
        self.btn_apply.setEnabled(True)
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось собрать группу", error)
            return
        self.summary, self.tc, self.env = payload
        if self.summary.empty:
            self.status.setText("В выбранных условиях нет сводных таблиц спонтанной ЭМГ.")
            return
        # Labels may have been edited after the members were read: the frames
        # carry the labels as they were at collection time, which is now.
        self._fill_channels()
        self._analyse()

    def _fill_channels(self) -> None:
        picked = {i.text() for i in self.channel_list.selectedItems()}
        names = sorted(self.summary["channel"].dropna().unique().tolist(),
                       key=lambda s: (len(s), s))
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_list.addItems(names)
        for i in range(self.channel_list.count()):
            if self.channel_list.item(i).text() in picked:
                self.channel_list.item(i).setSelected(True)
        self.channel_list.blockSignals(False)

    def _analyse(self) -> None:
        metric = self.metric_box.currentData()
        mode = self.norm_box.currentData()
        baseline = self.base_box.currentText()
        factors = (S.normalisation_factors(self.summary, baseline, metric="rms_uv")
                   if mode == S.NORM_BASELINE else pd.DataFrame())
        self.norm, self._missing = S.apply_normalisation(self.summary, factors, metric)
        if not self.tc.empty:
            # The time course is always RMS, so it is scaled by the RMS factor
            # whatever the metric chosen for the boxplots.
            self.norm_tc, _ = S.apply_normalisation(self.tc, factors, "rms_uv")
        else:
            self.norm_tc = pd.DataFrame()

        self._palette = {s: STATE_COLORS[i % len(STATE_COLORS)]
                         for i, s in enumerate(S.sort_states(m.state for m in self.members
                                                             if m.include))}
        n_sub = self.summary["subject"].nunique()
        bits = [f"{self.summary[['run', 'condition']].drop_duplicates().shape[0]} условий · "
                f"{n_sub} субъект(ов) · {self.summary['channel'].nunique()} каналов"]
        if mode == S.NORM_BASELINE and metric not in ("n_bursts", "burst_dur_s"):
            bits.append(f"нормировка на «{baseline}» по субъекту и каналу")
        if self._missing:
            shown = ", ".join(self._missing[:5])
            more = f" и ещё {len(self._missing) - 5}" if len(self._missing) > 5 else ""
            bits.append(f"без базовой записи (исключены из нормировки): {shown}{more}")
        if not self.env.empty:
            bits.append(f"огибающие вспышек: {self.env['channel'].nunique()} канал(ов)")
        self.status.setText("  ·  ".join(bits))
        self._autoscale()
        self._draw()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _autoscale(self) -> None:
        if self.norm_tc.empty:
            return
        axis = self.axis_box.currentData()
        x = pd.to_numeric(self.norm_tc[axis], errors="coerce").dropna()
        y = pd.to_numeric(self.norm_tc["value"], errors="coerce").dropna()
        if x.empty or y.empty:
            return
        for box, val in ((self.x_lo, float(x.min())), (self.x_hi, float(x.quantile(0.99))),
                         (self.y_lo, 0.0), (self.y_hi, float(y.quantile(0.99)) * 1.05)):
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

    def _channels_to_draw(self, frame: pd.DataFrame) -> list[str]:
        picked = self._picked_channels()
        available = sorted(frame["channel"].dropna().unique().tolist(),
                           key=lambda s: (len(s), s))
        return [c for c in available if picked is None or c in set(picked)]

    def _unit(self) -> str:
        return str(self.norm["unit"].iloc[0]) if not self.norm.empty else ""

    def _tc_unit(self) -> str:
        return str(self.norm_tc["unit"].iloc[0]) if not self.norm_tc.empty else "RMS, µV"

    def _xlabel(self) -> str:
        return "доля сегмента" if self.axis_box.currentData() == "frac" else "время, с"

    def _style(self, ax, title: str) -> None:
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, color="0.93", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _legend(self, fig, states) -> None:
        handles = [Line2D([], [], color=self._palette.get(s, "0.5"), label=s) for s in states]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(states), 6),
                   frameon=False, fontsize=8)

    def _draw(self) -> None:
        if self.norm.empty:
            return
        key = ["overlay", "mean", "box", "bursts"][self.views.currentIndex()]
        fig = self.figs[key]
        fig.clear()
        try:
            getattr(self, f"_draw_{key}")(fig)
        except Exception as exc:                      # noqa: BLE001 - drawn, not raised
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}", ha="center", va="center",
                    color="#b00", wrap=True)
            ax.set_axis_off()
        self.canvases[key].draw_idle()

    def _draw_overlay(self, fig) -> None:
        """Every member's RMS time course, per channel, before any averaging."""
        if self.norm_tc.empty:
            fig.add_subplot(111).text(0.5, 0.5, "Нет таблиц временного хода RMS.",
                                      ha="center", va="center", color="gray")
            return
        axis = self.axis_box.currentData()
        channels = self._channels_to_draw(self.norm_tc)
        states = S.sort_states(self.norm_tc["state"])
        nrow, ncol = _grid(len(channels))
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = self.norm_tc[self.norm_tc["channel"] == ch]
            n = 0
            for _, g in d.groupby(["run", "condition"]):
                g = g.dropna(subset=[axis, "value"]).sort_values(axis)
                if g.empty:
                    continue
                n += 1
                ax.plot(g[axis], g["value"], lw=0.9, alpha=0.7,
                        color=self._palette.get(str(g["state"].iloc[0]), "0.5"))
            self._style(ax, f"{ch}  (n={n})")
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        self._legend(fig, states)
        fig.supxlabel(self._xlabel(), fontsize=9)
        fig.supylabel(self._tc_unit(), fontsize=9)

    def _draw_mean(self, fig) -> None:
        """Mean ± SE across members on the shared axis — the «среднее средних»."""
        if self.norm_tc.empty:
            fig.add_subplot(111).text(0.5, 0.5, "Нет таблиц временного хода RMS.",
                                      ha="center", va="center", color="gray")
            return
        axis = self.axis_box.currentData()
        channels = self._channels_to_draw(self.norm_tc)
        states = S.sort_states(self.norm_tc["state"])
        gm = S.mean_timecourse(self.norm_tc, axis)
        nrow, ncol = _grid(len(channels))
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = gm[gm["channel"] == ch] if not gm.empty else gm
            for state in states:
                s = d[d["state"] == state].sort_values("x") if not d.empty else d
                if s.empty:
                    continue
                color = self._palette.get(state, "0.5")
                ax.plot(s["x"], s["mean"], lw=1.6, color=color,
                        label=f"{state} (n≤{int(s['n'].max())})")
                ax.fill_between(s["x"], s["mean"] - s["se"], s["mean"] + s["se"],
                                color=color, alpha=0.18, lw=0)
            self._style(ax, ch)
            if i == 0 and not d.empty:
                ax.legend(fontsize=7, frameon=False)
            self._apply_limits(ax)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supxlabel(self._xlabel(), fontsize=9)
        fig.supylabel(self._tc_unit(), fontsize=9)

    def _draw_box(self, fig) -> None:
        """One number per member per channel, across states; stats and contrast below."""
        channels = self._channels_to_draw(self.norm)
        states = S.sort_states(self.norm["state"])
        nrow, ncol = _grid(len(channels))
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = self.norm[self.norm["channel"] == ch]
            data, labels, used = [], [], []
            for state in states:
                vals = pd.to_numeric(d[d["state"] == state]["value"], errors="coerce").dropna()
                if vals.empty:
                    continue
                data.append(vals.to_numpy())
                labels.append(f"{state}\nn={len(vals)}")
                used.append(state)
            if not data:
                ax.set_axis_off()
                continue
            try:
                bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
            except TypeError:
                bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, state in zip(bp["boxes"], used):
                patch.set_facecolor(self._palette.get(state, "0.7"))
                patch.set_alpha(0.45)
            for med in bp["medians"]:
                med.set_color("0.15")
            for pos, vals in enumerate(data, start=1):
                jitter = (np.linspace(-0.13, 0.13, len(vals)) if len(vals) > 1
                          else np.array([0.0]))
                ax.plot(pos + jitter, vals, "o", ms=3, color="0.25", alpha=0.7, zorder=3)
            self._style(ax, ch)
            ax.tick_params(axis="x", labelsize=7)
            self._apply_limits(ax, x=False, y=(self.metric_box.currentData() == "rms_uv"))
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supylabel(self._unit(), fontsize=9)
        self._fill_stats()

    def _fill_stats(self) -> None:
        """Group statistics per (channel, state), then the contrast rows underneath."""
        picked = self._picked_channels()
        stats = S.group_stats(self.norm)
        if picked and not stats.empty:
            stats = stats[stats["channel"].isin(set(picked))]
        a, b = self.state_a.currentText(), self.state_b.currentText()
        con = S.contrast(self.norm, a, b) if a and b and a != b else pd.DataFrame()
        if picked and not con.empty:
            con = con[con["channel"].isin(set(picked))]
        if stats.empty and con.empty:
            self.stats.setRowCount(0)
            self.stats.setColumnCount(0)
            return
        if not con.empty:
            con = con.copy()
            con.insert(1, "state", f"{a} → {b}")
        table = pd.concat([stats, con], ignore_index=True, sort=False) \
            if not con.empty else stats
        self.stats.setRowCount(len(table))
        self.stats.setColumnCount(len(table.columns))
        self.stats.setHorizontalHeaderLabels([str(c) for c in table.columns])
        for i, (_, row) in enumerate(table.iterrows()):
            for j, col in enumerate(table.columns):
                v = row[col]
                if isinstance(v, float) and np.isnan(v):
                    text = ""
                elif isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    text = f"{v:.3g}"
                else:
                    text = str(v)
                self.stats.setItem(i, j, QTableWidgetItem(text))
        self.stats.resizeColumnsToContents()

    def _draw_bursts(self, fig) -> None:
        """Mean burst envelope per state: mean of the members' own means."""
        if self.env.empty:
            fig.add_subplot(111).text(
                0.5, 0.5, "В выбранных условиях нет сохранённых огибающих вспышек.",
                ha="center", va="center", color="gray")
            return
        env = self.env
        if self.norm_box.currentData() == S.NORM_BASELINE:
            factors = S.normalisation_factors(self.summary, self.base_box.currentText(),
                                              metric="rms_uv")
            env, _ = S.apply_normalisation(self.env.rename(columns={"value": "raw"}),
                                           factors, "raw")
        channels = self._channels_to_draw(env)
        states = S.sort_states(env["state"])
        gm = S.mean_burst_envelope(env)
        nrow, ncol = _grid(len(channels))
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // ncol][i % ncol]
            d = gm[gm["channel"] == ch]
            for state in states:
                s = d[d["state"] == state].sort_values("t")
                if s.empty:
                    continue
                color = self._palette.get(state, "0.5")
                ax.plot(s["t"], s["mean"], lw=1.6, color=color,
                        label=f"{state} (n={int(s['n'].max())})")
                ax.fill_between(s["t"], s["mean"] - s["se"], s["mean"] + s["se"],
                                color=color, alpha=0.18, lw=0)
            ax.axvline(0, color="0.4", lw=0.8, ls="--")
            self._style(ax, ch)
            if not d.empty:
                ax.legend(fontsize=7, frameon=False)
            self._apply_limits(ax, x=False)
        for j in range(len(channels), nrow * ncol):
            axes[j // ncol][j % ncol].set_axis_off()
        fig.supxlabel("время от центра вспышки, с", fontsize=9)
        fig.supylabel(self._tc_unit(), fontsize=9)

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        if self.norm.empty:
            return
        folder = QFileDialog.getExistingDirectory(self, "Куда выгрузить таблицы группы")
        if not folder:
            return
        out = Path(folder)
        written = []
        self.norm.to_csv(out / "spont_group_summary_long.csv", index=False)
        written.append("spont_group_summary_long.csv")
        stats = S.group_stats(self.norm)
        if not stats.empty:
            stats.to_csv(out / "spont_group_stats.csv", index=False)
            written.append("spont_group_stats.csv")
        if not self.norm_tc.empty:
            gm = S.mean_timecourse(self.norm_tc, self.axis_box.currentData())
            if not gm.empty:
                gm.to_csv(out / "spont_group_mean_timecourse.csv", index=False)
                written.append("spont_group_mean_timecourse.csv")
        if not self.env.empty:
            S.mean_burst_envelope(self.env).to_csv(
                out / "spont_group_mean_burst_envelope.csv", index=False)
            written.append("spont_group_mean_burst_envelope.csv")
        a, b = self.state_a.currentText(), self.state_b.currentText()
        if a and b and a != b:
            c = S.contrast(self.norm, a, b)
            if not c.empty:
                name = f"spont_group_contrast_{a}_vs_{b}.csv".replace(" ", "_").replace("/", "-")
                c.to_csv(out / name, index=False)
                written.append(name)
        pd.DataFrame([{"subject": m.subject, "state": m.state, "included": m.include,
                       "condition": m.condition, "run": str(m.root)}
                      for m in self.members]).to_csv(out / "spont_group_membership.csv",
                                                     index=False)
        written.append("spont_group_membership.csv")
        self.status.setText("Выгружено: " + ", ".join(written))

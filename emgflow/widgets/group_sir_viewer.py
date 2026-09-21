"""Group analysis of standard stimulation runs: pigs, patients, the high-quality dataset.

A standard SIR run is cut per (electrode configuration, stimulation amplitude), with
several stimuli at each amplitude, so its recruitment curve lies on the real current
axis in mA — unlike a Neurosoft export, whose only axis is the curve number (that is
the "Group: Neurosoft" tab). A recording gets a subject and a group read from its
folder path (the pig array: "pig 5_2 30 day after SCI" → subject 5_2, group "30 day";
a patient folder → the patient and the session); both can be edited by hand.

Electrode configurations differ between sessions (1+8 is in 44 of 52 pig sessions,
3+6 in five), and curves from different pairs are different stimulations, so one
configuration is compared at a time; its list says in how many recordings each is.

Three views, channel by channel, group A (blue) against group B (red) or all groups:

* Кривые рекрутирования — every recording's curve on the shared current axis, and each
  group's median and quartiles where at least three recordings and 30 % of the group
  were measured. Inside a recording's own range an amplitude with no detection counts
  as no response (0).
* Показатели кривой — one number per recording: the maximal response, the threshold
  current (the lowest current reaching 10 % of that curve's own maximum) or the current
  of the maximal response. The table under the plots compares the groups on the chosen
  number in every view — paired within subjects when the same animals are in both.
* Формы ответов — the mean response at each recording's strongest current, and each
  group's mean with a 95 % confidence band.

Recordings are read once, in the background; everything else works in memory. The
analysis is in src/group.py and src/group_stats.py; this file only draws.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QSpinBox,
                               QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

import src.group as G
import src.group_stats as ST

from .group_common import (records, subjects, GroupTab, begin_fill, comparison_display, dots_and_box, empty_panel,
                           end_fill, group_ticks, groups_display, label_space, median_band,
                           member_table, style_axes, top_legend)

METRIC_LABELS = {
    "amp_uv": "размах ответа, мкВ",
    "area_uvms": "площадь ответа, мкВ·мс",
    "onset_ms": "латентность начала, мс",
    "p1_ms": "латентность P1, мс",
}
NORM_UNITS = {
    G.NORM_BASELINE_MAX: "доля максимума базовой группы",
    G.NORM_BASELINE_TOPN: "доля top-3 базовой группы",
    G.NORM_OWN_MAX: "доля максимума своей записи",
}
#: The number a curve is compared on: for response sizes, and for latencies.
SUMMARIES_SIZE = {"max": "максимальный ответ", "threshold_x": "порог, мА",
                  "at_max_x": "ток максимального ответа, мА"}
SUMMARIES_LATENCY = {"at_max_value": "латентность при максимальном ответе",
                     "threshold_x": "порог, мА", "at_max_x": "ток максимального ответа, мА"}
VIEWS = [("curves", "Кривые рекрутирования"), ("summary", "Показатели кривой"),
         ("waves", "Формы ответов")]


class _PointReader(QThread):
    progress = Signal(int, int)
    done = Signal(object, object)

    def __init__(self, runs) -> None:
        super().__init__()
        self._runs = runs

    def run(self) -> None:
        try:
            points = G.collect_sir_points(self._runs, progress=lambda i, n: self.progress.emit(i, n))
            self.done.emit(points, None)
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class _WaveReader(QThread):
    progress = Signal(int, int)
    done = Signal(object)

    def __init__(self, jobs: list[tuple[str, str, dict]]) -> None:
        super().__init__()
        self._jobs = jobs

    def run(self) -> None:
        out = {}
        for i, (root, config, labels) in enumerate(self._jobs, 1):
            try:
                out[(root, config)] = (labels, G.peak_waveforms(root, config, labels))
            except Exception:                          # noqa: BLE001 - that recording has none
                out[(root, config)] = (labels, (None, {}))
            self.progress.emit(i, len(self._jobs))
        self.done.emit(out)


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_") or "group"


class GroupSIRViewer(GroupTab):
    """The group tab for standard stimulation runs."""
    VIEWS = VIEWS
    EXPORT_PREFIX = "sir_group"

    def __init__(self, session=None) -> None:
        super().__init__(session)
        self.runs: list[G.GroupRun] = []
        self.points_all = pd.DataFrame()
        self._pending: list[G.GroupRun] = []
        self._reader: _PointReader | None = None
        self._wave_reader: _WaveReader | None = None
        self._waves: dict = {}
        self.values = pd.DataFrame()
        self.summaries = pd.DataFrame()
        self.grid = np.array([])
        self._missing: list[str] = []
        self._summary_status = ""
        self._short: list[str] = []

        # ---- left: the recordings ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip("Найти в папке все обработанные записи SIR (не Neurosoft) и "
                                 "добавить их; субъект и группа читаются из пути.")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.group_box = QComboBox()
        for key, label in G.SIR_GROUP_MODES.items():
            self.group_box.addItem(label, key)
        self.group_box.currentIndexChanged.connect(lambda _: self._regroup())
        self.table = member_table(["", "Субъект", "Группа", "Подгруппа", "Запись"], stretch_col=4)
        self.table.setToolTip("Галочка — запись участвует. Субъекта, группу и подгруппу можно "
                              "поправить двойным щелчком.")
        self.table.itemChanged.connect(self._on_table_edit)
        self.btn_all = QPushButton("отметить все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("снять все")
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        self.min_amps = QSpinBox()
        self.min_amps.setRange(2, 30)
        self.min_amps.setValue(5)
        self.min_amps.setSuffix(" токов")
        self.min_amps.setToolTip("Запись, в которой при выбранной конфигурации измерено меньше "
                                 "токов, не участвует: у кривой из пары токов нет ни максимума, "
                                 "ни порога (у свиней на 60 day есть сессии из 1–3 токов).")
        self.min_amps.valueChanged.connect(lambda _: self.invalidate())
        self.split_sub = QCheckBox("делить группы по подгруппам")
        self.split_sub.setToolTip("Запись с заполненной подгруппой сравнивается в группе «группа · подгруппа»; записи без подгруппы остаются в своей группе.")
        self.split_sub.toggled.connect(lambda _: self._membership_changed(refill=True))
        self.members_note = QLabel("")
        self.members_note.setStyleSheet("color: #555; font-size: 11px;")

        # ---- top: what is compared ----
        self.config_box = QComboBox()
        self.config_box.setMinimumWidth(120)
        self.config_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.config_box.setToolTip("Конфигурация электродов. Кривые разных пар — разные "
                                   "стимуляции, поэтому сравнивается одна; в скобках — "
                                   "в скольких отмеченных записях она есть.")
        self.config_box.currentIndexChanged.connect(lambda _: self._on_config())
        self.metric_box = QComboBox()
        for key, label in METRIC_LABELS.items():
            self.metric_box.addItem(label, key)
        self.summary_box = QComboBox()
        self.summary_box.setToolTip("Одно число на кривую: его сравнивает таблица и вид "
                                    "«Показатели кривой».")
        self._fill_summaries()
        self.norm_box = QComboBox()
        for key, label in G.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        self.norm_box.setToolTip("Нормировка считается отдельно для каждого субъекта и канала. "
                                 "Латентности и токи не нормируются.")
        self.base_label = QLabel("базовая группа")
        self.base_box = QComboBox()
        self.metric_box.currentIndexChanged.connect(lambda _: (self._fill_summaries(),
                                                               self._on_analysis_control()))
        self.summary_box.currentIndexChanged.connect(lambda _: self.redraw())
        for box in (self.norm_box, self.base_box):
            box.currentIndexChanged.connect(lambda _: self._on_analysis_control())

        # ---- options that belong to one view ----
        self.show_curves = QCheckBox("отдельные записи")
        self.show_curves.toggled.connect(lambda _: self.redraw())
        self.show_traces = QCheckBox("отдельные записи")
        self.show_traces.toggled.connect(lambda _: self.redraw())
        self.view_options = {"curves": [self.show_curves], "summary": [],
                             "waves": [self.show_traces]}

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Группировать по</b>"))
        lv.addWidget(self.group_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Кривая не короче</b>"))
        row.addWidget(self.min_amps)
        row.addStretch(1)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Записи</b>"))
        lv.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        lv.addLayout(row)
        lv.addWidget(self.split_sub)
        lv.addWidget(self.members_note)

        self.build_layout(left, [QLabel("<b>Конфигурация</b>"), self.config_box,
                                 QLabel("<b>Показатель</b>"), self.metric_box,
                                 QLabel("<b>Число кривой</b>"), self.summary_box,
                                 QLabel("<b>Нормировка</b>"), self.norm_box,
                                 self.base_label, self.base_box])
        self._sync_base_visibility()
        self.refresh_now()

    def background_threads(self) -> list:
        return [self._reader, self._wave_reader]

    # ------------------------------------------------------------------ #
    # The recordings
    # ------------------------------------------------------------------ #
    def _mode(self) -> str:
        return self.group_box.currentData() or "state"

    def _scan(self) -> None:
        start = str(self.runs[-1].root.parent) if self.runs else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с обработанными записями SIR", start)
        if folder:
            n = self.add_runs(G.scan_standard_runs(Path(folder)))
            if not n:
                self.status.setText("Новых обработанных записей SIR в этой папке нет "
                                    "(записи Neurosoft — во вкладке «Group: Neurosoft»).")

    def add_runs(self, found: list[G.GroupRun]) -> int:
        known = {str(r.root) for r in self.runs}
        new = [r for r in found if str(r.root) not in known]
        if not new:
            return 0
        mode = self._mode()
        for r in new:
            if not r.tags:
                r.tags = {"subject": r.subject, "state": r.state}
            r.state = G.sir_group_of(r, mode)
        self.runs.extend(new)
        self.runs.sort(key=lambda r: (r.subject, G._state_order_key(r.tags.get("state", r.state)),
                                      r.root.name))
        self._pending.extend(new)
        self._read_pending()
        self._membership_changed(refill=True)
        return len(new)

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group."""
        root = Path(root)
        if G.looks_like_run(root) and G.is_standard_sir_run(root):
            subject, state = G.parse_labels(root)
            self.add_runs([G.GroupRun(root, subject, state,
                                      tags={"subject": subject, "state": state})])

    def _read_pending(self) -> None:
        if (self._reader is not None and self._reader.isRunning()) or not self._pending:
            return
        batch, self._pending = self._pending, []
        self.status.setText(f"Читаю {records(len(batch))}…")
        self._reader = _PointReader(batch)
        self._reader.progress.connect(lambda i, n: self.status.setText(f"Читаю записи: {i} из {n}…"))
        self._reader.done.connect(self._on_points)
        self._reader.start()

    def _on_points(self, points, error) -> None:
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось прочитать записи", error)
        elif points is not None and not points.empty:
            self.points_all = (points if self.points_all.empty
                               else pd.concat([self.points_all, points], ignore_index=True))
        self._read_pending()
        self._membership_changed()

    def _clear(self) -> None:
        self.runs, self._pending, self._waves = [], [], {}
        self.points_all = self.values = self.summaries = pd.DataFrame()
        self._fill_table()
        self.picker.set_groups([])
        self.channels.set_channels([])
        self.config_box.blockSignals(True)
        self.config_box.clear()
        self.config_box.blockSignals(False)
        for view in self.views.values():
            view.message("Группа пуста: добавьте папку с записями.")
            view.draw()
        self.stats.show_frame(None)
        self.members_note.setText("")
        self.status.setText("Группа пуста: добавьте папку с записями.")

    def _regroup(self) -> None:
        mode = self._mode()
        for r in self.runs:
            r.state = G.sir_group_of(r, mode)
        self._membership_changed(refill=True)

    def _fill_table(self) -> None:
        begin_fill(self.table)
        self.table.setRowCount(len(self.runs))
        for i, r in enumerate(self.runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if r.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            self.table.setItem(i, 1, QTableWidgetItem(r.subject))
            self.table.setItem(i, 2, QTableWidgetItem(r.state))
            self.table.setItem(i, 3, QTableWidgetItem(r.subgroup))
            name = QTableWidgetItem(r.root.name)
            name.setFlags(Qt.ItemIsEnabled)
            name.setToolTip(str(r.root))
            self.table.setItem(i, 4, name)
        end_fill(self.table)

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        if not 0 <= item.row() < len(self.runs):
            return
        run = self.runs[item.row()]
        if item.column() == 0:
            run.include = item.checkState() == Qt.Checked
        elif item.column() == 1:
            run.subject = item.text().strip() or run.subject
            if not self.points_all.empty:
                self.points_all.loc[self.points_all["run"] == str(run.root), "subject"] = run.subject
        elif item.column() == 2:
            run.state = item.text().strip() or run.state
        elif item.column() == 3:
            run.subgroup = item.text().strip()
        else:
            return
        self._membership_changed()

    def _set_all(self, on: bool) -> None:
        for r in self.runs:
            r.include = on
        self._membership_changed(refill=True)

    def _active_runs(self) -> list[G.GroupRun]:
        return [r for r in self.runs if r.include]

    def _group_of(self, member) -> str:
        """The group a recording is compared in: its group, split by the hand-typed subgroup."""
        sub = (member.subgroup or "").strip()
        return f"{member.state} · {sub}" if sub and self.split_sub.isChecked() else member.state

    def _membership_changed(self, refill: bool = False) -> None:
        if refill:
            self._fill_table()
        active = self._active_runs()
        roots = {str(r.root) for r in active}
        pts = self.points_all[self.points_all["run"].isin(roots)] if not self.points_all.empty \
            else self.points_all

        # configurations: most recordings first
        current = self.config_box.currentData()
        counts = (pts.groupby("config")["run"].nunique().sort_values(ascending=False)
                  if not pts.empty else pd.Series(dtype=int))
        self.config_box.blockSignals(True)
        self.config_box.clear()
        for cfg, n in counts.items():
            self.config_box.addItem(f"{cfg}  ({n})", cfg)
        if current in counts.index:
            self.config_box.setCurrentIndex(list(counts.index).index(current))
        self.config_box.blockSignals(False)

        self._sync_groups_and_channels()
        self.members_note.setText(f"отмечено {len(active)} из {len(self.runs)} записей · "
                                  f"{subjects(len({r.subject for r in active}))}")
        self._sync_base_visibility()
        self.invalidate()

    def _sync_groups_and_channels(self) -> None:
        """Groups, baseline list and channels of the recordings that have this configuration."""
        config = self.config_box.currentData()
        active = self._active_runs()
        have = set()
        if not self.points_all.empty and config is not None:
            sel = self.points_all[self.points_all["config"] == config]
            have = set(sel["run"])
            mine = sel[sel["run"].isin({str(r.root) for r in active})]
            per_channel = mine.groupby("channel")["run"].nunique()
            total = mine["run"].nunique()
            names = sorted(per_channel.index, key=lambda s: (len(str(s)), str(s)))
            # a channel only a few sessions recorded (the surface leads on one pig)
            # starts unticked: otherwise every comparison opens on empty panels
            common = {c for c, n in per_channel.items() if n >= max(2, 0.25 * total)}
            self.channels.set_channels(names, checked=common)
        in_config = [r for r in active if str(r.root) in have]
        info = []
        for g in G.sort_states(self._group_of(r) for r in in_config):
            rs = [r for r in in_config if self._group_of(r) == g]
            info.append((g, len(rs), len({r.subject for r in rs})))
        self.picker.set_groups(info)
        current = self.base_box.currentText()
        self.base_box.blockSignals(True)
        self.base_box.clear()
        self.base_box.addItems([g for g, _, _ in info])
        if current in [g for g, _, _ in info]:
            self.base_box.setCurrentText(current)
        self.base_box.blockSignals(False)

    def _on_config(self) -> None:
        self._sync_groups_and_channels()
        self.invalidate()

    def _fill_summaries(self) -> None:
        latency = (self.metric_box.currentData() or "").endswith("_ms")
        options = SUMMARIES_LATENCY if latency else SUMMARIES_SIZE
        current = self.summary_box.currentData()
        self.summary_box.blockSignals(True)
        self.summary_box.clear()
        for key, label in options.items():
            self.summary_box.addItem(label, key)
        if current in options:
            self.summary_box.setCurrentIndex(list(options).index(current))
        self.summary_box.blockSignals(False)

    def _on_analysis_control(self) -> None:
        self._sync_base_visibility()
        self.invalidate()

    def _sync_base_visibility(self) -> None:
        needs = self.norm_box.currentData() in (G.NORM_BASELINE_MAX, G.NORM_BASELINE_TOPN)
        self.base_label.setVisible(needs)
        self.base_box.setVisible(needs)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def has_data(self) -> bool:
        return (not self.points_all.empty and bool(self._active_runs())
                and self.config_box.currentData() is not None)

    def _latency(self) -> bool:
        return (self.metric_box.currentData() or "").endswith("_ms")

    def prepare(self) -> None:
        active = self._active_runs()
        config = self.config_box.currentData()
        state_of = {str(r.root): self._group_of(r) for r in active}
        subject_of = {str(r.root): r.subject for r in active}
        pts = self.points_all[self.points_all["run"].isin(state_of)
                              & (self.points_all["config"] == config)].copy()
        # a curve of a few currents has neither a maximum nor a threshold to compare
        n_amps = pts.groupby("run")["x_value"].nunique()
        self._short = sorted(n_amps.index[n_amps < self.min_amps.value()])
        pts = pts[~pts["run"].isin(self._short)]
        pts["state"] = pts["run"].map(state_of)
        pts["subject"] = pts["run"].map(subject_of)
        metric, mode = self.metric_box.currentData(), self.norm_box.currentData()
        base = self.base_box.currentText()
        factors = (G.normalisation_factors(pts, base, mode, metric=metric)
                   if mode != G.NORM_NONE and not self._latency() else pd.DataFrame())
        self.values, self._missing = G.apply_normalisation(pts, factors, metric)
        self.summaries = G.curve_summaries(self.values, "value")
        self.grid = G.current_grid(self.values)
        n_runs = self.values["run"].nunique() if not self.values.empty else 0
        bits = [f"конфигурация {config}: {records(n_runs)}",
                subjects(self.values['subject'].nunique() if n_runs else 0)]
        if mode != G.NORM_NONE and not self._latency():
            bits.append(f"нормировка: {G.NORM_LABELS[mode]}"
                        + (f" (базовая группа «{base}»)" if self.base_box.isVisible() else ""))
        if self._missing:
            bits.append(f"без базовой записи, исключены: {len(self._missing)} субъект/канал")
        if self._short:
            bits.append(f"короче {self.min_amps.value()} токов, не участвуют: "
                        f"{records(len(self._short))}")
        self._summary_status = "  ·  ".join(bits)
        self.status.setText(self._summary_status)

    def _unit(self) -> str:
        metric, mode = self.metric_box.currentData(), self.norm_box.currentData()
        if mode == G.NORM_NONE or self._latency():
            return METRIC_LABELS[metric]
        return f"{METRIC_LABELS[metric].split(',')[0]}, {NORM_UNITS[mode]}"

    def _summary_unit(self) -> str:
        key = self.summary_box.currentData()
        if key in ("threshold_x", "at_max_x"):
            return "ток, мА"
        return self._unit()

    def _p_values(self, frame: pd.DataFrame, value: str) -> dict:
        pair = self.picker.pair()
        if pair is None or frame.empty:
            return {}
        t = ST.comparison_table(frame, "channel", value, "state", *pair)
        return {r["channel"]: r["p"] for _, r in t.iterrows()} if not t.empty else {}

    @staticmethod
    def _curve_segments(frame: pd.DataFrame, latency: bool) -> list:
        """One line per recording of *frame* on the current axis (no response = 0 for sizes)."""
        segs = []
        for _, grp in frame.groupby("run"):
            c = grp.sort_values("x_value")
            y = c["value"] if latency else c["value"].fillna(0.0)
            seg = np.column_stack([c["x_value"].to_numpy(float), y.to_numpy(float)])
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) > 1:
                segs.append(seg)
        return segs

    @staticmethod
    def _present(channels: list[str], present) -> tuple[list[str], list[str]]:
        """(channels that have data in the compared groups, the ones that do not)."""
        present = set(present)
        return [c for c in channels if c in present], [c for c in channels if c not in present]

    @staticmethod
    def _missing_note(missing: list[str]) -> str:
        return f" · нет данных в этих группах: {', '.join(missing)}" if missing else ""

    @staticmethod
    def _title(name: str, p) -> str:
        if p is None or not np.isfinite(p):
            return name
        text = ST.format_p(p)
        return f"{name}     p {text}" if text.startswith("<") else f"{name}     p = {text}"

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def draw_view(self, key: str, view) -> None:
        groups = self.picker.groups()
        channels = self.channels.checked()
        if not groups:
            view.message("Нет групп с этой конфигурацией: отметьте записи или выберите другую "
                         "конфигурацию.")
            return
        if not channels:
            view.message("Отметьте хотя бы один канал.")
            return
        if self.values.empty and self._short:
            view.message(f"Все записи с конфигурацией {self.config_box.currentData()} короче "
                         f"{self.min_amps.value()} токов: уменьшите порог «Кривая не короче».")
            return
        getattr(self, f"_draw_{key}")(view, groups, self.picker.colors(), channels)

    def _draw_curves(self, view, groups, colors, channels) -> None:
        v = self.values[self.values["state"].isin(groups) & self.values["channel"].isin(channels)]
        channels, missing = self._present(channels, v["channel"])
        if not channels:
            view.message("На выбранных каналах в этих группах нет записей с этой конфигурацией.")
            return
        latency = self._latency()
        on_grid, bands = G.current_bands(v, self.grid, "value", "state", absent_as_zero=not latency)
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_curves.isChecked())
        for ax, ch in zip(axes, channels):
            d = v[v["channel"] == ch].dropna(subset=["x_value"])
            if d.empty:
                empty_panel(ax, ch, "в этих группах канал не записан")
                continue
            b_ch = bands[bands["channel"] == ch] if not bands.empty else bands
            sparse = []             # groups too small for a median here: their recordings are drawn
            for g in groups:
                b = b_ch[b_ch["state"] == g].sort_values("x") if not b_ch.empty else b_ch
                if b.empty and (d["state"] == g).any():
                    sparse.append(g)
                if self.show_curves.isChecked() or b.empty:
                    segs = self._curve_segments(d[d["state"] == g], latency)
                    if segs:
                        faint = not b.empty
                        ax.add_collection(LineCollection(
                            segs, colors=colors[g], linewidths=0.7 if faint else 1.3,
                            alpha=0.25 if faint else 0.85, zorder=1 if faint else 2))
                if not b.empty:
                    median_band(ax, b["x"], b["median"], b["q1"], b["q3"], colors[g])
            if self.grid.size:
                ax.set_xlim(float(self.grid[0]), float(self.grid[-1]))
            los, his = [], []
            if not b_ch.empty:
                los.append(float(b_ch["q1"].min()))
                his.append(float(b_ch["q3"].max()))
            if sparse:
                dv = d.loc[d["state"].isin(sparse), "value"]
                dv = dv if latency else dv.fillna(0.0)
                los.append(float(dv.min()))
                his.append(float(dv.max()))
            lo, hi = (min(los), max(his)) if los else (np.nan, np.nan)
            if latency:
                pad = 0.25 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.0
                if np.isfinite(lo):
                    ax.set_ylim(lo - pad, hi + pad)
            elif np.isfinite(hi) and hi > 0:
                ax.set_ylim(0, hi * (1.4 if not b_ch.empty else 1.15))
            style_axes(ax, ch, xlabel="ток стимуляции, мА", ylabel=self._unit())
        entries = [(g, colors[g], "line") for g in groups]
        if self.show_curves.isChecked():
            entries.append(("отдельные записи", "0.4", "faint"))
        top_legend(view.figure, entries, title=(
            "линия — медиана группы, полоса — квартили (где измерены хотя бы 3 записи и 30% "
            "группы; у групп меньше — сами записи) · конфигурация "
            f"{self.config_box.currentData()}" + ("" if latency else " · ток без ответа = 0")
            + self._missing_note(missing)))

    def _draw_summary(self, view, groups, colors, channels) -> None:
        key = self.summary_box.currentData()
        s = self.summaries[self.summaries["state"].isin(groups)
                           & self.summaries["channel"].isin(channels)] if not self.summaries.empty \
            else self.summaries
        channels, missing = self._present(
            channels, s.dropna(subset=[key])["channel"] if not s.empty else [])
        if not channels:
            view.message("На выбранных каналах в этих группах нет значений.")
            return
        pvals = self._p_values(s, key)
        axes = view.panels(len(channels), legend_entries=len(groups),
                           bottom_extra=label_space(groups))
        for i, (ax, ch) in enumerate(zip(axes, channels)):
            d = s[s["channel"] == ch].dropna(subset=[key]) if not s.empty else s
            present = [g for g in groups if not d.empty and (d["state"] == g).any()]
            if not present:
                empty_panel(ax, ch, "в этих группах нет значений")
                continue
            labels = []
            for k, g in enumerate(present):
                dg = d[d["state"] == g]
                dots_and_box(ax, k, dg[key].to_numpy(float), colors[g], seed=k + 7 * i)
                labels.append((g, len(dg)))
            group_ticks(ax, labels)
            style_axes(ax, self._title(ch, pvals.get(ch)), ylabel=self._summary_unit())
            if key != "at_max_value":
                ax.set_ylim(bottom=0)
        label = self.summary_box.currentText()
        note = {"max": "точка — запись: максимум кривой; кривая без ответов — 0",
                "threshold_x": "точка — запись: наименьший ток, где ответ ≥ 10% максимума кривой",
                "at_max_x": "точка — запись: ток, при котором ответ наибольший",
                "at_max_value": "точка — запись: латентность ответа при токе максимального ответа"}
        top_legend(view.figure, [(g, colors[g], "dot") for g in groups],
                   title=f"{label} · {note.get(key, '')} · конфигурация "
                         f"{self.config_box.currentData()}" + self._missing_note(missing))

    def _draw_waves(self, view, groups, colors, channels) -> None:
        config = self.config_box.currentData()
        s = self.summaries[self.summaries["state"].isin(groups)
                           & self.summaries["channel"].isin(channels)
                           & self.summaries["responded"]] if not self.summaries.empty \
            else self.summaries
        if s.empty:
            view.message("В этих группах на выбранных каналах нет ответов.")
            return
        jobs = []
        for run, g in s.groupby("run"):
            labels = dict(zip(g["channel"], g["at_max_label"]))
            cached = self._waves.get((run, config))
            if cached is None or any(cached[0].get(ch) != lab for ch, lab in labels.items()):
                merged = dict(cached[0]) if cached is not None else {}
                merged.update(labels)
                jobs.append((run, config, merged))
        if jobs:
            self._load_waves(jobs)
            view.message(f"Читаю формы ответов: {records(len(jobs))}…\nЭто делается один раз.")
            return
        state_of = dict(zip(s["run"], s["state"]))
        channels, missing = self._present(channels, s["channel"])
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_traces.isChecked())
        for ax, ch in zip(axes, channels):
            loaded = []
            for run in s.loc[s["channel"] == ch, "run"].unique():
                _, (times, waves) = self._waves.get((run, config), ({}, (None, {})))
                if times is not None and ch in waves:
                    loaded.append((state_of[run], run, times, waves[ch]))
            grid, bands, traces = G.waveform_bands(loaded)
            if grid.size == 0:
                empty_panel(ax, ch, "нет ответов")
                continue
            t = grid * 1e3
            # the whole epoch is drawn (pan or zoom out to see it); the view opens on the
            # early response, and its scale ignores the stimulus artifact of the first 3 ms
            shown = (t >= -5.0) & (t <= 60.0)
            post = t >= 3.0
            if self.show_traces.isChecked():
                for g in groups:
                    segs = [np.column_stack([t, w]) for gg, _, w in traces if gg == g]
                    if segs:
                        ax.add_collection(LineCollection(segs, colors=colors[g], linewidths=0.6,
                                                         alpha=0.2, zorder=1))
            lows, highs, counts = [], [], []
            for g in groups:
                b = bands[bands["group"] == g] if not bands.empty else bands
                if b.empty:
                    continue
                m, lo, hi = (b[c].to_numpy(float) for c in ("mean", "lo", "hi"))
                ax.plot(t, m, color=colors[g], lw=2.0, zorder=3)
                if np.isfinite(lo).any():
                    ax.fill_between(t, lo, hi, color=colors[g], alpha=0.2, lw=0, zorder=2)
                sel = post & shown
                lows.append(float(np.nanmin(np.where(np.isfinite(lo), lo, m)[sel])))
                highs.append(float(np.nanmax(np.where(np.isfinite(hi), hi, m)[sel])))
                counts.append(f"{g}: {int(b['n'].iloc[0])}")
            ax.axvspan(float(t[0]), 3.0, color="0.92", lw=0, zorder=0)
            ax.axhline(0, color="0.75", lw=0.6, zorder=0)
            if lows:
                span = (max(highs) - min(lows)) or 1.0
                ax.set_ylim(min(lows) - 0.12 * span, max(highs) + 0.12 * span)
            ax.set_xlim(max(float(t[0]), -5.0), min(float(t[-1]), 60.0))
            style_axes(ax, ch, xlabel="мс от стимула", ylabel="мкВ")
            if len(groups) <= 3:
                ax.text(0.99, 0.02, "записей — " + ", ".join(counts), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color="0.4")
        entries = [(g, colors[g], "line") for g in groups]
        if self.show_traces.isChecked():
            entries.append(("отдельные записи", "0.4", "faint"))
        top_legend(view.figure, entries, title=(
            "линия — средний ответ группы, полоса — 95% ДИ; у каждой записи — ответ на токе, где "
            f"он максимален · конфигурация {config} · мкВ · серым — до 3 мс (артефакт стимула)"
            + ("" if len(groups) <= 3 else " · число записей — в таблице")
            + self._missing_note(missing)))

    def _load_waves(self, jobs) -> None:
        if self._wave_reader is not None and self._wave_reader.isRunning():
            return
        self._wave_reader = _WaveReader(jobs)
        self._wave_reader.progress.connect(
            lambda i, n: self.status.setText(f"Читаю формы ответов: {i} из {n}…"))
        self._wave_reader.done.connect(self._on_waves)
        self._wave_reader.start()

    def _on_waves(self, out: dict) -> None:
        self._waves.update(out)
        self.status.setText(self._summary_status
                            + f"  ·  формы ответов прочитаны: {records(len(out))}")
        self._dirty.add("waves")
        self._timer.start()

    # ------------------------------------------------------------------ #
    # Table and export
    # ------------------------------------------------------------------ #
    def stats_for(self, key: str):
        value = self.summary_box.currentData()
        channels = self.channels.checked()
        groups = self.picker.groups()
        if self.summaries.empty or value is None:
            return None, None, "нет данных для таблицы"
        frame = self.summaries[self.summaries["state"].isin(groups)
                               & self.summaries["channel"].isin(channels)]
        if frame.empty:
            return None, None, "нет данных для таблицы"
        pair = self.picker.pair()
        if pair is not None:
            table = ST.comparison_table(frame, "channel", value, "state", *pair, units=channels)
            shown, bold = comparison_display(table, "channel", "канал", *pair)
            return shown, bold, ""
        table = ST.describe_groups(frame, "channel", value, "state", groups, units=channels)
        return groups_display(table, "channel", "канал"), None, ""

    def export_tables(self, out: Path) -> list[str]:
        written = []

        def save(df: pd.DataFrame, name: str) -> None:
            if df is not None and not df.empty:
                df.to_csv(out / name, index=False)
                written.append(name)

        config = _safe(self.config_box.currentData())
        save(self.values, f"sir_group_{config}_points_long.csv")
        save(self.summaries, f"sir_group_{config}_curve_summaries.csv")
        value = self.summary_box.currentData()
        groups = self.picker.groups()
        frame = self.summaries[self.summaries["state"].isin(groups)] if not self.summaries.empty \
            else self.summaries
        pair = self.picker.pair()
        if not frame.empty and value is not None:
            if pair is not None:
                save(ST.comparison_table(frame, "channel", value, "state", *pair),
                     f"sir_group_{config}_{value}_{_safe(pair[0])}_vs_{_safe(pair[1])}.csv")
            else:
                save(ST.describe_groups(frame, "channel", value, "state", groups),
                     f"sir_group_{config}_{value}_by_group.csv")
        if not self.values.empty and self.grid.size:
            _, bands = G.current_bands(self.values, self.grid, "value", "state",
                                       absent_as_zero=not self._latency())
            save(bands, f"sir_group_{config}_curve_bands.csv")
        save(pd.DataFrame([{"subject": r.subject, "group": r.state, "subgroup": r.subgroup,
                            "group_compared": self._group_of(r), "included": r.include,
                            "run": str(r.root)} for r in self.runs]), "sir_group_membership.csv")
        return written

"""Group analysis of Neurosoft recordings: does one group of recordings differ from another?

A group is a set of processed Neurosoft exports sharing a label read off the recording
names — cohort, stimulation level, side, … ("Группировать по"); the label can be edited
by hand in the list. The tab compares two groups, A in blue and B in red, or shows all
of them, channel by channel, in four views:

* Плато — A. Militskova's summary of a recording: the mean response of its last five
  curves, the end of the ramp. One dot per recording, a thin line for the range of its
  five curves, the group's median (bold line) and quartiles (box). The table under the
  plots compares the groups on this value.
* Кривые рекрутирования — every recording's ramp, counted from its last curve (the
  exports differ in length and starting current and carry no mA), with each group's
  median and quartile band where at least three recordings reach.
* Формы ответов — the plateau waveform (mean of the last five curves) of every
  recording, and each group's mean with a 95 % confidence band.
* Асимметрия — the left/right index of the channel pairs on the same five curves
  (src/asymmetry.py; the sides are provisional).

Recordings are read once, in the background, when a folder is added. Grouping, the
compared groups, metric, normalisation and channels all work on what is in memory;
waveforms are read the first time their view is opened and kept. The analysis lives in
src/group.py, src/group_stats.py and src/asymmetry.py; this file only draws.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

import src.asymmetry as A
import src.group as G
import src.group_stats as ST
from src.constants import TEXT_CURVES_RESP_TMIN

from .group_common import (plural, records, subjects, GroupTab, begin_fill, comparison_display, dots_and_box, empty_panel,
                           end_fill, group_ticks, groups_display, label_space, legend_note, median_band,
                           member_table, style_axes, top_legend)

#: The metrics offered, in this order, with the units they are read in.
METRIC_LABELS = {
    "amp_uv": "размах ответа, мкВ",
    "area_uvms": "площадь ответа, мкВ·мс",
    "onset_ms": "латентность начала, мс",
    "p1_ms": "латентность P1, мс",
    "p1_uv": "амплитуда P1, мкВ",
}
NORM_UNITS = {
    G.NORM_BASELINE_MAX: "доля максимума базовой группы",
    G.NORM_BASELINE_TOPN: "доля top-3 базовой группы",
    G.NORM_OWN_MAX: "доля максимума своей записи",
}
SCENARIO_LABELS = {
    "Recruitment": "кривые рекрутирования",
    "Jendrassik": "приём Ендрассика",
    "Paired stimulation": "парная стимуляция",
    "H-reflex": "H-рефлекс",
}
ALL_SCENARIOS = "__all__"
VIEWS = [
    ("plateau", f"Плато: последние {G.LAST_N} кривых"),
    ("curves", "Кривые рекрутирования"),
    ("waves", "Формы ответов"),
    ("asym", "Асимметрия"),
]


class _PointReader(QThread):
    """Reads the recordings' metrics (and which channels they recorded) off disk."""
    done = Signal(object, object, object)

    def __init__(self, runs) -> None:
        super().__init__()
        self._runs = runs

    def run(self) -> None:
        try:
            points = G.collect_points(self._runs)
            recorded = {}
            for run in self._runs:
                try:
                    recorded[str(run.root)] = A.recorded_channels(run.root)
                except Exception:                      # noqa: BLE001 - falls back to the table
                    recorded[str(run.root)] = None
            self.done.emit(points, recorded, None)
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.done.emit(None, None, f"{type(exc).__name__}: {exc}")


class _WaveReader(QThread):
    """Reads the plateau waveform of every recording, once."""
    progress = Signal(int, int)
    done = Signal(object)

    def __init__(self, roots: list[str]) -> None:
        super().__init__()
        self._roots = roots

    def run(self) -> None:
        out = {}
        for i, root in enumerate(self._roots, 1):
            try:
                out[root] = G.plateau_waveforms(root)
            except Exception:                          # noqa: BLE001 - that recording has none
                out[root] = (None, {})
            self.progress.emit(i, len(self._roots))
        self.done.emit(out)


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_") or "group"


class GroupViewer(GroupTab):
    """The Neurosoft group tab."""
    VIEWS = VIEWS
    EXPORT_PREFIX = "neurosoft_group"

    def __init__(self, session=None) -> None:
        super().__init__(session)
        self.runs: list[G.GroupRun] = []
        self.points_all = pd.DataFrame()
        self._recorded: dict = {}
        self._waves: dict = {}
        self._pending: list[G.GroupRun] = []
        self._reader: _PointReader | None = None
        self._wave_reader: _WaveReader | None = None
        self.raw_points = pd.DataFrame()
        self.values = pd.DataFrame()
        self.plateau = pd.DataFrame()
        self.plateau_curves = pd.DataFrame()
        self.hm = pd.DataFrame()
        self._asym = None
        self._missing: list[str] = []

        # ---- left: the recordings ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip("Найти в папке все обработанные записи Neurosoft и добавить их.")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.scenario_box = QComboBox()
        self.scenario_box.setToolTip("Какой протокол сравнивать: виды вкладки сделаны для "
                                     "кривых рекрутирования, остальные сценарии — отдельно.")
        self.scenario_box.currentIndexChanged.connect(lambda _: self._membership_changed(refill=True))
        self.group_box = QComboBox()
        for key, label in G.GROUP_MODES.items():
            self.group_box.addItem(label, key)
        self.group_box.setCurrentIndex(list(G.GROUP_MODES).index("cohort"))
        self.group_box.setToolTip("Из каких меток названия складывается группа записи. "
                                  "Группу можно поправить руками в списке.")
        self.group_box.currentIndexChanged.connect(lambda _: self._regroup())
        self.table = member_table(["", "Субъект", "Группа", "Запись"], stretch_col=3)
        self.table.setToolTip("Галочка — запись участвует. Группу можно переименовать двойным щелчком.")
        self.table.itemChanged.connect(self._on_table_edit)
        self.btn_all = QPushButton("отметить все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("снять все")
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        self.members_note = QLabel("")
        self.members_note.setStyleSheet("color: #555; font-size: 11px;")

        # ---- top: what is compared ----
        self.metric_box = QComboBox()
        for key, label in METRIC_LABELS.items():
            self.metric_box.addItem(label, key)
        self.norm_box = QComboBox()
        for key, label in G.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        self.norm_box.setToolTip("Нормировка считается отдельно для каждого субъекта и канала. "
                                 "Латентности не нормируются.")
        self.base_label = QLabel("базовая группа")
        self.base_box = QComboBox()
        for box in (self.metric_box, self.norm_box, self.base_box):
            box.currentIndexChanged.connect(lambda _: self._on_analysis_control())

        # ---- options that belong to one view ----
        self.show_five = QCheckBox(f"разброс {G.LAST_N} кривых внутри записи")
        self.show_five.setChecked(True)
        self.show_five.toggled.connect(lambda _: self.redraw())
        self.align_box = QComboBox()
        self.align_box.addItem("кривые от конца развертки", G.ALIGN_END)
        self.align_box.addItem("номер кривой", G.ALIGN_START)
        self.align_box.setToolTip("От конца: последние кривые (плато) совпадают у всех записей, "
                                  "хотя развертки разной длины.")
        self.align_box.currentIndexChanged.connect(lambda _: self.redraw())
        self.show_ramps = QCheckBox("отдельные записи")
        self.show_ramps.toggled.connect(lambda _: self.redraw())
        self.show_traces = QCheckBox("отдельные записи")
        self.show_traces.toggled.connect(lambda _: self.redraw())
        self.asym_box = QComboBox()
        self.asym_box.addItem("индекс (R − L) / (R + L)", "ai")
        self.asym_box.addItem("отношение R / L", "ratio")
        self.asym_box.currentIndexChanged.connect(lambda _: self.redraw())
        self.asym_note = QLabel(A.SIDES_NOTE_RU)
        self.asym_note.setStyleSheet("color: #777;")
        self.view_options = {
            "plateau": [self.show_five],
            "curves": [self.align_box, self.show_ramps],
            "waves": [self.show_traces],
            "asym": [self.asym_box, self.asym_note],
        }

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)
        for title, w in (("Сценарий", self.scenario_box), ("Группировать по", self.group_box)):
            lv.addWidget(QLabel(f"<b>{title}</b>"))
            lv.addWidget(w)
        lv.addWidget(QLabel("<b>Записи</b>"))
        lv.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        lv.addLayout(row)
        lv.addWidget(self.members_note)

        self.build_layout(left, [QLabel("<b>Показатель</b>"), self.metric_box,
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
        return self.group_box.currentData() or "cohort"

    def _scan(self) -> None:
        start = str(self.runs[-1].root.parent) if self.runs else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с обработанными записями Neurosoft", start)
        if folder:
            n = self.add_runs(G.scan_neurosoft([Path(folder)], self._mode()))
            if not n:
                self.status.setText("Новых обработанных записей в этой папке нет.")

    def add_runs(self, found: list[G.GroupRun]) -> int:
        known = {str(r.root) for r in self.runs}
        new = [r for r in found if str(r.root) not in known]
        if not new:
            return 0
        self.runs.extend(new)
        self.runs.sort(key=lambda r: (r.tags.get("cohort", ""), r.subject, r.root.name))
        self._pending.extend(new)
        self._read_pending()
        self._fill_scenarios()
        self._membership_changed(refill=True)
        return len(new)

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group."""
        root = Path(root)
        if G.looks_like_run(root):
            tags = G.parse_neurosoft(root)
            self.add_runs([G.GroupRun(root, tags["subject"], G.state_from_tags(tags, self._mode()),
                                      tags=tags)])

    def _read_pending(self) -> None:
        if (self._reader is not None and self._reader.isRunning()) or not self._pending:
            return
        batch, self._pending = self._pending, []
        copies = [G.GroupRun(r.root, r.subject, r.state, True, tags=r.tags) for r in batch]
        self.status.setText(f"Читаю {records(len(batch))}…")
        self._reader = _PointReader(copies)
        self._reader.done.connect(self._on_points)
        self._reader.start()

    def _on_points(self, points, recorded, error) -> None:
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось прочитать записи", error)
        else:
            if points is not None and not points.empty:
                points = points.copy()
                # A Neurosoft export is one configuration named after the recording;
                # one label for all, so it never splits the group.
                points["config"] = "Neurosoft"
                self.points_all = (points if self.points_all.empty
                                   else pd.concat([self.points_all, points], ignore_index=True))
            self._recorded.update(recorded or {})
        self._read_pending()
        self._membership_changed()

    def _clear(self) -> None:
        self.runs, self._pending = [], []
        self.points_all = self.raw_points = self.values = pd.DataFrame()
        self.plateau = self.plateau_curves = self.hm = pd.DataFrame()
        self._recorded, self._waves, self._asym = {}, {}, None
        self._fill_scenarios()
        self._fill_table()
        self.picker.set_groups([])
        self.channels.set_channels([])
        for view in self.views.values():
            view.message("Группа пуста: добавьте папку с записями.")
            view.draw()
        self.stats.show_frame(None)
        self.members_note.setText("")
        self.status.setText("Группа пуста: добавьте папку с записями.")

    def _fill_scenarios(self) -> None:
        counts: dict[str, int] = {}
        for r in self.runs:
            s = r.tags.get("scenario", "—")
            counts[s] = counts.get(s, 0) + 1
        current = self.scenario_box.currentData()
        present = sorted(counts, key=lambda s: (s != "Recruitment", s))
        self.scenario_box.blockSignals(True)
        self.scenario_box.clear()
        for s in present:
            self.scenario_box.addItem(f"{SCENARIO_LABELS.get(s, s)} ({counts[s]})", s)
        self.scenario_box.addItem(f"все сценарии ({len(self.runs)})", ALL_SCENARIOS)
        keys = present + [ALL_SCENARIOS]
        target = current if current in keys else (present[0] if present else ALL_SCENARIOS)
        self.scenario_box.setCurrentIndex(keys.index(target))
        self.scenario_box.blockSignals(False)

    def _visible_runs(self) -> list[G.GroupRun]:
        scenario = self.scenario_box.currentData()
        return [r for r in self.runs
                if scenario in (None, ALL_SCENARIOS) or r.tags.get("scenario", "—") == scenario]

    def _active_runs(self) -> list[G.GroupRun]:
        return [r for r in self._visible_runs() if r.include]

    def _regroup(self) -> None:
        mode = self._mode()
        for r in self.runs:
            if r.tags:
                r.state = G.state_from_tags(r.tags, mode)
        self._membership_changed(refill=True)

    def _fill_table(self) -> None:
        runs = self._visible_runs()
        begin_fill(self.table)
        self.table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if r.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            subject = QTableWidgetItem(r.subject)
            subject.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 1, subject)
            self.table.setItem(i, 2, QTableWidgetItem(r.state))
            name = QTableWidgetItem(r.root.name)
            name.setFlags(Qt.ItemIsEnabled)
            name.setToolTip(str(r.root) + "\n" + ", ".join(
                f"{k}: {v}" for k, v in r.tags.items() if v not in ("—", "") and k != "subject"))
            self.table.setItem(i, 3, name)
        end_fill(self.table)

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        runs = self._visible_runs()
        if not 0 <= item.row() < len(runs):
            return
        run = runs[item.row()]
        if item.column() == 0:
            run.include = item.checkState() == Qt.Checked
        elif item.column() == 2:
            run.state = item.text().strip() or run.state
        else:
            return
        self._membership_changed()

    def _set_all(self, on: bool) -> None:
        for r in self._visible_runs():
            r.include = on
        self._membership_changed(refill=True)

    def _membership_changed(self, refill: bool = False) -> None:
        if refill:
            self._fill_table()
        active = self._active_runs()
        info = []
        for g in G.sort_states(r.state for r in active):
            rs = [r for r in active if r.state == g]
            info.append((g, len(rs), len({r.subject for r in rs})))
        self.picker.set_groups(info)
        current = self.base_box.currentText()
        self.base_box.blockSignals(True)
        self.base_box.clear()
        self.base_box.addItems([g for g, _, _ in info])
        if current in [g for g, _, _ in info]:
            self.base_box.setCurrentText(current)
        self.base_box.blockSignals(False)
        if not self.points_all.empty:
            roots = {str(r.root) for r in active}
            names = self.points_all.loc[self.points_all["run"].isin(roots), "channel"].dropna().unique()
            self.channels.set_channels(sorted(names, key=lambda s: (len(s), s)))
        visible = self._visible_runs()
        self.members_note.setText(f"отмечено {len(active)} из {len(visible)} · "
                                  f"{subjects(len({r.subject for r in active}))} · "
                                  f"{plural(len(info), 'группа', 'группы', 'групп')}")
        self._sync_base_visibility()
        self.invalidate()

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
        return not self.points_all.empty and bool(self._active_runs())

    def prepare(self) -> None:
        active = self._active_runs()
        state_of = {str(r.root): r.state for r in active}
        pts = self.points_all[self.points_all["run"].isin(state_of)].copy()
        pts["state"] = pts["run"].map(state_of)
        self.raw_points = pts
        metric = self.metric_box.currentData()
        mode = self.norm_box.currentData()
        base = self.base_box.currentText()
        factors = (G.normalisation_factors(pts, base, mode, metric=metric)
                   if mode != G.NORM_NONE and not metric.endswith("_ms") else pd.DataFrame())
        self.values, self._missing = G.apply_normalisation(pts, factors, metric)
        self.plateau, self.plateau_curves = G.plateau_table(self.values, G.LAST_N)
        self.hm = self._hmax_mmax(pts) if metric == "amp_uv" else pd.DataFrame()
        self._asym = None

        bits = [records(len(active)), subjects(len({r.subject for r in active})),
                plural(len(set(state_of.values())), "группа", "группы", "групп")]
        if mode != G.NORM_NONE and not metric.endswith("_ms"):
            bits.append(f"нормировка: {G.NORM_LABELS[mode]}"
                        + (f" (базовая группа «{base}»)" if self.base_box.isVisible() else ""))
        if self._missing:
            bits.append(f"без базовой записи, исключены: {len(self._missing)} субъект/канал")
        self.status.setText("  ·  ".join(bits))

    @staticmethod
    def _hmax_mmax(pts: pd.DataFrame) -> pd.DataFrame:
        """Hmax/Mmax per H-reflex recording: the number clinicians read first."""
        comp = pts["channel"].map(lambda c: G.split_component(str(c))[1])
        if comp.notna().sum() == 0:
            return pd.DataFrame()
        d = pts[comp.notna()].assign(comp=comp[comp.notna()],
                                     raw=pts.loc[comp.notna(), "channel"].map(
                                         lambda c: G.split_component(str(c))[0]))
        peak = d.groupby(["subject", "state", "run", "raw", "comp"])["amp_uv"].max().unstack("comp")
        if not {"h", "m"} <= set(peak.columns):
            return pd.DataFrame()
        peak = peak.dropna(subset=["h", "m"])
        peak = peak[peak["m"] > 0].reset_index()
        peak["channel"] = peak["raw"] + " · Hmax/Mmax"
        peak["mean"] = peak["h"] / peak["m"]
        return peak[["subject", "state", "run", "channel", "mean"]]

    def _asymmetry(self):
        if self._asym is None:
            self._asym = G.asymmetry_last_curves(self.raw_points, self.metric_box.currentData(),
                                                 G.LAST_N, recorded=self._recorded)
        return self._asym

    def _unit(self) -> str:
        metric, mode = self.metric_box.currentData(), self.norm_box.currentData()
        if mode == G.NORM_NONE or metric.endswith("_ms"):
            return METRIC_LABELS[metric]
        return f"{METRIC_LABELS[metric].split(',')[0]}, {NORM_UNITS[mode]}"

    def _p_values(self, frame: pd.DataFrame, unit_col: str, value: str) -> dict:
        pair = self.picker.pair()
        if pair is None or frame.empty:
            return {}
        t = ST.comparison_table(frame, unit_col, value, "state", *pair)
        return {r[unit_col]: r["p"] for _, r in t.iterrows()} if not t.empty else {}

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
            view.message("Нет групп: отметьте записи в списке слева.")
            return
        if not channels:
            view.message("Отметьте хотя бы один канал.")
            return
        getattr(self, f"_draw_{key}")(view, groups, self.picker.colors(), channels)

    def _draw_plateau(self, view, groups, colors, channels) -> None:
        p = self.plateau[self.plateau["state"].isin(groups) & self.plateau["channel"].isin(channels)]
        pvals = self._p_values(p, "channel", "mean")
        unit = self._unit()
        latency = self.metric_box.currentData().endswith("_ms")
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_five.isChecked(),
                           bottom_extra=label_space(groups))
        for i, (ax, ch) in enumerate(zip(axes, channels)):
            d = p[p["channel"] == ch].dropna(subset=["mean"])
            present = [g for g in groups if (d["state"] == g).any()]
            if not present:
                empty_panel(ax, ch, "в этих группах нет ответов")
                continue
            labels = []
            for k, g in enumerate(present):
                dg = d[d["state"] == g]
                ranges = list(zip(dg["min"], dg["max"])) if self.show_five.isChecked() else None
                dots_and_box(ax, k, dg["mean"].to_numpy(float), colors[g], seed=k + 7 * i,
                             ranges=ranges)
                labels.append((g, len(dg)))
            group_ticks(ax, labels)
            style_axes(ax, self._title(ch, pvals.get(ch)), ylabel=unit)
            if not latency:
                ax.set_ylim(bottom=0)
        entries = [(g, colors[g], "dot") for g in groups]
        if self.show_five.isChecked():
            entries.append((f"разброс {G.LAST_N} кривых записи", "0.45", "faint"))
        top_legend(view.figure, entries)
        legend_note(view.figure, f"точка — запись (среднее последних {G.LAST_N} кривых) · черта — "
                             "медиана группы · рамка — квартили")

    def _draw_curves(self, view, groups, colors, channels) -> None:
        align = self.align_box.currentData()
        v = self.values[self.values["state"].isin(groups) & self.values["channel"].isin(channels)]
        ramps = G.aligned_ramps(v, align)
        bands = G.ramp_bands(ramps, group_col="state")
        unit = self._unit()
        latency = self.metric_box.currentData().endswith("_ms")
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_ramps.isChecked())
        for ax, ch in zip(axes, channels):
            d = ramps[ramps["channel"] == ch].dropna(subset=["x", "value"])
            if d.empty:
                empty_panel(ax, ch, "в этих группах нет ответов")
                continue
            b_ch = bands[bands["channel"] == ch]
            if self.show_ramps.isChecked():
                for g in groups:
                    segs = [grp.sort_values("x")[["x", "value"]].to_numpy(float)
                            for _, grp in d[d["state"] == g].groupby("run")]
                    segs = [s for s in segs if len(s) > 1]
                    if segs:
                        ax.add_collection(LineCollection(segs, colors=colors[g], linewidths=0.6,
                                                         alpha=0.2, zorder=1))
            for g in groups:
                b = b_ch[b_ch["state"] == g].sort_values("x")
                if not b.empty:
                    median_band(ax, b["x"], b["median"], b["q1"], b["q3"], colors[g])
            if align == G.ALIGN_END:
                ax.axvspan(-(G.LAST_N - 0.5), 0.5, color="0.88", alpha=0.6, lw=0, zorder=0)
            xs = b_ch["x"] if not b_ch.empty else d["x"]
            ax.set_xlim(float(xs.min()) - 1, float(xs.max()) + 1)
            if b_ch.empty:
                lo, hi = float(d["value"].quantile(0.02)), float(d["value"].quantile(0.95))
            else:
                lo, hi = float(b_ch["q1"].min()), float(b_ch["q3"].max())
            if latency:
                pad = 0.25 * (hi - lo) if hi > lo else 1.0
                ax.set_ylim(lo - pad, hi + pad)
            elif np.isfinite(hi) and hi > 0:
                ax.set_ylim(0, hi * 1.4)
            style_axes(ax, ch, ylabel=unit,
                       xlabel=("кривых до конца развертки (0 — последняя)" if align == G.ALIGN_END
                               else "номер кривой"))
        entries = [(g, colors[g], "line") for g in groups]
        if self.show_ramps.isChecked():
            entries.append(("отдельные записи", "0.4", "faint"))
        note = "линия — медиана группы, полоса — квартили (где есть хотя бы 30% записей группы)"
        if align == G.ALIGN_END:
            note += f" · серым — последние {G.LAST_N} кривых, по которым считается плато"
        top_legend(view.figure, entries, title=note)

    def _draw_waves(self, view, groups, colors, channels) -> None:
        active = [r for r in self._active_runs() if r.state in groups]
        missing = [str(r.root) for r in active if str(r.root) not in self._waves]
        if missing:
            self._load_waves(missing)
            view.message(f"Читаю формы ответов: {len(active) - len(missing)} из {len(active)} "
                         "записей готово…\nЭто делается один раз.")
            return
        t_art = TEXT_CURVES_RESP_TMIN * 1e3
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_traces.isChecked())
        for ax, ch in zip(axes, channels):
            raw, _ = G.split_component(ch)
            loaded = []
            for r in active:
                times, waves = self._waves.get(str(r.root), (None, {}))
                if times is not None and raw in waves:
                    loaded.append((r.state, str(r.root), times, waves[raw]))
            grid, bands, traces = G.waveform_bands(loaded)
            if grid.size == 0:
                empty_panel(ax, ch, "нет сохранённых эпох")
                continue
            t = grid * 1e3
            post = t >= t_art
            if self.show_traces.isChecked():
                for g in groups:
                    segs = [np.column_stack([t[post], w[post]]) for gg, _, w in traces if gg == g]
                    if segs:
                        ax.add_collection(LineCollection(segs, colors=colors[g], linewidths=0.6,
                                                         alpha=0.18, zorder=1))
            lows, highs, counts = [], [], []
            for g in groups:
                b = bands[bands["group"] == g]
                if b.empty:
                    continue
                m, lo, hi = (b[c].to_numpy(float) for c in ("mean", "lo", "hi"))
                ax.plot(t[post], m[post], color=colors[g], lw=2.0, zorder=3)
                if np.isfinite(lo).any():
                    ax.fill_between(t[post], lo[post], hi[post], color=colors[g], alpha=0.2, lw=0,
                                    zorder=2)
                lows.append(float(np.nanmin(np.where(np.isfinite(lo), lo, m)[post])))
                highs.append(float(np.nanmax(np.where(np.isfinite(hi), hi, m)[post])))
                counts.append(f"{g}: {int(b['n'].iloc[0])}")
            ax.axvspan(t[0], t_art, color="0.92", lw=0, zorder=0)
            ax.axhline(0, color="0.75", lw=0.6, zorder=0)
            if lows:
                span = max(highs) - min(lows) or 1.0
                ax.set_ylim(min(lows) - 0.12 * span, max(highs) + 0.12 * span)
            ax.set_xlim(t[0], t[-1])
            style_axes(ax, ch, xlabel="мс от стимула", ylabel="мкВ")
            if len(groups) <= 3:
                ax.text(0.99, 0.02, "записей — " + ", ".join(counts), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color="0.4")
        entries = [(g, colors[g], "line") for g in groups]
        if self.show_traces.isChecked():
            entries.append(("отдельные записи", "0.4", "faint"))
        top_legend(view.figure, entries, title=(
            f"линия — средний ответ группы, полоса — 95% ДИ; у каждой записи — среднее последних "
            f"{G.LAST_N} кривых · мкВ без нормировки · серым — артефакт стимула"
            + ("" if len(groups) <= 3 else " · число записей — в таблице")))

    def _load_waves(self, roots: list[str]) -> None:
        if self._wave_reader is not None and self._wave_reader.isRunning():
            return
        self._wave_reader = _WaveReader(roots)
        self._wave_reader.progress.connect(
            lambda i, n: self.status.setText(f"Читаю формы ответов: {i} из {n}…"))
        self._wave_reader.done.connect(self._on_waves)
        self._wave_reader.start()

    def _on_waves(self, out: dict) -> None:
        self._waves.update(out)
        self.status.setText(f"Формы ответов прочитаны: {records(len(out))}.")
        self._dirty.add("waves")
        self._timer.start()

    def _draw_asym(self, view, groups, colors, channels) -> None:
        _, rec, kind = self._asymmetry()
        if rec.empty:
            view.message("Асимметрия считается по парам каналов ch1–ch5 … ch4–ch8.\n"
                         "В этих записях таких пар нет (у H-рефлекса каналы одной конечности).")
            return
        mode = "diff" if kind == "latency" else self.asym_box.currentData()
        col = {"ai": "ai", "ratio": "ratio", "diff": "diff_ms"}[mode]
        names = set(rec["pair"])
        pairs = [(l, r) for l, r in A.PAIRS
                 if A.pair_name(l, r) in names and (l in channels or r in channels)]
        if not pairs:
            view.message("Отметьте каналы, входящие в пары ch1–ch5 … ch4–ch8.")
            return
        d_all = rec[rec["state"].isin(groups)].dropna(subset=[col])
        if mode == "ratio":
            d_all = d_all[d_all[col] > 0]
        pvals = self._p_values(d_all, "pair", col)
        ylabel = {"ai": "(R − L) / (R + L)", "ratio": "R / L", "diff": "R − L, мс"}[mode]
        axes = view.panels(len(pairs), legend_entries=len(groups), bottom_extra=label_space(groups))
        for i, (ax, (left, right)) in enumerate(zip(axes, pairs)):
            name = A.pair_name(left, right)
            d = d_all[d_all["pair"] == name]
            present = [g for g in groups if (d["state"] == g).any()]
            title = f"{left} (лево) ↔ {right} (право)"
            if not present:
                empty_panel(ax, title)
                continue
            labels = []
            for k, g in enumerate(present):
                dg = d[d["state"] == g]
                dots_and_box(ax, k, dg[col].to_numpy(float), colors[g], seed=k + 11 * i)
                labels.append((g, len(dg)))
            ax.axhline(1.0 if mode == "ratio" else 0.0, color="0.5", lw=0.9, ls="--", zorder=0)
            if mode == "ai":
                ax.set_ylim(-1.1, 1.1)
            elif mode == "ratio":
                ax.set_yscale("log")
            group_ticks(ax, labels)
            style_axes(ax, self._title(title, pvals.get(name)), ylabel=ylabel)
        top_legend(view.figure, [(g, colors[g], "dot") for g in groups])
        hint = {"ai": "0 — симметрично, +1 — ответ только справа, −1 — только слева",
                "ratio": "1 — симметрично", "diff": "0 — одинаковая латентность"}[mode]
        legend_note(view.figure, f"точка — запись (последние {G.LAST_N} кривых) · {hint} · "
                             f"{A.SIDES_NOTE_RU}")

    # ------------------------------------------------------------------ #
    # Table and export
    # ------------------------------------------------------------------ #
    def _stat_frame(self, key: str):
        groups = self.picker.groups()
        channels = self.channels.checked()
        if key == "asym":
            _, rec, kind = self._asymmetry()
            if rec.empty:
                return pd.DataFrame(), "pair", "пара каналов", "ai", []
            col = "diff_ms" if kind == "latency" else ("ratio" if self.asym_box.currentData() == "ratio"
                                                       else "ai")
            frame = rec[rec["state"].isin(groups)]
            units = [A.pair_name(l, r) for l, r in A.PAIRS
                     if (l in channels or r in channels) and A.pair_name(l, r) in set(frame["pair"])]
            return frame[frame["pair"].isin(units)], "pair", "пара каналов", col, units
        frame = self.plateau[self.plateau["state"].isin(groups) & self.plateau["channel"].isin(channels)]
        units = list(channels)
        if not self.hm.empty:
            raws = {G.split_component(c)[0] for c in channels}
            hm = self.hm[self.hm["state"].isin(groups)
                         & self.hm["channel"].map(lambda c: c.split(" · ")[0] in raws)]
            frame = pd.concat([frame, hm], ignore_index=True)
            units += sorted(hm["channel"].unique())
        return frame, "channel", "канал", "mean", units

    def stats_for(self, key: str):
        frame, unit_col, unit_label, value, units = self._stat_frame(key)
        if frame.empty:
            return None, None, "нет данных для таблицы"
        pair = self.picker.pair()
        if pair is not None:
            table = ST.comparison_table(frame, unit_col, value, "state", *pair, units=units)
            shown, bold = comparison_display(table, unit_col, unit_label, *pair)
            return shown, bold, ""
        table = ST.describe_groups(frame, unit_col, value, "state", self.picker.groups(), units=units)
        return groups_display(table, unit_col, unit_label), None, ""

    def export_tables(self, out: Path) -> list[str]:
        written = []

        def save(df: pd.DataFrame, name: str) -> None:
            if df is not None and not df.empty:
                df.to_csv(out / name, index=False)
                written.append(name)

        save(self.values, "neurosoft_group_points_long.csv")
        save(self.plateau, "neurosoft_group_plateau_by_recording.csv")
        save(self.plateau_curves, "neurosoft_group_plateau_curves.csv")
        pair = self.picker.pair()
        for key, stem in (("plateau", "plateau"), ("asym", "asymmetry")):
            frame, unit_col, _, value, units = self._stat_frame(key)
            if frame.empty:
                continue
            if pair is not None:
                save(ST.comparison_table(frame, unit_col, value, "state", *pair, units=units),
                     f"neurosoft_group_{stem}_{_safe(pair[0])}_vs_{_safe(pair[1])}.csv")
            else:
                save(ST.describe_groups(frame, unit_col, value, "state", self.picker.groups(),
                                        units=units), f"neurosoft_group_{stem}_by_group.csv")
        save(self._asymmetry()[1], "neurosoft_group_asymmetry_by_recording.csv")
        save(pd.DataFrame([{"subject": r.subject, "group": r.state, "included": r.include,
                            **{k: v for k, v in r.tags.items() if k != "subject"},
                            "run": str(r.root)} for r in self.runs]),
             "neurosoft_group_membership.csv")
        return written

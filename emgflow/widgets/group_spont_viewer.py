"""Group analysis of spontaneous EMG: does one group of conditions differ from another?

A member is one condition of one StartStop run ("Flex hand left stim 2"). Its group is
read from the names ("Группировать по": stimulation on or off, the task, the whole
condition, the run) and can be edited by hand. Its subject — the unit the statistics
pair on — is the task the condition name describes, so "Flex hand left stim 2" pairs
with "Flex hand left non stim". The tab compares two groups, A in blue and B in red,
or shows all of them, channel by channel, in three views:

* Сводка — one number per condition (RMS, amplitude, bursts, …): a dot each, the group
  median and quartiles; the comparison table is under the plots;
* Огибающая во времени — each group's mean RMS time course ± SE across conditions,
  along the segment (as a fraction of its length, or in seconds);
* Форма вспышек — each group's mean burst envelope ± SE (the mean of every condition's
  own mean, so a condition with ten bursts weighs as much as one with two).

Everything is read once, in the background, when a folder is added. The analysis is in
src/group_spont.py and src/group_stats.py; this file only draws.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

import src.group_spont as S
import src.group_stats as ST

from .group_common import (plural, GroupTab, begin_fill, comparison_display, dots_and_box, empty_panel,
                           end_fill, group_ticks, groups_display, label_space, legend_note, member_table,
                           style_axes, top_legend)

VIEWS = [("summary", "Сводка"), ("timecourse", "Огибающая во времени"),
         ("bursts", "Форма вспышек")]


class _Reader(QThread):
    done = Signal(object, object)

    def __init__(self, members, aliases) -> None:
        super().__init__()
        self._members, self._aliases = members, aliases

    def run(self) -> None:
        try:
            members = [S.SpontMember(m.root, m.condition, m.subject, m.state, True,
                                     m.parsed_subject, m.parsed_state) for m in self._members]
            self.done.emit((S.collect_summary(members, self._aliases),
                            S.collect_timecourses(members, self._aliases),
                            S.collect_burst_envelopes(members, self._aliases)), None)
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_") or "group"


class GroupSpontViewer(GroupTab):
    """The spontaneous-EMG group tab."""
    VIEWS = VIEWS
    EXPORT_PREFIX = "spont_group"

    def __init__(self, session=None) -> None:
        super().__init__(session)
        self.members: list[S.SpontMember] = []
        self.summary = self.tc = self.env = pd.DataFrame()
        self.norm = self.norm_tc = self.norm_env = pd.DataFrame()
        self._reader: _Reader | None = None
        self._reread = False
        self._missing: list[str] = []

        # ---- left: the conditions ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip("Найти в папке прогоны StartStop со спонтанной ЭМГ и добавить "
                                 "каждое их условие.")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.group_box = QComboBox()
        for key, label in S.GROUP_MODES.items():
            self.group_box.addItem(label, key)
        self.group_box.currentIndexChanged.connect(lambda _: self._regroup())
        self.table = member_table(["", "Задача", "Группа", "Условие", "Прогон"], stretch_col=3)
        self.table.setToolTip("Галочка — условие участвует. Группу можно переименовать двойным щелчком.")
        self.table.itemChanged.connect(self._on_table_edit)
        self.btn_all = QPushButton("отметить все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("снять все")
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        self.aliases = QLineEdit()
        self.aliases.setPlaceholderText("BB R=Biceps R; ch1=TA L")
        self.aliases.setToolTip("Одна мышца под разными именами в разных файлах: «имя в файле=общее "
                                "имя» через «;». Регистр, пробелы и подчёркивания и так не важны.")
        self.aliases.editingFinished.connect(self._read_all)
        self.members_note = QLabel("")
        self.members_note.setStyleSheet("color: #555; font-size: 11px;")

        # ---- top: what is compared ----
        self.metric_box = QComboBox()
        for key, label in S.METRICS.items():
            self.metric_box.addItem(label, key)
        self.norm_box = QComboBox()
        for key, label in S.NORM_LABELS.items():
            self.norm_box.addItem(label, key)
        self.norm_box.setToolTip("Нормировка на значение того же субъекта и канала в базовой "
                                 "группе. Число и длительность вспышек не нормируются.")
        self.base_label = QLabel("базовая группа")
        self.base_box = QComboBox()
        for box in (self.metric_box, self.norm_box, self.base_box):
            box.currentIndexChanged.connect(lambda _: self._on_analysis_control())

        # ---- options that belong to one view ----
        self.axis_box = QComboBox()
        self.axis_box.addItem("ось: доля сегмента (0–1)", "frac")
        self.axis_box.addItem("ось: секунды от начала", "t")
        self.axis_box.setToolTip("Сегменты разной длины сопоставимы только по доле сегмента.")
        self.axis_box.currentIndexChanged.connect(lambda _: self.redraw())
        self.show_members = QCheckBox("отдельные условия")
        self.show_members.toggled.connect(lambda _: self.redraw())
        self.view_options = {"summary": [], "timecourse": [self.axis_box, self.show_members],
                             "bursts": []}

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Группировать по</b>"))
        lv.addWidget(self.group_box)
        lv.addWidget(QLabel("<b>Условия</b>"))
        lv.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        lv.addLayout(row)
        lv.addWidget(self.members_note)
        lv.addWidget(QLabel("<b>Псевдонимы каналов</b>"))
        lv.addWidget(self.aliases)

        self.build_layout(left, [QLabel("<b>Показатель</b>"), self.metric_box,
                                 QLabel("<b>Нормировка</b>"), self.norm_box,
                                 self.base_label, self.base_box])
        self._sync_base_visibility()
        self.refresh_now()

    def background_threads(self) -> list:
        return [self._reader]

    # ------------------------------------------------------------------ #
    # The conditions
    # ------------------------------------------------------------------ #
    def _mode(self) -> str:
        return self.group_box.currentData() or "state"

    def _scan(self) -> None:
        start = str(self.members[-1].root.parent) if self.members else ""
        folder = QFileDialog.getExistingDirectory(self, "Папка с прогонами StartStop", start)
        if folder:
            n = self.add_members(S.scan_members(Path(folder)))
            if not n:
                self.status.setText("Новых условий со спонтанной ЭМГ в этой папке нет.")

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group (all its conditions)."""
        self.add_members(S.scan_members(Path(root), max_depth=0))

    def add_members(self, found: list[S.SpontMember]) -> int:
        known = {(str(m.root), m.condition) for m in self.members}
        new = [m for m in found if (str(m.root), m.condition) not in known]
        if not new:
            return 0
        mode = self._mode()
        for m in new:
            m.state = S.group_of(m, mode)
        self.members.extend(new)
        self.members.sort(key=lambda m: (m.subject, m.state, m.condition))
        self._fill_table()
        self._read_all()
        return len(new)

    def _read_all(self) -> None:
        if not self.members:
            return
        if self._reader is not None and self._reader.isRunning():
            self._reread = True
            return
        self.status.setText(f"Читаю {plural(len(self.members), 'условие', 'условия', 'условий')}…")
        self._reader = _Reader(list(self.members), S.parse_aliases(self.aliases.text()))
        self._reader.done.connect(self._on_read)
        self._reader.start()

    def _on_read(self, payload, error) -> None:
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось прочитать условия", error)
        else:
            self.summary, self.tc, self.env = payload
        if self._reread:
            self._reread = False
            self._read_all()
            return
        self._membership_changed()

    def _clear(self) -> None:
        self.members = []
        self.summary = self.tc = self.env = pd.DataFrame()
        self.norm = self.norm_tc = self.norm_env = pd.DataFrame()
        self._fill_table()
        self.picker.set_groups([])
        self.channels.set_channels([])
        for view in self.views.values():
            view.message("Группа пуста: добавьте папку с прогонами.")
            view.draw()
        self.stats.show_frame(None)
        self.members_note.setText("")
        self.status.setText("Группа пуста: добавьте папку с прогонами.")

    def _regroup(self) -> None:
        mode = self._mode()
        for m in self.members:
            m.state = S.group_of(m, mode)
        self._fill_table()
        self._membership_changed()

    def _fill_table(self) -> None:
        begin_fill(self.table)
        self.table.setRowCount(len(self.members))
        for i, m in enumerate(self.members):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if m.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            for col, text, editable in ((1, m.subject, False), (2, m.state, True),
                                        (3, m.condition, False), (4, m.root.name, False)):
                item = QTableWidgetItem(text)
                if not editable:
                    item.setFlags(Qt.ItemIsEnabled)
                if col == 4:
                    item.setToolTip(str(m.root))
                self.table.setItem(i, col, item)
        end_fill(self.table)

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        if not 0 <= item.row() < len(self.members):
            return
        m = self.members[item.row()]
        if item.column() == 0:
            m.include = item.checkState() == Qt.Checked
        elif item.column() == 2:
            m.state = item.text().strip() or m.state
        else:
            return
        self._membership_changed()

    def _set_all(self, on: bool) -> None:
        for m in self.members:
            m.include = on
        self._fill_table()
        self._membership_changed()

    def _active(self) -> list[S.SpontMember]:
        return [m for m in self.members if m.include]

    def _membership_changed(self) -> None:
        active = self._active()
        info = []
        for g in S.sort_states(m.state for m in active):
            ms = [m for m in active if m.state == g]
            info.append((g, len(ms), len({m.subject for m in ms})))
        self.picker.set_groups(info)
        current = self.base_box.currentText()
        self.base_box.blockSignals(True)
        self.base_box.clear()
        self.base_box.addItems([g for g, _, _ in info])
        if current in [g for g, _, _ in info]:
            self.base_box.setCurrentText(current)
        self.base_box.blockSignals(False)
        if not self.summary.empty:
            keys = {(str(m.root), m.condition) for m in active}
            s = self.summary[[k in keys for k in zip(self.summary["run"], self.summary["condition"])]]
            counts = s.dropna(subset=["rms_uv"]).groupby("channel")["run"].count()
            names = sorted(s["channel"].dropna().unique(), key=lambda c: (len(str(c)), str(c)))
            # channels recorded in at least two conditions start checked; the rest are one-offs
            self.channels.set_channels(names, checked={c for c in names if counts.get(c, 0) >= 2})
        self.members_note.setText(f"отмечено {len(active)} из {len(self.members)} условий · "
                                  f"{plural(len({m.subject for m in active}), 'задача', 'задачи', 'задач')} · "
                                  f"{plural(len(info), 'группа', 'группы', 'групп')}")
        self._sync_base_visibility()
        self.invalidate()

    def _on_analysis_control(self) -> None:
        self._sync_base_visibility()
        self.invalidate()

    def _sync_base_visibility(self) -> None:
        needs = self.norm_box.currentData() == S.NORM_BASELINE
        self.base_label.setVisible(needs)
        self.base_box.setVisible(needs)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def has_data(self) -> bool:
        return not self.summary.empty and bool(self._active())

    def prepare(self) -> None:
        active = self._active()
        labels = pd.DataFrame([{"run": str(m.root), "condition": m.condition,
                                "_group": m.state, "_subject": m.subject} for m in active])

        def relabel(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or labels.empty:
                return frame.iloc[0:0]
            f = frame.merge(labels, on=["run", "condition"], how="inner")
            f["state"], f["subject"] = f["_group"], f["_subject"]
            return f.drop(columns=["_group", "_subject"])

        summary, tc, env = relabel(self.summary), relabel(self.tc), relabel(self.env)
        metric, mode = self.metric_box.currentData(), self.norm_box.currentData()
        base = self.base_box.currentText()
        factors = (S.normalisation_factors(summary, base, metric="rms_uv")
                   if mode == S.NORM_BASELINE else pd.DataFrame())
        self.norm, self._missing = S.apply_normalisation(summary, factors, metric)
        self.norm_tc = S.apply_normalisation(tc, factors, "rms_uv")[0] if not tc.empty else tc
        self.norm_env = (S.apply_normalisation(env.rename(columns={"value": "raw"}), factors, "raw")[0]
                         if not env.empty else env)
        bits = [plural(len(active), "условие", "условия", "условий"),
                plural(len({m.subject for m in active}), "задача", "задачи", "задач"),
                plural(self.norm["channel"].nunique(), "канал", "канала", "каналов")]
        if mode == S.NORM_BASELINE:
            bits.append(f"нормировка на группу «{base}» того же субъекта и канала")
        if self._missing:
            bits.append(f"без базовой записи, исключены: {len(self._missing)} субъект/канал")
        self.status.setText("  ·  ".join(bits))

    def _unit(self, timecourse: bool = False) -> str:
        frame = self.norm_tc if timecourse else self.norm
        if frame.empty:
            return ""
        unit = str(frame["unit"].iloc[0])
        return "RMS, мкВ" if timecourse and unit == "rms_uv" else unit.replace("µV", "мкВ")

    def _p_values(self, frame: pd.DataFrame) -> dict:
        pair = self.picker.pair()
        if pair is None or frame.empty:
            return {}
        t = ST.comparison_table(frame, "channel", "value", "state", *pair)
        return {r["channel"]: r["p"] for _, r in t.iterrows()} if not t.empty else {}

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
            view.message("Нет групп: отметьте условия в списке слева.")
            return
        if not channels:
            view.message("Отметьте хотя бы один канал.")
            return
        getattr(self, f"_draw_{key}")(view, groups, self.picker.colors(), channels)

    def _draw_summary(self, view, groups, colors, channels) -> None:
        d_all = self.norm[self.norm["state"].isin(groups) & self.norm["channel"].isin(channels)]
        pvals = self._p_values(d_all)
        unit = self._unit()
        axes = view.panels(len(channels), legend_entries=len(groups),
                           bottom_extra=label_space(groups))
        for i, (ax, ch) in enumerate(zip(axes, channels)):
            d = d_all[d_all["channel"] == ch].dropna(subset=["value"])
            present = [g for g in groups if (d["state"] == g).any()]
            if not present:
                empty_panel(ax, ch, "в этих группах нет данных")
                continue
            labels = []
            for k, g in enumerate(present):
                dg = d[d["state"] == g]
                dots_and_box(ax, k, dg["value"].to_numpy(float), colors[g], seed=k + 7 * i)
                labels.append((g, len(dg)))
            group_ticks(ax, labels)
            style_axes(ax, self._title(ch, pvals.get(ch)), ylabel=unit)
            ax.set_ylim(bottom=0)
        top_legend(view.figure, [(g, colors[g], "dot") for g in groups])
        legend_note(view.figure, "точка — условие · черта — медиана группы · рамка — квартили")

    def _draw_timecourse(self, view, groups, colors, channels) -> None:
        if self.norm_tc.empty:
            view.message("В выбранных условиях нет таблиц RMS по окнам.")
            return
        axis = self.axis_box.currentData()
        tc = self.norm_tc[self.norm_tc["state"].isin(groups) & self.norm_tc["channel"].isin(channels)]
        means = S.mean_timecourse(tc, axis)
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_members.isChecked())
        for ax, ch in zip(axes, channels):
            d = tc[tc["channel"] == ch].dropna(subset=[axis, "value"])
            if d.empty:
                empty_panel(ax, ch, "в этих группах нет данных")
                continue
            if self.show_members.isChecked():
                for g in groups:
                    segs = [grp.sort_values(axis)[[axis, "value"]].to_numpy(float)
                            for _, grp in d[d["state"] == g].groupby(["run", "condition"])]
                    segs = [s for s in segs if len(s) > 1]
                    if segs:
                        ax.add_collection(LineCollection(segs, colors=colors[g], linewidths=0.6,
                                                         alpha=0.2, zorder=1))
            top, xs = [], []
            m_ch = means[means["channel"] == ch] if not means.empty else means
            for g in groups:
                s = m_ch[m_ch["state"] == g].sort_values("x") if not m_ch.empty else m_ch
                if s.empty:
                    continue
                ax.fill_between(s["x"], s["mean"] - s["se"], s["mean"] + s["se"], color=colors[g],
                                alpha=0.18, lw=0, zorder=2)
                ax.plot(s["x"], s["mean"], color=colors[g], lw=2.2, zorder=3)
                top.append(float((s["mean"] + s["se"]).max()))
                xs += [float(s["x"].min()), float(s["x"].max())]
            if not xs:
                xs = [float(d[axis].min()), float(d[axis].max())]
            ax.set_xlim(min(xs), max(xs))
            hi = max(top) if top else float(d["value"].quantile(0.95))
            if np.isfinite(hi) and hi > 0:
                ax.set_ylim(0, hi * 1.25)
            style_axes(ax, ch, ylabel=self._unit(timecourse=True),
                       xlabel="доля сегмента" if axis == "frac" else "время от начала сегмента, с")
        entries = [(f"{g}: среднее ± SE", colors[g], "line") for g in groups]
        if self.show_members.isChecked():
            entries.append(("отдельные условия", "0.4", "faint"))
        top_legend(view.figure, entries)
        legend_note(view.figure, "RMS по окнам вдоль сегмента; среднее — где есть хотя бы 2 условия")

    def _draw_bursts(self, view, groups, colors, channels) -> None:
        env = (self.norm_env[self.norm_env["state"].isin(groups)
                             & self.norm_env["channel"].isin(channels)]
               if not self.norm_env.empty else self.norm_env)
        if env.empty:
            view.message("В выбранных условиях и каналах нет сохранённых огибающих вспышек.")
            return
        means = S.mean_burst_envelope(env)
        shown = [c for c in channels if (env["channel"] == c).any()]
        axes = view.panels(len(shown), legend_entries=len(groups))
        for ax, ch in zip(axes, shown):
            m_ch = means[means["channel"] == ch]
            top = []
            for g in groups:
                s = m_ch[m_ch["state"] == g].sort_values("t")
                if s.empty:
                    continue
                ax.fill_between(s["t"], s["mean"] - s["se"], s["mean"] + s["se"], color=colors[g],
                                alpha=0.18, lw=0, zorder=2)
                ax.plot(s["t"], s["mean"], color=colors[g], lw=2.2, zorder=3)
                top.append(float((s["mean"] + s["se"]).max()))
            ax.axvline(0, color="0.5", lw=0.8, ls="--", zorder=0)
            if top and np.isfinite(max(top)) and max(top) > 0:
                ax.set_ylim(0, max(top) * 1.2)
            style_axes(ax, ch, xlabel="время от центра вспышки, с", ylabel=self._unit(timecourse=True))
        top_legend(view.figure, [(f"{g}: среднее ± SE", colors[g], "line") for g in groups])
        legend_note(view.figure, "средняя огибающая вспышки: сначала по вспышкам условия, потом по условиям")

    # ------------------------------------------------------------------ #
    # Table and export
    # ------------------------------------------------------------------ #
    def stats_for(self, key: str):
        channels = self.channels.checked()
        groups = self.picker.groups()
        frame = self.norm[self.norm["state"].isin(groups) & self.norm["channel"].isin(channels)]
        if frame.empty:
            return None, None, "нет данных для таблицы"
        pair = self.picker.pair()
        if pair is not None:
            table = ST.comparison_table(frame, "channel", "value", "state", *pair, units=channels)
            shown, bold = comparison_display(table, "channel", "канал", *pair)
            return shown, bold, ""
        table = ST.describe_groups(frame, "channel", "value", "state", groups, units=channels)
        return groups_display(table, "channel", "канал"), None, ""

    def export_tables(self, out: Path) -> list[str]:
        written = []

        def save(df: pd.DataFrame, name: str) -> None:
            if df is not None and not df.empty:
                df.to_csv(out / name, index=False)
                written.append(name)

        save(self.norm, "spont_group_summary_long.csv")
        groups = self.picker.groups()
        frame = self.norm[self.norm["state"].isin(groups)]
        pair = self.picker.pair()
        if pair is not None:
            save(ST.comparison_table(frame, "channel", "value", "state", *pair),
                 f"spont_group_{_safe(pair[0])}_vs_{_safe(pair[1])}.csv")
        else:
            save(ST.describe_groups(frame, "channel", "value", "state", groups),
                 "spont_group_by_group.csv")
        if not self.norm_tc.empty:
            save(S.mean_timecourse(self.norm_tc[self.norm_tc["state"].isin(groups)],
                                   self.axis_box.currentData()), "spont_group_mean_timecourse.csv")
        if not self.norm_env.empty:
            save(S.mean_burst_envelope(self.norm_env[self.norm_env["state"].isin(groups)]),
                 "spont_group_mean_burst_envelope.csv")
        save(pd.DataFrame([{"task": m.subject, "group": m.state, "included": m.include,
                            "condition": m.condition, "run": str(m.root)} for m in self.members]),
             "spont_group_membership.csv")
        return written

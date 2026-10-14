"""Group analysis of the Jendrassik manoeuvre: did the manoeuvre change the response?

Its own tab, and not a fifth view of "Group: Neurosoft", because the question has a
different shape. Everywhere else a recording contributes a LEVEL and two groups of
levels are compared. Here a recording contributes a DIFFERENCE: five stimuli at rest
against the five that follow with the manoeuvre, on the same channel, at the same
intensity, minutes apart (``src/jendrassik_trials.py``). So:

* zero is the interesting value, and it is drawn on every effect panel;
* the first test is one-sample — does the effect differ from zero in this group
  (``group_stats.vs_zero``) — and the comparison of two groups comes after it;
* no normalisation: a trial is already its own control.

Three views:

* Формы: покой и приём — the response itself. Rest is the control line (dashed), the
  manoeuvre is the solid line, and each has the transparent 95 % band of the group
  mean across recordings; "отдельные записи" puts every recording's own pair of
  curves faintly behind them, so a group mean can be checked against the subjects it
  is made of. "разница (приём − покой)" switches the panel to the effect itself, one
  band per group.
* Эффект: все пробы — the distribution of the effect, a dot per trial.
* Лучшие пробы — one dot per recording and channel: the trial where the manoeuvre
  acted most (``src/group_jendrassik.py`` chooses it exactly as the saved
  ``jendrassik_trials.csv`` does), with the spread of that recording's other trials
  behind the dot. Its statistics are computed separately from the "all trials" ones,
  and the two answer different questions: "does the manoeuvre do anything on
  average" against "how large is it where it works".

This tab replaced the "Приём Ендрассика" view of Group: Neurosoft (i-upri, PR #8),
which asked the same question of the best trials alone; everything that view did is
here, and what it brought with it — dots shaded by subject, the tooltip naming the
recording, zero kept inside the panel, the response area as a measure — is used by
both tabs from group_common now.

The analysis is in src/group_jendrassik.py, src/group_stats.py and src/group.py; this
file only draws.
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

import src.group as G
import src.group_jendrassik as JM
import src.group_stats as ST
from src.constants import TEXT_CURVES_RESP_TMIN

from .group_common import (GroupTab, begin_fill, comparison_display, dot_colors, dot_labels,
                           dots_and_box, empty_panel, end_fill, group_ticks, groups_display,
                           label_space, legend_note, member_table, plural, records,
                           shades_by_subject, short_name, style_axes, subjects, top_legend)
from .group_viewer import METRIC_LABELS

VIEWS = [
    ("waves", "Формы: покой и приём"),
    ("effect", "Эффект: все пробы"),
    ("best", "Лучшие пробы"),
]

#: Rest is drawn dashed and the manoeuvre solid, so a group keeps ONE colour: with two
#: groups a panel already carries four curves, and colouring the conditions as well
#: would leave nothing to tell the groups apart by.
REST_STYLE = dict(ls="--", lw=1.6)
ACT_STYLE = dict(ls="-", lw=2.2)

TRIAL_MODES = {"best": "лучшая проба каждой записи", "all": "все пробы вместе"}


class _PointReader(QThread):
    """Reads the recordings' per-curve metrics off disk."""
    done = Signal(object, object)

    def __init__(self, runs) -> None:
        super().__init__()
        self._runs = runs

    def run(self) -> None:
        try:
            self.done.emit(G.collect_points(self._runs), None)
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class _CurveWaveReader(QThread):
    """Reads every curve of every recording, once: the pentads are curves, not a plateau."""
    progress = Signal(int, int)
    done = Signal(object)

    def __init__(self, roots: list[str]) -> None:
        super().__init__()
        self._roots = roots

    def run(self) -> None:
        out = {}
        for i, root in enumerate(self._roots, 1):
            try:
                out[root] = G.curve_waveforms(root)
            except Exception:                          # noqa: BLE001 - that recording has none
                out[root] = (None, {})
            self.progress.emit(i, len(self._roots))
        self.done.emit(out)


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_") or "group"


class GroupJendrassikViewer(GroupTab):
    """The Jendrassik group tab."""
    VIEWS = VIEWS
    EXPORT_PREFIX = "jendrassik_group"

    def __init__(self, session=None) -> None:
        super().__init__(session)
        self.runs: list[G.GroupRun] = []
        self.points_all = pd.DataFrame()
        self.points = pd.DataFrame()
        self.trials = pd.DataFrame()
        self.best = pd.DataFrame()
        self._skipped: list[str] = []
        self._waves: dict = {}
        self._pending: list[G.GroupRun] = []
        self._reader: _PointReader | None = None
        self._wave_reader: _CurveWaveReader | None = None

        # ---- left: the recordings ----
        self.btn_scan = QPushButton("Добавить папку…")
        self.btn_scan.setToolTip("Найти в папке обработанные записи с приёмом Ендрассика "
                                 "(сценарий Jendrassik или JM в названии).")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear)
        self.group_box = QComboBox()
        for key, label in G.GROUP_MODES.items():
            self.group_box.addItem(label, key)
        # Level, not cohort: the manoeuvre was only ever recorded on patients, so
        # "SCI against control" is one group here and compares nothing.
        self.group_box.setCurrentIndex(list(G.GROUP_MODES).index("level"))
        self.group_box.setToolTip("Из каких меток названия складывается группа записи. "
                                  "Группу можно поправить руками в списке.")
        self.group_box.currentIndexChanged.connect(lambda _: self._regroup())
        self.table = member_table(["", "Субъект", "Группа", "Подгруппа", "Запись"], stretch_col=4)
        self.table.setToolTip("Галочка — запись участвует. Группу и подгруппу можно вписать "
                              "двойным щелчком.")
        self.table.itemChanged.connect(self._on_table_edit)
        self.btn_all = QPushButton("отметить все")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none = QPushButton("снять все")
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        self.split_sub = QCheckBox("делить группы по подгруппам")
        self.split_sub.setToolTip("Запись с заполненной подгруппой сравнивается в группе "
                                  "«группа · подгруппа»; записи без подгруппы остаются в своей.")
        self.split_sub.toggled.connect(lambda _: self._membership_changed(refill=True))
        self.members_note = QLabel("")
        self.members_note.setStyleSheet("color: #555; font-size: 11px;")

        # ---- top: what is compared ----
        self.metric_box = QComboBox()
        for key, label in METRIC_LABELS.items():
            self.metric_box.addItem(label, key)
        self.metric_box.setToolTip("На каком показателе считается эффект приёма. "
                                   "Лучшая проба всегда выбирается по размаху ответа, "
                                   "чтобы не менялась при смене показателя.")
        self.metric_box.currentIndexChanged.connect(lambda _: self.invalidate())
        self.effect_box = QComboBox()
        for key, (_, label) in JM.EFFECTS.items():
            self.effect_box.addItem(label, key)
        self.effect_box.setToolTip("Чем мерить эффект: процентом от покоя, разницей в "
                                   "единицах показателя или Cohen d (разница в долях "
                                   "разброса внутри пятёрок).")
        self.effect_box.currentIndexChanged.connect(lambda _: self.redraw())

        # ---- options that belong to one view ----
        self.trial_box = QComboBox()
        for key, label in TRIAL_MODES.items():
            self.trial_box.addItem(label, key)
        self.trial_box.setToolTip("По каким пробам усреднять формы ответов.")
        self.trial_box.currentIndexChanged.connect(lambda _: self.redraw())
        self.show_diff = QCheckBox("разница (приём − покой)")
        self.show_diff.setToolTip("Вместо двух кривых — сам эффект: одна кривая на группу "
                                  "с полосой 95% ДИ.")
        self.show_diff.toggled.connect(lambda _: self.redraw())
        self.show_traces = QCheckBox("отдельные записи")
        self.show_traces.setChecked(True)
        self.show_traces.toggled.connect(lambda _: self.redraw())
        self.show_spread = QCheckBox("разброс остальных проб записи")
        self.show_spread.setChecked(True)
        self.show_spread.toggled.connect(lambda _: self.redraw())
        self.clip_box = QCheckBox("обрезать выбросы (±1,5 IQR)")
        self.clip_box.setChecked(True)
        self.clip_box.setToolTip("Изменение в процентах не ограничено сверху: одна запись, "
                                 "где в покое ответа почти не было, даёт +1000% и "
                                 "расплющивает панель. Точки вне шкалы не теряются — их "
                                 "число написано в углу.")
        self.clip_box.toggled.connect(lambda _: self.redraw())
        # The same switch on both effect views: a widget can sit in only one entry of
        # view_options (the others are hidden), so there are two boxes kept in step.
        self.clip_box2 = QCheckBox("обрезать выбросы (±1,5 IQR)")
        self.clip_box2.setChecked(True)
        self.clip_box2.setToolTip(self.clip_box.toolTip())
        self.clip_box2.toggled.connect(lambda _: self.redraw())
        self.clip_box.toggled.connect(self.clip_box2.setChecked)
        self.clip_box2.toggled.connect(self.clip_box.setChecked)
        self.view_options = {
            "waves": [self.trial_box, self.show_diff, self.show_traces],
            "effect": [self.clip_box],
            "best": [self.show_spread, self.clip_box2],
        }

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)
        lv.addWidget(QLabel("<b>Группировать по</b>"))
        lv.addWidget(self.group_box)
        lv.addWidget(QLabel("<b>Записи с приёмом</b>"))
        lv.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        lv.addLayout(row)
        lv.addWidget(self.split_sub)
        lv.addWidget(self.members_note)

        self.build_layout(left, [QLabel("<b>Показатель</b>"), self.metric_box,
                                 QLabel("<b>Эффект</b>"), self.effect_box])
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
        folder = QFileDialog.getExistingDirectory(
            self, "Папка с обработанными записями Neurosoft", start)
        if not folder:
            return
        found = G.scan_neurosoft([Path(folder)], self._mode())
        jm = [r for r in found if JM.is_jendrassik_run(r)]
        n = self.add_runs(jm)
        if not n:
            self.status.setText("Записей с приёмом Ендрассика в этой папке нет"
                                + (f" (найдено {records(len(found))} других сценариев)."
                                   if found else "."))
        elif len(found) > len(jm):
            self.status.setText(f"Добавлено {records(n)} с приёмом; "
                                f"{len(found) - len(jm)} записей других сценариев пропущено.")

    def add_runs(self, found: list[G.GroupRun]) -> int:
        known = {str(r.root) for r in self.runs}
        new = [r for r in found if str(r.root) not in known]
        if not new:
            return 0
        self.runs.extend(new)
        self.runs.sort(key=lambda r: (r.tags.get("cohort", ""), r.subject, r.root.name))
        self._pending.extend(new)
        self._read_pending()
        self._membership_changed(refill=True)
        return len(new)

    def add_run(self, root: Path) -> None:
        """Put the run open in the main window into the group, if it is a Jendrassik one."""
        root = Path(root)
        if not G.looks_like_run(root):
            return
        tags = G.parse_neurosoft(root)
        run = G.GroupRun(root, tags["subject"], G.state_from_tags(tags, self._mode()), tags=tags)
        if JM.is_jendrassik_run(run):
            self.add_runs([run])

    def _read_pending(self) -> None:
        if (self._reader is not None and self._reader.isRunning()) or not self._pending:
            return
        batch, self._pending = self._pending, []
        copies = [G.GroupRun(r.root, r.subject, r.state, True, tags=r.tags) for r in batch]
        self.status.setText(f"Читаю {records(len(batch))}…")
        self._reader = _PointReader(copies)
        self._reader.done.connect(self._on_points)
        self._reader.start()

    def _on_points(self, points, error) -> None:
        if error:
            self.status.setText(error)
            QMessageBox.warning(self, "Не удалось прочитать записи", error)
        elif points is not None and not points.empty:
            points = points.copy()
            # A Neurosoft export is one configuration named after the recording; one
            # label for all of them, so the configuration never splits a group.
            points["config"] = "Neurosoft"
            self.points_all = (points if self.points_all.empty
                               else pd.concat([self.points_all, points], ignore_index=True))
        self._read_pending()
        self._membership_changed()

    def _clear(self) -> None:
        self.runs, self._pending = [], []
        self.points_all = self.points = pd.DataFrame()
        self.trials = self.best = pd.DataFrame()
        self._waves, self._skipped = {}, []
        self._fill_table()
        self.picker.set_groups([])
        self.channels.set_channels([])
        for view in self.views.values():
            view.message("Группа пуста: добавьте папку с записями.")
            view.draw()
        self.stats.show_frame(None)
        self.members_note.setText("")
        self.status.setText("Группа пуста: добавьте папку с записями.")

    def _regroup(self) -> None:
        mode = self._mode()
        for r in self.runs:
            if r.tags:
                r.state = G.state_from_tags(r.tags, mode)
        self._membership_changed(refill=True)

    def _active_runs(self) -> list[G.GroupRun]:
        return [r for r in self.runs if r.include]

    def _fill_table(self) -> None:
        begin_fill(self.table)
        self.table.setRowCount(len(self.runs))
        for i, r in enumerate(self.runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if r.include else Qt.Unchecked)
            self.table.setItem(i, 0, check)
            subject = QTableWidgetItem(r.subject)
            subject.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 1, subject)
            self.table.setItem(i, 2, QTableWidgetItem(r.state))
            self.table.setItem(i, 3, QTableWidgetItem(r.subgroup))
            name = QTableWidgetItem(r.root.name)
            name.setFlags(Qt.ItemIsEnabled)
            name.setToolTip(str(r.root) + "\n" + ", ".join(
                f"{k}: {v}" for k, v in r.tags.items() if v not in ("—", "") and k != "subject"))
            self.table.setItem(i, 4, name)
        end_fill(self.table)

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        if not 0 <= item.row() < len(self.runs):
            return
        run = self.runs[item.row()]
        if item.column() == 0:
            run.include = item.checkState() == Qt.Checked
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

    def _group_of(self, member) -> str:
        sub = (member.subgroup or "").strip()
        return f"{member.state} · {sub}" if sub and self.split_sub.isChecked() else member.state

    def _membership_changed(self, refill: bool = False) -> None:
        if refill:
            self._fill_table()
        active = self._active_runs()
        info = []
        for g in G.sort_states(self._group_of(r) for r in active):
            rs = [r for r in active if self._group_of(r) == g]
            info.append((g, len(rs), len({r.subject for r in rs})))
        self.picker.set_groups(info)
        if not self.points_all.empty:
            roots = {str(r.root) for r in active}
            names = self.points_all.loc[self.points_all["run"].isin(roots),
                                        "channel"].dropna().unique()
            self.channels.set_channels(sorted(names, key=lambda s: (len(s), s)))
        self.members_note.setText(f"отмечено {len(active)} из {len(self.runs)} · "
                                  f"{subjects(len({r.subject for r in active}))} · "
                                  f"{plural(len(info), 'группа', 'группы', 'групп')}")
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def has_data(self) -> bool:
        return not self.points_all.empty and bool(self._active_runs())

    def prepare(self) -> None:
        active = self._active_runs()
        state_of = {str(r.root): self._group_of(r) for r in active}
        pts = self.points_all[self.points_all["run"].isin(state_of)].copy()
        pts["state"] = pts["run"].map(state_of)
        self.points = pts
        self.trials, self._skipped = JM.trial_table(pts, self.metric_box.currentData())
        self.best = JM.best_only(self.trials)

        bits = [records(len(active)), subjects(len({r.subject for r in active})),
                plural(len(set(state_of.values())), "группа", "группы", "групп"),
                JM.summary(self.trials)]
        if self._skipped:
            bits.append(f"пропущено каналов с разрывами в нумерации кривых: "
                        f"{len(self._skipped)}")
        self.status.setText("  ·  ".join(bits))

    def _effect(self) -> tuple[str, str]:
        """``(column, axis label)`` of the effect measure chosen."""
        key = self.effect_box.currentData() or "delta_pct"
        col, label = JM.EFFECTS[key]
        if key == "delta":
            unit = METRIC_LABELS[self.metric_box.currentData()].split(", ")[-1]
            label = f"разница приём − покой, {unit}"
        return col, label

    def _frame_for(self, key: str) -> tuple[pd.DataFrame, str]:
        """The rows a view compares, and the word for what one row is."""
        if key == "effect" or (key == "waves" and self.trial_box.currentData() == "all"):
            return self.trials, "проб"
        return self.best, "записей"

    def _p_values(self, frame: pd.DataFrame, value: str, units: list[str]) -> dict:
        pair = self.picker.pair()
        if pair is None or frame.empty:
            return {}
        t = ST.comparison_table(frame, "channel", value, "state", *pair, units=units)
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
            view.message("Нет групп: отметьте записи в списке слева.")
            return
        if not channels:
            view.message("Отметьте хотя бы один канал.")
            return
        if self.trials.empty:
            view.message("Ни в одной записи не нашлось полной пары пятёрок.\n"
                         "Проба — это пять кривых покоя и пять следующих с приёмом; "
                         "запись короче десяти кривых сравнивать не с чем.")
            return
        getattr(self, f"_draw_{key}")(view, groups, self.picker.colors(), channels)

    def _draw_dots(self, view, groups, colors, channels, frame, dot: str, spread: bool) -> None:
        """The shared effect panel: a dot per row, the zero line, the group boxes."""
        value, ylabel = self._effect()
        d_all = frame[frame["state"].isin(groups) & frame["channel"].isin(channels)]
        d_all = d_all.dropna(subset=[value]) if not d_all.empty else d_all
        if d_all.empty:
            view.message("На этом показателе эффект не считается ни в одной пробе.\n"
                         "У латентностей это обычно значит, что ответ не найден.")
            return
        pvals = self._p_values(d_all, value, channels)
        ranges_of = self._trial_ranges(value) if spread else None
        clip = self.clip_box.isChecked()
        # One tone per subject, over every subject of the group and not only the ones
        # with a response on the channel drawn, so a person keeps their tone from
        # panel to panel; the tooltip gives the recording a dot came from.
        shade_of = shades_by_subject(d_all, groups, colors)
        hover = []
        axes = view.panels(len(channels), legend_entries=len(groups) + bool(spread),
                           bottom_extra=label_space(groups))
        for i, (ax, ch) in enumerate(zip(axes, channels)):
            d = d_all[d_all["channel"] == ch]
            present = [g for g in groups if (d["state"] == g).any()]
            if not present:
                empty_panel(ax, ch, "в этих группах нет проб")
                continue
            labels = []
            for k, g in enumerate(present):
                dg = d[d["state"] == g]
                ranges = ([ranges_of.get((r.run, r.channel), (np.nan, np.nan))
                           for r in dg.itertuples()] if ranges_of is not None else None)
                hover += [(ax, *point) for point in dots_and_box(
                    ax, k, dg[value].to_numpy(float), colors[g], seed=k + 7 * i,
                    ranges=ranges, colors=dot_colors(shade_of, g, dg["subject"]),
                    labels=dot_labels(dg))]
                labels.append((g, len(dg)))
            # Above the quartile box, which would wash it out, and below the dots.
            ax.axhline(0, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=1.5)
            group_ticks(ax, labels)
            style_axes(ax, self._title(ch, pvals.get(ch)), ylabel=ylabel)
            if clip:
                self._clip_outliers(ax, d[value].to_numpy(float))
            # Zero stays INSIDE the panel: autoscale puts it on the frame exactly when
            # every recording fell on one side of it — the case the line is there for.
            lo, hi = ax.get_ylim()
            span = (hi - lo) or 1.0
            ax.set_ylim(min(lo, -0.12 * span) if lo >= 0 else lo,
                        max(hi, 0.12 * span) if hi <= 0 else hi)
        entries = [(g, colors[g], "dot") for g in groups]
        if spread:
            entries.append(("разброс остальных проб записи", "0.45", "faint"))
        top_legend(view.figure, entries)
        view.set_hover(hover)
        note = (f"точка — {dot} · оттенок — субъект · черта — медиана группы · "
                "рамка — квартили · пунктир на нуле — приём ничего не изменил · "
                "наведите на точку, чтобы увидеть запись")
        if self.effect_box.currentData() == "delta_pct":
            note += " · процент не определён там, где в покое ответа нет"
        legend_note(view.figure, note)

    def _draw_effect(self, view, groups, colors, channels) -> None:
        self._draw_dots(view, groups, colors, channels, self.trials,
                        "одна проба (пятёрка покоя против пятёрки приёма)", spread=False)

    def _draw_best(self, view, groups, colors, channels) -> None:
        self._draw_dots(view, groups, colors, channels, self.best,
                        "лучшая проба записи", spread=self.show_spread.isChecked())

    @staticmethod
    def _clip_outliers(ax, values) -> None:
        """Tukey fences on the panel's own values, and a note of what fell outside.

        A percentage change has no upper bound — a recording whose rest pentad was
        almost silent reports +1000% — and one such dot flattens the box of every
        group onto the zero line. The axis is cut to q1 − 1.5·IQR … q3 + 1.5·IQR,
        never wider than the data itself, and what the cut hides is COUNTED in the
        corner: a hidden point the reader is not told about would be worse than an
        unreadable panel.
        """
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 4:
            return
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return
        lo = max(float(v.min()), q1 - 1.5 * iqr)
        hi = min(float(v.max()), q3 + 1.5 * iqr)
        n_out = int(((v < lo) | (v > hi)).sum())
        if not n_out:
            return
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(min(lo - pad, -pad), max(hi + pad, pad))
        ax.text(0.99, 0.97, f"{plural(n_out, 'точка', 'точки', 'точек')} вне шкалы",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="0.45")

    def _trial_ranges(self, value: str) -> dict:
        """``(min, max)`` of the effect over ALL trials of a recording and channel.

        Drawn behind the best-trial dot: it says whether the best trial stands out
        among the others or whether every intensity gave the same thing.
        """
        if self.trials.empty or value not in self.trials.columns:
            return {}
        agg = self.trials.groupby(["run", "channel"])[value].agg(["min", "max"])
        return {key: (row["min"], row["max"]) for key, row in agg.iterrows()}

    def _draw_waves(self, view, groups, colors, channels) -> None:
        active = [r for r in self._active_runs() if self._group_of(r) in groups]
        missing = [str(r.root) for r in active if str(r.root) not in self._waves]
        if missing:
            self._load_waves(missing)
            view.message(f"Читаю кривые: {len(active) - len(missing)} из {len(active)} "
                         "записей готово…\nЭто делается один раз.")
            return
        source, _ = self._frame_for("waves")
        diff = self.show_diff.isChecked()
        t_art = TEXT_CURVES_RESP_TMIN * 1e3
        axes = view.panels(len(channels), legend_entries=len(groups) + self.show_traces.isChecked())
        for ax, ch in zip(axes, channels):
            raw, _ = G.split_component(ch)
            rest_loaded, act_loaded, diff_loaded = [], [], []
            for r in active:
                times, waves = self._waves.get(str(r.root), (None, {}))
                if times is None or raw not in waves:
                    continue
                rows = source[(source["run"] == str(r.root)) & (source["channel"] == ch)]
                if rows.empty:
                    continue
                rest, act = JM.condition_means(waves[raw], JM.spans(rows))
                if rest is None or act is None:
                    continue
                g = self._group_of(r)
                rest_loaded.append((g, str(r.root), times, rest))
                act_loaded.append((g, str(r.root), times, act))
                diff_loaded.append((g, str(r.root), times, act - rest))
            if not act_loaded:
                empty_panel(ax, ch, "нет сохранённых эпох для этих проб")
                continue

            lows, highs, counts = [], [], []
            series = ([(diff_loaded, ACT_STYLE, 0.22)] if diff
                      else [(rest_loaded, REST_STYLE, 0.10), (act_loaded, ACT_STYLE, 0.22)])
            grid = None
            for loaded, style, alpha in series:
                grid, bands, traces = G.waveform_bands(loaded)
                if grid.size == 0:
                    continue
                t = grid * 1e3
                post = t >= t_art
                if self.show_traces.isChecked():
                    for g in groups:
                        segs = [np.column_stack([t[post], w[post]]) for gg, _, w in traces
                                if gg == g]
                        if segs:
                            ax.add_collection(LineCollection(
                                segs, colors=colors[g], linewidths=0.6, alpha=0.16,
                                linestyles=style["ls"], zorder=1))
                for g in groups:
                    b = bands[bands["group"] == g]
                    if b.empty:
                        continue
                    m, lo, hi = (b[c].to_numpy(float) for c in ("mean", "lo", "hi"))
                    ax.plot(t[post], m[post], color=colors[g], zorder=3, **style)
                    if np.isfinite(lo).any():
                        ax.fill_between(t[post], lo[post], hi[post], color=colors[g],
                                        alpha=alpha, lw=0, zorder=2)
                    lows.append(float(np.nanmin(np.where(np.isfinite(lo), lo, m)[post])))
                    highs.append(float(np.nanmax(np.where(np.isfinite(hi), hi, m)[post])))
                    if style is ACT_STYLE:
                        counts.append(f"{short_name(g, 14)}: {int(b['n'].iloc[0])}")
            if grid is None or grid.size == 0:
                empty_panel(ax, ch, "кривые не совпали по времени")
                continue
            t = grid * 1e3
            ax.axvspan(t[0], t_art, color="0.92", lw=0, zorder=0)
            ax.axhline(0, color="0.75", lw=0.6, zorder=0)
            if lows:
                span = max(highs) - min(lows) or 1.0
                ax.set_ylim(min(lows) - 0.12 * span, max(highs) + 0.12 * span)
            ax.set_xlim(t[0], t[-1])
            style_axes(ax, ch, xlabel="мс от стимула", ylabel="мкВ")
            if len(groups) <= 3 and counts:
                ax.text(0.99, 0.02, "записей — " + ", ".join(counts), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color="0.4")
        entries = [(g, colors[g], "line") for g in groups]
        if self.show_traces.isChecked():
            entries.append(("отдельные записи", "0.4", "faint"))
        mode = TRIAL_MODES[self.trial_box.currentData() or "best"]
        note = (f"линия — разница приём − покой, полоса — 95% ДИ по записям · {mode}"
                if diff else
                f"пунктир — покой, сплошная — приём, полоса — 95% ДИ по записям · {mode}")
        top_legend(view.figure, entries, title=note + " · мкВ · серым — артефакт стимула")

    def _load_waves(self, roots: list[str]) -> None:
        if self._wave_reader is not None and self._wave_reader.isRunning():
            return
        self._wave_reader = _CurveWaveReader(roots)
        self._wave_reader.progress.connect(
            lambda i, n: self.status.setText(f"Читаю кривые: {i} из {n}…"))
        self._wave_reader.done.connect(self._on_waves)
        self._wave_reader.start()

    def _on_waves(self, out: dict) -> None:
        self._waves.update(out)
        self.status.setText(f"Кривые прочитаны: {records(len(out))}.")
        self._dirty.add("waves")
        self._timer.start()

    # ------------------------------------------------------------------ #
    # Table and export
    # ------------------------------------------------------------------ #
    def stats_for(self, key: str):
        frame, word = self._frame_for(key)
        groups = self.picker.groups()
        channels = self.channels.checked()
        if frame.empty or not groups or not channels:
            return None, None, "нет данных для таблицы"
        value, _ = self._effect()
        d = frame[frame["state"].isin(groups) & frame["channel"].isin(channels)]
        if d.empty or d[value].notna().sum() == 0:
            return None, None, "на этом показателе эффект не считается"
        zero = ST.vs_zero(d, "channel", value, "state", groups, units=channels)
        pair = self.picker.pair()
        if pair is None:
            return (groups_display(zero, "channel", "канал", unit_word=word,
                                   columns={"p: эффект ≠ 0": "p0", "тест": "test0"}),
                    [bool(np.isfinite(p) and p < 0.05) for p in zero["p0"]], "")
        table = ST.comparison_table(d, "channel", value, "state", *pair, units=channels)
        extra = {f"p: {short_name(g, 16)} ≠ 0":
                 {r["channel"]: ST.format_p(r["p0"]) for _, r in zero.iterrows()
                  if r["group"] == g}
                 for g in pair}
        shown, bold = comparison_display(table, "channel", "канал", *pair,
                                         unit_word=word, extra=extra)
        return shown, bold, ""

    def export_tables(self, out: Path) -> list[str]:
        written = []

        def save(df: pd.DataFrame, name: str) -> None:
            if df is not None and not df.empty:
                df.to_csv(out / name, index=False)
                written.append(name)

        save(self.trials, "jendrassik_group_trials.csv")
        save(self.best, "jendrassik_group_best_trials.csv")
        groups, channels = self.picker.groups(), self.channels.checked()
        value, _ = self._effect()
        pair = self.picker.pair()
        for key, stem in (("effect", "all_trials"), ("best", "best_trials")):
            frame, _ = self._frame_for(key)
            if frame.empty:
                continue
            d = frame[frame["state"].isin(groups) & frame["channel"].isin(channels)]
            if d.empty:
                continue
            save(ST.vs_zero(d, "channel", value, "state", groups, units=channels),
                 f"jendrassik_group_{stem}_vs_zero.csv")
            if pair is not None:
                save(ST.comparison_table(d, "channel", value, "state", *pair, units=channels),
                     f"jendrassik_group_{stem}_{_safe(pair[0])}_vs_{_safe(pair[1])}.csv")
        save(pd.DataFrame([{"subject": r.subject, "group": r.state, "subgroup": r.subgroup,
                            "group_compared": self._group_of(r), "included": r.include,
                            **{k: v for k, v in r.tags.items() if k != "subject"},
                            "run": str(r.root)} for r in self.runs]),
             "jendrassik_group_membership.csv")
        return written

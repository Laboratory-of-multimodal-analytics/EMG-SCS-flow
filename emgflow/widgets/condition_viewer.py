"""Interactive surface for the Condition test.

The question this protocol asks is how the spinal response changes when another
stimulus precedes it, and by how much — so the view is built around comparing an
inter-stimulus interval against the control, not around browsing images:

  - pick a channel; amplitude-vs-ISI and the curves redraw;
  - click a point on the amplitude curve (or use the ISI list) to load that
    interval — its individual sweeps are drawn with the per-condition mean over
    them, and the control mean is overlaid for reference, so facilitation or
    suppression is read directly off the pair;
  - the waterfall shows every condition at once, aligned on the artifact, with
    the selected one highlighted;
  - persistence is plotted beside amplitude: a response can stop appearing on
    some sweeps without the mean amplitude moving much, and that is a different
    finding from the response getting smaller.

Everything is read from the arrays and tables the analysis wrote, so the GUI
never recomputes a detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHeaderView, QLabel, QListWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..results import COND_DIR, mode_dir

CONTROL = "control"


def _fmt(lab: float) -> str:
    return CONTROL if float(lab) == 0.0 else f"{float(lab):.0f} мс"


class ConditionViewer(QWidget):
    """Amplitude vs ISI + the sweeps behind it, for one Condition run."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.base: Path | None = None
        self.summary = pd.DataFrame()
        self.per_curve = pd.DataFrame()
        self.times: np.ndarray | None = None
        self.labels: list[float] = []
        self.curve_cond: np.ndarray | None = None
        self.channel: str | None = None
        self.selected: float | None = None
        self._aligned: dict[str, np.ndarray] = {}
        self._means: dict[str, np.ndarray] = {}
        self._markers: dict[str, np.ndarray] = {}
        self._present: dict[str, np.ndarray] = {}

        self.header = QLabel("<b>No Condition results in this run</b>")
        self.header.setWordWrap(True)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.channel_list.currentTextChanged.connect(self._on_channel)

        self.cond_list = QListWidget()
        self.cond_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cond_list.currentRowChanged.connect(self._on_condition)

        self.show_sweeps = QCheckBox("отдельные свипы")
        self.show_sweeps.setChecked(True)
        self.show_sweeps.toggled.connect(lambda _: self._draw())
        self.show_control = QCheckBox("наложить control")
        self.show_control.setChecked(True)
        self.show_control.toggled.connect(lambda _: self._draw())
        self.show_markers = QCheckBox("маркеры onset/P1/P2")
        self.show_markers.setChecked(True)
        self.show_markers.toggled.connect(lambda _: self._draw())
        self.mark_absent = QCheckBox("красным — свипы без ответа")
        self.mark_absent.setChecked(True)
        self.mark_absent.toggled.connect(lambda _: self._draw())

        self.stats = QTableWidget(0, 4)
        self.stats.setHorizontalHeaderLabels(["ISI", "N", "ампл. мВ", "persist."])
        self.stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats.setEditTriggers(QAbstractItemView.NoEditTriggers)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(self.header)
        lv.addWidget(QLabel("<b>Каналы</b>"))
        lv.addWidget(self.channel_list, 1)
        lv.addWidget(QLabel("<b>Интервал (ISI)</b>"))
        lv.addWidget(self.cond_list, 1)
        lv.addWidget(self.show_sweeps)
        lv.addWidget(self.show_control)
        lv.addWidget(self.show_markers)
        lv.addWidget(self.mark_absent)
        lv.addWidget(QLabel("<b>По условиям</b>"))
        lv.addWidget(self.stats, 1)

        self.fig = Figure(figsize=(8, 7), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        hint = QLabel("Клик по точке на графике амплитуды — поднять этот интервал ниже.")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        page_amp = QWidget()
        pv = QVBoxLayout(page_amp)
        pv.setContentsMargins(2, 2, 2, 2)
        pv.addWidget(self.canvas, 1)
        pv.addWidget(hint)

        # Waterfall lives on its own page: it wants the full height, and it
        # answers a different question — how the whole ISI series moves at once,
        # rather than one interval against the control.
        self.wf_fig = Figure(figsize=(8, 10), layout="constrained")
        self.wf_canvas = FigureCanvasQTAgg(self.wf_fig)
        wf_scroll = QWidget()
        wv = QVBoxLayout(wf_scroll)
        wv.setContentsMargins(2, 2, 2, 2)
        wv.addWidget(self.wf_canvas, 1)
        wf_hint = QLabel("Control внизу, интервал растёт вверх. Выбранный выделен красным.")
        wf_hint.setStyleSheet("color: gray; font-size: 10px;")
        wv.addWidget(wf_hint)

        right = QTabWidget()
        right.addTab(page_amp, "Ответ vs интервал")
        right.addTab(wf_scroll, "Waterfall")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([280, 900])
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, root: Path) -> None:
        self.base = mode_dir(Path(root), COND_DIR)
        arr = self.base / "arrays"
        excel = self.base / "Excel"
        self._aligned, self._means = {}, {}
        self.channel_list.blockSignals(True); self.channel_list.clear()
        self.channel_list.blockSignals(False)
        self.cond_list.blockSignals(True); self.cond_list.clear()
        self.cond_list.blockSignals(False)

        summary_csv = excel / "condition_summary.csv"
        if not summary_csv.exists() or not arr.exists():
            self.header.setText("<b>No Condition results in this run</b>")
            self._blank("This run produced no Condition-test outputs.")
            return

        self.summary = pd.read_csv(summary_csv)
        pc = excel / "condition_amplitudes_per_curve.csv"
        self.per_curve = pd.read_csv(pc) if pc.exists() else pd.DataFrame()
        self.times = np.load(arr / "times_ms.npy") if (arr / "times_ms.npy").exists() else None
        self.labels = (list(np.load(arr / "condition_labels.npy"))
                       if (arr / "condition_labels.npy").exists() else [])
        cc = arr / "curve_condition.npy"
        self.curve_cond = np.load(cc) if cc.exists() else None
        for f in sorted(arr.glob("*_curves_aligned.npy")):
            self._aligned[f.name[: -len("_curves_aligned.npy")]] = np.load(f)
        for f in sorted(arr.glob("*_condition_means.npy")):
            self._means[f.name[: -len("_condition_means.npy")]] = np.load(f)
        for f in sorted(arr.glob("*_markers_ms.npy")):
            self._markers[f.name[: -len("_markers_ms.npy")]] = np.load(f)
        for f in sorted(arr.glob("*_present.npy")):
            self._present[f.name[: -len("_present.npy")]] = np.load(f)

        chans = sorted(self.summary["Channel"].astype(str).unique(),
                       key=lambda s: (len(s), s))
        n_curves = 0 if self.curve_cond is None else len(self.curve_cond)
        self.header.setText(
            f"<b>Condition test</b><br><span style='color:gray'>"
            f"{len(self.labels)} интервалов · {len(chans)} каналов · {n_curves} кривых</span>"
        )
        self.channel_list.blockSignals(True)
        self.channel_list.addItems(chans)
        self.channel_list.blockSignals(False)
        self.cond_list.blockSignals(True)
        self.cond_list.addItems([_fmt(l) for l in self.labels])
        self.cond_list.blockSignals(False)
        if chans:
            self.channel_list.setCurrentRow(0)
        else:
            self._blank("No responding channels.")

    # ------------------------------------------------------------------ #
    def _on_channel(self, ch: str) -> None:
        self.channel = ch or None
        # Default to the first non-control interval — control is the reference,
        # so it is the least interesting thing to open on.
        if self.selected is None and len(self.labels) > 1:
            self.selected = self.labels[1]
            self.cond_list.blockSignals(True); self.cond_list.setCurrentRow(1)
            self.cond_list.blockSignals(False)
        self._draw()

    def _on_condition(self, row: int) -> None:
        if 0 <= row < len(self.labels):
            self.selected = self.labels[row]
            self._draw()

    def _on_click(self, event) -> None:
        if event.inaxes is not getattr(self, "_ax_amp", None) or event.xdata is None:
            return
        if not self.labels:
            return
        i = int(np.clip(round(event.xdata), 0, len(self.labels) - 1))
        self.selected = self.labels[i]
        self.cond_list.blockSignals(True); self.cond_list.setCurrentRow(i)
        self.cond_list.blockSignals(False)
        self._draw()

    def _blank(self, text: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", wrap=True)
        ax.set_axis_off()
        self._ax_amp = None
        self.stats.setRowCount(0)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        ch = self.channel
        if not ch or self.summary.empty or self.times is None:
            return
        d = self.summary[self.summary["Channel"].astype(str) == ch]
        if d.empty:
            self._blank(f"{ch}: нет данных."); return

        def _key(v):
            return 0.0 if str(v) == CONTROL else float(v)
        d = d.assign(_isi=d["Condition (ISI ms)"].map(_key)).sort_values("_isi")
        xs = np.arange(len(d))
        amp = d["Amplitude mean mV"].to_numpy(float)
        se = d["Amplitude SE mV"].to_numpy(float)
        pers = d["Persistence"].to_numpy(float)
        tags = [_fmt(v) for v in d["_isi"]]

        self.fig.clear()
        ax_amp, ax_pers, ax_cur = self.fig.subplots(3, 1, height_ratios=[1.1, 0.55, 1.5])
        self._ax_amp = ax_amp

        ctrl = amp[0] if len(amp) and d["_isi"].iloc[0] == 0.0 else np.nan
        if np.isfinite(ctrl):
            ax_amp.axhline(ctrl, color="0.6", lw=1.0, ls="--", zorder=1,
                           label=f"control = {ctrl:.3f} мВ")
        ax_amp.errorbar(xs, amp, yerr=se, marker="s", ms=5, lw=1.4, color="k",
                        capsize=3, zorder=3)
        sel_i = None
        if self.selected is not None:
            hits = np.where(d["_isi"].to_numpy() == float(self.selected))[0]
            if len(hits):
                sel_i = int(hits[0])
                ax_amp.plot([sel_i], [amp[sel_i]], "o", ms=13, mfc="none",
                            mec="#ff8800", mew=2.2, zorder=4)
        ax_amp.set_xticks(xs); ax_amp.set_xticklabels(tags, rotation=45, fontsize=7)
        ax_amp.set_ylabel("Амплитуда, мВ")
        ax_amp.set_title(f"{ch} — ответ против интервала", fontsize=10, loc="left")
        ax_amp.grid(True, color="0.92", lw=0.6)
        if np.isfinite(ctrl):
            ax_amp.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax_amp.spines[s].set_visible(False)

        ax_pers.bar(xs, pers, color="0.55")
        if sel_i is not None:
            ax_pers.bar([sel_i], [pers[sel_i]], color="#ff8800")
        ax_pers.set_ylim(0, 1.05); ax_pers.set_xticks(xs)
        ax_pers.set_xticklabels(tags, rotation=45, fontsize=7)
        ax_pers.set_ylabel("доля с ответом")
        ax_pers.grid(True, axis="y", color="0.92", lw=0.6)
        for s in ("top", "right"):
            ax_pers.spines[s].set_visible(False)

        self._draw_curves(ax_cur, ch)
        self.canvas.draw_idle()
        self._draw_waterfall(ch)
        self._fill_stats(d, tags, amp, pers)

    def _draw_curves(self, ax, ch: str) -> None:
        t = self.times
        means = self._means.get(ch)
        aligned = self._aligned.get(ch)
        if means is None or self.selected is None:
            ax.text(0.5, 0.5, "Нет сохранённых кривых для этого канала.",
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_axis_off(); return

        i = self.labels.index(self.selected) if self.selected in self.labels else 0
        present = self._present.get(ch)
        n_abs = 0
        if self.show_sweeps.isChecked() and aligned is not None and self.curve_cond is not None:
            rows = np.where(self.curve_cond == float(self.selected))[0]
            for k in rows:
                if k >= len(aligned):
                    continue
                ok = True if present is None or k >= len(present) else bool(present[k])
                if not ok:
                    n_abs += 1
                red = self.mark_absent.isChecked() and not ok
                ax.plot(t, aligned[k], color="#d62728" if red else "0.6",
                        lw=0.6, alpha=0.85 if red else 0.55, zorder=2 if red else 1)
        if self.show_control.isChecked() and 0.0 in self.labels:
            ci = self.labels.index(0.0)
            ax.plot(t, means[ci], color="0.35", lw=1.4, ls="--", zorder=3, label="control")
        ax.plot(t, means[i], color="#d73027", lw=2.0, zorder=4, label=_fmt(self.selected))
        ax.axvline(0, color="tab:blue", lw=1.0, ls=":", zorder=2)
        # The latencies the amplitudes were actually sampled at — this is how you
        # check the numbers came off the response and not off the artifact tail.
        mk = self._markers.get(ch)
        if self.show_markers.isChecked() and mk is not None:
            for t_ms, col, nm in zip(mk, ("tab:blue", "tab:red", "tab:green"),
                                     ("onset", "P1", "P2")):
                if np.isfinite(t_ms):
                    ax.axvline(float(t_ms), color=col, lw=1.1, ls="--", alpha=0.85,
                               zorder=5, label=nm)

        seg = means[np.isfinite(means)]
        if seg.size:
            lo, hi = float(np.nanmin(means)), float(np.nanmax(means))
            pad = 0.1 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("мс относительно артефакта")
        ax.set_ylabel("мВ")
        extra = f" · {n_abs} свипов без ответа" if n_abs else ""
        ax.set_title(f"{ch} — {_fmt(self.selected)}: свипы и среднее, control пунктиром{extra}",
                     fontsize=10, loc="left")
        ax.grid(True, color="0.92", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _draw_waterfall(self, ch: str) -> None:
        """Every condition stacked and aligned on the artifact, selected in red.

        The per-interval view answers "how does this one compare with control";
        this one answers "where along the series does the response turn over",
        which is only visible with all the intervals in one picture.
        """
        self.wf_fig.clear()
        means = self._means.get(ch)
        t = self.times
        if means is None or t is None or not self.labels:
            ax = self.wf_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Нет сохранённых кривых.", ha="center", va="center",
                    color="gray"); ax.set_axis_off(); self.wf_canvas.draw_idle(); return

        ax = self.wf_fig.add_subplot(111)
        span = np.nanmax([np.nanmax(w) - np.nanmin(w) for w in means]) or 1.0
        step = span * 1.1
        for row, lab in enumerate(self.labels):
            y = means[row] + row * step
            sel = self.selected is not None and float(lab) == float(self.selected)
            ax.plot(t, y, color="#d73027" if sel else "k", lw=1.6 if sel else 0.8, zorder=3 if sel else 2)
            ax.text(t[0], row * step, _fmt(lab) + "  ", ha="right", va="center",
                    fontsize=8, color="#d73027" if sel else "0.25",
                    fontweight="bold" if sel else "normal")
            ax.plot(0, row * step, "|", color="tab:blue", ms=9, mew=1.4, zorder=4)
        mk = self._markers.get(ch)
        if self.show_markers.isChecked() and mk is not None:
            for t_ms, col in zip(mk, ("tab:blue", "tab:red", "tab:green")):
                if np.isfinite(t_ms):
                    ax.axvline(float(t_ms), color=col, lw=1.0, ls="--", alpha=0.6, zorder=1)
        ax.set_yticks([])
        ax.set_xlabel("мс относительно артефакта")
        ax.set_title(f"{ch} — все интервалы, выровнено по артефакту", fontsize=10, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        self.wf_canvas.draw_idle()

    def _fill_stats(self, d, tags, amp, pers) -> None:
        n = d["N"].to_numpy()
        self.stats.setRowCount(len(tags))
        for r, tag in enumerate(tags):
            vals = [tag, str(int(n[r])), f"{amp[r]:.3f}", f"{pers[r]:.2f}"]
            for c, v in enumerate(vals):
                self.stats.setItem(r, c, QTableWidgetItem(v))

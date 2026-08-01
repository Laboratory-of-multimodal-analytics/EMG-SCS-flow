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
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHeaderView, QLabel, QListWidget, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from src.condition import load_overrides, recompute_corrections, save_overrides

from ..results import COND_DIR, mode_dir
from ..templates import build_template
from .template_editor import TemplateEditor

CONTROL = "control"

#: Red is reserved for the SELECTED interval — it is the thing in focus. A sweep
#: the presence gate rejected gets its own hue instead, or the two read as the
#: same annotation at a glance.
SELECTED_COLOR = "#d73027"
ABSENT_COLOR = "#7b3294"


def _plural_curves(n: int) -> str:
    """1 кривая / 2 кривые / 5 кривых.

    "Кривая", not "свип": it is the word the station itself uses ("количество
    кривых в файле") and the one the clinicians use, so the labels match what
    they already read on the machine.
    """
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return f"{n} кривая"
    if n10 in (2, 3, 4) and n100 not in (12, 13, 14):
        return f"{n} кривые"
    return f"{n} кривых"


def _fmt(lab: float) -> str:
    return CONTROL if float(lab) == 0.0 else f"{float(lab):.0f} мс"


class ConditionViewer(QWidget):
    """Amplitude vs ISI + the curves behind it, for one Condition run."""

    rerun_requested = Signal()
    #: Corrections applied and the tables on disk rewritten — see ScenarioViewer.
    results_changed = Signal()

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
        self._means_time: dict[str, np.ndarray] = {}
        self.root: Path | None = None
        self.overrides: dict = {}
        self._span: tuple[float, float] | None = None
        self._selector = None
        self.times_full: np.ndarray | None = None
        self.cond_artifact: np.ndarray | None = None

        self.header = QLabel("<b>No Condition results in this run</b>")
        self.header.setWordWrap(True)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.channel_list.currentTextChanged.connect(self._on_channel)

        self.cond_list = QListWidget()
        self.cond_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cond_list.currentRowChanged.connect(self._on_condition)

        self.show_sweeps = QCheckBox("отдельные кривые")
        self.show_sweeps.setChecked(True)
        self.show_sweeps.toggled.connect(lambda _: self._draw())
        self.show_control = QCheckBox("наложить control")
        self.show_control.setChecked(True)
        self.show_control.toggled.connect(lambda _: self._draw())
        self.show_markers = QCheckBox("маркеры onset/P1/P2")
        self.show_markers.setChecked(True)
        self.show_markers.toggled.connect(lambda _: self._draw())
        self.mark_absent = QCheckBox("выделять кривые без ответа")
        self.mark_absent.setChecked(True)
        self.mark_absent.toggled.connect(lambda _: self._draw())
        # Off = real recording time, which is where the travel of the response
        # along the curve is visible; alignment makes the shapes comparable but
        # removes exactly that.
        self.wf_aligned = QCheckBox("waterfall: выровнять по артефакту")
        self.wf_aligned.setChecked(True)
        self.wf_aligned.toggled.connect(lambda _: self._draw())

        # ---- manual correction, mirroring the SIR crop review ----
        self.btn_window = QPushButton("Задать окно ответа")
        self.btn_window.setEnabled(False)
        self.btn_window.setToolTip(
            "Протяните мышью по нижнему графику над нужным ответом, затем нажмите.\n"
            "Откроется редактор, где onset, P1 и P2 можно расставить кликами.\n"
            "Правка сохраняется только для этого канала, в общий банк не идёт."
        )
        self.btn_window.clicked.connect(self._set_window)
        self.btn_reject = QPushButton("Ответа нет")
        self.btn_reject.setToolTip("Пометить канал как не отвечающий и обнулить его амплитуды.")
        self.btn_reject.clicked.connect(self._toggle_reject)
        self.btn_clear = QPushButton("Снять правку")
        self.btn_clear.clicked.connect(self._clear_override)
        # Local: the correction touches one channel, and everything needed to
        # redo it is already saved, so re-parsing the raw export would be minutes
        # of work to answer a question about one trace.
        self.btn_rerun = QPushButton("Пересчитать по правкам")
        self.btn_rerun.setToolTip(
            "Внесите все правки, затем нажмите один раз.\n"
            "Пересчитываются все поправленные каналы из сохранённых кривых;\n"
            "исходный файл не перечитывается."
        )
        self.btn_rerun.clicked.connect(self._recompute)
        self.ov_label = QLabel("правок нет")
        self.ov_label.setStyleSheet("color: gray; font-size: 10px;")
        self.ov_label.setWordWrap(True)

        self.stats = QTableWidget(0, 4)
        self.stats.setHorizontalHeaderLabels(["ISI", "N", "ампл. мкВ", "persist."])
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
        lv.addWidget(self.wf_aligned)
        lv.addWidget(QLabel("<b>Ручная правка канала</b>"))
        for b in (self.btn_window, self.btn_reject, self.btn_clear, self.btn_rerun):
            lv.addWidget(b)
        lv.addWidget(self.ov_label)
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
        self.root = Path(root)
        self.overrides = load_overrides(self.root)
        self._span = None
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
        for f in sorted(arr.glob("*_condition_means_time.npy")):
            self._means_time[f.name[: -len("_condition_means_time.npy")]] = np.load(f)
        tf = arr / "times_full_ms.npy"
        self.times_full = np.load(tf) if tf.exists() else None
        ca = arr / "condition_artifact_ms.npy"
        self.cond_artifact = np.load(ca) if ca.exists() else None

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
        amp = d["Amplitude mean uV"].to_numpy(float)
        se = d["Amplitude SE uV"].to_numpy(float)
        pers = d["Persistence"].to_numpy(float)
        tags = [_fmt(v) for v in d["_isi"]]

        self.fig.clear()
        ax_amp, ax_pers, ax_cur = self.fig.subplots(3, 1, height_ratios=[1.1, 0.55, 1.5])
        self._ax_amp = ax_amp

        ctrl = amp[0] if len(amp) and d["_isi"].iloc[0] == 0.0 else np.nan
        if np.isfinite(ctrl):
            ax_amp.axhline(ctrl, color="0.6", lw=1.0, ls="--", zorder=1,
                           label=f"control = {ctrl:.0f} мкВ")
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
        ax_amp.set_ylabel("Амплитуда, мкВ")
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
        # Drag over the response to define a template window, exactly as the SIR
        # crop review does it.
        self._selector = SpanSelector(
            ax_cur, self._on_span, "horizontal", useblit=False,
            props=dict(alpha=0.25, facecolor="#ffb703"),
        )
        self.canvas.draw_idle()
        self._refresh_override_label()
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
                    n_abs += 1                     # counted before use in the label
                flag = self.mark_absent.isChecked() and not ok
                ax.plot(t, aligned[k], color=ABSENT_COLOR if flag else "0.6",
                        lw=0.7 if flag else 0.6, alpha=0.9 if flag else 0.55,
                        zorder=2 if flag else 1,
                        label="кривая без ответа" if (flag and n_abs == 1) else None)
        if self.show_control.isChecked() and 0.0 in self.labels:
            ci = self.labels.index(0.0)
            ax.plot(t, means[ci], color="0.35", lw=1.4, ls="--", zorder=3, label="control")
        ax.plot(t, means[i], color=SELECTED_COLOR, lw=2.0, zorder=4,
                label=f"{_fmt(self.selected)} — среднее")
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
        ax.set_ylabel("мкВ")
        cur = self.overrides.get(ch, {})
        if cur.get("window_ms"):
            a, b = cur["window_ms"]
            ax.axvspan(a, b, color="#ffb703", alpha=0.18, lw=0, zorder=0)
        elif self._span is not None:
            ax.axvspan(self._span[0], self._span[1], color="#ffb703", alpha=0.12, lw=0, zorder=0)
        extra = f" · {_plural_curves(n_abs)} без ответа" if n_abs else ""
        if cur.get("reject"):
            extra += " · ПОМЕЧЕН КАК БЕЗ ОТВЕТА"
        ax.set_title(f"{ch} — {_fmt(self.selected)}: кривые и среднее, control пунктиром{extra}",
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
        aligned = self.wf_aligned.isChecked()
        if aligned:
            means, t = self._means.get(ch), self.times
        else:
            means, t = self._means_time.get(ch), self.times_full
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
            ax.plot(t, y, color=SELECTED_COLOR if sel else "k",
                    lw=1.6 if sel else 0.8, zorder=3 if sel else 2)
            ax.text(t[0], row * step, _fmt(lab) + "  ", ha="right", va="center",
                    fontsize=8, color=SELECTED_COLOR if sel else "0.25",
                    fontweight="bold" if sel else "normal")
            # Artifact marker: at zero when aligned, at its real position when not.
            a_ms = 0.0
            if not aligned and self.cond_artifact is not None and row < len(self.cond_artifact):
                a_ms = float(self.cond_artifact[row])
            ax.plot(a_ms, row * step, "|", color="tab:blue", ms=9, mew=1.4, zorder=4)
        mk = self._markers.get(ch)
        if aligned and self.show_markers.isChecked() and mk is not None:
            for t_ms, col in zip(mk, ("tab:blue", "tab:red", "tab:green")):
                if np.isfinite(t_ms):
                    ax.axvline(float(t_ms), color=col, lw=1.0, ls="--", alpha=0.6, zorder=1)
        ax.set_yticks([])
        ax.set_xlabel("мс относительно артефакта" if aligned else "мс (реальное время кривой)")
        ax.set_title(
            f"{ch} — все интервалы, "
            + ("выровнено по артефакту" if aligned else "реальное время: ответ едет за артефактом"),
            fontsize=10, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        self.wf_canvas.draw_idle()

    # ---- manual correction ------------------------------------------- #
    def _on_span(self, lo: float, hi: float) -> None:
        if hi - lo < 0.5:                      # a click, not a drag
            return
        self._span = (float(lo), float(hi))
        self.btn_window.setEnabled(True)
        self.btn_window.setText(f"Задать окно {lo:.0f}–{hi:.0f} мс")
        self._draw()

    def _set_window(self) -> None:
        """Window -> editor -> per-channel override.

        The same editor the SIR review uses, so onset/P1/P2 mean the same thing
        and are placed the same way. What is saved is the channel's own
        correction, NOT a new entry in the shared bank: a template added to the
        bank changes matching for every channel in the run, which is not what
        drawing on one channel should do.
        """
        if not self.channel or self._span is None or self.root is None:
            return
        lo, hi = self._span
        means = self._means.get(self.channel)
        if means is None or self.times is None:
            return
        t_s = np.asarray(self.times) / 1e3
        win = (self.times >= lo) & (self.times <= hi)
        ref = int(np.argmax([np.ptp(np.nan_to_num(m)[win]) for m in means]))
        wave = np.nan_to_num(means[ref])
        try:
            tpl = build_template(t_s, wave, lo / 1e3, hi / 1e3,
                                 source=f"{self.channel} · {_fmt(self.labels[ref])} · "
                                        f"{lo:.0f}–{hi:.0f} мс от артефакта")
        except Exception as exc:
            self.ov_label.setText(f"<span style='color:#b00'>Не удалось построить шаблон: {exc}</span>")
            return
        dlg = TemplateEditor(tpl, self)
        if not dlg.exec():
            return
        t = dlg.edited_template()
        self.overrides.setdefault(self.channel, {}).pop("reject", None)
        self.overrides[self.channel]["window_ms"] = [round(lo, 2), round(hi, 2)]
        self.overrides[self.channel]["markers_ms"] = {
            "onset": round(float(t.times[int(t.onset_idx)]) * 1e3, 2),
            "p1": round(float(t.times[int(t.peak1_idx)]) * 1e3, 2),
            "p2": round(float(t.times[int(t.peak2_idx)]) * 1e3, 2),
        }
        self._save_overrides()

    def _toggle_reject(self) -> None:
        if not self.channel or self.root is None:
            return
        cur = self.overrides.get(self.channel, {})
        if cur.get("reject"):
            self.overrides.pop(self.channel, None)
        else:
            self.overrides[self.channel] = {"reject": True}
        self._save_overrides()

    def _clear_override(self) -> None:
        if not self.channel or self.root is None:
            return
        self.overrides.pop(self.channel, None)
        self._span = None
        self.btn_window.setEnabled(False)
        self.btn_window.setText("Задать окно ответа")
        self._save_overrides()

    def _recompute(self) -> None:
        """Apply every correction at once — corrections are made one at a time
        but the run is judged as a whole."""
        if self.root is None:
            return
        ch, sel = self.channel, self.selected
        try:
            done = recompute_corrections(self.root)
        except Exception as exc:
            self.ov_label.setText(f"<span style='color:#b00'>Не удалось пересчитать: {exc}</span>")
            return
        if not done:
            self.ov_label.setText("Правок нет — пересчитывать нечего.")
            return
        self.load(self.root)                     # re-read the updated tables and arrays
        # Back to the same channel AND the same condition: corrections are made
        # one interval at a time, and losing the place after each pass turns a
        # quick pass over a run into re-navigating it.
        rows = [self.channel_list.item(i).text() for i in range(self.channel_list.count())]
        if ch in rows:
            self.channel_list.setCurrentRow(rows.index(ch))
        if sel is not None and sel in self.labels:
            self.cond_list.setCurrentRow(self.labels.index(sel))
        self.results_changed.emit()
        parts = [(f"{d['channel']}: {d['error']}" if "error" in d
                  else f"{d['channel']} — {d['source']}") for d in done]
        self.ov_label.setText("Пересчитано: " + "; ".join(parts))

    def _save_overrides(self) -> None:
        save_overrides(self.root, self.overrides)
        self._refresh_override_label()
        self._draw()

    def _refresh_override_label(self) -> None:
        cur = self.overrides.get(self.channel or "", {})
        if cur.get("reject"):
            txt = f"<b>{self.channel}</b>: помечен как без ответа"
        elif cur.get("window_ms"):
            a, b = cur["window_ms"]
            txt = f"<b>{self.channel}</b>: окно {a:.0f}–{b:.0f} мс"
            mk = cur.get("markers_ms")
            if mk:
                txt += (f", маркеры onset {mk['onset']:.1f} / P1 {mk['p1']:.1f}"
                        f" / P2 {mk['p2']:.1f} мс")
        else:
            txt = "правок нет"
        others = [c for c in self.overrides if c != self.channel]
        if others:
            txt += f" · ещё правок: {', '.join(sorted(others))}"
        txt += "<br>Правки применяются при пересчёте."
        self.ov_label.setText(txt)
        self.btn_reject.setText("Вернуть ответ" if cur.get("reject") else "Ответа нет")

    def _fill_stats(self, d, tags, amp, pers) -> None:
        n = d["N"].to_numpy()
        self.stats.setRowCount(len(tags))
        for r, tag in enumerate(tags):
            vals = [tag, str(int(n[r])), f"{amp[r]:.0f}", f"{pers[r]:.2f}"]
            for c, v in enumerate(vals):
                self.stats.setItem(r, c, QTableWidgetItem(v))

"""Interactive surface for the three Neurosoft protocols.

The pipeline writes finished PNGs for the record; this is the view you actually
work in, so it is driven by the data, not by those images:

  - pick a channel, and the response-vs-curve plot and the waveforms redraw;
  - click a point on that plot (or drag the slider) to pull up ONE curve — its
    waveform is drawn bold over the rest, with its onset/P1/P2 markers;
  - colour switches with the protocol: a recruitment sweep is coloured by curve
    order (later = darker), a Jendrassik / paired file by amplitude group, since
    what matters there is which group a repeat fell into, not when it happened;
  - amplitude groups carry their mean ± SD band and a live statistics table.

Everything is recomputed from the per-curve metrics and the saved epochs, so it
stays in step with re-runs and with channel rejections made in Crop review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.jendrassik import CURVE_GROUPS
from src.neurosoft import intensities_from_name
from src.plotting import SWEEP_CMAP
from src.recruitment import choose_n_groups, cluster_amplitudes
from matplotlib.widgets import SpanSelector

from src.sir_review import (FALSE, MARKERS, MISSED, WINDOW, curve_mark, load_overrides,
                            mark_curve, recompute_corrections, save_overrides)

from ..templates import build_template
from .template_editor import TemplateEditor

from ..results import SIRResults

#: Folder the pipeline wrote -> (tab label, default colouring, number of groups).
#: Only Jendrassik fixes the count, and from the protocol: a block of test stimuli
#: and a block with the manoeuvre, per intensity. Elsewhere None means each
#: channel's own amplitudes decide — nothing about a recruitment ramp or a
#: paired-pulse run says how many response levels it should have.
SCENARIOS = {
    "Recruitment": ("Recruitment curve", "curve order", None),
    "Jendrassik": ("Jendrassik manoeuvre", "amplitude group", CURVE_GROUPS),
    "Paired stimulation": ("Paired stimulation", "amplitude group", None),
}

#: Group colours, matching src/jendrassik.py so the GUI and the PNGs agree.
GROUP_COLORS = ["#4575b4", "#f0a202", "#d73027", "#4d9221", "#7b3294"]


def _assign_groups(amps: pd.Series, n_groups: int | None) -> pd.Series:
    """Group one channel's curves by response amplitude (low..high).

    Uses the pipeline's own clustering so the on-screen grouping is exactly the
    one in the exported tables — equal-sized quantile bins would cut through the
    two protocol blocks instead of finding them.

    ``n_groups=None`` means the channel's own amplitudes decide, as they do for
    recruitment and paired stimulation.
    """
    # A channel the detector found nothing on has no amplitudes to group — it is
    # still shown (its curves plotted), just with no groups.
    if amps.dropna().empty:
        return pd.Series([None] * len(amps), index=amps.index, dtype=object)
    if n_groups is None:
        n_groups = choose_n_groups(amps.to_numpy(float))
    labels = [f"G{i + 1}" for i in range(n_groups)]
    idx = cluster_amplitudes(amps.to_numpy(float), n_groups)
    return pd.Series([labels[i] if i >= 0 else None for i in idx],
                     index=amps.index, dtype=object)


class ScenarioViewer(QWidget):
    """Response-vs-curve plot + the curves themselves, for one Neurosoft run."""

    #: Emitted once corrections have been applied and the tables on disk rewritten,
    #: so the other surfaces onto the same run stop showing the previous numbers.
    results_changed = Signal()

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.results: SIRResults | None = None
        self.scenario: str | None = None
        self.by_curve: pd.DataFrame | None = None
        self.times: np.ndarray | None = None
        self.waves: dict[str, np.ndarray] = {}
        self.channel: str | None = None
        self.n_groups: int | None = None
        self.root: Path | None = None
        self.overrides: dict = {}
        self._span: tuple[float, float] | None = None
        self._selector = None
        self.selected_curve: int | None = None
        self._points = None          # curve numbers behind the top plot, for click hit-testing

        # ---- left: channels + controls ----
        self.header = QLabel("<b>No Neurosoft scenario in this run</b>")
        self.header.setWordWrap(True)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.channel_list.currentTextChanged.connect(self._on_channel)

        self.color_box = QComboBox()
        self.color_box.addItems(["colour by curve order", "colour by amplitude group"])
        self.color_box.currentIndexChanged.connect(lambda _: self._draw())

        self.show_all = QCheckBox("show all curves")
        self.show_all.setChecked(True)
        self.show_all.toggled.connect(lambda _: self._draw())

        self.show_band = QCheckBox("group mean ± SD")
        self.show_band.setChecked(True)
        self.show_band.toggled.connect(lambda _: self._draw())

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.curve_label = QLabel("curve: —")

        # ---- per-curve corrections ----
        # Detection here is per curve, so its mistakes are per curve: one sweep
        # picks up something spurious, another loses a response its neighbours
        # kept. Neither is worth moving a global threshold for.
        self.btn_false = QPushButton("Ложный ответ")
        self.btn_false.setToolTip("Убрать детекцию с выбранной кривой.")
        self.btn_false.clicked.connect(lambda: self._mark(FALSE))
        self.btn_missed = QPushButton("Потерянный ответ")
        self.btn_missed.setToolTip(
            "Досчитать ответ на выбранной кривой по латентностям,\n"
            "на которых сходятся принятые кривые этого канала."
        )
        self.btn_missed.clicked.connect(lambda: self._mark(MISSED))
        self.btn_unmark = QPushButton("Снять пометку")
        self.btn_unmark.clicked.connect(lambda: self._mark(None))
        self.btn_window = QPushButton("Задать окно и маркеры")
        self.btn_window.setEnabled(False)
        self.btn_window.setToolTip(
            "Протяните мышью по нижнему графику над ответом, затем нажмите.\n"
            "Откроется редактор, где onset, P1 и P2 ставятся кликами.\n"
            "Канал перемеряется по ним целиком; в общий банк это не идёт."
        )
        self.btn_window.clicked.connect(self._set_window)
        self.btn_apply = QPushButton("Пересчитать по правкам")
        self.btn_apply.setToolTip("Отметьте всё нужное, затем нажмите один раз.")
        self.btn_apply.clicked.connect(self._recompute)
        self.mark_label = QLabel("пометок нет")
        self.mark_label.setStyleSheet("color: gray; font-size: 10px;")
        self.mark_label.setWordWrap(True)

        self.stats = QTableWidget(0, 5)
        self.stats.setHorizontalHeaderLabels(["Group", "N", "mean µV", "SD", "median"])
        self.stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats.setMaximumHeight(150)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(self.header)
        lv.addWidget(QLabel("<b>Channels</b>"))
        lv.addWidget(self.channel_list, 1)
        lv.addWidget(self.color_box)
        lv.addWidget(self.show_all)
        lv.addWidget(self.show_band)
        lv.addWidget(self.curve_label)
        lv.addWidget(self.slider)
        lv.addWidget(QLabel("<b>Правка выбранной кривой</b>"))
        for b in (self.btn_false, self.btn_missed, self.btn_unmark,
                  self.btn_window, self.btn_apply):
            lv.addWidget(b)
        lv.addWidget(self.mark_label)
        lv.addWidget(QLabel("<b>Amplitude groups</b>"))
        lv.addWidget(self.stats)

        # ---- right: the two plots ----
        self.fig = Figure(figsize=(7, 7), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.hint = QLabel(
            "Click a point on the top plot to pull that curve up below, or drag the slider."
        )
        self.hint.setStyleSheet("color: gray; font-size: 10px;")

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(2, 2, 2, 2)
        rv.addWidget(self.canvas, 1)
        rv.addWidget(self.hint)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([260, 900])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, root: Path) -> None:
        self.root = Path(root)
        self.overrides = load_overrides(self.root)
        self.results = SIRResults(Path(root))
        self.scenario = self.results.scenario
        self.by_curve = None
        self.times, self.waves = None, {}
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_list.blockSignals(False)

        if self.scenario is None:
            self.header.setText(
                "<b>No Neurosoft scenario in this run</b><br>"
                "<span style='color:gray'>This surface covers the recruitment, "
                "Jendrassik and paired-stimulation exports.</span>"
            )
            self._blank("This run produced no Neurosoft scenario outputs.")
            return

        label, default_colour, self.n_groups = SCENARIOS.get(
            self.scenario, (self.scenario, "curve order", None))
        crops = self.results.crops
        if not crops:
            self.header.setText(f"<b>{label}</b>")
            self._blank("No epochs were saved for this run.")
            return

        crop = crops[0]                      # a curve export is always a single crop
        # Group count follows the intensities named in the file, exactly as the
        # pipeline does it, so the on-screen groups are the exported ones.
        if self.scenario == "Jendrassik":
            intensities = intensities_from_name(crop.config)
            self.n_groups = 2 * max(len(intensities), 1)
        self.by_curve = self.results.recruitment_by_curve(crop.config, self.session)
        try:
            epochs = self.results.load_epochs(crop)
            self.times = epochs.times
            data = epochs.get_data() * 1e6   # -> µV
            self.waves = {ch: data[:, i, :] for i, ch in enumerate(epochs.ch_names)}
        except Exception as exc:
            self._blank(f"Cannot read epochs:\n{exc}")
            return

        n_curves = len(next(iter(self.waves.values()))) if self.waves else 0
        self.header.setText(
            f"<b>{label}</b><br><span style='color:gray'>{crop.config}<br>"
            f"{n_curves} curves · {len(self.waves)} channels</span>"
        )
        self.color_box.blockSignals(True)
        self.color_box.setCurrentIndex(0 if default_colour == "curve order" else 1)
        self.color_box.blockSignals(False)

        # Show EVERY channel, not just the responders: a channel the detector
        # found nothing on still needs looking at (to place markers by hand, or
        # to confirm it really is silent). Non-responders are greyed.
        responders = set(self._responders())
        all_chans = sorted(self.waves, key=lambda s: (len(s), s))
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        for ch in all_chans:
            item = QListWidgetItem(ch)
            if ch not in responders:
                item.setForeground(Qt.gray)
                item.setToolTip("нет детекций")
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)
        if self.channel_list.count():
            self.channel_list.setCurrentRow(0)   # fires _on_channel -> _draw
        else:
            self._blank("No channels.")

    def _responders(self) -> list[str]:
        """Channels with at least one detected response, in natural order."""
        if self.by_curve is None or self.by_curve.empty:
            return []
        # A channel counts as responding if ANY curve carried a detection — the
        # frame now holds a row per curve, undetected ones included.
        got = self.by_curve.groupby("Channel")["amp_uv"].apply(lambda s: s.notna().any())
        chans = [c for c, ok in got.items() if ok]
        return sorted((c for c in chans if c in self.waves), key=lambda s: (len(s), s))

    # ------------------------------------------------------------------ #
    def _on_channel(self, ch: str) -> None:
        self.channel = ch or None
        self.selected_curve = None
        n = len(self.waves.get(ch, []))
        self.slider.blockSignals(True)
        self.slider.setEnabled(n > 0)
        self.slider.setRange(1, max(n, 1))
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self._draw()

    def _on_slider(self, value: int) -> None:
        self.selected_curve = int(value)
        self._draw()

    def _on_click(self, event) -> None:
        """Select the curve nearest the click on the top plot."""
        if event.inaxes is None or self._points is None or event.xdata is None:
            return
        if event.inaxes is not self._ax_top:
            return
        curves = self._points
        if not len(curves):
            return
        nearest = int(curves[int(np.argmin(np.abs(np.asarray(curves) - event.xdata)))])
        self.selected_curve = nearest
        self.slider.blockSignals(True)
        self.slider.setValue(nearest)
        self.slider.blockSignals(False)
        self._draw()

    # ------------------------------------------------------------------ #
    def _blank(self, text: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", wrap=True)
        ax.set_axis_off()
        self._points = None
        self._ax_top = None
        self.stats.setRowCount(0)
        self.canvas.draw_idle()

    def _provisional_amplitudes(self, g: pd.DataFrame, waves: np.ndarray) -> np.ndarray:
        """What each undetected curve reads where its neighbours' response sits.

        Undetected curves have no point to click and no place on the plot, which
        makes the one thing they are there for — deciding whether a response was
        missed or genuinely absent — impossible to do by eye. Each is measured at
        the P1 latency of the NEAREST curve on this channel that did get one, so
        a missed response shows up at roughly its neighbours' height while a
        truly silent curve sits near zero.

        Returns NaN for curves that already have an amplitude, and for every
        curve when the channel has no detection at all to borrow a latency from.
        These values are for the plot only; the tables keep the NaN.
        """
        curves = g["curve"].to_numpy(int)
        amps = g["amp_uv"].to_numpy(float)
        p1 = g["p1_ms"].to_numpy(float)
        out = np.full(len(curves), np.nan)
        ref = np.flatnonzero(np.isfinite(amps) & np.isfinite(p1))
        if not ref.size or self.times is None:
            return out
        t_ms = np.asarray(self.times) * 1e3
        for j in np.flatnonzero(~np.isfinite(amps)):
            if not (0 < curves[j] <= len(waves)):
                continue
            nearest = ref[int(np.argmin(np.abs(curves[ref] - curves[j])))]
            i = int(np.argmin(np.abs(t_ms - p1[nearest])))
            out[j] = abs(float(waves[curves[j] - 1][i]))
        return out

    def _groups_for(self, ch: str) -> pd.DataFrame:
        g = self.by_curve[self.by_curve["Channel"] == ch].sort_values("curve").copy()
        g["group"] = _assign_groups(g["amp_uv"], self.n_groups)
        return g

    def _draw(self) -> None:
        if not self.channel or self.by_curve is None or self.times is None:
            return
        ch = self.channel
        if ch not in self.waves:
            self._blank(f"{ch} has no saved waveforms.")
            return

        g = self._groups_for(ch)
        by_group = self.color_box.currentIndex() == 1
        # A channel keeps only the groups its own amplitudes produced.
        labels = sorted(g["group"].dropna().unique(), key=lambda s: int(str(s)[1:]))
        waves = self.waves[ch]
        t_ms = np.asarray(self.times) * 1e3

        self.fig.clear()
        ax_top, ax_bot = self.fig.subplots(2, 1, height_ratios=[1, 1.4])
        self._ax_top = ax_top

        # ---- top: response size against curve number ----
        curves = g["curve"].to_numpy(int)
        amps = g["amp_uv"].to_numpy(float)
        self._points = curves
        got = np.isfinite(amps)
        # What the curves with no detection would read, measured where the
        # nearest detected curve has its P1. Drawn hollow and never written to
        # the tables, which keep the NaN: this is something to aim at and to
        # judge by eye, not a measurement.
        shown = self._provisional_amplitudes(g, waves)
        # The line is drawn over the full curve range; NaNs break it, so curves
        # with no detection show up as gaps instead of shifting the axis.
        if by_group:
            for gi, lab in enumerate(labels):
                m = ((g["group"] == lab).to_numpy()) & got
                if m.any():
                    ax_top.plot(curves[m], amps[m], "o", ms=5, color=GROUP_COLORS[gi],
                                label=f"{lab} (n={int(m.sum())})")
            ax_top.plot(curves, amps, "-", lw=0.8, color="0.7", zorder=0)
            ax_top.legend(fontsize=7, frameon=False, ncol=len(labels))
        else:
            norm = mcolors.Normalize(vmin=1, vmax=max(int(curves.max()), 2))
            ax_top.plot(curves, amps, "-", lw=0.9, color="0.75", zorder=0)
            ax_top.scatter(curves[got], amps[got], s=26,
                           c=[SWEEP_CMAP(norm(c)) for c in curves[got]], zorder=3)
        miss = (~got) & np.isfinite(shown)
        if miss.any():
            ax_top.plot(curves[miss], shown[miss], "o", ls="none", ms=6,
                        mfc="none", mec="0.45", mew=1.1, zorder=2,
                        label="без детекции")
        n_missing = int((~got).sum())
        if len(curves):
            ax_top.set_xlim(0.5, float(curves.max()) + 0.5)
        # Marked curves: a cross where a detection was called false, a hollow
        # ring where one was called missed — both visible before recomputing.
        per_ch = self.overrides.get(ch, {})
        for cno in per_ch.get(FALSE, []):
            if cno in set(curves.tolist()):
                j = list(curves).index(cno)
                if got[j]:
                    ax_top.plot([cno], [amps[j]], "x", ms=9, mew=2, color="#d62728", zorder=6)
        for cno in per_ch.get(MISSED, []):
            if cno in set(curves.tolist()):
                j = list(curves).index(cno)
                y = amps[j] if got[j] else shown[j]
                if np.isfinite(y):
                    ax_top.plot([cno], [y], "o", ms=10, mfc="none", mec="#2ca02c",
                                mew=2, zorder=6)
        if self.selected_curve is not None and self.selected_curve in set(curves.tolist()):
            j = list(curves).index(self.selected_curve)
            ax_top.axvline(self.selected_curve, color="#ffb703", lw=1.2, zorder=1)
            y = amps[j] if got[j] else shown[j]
            if np.isfinite(y):
                ax_top.plot([self.selected_curve], [y], "o", ms=11, mfc="none",
                            mec="#ff8800", mew=2.0, zorder=4)
        ax_top.set_xlabel(
            "curve number"
            + (f"   ({n_missing} of {len(curves)} without a detection)" if n_missing else "")
        )
        ax_top.set_ylabel("PTP / |P1|, µV")
        ax_top.set_title(f"{ch} — response vs curve", fontsize=10, loc="left")
        ax_top.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax_top.spines[side].set_visible(False)

        # ---- bottom: the curves themselves ----
        post = np.asarray(self.times) >= 0.002     # scale past the stimulus artifact
        group_of = dict(zip(g["curve"], g["group"]))
        # Curves whose response was not found are still drawn, in pale grey. The
        # colour scale says where a curve sits in the run, not whether anything
        # was measured on it, so without this a missed response is invisible —
        # and telling "no response here" from "the detector lost it" is the whole
        # point of looking at the curves.
        found = set(g.loc[g["amp_uv"].notna(), "curve"].astype(int))
        if self.show_all.isChecked():
            norm = mcolors.Normalize(vmin=1, vmax=max(len(waves), 2))
            for i in range(len(waves)):
                cno = i + 1
                missing = cno not in found
                if by_group:
                    lab = group_of.get(cno)
                    if lab is None:
                        continue
                    col = GROUP_COLORS[labels.index(lab)] if lab in labels else "0.6"
                else:
                    col = "0.78" if missing else SWEEP_CMAP(norm(cno))
                ax_bot.plot(t_ms, waves[i], color=col, lw=0.5,
                            alpha=0.30 if missing else 0.35,
                            zorder=0 if missing else 1)

        if by_group and self.show_band.isChecked():
            for gi, lab in enumerate(labels):
                rows = [c - 1 for c, l in group_of.items() if l == lab and 0 < c <= len(waves)]
                if not rows:
                    continue
                block = waves[sorted(rows)]
                m = block.mean(axis=0)
                sd = block.std(axis=0, ddof=1) if len(block) > 1 else np.zeros_like(m)
                ax_bot.fill_between(t_ms, m - sd, m + sd, color=GROUP_COLORS[gi],
                                    alpha=0.18, lw=0, zorder=2)
                ax_bot.plot(t_ms, m, color=GROUP_COLORS[gi], lw=1.8, zorder=3,
                            label=f"{lab} mean")
            if labels:   # a silent channel has no groups, hence no legend
                ax_bot.legend(fontsize=7, frameon=False, ncol=len(labels))

        if self.selected_curve is not None and 0 < self.selected_curve <= len(waves):
            sel = waves[self.selected_curve - 1]
            ax_bot.plot(t_ms, sel, color="#111111", lw=1.8, zorder=5,
                        label=f"curve {self.selected_curve}")
            row = g[g["curve"] == self.selected_curve]
            if not row.empty:
                # all three markers on the selected curve: onset (blue), P1 (red),
                # P2 (green) — sat on the waveform at each marker's latency.
                for col_name, colour in (("onset_ms", "tab:blue"),
                                         ("p1_ms", "red"),
                                         ("p2_ms", "tab:green")):
                    if col_name not in row.columns:
                        continue
                    val = row[col_name].iloc[0]
                    if not np.isfinite(val):
                        continue
                    idx = int(np.argmin(np.abs(t_ms - float(val))))
                    ax_bot.plot([float(val)], [sel[idx]], "o", color=colour, ms=7, zorder=6)
            if not row.empty and np.isfinite(row["amp_uv"].iloc[0]):
                self.curve_label.setText(
                    f"curve: {self.selected_curve} · {row['amp_uv'].iloc[0]:.1f} µV")
            else:
                self.curve_label.setText(
                    f"curve: {self.selected_curve} · ответ не найден")
        else:
            self.curve_label.setText("curve: —")
        self._refresh_mark_label()

        seg = waves[:, post]
        if seg.size and np.isfinite(seg).any():
            lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            ax_bot.set_ylim(lo - pad, hi + pad)
        cur_win = self.overrides.get(ch, {}).get(WINDOW)
        if cur_win:
            ax_bot.axvspan(cur_win[0], cur_win[1], color="#ffb703", alpha=0.18, lw=0, zorder=0)
        elif self._span is not None:
            ax_bot.axvspan(self._span[0], self._span[1], color="#ffb703", alpha=0.12, lw=0, zorder=0)
        mk = self.overrides.get(ch, {}).get(MARKERS)
        if mk:
            for key, col in (("onset", "tab:blue"), ("p1", "tab:red"), ("p2", "tab:green")):
                v = mk.get(key)
                if v is not None and np.isfinite(v):
                    ax_bot.axvline(float(v), color=col, lw=1.1, ls="--", alpha=0.85, zorder=6)
        ax_bot.axhline(0, color="0.8", lw=0.6)
        ax_bot.set_xlabel("ms from stimulus")
        ax_bot.set_ylabel("µV")
        ax_bot.set_title(
            f"{ch} — curves, "
            + ("coloured by amplitude group" if by_group else "coloured by order (later = darker)"),
            fontsize=10, loc="left",
        )
        ax_bot.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax_bot.spines[side].set_visible(False)

        self._selector = SpanSelector(
            ax_bot, self._on_span, "horizontal", useblit=False,
            props=dict(alpha=0.25, facecolor="#ffb703"),
        )
        self.canvas.draw_idle()
        self._fill_stats(g, labels)

    # ---- per-curve corrections --------------------------------------- #
    def _on_span(self, lo: float, hi: float) -> None:
        if hi - lo < 0.5:
            return
        self._span = (float(lo), float(hi))
        self.btn_window.setEnabled(True)
        self.btn_window.setText(f"Задать окно {lo:.0f}–{hi:.0f} мс")
        self._draw()

    def _set_window(self) -> None:
        """Window -> the shared marker editor -> a per-channel correction."""
        if not self.channel or self._span is None or self.root is None:
            return
        lo, hi = self._span
        waves = self.waves.get(self.channel)
        if waves is None or self.times is None:
            return
        t_s = np.asarray(self.times)
        win = (self.times * 1e3 >= lo) & (self.times * 1e3 <= hi)
        strongest = int(np.argmax(np.ptp(np.nan_to_num(waves)[:, win], axis=1)))
        try:
            tpl = build_template(t_s, np.nan_to_num(waves[strongest]), lo / 1e3, hi / 1e3,
                                 source=f"{self.channel} · кривая {strongest + 1} · "
                                        f"{lo:.0f}–{hi:.0f} мс")
        except Exception as exc:
            self.mark_label.setText(f"<span style='color:#b00'>Не удалось построить шаблон: {exc}</span>")
            return
        dlg = TemplateEditor(tpl, self)
        if not dlg.exec():
            return
        t = dlg.edited_template()
        ch = self.overrides.setdefault(self.channel, {})
        ch[WINDOW] = [round(lo, 2), round(hi, 2)]
        ch[MARKERS] = {
            "onset": round(float(t.times[int(t.onset_idx)]) * 1e3, 2),
            "p1": round(float(t.times[int(t.peak1_idx)]) * 1e3, 2),
            "p2": round(float(t.times[int(t.peak2_idx)]) * 1e3, 2),
        }
        save_overrides(self.root, self.overrides)
        self._refresh_mark_label()
        self._draw()

    def _mark(self, kind) -> None:
        if not self.channel or self.selected_curve is None or self.root is None:
            self.mark_label.setText("Сначала выберите кривую — кликом по графику или слайдером.")
            return
        self.overrides = mark_curve(self.overrides, self.channel,
                                    int(self.selected_curve), kind)
        save_overrides(self.root, self.overrides)
        self._refresh_mark_label()
        self._draw()

    def _refresh_mark_label(self) -> None:
        cur = curve_mark(self.overrides, self.channel or "", int(self.selected_curve or -1))
        word = {FALSE: "ложный", MISSED: "потерянный"}.get(cur, "нет пометки")
        total = sum(len(v.get(FALSE, [])) + len(v.get(MISSED, []))
                    for v in self.overrides.values())
        per_ch = self.overrides.get(self.channel or "", {})
        bits = []
        if per_ch.get(FALSE):
            bits.append(f"ложные: {', '.join(map(str, per_ch[FALSE]))}")
        if per_ch.get(MISSED):
            bits.append(f"потерянные: {', '.join(map(str, per_ch[MISSED]))}")
        txt = f"кривая {self.selected_curve}: <b>{word}</b>"
        mk = per_ch.get(MARKERS)
        if mk:
            bits.append(f"маркеры onset {mk['onset']:.1f} / P1 {mk['p1']:.1f} / P2 {mk['p2']:.1f} мс")
        if bits:
            txt += "<br>" + f"{self.channel}: " + "; ".join(bits)
        if total:
            txt += f"<br>всего пометок в файле: {total}"
        self.mark_label.setText(txt)

    def _recompute(self) -> None:
        if self.root is None:
            return
        ch, cur, colour = self.channel, self.selected_curve, self.color_box.currentIndex()
        try:
            done = recompute_corrections(self.root, self.scenario_key())
        except Exception as exc:
            self.mark_label.setText(f"<span style='color:#b00'>Не удалось пересчитать: {exc}</span>")
            return
        if not done:
            self.mark_label.setText("Пометок нет — пересчитывать нечего.")
            return
        self.load(self.root)
        # Put the view back exactly where it was. Correcting is iterative — mark,
        # recompute, look, mark again — and having to find the channel and the
        # curve again after every pass is what makes it unusable.
        rows = [self.channel_list.item(i).text() for i in range(self.channel_list.count())]
        if ch in rows:
            self.channel_list.setCurrentRow(rows.index(ch))
        elif ch:
            # The channel can leave the responder list once its detections are
            # taken away. Say so rather than silently landing on another one.
            self.mark_label.setText(
                f"<span style='color:#b00'>{ch} больше не проходит как отвечающий "
                "канал — все его ответы сняты.</span>")
        self.color_box.blockSignals(True)
        self.color_box.setCurrentIndex(colour)
        self.color_box.blockSignals(False)
        if cur is not None:
            self.selected_curve = cur
        self._draw()
        self._refresh_mark_label()
        self.mark_label.setText(self.mark_label.text() + "<br>Пересчитано: " + "; ".join(
            f"{d['channel']}/{d['curve']} — {d['kind']}" for d in done))
        # Crop review shows the same run from its own copy of the tables; left
        # alone it keeps drawing the numbers from before the correction.
        self.results_changed.emit()

    def scenario_key(self) -> str | None:
        from src.neurosoft import JENDRASSIK, PAIRED, RECRUITMENT
        return {"Recruitment": RECRUITMENT, "Jendrassik": JENDRASSIK,
                "Paired stimulation": PAIRED}.get(self.scenario or "")

    def _fill_stats(self, g: pd.DataFrame, labels: list[str]) -> None:
        rows = []
        for lab in labels:
            d = g.loc[g["group"] == lab, "amp_uv"].dropna()
            if d.empty:
                continue
            rows.append((lab, len(d), d.mean(),
                         d.std(ddof=1) if len(d) > 1 else 0.0, d.median()))
        self.stats.setRowCount(len(rows))
        for r, (lab, n, mean, sd, med) in enumerate(rows):
            for c, v in enumerate([lab, str(n), f"{mean:.1f}", f"{sd:.1f}", f"{med:.1f}"]):
                self.stats.setItem(r, c, QTableWidgetItem(v))

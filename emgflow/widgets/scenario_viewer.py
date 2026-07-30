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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from src.jendrassik import CURVE_GROUPS
from src.neurosoft import intensities_from_name
from src.plotting import SWEEP_CMAP
from src.recruitment import RECRUITMENT_N_GROUPS, cluster_amplitudes

from ..results import SIRResults

#: Folder the pipeline wrote -> (tab label, default colouring, number of groups).
#: Two groups on the Jendrassik/paired files: those runs are a block of test
#: stimuli and a block with the manoeuvre at the same intensity, which is the
#: split the clinician reads off them.
SCENARIOS = {
    "Recruitment": ("Recruitment curve", "curve order", RECRUITMENT_N_GROUPS),
    "Jendrassik": ("Jendrassik manoeuvre", "amplitude group", CURVE_GROUPS),
    "Paired stimulation": ("Paired stimulation", "amplitude group", CURVE_GROUPS),
}

#: Group colours, matching src/jendrassik.py so the GUI and the PNGs agree.
GROUP_COLORS = ["#4575b4", "#f0a202", "#d73027", "#4d9221", "#7b3294"]


def _assign_groups(amps: pd.Series, n_groups: int) -> pd.Series:
    """Group one channel's curves by response amplitude (low..high).

    Uses the pipeline's own clustering so the on-screen grouping is exactly the
    one in the exported tables — equal-sized quantile bins would cut through the
    two protocol blocks instead of finding them.
    """
    labels = [f"G{i + 1}" for i in range(n_groups)]
    idx = cluster_amplitudes(amps.to_numpy(float), n_groups)
    return pd.Series([labels[i] if i >= 0 else None for i in idx],
                     index=amps.index, dtype=object)


class ScenarioViewer(QWidget):
    """Response-vs-curve plot + the curves themselves, for one Neurosoft run."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.session = session
        self.results: SIRResults | None = None
        self.scenario: str | None = None
        self.by_curve: pd.DataFrame | None = None
        self.times: np.ndarray | None = None
        self.waves: dict[str, np.ndarray] = {}
        self.channel: str | None = None
        self.n_groups: int = RECRUITMENT_N_GROUPS
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
            self.scenario, (self.scenario, "curve order", RECRUITMENT_N_GROUPS))
        crops = self.results.crops
        if not crops:
            self.header.setText(f"<b>{label}</b>")
            self._blank("No epochs were saved for this run.")
            return

        crop = crops[0]                      # a curve export is always a single crop
        # Group count follows the intensities named in the file, exactly as the
        # pipeline does it, so the on-screen groups are the exported ones.
        if self.scenario != "Recruitment":
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

        responders = self._responders()
        self.channel_list.blockSignals(True)
        self.channel_list.addItems(responders or sorted(self.waves))
        self.channel_list.blockSignals(False)
        if self.channel_list.count():
            self.channel_list.setCurrentRow(0)   # fires _on_channel -> _draw
        else:
            self._blank("No responding channels.")

    def _responders(self) -> list[str]:
        """Channels with at least one detected response, in natural order."""
        if self.by_curve is None or self.by_curve.empty:
            return []
        chans = self.by_curve["Channel"].unique()
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
        labels = [f"G{i + 1}" for i in range(self.n_groups)]
        waves = self.waves[ch]
        t_ms = np.asarray(self.times) * 1e3

        self.fig.clear()
        ax_top, ax_bot = self.fig.subplots(2, 1, height_ratios=[1, 1.4])
        self._ax_top = ax_top

        # ---- top: response size against curve number ----
        curves = g["curve"].to_numpy(int)
        amps = g["amp_uv"].to_numpy(float)
        self._points = curves
        if by_group:
            for gi, lab in enumerate(labels):
                m = (g["group"] == lab).to_numpy()
                if m.any():
                    ax_top.plot(curves[m], amps[m], "o", ms=5, color=GROUP_COLORS[gi],
                                label=f"{lab} (n={int(m.sum())})")
            ax_top.plot(curves, amps, "-", lw=0.8, color="0.7", zorder=0)
            ax_top.legend(fontsize=7, frameon=False, ncol=len(labels))
        else:
            norm = mcolors.Normalize(vmin=1, vmax=max(int(curves.max()), 2))
            ax_top.plot(curves, amps, "-", lw=0.9, color="0.75", zorder=0)
            ax_top.scatter(curves, amps, s=26, c=[SWEEP_CMAP(norm(c)) for c in curves], zorder=3)
        if self.selected_curve is not None and self.selected_curve in set(curves.tolist()):
            y = float(amps[list(curves).index(self.selected_curve)])
            ax_top.axvline(self.selected_curve, color="#ffb703", lw=1.2, zorder=1)
            ax_top.plot([self.selected_curve], [y], "o", ms=11, mfc="none",
                        mec="#ff8800", mew=2.0, zorder=4)
        ax_top.set_xlabel("curve number")
        ax_top.set_ylabel("response (µV)")
        ax_top.set_title(f"{ch} — response vs curve", fontsize=10, loc="left")
        ax_top.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax_top.spines[side].set_visible(False)

        # ---- bottom: the curves themselves ----
        post = np.asarray(self.times) >= 0.002     # scale past the stimulus artifact
        group_of = dict(zip(g["curve"], g["group"]))
        if self.show_all.isChecked():
            norm = mcolors.Normalize(vmin=1, vmax=max(len(waves), 2))
            for i in range(len(waves)):
                cno = i + 1
                if by_group:
                    lab = group_of.get(cno)
                    if lab is None:
                        continue
                    col = GROUP_COLORS[labels.index(lab)] if lab in labels else "0.6"
                else:
                    col = SWEEP_CMAP(norm(cno))
                ax_bot.plot(t_ms, waves[i], color=col, lw=0.5, alpha=0.35, zorder=1)

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
            ax_bot.legend(fontsize=7, frameon=False, ncol=len(labels))

        if self.selected_curve is not None and 0 < self.selected_curve <= len(waves):
            sel = waves[self.selected_curve - 1]
            ax_bot.plot(t_ms, sel, color="#111111", lw=1.8, zorder=5,
                        label=f"curve {self.selected_curve}")
            row = g[g["curve"] == self.selected_curve]
            if not row.empty and np.isfinite(row["p1_ms"].iloc[0]):
                p1 = float(row["p1_ms"].iloc[0])
                idx = int(np.argmin(np.abs(t_ms - p1)))
                ax_bot.plot([p1], [sel[idx]], "o", color="red", ms=7, zorder=6)
            self.curve_label.setText(
                f"curve: {self.selected_curve}"
                + (f" · {row['amp_uv'].iloc[0]:.1f} µV" if not row.empty else "")
            )
        else:
            self.curve_label.setText("curve: —")

        seg = waves[:, post]
        if seg.size and np.isfinite(seg).any():
            lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            ax_bot.set_ylim(lo - pad, hi + pad)
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

        self.canvas.draw_idle()
        self._fill_stats(g, labels)

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

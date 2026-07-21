"""Spontaneous-EMG review: RMS envelopes with detected bursts, and antagonist co-activation.

A separate review surface, not a variant of the crop viewer: here the object of interest is
a burst on the envelope, not an evoked peak. The burst thresholds — above all the absolute
floor in µV, which is what separates an active muscle from a noisy low-amplitude channel —
are the knobs you drag while watching the shading update.

Two sub-tabs share one condition selector: "Envelopes" (per-channel RMS envelope with its
bursts) and "Antagonist co-activation" (the L-shape plots, one point per merged burst
episode of an antagonist pair — click a point to see that episode's envelope chunk).
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..results import SpontaneousResults
from .coactivation_viewer import CoactivationViewer


class SpontaneousViewer(QWidget):
    rerun_requested = Signal()

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: SpontaneousResults | None = None
        self.condition = ""

        self.cond_box = QComboBox()
        self.cond_box.currentTextChanged.connect(self._on_condition)

        self.env_list = QListWidget()
        self.env_list.currentRowChanged.connect(lambda _: self._draw())

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Exported envelopes</b>"))
        lv.addWidget(self.env_list, 1)
        self.summary_btn = QPushButton("Per-channel summary")
        self.summary_btn.clicked.connect(self._show_summary)
        lv.addWidget(self.summary_btn)

        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)

        bar = QHBoxLayout()
        note = QLabel(
            "Bursts come from the pipeline. To change what counts as a burst, edit the "
            "Bursts group in Settings and re-run."
        )
        note.setStyleSheet("color: gray; font-size: 11px;")
        bar.addWidget(note)
        bar.addStretch()
        self.export_btn = QPushButton("Export this envelope")
        self.export_btn.setToolTip("Copy the burst envelope out for the stimulator (FES).")
        self.export_btn.clicked.connect(self._export)
        self.rerun_btn = QPushButton("Re-run with current burst settings")
        self.rerun_btn.clicked.connect(self.rerun_requested)
        bar.addWidget(self.export_btn)
        bar.addWidget(self.rerun_btn)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.canvas, 1)
        rv.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 900])
        envelopes_tab = QWidget()
        ev_layout = QVBoxLayout(envelopes_tab)
        ev_layout.setContentsMargins(0, 0, 0, 0)
        ev_layout.addWidget(split)

        self.coactivation_viewer = CoactivationViewer(session)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.addTab(envelopes_tab, "Envelopes")
        self.inner_tabs.addTab(self.coactivation_viewer, "Antagonist co-activation")

        cond_bar = QHBoxLayout()
        cond_bar.addWidget(QLabel("<b>Condition</b>"))
        cond_bar.addWidget(self.cond_box)
        cond_bar.addStretch()

        root = QVBoxLayout(self)
        root.addLayout(cond_bar)
        root.addWidget(self.inner_tabs, 1)

    # ------------------------------------------------------------------ #
    def load(self, results: SpontaneousResults) -> None:
        self.results = results
        self.coactivation_viewer.load(results)
        self.cond_box.blockSignals(True)
        self.cond_box.clear()
        self.cond_box.addItems(results.conditions())
        self.cond_box.blockSignals(False)
        conds = results.conditions()
        if conds:
            self._on_condition(conds[0])

    def _on_condition(self, condition: str) -> None:
        if not condition or self.results is None:
            return
        self.condition = condition
        self.env_list.clear()
        for f in self.results.envelope_files(condition):
            self.env_list.addItem(QListWidgetItem(f.stem))
        if self.env_list.count():
            self.env_list.setCurrentRow(0)
        else:
            self._draw()
        self.coactivation_viewer.set_condition(condition)

    def _current_file(self) -> Path | None:
        if self.results is None:
            return None
        row = self.env_list.currentRow()
        files = self.results.envelope_files(self.condition)
        if row < 0 or row >= len(files):
            return None
        return files[row]

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        path = self._current_file()
        if path is None or self.results is None:
            ax.text(0.5, 0.5, "No envelopes exported for this condition.\n"
                              "Either no channel had bursts, or the spontaneous analysis is off.",
                    ha="center", va="center", color="gray")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        times, rms = self.results.read_envelope(path)
        ax.plot(times, rms, color="#1f77b4", lw=1.2)
        ax.fill_between(times, 0, rms, color="#1f77b4", alpha=0.15)

        is_burst = "burst" in path.stem
        ax.set_xlabel("time from burst centre (s)" if is_burst else "time (s)")
        ax.set_ylabel("RMS (µV)")
        ax.set_title(path.stem, fontsize=10)
        if is_burst:
            ax.axvline(0.0, color="#d62728", lw=0.9, ls=":")

        bursts = self.results.bursts(self.condition)
        if not bursts.empty:
            chan = path.stem.split("_burst")[0].split("_fullenvelope")[0]
            rows = bursts[bursts["Channel"].astype(str) == chan]
            if not rows.empty:
                ax.text(
                    0.99, 0.97, f"{len(rows)} burst(s) on {chan}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
                )
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        src = self._current_file()
        if src is None:
            QMessageBox.information(self, "Nothing to export", "Select an envelope first.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export envelope", src.name, "Text (*.txt);;CSV (*.csv)")
        if not dest:
            return
        Path(dest).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        QMessageBox.information(self, "Exported", f"Saved:\n{dest}")

    def _show_summary(self) -> None:
        if self.results is None:
            return
        table = self.results.summary(self.condition)
        if table.empty:
            QMessageBox.information(self, "Summary", "No spontaneous-EMG summary for this condition.")
            return
        QMessageBox.information(self, "Per-channel summary", table.to_string(index=False, max_rows=40))

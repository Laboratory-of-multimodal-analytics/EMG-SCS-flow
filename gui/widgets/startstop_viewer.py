"""StartStop epoch browser — the clinicians' surface.

Walk the detections found by the pipeline, flag the ones you don't like, drag a window over
a pattern and describe it, and export the selected waveform to CSV for the stimulator.
Reads 'Detections raw/<cond>_detections_raw.fif' — the pipeline's own annotated recording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ..results import Detection, StartStopResults
from ..review_store import STATUS_LABEL, STATUSES, Annotation, ReviewStore

STATUS_COLOUR = {
    "ok": None,
    "bad": "#c8a0a0",
    "false_positive": "#d08a8a",
    "complex": "#c9b483",
}


class StartStopViewer(QWidget):
    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: StartStopResults | None = None
        self.store: ReviewStore | None = None
        self.detections: list[Detection] = []
        self.condition: str = ""
        self._span: tuple[float, float] | None = None
        self._selector: SpanSelector | None = None

        # ---- left ----
        self.cond_box = QComboBox()
        self.cond_box.currentTextChanged.connect(self._on_condition)

        self.det_list = QListWidget()
        self.det_list.currentRowChanged.connect(lambda _: self._draw())

        self.status_box = QComboBox()
        self.status_box.addItems([STATUS_LABEL[s] for s in STATUSES])
        self.status_box.currentIndexChanged.connect(self._on_status)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Condition</b>"))
        lv.addWidget(self.cond_box)
        lv.addWidget(QLabel("<b>Detections</b>"))
        lv.addWidget(self.det_list, 1)
        lv.addWidget(QLabel("Status of the selected epoch"))
        lv.addWidget(self.status_box)
        why = QPushButton("Why was nothing found?")
        why.setToolTip("The pipeline's own rejected-anchor reasons for this condition.")
        why.clicked.connect(self._show_why)
        lv.addWidget(why)

        # ---- right ----
        self.fig = Figure(figsize=(9, 7), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)

        bar = QHBoxLayout()
        self.window_label = QLabel("Drag over the plot to select a window.")
        self.window_label.setStyleSheet("color: gray;")
        bar.addWidget(self.window_label)
        bar.addStretch()
        self.annotate_btn = QPushButton("Annotate selection")
        self.annotate_btn.clicked.connect(self._annotate)
        self.export_btn = QPushButton("Export selection to CSV")
        self.export_btn.setToolTip("The response waveform to feed the stimulator (FES).")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.annotate_btn)
        bar.addWidget(self.export_btn)

        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(90)
        self.ann_list.itemDoubleClicked.connect(self._delete_annotation)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.canvas, 1)
        rv.addLayout(bar)
        rv.addWidget(QLabel("Annotations (double-click to delete)"))
        rv.addWidget(self.ann_list)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([280, 900])
        root = QVBoxLayout(self)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, results: StartStopResults, store: ReviewStore) -> None:
        self.results = results
        self.store = store
        self.cond_box.blockSignals(True)
        self.cond_box.clear()
        self.cond_box.addItems(results.conditions())
        self.cond_box.blockSignals(False)
        if results.conditions():
            self._on_condition(results.conditions()[0])

    def _on_condition(self, condition: str) -> None:
        if not condition or self.results is None:
            return
        self.condition = condition
        self.detections = self.results.detections(condition)
        self.det_list.clear()
        for det in self.detections:
            item = QListWidgetItem(det.label)
            status = self.store.review.status(condition, det.index) if self.store else "ok"
            colour = STATUS_COLOUR.get(status)
            if colour:
                item.setBackground(Qt.GlobalColor.transparent)
                item.setForeground(Qt.darkRed if status != "complex" else Qt.darkYellow)
                item.setText(f"{det.label}   [{STATUS_LABEL[status]}]")
            self.det_list.addItem(item)
        if self.detections:
            self.det_list.setCurrentRow(0)
        else:
            self._draw()

    def _current(self) -> Detection | None:
        row = self.det_list.currentRow()
        if row < 0 or row >= len(self.detections):
            return None
        return self.detections[row]

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self.fig.clear()
        self._span = None
        self.window_label.setText("Drag over the plot to select a window.")
        det = self._current()
        if det is None or self.results is None:
            self.canvas.draw_idle()
            return

        # Sync the status box without re-triggering it.
        status = self.store.review.status(self.condition, det.index) if self.store else "ok"
        self.status_box.blockSignals(True)
        self.status_box.setCurrentIndex(STATUSES.index(status))
        self.status_box.blockSignals(False)

        raw = self.results.raw(self.condition)
        sfreq = raw.info["sfreq"]
        # ±1 s of context around the detection, as in the pipeline's own 'Raw epochs' PNGs.
        pad = 1.0
        t0 = max(det.time - pad, raw.times[0])
        t1 = min(det.time + pad, raw.times[-1])
        s0, s1 = int(t0 * sfreq), int(t1 * sfreq)

        chans = [c for c in raw.ch_names if "art" not in c.lower()]
        data = raw.get_data(picks=chans, start=s0, stop=s1) * 1e6
        times = np.arange(s0, s1) / sfreq

        axes = self.fig.subplots(len(chans), 1, sharex=True, squeeze=False)[:, 0]
        for ax, ch, row in zip(axes, chans, range(len(chans))):
            responding = ch in det.channels
            ax.plot(times, data[row], lw=0.6,
                    color="#1f77b4" if responding else "0.75", zorder=2)
            ax.axvline(det.time, color="#d62728", lw=1.0, zorder=3)
            ax.set_ylabel(ch, rotation=0, ha="right", va="center", fontsize=8,
                          fontweight="bold" if responding else "normal")
            ax.tick_params(labelsize=7)

            for ann in (self.store.review.annotations_for(self.condition) if self.store else []):
                if ann.tmax >= t0 and ann.tmin <= t1:
                    ax.axvspan(ann.tmin, ann.tmax, color="#ffd27f", alpha=0.35, zorder=1)

        axes[-1].set_xlabel("time (s)")
        self.fig.suptitle(
            f"{self.condition} — detection {det.index + 1}/{len(self.detections)} "
            f"on {'+'.join(det.channels)}  (bold = responding muscle)", fontsize=10,
        )

        self._selector = SpanSelector(
            axes[-1], self._on_span, "horizontal", useblit=True,
            props=dict(alpha=0.3, facecolor="#ffb703"), interactive=True,
        )
        # An interactive SpanSelector seeds a zero-width span at x=0, which autoscales the
        # shared x-axis back to the origin and squashes the response into a corner. Pin the
        # limits to the window we actually drew.
        axes[-1].set_xlim(t0, t1)
        self.canvas.draw_idle()
        self._refresh_annotations()

    def _on_span(self, tmin: float, tmax: float) -> None:
        self._span = (float(tmin), float(tmax))
        self.window_label.setText(
            f"Selected {tmin:.3f} – {tmax:.3f} s  ({1000 * (tmax - tmin):.0f} ms)"
        )

    # ------------------------------------------------------------------ #
    def _on_status(self, idx: int) -> None:
        det = self._current()
        if det is None or self.store is None:
            return
        self.store.review.set_status(self.condition, det.index, STATUSES[idx])
        self.store.save()
        row = self.det_list.currentRow()
        self._on_condition(self.condition)
        self.det_list.setCurrentRow(row)

    def _annotate(self) -> None:
        det = self._current()
        if self.store is None or self._span is None:
            QMessageBox.information(self, "No selection", "Drag over the plot to select a window first.")
            return
        text, ok = QInputDialog.getText(self, "Annotate selection", "Description (letters only):")
        if not ok or not text.strip():
            return
        self.store.review.add_annotation(Annotation(
            condition=self.condition,
            detection_index=det.index if det else -1,
            tmin=self._span[0], tmax=self._span[1], text=text.strip(),
        ))
        self.store.save()
        self._draw()

    def _refresh_annotations(self) -> None:
        self.ann_list.clear()
        if self.store is None:
            return
        for ann in self.store.review.annotations_for(self.condition):
            self.ann_list.addItem(
                QListWidgetItem(f"{ann.tmin:.3f}–{ann.tmax:.3f} s   {ann.text}")
            )

    def _delete_annotation(self, item: QListWidgetItem) -> None:
        if self.store is None:
            return
        anns = self.store.review.annotations_for(self.condition)
        row = self.ann_list.row(item)
        if 0 <= row < len(anns):
            self.store.review.remove_annotation(anns[row])
            self.store.save()
            self._draw()

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        det = self._current()
        if self.results is None or self.store is None or self._span is None:
            QMessageBox.information(self, "No selection", "Drag over the plot to select a window first.")
            return
        chans = det.channels if det else []
        channel = chans[0] if chans else ""
        raw = self.results.raw(self.condition)
        available = [c for c in raw.ch_names if "art" not in c.lower()]
        channel, ok = QInputDialog.getItem(
            self, "Export to CSV", "Channel to export:", available,
            available.index(channel) if channel in available else 0, False,
        )
        if not ok:
            return

        sfreq = raw.info["sfreq"]
        s0, s1 = int(self._span[0] * sfreq), int(self._span[1] * sfreq)
        values = raw.get_data(picks=[channel], start=s0, stop=s1)[0] * 1e6
        times = np.arange(s0, s1) / sfreq

        default = Path(self.results.root) / "review" / f"{self.condition}_{channel}_response.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export response waveform", str(default), "CSV (*.csv)")
        if not path:
            return
        self.store.export_response_csv(Path(path), times, values, channel)
        QMessageBox.information(self, "Exported", f"Saved:\n{path}")

    def _show_why(self) -> None:
        if self.results is None:
            return
        table = self.results.why_empty(self.condition)
        if table.empty:
            QMessageBox.information(self, "Rejected anchors", "No rejected candidates recorded.")
            return
        QMessageBox.information(self, "Why nothing was found", table.to_string(index=False, max_rows=50))

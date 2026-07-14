"""StartStop review — the clinicians' surface.

Three views of the same condition, because a StartStop recording holds two different kinds
of object and they need different eyes:

  * Detection — one detected response, in context, with a window you can annotate and
    export to the stimulator.
  * Overlay   — every detection of a channel superimposed, like the SIR crop view: that is
    how you see whether a response is stereotyped or whether the detector is firing on noise.
  * Segment   — the whole condition end to end, with bursts shaded and their RMS envelopes
    drawn on top.

Flags and free-text annotations attach to detections and to bursts alike, and survive a
re-run (they live in review/review.json, not in the pipeline's outputs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..results import Detection, SpontaneousResults, StartStopResults
from ..review_store import STATUS_LABEL, STATUSES, Annotation, ReviewStore
from .plot_canvas import FitPlotCanvas, VerticalPlotCanvas, style_channel_axis

BURST_COLOUR = "#ff9f1c"
ENV_COLOUR = "#c1121f"
DET_COLOUR = "#d62728"


class StartStopViewer(QWidget):
    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: StartStopResults | None = None
        self.spont: SpontaneousResults | None = None
        self.store: ReviewStore | None = None
        self.detections: list[Detection] = []
        self.condition = ""
        self._span: tuple[float, float] | None = None
        self._span_det: int | None = None       # which detection the span belongs to
        self._selectors: list[SpanSelector] = []
        self._seg_selectors: list[SpanSelector] = []

        # ---- left ----
        self.cond_box = QComboBox()
        self.cond_box.currentTextChanged.connect(self._on_condition)

        self.det_list = QListWidget()
        self.det_list.currentRowChanged.connect(lambda _: self._redraw())

        self.status_box = QComboBox()
        self.status_box.addItems([STATUS_LABEL[s] for s in STATUSES])
        self.status_box.currentIndexChanged.connect(self._on_status)

        self.burst_list = QListWidget()
        self.burst_list.currentRowChanged.connect(lambda _: self._redraw())

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Condition</b>"))
        lv.addWidget(self.cond_box)
        lv.addWidget(QLabel("<b>Detections</b>"))
        lv.addWidget(self.det_list, 2)
        lv.addWidget(QLabel("Status of the selected epoch"))
        lv.addWidget(self.status_box)
        lv.addWidget(QLabel("<b>Bursts</b> (spontaneous EMG)"))
        lv.addWidget(self.burst_list, 1)
        why = QPushButton("Why was nothing found?")
        why.setToolTip("The pipeline's own rejected-anchor reasons for this condition.")
        why.clicked.connect(self._show_why)
        lv.addWidget(why)

        # ---- right: three views ----
        self.detail = VerticalPlotCanvas(row_inches=1.5)
        self.overlay = FitPlotCanvas()
        self.segment = VerticalPlotCanvas(row_inches=1.4)

        self.views = QTabWidget()
        self.views.addTab(self.detail, "Detection")
        self.views.addTab(self.overlay, "Overlay")
        self.views.addTab(self.segment, "Segment (bursts + envelopes)")
        self.views.currentChanged.connect(lambda _: self._redraw())

        self.window_label = QLabel("Drag over any channel — in Detection or in Segment — to select a window.")
        self.window_label.setStyleSheet("color: gray;")
        self.annotate_btn = QPushButton("Annotate selection")
        self.annotate_btn.clicked.connect(self._annotate)
        self.export_btn = QPushButton("Export selection to CSV")
        self.export_btn.setToolTip("The response waveform to feed the stimulator (FES).")
        self.export_btn.clicked.connect(self._export)
        self.annotate_burst_btn = QPushButton("Annotate selected burst")
        self.annotate_burst_btn.clicked.connect(self._annotate_burst)

        bar = QHBoxLayout()
        bar.addWidget(self.window_label, 1)
        bar.addWidget(self.annotate_btn)
        bar.addWidget(self.annotate_burst_btn)
        bar.addWidget(self.export_btn)

        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(90)
        self.ann_list.itemDoubleClicked.connect(self._delete_annotation)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.views, 1)
        rv.addLayout(bar)
        rv.addWidget(QLabel("Annotations (double-click to delete)"))
        rv.addWidget(self.ann_list)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 1100])
        root = QVBoxLayout(self)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, results: StartStopResults, store: ReviewStore,
             spont: SpontaneousResults | None = None) -> None:
        self.results = results
        self.store = store
        self.spont = spont
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

        self.det_list.blockSignals(True)
        self.det_list.clear()
        for det in self.detections:
            status = self.store.review.status(condition, det.index) if self.store else "ok"
            text = det.label + (f"   [{STATUS_LABEL[status]}]" if status != "ok" else "")
            item = QListWidgetItem(text)
            if status in ("bad", "false_positive"):
                item.setForeground(Qt.red)
            elif status == "complex":
                item.setForeground(Qt.darkYellow)
            self.det_list.addItem(item)
        self.det_list.blockSignals(False)

        self.burst_list.clear()
        for _, b in self._bursts().iterrows():
            self.burst_list.addItem(QListWidgetItem(
                f"{b['Channel']}  #{int(b['Burst'])}  {float(b['Start_s']):.2f}–"
                f"{float(b['End_s']):.2f} s  ({float(b['RMS_peak_uV']):.0f} µV)"
            ))

        if self.detections:
            self.det_list.setCurrentRow(0)
        else:
            self._redraw()

    # ------------------------------------------------------------------ #
    def _bursts(self):
        import pandas as pd
        if self.spont is None:
            return pd.DataFrame()
        return self.spont.bursts(self.condition)

    def _current(self) -> Detection | None:
        row = self.det_list.currentRow()
        if row < 0 or row >= len(self.detections):
            return None
        return self.detections[row]

    def _redraw(self) -> None:
        idx = self.views.currentIndex()
        if idx == 0:
            self._draw_detection()
        elif idx == 1:
            self._draw_overlay()
        else:
            self._draw_segment()
        self._refresh_annotations()

    # ------------------------------------------------------------------ #
    def _draw_detection(self) -> None:
        det = self._current()
        if det is None or self.results is None:
            self.detail.message("No detection selected.")
            return

        # The span survives a redraw (that is how it stays on screen after the drag). A span
        # drawn on THIS detection is cleared only when you move to another one; a span drawn
        # on the whole segment (_span_det is None) is not owned by any detection, so it stays.
        if self._span_det is not None and self._span_det != det.index:
            self._span = None
            self._span_det = None
            self.window_label.setText("Drag over any channel to select a window.")

        status = self.store.review.status(self.condition, det.index) if self.store else "ok"
        self.status_box.blockSignals(True)
        self.status_box.setCurrentIndex(STATUSES.index(status))
        self.status_box.blockSignals(False)

        raw = self.results.raw(self.condition)
        sf = raw.info["sfreq"]
        pad = 1.0
        t0 = max(det.time - pad, 0.0)
        t1 = min(det.time + pad, float(raw.times[-1]))
        s0, s1 = int(t0 * sf), int(t1 * sf)

        chans = [c for c in raw.ch_names if "art" not in c.lower()]
        data = raw.get_data(picks=chans, start=s0, stop=s1) * 1e6
        times = np.arange(s0, s1) / sf

        axes = self.detail.make_axes(len(chans))
        for ax, ch, row in zip(axes, chans, range(len(chans))):
            responding = ch in det.channels
            ax.plot(times, data[row], lw=0.7,
                    color="#1f77b4" if responding else "0.72", zorder=2)
            ax.axvline(det.time, color=DET_COLOUR, lw=1.2, zorder=3)
            style_channel_axis(ax, ch, bold=responding, muted=not responding, compact=True)
            for ann in (self.store.review.annotations_for(self.condition) if self.store else []):
                if ann.tmax >= t0 and ann.tmin <= t1:
                    ax.axvspan(ann.tmin, ann.tmax, color="#ffd27f", alpha=0.35, zorder=1)
            # Keep the current selection visible on every channel, not only where it was drawn.
            if self._span is not None:
                ax.axvspan(self._span[0], self._span[1],
                           facecolor="#ffb703", alpha=0.3, lw=0, zorder=1)

        axes[-1].set_xlabel("time (s)")
        self.detail.fig.suptitle(
            f"{self.condition} — detection {det.index + 1}/{len(self.detections)} "
            f"on {'+'.join(det.channels)}  (bold = responding muscle)", fontsize=10,
        )

        # A selector on EVERY channel: dragging over any muscle must select, not only over
        # the bottom row. useblit=False because blitting does not survive the redraws and the
        # scroll-area resizes this canvas goes through — with it on, the drag drew nothing.
        self._selectors = [
            SpanSelector(
                ax, self._on_span, "horizontal", useblit=False,
                props=dict(alpha=0.3, facecolor="#ffb703"), drag_from_anywhere=True,
            )
            for ax in axes
        ]
        axes[-1].set_xlim(t0, t1)
        self.detail.draw()

    # ------------------------------------------------------------------ #
    def _draw_overlay(self) -> None:
        """Every detection of a channel, superimposed — the SIR view, for single responses."""
        if self.results is None or not self.detections:
            self.overlay.message("No detections in this condition.")
            return
        raw = self.results.raw(self.condition)
        sf = float(raw.info["sfreq"])
        tmin, tmax = -0.05, 0.10
        n0, n1 = int(tmin * sf), int(tmax * sf)
        rel = np.arange(n0, n1) / sf

        chans = sorted({c for d in self.detections for c in d.channels})
        axes = self.overlay.make_axes(len(chans))
        for ax, ch in zip(axes, chans):
            dets = [d for d in self.detections if ch in d.channels]
            traces = []
            for d in dets:
                s = int(d.time * sf)
                a, b = s + n0, s + n1
                if a < 0 or b > raw.n_times:
                    continue
                status = self.store.review.status(self.condition, d.index) if self.store else "ok"
                seg = raw.get_data(picks=[ch], start=a, stop=b)[0] * 1e6
                rejected = status in ("bad", "false_positive")
                ax.plot(rel, seg, lw=0.8, alpha=0.35,
                        color="#c0392b" if rejected else "gray", zorder=2)
                if not rejected:
                    traces.append(seg)
            if traces:
                m = np.mean(np.vstack(traces), axis=0)
                ax.plot(rel, m, color="black", lw=2.0, zorder=3)
            ax.axvline(0.0, color=DET_COLOUR, lw=1.0, ls=":", zorder=4)
            style_channel_axis(
                ax, f"{ch}\n{len(traces)} kept / {len(dets)}", bold=True, compact=True
            )
        axes[-1].set_xlabel("time from detection (s)")
        self.overlay.fig.suptitle(
            f"{self.condition} — detections overlaid (red = flagged, black = mean of the kept)",
            fontsize=10,
        )
        self.overlay.refresh()

    # ------------------------------------------------------------------ #
    def _draw_segment(self) -> None:
        """The whole condition, with bursts shaded and their envelopes on top."""
        if self.results is None:
            self.segment.message("No results loaded.")
            return
        raw = self.results.raw(self.condition)
        sf = float(raw.info["sfreq"])
        data = raw.get_data() * 1e6
        times = raw.times
        chans = [c for c in raw.ch_names if "art" not in c.lower()]

        bursts = self._bursts()
        envs = self.spont.envelopes_on_segment(self.condition) if self.spont else {}
        sel_burst = self._selected_burst()

        axes = self.segment.make_axes(len(chans))
        for ax, ch in zip(axes, chans):
            i = raw.ch_names.index(ch)
            ax.plot(times, data[i], lw=0.5, color="#1f77b4", zorder=2)

            if len(bursts):
                for _, b in bursts[bursts["Channel"].astype(str) == ch].iterrows():
                    is_sel = (
                        sel_burst is not None
                        and str(sel_burst["Channel"]) == ch
                        and int(sel_burst["Burst"]) == int(b["Burst"])
                    )
                    ax.axvspan(float(b["Start_s"]), float(b["End_s"]),
                               facecolor=BURST_COLOUR, edgecolor=BURST_COLOUR,
                               alpha=0.45 if is_sel else 0.2,
                               lw=1.2 if is_sel else 0, zorder=1)

            if ch in envs:
                et, ev = envs[ch]
                ax.plot(et, ev, color=ENV_COLOUR, lw=1.1, zorder=4)
                ax.plot(et, -ev, color=ENV_COLOUR, lw=1.1, zorder=4)

            for d in self.detections:
                if ch in d.channels:
                    ax.axvline(d.time, color=DET_COLOUR, lw=0.9, zorder=3)

            for ann in (self.store.review.annotations_for(self.condition) if self.store else []):
                ax.axvspan(ann.tmin, ann.tmax, color="#ffd27f", alpha=0.3, lw=0, zorder=0)

            if self._span is not None:
                ax.axvspan(self._span[0], self._span[1],
                           facecolor="#ffb703", alpha=0.3, lw=0, zorder=5)

            has = (len(bursts) and (bursts["Channel"].astype(str) == ch).any()) or \
                  any(ch in d.channels for d in self.detections)
            style_channel_axis(ax, ch, bold=bool(has), muted=not has, compact=True)

        axes[-1].set_xlabel("time (s)")
        axes[-1].set_xlim(times[0], times[-1])
        self.segment.fig.suptitle(
            f"{self.condition} — whole segment  "
            f"(orange = burst, red = RMS envelope, red line = detection)", fontsize=10,
        )
        # Selecting a window on the whole segment is the natural way to annotate a stretch of
        # activity, so this view gets the same drag as the Detection view — on every channel.
        self._seg_selectors = [
            SpanSelector(
                ax, self._on_span_segment, "horizontal", useblit=False,
                props=dict(alpha=0.3, facecolor="#ffb703"), drag_from_anywhere=True,
            )
            for ax in axes
        ]
        self.segment.draw()

    def _on_span_segment(self, tmin: float, tmax: float) -> None:
        if abs(tmax - tmin) < 1e-6:
            return
        self._span = (float(tmin), float(tmax))
        self._span_det = None      # a segment-wide window is not tied to one detection
        self.window_label.setText(
            f"Selected {tmin:.3f} – {tmax:.3f} s  ({tmax - tmin:.2f} s) — annotate or export it"
        )
        self._draw_segment()

    def _selected_burst(self):
        b = self._bursts()
        row = self.burst_list.currentRow()
        if len(b) == 0 or row < 0 or row >= len(b):
            return None
        return b.iloc[row]

    # ------------------------------------------------------------------ #
    def _on_span(self, tmin: float, tmax: float) -> None:
        if abs(tmax - tmin) < 1e-6:   # a plain click, not a drag
            return
        self._span = (float(tmin), float(tmax))
        det = self._current()
        self._span_det = det.index if det else None
        self.window_label.setText(
            f"Selected {tmin:.3f} – {tmax:.3f} s  ({1000 * (tmax - tmin):.0f} ms)  "
            "— annotate or export it"
        )
        self._draw_detection()   # paint the selection across every channel

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
            QMessageBox.information(self, "No selection",
                                    "Drag over the Detection view to select a window first.")
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
        self._redraw()

    def _annotate_burst(self) -> None:
        """Bursts get the same free-text layer as detections."""
        b = self._selected_burst()
        if self.store is None or b is None:
            QMessageBox.information(self, "No burst", "Select a burst in the list first.")
            return
        text, ok = QInputDialog.getText(
            self, "Annotate burst",
            f"{b['Channel']} #{int(b['Burst'])} — description:",
        )
        if not ok or not text.strip():
            return
        self.store.review.add_annotation(Annotation(
            condition=self.condition,
            detection_index=-1,
            tmin=float(b["Start_s"]), tmax=float(b["End_s"]),
            text=text.strip(), channel=str(b["Channel"]),
        ))
        self.store.save()
        self._redraw()

    def _refresh_annotations(self) -> None:
        self.ann_list.clear()
        if self.store is None:
            return
        for ann in self.store.review.annotations_for(self.condition):
            who = f" [{ann.channel}]" if ann.channel else ""
            self.ann_list.addItem(
                QListWidgetItem(f"{ann.tmin:.3f}–{ann.tmax:.3f} s{who}   {ann.text}")
            )

    def _delete_annotation(self, item: QListWidgetItem) -> None:
        if self.store is None:
            return
        anns = self.store.review.annotations_for(self.condition)
        row = self.ann_list.row(item)
        if 0 <= row < len(anns):
            self.store.review.remove_annotation(anns[row])
            self.store.save()
            self._redraw()

    # ------------------------------------------------------------------ #
    def _export(self) -> None:
        det = self._current()
        if self.results is None or self.store is None or self._span is None:
            QMessageBox.information(self, "No selection",
                                    "Drag over the Detection view to select a window first.")
            return
        raw = self.results.raw(self.condition)
        available = [c for c in raw.ch_names if "art" not in c.lower()]
        default = det.channels[0] if det and det.channels else available[0]
        channel, ok = QInputDialog.getItem(
            self, "Export to CSV", "Channel to export:", available,
            available.index(default) if default in available else 0, False,
        )
        if not ok:
            return

        sf = raw.info["sfreq"]
        s0, s1 = int(self._span[0] * sf), int(self._span[1] * sf)
        values = raw.get_data(picks=[channel], start=s0, stop=s1)[0] * 1e6
        times = np.arange(s0, s1) / sf

        target = Path(self.results.root) / "review" / f"{self.condition}_{channel}_response.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export response waveform", str(target), "CSV (*.csv)"
        )
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
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Why nothing was found")
        dlg.setText(f"{len(table)} channel/reason combinations.")
        dlg.setDetailedText(table.to_string(index=False))
        dlg.exec()

"""SIR per-crop review: the interactive replacement for the static PNG + re-run loop.

Left: the crop list (config @ amplitude) with detection counts, so "which crops are 0" is
visible at a glance. Right: epoch overlay + mean with onset/P1/P2 markers, one row per
channel, and a live per-channel table underneath.

Two interaction modes:
  * Markers  — click a peak to force P1 onto it; shift/ctrl/alt-click to reject, whitelist
    or exclude. Rejecting updates the plot AND the table immediately.
  * Template — drag over a response; it becomes a matching template the next run detects by.

Every edit is written at the most specific (config, amp, channel) key, so it is additive:
it can never alter a crop you did not name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..results import Crop, SIRResults
from .plot_canvas import VerticalPlotCanvas, style_channel_axis

# P1 is drawn red and P2 green — the convention the whole project reads by.
C_P1, C_P2, C_ONSET = "#d62728", "#2ca02c", "#7f7f7f"


class SIRViewer(QWidget):
    rerun_requested = Signal()
    template_made = Signal(str)  # template name

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: SIRResults | None = None
        self.crop: Crop | None = None
        self._axes: dict[str, object] = {}     # channel -> axis, for click hit-testing
        self._selectors: list[SpanSelector] = []
        self._selection: tuple[str, float, float] | None = None  # channel, t0, t1

        # ---- left: crops ----
        self.crop_list = QListWidget()
        self.crop_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.crop_list.currentRowChanged.connect(self._on_crop_selected)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Crops</b> (config @ amplitude)"))
        lv.addWidget(self.crop_list)
        self.exclude_config_btn = QPushButton("Exclude this config")
        self.exclude_config_btn.clicked.connect(self._toggle_exclude_config)
        lv.addWidget(self.exclude_config_btn)
        self.recruit_btn = QPushButton("Recruitment table")
        self.recruit_btn.clicked.connect(self.show_recruitment)
        lv.addWidget(self.recruit_btn)

        # ---- right: mode bar + canvas + table ----
        self.mode_markers = QRadioButton("Markers")
        self.mode_markers.setChecked(True)
        self.mode_markers.toggled.connect(self._on_mode_changed)
        self.mode_template = QRadioButton("Template (drag to select a response)")
        self.mode_template.toggled.connect(self._on_mode_changed)

        self.polarity = QComboBox()
        self.polarity.addItems(["auto (sign at click)", "force positive ↑", "force negative ↓"])
        self.bind_window = QCheckBox("Bind P1 to ±3 ms")
        self.bind_window.setChecked(True)
        self.bind_window.setToolTip(
            "On: writes (min_lat, max_lat) so P1 cannot drift to a stronger later peak.\n"
            "Off: writes a bare min_lat — P1 = the dominant deflection at or after the click.\n"
            "A click before t=0 forces a peak BEFORE the stimulus, which is allowed."
        )

        self.make_template_btn = QPushButton("Make template from selection")
        self.make_template_btn.setEnabled(False)
        self.make_template_btn.clicked.connect(self._make_template)
        self.only_user_templates = QCheckBox("Detect by my templates only")
        self.only_user_templates.setToolTip(
            "On: the next run matches against YOUR templates alone.\n"
            "Off: your templates are added on top of the 26 stock ones."
        )
        self.only_user_templates.toggled.connect(
            lambda v: setattr(self.session, "template_only_user", bool(v))
        )

        # One tall row per channel, scrolled — as in the pipeline's own panels. A slider
        # trades rows-on-screen against row height.
        self.zoom = QSlider(Qt.Horizontal)
        self.zoom.setRange(9, 40)      # tenths of an inch per channel row
        self.zoom.setValue(20)
        self.zoom.setFixedWidth(110)
        self.zoom.setToolTip("Height of each channel row")
        self.zoom.valueChanged.connect(lambda v: self.plot.set_row_height(v / 10.0))

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Mode:"))
        bar.addWidget(self.mode_markers)
        bar.addWidget(self.mode_template)
        bar.addSpacing(16)
        bar.addWidget(self.polarity)
        bar.addWidget(self.bind_window)
        bar.addSpacing(16)
        bar.addWidget(self.make_template_btn)
        bar.addWidget(self.only_user_templates)
        bar.addStretch()
        bar.addWidget(QLabel("Row height:"))
        bar.addWidget(self.zoom)

        self.plot = VerticalPlotCanvas(row_inches=2.0)
        self.plot.canvas.mpl_connect("button_press_event", self._on_click)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Channel", "Detections", "Epochs", "P1 (ms)", "PTP (µV)", "Status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(180)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_table_double_click)

        self.hint = QLabel(
            "Markers mode — click: force P1 · Shift+click: reject (false positive) · "
            "Ctrl+click: whitelist · Alt+click: exclude channel · right-click: clear. "
            "Double-click a table row to reject/restore it."
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 11px;")

        self.rerun_btn = QPushButton("Apply edits and re-run")
        self.rerun_btn.clicked.connect(self.rerun_requested)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addLayout(bar)
        rv.addWidget(self.plot, 1)
        rv.addWidget(self.hint)
        rv.addWidget(self.table)
        rv.addWidget(self.rerun_btn)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([260, 1000])

        root = QVBoxLayout(self)
        root.addWidget(split)
        self._on_mode_changed()

    # ------------------------------------------------------------------ #
    def load(self, results: SIRResults) -> None:
        self.results = results
        self._refresh_crop_list()
        if results.crops:
            self.crop_list.setCurrentRow(0)

    def _refresh_crop_list(self) -> None:
        """Rebuild the list; counts reflect manual rejections without a re-run."""
        if self.results is None:
            return
        row = self.crop_list.currentRow()
        self.crop_list.blockSignals(True)
        self.crop_list.clear()
        for crop in self.results.crops:
            n = self.results.detection_count(crop, self.session)
            if crop.config in self.session.exclude_configs:
                item = QListWidgetItem(f"{crop.label}    [excluded]")
                item.setForeground(Qt.gray)
            else:
                item = QListWidgetItem(f"{crop.label}    [{n}]")
                if n == 0:
                    item.setForeground(Qt.red)
            self.crop_list.addItem(item)
        self.crop_list.blockSignals(False)
        if 0 <= row < self.crop_list.count():
            self.crop_list.setCurrentRow(row)

    def _on_crop_selected(self, row: int) -> None:
        if self.results is None or row < 0 or row >= len(self.results.crops):
            return
        self.crop = self.results.crops[row]
        self._selection = None
        self.make_template_btn.setEnabled(False)
        self._draw()

    # ------------------------------------------------------------------ #
    def _on_mode_changed(self) -> None:
        template = self.mode_template.isChecked()
        self.polarity.setEnabled(not template)
        self.bind_window.setEnabled(not template)
        self.only_user_templates.setEnabled(template)
        self.hint.setText(
            "Template mode — drag over the response you want detected; release, then "
            "“Make template from selection”. The next run matches against it."
            if template else
            "Markers mode — click: force P1 · Shift+click: reject (false positive) · "
            "Ctrl+click: whitelist · Alt+click: exclude channel · right-click: clear. "
            "Double-click a table row to reject/restore it."
        )
        self._draw()

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        if self.results is None or self.crop is None:
            return
        self._axes.clear()
        self._selectors.clear()

        try:
            epochs = self.results.load_epochs(self.crop)
        except Exception as exc:  # a crop can exist without a readable epoch file
            self.plot.message(f"Cannot read epochs:\n{exc}")
            return

        chans = [c for c in epochs.ch_names if "art" not in c.lower()]
        if not chans:
            self.plot.message("This crop has no EMG channels.")
            return

        times = epochs.times
        data = epochs.get_data(picks=chans) * 1e6  # -> µV
        mean = data.mean(axis=0)
        template_mode = self.mode_template.isChecked()

        axes = self.plot.make_axes(len(chans))
        for ax, ch, idx in zip(axes, chans, range(len(chans))):
            self._axes[ch] = ax
            state = self.session.edit_state(self.crop.config, self.crop.amp, ch)
            rejected = state["suppressed"] or state["channel_excluded"] or state["config_excluded"]

            ax.plot(times, data[:, idx, :].T, color="0.82", lw=0.5, zorder=1)
            ax.plot(times, mean[idx], color="black", lw=1.6, zorder=3)
            ax.axvline(0.0, color="0.45", lw=0.9, ls=":", zorder=2)
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=1)

            # Scale each row to its own RESPONSE, not to the whole epoch: the stimulus
            # artifact and pre-stimulus drift are often an order of magnitude larger and
            # would flatten the very deflection being judged.
            resp = mean[idx][times > 0]
            span = float(np.nanmax(np.abs(resp))) if resp.size and np.isfinite(resp).any() else 0.0
            if span > 0:
                ax.set_ylim(-1.6 * span, 1.6 * span)

            # Markers vanish the moment a detection is rejected — the plot and the table
            # must never disagree about what counts as detected.
            m = self.results.markers(self.crop, ch)
            if not m.empty and not rejected:
                for col, colour, name in (
                    ("Onset latency", C_ONSET, "onset"),
                    ("Peak1 latency", C_P1, "P1"),
                    ("Peak2 latency", C_P2, "P2"),
                ):
                    lat = m[col].to_numpy(dtype=float)
                    lat = lat[~np.isnan(lat)]
                    if lat.size:
                        ax.axvline(float(np.median(lat)), color=colour, lw=1.4, alpha=0.9, zorder=4)

            tags = []
            if state["forced"]:
                tags.append(f"forced {state['forced'][0]}")
            if state["suppressed"]:
                tags.append("REJECTED")
            if state["whitelisted"]:
                tags.append("whitelisted")
            if state["channel_excluded"]:
                tags.append("EXCLUDED")
            style_channel_axis(
                ax,
                ch + (f"\n[{', '.join(tags)}]" if tags else ""),
                bold=bool(m["Peak1 latency"].notna().any()) if not m.empty else False,
                muted=rejected,
            )
            if rejected:
                ax.set_facecolor("#f7f0f0")

            if self._selection and self._selection[0] == ch:
                ax.axvspan(self._selection[1], self._selection[2],
                           color="#ffb703", alpha=0.3, zorder=5)

            if template_mode:
                self._selectors.append(SpanSelector(
                    ax, lambda a, b, c=ch: self._on_span(c, a, b), "horizontal",
                    useblit=True, props=dict(alpha=0.3, facecolor="#ffb703"),
                ))

        axes[-1].set_xlabel("time (s)", fontsize=10)
        axes[-1].set_xlim(times[0], times[-1])
        self.plot.fig.suptitle(
            f"{self.crop.label}   —   {len(epochs)} epochs   "
            f"(red = P1, green = P2, grey = onset; each row scaled to its own response)",
            fontsize=11,
        )
        self.plot.draw()

        self._refresh_table()
        excluded = self.crop.config in self.session.exclude_configs
        self.exclude_config_btn.setText(
            "Un-exclude this config" if excluded else "Exclude this config"
        )

    # ------------------------------------------------------------------ #
    def _refresh_table(self) -> None:
        if self.results is None or self.crop is None:
            return
        # Pass the plotted channels so quiet muscles get a row too, not just the ones the
        # pipeline emitted metrics for.
        df = self.results.channel_summary(
            self.crop, self.session, channels=list(self._axes) or None
        )
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, key in enumerate(["Channel", "Detections", "Epochs", "P1 (ms)", "PTP (µV)", "Status"]):
                val = row[key]
                if isinstance(val, float):
                    text = "—" if not np.isfinite(val) else f"{val:.2f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                if row["Status"] == "rejected":
                    item.setForeground(Qt.red)
                elif row["Status"] == "whitelisted":
                    item.setForeground(Qt.darkGreen)
                self.table.setItem(r, c, item)

    def _on_table_double_click(self, row: int, _col: int) -> None:
        """Reject / restore straight from the table."""
        if self.crop is None:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        self.session.toggle_suppress(self.crop.config, self.crop.amp, item.text())
        self._draw()
        self._refresh_crop_list()

    # ------------------------------------------------------------------ #
    def _channel_at(self, event) -> str | None:
        for ch, ax in self._axes.items():
            if event.inaxes is ax:
                return ch
        return None

    def _on_span(self, channel: str, t0: float, t1: float) -> None:
        if abs(t1 - t0) < 1e-4:
            return
        self._selection = (channel, float(t0), float(t1))
        self.make_template_btn.setEnabled(True)
        self.make_template_btn.setText(
            f"Make template from {channel}  ({1000 * (t1 - t0):.0f} ms)"
        )

    def _on_click(self, event) -> None:
        if self.crop is None or event.inaxes is None or self.mode_template.isChecked():
            return
        ch = self._channel_at(event)
        if ch is None:
            return
        cfg, amp = self.crop.config, self.crop.amp
        mods = event.guiEvent.modifiers() if event.guiEvent is not None else Qt.NoModifier

        if event.button == 3:  # right-click clears the forced marker
            self.session.clear_force(cfg, amp, ch)
        elif mods & Qt.AltModifier:
            self.session.toggle_exclude_channel(ch)
        elif mods & Qt.ShiftModifier:
            self.session.toggle_suppress(cfg, amp, ch)
        elif mods & Qt.ControlModifier:
            self.session.toggle_whitelist(cfg, amp, ch)
        else:
            t = float(event.xdata)
            mode = self.polarity.currentIndex()
            positive = (mode == 1) if mode in (1, 2) else float(event.ydata) >= 0.0
            if self.bind_window.isChecked():
                self.session.force_p1(cfg, amp, ch, positive, t - 0.003, t + 0.003)
            else:
                self.session.force_p1(cfg, amp, ch, positive, t)

        self._draw()
        self._refresh_crop_list()

    def _toggle_exclude_config(self) -> None:
        if self.crop is None:
            return
        self.session.toggle_exclude_config(self.crop.config)
        self._draw()
        self._refresh_crop_list()

    # ------------------------------------------------------------------ #
    def _make_template(self) -> None:
        """Turn the dragged-out response into a template the next run detects by."""
        from ..templates import TemplateBank, build_template

        if self.results is None or self.crop is None or self._selection is None:
            return
        channel, t0, t1 = self._selection
        try:
            times, mean = self.results.mean_wave(self.crop, channel)
            tpl = build_template(times, mean, t0, t1,
                                 source=f"{self.crop.label} / {channel}")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot build template", str(exc))
            return

        if self.session.template_dir is None:
            base = self.session.output_dir or self.results.root
            self.session.template_dir = Path(base) / "templates"
        bank = TemplateBank(
            self.session.template_dir,
            include_stock=not self.only_user_templates.isChecked(),
        )
        name = bank.add(tpl)
        self.session.template_only_user = self.only_user_templates.isChecked()

        m = tpl.markers_s
        QMessageBox.information(
            self, "Template created",
            f"{name} built from {self.crop.label} / {channel}.\n\n"
            f"onset {1000 * m['onset']:.1f} ms · P1 {1000 * m['peak1']:.1f} ms · "
            f"P2 {1000 * m['peak2']:.1f} ms\n\n"
            f"Bank: {bank.count()} template(s) in\n{self.session.template_dir}\n\n"
            "Re-run to detect with it.",
        )
        self.template_made.emit(name)

    # ------------------------------------------------------------------ #
    def show_recruitment(self) -> None:
        if self.results is None:
            return
        table = self.results.recruitment_table(self.session)
        if table.empty:
            QMessageBox.information(self, "Recruitment", "No detections in this run.")
            return
        zeros = int((table.detections == 0).sum())
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Recruitment table")
        dlg.setText(f"{len(table)} (config, amplitude, channel) rows — {zeros} with no detections.")
        dlg.setDetailedText(table.to_string(index=False))
        dlg.exec()

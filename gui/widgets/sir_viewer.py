"""SIR per-crop review: the interactive replacement for the static PNG + re-run loop.

Layout: crops on the left, a narrow channel column in the middle (every channel visible at
once — SIR epochs are short, so the time axis can be squeezed and nothing needs scrolling),
and on the right the live per-channel table plus recruitment curves.

Markers are drawn the way the pipeline's own panels draw them: a point per epoch, onset
blue, P1 red, P2 green — not a line at the median, which hid how consistent the detection
actually was across trials.

Two interaction modes:
  * Markers  — click a peak to force P1; shift/ctrl/alt-click to reject, whitelist, exclude.
  * Template — drag over a response, check its markers, and it becomes a matching template.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..results import Crop, SIRResults
from .plot_canvas import C_ONSET, C_P1, C_P2, MARKER_SIZE, FitPlotCanvas, style_channel_axis
from .recruitment_panel import RecruitmentPanel
from .template_editor import TemplateEditor


class SIRViewer(QWidget):
    rerun_requested = Signal()
    template_made = Signal(str)

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: SIRResults | None = None
        self.crop: Crop | None = None
        self._axes: dict[str, object] = {}
        self._selectors: list[SpanSelector] = []
        self._selection: tuple[str, float, float] | None = None  # channel, t0, t1

        # ---- left: crops ----
        self.crop_list = QListWidget()
        self.crop_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.crop_list.currentRowChanged.connect(self._on_crop_selected)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(QLabel("<b>Crops</b> (config @ amplitude)"))
        lv.addWidget(self.crop_list)
        self.exclude_config_btn = QPushButton("Exclude this config")
        self.exclude_config_btn.clicked.connect(self._toggle_exclude_config)
        lv.addWidget(self.exclude_config_btn)

        # ---- middle: the channel column ----
        self.mode_markers = QRadioButton("Markers")
        self.mode_markers.setChecked(True)
        self.mode_markers.toggled.connect(self._on_mode_changed)
        self.mode_template = QRadioButton("Template")
        self.mode_template.setToolTip("Drag over a response to turn it into a matching template.")
        self.mode_template.toggled.connect(self._on_mode_changed)

        self.polarity = QComboBox()
        self.polarity.addItems(["auto", "force ↑", "force ↓"])
        self.polarity.setToolTip("Polarity of the forced P1. 'auto' = the sign under the cursor.")
        self.bind_window = QCheckBox("±3 ms")
        self.bind_window.setChecked(True)
        self.bind_window.setToolTip(
            "On: writes (min_lat, max_lat) so P1 cannot drift to a stronger later peak.\n"
            "Off: writes a bare min_lat — P1 = the dominant deflection at or after the click."
        )

        self.make_template_btn = QPushButton("Make template…")
        self.make_template_btn.setEnabled(False)
        self.make_template_btn.clicked.connect(self._make_template)
        self.only_user_templates = QCheckBox("mine only")
        self.only_user_templates.setToolTip(
            "On: the next run matches against YOUR templates alone.\n"
            "Off: your templates are added on top of the 26 stock ones."
        )
        self.only_user_templates.toggled.connect(
            lambda v: setattr(self.session, "template_only_user", bool(v))
        )

        # One compact row: every pixel spent here is a pixel the channel stack loses.
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for wdg in (self.mode_markers, self.mode_template, self.polarity,
                    self.bind_window, self.make_template_btn, self.only_user_templates):
            bar.addWidget(wdg)
        bar.addStretch()

        self.plot = FitPlotCanvas()
        self.plot.mpl_connect("button_press_event", self._on_click)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 10px;")
        self.hint.setMaximumHeight(28)

        middle = QWidget()
        mv = QVBoxLayout(middle)
        mv.setContentsMargins(2, 2, 2, 2)
        mv.setSpacing(2)
        mv.addLayout(bar)
        mv.addWidget(self.plot, 1)
        mv.addWidget(self.hint)

        # ---- right: table + recruitment ----
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Channel", "Detections", "Epochs", "P1 (ms)", "PTP (µV)", "Status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_table_double_click)
        self.table.itemSelectionChanged.connect(self._on_table_selection)

        self.recruitment = RecruitmentPanel()

        self.rerun_btn = QPushButton("Apply edits and re-run")
        self.rerun_btn.clicked.connect(self.rerun_requested)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.addWidget(QLabel("<b>Channels in this crop</b> — double-click a row to reject / restore"))
        rv.addWidget(self.table, 1)
        rv.addWidget(self.recruitment, 2)
        rv.addWidget(self.rerun_btn)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(middle)
        split.addWidget(right)
        split.setSizes([220, 430, 620])

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
        self.make_template_btn.setText("Make template from selection…")
        self._draw()
        self._refresh_recruitment()

    # ------------------------------------------------------------------ #
    def _on_mode_changed(self) -> None:
        template = self.mode_template.isChecked()
        self.polarity.setEnabled(not template)
        self.bind_window.setEnabled(not template)
        self.only_user_templates.setEnabled(template)
        self.make_template_btn.setEnabled(template and self._selection is not None)
        self.hint.setText(
            "Template mode — drag over the response you want detected, then "
            "“Make template from selection…” to check its markers and save it."
            if template else
            "Click: force P1 · Shift+click: reject · Ctrl+click: whitelist · "
            "Alt+click: exclude channel · right-click: clear."
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
        except Exception as exc:
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

            ax.plot(times, data[:, idx, :].T, color="gray", alpha=0.7, lw=0.7, zorder=1)
            ax.plot(times, mean[idx], color="black", lw=2.0, zorder=3)
            ax.axvline(0.0, color="0.45", lw=0.8, ls=":", zorder=2)

            # Scale each row to its own post-stimulus response: the artifact and pre-stimulus
            # drift are often an order of magnitude larger and would flatten the deflection
            # being judged.
            resp = mean[idx][times > 0]
            span = float(np.nanmax(np.abs(resp))) if resp.size and np.isfinite(resp).any() else 0.0
            if span > 0:
                ax.set_ylim(-1.7 * span, 1.7 * span)

            m = self.results.markers(self.crop, ch)
            detected = (not m.empty) and bool(m["Peak1 latency"].notna().any())
            if detected and not rejected:
                self._scatter_markers(ax, m, times, data[:, idx, :])

            if state["forced"]:
                self._draw_forced(ax, state["forced"])

            style_channel_axis(
                ax, ch + (self._tag(state) or ""),
                bold=detected and not rejected, muted=rejected, compact=True,
            )
            if rejected:
                ax.set_facecolor("#f7f0f0")

            if self._selection and self._selection[0] == ch:
                ax.axvspan(self._selection[1], self._selection[2],
                           color="#ffb703", alpha=0.35, lw=0, zorder=0)

            if template_mode:
                self._selectors.append(SpanSelector(
                    ax, lambda a, b, c=ch: self._on_span(c, a, b), "horizontal",
                    useblit=False, props=dict(alpha=0.3, facecolor="#ffb703"),
                ))

        axes[-1].set_xlabel("time (s)", fontsize=8)
        axes[-1].set_xlim(times[0], times[-1])
        self.plot.fig.suptitle(
            f"{self.crop.label} · {len(epochs)} epochs · onset•blue P1•red P2•green",
            fontsize=8,
        )
        self.plot.refresh()
        self._refresh_table()

    @staticmethod
    def _tag(state) -> str:
        tags = []
        if state["forced"]:
            pol, val = state["forced"]
            tags.append(f"forced {'↑' if pol == 'pos' else '↓'}")
        if state["suppressed"]:
            tags.append("REJECTED")
        if state["whitelisted"]:
            tags.append("whitelisted")
        if state["channel_excluded"]:
            tags.append("EXCLUDED")
        return f"\n[{', '.join(tags)}]" if tags else ""

    def _scatter_markers(self, ax, m, times, ch_data) -> None:
        """A point per epoch, on that epoch's own trace — as the pipeline's panels do it."""
        for _, row in m.iterrows():
            ep = int(row["Epoch"]) if not np.isnan(row.get("Epoch", np.nan)) else None
            if ep is None or ep >= ch_data.shape[0]:
                continue
            for col, colour in (
                ("Onset latency", C_ONSET), ("Peak1 latency", C_P1), ("Peak2 latency", C_P2)
            ):
                t = row.get(col, np.nan)
                if t is None or (isinstance(t, float) and np.isnan(t)):
                    continue
                i = int(np.argmin(np.abs(times - float(t))))
                ax.scatter(float(t), ch_data[ep, i], color=colour, s=MARKER_SIZE, zorder=5)

    def _draw_forced(self, ax, forced) -> None:
        """Show exactly which latency was forced and which way P1 must point."""
        pol, val = forced
        up = pol == "pos"
        colour = "#8e44ad"
        if isinstance(val, (tuple, list)):
            lo, hi = float(val[0]), float(val[1])
            ax.axvspan(lo, hi, color=colour, alpha=0.12, lw=0, zorder=0)
            t = 0.5 * (lo + hi)
            label = f"forced {'↑' if up else '↓'}  {1000 * lo:.0f}…{1000 * hi:.0f} ms"
        else:
            t = float(val)
            ax.axvline(t, color=colour, lw=1.0, ls="--", zorder=4)
            label = f"forced {'↑' if up else '↓'}  ≥ {1000 * t:.0f} ms"
        ax.annotate(
            label, xy=(t, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, -10), textcoords="offset points",
            ha="center", va="top", fontsize=7, color=colour,
        )
        lo_y, hi_y = ax.get_ylim()
        ax.annotate(
            "↑" if up else "↓", xy=(t, hi_y * 0.6 if up else lo_y * 0.6),
            ha="center", va="center", fontsize=14, color=colour, zorder=6,
        )

    # ------------------------------------------------------------------ #
    def _refresh_table(self) -> None:
        if self.results is None or self.crop is None:
            return
        df = self.results.channel_summary(
            self.crop, self.session, channels=list(self._axes) or None
        )
        self.table.blockSignals(True)
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, key in enumerate(["Channel", "Detections", "Epochs", "P1 (ms)", "PTP (µV)", "Status"]):
                val = row[key]
                text = ("—" if not np.isfinite(val) else f"{val:.2f}") if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                if row["Status"] == "rejected":
                    item.setForeground(Qt.red)
                elif row["Status"] == "whitelisted":
                    item.setForeground(Qt.darkGreen)
                elif row["Status"] == "no response":
                    item.setForeground(Qt.gray)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)

    def _on_table_selection(self) -> None:
        items = self.table.selectedItems()
        if items:
            self.recruitment.set_highlight(self.table.item(items[0].row(), 0).text())

    def _on_table_double_click(self, row: int, _col: int) -> None:
        if self.crop is None:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        self.session.toggle_suppress(self.crop.config, self.crop.amp, item.text())
        self._draw()
        self._refresh_crop_list()
        self._refresh_recruitment()

    def _refresh_recruitment(self) -> None:
        if self.results is None or self.crop is None:
            return
        df = self.results.recruitment_curves(self.crop.config, self.session)
        self.recruitment.show_config(df, self.crop.config)

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
            f"Make template from {channel} ({1000 * (t1 - t0):.0f} ms)…"
        )
        self._draw()  # keep the selection visible instead of letting it vanish

    def _on_click(self, event) -> None:
        if self.crop is None or event.inaxes is None or self.mode_template.isChecked():
            return
        ch = self._channel_at(event)
        if ch is None:
            return
        cfg, amp = self.crop.config, self.crop.amp
        mods = event.guiEvent.modifiers() if event.guiEvent is not None else Qt.NoModifier

        if event.button == 3:
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
        self._refresh_recruitment()

    def _toggle_exclude_config(self) -> None:
        if self.crop is None:
            return
        self.session.toggle_exclude_config(self.crop.config)
        self._draw()
        self._refresh_crop_list()

    # ------------------------------------------------------------------ #
    def _make_template(self) -> None:
        from ..templates import TemplateBank, build_template

        if self.results is None or self.crop is None or self._selection is None:
            return
        channel, t0, t1 = self._selection
        try:
            times, mean = self.results.mean_wave(self.crop, channel)
            tpl = build_template(times, mean, t0, t1, source=f"{self.crop.label} / {channel}")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot build template", str(exc))
            return

        editor = TemplateEditor(tpl, self)
        if not editor.exec():
            return
        tpl = editor.edited_template()

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
            self, "Template saved",
            f"{name} — from {self.crop.label} / {channel}\n\n"
            f"onset {1000 * m['onset']:.1f} ms · P1 {1000 * m['peak1']:.1f} ms · "
            f"P2 {1000 * m['peak2']:.1f} ms\n\n"
            f"Bank: {bank.count()} template(s).\nRe-run to detect with it.",
        )
        self.template_made.emit(name)

    # ------------------------------------------------------------------ #
    def show_recruitment(self) -> None:
        """Kept for the View menu: the curves live in the right-hand panel now."""
        if self.results is None or self.crop is None:
            return
        self._refresh_recruitment()

"""SIR per-crop review: the interactive replacement for the static PNG + re-run loop.

Left: the crop list (config @ amplitude) with detection counts, so "which crops are 0" is
visible at a glance. Right: epoch overlay + mean with onset/P1/P2 markers, one row per
channel. Clicking a channel's trace forces P1 onto that peak; the edit is written as the
most specific (config, amp, channel) key, so it can never leak onto a crop you didn't name.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ..results import Crop, SIRResults

# P1 is drawn red and P2 green — the convention the whole project reads by.
C_P1, C_P2, C_ONSET = "#d62728", "#2ca02c", "#7f7f7f"


class SIRViewer(QWidget):
    rerun_requested = Signal()

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: SIRResults | None = None
        self.crop: Crop | None = None
        self._axes: dict[str, object] = {}   # channel -> axis, for click hit-testing
        self._pending_polarity = "auto"

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

        # ---- right: canvas + edit bar ----
        self.fig = Figure(figsize=(8, 9), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Click a peak to force P1:"))
        self.polarity = QComboBox()
        self.polarity.addItems(["auto (sign at click)", "force positive ↑", "force negative ↓"])
        bar.addWidget(self.polarity)
        self.bind_window = QCheckBox("Bind P1 to a ±3 ms window around the click")
        self.bind_window.setChecked(True)
        self.bind_window.setToolTip(
            "On: writes (min_lat, max_lat) so P1 cannot drift to a stronger later peak.\n"
            "Off: writes a bare min_lat — P1 = the dominant deflection at or after the click.\n"
            "A click before t=0 forces a peak BEFORE the stimulus, which is allowed."
        )
        bar.addWidget(self.bind_window)
        bar.addStretch()

        self.hint = QLabel("Shift+click = suppress · Ctrl+click = whitelist (bypass gates) · "
                           "Alt+click = exclude channel · right-click = clear")
        self.hint.setStyleSheet("color: gray; font-size: 11px;")

        self.rerun_btn = QPushButton("Apply edits and re-run")
        self.rerun_btn.clicked.connect(self.rerun_requested)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addLayout(bar)
        rv.addWidget(self.canvas, 1)
        rv.addWidget(self.hint)
        rv.addWidget(self.rerun_btn)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([260, 900])

        root = QVBoxLayout(self)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, results: SIRResults) -> None:
        self.results = results
        self.crop_list.clear()
        for crop in results.crops:
            n = results.detection_count(crop)
            item = QListWidgetItem(f"{crop.label}    [{n}]")
            if crop.config in self.session.exclude_configs:
                item.setForeground(Qt.gray)
                item.setText(f"{crop.label}    [excluded]")
            elif n == 0:
                item.setForeground(Qt.red)
            self.crop_list.addItem(item)
        if results.crops:
            self.crop_list.setCurrentRow(0)

    def _on_crop_selected(self, row: int) -> None:
        if self.results is None or row < 0 or row >= len(self.results.crops):
            return
        self.crop = self.results.crops[row]
        self._draw()

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        if self.results is None or self.crop is None:
            return
        self.fig.clear()
        self._axes.clear()

        try:
            epochs = self.results.load_epochs(self.crop)
        except Exception as exc:  # a crop can exist without a readable epoch file
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Cannot read epochs:\n{exc}", ha="center", va="center")
            self.canvas.draw_idle()
            return

        chans = [c for c in epochs.ch_names if "art" not in c.lower()]
        if not chans:
            self.canvas.draw_idle()
            return

        times = epochs.times
        data = epochs.get_data(picks=chans) * 1e6  # -> µV
        mean = data.mean(axis=0)

        axes = self.fig.subplots(len(chans), 1, sharex=True, squeeze=False)[:, 0]
        for ax, ch, idx in zip(axes, chans, range(len(chans))):
            self._axes[ch] = ax
            state = self.session.edit_state(self.crop.config, self.crop.amp, ch)

            ax.plot(times, data[:, idx, :].T, color="0.8", lw=0.4, zorder=1)
            ax.plot(times, mean[idx], color="black", lw=1.4, zorder=3)
            ax.axvline(0.0, color="0.5", lw=0.8, ls=":", zorder=2)

            m = self.results.markers(self.crop, ch)
            if not m.empty and not state["suppressed"]:
                for col, colour in (
                    ("Onset latency", C_ONSET), ("Peak1 latency", C_P1), ("Peak2 latency", C_P2)
                ):
                    lat = m[col].to_numpy(dtype=float)
                    lat = lat[~np.isnan(lat)]
                    if lat.size:
                        t = float(np.median(lat))
                        ax.axvline(t, color=colour, lw=1.2, alpha=0.9, zorder=4)

            tags = []
            if state["forced"]:
                tags.append(f"forced {state['forced'][0]}")
            if state["suppressed"]:
                tags.append("SUPPRESSED")
            if state["whitelisted"]:
                tags.append("whitelisted")
            if state["channel_excluded"]:
                tags.append("EXCLUDED")
            title = ch + (f"   [{', '.join(tags)}]" if tags else "")
            ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
            ax.tick_params(labelsize=7)
            if state["suppressed"] or state["channel_excluded"]:
                ax.set_facecolor("#f5f0f0")

        axes[-1].set_xlabel("time (s)")
        self.fig.suptitle(
            f"{self.crop.label}   —   {len(epochs)} epochs   "
            f"(red = P1, green = P2, grey = onset)", fontsize=10,
        )
        self.canvas.draw_idle()

        excluded = self.crop.config in self.session.exclude_configs
        self.exclude_config_btn.setText(
            "Un-exclude this config" if excluded else "Exclude this config"
        )

    # ------------------------------------------------------------------ #
    def _channel_at(self, event) -> str | None:
        for ch, ax in self._axes.items():
            if event.inaxes is ax:
                return ch
        return None

    def _on_click(self, event) -> None:
        if self.crop is None or event.inaxes is None:
            return
        ch = self._channel_at(event)
        if ch is None:
            return
        cfg, amp = self.crop.config, self.crop.amp
        mods = event.guiEvent.modifiers() if event.guiEvent is not None else Qt.NoModifier

        if event.button == 3:  # right-click clears the forced marker
            self.session.clear_force(cfg, amp, ch)
            self._draw()
            return

        if mods & Qt.AltModifier:
            self.session.toggle_exclude_channel(ch)
            self._draw()
            return
        if mods & Qt.ShiftModifier:
            self.session.toggle_suppress(cfg, amp, ch)
            self._draw()
            return
        if mods & Qt.ControlModifier:
            self.session.toggle_whitelist(cfg, amp, ch)
            self._draw()
            return

        t = float(event.xdata)
        mode = self.polarity.currentIndex()
        if mode == 1:
            positive = True
        elif mode == 2:
            positive = False
        else:
            positive = float(event.ydata) >= 0.0  # sign of the deflection under the cursor

        if self.bind_window.isChecked():
            self.session.force_p1(cfg, amp, ch, positive, t - 0.003, t + 0.003)
        else:
            self.session.force_p1(cfg, amp, ch, positive, t)
        self._draw()

    def _toggle_exclude_config(self) -> None:
        if self.crop is None:
            return
        self.session.toggle_exclude_config(self.crop.config)
        if self.results:
            row = self.crop_list.currentRow()
            self.load(self.results)
            self.crop_list.setCurrentRow(row)

    # ------------------------------------------------------------------ #
    def show_recruitment(self) -> None:
        if self.results is None:
            return
        table = self.results.recruitment_table()
        if table.empty:
            QMessageBox.information(self, "Recruitment", "No detections in this run.")
            return
        QMessageBox.information(self, "Recruitment table", table.to_string(index=False, max_rows=60))

"""Place one curve's onset, P1 and P2 by hand.

Opened from the scenario tab on a curve the detector doubts (clicking its point or
its line in the list), or on any curve with the "Поставить пики вручную" button.
The curve is drawn at its own sampling rate, the channel's other curves in grey
behind it, and its current markers where the detector put them (or, for a curve
without a response, where the channel's responses have theirs).

A click moves the chosen marker and passes on to the next one, so onset, P1, P2
are three clicks. P1 and P2 snap to the extremum within ±0.5 ms on the side of the
click — above zero the maximum, below it the minimum — because a peak is placed
on a peak and a click is never that exact; untick the box to place them freely.
"Ответа нет" marks the curve as a false response instead of placing anything.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout,
)

from .plot_canvas import C_ONSET, C_P1, C_P2

MARKERS = (("onset", "onset", C_ONSET), ("p1", "P1", C_P1), ("p2", "P2", C_P2))
#: Half-width of the snap to the extremum, ms.
SNAP_MS = 0.5
#: How far the view reaches, ms: the network reads 60 ms, responses end well before.
VIEW_MS = 60.0
#: The doctors' peak-to-peak threshold, for the warning only (the hand markers are kept).
PTP_MIN_UV = 30.0

HAND = "hand"
FALSE = "false"


class CurvePeakEditor(QDialog):
    """``t_ms`` (n,), ``curve_uv`` (n,), ``others_uv`` (m, n) in µV; ``initial_ms`` by marker."""

    def __init__(self, t_ms, curve_uv, others_uv, initial_ms: dict, title: str,
                 note: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Пики вручную — {title}")
        self.resize(980, 600)
        self.t = np.asarray(t_ms, dtype=float)
        self.y = np.asarray(curve_uv, dtype=float)
        self.others = np.asarray(others_uv, dtype=float) if others_uv is not None else None
        self.ms = {k: (float(initial_ms[k]) if initial_ms.get(k) is not None
                       and np.isfinite(initial_ms[k]) else None) for k, _, _ in MARKERS}
        self.result_kind: str | None = None

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Клик по кривой ставит:"))
        self.radios: dict[str, QRadioButton] = {}
        for key, label, colour in MARKERS:
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color: {colour};")
            self.radios[key] = rb
            bar.addWidget(rb)
        self.radios["onset" if self.ms["onset"] is None else "p1"].setChecked(True)
        self.snap = QCheckBox(f"P1/P2 к экстремуму ±{SNAP_MS:g} мс")
        self.snap.setChecked(True)
        bar.addSpacing(16)
        bar.addWidget(self.snap)
        bar.addStretch()

        self.note = QLabel(note)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #7b3294;")
        self.note.setVisible(bool(note))
        self.info = QLabel()
        self.info.setWordWrap(True)

        self.fig = Figure(figsize=(9, 4.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.btn_save = QPushButton("Сохранить пики")
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._save)
        btn_none = QPushButton("Ответа нет")
        btn_none.setToolTip("Отметить кривую как ложный ответ вместо пиков.")
        btn_none.clicked.connect(self._no_response)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch()
        for b in (btn_none, btn_cancel, self.btn_save):
            buttons.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addWidget(self.note)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.info)
        layout.addLayout(buttons)
        self._draw()

    # ------------------------------------------------------------------ #
    def _value(self, ms) -> float:
        if ms is None:
            return np.nan
        return float(self.y[int(np.argmin(np.abs(self.t - ms)))])

    def _draw(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        view = (self.t >= 0) & (self.t <= VIEW_MS)
        if self.others is not None and len(self.others):
            ax.plot(self.t[view], self.others[:, view].T, color="0.82", lw=0.5, zorder=1)
        ax.plot(self.t[view], self.y[view], color="black", lw=1.5, zorder=3)
        for key, label, colour in MARKERS:
            ms = self.ms[key]
            if ms is None:
                continue
            ax.axvline(ms, color=colour, lw=0.8, ls="--", alpha=0.6, zorder=2)
            ax.plot([ms], [self._value(ms)], "o", color=colour, ms=8, zorder=5)
            below = self._value(ms) < 0      # a negative peak is labelled under it
            ax.annotate(f"{label}\n{ms:.2f} мс", (ms, self._value(ms)),
                        textcoords="offset points", xytext=(0, -26 if below else 10),
                        ha="center", fontsize=8, color=colour)
        # Scaled past the stimulus artifact, as on the scenario tab: it is an order of
        # magnitude larger than the response and would flatten it.
        post = view & (self.t >= 2.0)
        seg = self.y[post]
        if self.others is not None and len(self.others):
            seg = np.concatenate([seg, self.others[:, post].ravel()])
        if seg.size and np.isfinite(seg).any():
            lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))
            pad = 0.08 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlim(0, min(VIEW_MS, float(self.t[-1])))
        ax.set_xlabel("ms from stimulus")
        ax.set_ylabel("µV")
        ax.grid(True, color="0.92", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        self.canvas.draw_idle()
        self._update_info()

    def _update_info(self) -> None:
        on, p1, p2 = self.ms["onset"], self.ms["p1"], self.ms["p2"]
        v1, v2 = self._value(p1), self._value(p2)
        bits = []
        if on is not None:
            bits.append(f"onset {on:.2f} мс")
        if p1 is not None:
            bits.append(f"P1 {p1:.2f} мс, {v1:.0f} мкВ")
        if p2 is not None:
            bits.append(f"P2 {p2:.2f} мс, {v2:.0f} мкВ")
        warn = []
        if np.isfinite(v1) and np.isfinite(v2):
            ptp = abs(v1 - v2)
            bits.append(f"размах {ptp:.0f} мкВ")
            if ptp < PTP_MIN_UV:
                warn.append(f"размах меньше {PTP_MIN_UV:g} мкВ — по критериям врачей это не "
                            f"ответ, но пики сохранятся так, как поставлены")
            if np.sign(v1) == np.sign(v2):
                warn.append("P1 и P2 одного знака")
        order = [x for x in (on, p1, p2) if x is not None]
        if order != sorted(order):
            warn.append("порядок нарушен: ожидается onset ≤ P1 ≤ P2")
        if p1 is None or p2 is None:
            warn.append("поставьте P1 и P2")
        text = " · ".join(bits) or "пики не поставлены"
        if warn:
            text += "<br><span style='color:#b00'>⚠ " + "; ".join(warn) + "</span>"
        self.info.setText(text)
        self.btn_save.setEnabled(p1 is not None and p2 is not None)

    # ------------------------------------------------------------------ #
    def _which(self) -> str:
        return next(k for k, _, _ in MARKERS if self.radios[k].isChecked())

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        key = self._which()
        ms = float(event.xdata)
        i = int(np.argmin(np.abs(self.t - ms)))
        if key in ("p1", "p2") and self.snap.isChecked():
            w = np.flatnonzero(np.abs(self.t - self.t[i]) <= SNAP_MS)
            if w.size:
                above = event.ydata is not None and float(event.ydata) >= 0
                i = int(w[np.argmax(self.y[w])] if above else w[np.argmin(self.y[w])])
        self.ms[key] = float(self.t[i])
        # onset -> P1 -> P2: three clicks place a curve
        keys = [k for k, _, _ in MARKERS]
        nxt = keys[min(keys.index(key) + 1, len(keys) - 1)]
        self.radios[nxt].setChecked(True)
        self._draw()

    def _save(self) -> None:
        self.result_kind = HAND
        self.accept()

    def _no_response(self) -> None:
        self.result_kind = FALSE
        self.accept()

    def markers_ms(self) -> dict:
        """The placed onset/P1/P2 in ms (None where not placed)."""
        return dict(self.ms)

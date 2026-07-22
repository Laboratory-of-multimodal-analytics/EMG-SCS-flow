"""Antagonist co-activation review: the L-shape plots, made interactive.

Each point is one merged burst episode for an antagonist muscle pair (see
`_build_coactivation_episodes` in the pipeline), plotted at its
(RMS_A, RMS_B), normalised per pair. Clicking a point shows that episode's
envelope chunk on the right, both muscles, with the burst window shaded —
so the L-shape point and the burst it came from can be checked against
each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from ..results import SpontaneousResults

N_COLS = 2


class CoactivationViewer(QWidget):
    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.results: SpontaneousResults | None = None
        self.condition = ""
        self.episodes = pd.DataFrame()
        self.envelopes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # id(scatter artist) -> (pair_df, axes, muscle_a, muscle_b)
        self._artist_info: dict[int, tuple[pd.DataFrame, object, str, str]] = {}
        self._highlight = None  # the marker artist drawn over the selected point

        self.l_fig = Figure(figsize=(7, 7), layout="constrained")
        self.l_canvas = FigureCanvasQTAgg(self.l_fig)
        self.l_canvas.mpl_connect("pick_event", self._on_pick)

        self.detail_fig = Figure(figsize=(7, 4), layout="constrained")
        self.detail_canvas = FigureCanvasQTAgg(self.detail_fig)

        self.info_label = QLabel("Click a point on the left to see its burst episode.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 4px;")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>L-shape: antagonist co-activation episodes</b>"))
        lv.addWidget(self.l_canvas, 1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("<b>Selected episode</b>"))
        rv.addWidget(self.info_label)
        rv.addWidget(self.detail_canvas, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([700, 700])
        root = QVBoxLayout(self)
        root.addWidget(split)

    # ------------------------------------------------------------------ #
    def load(self, results: SpontaneousResults) -> None:
        self.results = results

    def set_condition(self, condition: str) -> None:
        if not condition or self.results is None:
            return
        self.condition = condition
        self.episodes = self.results.coactivation_episodes(condition)
        self.envelopes = self.results.envelopes_on_segment(condition)
        self._highlight = None
        self._draw_l_shapes()
        self._draw_detail(None)

    # ------------------------------------------------------------------ #
    def _draw_l_shapes(self) -> None:
        self.l_fig.clear()
        self._artist_info.clear()
        df = self.episodes

        if df.empty:
            ax = self.l_fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No antagonist co-activation episodes for this condition.",
                ha="center", va="center", color="gray",
            )
            ax.set_axis_off()
            self.l_canvas.draw_idle()
            return

        pairs = list(dict.fromkeys(df["Pair"]))
        n_pairs = len(pairs)
        n_cols = N_COLS
        n_rows = int(np.ceil(n_pairs / n_cols))
        axes = self.l_fig.subplots(n_rows, n_cols, squeeze=False)

        for idx, pair_name in enumerate(pairs):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            pair_df = df[df["Pair"] == pair_name].dropna(
                subset=["RMS_A_uV", "RMS_B_uV"],
            ).reset_index(drop=True)
            if pair_df.empty:
                ax.set_visible(False)
                continue

            muscle_a = pair_df["Muscle_A"].iloc[0]
            muscle_b = pair_df["Muscle_B"].iloc[0]
            max_a = pair_df["RMS_A_uV"].max()
            max_b = pair_df["RMS_B_uV"].max()
            xs = pair_df["RMS_A_uV"] / max_a if max_a > 0 else pair_df["RMS_A_uV"] * 0.0
            ys = pair_df["RMS_B_uV"] / max_b if max_b > 0 else pair_df["RMS_B_uV"] * 0.0

            artist = ax.scatter(
                xs, ys, s=28, alpha=0.55, color="#1f77b4", edgecolors="none", picker=5,
            )
            self._artist_info[id(artist)] = (pair_df, ax, muscle_a, muscle_b)

            max_val = max(float(xs.max()), float(ys.max()))
            limit = max_val * 1.05 if max_val > 0 else 1.0
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_aspect("equal", "box")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xlabel(f"{muscle_a} RMS (norm.)")
            ax.set_ylabel(f"{muscle_b} RMS (norm.)")
            ax.set_title(f"{muscle_a} vs {muscle_b}", fontsize=9)

        for j in range(n_pairs, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axes[row][col].set_visible(False)

        self.l_canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _on_pick(self, event) -> None:
        info = self._artist_info.get(id(event.artist))
        if info is None or len(event.ind) == 0:
            return
        pair_df, ax, muscle_a, muscle_b = info
        i = int(event.ind[0])
        if i >= len(pair_df):
            return
        row = pair_df.iloc[i]

        max_a = pair_df["RMS_A_uV"].max()
        max_b = pair_df["RMS_B_uV"].max()
        xi = row["RMS_A_uV"] / max_a if max_a > 0 else 0.0
        yi = row["RMS_B_uV"] / max_b if max_b > 0 else 0.0
        self._highlight_point(ax, xi, yi)
        self._draw_detail(row)

    def _highlight_point(self, ax, xi: float, yi: float) -> None:
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None
        self._highlight = ax.scatter(
            [xi], [yi], s=160, facecolors="none", edgecolors="#d62728",
            linewidths=2, zorder=5,
        )
        self.l_canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _draw_detail(self, row) -> None:
        self.detail_fig.clear()
        ax = self.detail_fig.add_subplot(111)

        if row is None:
            ax.text(0.5, 0.5, "Click a point on the left.", ha="center", va="center", color="gray")
            ax.set_axis_off()
            self.info_label.setText("Click a point on the left to see its burst episode.")
            self.detail_canvas.draw_idle()
            return

        muscle_a, muscle_b = str(row["Muscle_A"]), str(row["Muscle_B"])
        t0, t1 = float(row["Start_s"]), float(row["End_s"])
        pad = max(0.3 * (t1 - t0), 0.5)
        win0, win1 = t0 - pad, t1 + pad

        drew_any = False
        for muscle, color in ((muscle_a, "#1f77b4"), (muscle_b, "#ff7f0e")):
            env = self.envelopes.get(muscle)
            if env is None:
                continue
            t, v = env
            mask = (t >= win0) & (t <= win1)
            if not np.any(mask):
                continue
            ax.plot(t[mask], v[mask], color=color, lw=1.3, label=muscle)
            drew_any = True

        ax.axvspan(t0, t1, color="#d62728", alpha=0.15, label="burst episode")
        ax.set_xlim(win0, win1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("RMS (µV)")
        if drew_any:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        else:
            ax.text(
                0.5, 0.5, "Envelope not exported for these channels.",
                ha="center", va="center", color="gray", transform=ax.transAxes,
            )
        ax.set_title(f"{muscle_a} vs {muscle_b} — episode {int(row['Episode'])}", fontsize=10)
        self.detail_canvas.draw_idle()

        self.info_label.setText(
            f"<b>{muscle_a} vs {muscle_b}</b> — episode {int(row['Episode'])}: "
            f"{t0:.3f}–{t1:.3f} s (duration {t1 - t0:.3f} s)<br>"
            f"RMS: {muscle_a} = {row['RMS_A_uV']:.1f} µV, {muscle_b} = {row['RMS_B_uV']:.1f} µV"
        )

"""Read-only gallery for PNG plots + CSV tables the pipeline wrote.

Used by the Condition-test surface, which is a folder of PNGs and CSVs. The
Neurosoft scenarios moved off this widget onto an interactive surface
(widgets/scenario_viewer.py) — browsing finished images is fine for a
per-condition waterfall, but not for picking curves out of a sweep.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QLabel, QListWidget, QListWidgetItem, QScrollArea,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class ResultsGallery(QWidget):
    """Left: a grouped list of PNGs/CSVs. Right: the selected image or table."""

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None

        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentItemChanged.connect(self._on_item)

        self.image_label = QLabel("Nothing to show yet.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.image_scroll)   # index 0
        self.stack.addWidget(self.table)          # index 1

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.stack)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 1000])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    # -- population -------------------------------------------------------- #
    def set_sections(self, sections: list[tuple[str, list[tuple[str, Path, str]]]]) -> None:
        """sections = [(header, [(label, path, kind)])], kind in {'image','csv'}."""
        self.list.blockSignals(True)
        self.list.clear()
        first_row = None
        for header, entries in sections:
            if not entries:
                continue
            head = QListWidgetItem(header)
            head.setFlags(Qt.NoItemFlags)             # a non-selectable group title
            f = head.font(); f.setBold(True); head.setFont(f)
            head.setForeground(Qt.gray)
            self.list.addItem(head)
            for label, path, kind in entries:
                it = QListWidgetItem("   " + label)
                it.setData(Qt.UserRole, (str(path), kind))
                self.list.addItem(it)
                if first_row is None:
                    first_row = self.list.row(it)
        self.list.blockSignals(False)
        if first_row is not None:
            self.list.setCurrentRow(first_row)
        else:
            self._pixmap = None
            self.image_label.setText("No outputs found in this folder.")
            self.stack.setCurrentIndex(0)

    # -- display ----------------------------------------------------------- #
    def _on_item(self, cur: QListWidgetItem | None, _prev=None) -> None:
        if cur is None:
            return
        data = cur.data(Qt.UserRole)
        if not data:
            return
        path, kind = data
        if kind == "image":
            self._show_image(Path(path))
        else:
            self._show_csv(Path(path))

    def _show_image(self, path: Path) -> None:
        pm = QPixmap(str(path))
        self._pixmap = pm if not pm.isNull() else None
        self.stack.setCurrentIndex(0)
        self._rescale()
        if self._pixmap is None:
            self.image_label.setText(f"Could not load {path.name}")

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        w = max(self.image_scroll.viewport().width() - 4, 100)
        scaled = self._pixmap.scaledToWidth(
            min(w, self._pixmap.width()), Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _show_csv(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.stack.setCurrentIndex(0)
            self.image_label.setText(f"Could not read {path.name}: {exc}")
            self._pixmap = None
            return
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))
        self.table.resizeColumnsToContents()
        self.stack.setCurrentIndex(1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.stack.currentIndex() == 0:
            self._rescale()


def _images(folder: Path) -> list[tuple[str, Path, str]]:
    if not folder.is_dir():
        return []
    return [(p.name, p, "image") for p in sorted(folder.glob("*.png"))]


def _csvs(folder: Path) -> list[tuple[str, Path, str]]:
    if not folder.is_dir():
        return []
    return [(p.name, p, "csv") for p in sorted(folder.glob("*.csv"))]


class ConditionViewer(QWidget):
    """Condition-test outputs: waterfalls, amplitude-vs-condition, tables."""

    def __init__(self, session=None) -> None:
        super().__init__()
        self.gallery = ResultsGallery()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.gallery)

    def load(self, root: Path) -> None:
        from ..results import COND_DIR, mode_dir
        base = mode_dir(Path(root), COND_DIR)
        self.gallery.set_sections([
            ("Amplitude vs condition", _images(base / "Amplitude vs condition")),
            ("Curves per condition", _images(base / "Curves per condition")),
            ("Waterfall", _images(base / "Waterfall")),
            ("Tables (Excel)", _csvs(base / "Excel")),
        ])

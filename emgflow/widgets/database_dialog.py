"""Browse a results tree and open an already-processed recording without re-running it."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..results import ProcessedRun, scan_processed


class _ScanWorker(QThread):
    done = Signal(object)

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.base = base

    def run(self) -> None:  # noqa: D102
        try:
            self.done.emit(scan_processed(self.base))
        except Exception:
            self.done.emit([])


class DatabaseDialog(QDialog):
    """Lists every output root under a folder that already holds results."""

    def __init__(self, parent=None, base: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processed recordings")
        self.resize(1000, 560)
        self.selected: ProcessedRun | None = None
        self._runs: list[ProcessedRun] = []
        self._worker: _ScanWorker | None = None

        self.path_label = QLabel(str(base) if base else "No folder chosen")
        self.path_label.setStyleSheet("color: gray;")

        choose = QPushButton("Choose folder…")
        choose.clicked.connect(self.choose_folder)
        self.rescan_btn = QPushButton("Rescan")
        self.rescan_btn.clicked.connect(lambda: self.scan(Path(self.path_label.text())))

        top = QHBoxLayout()
        top.addWidget(choose)
        top.addWidget(self.rescan_btn)
        top.addWidget(self.path_label, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Recording", "Mode", "Crops / conditions", "Metrics table", "Spontaneous", "Last run"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self.accept_selection)

        self.open_btn = QPushButton("Open results")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.accept_selection)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.table.itemSelectionChanged.connect(
            lambda: self.open_btn.setEnabled(bool(self.table.selectedItems()))
        )

        bottom = QHBoxLayout()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: gray;")
        bottom.addWidget(self.count_label, 1)
        bottom.addWidget(self.open_btn)
        bottom.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(self.table, 1)
        layout.addLayout(bottom)

        if base:
            self.scan(Path(base))

    # ------------------------------------------------------------------ #
    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder to scan for results")
        if path:
            self.scan(Path(path))

    def scan(self, base: Path) -> None:
        if not base.exists():
            return
        self.path_label.setText(str(base))
        self.progress.show()
        self.rescan_btn.setEnabled(False)
        self.count_label.setText("Scanning…")
        self._worker = _ScanWorker(base)
        self._worker.done.connect(self._populate)
        self._worker.start()

    def _populate(self, runs: list[ProcessedRun]) -> None:
        self.progress.hide()
        self.rescan_btn.setEnabled(True)
        self._runs = runs
        self.table.setRowCount(len(runs))
        for r, run in enumerate(runs):
            when = _dt.datetime.fromtimestamp(run.mtime).strftime("%Y-%m-%d %H:%M")
            cells = [
                run.name,
                run.mode_label,
                f"{run.n_items} {run.items_label}",
                f"{run.size_mb:.0f} MB" if run.size_mb else "—",
                "yes" if run.has_spontaneous else "—",
                when,
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(str(run.root))
                if run.n_items == 0:
                    item.setForeground(Qt.red)
                self.table.setItem(r, c, item)
        self.count_label.setText(
            f"{len(runs)} processed recording(s)" if runs
            else "Nothing processed found under this folder."
        )

    def accept_selection(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return
        self.selected = self._runs[min(rows)]
        self.accept()

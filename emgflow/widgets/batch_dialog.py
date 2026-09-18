"""Process several recordings one after another (File → Process several recordings…).

Files are added one by one or by folder, each row says what the file is and
whether it already has results, and Run processes the ticked rows in order with
the pipeline the Run button uses (see ``emgflow/batch.py`` for the settings each
file gets). A file that fails is marked and the batch goes on. "Stop after this
file" ends the batch between files: a run cannot be cut halfway without leaving
half-written outputs. The window can be closed while the batch runs; it keeps
going and the progress bar of the main window shows it.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from ..batch import (BatchItem, describe_item, find_recordings, mark_duplicates, output_root,
                     processed_mode, session_for)

MODES = [("Stimulation-induced (SIR)", "sir"), ("StartStop", "startstop")]
_MODE_INDEX = {m[1]: i for i, m in enumerate(MODES)}
COL_CHECK, COL_FILE, COL_KIND, COL_MODE, COL_DONE, COL_STATUS = range(6)
_GREEN, _RED, _GREY = QColor("#1b7837"), QColor("#b2182b"), QColor("#777777")


class _AddThread(QThread):
    """Finds and describes files off the disk: a Drive file is fetched to answer
    what it is, so this never runs on the GUI thread."""
    done = Signal(object)

    def __init__(self, default_mode: str, paths: list[Path] | None = None,
                 folder: Path | None = None) -> None:
        super().__init__()
        self.default_mode = default_mode
        self.paths = paths or []
        self.folder = folder

    def run(self) -> None:  # noqa: D102
        try:
            paths = find_recordings(self.folder) if self.folder is not None else self.paths
            self.done.emit([describe_item(p, self.default_mode) for p in paths])
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.done.emit(exc)


class BatchDialog(QDialog):
    """A queue of recordings processed one after another."""

    def __init__(self, main) -> None:
        super().__init__(main)
        self.main = main
        self.controller = main.controller
        self.items: list[BatchItem] = []
        self._queue: list[int] = []          # rows of the running batch, in run order
        self._running = False
        self._stopping = False
        self._t0 = 0.0
        self._adder: _AddThread | None = None
        self.setWindowTitle("Process several recordings")
        self.resize(1150, 620)

        add_files = QPushButton("Add files…")
        add_files.clicked.connect(self.add_files)
        add_folder = QPushButton("Add folder…")
        add_folder.setToolTip("Every recording under the folder: .mat, .fif, .edf and Neurosoft .txt "
                              "exports. Results folders are skipped.")
        add_folder.clicked.connect(self.add_folder)
        self.clear_btn = QPushButton("Clear list")
        self.clear_btn.clicked.connect(self.clear)
        self.mode_box = QComboBox()
        self.mode_box.addItems([m[0] for m in MODES])
        self.mode_box.setToolTip("Mode for .mat / .fif / .edf recordings; every row that has not "
                                 "run yet follows it. Neurosoft .txt exports always run as SIR.")
        self.mode_box.currentIndexChanged.connect(self._mode_for_all)
        self.skip_done = QCheckBox("Skip recordings that already have results")
        self.skip_done.setChecked(True)
        self.where_box = QComboBox()
        self.where_box.addItems(["Results next to each recording", "Results into one folder…"])
        self.where_box.currentIndexChanged.connect(self._on_where)
        self.out_base: Path | None = None

        top = QHBoxLayout()
        for w in (add_files, add_folder, self.clear_btn):
            top.addWidget(w)
        top.addSpacing(16)
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.mode_box)
        top.addSpacing(16)
        top.addWidget(self.skip_done)
        top.addStretch(1)
        top.addWidget(self.where_box)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["", "Recording", "Kind", "Mode", "Results already",
                                              "Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        for col, width in ((COL_CHECK, 28), (COL_KIND, 110), (COL_MODE, 190), (COL_DONE, 120),
                           (COL_STATUS, 260)):
            self.table.setColumnWidth(col, width)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self.table.doubleClicked.connect(lambda _: self.open_selected())

        self.note = QLabel(
            "Settings: a recording processed before re-runs with its own saved settings and hand "
            "edits; any other takes the Settings tab (without the hand edits of the recording on "
            "screen). Neurosoft exports get their windows fitted to the file and keep the scenario "
            "of their previous run.")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #555; font-size: 11px;")

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v of %m recordings")
        self.progress.hide()
        self.current = QLabel("")
        self.current.setStyleSheet("color: #555;")

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop after this recording")
        self.stop_btn.clicked.connect(self.stop)
        self.open_btn = QPushButton("Open results")
        self.open_btn.clicked.connect(self.open_selected)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        bottom = QHBoxLayout()
        bottom.addWidget(self.current, 1)
        for w in (self.open_btn, self.stop_btn, self.run_btn, close):
            bottom.addWidget(w)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.note)
        lay.addWidget(self.progress)
        lay.addLayout(bottom)

        self.controller.batch_file_started.connect(self._on_file_started)
        self.controller.batch_file_done.connect(self._on_file_done)
        self.controller.batch_finished.connect(self._on_finished)
        self.controller.progress.connect(self._on_progress)
        self.controller.busy_changed.connect(lambda _: self._sync_buttons())
        self._sync_buttons()

    # ------------------------------------------------------------------ #
    # The list
    # ------------------------------------------------------------------ #
    def _default_mode(self) -> str:
        return MODES[self.mode_box.currentIndex()][1]

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Recordings to process", "",
            "Recordings (*.mat *.fif *.edf *.txt);;All files (*)")
        if paths:
            self._add(_AddThread(self._default_mode(), paths=[Path(p) for p in paths]))

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder with recordings")
        if folder:
            self._add(_AddThread(self._default_mode(), folder=Path(folder)))

    def _add(self, thread: _AddThread) -> None:
        if self._adder is not None and self._adder.isRunning():
            return
        self._adder = thread
        thread.done.connect(self._on_added)
        self.current.setText("Reading the files…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        thread.start()

    def _on_added(self, found) -> None:
        QApplication.restoreOverrideCursor()
        if isinstance(found, Exception):
            self.current.setText("")
            QMessageBox.warning(self, "Could not read the files", str(found))
            return
        known = {it.path for it in self.items}
        new = [it for it in found if it.path not in known]
        self.items.extend(new)
        mark_duplicates(self.items, self.out_base)
        self._fill()
        n_off = sum(not it.supported for it in new)
        self.current.setText(f"Added {len(new)} recording(s)"
                             + (f", {n_off} cannot be processed (see Status)" if n_off else "")
                             + (f"; {len(found) - len(new)} already in the list"
                                if len(found) > len(new) else "") + ".")

    def clear(self) -> None:
        if self._running:
            return
        self.items = []
        self._fill()
        self.current.setText("")

    def _fill(self) -> None:
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(len(self.items))
        for row, it in enumerate(self.items):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
                           if it.supported else Qt.ItemIsUserCheckable)
            previous = self.table.item(row, COL_CHECK)
            ticked = previous.checkState() == Qt.Checked if previous is not None else it.supported
            check.setCheckState(Qt.Checked if ticked and it.supported else Qt.Unchecked)
            self.table.setItem(row, COL_CHECK, check)
            name = QTableWidgetItem(it.path.name)
            name.setToolTip(str(it.path))
            self.table.setItem(row, COL_FILE, name)
            self.table.setItem(row, COL_KIND, QTableWidgetItem(it.kind))
            self._set_mode_cell(row, it)
            self.table.setItem(row, COL_DONE, QTableWidgetItem(
                (it.done_mode or "").replace("sir", "SIR").replace("startstop", "StartStop")
                .replace("condition", "Condition test") or "—"))
            self._set_status(row)
        self.table.setUpdatesEnabled(True)
        self._sync_buttons()

    def _set_mode_cell(self, row: int, it: BatchItem) -> None:
        if it.neurosoft or not it.supported:
            self.table.removeCellWidget(row, COL_MODE)
            self.table.setItem(row, COL_MODE, QTableWidgetItem(
                "SIR (Neurosoft, scenario auto)" if it.neurosoft else ""))
            return
        box = self.table.cellWidget(row, COL_MODE)
        if not isinstance(box, QComboBox):
            box = QComboBox()
            box.addItems([m[0] for m in MODES])
            box.currentIndexChanged.connect(
                lambda i, it=it: setattr(it, "mode", MODES[i][1]))
            self.table.setCellWidget(row, COL_MODE, box)
        box.blockSignals(True)
        box.setCurrentIndex(_MODE_INDEX.get(it.mode, 0))
        box.blockSignals(False)
        box.setEnabled(not self._running)

    def _mode_for_all(self) -> None:
        mode = self._default_mode()
        for row, it in enumerate(self.items):
            if it.supported and not it.neurosoft and it.status in ("waiting", "skipped"):
                it.mode = mode
                self._set_mode_cell(row, it)

    def _on_where(self, index: int) -> None:
        if index == 1:
            folder = QFileDialog.getExistingDirectory(self, "Folder for all the results")
            if not folder:
                self.where_box.setCurrentIndex(0)
                return
            self.out_base = Path(folder)
            self.where_box.setItemText(1, f"Results into {self.out_base.name}/")
            self.where_box.setToolTip(str(self.out_base))
        else:
            self.out_base = None
            self.where_box.setItemText(1, "Results into one folder…")
            self.where_box.setToolTip("")
        for it in self.items:
            if it.supported:
                it.done_mode = processed_mode(output_root(it.path, self.out_base))
        self._fill()

    def _set_status(self, row: int) -> None:
        it = self.items[row]
        text, color, tip = it.status, None, it.error or it.note
        if not it.supported:
            text, color = f"not processed: {it.note}", _GREY
        elif it.status.startswith("done"):
            color = _GREEN
        elif it.status.startswith("failed"):
            color = _RED
        elif it.status.startswith(("skipped", "not run")):
            color = _GREY
        item = QTableWidgetItem(text)
        if color is not None:
            item.setForeground(color)
        if tip:
            item.setToolTip(tip)
        self.table.setItem(row, COL_STATUS, item)

    def _checked(self, row: int) -> bool:
        item = self.table.item(row, COL_CHECK)
        return item is not None and item.checkState() == Qt.Checked

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self.controller.busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        rows, sessions = [], []
        for row, it in enumerate(self.items):
            if not it.supported or not self._checked(row):
                continue
            if self.skip_done.isChecked() and it.done_mode:
                it.status, it.error = "skipped: has results", ""
                self._set_status(row)
                continue
            try:
                sessions.append(session_for(it, self.main.session, self.out_base))
            except Exception as exc:                   # noqa: BLE001 - shown in the row
                it.status, it.error = f"failed: {exc}", repr(exc)
                self._set_status(row)
                continue
            it.status, it.error, it.output = "waiting", "", None
            self._set_status(row)
            rows.append(row)
        if not rows:
            QMessageBox.information(self, "Nothing to run",
                                    "No ticked recording is left to process"
                                    + (" (the ones with results are skipped)."
                                       if self.skip_done.isChecked() else "."))
            return
        self._queue, self._running, self._stopping = rows, True, False
        self.progress.setRange(0, len(rows))
        self.progress.setValue(0)
        self.progress.show()
        self.main.log_view.clear()
        self.main.log_dock.show()
        if not self.controller.start_batch(sessions):
            self._running = False
            QMessageBox.information(self, "Busy", "A run is already in progress.")
        self._sync_buttons()

    def stop(self) -> None:
        if self._running:
            self._stopping = True
            self.controller.stop_batch()
            self.current.setText(self.current.text() + "  — stopping after this recording")
            self._sync_buttons()

    def _on_file_started(self, i: int) -> None:
        row = self._queue[i]
        it = self.items[row]
        it.status = "running…"
        self._set_status(row)
        self.table.selectRow(row)
        self._t0 = time.time()
        self.current.setText(f"{i + 1} of {len(self._queue)}: {it.path.name}")

    def _on_progress(self, phase: str, n: int, total: int) -> None:
        if self._running and self._queue and total > 0:
            base = self.current.text().split("   [")[0]
            self.current.setText(f"{base}   [{phase} {n}/{total}]")

    def _on_file_done(self, i: int, out, error: str) -> None:
        row = self._queue[i]
        it = self.items[row]
        seconds = time.time() - self._t0
        if error:
            last = error.strip().splitlines()[-1] if error.strip() else "error"
            it.status, it.error = f"failed: {last}", error
        else:
            it.status, it.output, it.error = f"done in {seconds:.0f} s", Path(out), ""
            it.done_mode = processed_mode(Path(out))
        self._set_status(row)
        self.table.setItem(row, COL_DONE, QTableWidgetItem(
            (it.done_mode or "—").replace("sir", "SIR").replace("startstop", "StartStop")))
        self.progress.setValue(i + 1)

    def _on_finished(self) -> None:
        for row in self._queue:
            it = self.items[row]
            if it.status in ("waiting", "running…"):
                it.status = "not run (stopped)"
                self._set_status(row)
        done = sum(self.items[r].status.startswith("done") for r in self._queue)
        failed = sum(self.items[r].status.startswith("failed") for r in self._queue)
        stopped = sum(self.items[r].status.startswith("not run") for r in self._queue)
        self._running = False
        self.current.setText(f"Finished: {done} done, {failed} failed"
                             + (f", {stopped} not run (stopped)" if stopped else "")
                             + ". Double-click a row to open its results.")
        for row, it in enumerate(self.items):
            self._set_mode_cell(row, it)
        self._sync_buttons()
        if failed:
            self.show()
            self.raise_()

    # ------------------------------------------------------------------ #
    def _selected_root(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        it = self.items[rows[0].row()]
        if it.output is not None:
            return it.output
        root = output_root(it.path, self.out_base) if it.supported else None
        return root if root is not None and processed_mode(root) else None

    def open_selected(self) -> None:
        root = self._selected_root()
        if root is not None:
            self.main.load_results(root)
            self.main.raise_()

    def _sync_buttons(self) -> None:
        busy = self.controller.busy
        self.run_btn.setEnabled(not busy and any(it.supported for it in self.items))
        self.stop_btn.setEnabled(self._running and not self._stopping)
        self.clear_btn.setEnabled(not self._running)
        self.open_btn.setEnabled(self._selected_root() is not None)
        if not self._running and not busy:
            self.progress.setVisible(self.progress.value() > 0)

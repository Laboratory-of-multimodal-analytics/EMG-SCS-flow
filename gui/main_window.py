"""Application shell: file, mode switch, settings, the three review surfaces, log."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QWidget, QVBoxLayout,
)

from .results import SIRResults, SpontaneousResults, StartStopResults
from .review_store import ReviewStore
from .runner import RunController
from .session import Session
from .widgets.settings_panel import SettingsPanel
from .widgets.sir_viewer import SIRViewer
from .widgets.spontaneous_viewer import SpontaneousViewer
from .widgets.startstop_viewer import StartStopViewer

MODES = [("Stimulation-induced (SIR)", "sir"), ("StartStop", "startstop")]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EMG-SCS-flow")
        self.resize(1500, 950)

        self.session = Session()
        self.controller = RunController()
        self.controller.log.connect(self._log)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.busy_changed.connect(self._on_busy)
        self.store: ReviewStore | None = None

        # ---- top bar ----
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold;")

        open_btn = QPushButton("Open recording…")
        open_btn.clicked.connect(self.open_file)

        self.mode_box = QComboBox()
        self.mode_box.addItems([m[0] for m in MODES])
        self.mode_box.currentIndexChanged.connect(self._on_mode)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        self.run_btn.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.hide()
        self.progress.setMaximumWidth(160)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.file_label, 1)
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.mode_box)
        top.addWidget(self.run_btn)
        top.addWidget(self.progress)

        # ---- tabs ----
        self.settings_panel = SettingsPanel(self.session)
        self.sir_viewer = SIRViewer(self.session)
        self.sir_viewer.rerun_requested.connect(self.run)
        self.startstop_viewer = StartStopViewer(self.session)
        self.spontaneous_viewer = SpontaneousViewer(self.session)
        self.spontaneous_viewer.rerun_requested.connect(self.run)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.settings_panel, "Settings")
        self.tabs.addTab(self.sir_viewer, "Crop review (SIR)")
        self.tabs.addTab(self.startstop_viewer, "Epoch browser (StartStop)")
        self.tabs.addTab(self.spontaneous_viewer, "Spontaneous EMG")

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addLayout(top)
        cv.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # ---- log dock ----
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        self._build_menu()
        self._sync_tabs()

    # ------------------------------------------------------------------ #
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot in (
            ("Open recording…", self.open_file),
            ("Load session…", self.load_session),
            ("Save session…", self.save_session),
            ("Export runner script…", self.export_runner),
            ("Export reviewed metrics…", self.export_reviewed_metrics),
        ):
            act = QAction(text, self)
            act.triggered.connect(slot)
            file_menu.addAction(act)

        view_menu = self.menuBar().addMenu("&View")
        act = QAction("Recruitment table", self)
        act.triggered.connect(self.sir_viewer.show_recruitment)
        view_menu.addAction(act)

    def _log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # ------------------------------------------------------------------ #
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "", "Recordings (*.mat *.fif *.edf);;All files (*)"
        )
        if not path:
            return
        self.session.input_path = Path(path)
        # Default next to the input, mirroring the pipeline's own convention.
        self.session.output_dir = Path(path).parent / "results" / Path(path).stem
        self.file_label.setText(str(path))
        self.run_btn.setEnabled(True)
        self._log(f"Loaded {path}")
        self._log(f"Outputs will go to {self.session.output_dir}")

    def _on_mode(self, idx: int) -> None:
        self.session.set_mode(MODES[idx][1])
        self.settings_panel.rebuild()
        self._sync_tabs()

    def _sync_tabs(self) -> None:
        """Grey out the surfaces the current mode cannot produce."""
        sir = self.session.mode == "sir"
        self.tabs.setTabEnabled(1, sir)
        self.tabs.setTabEnabled(2, not sir)
        self.tabs.setTabEnabled(3, not sir)  # spontaneous only runs inside StartStop
        self.tabs.setCurrentIndex(0 if not self._has_results() else (1 if sir else 2))

    def _has_results(self) -> bool:
        return self.session.output_dir is not None and Path(self.session.output_dir).exists()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if self.session.input_path is None:
            QMessageBox.information(self, "No file", "Open a recording first.")
            return
        if self.controller.busy:
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        self.log_view.clear()
        self.controller.start(self.session)

    def _on_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy and self.session.input_path is not None)
        self.progress.setVisible(busy)

    def _on_failed(self, tb: str) -> None:
        self._log(tb)
        QMessageBox.critical(self, "Run failed", tb.strip().splitlines()[-1])

    def _on_finished(self, output_root: Path) -> None:
        self.session.output_dir = output_root
        self.store = ReviewStore(output_root)

        # Persist the exact settings that produced these outputs, next to them.
        self.session.save_json(Path(output_root) / "review" / "session.json")

        if self.session.mode == "sir":
            results = SIRResults(output_root)
            if not results.ok:
                self._log("No crops found — check the annotations.")
            self.sir_viewer.load(results)
            self.tabs.setCurrentIndex(1)
        else:
            ss = StartStopResults(output_root)
            if ss.ok:
                self.startstop_viewer.load(ss, self.store)
            else:
                self._log("No detections saved — is 'Save annotated .fif of detections' on?")
            sp = SpontaneousResults(output_root)
            if sp.ok:
                self.spontaneous_viewer.load(sp)
            self.tabs.setCurrentIndex(2)

    # ------------------------------------------------------------------ #
    def save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.json", "JSON (*.json)")
        if path:
            self.session.save_json(Path(path))
            self._log(f"Session saved to {path}")

    def load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "JSON (*.json)")
        if not path:
            return
        self.session = Session.load_json(Path(path))
        # Rebuild the widgets that hold a reference to the old session object.
        self.settings_panel.session = self.session
        self.settings_panel.rebuild()
        for w in (self.sir_viewer, self.startstop_viewer, self.spontaneous_viewer):
            w.session = self.session
        self.mode_box.setCurrentIndex(0 if self.session.mode == "sir" else 1)
        self.file_label.setText(str(self.session.input_path or "No file loaded"))
        self.run_btn.setEnabled(self.session.input_path is not None)
        self._log(f"Session loaded from {path}")

    def export_runner(self) -> None:
        if self.session.input_path is None:
            QMessageBox.information(self, "No file", "Open a recording first.")
            return
        default = f"run_{Path(self.session.input_path).stem}.py"
        path, _ = QFileDialog.getSaveFileName(self, "Export runner script", default, "Python (*.py)")
        if not path:
            return
        Path(path).write_text(self.session.to_runner_script(), encoding="utf-8")
        self._log(f"Runner script written to {path}")
        QMessageBox.information(
            self, "Exported",
            "The script reproduces this session headlessly — same settings, same edits, "
            "same outputs.",
        )

    def export_reviewed_metrics(self) -> None:
        """The automated metrics plus the manual review columns, as a NEW file."""
        if self.store is None or self.session.output_dir is None:
            QMessageBox.information(self, "No results", "Run the pipeline first.")
            return
        if self.session.mode == "sir":
            QMessageBox.information(
                self, "StartStop only",
                "The manual review layer applies to the StartStop epoch browser.",
            )
            return
        results = StartStopResults(self.session.output_dir)
        augmented = self.store.augment_metrics(results.metrics)
        if augmented.empty:
            QMessageBox.information(self, "Nothing to export", "The metrics table is empty.")
            return
        default = str(Path(self.session.output_dir) / "review" / "metrics_with_review.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export reviewed metrics", default, "CSV (*.csv)")
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        augmented.to_csv(path, index=False)
        self._log(f"Reviewed metrics written to {path}")

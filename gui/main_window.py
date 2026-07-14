"""Application shell: file, mode switch, settings, the three review surfaces, log."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QWidget, QVBoxLayout,
)

from .results import SIRResults, SpontaneousResults, StartStopResults, detect_mode
from .review_store import ReviewStore
from .runner import RunController
from .session import Session
from .widgets.database_dialog import DatabaseDialog
from .widgets.raw_browser import RawBrowser
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
        self.controller.progress.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.busy_changed.connect(self._on_busy)
        self.store: ReviewStore | None = None
        self.last_scan_dir: Path | None = None

        # ---- top bar ----
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold;")

        open_btn = QPushButton("Open recording…")
        open_btn.clicked.connect(self.open_file)
        browse_btn = QPushButton("Processed recordings…")
        browse_btn.setToolTip("Scan a folder for runs that already have results and open one.")
        browse_btn.clicked.connect(self.browse_database)

        self.mode_box = QComboBox()
        self.mode_box.addItems([m[0] for m in MODES])
        self.mode_box.currentIndexChanged.connect(self._on_mode)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        self.run_btn.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.progress.setMinimumWidth(260)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(browse_btn)
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

        self.raw_browser = RawBrowser()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.settings_panel, "Settings")
        self.tabs.addTab(self.sir_viewer, "Crop review (SIR)")
        self.tabs.addTab(self.startstop_viewer, "Epoch browser (StartStop)")
        self.tabs.addTab(self.spontaneous_viewer, "Spontaneous EMG")
        self.tabs.addTab(self.raw_browser, "Raw")

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
            ("Open results folder…", self.open_results),
            ("Browse processed recordings…", self.browse_database),
            (None, None),
            ("Load session…", self.load_session),
            ("Save session…", self.save_session),
            ("Export runner script…", self.export_runner),
            ("Export reviewed metrics…", self.export_reviewed_metrics),
        ):
            if text is None:
                file_menu.addSeparator()
                continue
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
        """Grey out the surfaces the current mode cannot produce. Raw always stays open."""
        sir = self.session.mode == "sir"
        self.tabs.setTabEnabled(1, sir)
        self.tabs.setTabEnabled(2, not sir)
        self.tabs.setTabEnabled(3, not sir)  # spontaneous only runs inside StartStop
        self.tabs.setTabEnabled(4, True)     # raw view works for any recording
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)

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
        if busy:
            # Until the first tqdm tick arrives we do not know the total, so stay busy-looking.
            self.progress.setRange(0, 0)
            self.progress.setFormat("starting…")

    def _on_progress(self, phase: str, n: int, total: int) -> None:
        """Determinate progress, driven by the pipeline's own tqdm loops.

        SIR ticks over crops ("PASS 2: detect epochs"), StartStop over conditions
        ("STARTSTOP: detect by condition").
        """
        if total <= 0:
            return
        self.progress.setRange(0, total)
        self.progress.setValue(n)
        self.progress.setFormat(f"{phase} — %v/%m")

    def _on_failed(self, tb: str) -> None:
        self._log(tb)
        QMessageBox.critical(self, "Run failed", tb.strip().splitlines()[-1])

    def _on_finished(self, output_root: Path) -> None:
        self.session.output_dir = output_root
        # Persist the exact settings that produced these outputs, next to them.
        self.session.save_json(Path(output_root) / "review" / "session.json")
        self.show_results(output_root, self.session.mode)

    # ------------------------------------------------------------------ #
    def show_results(self, output_root: Path, mode: str) -> None:
        """Populate the review surfaces from an output root — freshly produced or loaded."""
        output_root = Path(output_root)
        self.store = ReviewStore(output_root)

        if mode == "sir":
            results = SIRResults(output_root)
            if not results.ok:
                self._log("No crops found — check the annotations.")
            self.sir_viewer.load(results)
            self._load_raw_sir(results)
            self.tabs.setCurrentIndex(1)
        else:
            ss = StartStopResults(output_root)
            sp = SpontaneousResults(output_root)
            if ss.ok:
                self.startstop_viewer.load(ss, self.store, sp if sp.ok else None)
                self._load_raw_startstop(ss, sp)
            else:
                self._log("No detections saved — is 'Save annotated .fif of detections' on?")
            if sp.ok:
                self.spontaneous_viewer.load(sp)
            self.tabs.setCurrentIndex(2)

    def _load_raw_sir(self, results: SIRResults) -> None:
        """The raw tab shows the preprocessed recording the crops were cut from."""
        path = results.raw_path()
        if path is None:
            return
        try:
            import mne
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        except Exception as exc:
            self._log(f"Raw view unavailable: {exc}")
            return
        self.raw_browser.set_recording(raw, title=path.stem)

    def _load_raw_startstop(self, ss: StartStopResults, sp: SpontaneousResults) -> None:
        """Raw view of the first condition, with its detections, bursts and envelopes."""
        conds = ss.conditions()
        if not conds:
            return
        cond = conds[0]
        try:
            raw = ss.raw(cond)
        except Exception as exc:
            self._log(f"Raw view unavailable: {exc}")
            return
        dets = [(d.time, "+".join(d.channels)) for d in ss.detections(cond)]
        self.raw_browser.set_recording(
            raw, title=cond, detections=dets,
            bursts=sp.bursts(cond) if sp.ok else None,
            envelopes=sp.envelopes_on_segment(cond) if sp.ok else None,
        )

    def open_results(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open an output folder")
        if path:
            self.load_results(Path(path))

    def load_results(self, root: Path) -> None:
        """Open results that already exist on disk, without re-running anything."""
        mode = detect_mode(root)
        if mode is None:
            QMessageBox.warning(
                self, "No results here",
                f"{root}\n\ndoes not contain pipeline results.\n\n"
                "Note that the pipeline creates both mode folders on every run, so an empty "
                "'Stimulation-induced responses' folder does not mean SIR results exist.",
            )
            return

        # Re-use the settings that produced this run, if the GUI wrote them.
        sess_file = root / "review" / "session.json"
        if sess_file.exists():
            try:
                self.session = Session.load_json(sess_file)
                self._rebind_session()
                self._log(f"Restored the session that produced these results: {sess_file}")
            except Exception as exc:
                self._log(f"Could not restore session.json ({exc}); using defaults.")

        self.session.output_dir = root
        self.session.set_mode(mode)
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(0 if mode == "sir" else 1)
        self.mode_box.blockSignals(False)
        self.settings_panel.rebuild()
        self._sync_tabs()

        self.file_label.setText(f"{root.name}   (loaded results)")
        self.run_btn.setEnabled(self.session.input_path is not None)
        self._log(f"Loaded {mode.upper()} results from {root}")
        self.show_results(root, mode)

    def browse_database(self) -> None:
        base = self.last_scan_dir
        if base is None and self.session.output_dir is not None:
            base = Path(self.session.output_dir).parent
        dlg = DatabaseDialog(self, base)
        if dlg.exec() and dlg.selected is not None:
            self.last_scan_dir = Path(dlg.path_label.text())
            self.load_results(dlg.selected.root)

    def _rebind_session(self) -> None:
        """Point the widgets at a freshly loaded Session object."""
        self.settings_panel.session = self.session
        for w in (self.sir_viewer, self.startstop_viewer, self.spontaneous_viewer):
            w.session = self.session

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
        self._rebind_session()
        self.settings_panel.rebuild()
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
            results = SIRResults(self.session.output_dir)
            augmented = self.store.augment_sir_metrics(results.metrics, self.session)
        else:
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

"""Application shell: file, mode switch, settings, the three review surfaces, log."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QWidget, QVBoxLayout,
)

from src.text_curves import is_text_curves_file, text_curves_window_defaults

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
from .widgets.condition_viewer import ConditionViewer
from .widgets.scenario_viewer import ScenarioViewer

MODES = [("Stimulation-induced (SIR)", "sir"), ("StartStop", "startstop"),
         ("Condition test", "condition")]
# combobox index per mode
_MODE_INDEX = {m[1]: i for i, m in enumerate(MODES)}
# tab indices (kept in one place so the sync logic below stays readable)
TAB_SETTINGS, TAB_SIR, TAB_RECRUIT, TAB_STARTSTOP, TAB_SPONT, TAB_COND, TAB_RAW = range(7)


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
        self._raw_source = None
        self._raw_loaded = None

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
        self.recruitment_viewer = ScenarioViewer(self.session)
        self.startstop_viewer = StartStopViewer(self.session)
        self.spontaneous_viewer = SpontaneousViewer(self.session)
        self.spontaneous_viewer.rerun_requested.connect(self.run)
        self.condition_viewer = ConditionViewer(self.session)
        self.condition_viewer.rerun_requested.connect(self.run)

        self.raw_browser = RawBrowser()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.settings_panel, "Settings")            # TAB_SETTINGS
        self.tabs.addTab(self.sir_viewer, "Crop review (SIR)")        # TAB_SIR
        self.tabs.addTab(self.recruitment_viewer, "Recruitment")      # TAB_RECRUIT
        self.tabs.addTab(self.startstop_viewer, "Epoch browser (StartStop)")  # TAB_STARTSTOP
        self.tabs.addTab(self.spontaneous_viewer, "Spontaneous EMG")  # TAB_SPONT
        self.tabs.addTab(self.condition_viewer, "Condition test")     # TAB_COND
        self.tabs.addTab(self.raw_browser, "Raw")                     # TAB_RAW
        self.tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addLayout(top)
        cv.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # ---- log dock ----
        # Kept deliberately short: the channel stack needs the vertical space far more than
        # the log does. Toggle it from View, or drag the splitter if you want it taller.
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMaximumHeight(70)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()  # shown automatically while a run is going

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
        log_act = QAction("Show log", self)
        log_act.setCheckable(True)
        log_act.setChecked(False)
        log_act.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(
            lambda v: log_act.setChecked(bool(v))
        )
        view_menu.addAction(log_act)

    def _log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # ------------------------------------------------------------------ #
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "", "Recordings (*.mat *.fif *.edf *.txt);;All files (*)"
        )
        if not path:
            return
        self.session.input_path = Path(path)
        # Ask the pipeline itself where it would write, so the GUI and the CLI
        # never disagree about the output location.
        from src.pipeline import _default_output_root_for_input
        self.session.output_dir = _default_output_root_for_input(Path(path))
        self.file_label.setText(str(path))
        self.run_btn.setEnabled(True)
        self._log(f"Loaded {path}")
        self._log(f"Outputs will go to {self.session.output_dir}")
        self._adapt_to_text_curves(Path(path))

    def _adapt_to_text_curves(self, path: Path) -> None:
        """Show the windows a pre-epoched text export actually runs with.

        run_pipeline adapts these windows itself for this format (the file has no
        pre-stimulus data, so the standard windows do not fit). Mirroring them
        into the settings keeps the panel honest: what is on screen is what runs,
        and the user can still override any of them by hand.
        """
        if not is_text_curves_file(path):
            return
        try:
            adapted = text_curves_window_defaults(path)
        except (OSError, ValueError) as exc:
            self._log(f"Could not read text curves windows: {exc}")
            return
        for name, value in adapted.items():
            self.session.set(name, value)
        self.settings_panel.rebuild()
        self._log(
            "Pre-cut epochs: artifact search and epoching are skipped; windows set to "
            + ", ".join(f"{k}={v:g}" for k, v in adapted.items())
        )

    def _on_mode(self, idx: int) -> None:
        self.session.set_mode(MODES[idx][1])
        self.settings_panel.rebuild()
        self._sync_tabs()

    def _sync_scenario_tab(self) -> None:
        """Name the scenario tab after the scenario this run actually produced.

        The three Neurosoft protocols are mutually exclusive, so a tab labelled
        "Recruitment" on a Jendrassik run is simply wrong; a non-Neurosoft SIR run
        produces none of them and the tab goes back to its generic name.
        """
        from .widgets.scenario_viewer import SCENARIOS
        folder = getattr(self.recruitment_viewer, "scenario", None)
        label = SCENARIOS.get(folder, (None,))[0] if folder else None
        self.tabs.setTabText(TAB_RECRUIT, label or "Recruitment")

    def _sync_tabs(self) -> None:
        """Grey out the surfaces the current mode cannot produce."""
        mode = self.session.mode
        sir = mode == "sir"
        startstop = mode == "startstop"
        condition = mode == "condition"
        self.tabs.setTabEnabled(TAB_SIR, sir)
        self.tabs.setTabEnabled(TAB_RECRUIT, sir)         # recruitment lives in SIR
        self.tabs.setTabEnabled(TAB_STARTSTOP, startstop)
        self.tabs.setTabEnabled(TAB_SPONT, startstop)     # spontaneous is StartStop-only
        self.tabs.setTabEnabled(TAB_COND, condition)
        self.tabs.setTabEnabled(TAB_RAW, sir or startstop)  # condition saves no raw
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
        self.log_dock.show()  # the log matters while a run is going; it is hidden otherwise
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
        # A .txt run auto-routes to Condition or SIR by content, so trust the
        # output on disk for which viewer to open, not the requested run mode.
        actual = detect_mode(output_root) or self.session.mode
        if actual != self.session.mode:
            self.session.set_mode(actual)
            self.mode_box.blockSignals(True)
            self.mode_box.setCurrentIndex(_MODE_INDEX.get(actual, 0))
            self.mode_box.blockSignals(False)
            self.settings_panel.rebuild()
            self._sync_tabs()
        self.show_results(output_root, actual)

    # ------------------------------------------------------------------ #
    def show_results(self, output_root: Path, mode: str) -> None:
        """Populate the review surfaces from an output root — freshly produced or loaded."""
        output_root = Path(output_root)
        self.store = ReviewStore(output_root)

        if mode == "condition":
            self.condition_viewer.load(output_root)
            self.tabs.setCurrentIndex(TAB_COND)
            self._raw_source = None      # condition runs save no raw recording
            self._raw_loaded = None
            return

        if mode == "sir":
            results = SIRResults(output_root)
            if not results.ok:
                self._log("No crops found — check the annotations.")
            self.sir_viewer.load(results)
            self.recruitment_viewer.load(output_root)   # empty if no Neurosoft scenario ran
            self._sync_scenario_tab()
            self.tabs.setCurrentIndex(TAB_SIR)
        else:
            ss = StartStopResults(output_root)
            sp = SpontaneousResults(output_root)
            if ss.ok:
                self.startstop_viewer.load(ss, self.store, sp if sp.ok else None)
            else:
                self._log("No detections saved — is 'Save annotated .fif of detections' on?")
            if sp.ok:
                self.spontaneous_viewer.load(sp)
            self.tabs.setCurrentIndex(TAB_STARTSTOP)

        # The raw recording is read only when the Raw tab is actually opened: on a network
        # drive a preprocessed .fif is hundreds of MB, and loading it eagerly stalled the
        # window for minutes after every open.
        self._raw_source = (mode, output_root)
        self._raw_loaded = None

    # ------------------------------------------------------------------ #
    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_RAW:
            self._ensure_raw_loaded()

    def _ensure_raw_loaded(self) -> None:
        src = getattr(self, "_raw_source", None)
        if src is None or self._raw_loaded == src:
            return
        mode, root = src
        self.raw_browser.plot.message("Loading the recording…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if mode == "sir":
                self._load_raw_sir(SIRResults(root))
            else:
                self._load_raw_startstop(StartStopResults(root), SpontaneousResults(root))
            self._raw_loaded = src
        finally:
            QApplication.restoreOverrideCursor()

    def _load_raw_sir(self, results: SIRResults) -> None:
        """The raw tab shows the preprocessed recording the crops were cut from."""
        path = results.raw_path()
        if path is None:
            self.raw_browser.plot.message("This run saved no raw recording.")
            return
        try:
            import mne
            # preload=False: these files run to ~1 GB (a 97-minute recording at 4 kHz).
            # Preloading pulled the whole thing over the network — 116 s on a Drive path —
            # when the browser only ever draws the visible window, which reads in ~10 ms.
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        except Exception as exc:
            self.raw_browser.plot.message(f"Raw view unavailable:\n{exc}")
            self._log(f"Raw view unavailable: {exc}")
            return
        self.raw_browser.set_recording(raw, title=path.stem)

    def _load_raw_startstop(self, ss: StartStopResults, sp: SpontaneousResults) -> None:
        """Raw view of the first condition, with its detections, bursts and envelopes."""
        conds = ss.conditions()
        if not conds:
            self.raw_browser.plot.message("This run saved no annotated recording.")
            return
        cond = conds[0]
        try:
            raw = ss.raw(cond)
        except Exception as exc:
            self.raw_browser.plot.message(f"Raw view unavailable:\n{exc}")
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
                "Pick the run folder itself — the one holding 'results/'. Older runs may nest "
                "their outputs under 'Stimulation-induced responses/' or 'StartStop analysis/'; "
                "both layouts are read.",
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

        # Recover the recording these results came from, so "Apply edits and
        # re-run" acts on the run being looked at instead of asking for a file.
        # session.json only exists for GUI runs; the manifest covers the rest.
        from src.pipeline import input_from_run_manifest

        src_path = self.session.input_path
        if src_path is None or not Path(src_path).exists():
            src_path = input_from_run_manifest(root)
        if src_path is not None:
            self.session.input_path = Path(src_path)
            self._log(f"These results came from {src_path}")
            self._adapt_to_text_curves(Path(src_path))
        else:
            self.session.input_path = None
            self._log(
                "The source recording could not be located from this folder — "
                "re-running will ask for it. Everything else works."
            )

        self.session.output_dir = root
        self.session.set_mode(mode)
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(_MODE_INDEX.get(mode, 0))
        self.mode_box.blockSignals(False)
        self.settings_panel.rebuild()
        self._sync_tabs()

        self.file_label.setText(
            f"{root.name}   (loaded results)"
            if self.session.input_path is None
            else f"{Path(self.session.input_path).name}   (loaded results — re-run works)"
        )
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

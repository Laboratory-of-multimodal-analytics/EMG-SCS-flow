"""Background execution of the real pipeline, with its log and progress streamed to the UI."""

from __future__ import annotations

import io
import logging
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

# The pipeline drives two tqdm loops, one per mode:
#   SIR mode:       "PASS 2: detect epochs"          (over crops)
#   StartStop mode: "STARTSTOP: detect by condition" (over conditions)
# tqdm renders to stderr and rewrites the line with \r, so we parse the carriage-return
# fragments rather than whole lines.
# The desc itself contains a colon ("STARTSTOP: detect by condition"), so match everything
# up to the percentage rather than stopping at the first colon.
_TQDM_RE = re.compile(r"^(?P<desc>.*?)\s*:?\s*\d+%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)")

# Coarse stage messages the pipeline prints, so the bar means something before the loop starts.
_STAGE_RE = re.compile(r"^\[(SIR|STARTSTOP)\]\s*(?P<msg>.+?)\.{0,3}$")


class _LogStream(io.TextIOBase):
    """Funnel print() output into a Qt signal, line by line."""

    def __init__(self, emit) -> None:
        super().__init__()
        self._emit = emit
        self._buf = ""

    def write(self, text: str) -> int:  # noqa: D102
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit(line)
        return len(text)

    def flush(self) -> None:  # noqa: D102
        if self._buf.strip():
            self._emit(self._buf)
        self._buf = ""


class _ProgressStream(io.TextIOBase):
    """Parse tqdm's stderr chatter into (phase, n, total) progress updates."""

    def __init__(self, emit_progress, emit_log) -> None:
        super().__init__()
        self._progress = emit_progress
        self._log = emit_log
        self._buf = ""
        self._last: tuple[str, int] | None = None

    def write(self, text: str) -> int:  # noqa: D102
        self._buf += text
        # tqdm redraws with \r; split on both so we see every refresh.
        parts = re.split(r"[\r\n]", self._buf)
        self._buf = parts.pop()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = _TQDM_RE.search(part)
            if m:
                phase = m.group("desc").strip().rstrip(":") or "Working"
                n, total = int(m.group("n")), int(m.group("total"))
                if self._last != (phase, n):
                    self._last = (phase, n)
                    self._progress(phase, n, total)
                continue
            if not part.startswith(("  ", "Reading", "Isotrak", "Adding", "Ready")):
                self._log(part)
        return len(text)

    def flush(self) -> None:  # noqa: D102
        self._buf = ""


class _QtLogHandler(logging.Handler):
    def __init__(self, emit) -> None:
        super().__init__()
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._emit(self.format(record))
        except Exception:
            pass


def run_session(session, emit_log, emit_progress) -> Path:
    """One run of the real pipeline for *session*, its output and progress streamed out.

    The session's settings and edits are applied to the imported src.pipeline module
    right before the call, so the run is exactly what the equivalent runner script does.
    """
    import src.pipeline as P
    from src import run_pipeline

    session.apply_to_pipeline(P)

    handler = _QtLogHandler(emit_log)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)

    emit_log(f"Running {session.mode.upper()} mode on {session.input_path}")
    if session.template_dir is not None:
        emit_log(f"Template bank: {session.template_dir}")

    out_stream = _LogStream(emit_log)
    err_stream = _ProgressStream(emit_progress, emit_log)
    try:
        with redirect_stdout(out_stream), redirect_stderr(err_stream):
            out = run_pipeline(
                session.input_path,
                output_dir=session.output_dir,
                startstop_mode=(session.mode == "startstop"),
                force_scenario=session.force_scenario,
                **session.kwargs(),
            )
        out_stream.flush()
        err_stream.flush()
    finally:
        root.removeHandler(handler)
    emit_log(f"Done. Outputs under: {out}")
    return Path(out)


class PipelineWorker(QObject):
    """Runs run_pipeline() off the GUI thread."""

    log = Signal(str)
    progress = Signal(str, int, int)  # phase, n, total
    finished = Signal(object)   # Path to the output root
    failed = Signal(str)

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session

    def run(self) -> None:
        try:
            self.finished.emit(run_session(self.session, self.log.emit, self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchWorker(QObject):
    """Runs several recordings one after another, each with its own session.

    A recording that fails is reported and the batch goes on. ``stop()`` takes
    effect between recordings: a run cut halfway would leave half-written outputs.
    Each finished run saves its session next to its results, as a single run does.
    """

    log = Signal(str)
    progress = Signal(str, int, int)
    file_started = Signal(int)
    file_done = Signal(int, object, str)   # index, output root (or None), traceback ("" if ok)
    finished = Signal()

    def __init__(self, sessions) -> None:
        super().__init__()
        self.sessions = list(sessions)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for i, session in enumerate(self.sessions):
            if self._stop:
                break
            self.file_started.emit(i)
            try:
                out = run_session(session, self.log.emit, self.progress.emit)
                session.output_dir = out
                session.save_json(Path(out) / "review" / "session.json")
                self.file_done.emit(i, out, "")
            except Exception:
                tb = traceback.format_exc()
                self.log.emit(tb)
                self.file_done.emit(i, None, tb)
        self.finished.emit()


class RunController(QObject):
    """Owns the worker thread and keeps the UI from launching two runs at once."""

    log = Signal(str)
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)
    batch_file_started = Signal(int)
    batch_file_done = Signal(int, object, str)
    batch_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, session) -> bool:
        if self.busy:
            return False
        self._thread = QThread()
        self._worker = PipelineWorker(session)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        self.busy_changed.emit(True)
        return True

    def start_batch(self, sessions) -> bool:
        """Run *sessions* one after another (see BatchWorker)."""
        if self.busy or not sessions:
            return False
        self._thread = QThread()
        self._worker = BatchWorker(sessions)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log)
        self._worker.progress.connect(self.progress)
        self._worker.file_started.connect(self.batch_file_started)
        self._worker.file_done.connect(self.batch_file_done)
        self._worker.finished.connect(self._on_batch_finished)
        self._thread.start()
        self.busy_changed.emit(True)
        return True

    def stop_batch(self) -> None:
        if isinstance(self._worker, BatchWorker):
            self._worker.stop()

    def _on_batch_finished(self) -> None:
        self._teardown()
        self.batch_finished.emit()

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.busy_changed.emit(False)

    def _on_finished(self, out: Path) -> None:
        self._teardown()
        self.finished.emit(out)

    def _on_failed(self, tb: str) -> None:
        self._teardown()
        self.failed.emit(tb)

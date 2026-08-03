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


class PipelineWorker(QObject):
    """Runs run_pipeline() off the GUI thread.

    The session's settings and edits are applied to the imported src.pipeline module
    right before the call, so the run is exactly what the equivalent runner script does.
    """

    log = Signal(str)
    progress = Signal(str, int, int)  # phase, n, total
    finished = Signal(object)   # Path to the output root
    failed = Signal(str)

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session

    def run(self) -> None:
        try:
            import src.pipeline as P
            from src import run_pipeline

            self.session.apply_to_pipeline(P)

            handler = _QtLogHandler(self.log.emit)
            handler.setFormatter(logging.Formatter("%(message)s"))
            root = logging.getLogger()
            root.addHandler(handler)

            self.log.emit(f"Running {self.session.mode.upper()} mode on {self.session.input_path}")
            if self.session.template_dir is not None:
                self.log.emit(f"Template bank: {self.session.template_dir}")

            out_stream = _LogStream(self.log.emit)
            err_stream = _ProgressStream(self.progress.emit, self.log.emit)
            try:
                with redirect_stdout(out_stream), redirect_stderr(err_stream):
                    out = run_pipeline(
                        self.session.input_path,
                        output_dir=self.session.output_dir,
                        startstop_mode=(self.session.mode == "startstop"),
                        force_scenario=self.session.force_scenario,
                        **self.session.kwargs(),
                    )
                out_stream.flush()
                err_stream.flush()
            finally:
                root.removeHandler(handler)

            self.log.emit(f"Done. Outputs under: {out}")
            self.finished.emit(Path(out))
        except Exception:
            self.failed.emit(traceback.format_exc())


class RunController(QObject):
    """Owns the worker thread and keeps the UI from launching two runs at once."""

    log = Signal(str)
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)

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

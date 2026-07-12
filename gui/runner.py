"""Background execution of the real pipeline, with its log streamed to the UI."""

from __future__ import annotations

import io
import logging
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal


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
            stream = _LogStream(self.log.emit)
            try:
                with redirect_stdout(stream):
                    out = run_pipeline(
                        self.session.input_path,
                        output_dir=self.session.output_dir,
                        startstop_mode=(self.session.mode == "startstop"),
                        **self.session.kwargs(),
                    )
                stream.flush()
            finally:
                root.removeHandler(handler)

            self.log.emit(f"Done. Outputs under: {out}")
            self.finished.emit(Path(out))
        except Exception:
            self.failed.emit(traceback.format_exc())


class RunController(QObject):
    """Owns the worker thread and keeps the UI from launching two runs at once."""

    log = Signal(str)
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

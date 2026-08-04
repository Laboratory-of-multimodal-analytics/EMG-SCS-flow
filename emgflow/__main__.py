"""Entry point:  python -m emgflow   (from the repository root)"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

# Guard a matplotlib SpanSelector crash that aborts template/window drawing.
import emgflow._mpl_patches  # noqa: E402,F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from emgflow.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("EMG-SCS-flow")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""The library versions EMG-SCS-flow is tested with, and a check against what is installed.

A plotting or array library that renames a function breaks a run in the middle
(matplotlib dropped ``cm.get_cmap`` in 3.9 and ``boxplot(labels=)`` in 3.11, NumPy
dropped ``np.trapz``). The tested set lives in ``requirements-lock.txt``; the GUI
says at start-up when the installed versions differ from it, so a failure on
another computer can be traced to its versions at once.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
TESTED_PYTHON = (3, 13)


def tested_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            out[name.strip()] = ver.strip()
    return out


def version_mismatches() -> list[str]:
    """``"numpy 2.5.3 (tested 2.3.5)"`` for every package that differs from the lock."""
    try:
        tested = tested_versions()
    except OSError:
        return []
    msgs = []
    if sys.version_info[:2] != TESTED_PYTHON:
        msgs.append(f"Python {sys.version_info[0]}.{sys.version_info[1]} "
                    f"(tested {TESTED_PYTHON[0]}.{TESTED_PYTHON[1]})")
    for name, want in tested.items():
        try:
            have = version(name)
        except PackageNotFoundError:
            msgs.append(f"{name} not installed (tested {want})")
            continue
        if have != want:
            msgs.append(f"{name} {have} (tested {want})")
    return msgs

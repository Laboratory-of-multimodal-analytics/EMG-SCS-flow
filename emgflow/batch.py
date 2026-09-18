"""Processing several recordings in one go: which files, and the session each one runs with.

The GUI runs one recording at a time, with the session on screen. A batch runs a
list of them one after another, each with its own copy of that session:

* settings come from the Settings tab — except for a recording the GUI already
  processed, which re-runs with the settings and hand edits saved next to its
  results (``review/session.json``), exactly as opening it and pressing Run would;
* hand edits of the recording on screen (whitelist, suppress, excluded channels
  and configurations) belong to that recording and are never carried over;
* a Neurosoft text export always runs in SIR mode with its windows fitted to the
  file, and keeps the scenario of its previous run if it has one — as opening it does;
* results go next to each recording (the pipeline's default), or into one folder.

Nothing here touches Qt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.text_curves import is_text_curves_file, text_curves_window_defaults

from .results import detect_mode
from .session import Session
from .settings_spec import defaults_for

RECORDING_SUFFIXES = (".mat", ".fif", ".edf", ".txt")
KIND_NEUROSOFT = "Neurosoft .txt"
#: Scenarios a run manifest may carry (the GUI's scenario selector offers the same).
KNOWN_SCENARIOS = ("recruitment", "jendrassik", "paired", "condition", "hreflex")
#: The windows a Neurosoft export gets fitted to its own curves (text_curves_window_defaults).
_TEXT_WINDOWS = ("epoch_tmin", "epoch_tmax", "baseline_tmin", "baseline_tmax",
                 "resp_tmin", "resp_tmax")


@dataclass
class BatchItem:
    path: Path
    kind: str                   # what the file is, as shown in the list
    supported: bool
    note: str = ""              # why it is not run, when it is not
    mode: str = "sir"           # run mode of a .mat / .fif / .edf recording
    done_mode: str | None = None  # mode of the results already next to it, if any
    status: str = "waiting"
    output: Path | None = None
    error: str = ""

    @property
    def neurosoft(self) -> bool:
        return self.kind == KIND_NEUROSOFT


def describe(path: Path) -> tuple[str, bool, str]:
    """``(kind, supported, note)`` of one file. A .txt is opened to tell which text it is."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        if is_text_curves_file(path):
            return KIND_NEUROSOFT, True, ""
        return "text", False, ("not a Neurosoft curves export (a LabChart text export is not "
                               "read; export .mat or convert to .fif)")
    if suffix == ".mat":
        return "LabChart .mat", True, ""
    if suffix == ".fif":
        return "FIF", True, ""
    if suffix == ".edf":
        return "EDF", True, ""
    return suffix or "?", False, "unsupported format"


def _is_output_root(d: Path) -> bool:
    """A folder the pipeline wrote (it holds results/ or review/), never a recording folder."""
    return (d / "results").is_dir() or (d / "review").is_dir()


def find_recordings(folder: Path, max_depth: int = 6) -> list[Path]:
    """Every recording under *folder*, skipping the results folders themselves.

    A .txt counts only when it is a Neurosoft curves export (``*_annotations.txt``
    and LabChart text exports are not recordings the pipeline reads), and epoch
    files saved by earlier runs are left out.
    """
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        for p in entries:
            if p.name.startswith((".", "~$")):
                continue
            if p.is_dir():
                if p.name not in ("results", "review") and not _is_output_root(p):
                    walk(p, depth + 1)
                continue
            if p.suffix.lower() not in RECORDING_SUFFIXES:
                continue
            if p.name.endswith(("_annotations.txt", "-epo.fif")):
                continue
            if p.suffix.lower() == ".txt" and not is_text_curves_file(p):
                continue
            found.append(p)

    walk(Path(folder), 0)
    return found


def output_root(path: Path, out_base: Path | None = None) -> Path:
    """Where a recording's results go: the pipeline's own choice, or the same name in *out_base*."""
    from src.pipeline import _default_output_root_for_input

    default = _default_output_root_for_input(Path(path))
    return default if out_base is None else Path(out_base) / default.name


def processed_mode(root: Path) -> str | None:
    """The mode of the results already in *root*, or None when there are none."""
    try:
        return detect_mode(root) if Path(root).is_dir() else None
    except Exception:
        return None


def describe_item(path: Path, default_mode: str) -> BatchItem:
    kind, supported, note = describe(Path(path))
    item = BatchItem(Path(path), kind, supported, note)
    item.done_mode = processed_mode(output_root(item.path)) if supported else None
    # a recording processed before keeps the mode it was processed in
    item.mode = "startstop" if item.done_mode == "startstop" else (
        "sir" if item.done_mode else default_mode)
    return item


def mark_duplicates(items: list[BatchItem], out_base: Path | None = None) -> None:
    """Two files that would write the same results folder (``x.mat`` next to its
    converted ``x.fif``): keep the .fif — the prepared copy — and leave the rest out."""
    by_root: dict[Path, list[BatchItem]] = {}
    for it in items:
        if it.supported:
            by_root.setdefault(output_root(it.path, out_base), []).append(it)
    for group in by_root.values():
        if len(group) < 2:
            continue
        keep = next((it for it in group if it.path.suffix.lower() == ".fif"), group[0])
        for it in group:
            if it is not keep:
                it.supported = False
                it.note = f"same results folder as {keep.path.name}"


def scenario_of_run(root: Path) -> str | None:
    """The scenario recorded in a run's manifest, if any."""
    try:
        data = json.loads((Path(root) / "review" / "run.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    scen = data.get("scenario")
    return scen if scen in KNOWN_SCENARIOS else None


def session_for(item: BatchItem, base: Session, out_base: Path | None = None) -> Session:
    """The session one recording of a batch runs with (see the module docstring)."""
    root = output_root(item.path, out_base)
    s: Session | None = None
    own = root / "review" / "session.json"
    if own.exists():
        try:
            s = Session.load_json(own)
        except Exception:
            s = None
    if s is None:
        s = Session.from_dict(base.to_dict())
        s.force_keys, s.suppress_keys = set(), set()
        s.exclude_channels, s.exclude_configs = set(), set()
        s.force_scenario = None
        fitted_to_curves = (base.input_path is not None
                            and is_text_curves_file(Path(base.input_path)))
    else:
        fitted_to_curves = False

    mode = "sir" if item.neurosoft else item.mode
    s.set_mode(mode)
    if fitted_to_curves and not item.neurosoft:
        # the windows on screen were fitted to a Neurosoft export opened last; they
        # would cut a continuous recording wrongly
        defaults = defaults_for(mode)
        for key in _TEXT_WINDOWS:
            if key in defaults:
                s.settings[key] = defaults[key]
    s.input_path = item.path
    s.output_dir = root
    s.force_scenario = scenario_of_run(root) or s.force_scenario
    if item.neurosoft:
        from src.neurosoft import scenario_from_name

        scenario = s.force_scenario or scenario_from_name(item.path)
        for name, value in text_curves_window_defaults(item.path, scenario).items():
            s.set(name, value)
    return s

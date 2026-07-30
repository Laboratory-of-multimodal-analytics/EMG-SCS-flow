"""I/O helpers for the EMG pipeline."""

from __future__ import annotations

from pathlib import Path

STIMULATION_INDUCED_FOLDER = "Stimulation-induced responses"
STARTSTOP_FOLDER = "StartStop analysis"
CONDITION_FOLDER = "Condition test"

#: Folders that identify which analysis produced a ``results/`` tree. Used to
#: decide whether the mode level is needed, and by readers to find outputs in
#: either layout.
_MODE_MARKERS: dict[str, tuple[str, ...]] = {
    STIMULATION_INDUCED_FOLDER: (
        "Stimulus-centered epochs", "Recruitment", "Jendrassik",
        "Paired stimulation", "Template overlays",
    ),
    STARTSTOP_FOLDER: (
        "Detections raw", "Spontaneous EMG", "Raw epochs",
        "Plots grouped by condition",
    ),
    CONDITION_FOLDER: (
        "Waterfall", "Amplitude vs condition", "Curves per condition",
    ),
}


def ensure_dir(path: Path) -> Path:
    """Create *path* on first write. Use this instead of creating trees up front."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _holds_other_mode(output_root: Path, own_folder: str) -> bool:
    """True if this output root already carries a DIFFERENT analysis's results.

    Checks both layouts: a named mode folder from an older run, and flattened
    outputs sitting directly in ``results/``.
    """
    output_root = Path(output_root)
    results = output_root / "results"
    for folder, markers in _MODE_MARKERS.items():
        if folder == own_folder:
            continue
        if (results / folder).exists() or (output_root / "data" / folder).exists():
            return True
        if results.exists() and any((results / m).exists() for m in markers):
            return True
    return False


def resolve_mode_dirs(output_root: Path, folder: str) -> tuple[Path, Path]:
    """``(results_dir, data_dir)`` for one analysis mode.

    The ``<mode>/`` level exists only to keep two analyses apart in one output
    root. A run produces exactly one of them, so when nothing else is there the
    level is pure nesting and is dropped — results land straight in ``results/``.
    It comes back only if the root already holds another analysis's outputs,
    where the names would otherwise collide (both modes write ``Excel/
    Large_dataset_emg_response_metrics.csv``).
    """
    output_root = Path(output_root)
    if _holds_other_mode(output_root, folder):
        return output_root / "results" / folder, output_root / "data" / folder
    return output_root / "results", output_root / "data"


def find_mode_dir(output_root: Path, folder: str) -> Path:
    """Where a finished run's *folder*-mode results actually live (reader side).

    Prefers the nested layout when it exists, so both old and new output roots
    read the same way.
    """
    nested = Path(output_root) / "results" / folder
    return nested if nested.exists() else Path(output_root) / "results"


def build_output_dirs(output_root: Path, startstop_mode: bool = False) -> dict[str, Path]:
    """Resolve the output directory layout WITHOUT creating it.

    Directories are created lazily, at the moment something is actually written
    into them (``ensure_dir``). Creating the tree up front produced a StartStop
    branch on every SIR run and vice versa, so a run left behind a dozen empty
    folders and ``results/`` could not be read as "this analysis happened".
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    folder = STARTSTOP_FOLDER if startstop_mode else STIMULATION_INDUCED_FOLDER
    results_dir, data_dir = resolve_mode_dirs(output_root, folder)
    # The inactive mode's paths are only ever used to name things; nothing is
    # written there on this run.
    other = STIMULATION_INDUCED_FOLDER if startstop_mode else STARTSTOP_FOLDER
    other_results, other_data = resolve_mode_dirs(output_root, other)

    stim_results_dir = other_results if startstop_mode else results_dir
    stim_data_dir = other_data if startstop_mode else data_dir
    startstop_dir = results_dir if startstop_mode else other_results

    return {
        "output_root": output_root,
        "data_dir": data_dir,
        "data_root": output_root / "data",
        "startstop_data_dir": other_data if not startstop_mode else data_dir,
        "stim_data_dir": stim_data_dir,
        "crops_dir": stim_data_dir / "annot_crops_fif",
        "results_dir": output_root / "results",
        "stim_results_dir": stim_results_dir,
        "epochs_dir": stim_results_dir / "Stimulus-centered epochs",
        "excel_dir": results_dir / "Excel",
        "boxplot_dir": results_dir / "Boxplots",
        "plots_grid_dir": results_dir / "Plots with grid and markers",
        "plots_plain_dir": results_dir / "Plots without grid and markers",
        "plots_grouped_dir": results_dir / "Plots grouped by amplitude",
        "templates_dir": results_dir / "Templates",
        "startstop_dir": startstop_dir,
    }


def list_crop_files(crops_dir: Path) -> list[str]:
    crops_dir = Path(crops_dir)
    if not crops_dir.exists():
        return []
    files = []
    for file_path in crops_dir.iterdir():
        if not file_path.is_file():
            continue
        name = file_path.name
        if "(" in name or ")" in name:
            continue
        files.append(name)
    return files

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
        "Paired stimulation", "H-reflex", "Template overlays",
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
        # Every table of a run goes here, whatever produced it — the per-epoch
        # metrics, the scenario deliverables, the condition summaries. They used
        # to be split between this folder and one inside each scenario's, which
        # meant knowing which scenario a file was before knowing where to look
        # for its numbers. Figures still live with their scenario.
        "excel_dir": results_dir / "Excel",
        "boxplot_dir": results_dir / "Boxplots",
        "plots_grid_dir": results_dir / "Plots with grid and markers",
        "plots_plain_dir": results_dir / "Plots without grid and markers",
        "plots_grouped_dir": results_dir / "Plots grouped by amplitude",
        "templates_dir": results_dir / "Templates",
        "startstop_dir": startstop_dir,
    }


#: Per-scenario deliverable: the figure folder, and the tables it owns in the
#: shared Excel folder. Filenames are exact or prefixes, matched by ``startswith``.
SCENARIO_OUTPUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "recruitment": ("Recruitment", ("recruitment_", "stats_top", "stats_amplitude_groups")),
    "jendrassik": ("Jendrassik", ("curves_by_amplitude_group", "stats_amplitude_groups")),
    "paired": ("Paired stimulation", ("curves_by_amplitude_group", "stats_amplitude_groups")),
    "hreflex": ("H-reflex", ("hreflex_", "stats_hm")),
}


def clear_other_scenario_outputs(results_dir: Path, keep: str) -> list[str]:
    """Remove the deliverables of every scenario except *keep*.

    A recording can be re-run under a different scenario — that is what the GUI's
    scenario selector is for, and it is how the H-reflex files are being
    reprocessed. Without this the previous scenario's folder and tables simply
    stay put, and the run root then claims two mutually exclusive protocols at
    once with nothing marking which numbers are current. The reader picks one and
    the other silently becomes a trap, differing by however much detection has
    changed since.

    Only this pipeline's own output names are touched, and only inside
    ``results/``. Returns what was removed, for the run log.
    """
    import shutil

    results_dir = Path(results_dir)
    removed: list[str] = []
    excel = results_dir / "Excel"
    for scenario, (folder, prefixes) in SCENARIO_OUTPUTS.items():
        if scenario == keep:
            continue
        d = results_dir / folder
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(folder + "/")
        if not excel.is_dir():
            continue
        keep_prefixes = SCENARIO_OUTPUTS.get(keep, ("", ()))[1]
        for f in excel.glob("*.csv"):
            if f.name.startswith(prefixes) and not f.name.startswith(keep_prefixes):
                f.unlink(missing_ok=True)
                removed.append(f.name)
    return removed


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

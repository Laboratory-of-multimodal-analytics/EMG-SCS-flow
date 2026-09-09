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


#: Per-scenario deliverable: the figure folders, and the tables it owns in the
#: shared Excel folder. Filenames are exact or prefixes, matched by ``startswith``.
#:
#: The Condition test is in here with the other four. It is the fifth reading of
#: the same export and just as mutually exclusive with them — it only differs in
#: writing several folders instead of one. Leaving it out is what let a
#: Condition run survive a re-run under another scenario, and its leftovers then
#: told every reader the recording was still a Condition test.
SCENARIO_OUTPUTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "recruitment": (("Recruitment",),
                    ("recruitment_", "stats_top", "stats_amplitude_groups")),
    "jendrassik": (("Jendrassik",),
                   ("curves_by_amplitude_group", "stats_amplitude_groups")),
    "paired": (("Paired stimulation",),
               ("curves_by_amplitude_group", "stats_amplitude_groups")),
    "hreflex": (("H-reflex",), ("hreflex_", "stats_hm")),
    "condition": (("Waterfall", "Curves per condition", "Amplitude vs condition",
                   "arrays"), ("condition_",)),
}


def other_scenario_output_paths(output_root: Path, keep: str) -> list[Path]:
    """Everything the scenarios other than *keep* left in this run root.

    A recording can be re-run under a different scenario — that is what the GUI's
    scenario selector is for, and it is how the H-reflex files are being
    reprocessed. Left in place, the previous scenario's folders and tables sit
    beside the new ones and the root then claims two mutually exclusive protocols
    at once with nothing marking which numbers are current. The reader picks one
    and the other silently becomes a trap, differing by however much detection has
    changed since.

    Collected here rather than deleted: the caller holds them aside and only
    discards them once the new run has actually produced something — see
    ``SupersededOutputs``.

    Looked for in ``results/`` and in both mode subfolders, because a root that
    has been re-run before may hold the previous scenario in the other layout.
    """
    base = Path(output_root) / "results"
    keep_prefixes = SCENARIO_OUTPUTS.get(keep, ((), ()))[1]
    found: list[Path] = []
    for results_dir in (base, base / STIMULATION_INDUCED_FOLDER, base / CONDITION_FOLDER):
        if not results_dir.is_dir():
            continue
        excel = results_dir / "Excel"
        for scenario, (folders, prefixes) in SCENARIO_OUTPUTS.items():
            if scenario == keep:
                continue
            found += [d for d in (results_dir / f for f in folders) if d.is_dir()]
            if not excel.is_dir():
                continue
            found += [f for f in excel.glob("*.csv")
                      if f.name.startswith(prefixes) and not f.name.startswith(keep_prefixes)]
    return found


#: What a stimulation-induced run leaves in a results directory besides its
#: scenario deliverable. Superseded when a Neurosoft export is re-read as a
#: Condition test: the two are readings of the SAME file, so the abandoned one's
#: epochs, panels and metrics are not a second analysis to keep beside the new
#: one, they are the answer that was just replaced.
SIR_MODE_OUTPUTS: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("Stimulus-centered epochs", "Plots with grid and markers",
     "Plots without grid and markers", "Plots grouped by amplitude", "Boxplots",
     "Template overlays", "Template overlays per amplitude", "Templates"),
    ("Large_dataset_emg_response_metrics", "Summary_stats_by_config"),
)


def sir_mode_output_paths(output_root: Path) -> list[Path]:
    """The stimulation-induced outputs of a Neurosoft run, in every layout.

    Only for text ``curves`` exports being re-read as a Condition test. A .fif or
    .mat root may legitimately hold a SIR and a StartStop analysis side by side;
    a Neurosoft export has exactly one reading at a time.
    """
    base = Path(output_root) / "results"
    folders, prefixes = SIR_MODE_OUTPUTS
    found: list[Path] = []
    for results_dir in (base, base / STIMULATION_INDUCED_FOLDER):
        if not results_dir.is_dir():
            continue
        found += [d for d in (results_dir / f for f in folders) if d.is_dir()]
        excel = results_dir / "Excel"
        if excel.is_dir():
            found += [f for f in excel.iterdir()
                      if f.is_file() and f.name.startswith(prefixes)]
    return found


def stale_mode_layout_paths(output_root: Path, folder: str, results_dir: Path) -> list[Path]:
    """This analysis's outputs sitting in the layout the run did NOT pick.

    ``resolve_mode_dirs`` drops the ``<mode>/`` level whenever the root holds
    nothing else, while ``find_mode_dir`` — the reader side, and the writer side
    of the scenario deliverables — prefers that level whenever it exists. So a
    root that once held a second analysis keeps a ``results/<mode>/`` tree, and
    the next run writes its epochs and tables flat while its scenario folder
    lands inside the old subfolder. The run then reads back as two disagreeing
    copies of itself, which no amount of re-running repairs: that is the state
    that could only be cleared by deleting the folder by hand.

    A run resolving to the flat layout has already established that nothing else
    lives in this root, so the leftover subfolder can only be an older run of
    this same analysis.
    """
    root = Path(output_root)
    if Path(results_dir) != root / "results":
        # This run kept the mode level (another analysis really is present), so
        # both layouts are legitimately in use and nothing here is stale.
        return []
    return [d for d in (root / "results" / folder, root / "data" / folder) if d.is_dir()]


#: Where the previous reading waits while the new one is being computed. Outside
#: ``results/`` and dot-prefixed, so no mode probe, scenario probe or folder scan
#: can mistake its contents for outputs of this run.
SUPERSEDED_DIR = ".superseded"

#: Names the store's contents, so a rollback does not have to infer them.
MANIFEST_NAME = "_superseded.json"


class SupersededOutputs:
    """The previous reading's outputs, held aside until this run succeeds.

    Switching the scenario replaces one reading of a recording with another, and
    the outputs of the old one have to be gone before the new run resolves its
    layout — they are mode markers, and leaving them visible is what pushed a
    re-run into the wrong folder. But *deleting* them at that moment means a run
    that then fails, or a scenario switched by accident, takes the previous
    results with it.

    So they are moved into ``<root>/.superseded/`` first, keeping their relative
    paths, and the run decides their fate: ``commit`` on success, ``restore`` on
    failure. A store found on entry belongs to a run that never reached either
    and is put back before anything else happens.
    """

    def __init__(self, output_root: Path):
        self.root = Path(output_root)
        self.store = self.root / SUPERSEDED_DIR
        self.moved: list[tuple[Path, Path]] = []
        self.names: list[str] = []

    def take(self, paths: list[Path]) -> None:
        """Move *paths* into the store, remembering where each came from."""
        import shutil

        for src in paths:
            src = Path(src)
            if not src.exists():
                continue
            try:
                rel = src.relative_to(self.root)
            except ValueError:
                continue
            dst = self.store / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
            shutil.move(str(src), str(dst))
            self.moved.append((src, dst))
            self.names.append(str(rel) + ("/" if dst.is_dir() else ""))
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Record what was set aside, so a crashed run can be undone next time.

        The store alone does not say it: a folder in there could be a whole tree
        that was moved or the parent of one file that was, and guessing wrong
        either loses the rest of the tree or resurrects an empty shell.
        """
        import json

        if not self.moved:
            return
        self.store.mkdir(parents=True, exist_ok=True)
        rels = [str(src.relative_to(self.root)) for src, _ in self.moved]
        (self.store / MANIFEST_NAME).write_text(
            json.dumps({"paths": rels}, ensure_ascii=False, indent=2), encoding="utf-8")

    def restore(self) -> list[str]:
        """Put everything back where it came from. Safe to call twice."""
        import shutil

        back: list[str] = []
        for src, dst in reversed(self.moved):
            if not dst.exists():
                continue
            src.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                # The failed run recreated this path; its content is a partial
                # answer and the superseded one is the answer that worked.
                shutil.rmtree(src, ignore_errors=True) if src.is_dir() else src.unlink()
            shutil.move(str(dst), str(src))
            back.append(str(src.relative_to(self.root)))
        self.moved.clear()
        self.names.clear()
        shutil.rmtree(self.store, ignore_errors=True)
        return back

    def commit(self) -> list[str]:
        """Discard the superseded outputs — the new run produced its own."""
        import shutil

        shutil.rmtree(self.store, ignore_errors=True)
        names, self.names = self.names, []
        self.moved.clear()
        return names


def restore_interrupted_supersede(output_root: Path) -> list[str]:
    """Put back a store left by a run that never finished.

    Called before a new run touches anything. Whatever is in there was the last
    complete answer for this recording, and the run that set it aside neither
    committed nor rolled it back — so it is still the answer.
    """
    import json
    import shutil

    root = Path(output_root)
    store = root / SUPERSEDED_DIR
    manifest = store / MANIFEST_NAME
    if not manifest.is_file():
        # No manifest: either no store at all, or one from a version that did not
        # write one. Leave it alone rather than guess what its shape means.
        return []
    try:
        rels = json.loads(manifest.read_text(encoding="utf-8"))["paths"]
    except (OSError, ValueError, KeyError):
        return []
    back: list[str] = []
    for rel in rels:
        src, dst = store / rel, root / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
        shutil.move(str(src), str(dst))
        back.append(rel)
    shutil.rmtree(store, ignore_errors=True)
    return back


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

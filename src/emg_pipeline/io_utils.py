"""I/O helpers for the EMG pipeline."""

from __future__ import annotations

from pathlib import Path

STIMULATION_INDUCED_FOLDER = "Stimulation-induced responses"


def build_output_dirs(output_root: Path, startstop_mode: bool = False) -> dict[str, Path]:
    """Build output directories. When startstop_mode is True only StartStop dirs are created.
    When False, stimulation-induced dirs are created under results/ (same level as StartStop analysis)."""
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = output_root / "data"
    results_dir = output_root / "results"
    startstop_dir = results_dir / "StartStop analysis"
    sir = results_dir / STIMULATION_INDUCED_FOLDER

    if startstop_mode:
        # StartStop only: create just what's needed for StartStop and shared data
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        startstop_dir.mkdir(parents=True, exist_ok=True)
        # Return placeholder paths for stim dirs (unused when startstop_mode True)
        return {
            "output_root": output_root,
            "data_dir": data_dir,
            "crops_dir": sir / "data" / "annot_crops_fif",
            "results_dir": results_dir,
            "epochs_dir": sir / "Stimulus-centered epochs",
            "excel_dir": sir / "Excel",
            "boxplot_dir": sir / "Boxplots",
            "plots_grid_dir": sir / "Plots with grid and markers",
            "plots_plain_dir": sir / "Plots without grid and markers",
            "plots_grouped_dir": sir / "Plots grouped by amplitude",
            "templates_dir": sir / "Templates",
            "startstop_dir": startstop_dir,
        }
    # Stimulation-induced flow: create folder tree under results/Stimulation-induced responses (same level as StartStop analysis)
    crops_dir = sir / "data" / "annot_crops_fif"
    epochs_dir = sir / "Stimulus-centered epochs"
    excel_dir = sir / "Excel"
    boxplot_dir = sir / "Boxplots"
    plots_grid_dir = sir / "Plots with grid and markers"
    plots_plain_dir = sir / "Plots without grid and markers"
    plots_grouped_dir = sir / "Plots grouped by amplitude"
    templates_dir = sir / "Templates"

    for path in [
        data_dir,
        results_dir,
        startstop_dir,
        sir,
        crops_dir,
        epochs_dir,
        excel_dir,
        boxplot_dir,
        plots_grid_dir,
        plots_plain_dir,
        plots_grouped_dir,
        templates_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "data_dir": sir / "data",
        "crops_dir": crops_dir,
        "results_dir": results_dir,
        "epochs_dir": epochs_dir,
        "excel_dir": excel_dir,
        "boxplot_dir": boxplot_dir,
        "plots_grid_dir": plots_grid_dir,
        "plots_plain_dir": plots_plain_dir,
        "plots_grouped_dir": plots_grouped_dir,
        "templates_dir": templates_dir,
        "startstop_dir": startstop_dir,
    }


def list_crop_files(crops_dir: Path) -> list[str]:
    files = []
    for file_path in crops_dir.iterdir():
        if not file_path.is_file():
            continue
        name = file_path.name
        if "(" in name or ")" in name:
            continue
        files.append(name)
    return files

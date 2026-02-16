"""I/O helpers for the EMG pipeline."""

from __future__ import annotations

from pathlib import Path

STIMULATION_INDUCED_FOLDER = "Stimulation-induced responses"
STARTSTOP_FOLDER = "StartStop analysis"


def build_output_dirs(output_root: Path, startstop_mode: bool = False) -> dict[str, Path]:
    """Build output directories.

    Data artifacts are stored under:
      - data/StartStop analysis/
      - data/Stimulation-induced responses/

    Analysis outputs are stored under:
      - results/StartStop analysis/
      - results/Stimulation-induced responses/
    """
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = output_root / "data"
    results_dir = output_root / "results"
    startstop_data_dir = data_root / STARTSTOP_FOLDER
    stim_data_dir = data_root / STIMULATION_INDUCED_FOLDER
    startstop_dir = results_dir / STARTSTOP_FOLDER
    stim_results_dir = results_dir / STIMULATION_INDUCED_FOLDER

    # Stimulation-induced flow directory tree.
    crops_dir = stim_data_dir / "annot_crops_fif"
    epochs_dir = stim_results_dir / "Stimulus-centered epochs"
    excel_dir = stim_results_dir / "Excel"
    boxplot_dir = stim_results_dir / "Boxplots"
    plots_grid_dir = stim_results_dir / "Plots with grid and markers"
    plots_plain_dir = stim_results_dir / "Plots without grid and markers"
    plots_grouped_dir = stim_results_dir / "Plots grouped by amplitude"
    templates_dir = stim_results_dir / "Templates"

    for path in [
        data_root,
        startstop_data_dir,
        stim_data_dir,
        results_dir,
        startstop_dir,
        stim_results_dir,
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
        "data_dir": startstop_data_dir if startstop_mode else stim_data_dir,
        "data_root": data_root,
        "startstop_data_dir": startstop_data_dir,
        "stim_data_dir": stim_data_dir,
        "crops_dir": crops_dir,
        "results_dir": results_dir,
        "stim_results_dir": stim_results_dir,
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

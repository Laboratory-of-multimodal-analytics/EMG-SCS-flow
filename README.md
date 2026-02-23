# EMG-SCS-flow

EMG processing pipeline for SCS and related protocols.  
It is designed for practical use by researchers and clinicians: load recording -> parse annotations -> detect responses -> export tables and plots.

This repository is in active development. Always validate outputs against your clinical/research ground truth before making decisions.

---


## What The Pipeline Produces

For each recording, the pipeline saves:

- Preprocessed data and cropped files by protocol annotations.
- Epoch-level response metrics (onset, P1, P2, amplitudes, PTP).
- Summary Excel tables for downstream statistics.
- QC figures (epoch grids, grouped plots, templates, boxplots).

---

## Analysis Modes

| Mode | Typical use case | Main outputs |
|------|------------------|--------------|
| `StartStop` | Start/stop protocol segments with template matching | Start/stop response summaries and plots |
| `Stimulation-induced` | Stimulus-centered epoch analysis by config/amplitude | Epoch metrics, grouped amplitude/config plots, templates |

You can select mode with:

- `startstop_mode=True/False` when calling `run_pipeline(...)`, or
- CLI flags `--startstop` / `--no-startstop`.

---

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

Install:

```bash
cd /path/to/EMG-SCS-flow
pip install -r requirements.txt
```

---

## Quick Start (Recommended: Python/Notebook)

Use this from a notebook or script. It is the most robust workflow for this repository.

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/EMG-SCS-flow")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import run_pipeline

input_path = PROJECT_ROOT / "path/to/recording.mat"   # .mat, .fif, or .edf
output_dir = PROJECT_ROOT / "results/my_run_name"

out = run_pipeline(
    input_path,
    output_dir=output_dir,
    startstop_mode=False,          # False = stimulation-induced mode
    template_mode="per_file_mean", # only used in stimulation-induced mode
)
print(out)
```

---

## Command Line Usage

From project root:

```bash
python run.py /path/to/recording.mat --no-startstop --template-mode per_file_mean --output-dir /path/to/output
```

Common options:

- `--startstop` or `--no-startstop`
- `--template-mode {per_file_mean|config_grand_avg|max_amp_only}`
- `--output-dir DIR`

---

## Input Data Expectations

- Supported files: `.edf`, `.fif`, `.mat`
- Annotations are required for meaningful analysis in both modes.
- Stimulation-induced mode expects annotation labels that encode configuration and amplitude.

Notes:

- Annotation parsing rules are implemented in `src/annotations.py`.
- If labels are inconsistent, outputs may be incomplete or grouped incorrectly.

---

## Output Structure

Inside your selected output root, the pipeline creates two branches:

- `data/...` -> intermediate/prepared data
- `results/...` -> analysis tables and figures

Mode-specific subfolders:

- `data/StartStop analysis/`
- `data/Stimulation-induced responses/`
- `results/StartStop analysis/`
- `results/Stimulation-induced responses/`

In stimulation-induced mode, check:

- `results/Stimulation-induced responses/Excel/Large_dataset_emg_response_metrics.csv`
- `results/Stimulation-induced responses/Excel/Summary_stats_by_config_amp_channel.csv`

---

## Key Settings You May Tune

Edit `src/constants.py`:

- Filter settings (`RAW_BANDPASS_*`)
- Re-referencing flags (`ARTIFACT_REREF`, `CAR_REREF`, `LATERAL_CAR_REREF`)
- Detection thresholds (prominence, amplitude/PTP minima, correlation thresholds)
- Onset logic (`STIM_USE_ONSET`, onset constraints)

Most tuning work should happen by changing constants, then rerunning on a known validation set.

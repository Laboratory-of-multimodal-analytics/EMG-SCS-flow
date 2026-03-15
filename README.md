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
- QC figures (epoch grids, grouped plots, template overlays, boxplots).

---

## Analysis Modes

| Mode | Typical use case | Main outputs |
|------|------------------|--------------|
| `StartStop` | Start/stop protocol segments with template matching | Start/stop response summaries and plots |
| `Stimulation-induced` | Stimulus-centered epoch analysis by config/amplitude | Epoch metrics, grouped amplitude/config plots, template overlays |

You can select mode with:

- `startstop_mode=True/False` when calling `run_pipeline(...)`, or
- CLI flags `--startstop` / `--no-startstop`.

---

## Stimulation-Induced Response Detection

The pipeline uses a **pre-computed template bank** (`templates/` directory) to detect stimulation-induced responses (SIR). The detection works in two passes:

1. **PASS 1 — Template matching**: For each configuration x channel, the mean waveform across all epochs is correlated against the 8 pre-computed templates at multiple time scales and polarities. The best-matching template defines the expected onset, P1, and P2 latencies.

2. **PASS 2 — Epoch-level detection**: For each individual epoch, peaks are searched near the template-derived latencies. Detections must pass amplitude thresholds (peak >= 10 µV, PTP >= 30 µV) and channel-level consistency checks (valid fraction, median epoch-template correlation, artifact correlation rejection).

Templates are stored at a fixed 2000 Hz sampling rate and are automatically resampled to the data's native rate via interpolation.

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

input_path = PROJECT_ROOT / "path/to/recording.edf"   # .mat, .fif, or .edf
output_dir = PROJECT_ROOT / "results/my_run_name"

out = run_pipeline(
    input_path,
    output_dir=output_dir,
    startstop_mode=False,          # False = stimulation-induced mode
)
print(out)
```

---

## Command Line Usage

From project root:

```bash
python run.py /path/to/recording.edf --no-startstop --output-dir /path/to/output
```

Common options:

- `--startstop` or `--no-startstop`
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
- `results/Stimulation-induced responses/Template overlays/` — mean waveform with matched template overlay per configuration

---

## Key Settings You May Tune

Edit `src/constants.py`:

- **Filtering**: `RAW_BANDPASS_L_FREQ`, `RAW_BANDPASS_H_FREQ` — set to `None` to skip bandpass/notch filtering
- **Artifact peak detection**: `ART_PEAK_HEIGHT` (V), `ART_PEAK_WIDTH_MS` (ms) — set to `None` to fall back to threshold-based detection (`THRESH`)
- **Re-referencing**: `ARTIFACT_REREF`, `CAR_REREF`, `LATERAL_CAR_REREF`
- **Epoch window**: `EPOCH_TMIN`, `EPOCH_TMAX`
- **Response window**: `RESP_TMIN`, `RESP_TMAX`
- **Amplitude thresholds**: `STIM_PEAK_AMP_MIN_UV`, `STIM_PTP_MIN_UV`, `STIM_P1_ABS_MIN_UV`
- **Template matching**: `SIR_TM_SCALES`, `SIR_TM_MIN_CORR`
- **Channel QC**: `STIM_EPOCH_ARTIFACT_CORR_REJECTION`, `STIM_EPOCH_ARTIFACT_ABS_CORR_THR`

Most tuning work should happen by changing constants, then rerunning on a known validation set.

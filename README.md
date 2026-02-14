# EMG-SCS-flow

Pipeline for processing EMG data in spinal cord stimulation and other protocols: preprocessing, event detection, epoching, and extraction of response metrics.

**Status of the toolbox:** Draft / MVP. Suitable for researchers and clinicians to run on their own data; parameters and outputs may change.

---

## Two analysis modes

| Mode | Use case | Output location |
|------|----------|-----------------|
| **StartStop** | Start/stop protocol with annotation-defined conditions and template-matched events | `results/StartStop analysis/` |
| **Stimulation-induced** | Stimulus-centered epochs from config/amplitude annotations and artifact-based event detection | `results/Stimulation-induced responses/` |

The mode is set in `src/emg_pipeline/constants.py` (`STARTSTOP_MODE = True` or `False`) or from the command line with `--startstop` / `--no-startstop`.

---

## Requirements

- Python 3.9+
- See `requirements.txt`

---

## Installation

```bash
cd /path/
pip install -r requirements.txt
```

No package install is required: run from the project root so that source code is on the path (the run script does this for you).

---

## How to run

### From the command line

From the **project root** (the folder that contains `src` and `run.py`):

```bash
# StartStop mode (template matching on start/stop segments)
python run.py path/to/your_file.fif --startstop --output-dir path/to/output

# Stimulation-induced mode (config/amplitude crops, stimulus-centered epochs)
python run.py path/to/your_file.edf --no-startstop --output-dir path/to/output

# .mat files are supported (converted to FIF internally)
python run.py path/to/your_file.mat --startstop
```

If you omit `--output-dir`, outputs are written under `results/<filename_stem>/` next to your data.

**Options:**

- `--startstop` — run in StartStop mode (template-based event detection).
- `--no-startstop` — run in stimulation-induced mode (annotation-based crops and stimulus-centered epochs).
- `--output-dir DIR` — output directory root.
- `--template-mode {per_file_mean|config_grand_avg|max_amp_only}` — (stimulation-induced mode) template for detection.

### From a Python script or notebook

```python
import sys
from pathlib import Path

# If running from another directory, add project root
ROOT = Path(__file__).resolve().parent  
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from emg_pipeline import run_pipeline

# StartStop mode
out = run_pipeline("path/to/file.fif", output_dir="path/to/out", startstop_mode=True)

# Stimulation-induced mode
out = run_pipeline("path/to/file.edf", output_dir="path/to/out", startstop_mode=False)

print(out)  # output root path
```

---

## Input data

- **Formats:** EDF, FIF, or MAT (MAT converted to FIF on the fly).
- **Annotations:** Required for both modes. StartStop uses condition names and start/stop markers; stimulation-induced uses config and amplitude labels (see `src/emg_pipeline/annotations.py`).
- **Channels:** Artifact channels can be set in `constants.py` (`ARTCHAN`). Rereferencing (artifact subtraction and/or CAR) is applied before processing; artifact channels are left unchanged for stimulus-onset detection.

---

## Outputs

- **StartStop:** `results/StartStop analysis/` — Excel (metrics, template matches), plots, templates, and (if enabled) raw epochs.
- **Stimulation-induced:** `results/Stimulation-induced responses/` — stimulus-centered epochs, Excel, boxplots, and plots.

Preprocessed raw is saved under `data/` in the corresponding output tree.

---

## Configuration

Edit `src/emg_pipeline/constants.py` to change:

- `STARTSTOP_MODE` — default analysis mode.
- Filtering, thresholds, template-matching parameters, and artifact/CAR rereferencing.



---

## License and citation

Use and cite as appropriate for your context. This is research/clinical-support software; validate results for your own protocols and equipment.

"""Reader for the text ``curves`` export (pre-cut epochs, no annotations).

Format produced by the clinical EP station, one file per recording session::

    33 - curves count (количество кривых в файле)

    1 - curves series number (номер серии кривых)
    1 - curve number (номер кривой)
    5E-05 - sampling step in seconds (шаг квантования, с)
    8 - channels count (число каналов)
    2000 - samples per one channel (число отсчетов на один канал)

    Values, Volt (значения, В):

    -0.0002660698 4.6878E-05 ... (2000 rows x 8 whitespace-separated columns)

    2 - curves series number (номер серии кривых)
    ...

Every "curve" is one already-cut stimulus epoch: the station triggers on the
stimulus, so sample 0 is the trigger and there is NO pre-stimulus data (the
first ~13 samples are a constant pad, the stimulus artifact lands at ~1 ms).
This is why the SIR pipeline skips artifact peak-finding and epoching for this
format — the epochs are handed to it ready-made.

The export carries NO channel names, NO configuration and NO amplitude labels.
Channels are therefore numbered by column position (``ch1``..``chN``) and the
whole file is treated as a single SIR crop. Column order does NOT imply muscle
identity — that has to come from the recording protocol.
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np

from .constants import TEXT_CURVES_RESP_TMAX, TEXT_CURVES_RESP_TMIN

#: Channels have no names in the export; label them by column position.
CH_PREFIX = "ch"

#: Header line that marks the start of a curve's numeric block.
_VALUES_MARKER = "Values"

#: A data row starts with a number; header lines always contain " - ".
_NUM_RE = re.compile(r"^\s*-?\d")

_CURVE_RE = re.compile(r"^\s*(\d+)\s+-\s+curve number")
_STEP_RE = re.compile(r"^\s*([\d.E+-]+)\s+-\s+sampling step")
_NCHAN_RE = re.compile(r"^\s*(\d+)\s+-\s+channels count")


def is_text_curves_file(path: str | Path) -> bool:
    """True if *path* looks like this text export (cheap header sniff)."""
    path = Path(path)
    if path.suffix.lower() != ".txt":
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(12)]
    except OSError:
        return False
    joined = "".join(head)
    return "curves count" in joined and "curve number" in joined


def read_text_curves(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Parse the export.

    Returns ``(data, sfreq, ch_names)`` where *data* is
    ``(n_curves, n_channels, n_samples)`` in **volts** (the file's native unit),
    matching the shape MNE's ``EpochsArray`` expects.
    """
    path = Path(path)
    curves: list[np.ndarray] = []
    current: list[list[float]] = []
    step_s: float | None = None
    n_chan: int | None = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if _CURVE_RE.match(line):
                if current:
                    curves.append(np.asarray(current, dtype=float))
                    current = []
                continue
            m = _STEP_RE.match(line)
            if m:
                step_s = float(m.group(1))
                continue
            m = _NCHAN_RE.match(line)
            if m:
                n_chan = int(m.group(1))
                continue
            if _VALUES_MARKER in line or " - " in line or not _NUM_RE.match(line):
                continue
            row = [float(tok) for tok in line.split()]
            if n_chan is not None and len(row) != n_chan:
                continue
            current.append(row)
    if current:
        curves.append(np.asarray(current, dtype=float))

    if not curves:
        raise ValueError(f"No curves parsed from {path}")
    if step_s is None or step_s <= 0:
        raise ValueError(f"No usable sampling step in {path}")

    lengths = {c.shape for c in curves}
    if len(lengths) != 1:
        raise ValueError(f"Curves of differing shape in {path}: {sorted(lengths)}")

    # (n_curves, n_samples, n_channels) -> (n_curves, n_channels, n_samples)
    data = np.stack(curves).transpose(0, 2, 1)
    sfreq = 1.0 / step_s
    ch_names = [f"{CH_PREFIX}{i + 1}" for i in range(data.shape[1])]
    return data, sfreq, ch_names


def text_curves_epochs(path: str | Path, tmin: float = 0.0) -> mne.EpochsArray:
    """Load the export straight into an ``EpochsArray`` (stimulus at ``tmin``)."""
    data, sfreq, ch_names = read_text_curves(path)
    info = mne.create_info(ch_names, sfreq, ["eeg"] * len(ch_names))
    return mne.EpochsArray(data, info, tmin=tmin, baseline=None, verbose=False)


def text_curves_crop_name(path: str | Path, amp_label: str = "all") -> str:
    """Synthetic crop filename for a whole-file crop.

    The SIR pipeline re-parses configuration and amplitude out of the crop
    filename (``configuration = name.split("-")[0]`` and the amplitude from the
    first ``_``), so both separators must be scrubbed from the stem or the
    configuration label would be truncated at the first hyphen. The trailing
    ``-`` then terminates the configuration token, exactly as real config
    headers like ``1+8-`` do.
    """
    stem = Path(path).stem
    safe = re.sub(r"[-_]", ".", stem)
    safe = re.sub(r"[^\w\+\. ]", "_", safe, flags=re.UNICODE)
    return f"{safe}-_{amp_label}.fif"


def text_curves_window_defaults(path: str | Path) -> dict[str, float]:
    """Epoch/baseline/response windows that fit a text ``curves`` export.

    Single source of truth for both entry points: ``run_pipeline`` applies these
    to any window argument the caller left at its module default, and the GUI
    writes them into its settings when such a file is opened, so the values on
    screen are the values that run.
    """
    data, sfreq, _ = read_text_curves(path)
    n_samples = data.shape[2]
    b_tmin, b_tmax = flat_prestim_baseline(data, sfreq=sfreq)
    return {
        "epoch_tmin": 0.0,
        "epoch_tmax": (n_samples - 1) / sfreq,
        "baseline_tmin": b_tmin,
        "baseline_tmax": b_tmax,
        "resp_tmin": TEXT_CURVES_RESP_TMIN,
        "resp_tmax": TEXT_CURVES_RESP_TMAX,
    }


def flat_prestim_baseline(path_or_data, sfreq: float | None = None) -> tuple[float, float]:
    """Time window (s) of the constant pad that precedes the stimulus artifact.

    The station pads the start of each curve with a repeated sample before the
    artifact rises. That pad is the only pre-artifact data in the file, and
    because it is *constant* its mean is an exact DC estimate (zero variance) —
    which is what baseline correction needs here. Returns ``(0.0, t_end)``.
    """
    if isinstance(path_or_data, (str, Path)):
        data, sfreq, _ = read_text_curves(path_or_data)
    else:
        data = np.asarray(path_or_data)
        if sfreq is None:
            raise ValueError("sfreq is required when passing an array")
    first = data[0]                      # (n_channels, n_samples)
    n_flat = 1
    while n_flat < first.shape[1] and np.allclose(first[:, n_flat], first[:, 0]):
        n_flat += 1
    # Keep at least one sample; end the window on the last constant sample.
    return 0.0, max(n_flat - 1, 1) / float(sfreq)

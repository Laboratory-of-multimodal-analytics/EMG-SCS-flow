"""Read what the pipeline actually wrote.

The GUI never recomputes detections — it renders the pipeline's own outputs, so what you
see is what the scripted run produces. Folder names here mirror io_utils.build_output_dirs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SIR_DIR = "Stimulation-induced responses"
SS_DIR = "StartStop analysis"

METRIC_COLS = [
    "Configuration", "Stim. amplitude", "Epoch", "Channel",
    "Onset latency", "Peak1 latency", "Peak2 latency",
    "Peak1 value", "Peak2 value", "PTP amplitude",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLS)
    df = pd.read_csv(path)
    # "Time series" holds the epoch waveform as a stringified list; we read waveforms from
    # the saved epoch files instead, and dropping it keeps the table light.
    return df.drop(columns=[c for c in ("Time series",) if c in df.columns])


# --------------------------------------------------------------------------- #
# SIR
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    """One (config, amplitude) block — the unit of work in SIR mode."""
    config: str
    amp: str
    epochs_path: Path

    @property
    def label(self) -> str:
        return f"{self.config} @ {self.amp}"


def _parse_crop_stem(stem: str) -> tuple[str, str] | None:
    """'1+2-_9-epo' -> ('1+2', '9').  Config = everything before the first '-',
    matching the pipeline's own `filename.split('-')[0]` convention. Amplitude labels are
    opaque strings ('2' != '2,0' != '02'), so they are never normalised."""
    name = stem[:-4] if stem.endswith("-epo") else stem
    name = re.sub(r"\(\d+\)$", "", name)  # duplicate-crop suffix, e.g. '..._9(1)'
    if "_" not in name:
        return None
    head, amp = name.rsplit("_", 1)
    config = head.split("-")[0]
    return config, amp


class SIRResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.results_dir = self.root / "results" / SIR_DIR
        self.metrics = _read_csv(self.results_dir / "Excel" / "Large_dataset_emg_response_metrics.csv")
        self.crops: list[Crop] = []
        epochs_dir = self.results_dir / "Stimulus-centered epochs"
        for f in sorted(epochs_dir.glob("*-epo.fif")):
            parsed = _parse_crop_stem(f.stem)
            if parsed:
                self.crops.append(Crop(parsed[0], parsed[1], f))

    @property
    def ok(self) -> bool:
        return bool(self.crops)

    def channels(self) -> list[str]:
        if not self.metrics.empty:
            return sorted(self.metrics["Channel"].dropna().unique().tolist())
        return []

    def load_epochs(self, crop: Crop) -> mne.Epochs:
        return mne.read_epochs(crop.epochs_path, preload=True, verbose="ERROR")

    def markers(self, crop: Crop, channel: str) -> pd.DataFrame:
        """Per-epoch onset/P1/P2 for one channel of one crop."""
        if self.metrics.empty:
            return pd.DataFrame(columns=METRIC_COLS)
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
            & (m["Channel"].astype(str) == channel)
        )
        return m[sel]

    def detection_count(self, crop: Crop) -> int:
        if self.metrics.empty:
            return 0
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
        )
        return int(m.loc[sel, "Peak1 latency"].notna().sum())

    def recruitment_table(self) -> pd.DataFrame:
        """Detections per (config, amplitude, channel) — the 'which crops are 0' view."""
        if self.metrics.empty:
            return pd.DataFrame()
        m = self.metrics.copy()
        m["detected"] = m["Peak1 latency"].notna()
        g = (
            m.groupby(["Configuration", "Stim. amplitude", "Channel"], dropna=False)
            .agg(
                detections=("detected", "sum"),
                epochs=("detected", "size"),
                p1_lat_ms=("Peak1 latency", lambda s: 1000 * np.nanmean(s) if s.notna().any() else np.nan),
                ptp_uv=("PTP amplitude", lambda s: np.nanmean(s) if s.notna().any() else np.nan),
            )
            .reset_index()
        )
        return g


# --------------------------------------------------------------------------- #
# StartStop
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One detected response: an annotation on the saved '<cond>_detections_raw.fif'."""
    index: int
    condition: str
    time: float          # seconds from the START of the saved segment (see `detections`)
    abs_time: float      # onset as stored, i.e. in the ORIGINAL recording's time base
    channels: list[str]  # marker description, e.g. 'ECR L+TR R'

    @property
    def label(self) -> str:
        return f"#{self.index + 1}  {self.time:7.3f} s  —  {'+'.join(self.channels)}"


class StartStopResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.results_dir = self.root / "results" / SS_DIR
        self.metrics = _read_csv(self.results_dir / "Excel" / "Large_dataset_emg_response_metrics.csv")
        self.channel_qc = _read_csv(self.results_dir / "Excel" / "STARTSTOP_channel_qc.csv")
        self.discarded = _read_csv(self.results_dir / "Excel" / "STARTSTOP_template_anchor_discarded.csv")

        self.raw_paths: dict[str, Path] = {}
        for f in sorted((self.results_dir / "Detections raw").glob("*_detections_raw.fif")):
            self.raw_paths[f.stem.replace("_detections_raw", "")] = f
        self._raw_cache: dict[str, mne.io.BaseRaw] = {}

    @property
    def ok(self) -> bool:
        return bool(self.raw_paths)

    def conditions(self) -> list[str]:
        return list(self.raw_paths)

    def raw(self, condition: str) -> mne.io.BaseRaw:
        if condition not in self._raw_cache:
            self._raw_cache[condition] = mne.io.read_raw_fif(
                self.raw_paths[condition], preload=True, verbose="ERROR"
            )
        return self._raw_cache[condition]

    def detections(self, condition: str) -> list[Detection]:
        """Detections with times made relative to the saved segment.

        The saved .fif is the CONCATENATED start segment (e.g. 11 s long), but its
        annotation onsets are kept in the original recording's time base (e.g. 1023 s).
        Subtracting `first_time` is what puts a marker back on the data it belongs to —
        without it every detection lands far past the end of the file.
        """
        raw = self.raw(condition)
        offset = float(raw.first_time)
        out: list[Detection] = []
        for i, ann in enumerate(raw.annotations):
            desc = str(ann["description"])
            abs_t = float(ann["onset"])
            out.append(Detection(i, condition, abs_t - offset, abs_t, desc.split("+")))
        return out

    def why_empty(self, condition: str) -> pd.DataFrame:
        """Rejected template anchors with the pipeline's own reason — the 'why did I find
        nothing here' view, straight from STARTSTOP_template_anchor_discarded.csv."""
        if self.discarded.empty or "Reason" not in self.discarded.columns:
            return pd.DataFrame()
        d = self.discarded
        col = "Configuration" if "Configuration" in d.columns else None
        if col:
            d = d[d[col].astype(str) == condition]
        return d.groupby(["Channel", "Reason"]).size().reset_index(name="n")


# --------------------------------------------------------------------------- #
# Spontaneous EMG (lives inside the StartStop tree)
# --------------------------------------------------------------------------- #
class SpontaneousResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.base = self.root / "results" / SS_DIR / "Spontaneous EMG"

    @property
    def ok(self) -> bool:
        return self.base.exists() and any(self.base.iterdir())

    def conditions(self) -> list[str]:
        if not self.base.exists():
            return []
        return sorted(d.name for d in self.base.iterdir() if d.is_dir())

    def bursts(self, condition: str) -> pd.DataFrame:
        p = self.base / condition / "Excel" / f"Spontaneous_EMG_bursts_{condition}.csv"
        return _read_csv(p) if p.exists() else pd.DataFrame()

    def summary(self, condition: str) -> pd.DataFrame:
        p = self.base / condition / "Excel" / f"Spontaneous_EMG_summary_{condition}.csv"
        return _read_csv(p) if p.exists() else pd.DataFrame()

    def envelope_files(self, condition: str) -> list[Path]:
        d = self.base / condition / "Envelopes"
        return sorted(d.glob("*.txt")) if d.exists() else []

    @staticmethod
    def read_envelope(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Tab-delimited: (time_from_burst_center_s | time_s), rms_uV."""
        df = pd.read_csv(path, sep="\t")
        return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)

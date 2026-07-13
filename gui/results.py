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
    # "Time series" holds every epoch's waveform as a stringified list, which is what makes
    # these files 50-120 MB. Skipping the column at PARSE time (not after) is the difference
    # between seconds and a minute on a network drive; waveforms come from the epoch .fif
    # files anyway.
    try:
        return pd.read_csv(path, usecols=lambda c: c != "Time series")
    except ValueError:
        return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Finding already-processed recordings on disk
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedRun:
    """One output root that already holds results.

    Deliberately cheap to build: no metrics CSV is read here. Those files run to 50-120 MB
    (the 'Time series' column), and reading one per run would make a scan take minutes.
    """
    root: Path
    mode: str            # "sir" | "startstop"
    name: str
    n_items: int         # crops (SIR) or saved conditions (StartStop)
    has_spontaneous: bool
    mtime: float
    size_mb: float

    @property
    def mode_label(self) -> str:
        return "Stimulation-induced" if self.mode == "sir" else "StartStop"

    @property
    def items_label(self) -> str:
        return "crops" if self.mode == "sir" else "conditions"


def detect_mode(root: Path) -> str | None:
    """Which mode produced this output root — judged by CONTENT, not by folder names.

    `build_output_dirs` creates both trees on every run, so an empty
    'Stimulation-induced responses/' folder proves nothing.
    """
    root = Path(root)
    sir_epochs = root / "results" / SIR_DIR / "Stimulus-centered epochs"
    sir_csv = root / "results" / SIR_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv"
    if (sir_epochs.exists() and any(sir_epochs.glob("*-epo.fif"))) or sir_csv.exists():
        return "sir"

    ss_raw = root / "results" / SS_DIR / "Detections raw"
    ss_csv = root / "results" / SS_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv"
    if (ss_raw.exists() and any(ss_raw.glob("*.fif"))) or ss_csv.exists():
        return "startstop"
    return None


# Never descend into these: they are the pipeline's own output internals and can hold
# thousands of PNGs. On a network drive (Google Drive) walking them costs minutes.
_PRUNE = {
    "data", "results", "review", "templates", "Excel", "Boxplots", "Envelopes",
    "Plots", "Raw epochs", "Templates", "Detections raw", "Spontaneous EMG",
    "Stimulus-centered epochs", "Template overlays", "annot_crops_fif", "__pycache__",
}


def scan_processed(base: Path, max_depth: int = 4) -> list[ProcessedRun]:
    """Walk a directory tree and list every output root that already holds results.

    Uses os.scandir (whose is_dir() comes from the directory entry, not a separate stat per
    file) and prunes the pipeline's own output folders — otherwise a run's thousands of
    plot PNGs turn a scan of a Drive folder into minutes of stat calls.
    """
    import os

    base = Path(base)
    found: list[ProcessedRun] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(os.scandir(d))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return

        subdirs = [
            e for e in entries
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
        ]
        if any(e.name == "results" for e in subdirs):
            mode = detect_mode(d)
            if mode:
                found.append(_describe(d, mode))
                return  # an output root never contains another one

        for e in subdirs:
            if e.name not in _PRUNE:
                walk(Path(e.path), depth + 1)

    walk(base, 0)
    found.sort(key=lambda r: r.mtime, reverse=True)
    return found


def _describe(root: Path, mode: str) -> ProcessedRun:
    """Summarise a run from directory metadata alone — stat and listings, never a CSV read."""
    if mode == "sir":
        items = root / "results" / SIR_DIR / "Stimulus-centered epochs"
        pattern = "*-epo.fif"
        csv = root / "results" / SIR_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv"
    else:
        items = root / "results" / SS_DIR / "Detections raw"
        pattern = "*.fif"
        csv = root / "results" / SS_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv"

    n_items = len(list(items.glob(pattern))) if items.exists() else 0

    size_mb, mtime = 0.0, 0.0
    if csv.exists():
        st = csv.stat()
        size_mb, mtime = st.st_size / 1e6, st.st_mtime
    if not mtime:
        mtime = root.stat().st_mtime

    spont = root / "results" / SS_DIR / "Spontaneous EMG"
    return ProcessedRun(
        root=root,
        mode=mode,
        name=root.name,
        n_items=n_items,
        has_spontaneous=spont.is_dir() and next(spont.iterdir(), None) is not None,
        mtime=mtime,
        size_mb=size_mb,
    )


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

    def detection_count(self, crop: Crop, session=None) -> int:
        """Detections in a crop. With a session, rejected ones stop counting immediately —
        no re-run needed for the number to reflect what you just marked."""
        if self.metrics.empty:
            return 0
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
        )
        rows = m[sel]
        if session is not None and not rows.empty:
            # A plain list of bools would be read as column labels by pandas 2.x, so the mask
            # has to be an ndarray.
            keep = np.array([
                not self._is_rejected(session, crop.config, crop.amp, str(ch))
                for ch in rows["Channel"]
            ], dtype=bool)
            rows = rows[keep]
        if "Peak1 latency" not in rows.columns:
            return 0
        return int(rows["Peak1 latency"].notna().sum())

    @staticmethod
    def _is_rejected(session, config: str, amp: str, channel: str) -> bool:
        return (
            (config, amp, channel) in session.suppress_keys
            or channel in session.exclude_channels
            or config in session.exclude_configs
        )

    def channel_summary(self, crop: Crop, session=None) -> pd.DataFrame:
        """Per-channel view of one crop, for the live table under the plot."""
        if self.metrics.empty:
            return pd.DataFrame()
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
        )
        rows = []
        for ch, grp in m[sel].groupby("Channel"):
            rejected = session is not None and self._is_rejected(session, crop.config, crop.amp, str(ch))
            det = 0 if rejected else int(grp["Peak1 latency"].notna().sum())
            p1 = grp["Peak1 latency"].to_numpy(dtype=float)
            ptp = grp["PTP amplitude"].to_numpy(dtype=float)
            status = "rejected" if rejected else ("detected" if det else "—")
            if session is not None and (crop.config, crop.amp, str(ch)) in session.force_keys:
                status = "whitelisted"
            rows.append({
                "Channel": str(ch),
                "Detections": det,
                "Epochs": int(len(grp)),
                "P1 (ms)": np.nan if rejected or not np.isfinite(p1).any() else 1000 * np.nanmean(p1),
                "PTP (µV)": np.nan if rejected or not np.isfinite(ptp).any() else np.nanmean(ptp),
                "Status": status,
            })
        return pd.DataFrame(rows)

    def recruitment_table(self, session=None) -> pd.DataFrame:
        """Detections per (config, amplitude, channel) — the 'which crops are 0' view.

        With a session, manual rejections are applied here too, so the table and the plots
        always tell the same story.
        """
        if self.metrics.empty:
            return pd.DataFrame()
        m = self.metrics.copy()
        m["detected"] = m["Peak1 latency"].notna()
        if session is not None:
            rejected = np.array([
                self._is_rejected(session, str(c), str(a), str(ch))
                for c, a, ch in zip(m["Configuration"], m["Stim. amplitude"], m["Channel"])
            ], dtype=bool)
            m.loc[rejected, "detected"] = False
            m.loc[rejected, ["Peak1 latency", "PTP amplitude"]] = np.nan
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

    def mean_wave(self, crop: Crop, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Times (s) and the mean epoch waveform (µV) — the curve a template is built from."""
        epochs = self.load_epochs(crop)
        data = epochs.get_data(picks=[channel])[:, 0, :] * 1e6
        return epochs.times, data.mean(axis=0)


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

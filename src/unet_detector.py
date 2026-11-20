"""The network detector: an EMG-SCS-detector model applied to one crop's epochs.

Stimulation-induced mode has two interchangeable detectors (DETECTOR in
constants.py). Both answer the same question per curve and channel — is there a
response, and where are its P1 and P2 — and the pipeline treats their answers
the same way afterwards. This module is the network side.

The network itself is not reimplemented here. The model is run by the
detector's own ``predict.predict`` on the epochs this pipeline has just saved,
so the toolbox gives exactly the answer ``predict.py`` gives on the same file:
the ensemble's p(response), its onset/P1/P2 (with the model's peak polishing),
the doctors' five criteria, and the queue of curves the network doubts. Only the
queue thresholds are this toolbox's (UNET_REVIEW_* in constants.py).

``predict`` reads an epochs file and refuses paths on Google Drive, where results
usually live next to the recording, so the crop is handed over through a local
temporary copy.

Where the network may run is decided once per recording (``why_not``): it was
trained on Neurosoft curve exports — 20 kHz, stimulus at t = 0, 60 ms read — of
recruitment, Jendrassik and paired-stimulation runs. Anything else goes to the
template detector, and the reason is printed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNET = "unet"
TEMPLATES = "templates"
DETECTORS = (UNET, TEMPLATES)

#: The network's input contract (data/preprocess.py of the detector).
SOURCE_SFREQ = 20000.0
WINDOW_SAMPLES = 1200          # 60 ms at 20 kHz

#: The queue's reasons about the peaks rather than the response (post/queue.py).
PEAK_REASONS = ("сети расходятся в пике", "две волны")

#: The detector's per-curve doubt measures, as predict.py names them.
MEASURES = ("p_response", "nets_yes_share", "P1_spread_ms", "P2_spread_ms",
            "P1_rival", "P2_rival")


@dataclass(frozen=True)
class ReviewSettings:
    """Thresholds of the network's doctor queue (post/queue.QueueConfig fields)."""
    p_margin: float | None = None
    vote_split: tuple | None = None
    spread_ms: float | None = 2.0
    rival_ratio: float | None = 0.80
    criteria_doubt: bool = False
    #: the peaks whose spread across the nets counts; the others' is ignored
    spread_peaks: tuple = ("P1",)
    #: the peaks whose heatmap is checked for two waves; the others' is ignored
    rival_peaks: tuple = ("P1",)


def detector_dir(configured: str | os.PathLike | None = "") -> Path:
    """The EMG-SCS-detector checkout: the setting, then $EMG_SCS_DETECTOR, then a sibling folder."""
    for cand in (configured, os.environ.get("EMG_SCS_DETECTOR", "")):
        if cand:
            return Path(cand).expanduser()
    return Path(__file__).resolve().parents[2] / "EMG-SCS-detector"


def model_dir(configured: str | os.PathLike | None = "",
              detector: Path | None = None) -> Path:
    """The model folder: the setting, else <detector>/models/unet_mix."""
    if configured:
        return Path(configured).expanduser()
    return (detector or detector_dir()) / "models" / "unet_mix"


def model_name(model: Path) -> str:
    return Path(model).name


def missing(detector: Path, model: Path) -> str | None:
    """Why the network cannot be loaded at all, or None."""
    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed in this environment (pip install torch)"
    if not (Path(detector) / "predict.py").is_file():
        return (f"no EMG-SCS-detector code at {detector} (set UNET_DETECTOR_DIR or "
                f"the EMG_SCS_DETECTOR environment variable)")
    if not (Path(model) / "config.json").is_file() or not (Path(model) / "nets").is_dir():
        return f"no model at {model} (config.json + nets/; set UNET_MODEL_DIR)"
    return None


def input_mismatch(epochs) -> str | None:
    """Why these epochs are not what the network was trained on, or None."""
    sfreq = float(epochs.info["sfreq"])
    if not np.isclose(sfreq, SOURCE_SFREQ):
        return f"epochs at {sfreq:g} Hz, the network reads {SOURCE_SFREQ:g} Hz"
    if not np.isclose(float(epochs.times[0]), 0.0, atol=1e-9):
        return (f"epochs start at {epochs.times[0] * 1e3:.1f} ms, the network needs them "
                f"to start at the stimulus")
    if len(epochs.times) < WINDOW_SAMPLES:
        return (f"epochs are {len(epochs.times) / sfreq * 1e3:.0f} ms long, the network "
                f"reads {WINDOW_SAMPLES / SOURCE_SFREQ * 1e3:.0f} ms")
    return None


def why_not(*, pre_epoched, scenario, net_scenarios, detector: Path, model: Path) -> str | None:
    """Why the network is not used on this recording, or None when it is.

    ``pre_epoched`` is the pipeline's {crop: epochs} of a curve export (None for
    a continuous recording), ``scenario`` its Neurosoft scenario.
    """
    if pre_epoched is None:
        return ("the network was trained on Neurosoft curve exports only; "
                "this is a continuous recording")
    if scenario not in net_scenarios:
        return f"the network was not trained on the {scenario} scenario"
    for ep in pre_epoched.values():
        bad = input_mismatch(ep)
        if bad:
            return bad
    return missing(detector, model)


# --------------------------------------------------------------------------- #
# Loading: once per model folder and process. The GUI runs the pipeline on a
# worker thread and a batch runs file after file, so the nets stay in memory.
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_LOADED: dict[tuple[str, str], tuple] = {}


def _load(detector: Path, model: Path):
    key = (str(Path(detector).resolve()), str(Path(model).resolve()))
    with _LOCK:
        if key not in _LOADED:
            if key[0] not in sys.path:
                sys.path.insert(0, key[0])
            import predict as det_predict
            from ensemble import Ensemble, Families
            from post import queue as det_queue

            cfg = det_predict.ModelConfig.load(key[1])
            nets = os.path.join(key[1], "nets")
            ens = (Ensemble.from_dir(nets, "cpu") if cfg.families is None
                   else Families.from_dir(nets, "cpu"))
            _LOADED[key] = (det_predict, det_queue, cfg, ens)
        return _LOADED[key]


def detect(epochs, *, detector: Path, model: Path, review: ReviewSettings) -> pd.DataFrame:
    """One crop -> the network's answer, one row per (curve, channel).

    Columns as in predict.py: ``Curve`` (1-based), ``Channel``, ``Response``,
    ``Onset ms``, ``P1 ms``, ``P2 ms``, ``P1 uV``, ``P2 uV``, ``PTP uV`` (filled only
    where Response), the doubt measures, ``failed criteria``, plus ``queue`` (bool)
    and ``queue reasons`` (Russian, shown to the doctor) from this toolbox's
    thresholds.
    """
    det_predict, det_queue, cfg, ens = _load(detector, model)
    with tempfile.TemporaryDirectory(prefix="emgflow_unet_") as tmp:
        path = Path(tmp) / "crop-epo.fif"
        epochs.save(path, overwrite=True, verbose="ERROR")
        table = det_predict.predict(str(path), str(model), queue=True, ensemble=ens)

    # The queue again, with this toolbox's thresholds. predict.py builds it from the
    # model's config.json; the measures it is built from are in the table already.
    # A peak left out of spread_peaks / rival_peaks has that measure blanked for the
    # queue only (NaN never reaches a threshold); the table keeps the measured value.
    measures = table[list(MEASURES)].copy()
    for pk in ("P1", "P2"):
        if pk not in tuple(review.spread_peaks):
            measures[f"{pk}_spread_ms"] = np.nan
        if pk not in tuple(review.rival_peaks):
            measures[f"{pk}_rival"] = np.nan
    failed = table["failed criteria"].fillna("").astype(str).to_numpy()
    q = det_queue.build_queue(
        measures, table["Response"].to_numpy(bool), failed == "",
        float(cfg.threshold),
        det_queue.QueueConfig(p_margin=review.p_margin,
                              vote_split=tuple(review.vote_split) if review.vote_split else None,
                              spread_ms=review.spread_ms, rival_ratio=review.rival_ratio,
                              criteria_doubt=bool(review.criteria_doubt)),
        failed=failed)
    # Doubt about where the peaks are is only worth a look on a curve the network
    # answered "response": without one there are no peaks to check. The detector
    # asks for p >= threshold only, so a curve whose peaks then failed the
    # criteria would be listed for its peaks with no response on it.
    response = table["Response"].to_numpy(bool)
    reasons = []
    for text, yes in zip(q["reasons"], response):
        parts = [x for x in str(text).split("; ") if x]
        if not yes:
            parts = [x for x in parts if not x.startswith(PEAK_REASONS)]
        reasons.append("; ".join(parts))
    table["queue"] = np.array([bool(r) for r in reasons])
    table["queue reasons"] = reasons
    table["threshold"] = float(cfg.threshold)
    return table

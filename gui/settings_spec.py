"""Declarative registry of every pipeline lever the GUI exposes.

One entry per knob. `target` says HOW the value reaches the pipeline:

  "kwarg"   -> passed to run_pipeline(...) as a keyword argument
  "global"  -> assigned onto the src.pipeline module (P.<NAME> = value).
               NOTE: pipeline.py does `from .constants import (...)`, i.e. binds by
               value at import time, so patching src.constants has no effect.
  "default" -> bound as a *function default argument* inside pipeline.py and never
               passed by its caller. Neither of the above works; we rebind the
               function's __defaults__ instead (see session.apply_to_pipeline).

`mode` limits the lever to a branch: "sir", "startstop", or "both".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Lever:
    name: str
    label: str
    kind: str  # bool | int | float | str | choice | floats | opt_float
    default: Any
    group: str
    mode: str = "both"
    target: str = "global"
    doc: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)
    # for "default" levers: the pipeline function whose signature holds the value
    owner: str = ""


# --------------------------------------------------------------------------- #
# Shared: IO, filtering, artifact, re-referencing
# --------------------------------------------------------------------------- #
SHARED: list[Lever] = [
    Lever("raw_notch_freq", "Notch base (Hz)", "opt_float", 50.0, "Filtering", "both", "kwarg",
          "Full harmonic comb up to Nyquist. Empty = no notch."),
    Lever("raw_bandpass_l_freq", "Band-pass low (Hz)", "opt_float", 60.0, "Filtering", "both", "kwarg",
          "Empty = skip the low edge. Empty on BOTH edges = notch-only."),
    Lever("raw_bandpass_h_freq", "Band-pass high (Hz)", "opt_float", 300.0, "Filtering", "both", "kwarg",
          "Empty = skip the high edge."),
    Lever("MAT_DIVIDE_BY_1000", "MAT: divide by 1000", "bool", True, "Filtering", "both", "global",
          "LabChart unit rescaling on .mat load."),

    Lever("ARTCHAN", "Artifact channel(s)", "str", "", "Artifact", "both", "global",
          "Comma-separated. Empty = auto-detect by name containing 'art'. "
          "StartStop tolerates no artifact channel; SIR raises."),
    Lever("THRESH", "Adaptive artifact threshold (x std)", "float", 4.0, "Artifact", "sir", "global",
          "Used when the explicit peak height is empty."),
    Lever("art_min_distance_ms", "Min distance between stimuli (ms)", "float", 100.0, "Artifact", "sir", "kwarg", ""),

    Lever("ARTIFACT_REREF", "Artifact re-reference", "bool", False, "Re-referencing", "both", "global",
          "Subtract the artifact channel from every EMG channel."),
    Lever("CAR_REREF", "Common average reference", "bool", False, "Re-referencing", "both", "global", ""),
    Lever("LATERAL_CAR_REREF", "Lateral (L/R) CAR", "bool", False, "Re-referencing", "both", "global",
          "Only has any effect when CAR is ON — it is read inside the CAR routine. "
          "Side is inferred from channel-name suffixes (_l / _r / trailing ' l' / ' r')."),
]

# --------------------------------------------------------------------------- #
# SIR (stimulation-induced) mode
# --------------------------------------------------------------------------- #
SIR: list[Lever] = [
    Lever("epoch_tmin", "Epoch start (s)", "float", -0.05, "Windows", "sir", "kwarg", ""),
    Lever("epoch_tmax", "Epoch end (s)", "float", 0.10, "Windows", "sir", "kwarg", ""),
    Lever("baseline_tmin", "Baseline start (s)", "float", -0.04, "Windows", "sir", "kwarg", ""),
    Lever("baseline_tmax", "Baseline end (s)", "float", -0.01, "Windows", "sir", "kwarg", ""),
    Lever("resp_tmin", "Response search start (s)", "float", -0.01, "Windows", "sir", "kwarg", ""),
    Lever("resp_tmax", "Response search end (s)", "float", 0.02, "Windows", "sir", "kwarg", ""),

    Lever("SIR_TEMPLATE_PER_AMPLITUDE", "Template per amplitude", "bool", True, "Templates (PASS1)", "sir", "global",
          "Off = one template per (config, channel)."),
    Lever("sir_template_ref", "Reference crop", "choice", "max_amp", "Templates (PASS1)", "sir", "kwarg",
          "Which crop the template is built from.", ("max_amp", "max_resp")),
    Lever("sir_ref_method", "Reference alignment", "choice", "single", "Templates (PASS1)", "sir", "kwarg",
          "", ("single", "per_crop", "consensus")),
    Lever("sir_ref_exclude_top", "Exclude top-N crops from reference", "int", 0, "Templates (PASS1)", "sir", "kwarg", ""),
    Lever("SIR_TM_MIN_CORR", "Min template correlation", "float", 0.7, "Templates (PASS1)", "sir", "global", ""),
    Lever("sir_min_rel_to_template", "Min peak size vs template", "float", 0.1, "Templates (PASS1)", "sir", "kwarg", ""),

    Lever("sir_align_artifact", "Align epochs to artifact", "bool", False, "Alignment", "sir", "kwarg", ""),
    Lever("sir_align_win_ms", "Alignment window (ms)", "float", 8.0, "Alignment", "sir", "kwarg", ""),
    Lever("sir_align_max_shift_ms", "Max alignment shift (ms)", "float", 6.0, "Alignment", "sir", "kwarg", ""),
    Lever("sir_align_first_peak", "Centre on artifact first peak", "bool", False, "Alignment", "sir", "kwarg", ""),
    Lever("sir_align_first_peak_frac", "First-peak fraction", "float", 0.4, "Alignment", "sir", "kwarg", ""),
    Lever("SIR_ALIGN_ONSET", "Centre on artifact ONSET", "bool", False, "Alignment", "sir", "global",
          "Shape-independent landmark: fixes over/undershoot when the biggest artifact peak "
          "differs between crops."),
    Lever("SIR_ALIGN_SKIP_XCORR", "Skip cross-correlation", "bool", False, "Alignment", "sir", "global",
          "The xcorr locks events onto the dominant artifact peak and smears a small leading "
          "bump; skip it to keep a clean first bump to lock onto."),

    Lever("STIM_PEAK_AMP_MIN_UV", "Min peak amplitude (uV)", "float", 3.0, "Gates (PASS2)", "sir", "global", ""),
    Lever("STIM_PTP_MIN_UV", "Min peak-to-peak (uV)", "float", 5.0, "Gates (PASS2)", "sir", "global", ""),
    Lever("STIM_P1_ABS_MIN_UV", "Min |P1| (uV)", "float", 3.0, "Gates (PASS2)", "sir", "global", ""),
    Lever("MIN_VALID_EPOCHS", "Min valid epochs", "int", 5, "Gates (PASS2)", "sir", "global", ""),
    Lever("sir_require_both_peaks", "Require both P1 and P2", "bool", False, "Gates (PASS2)", "sir", "kwarg", ""),
    Lever("STIM_MIN_INTER_TRIAL_CORR", "Min inter-trial corr", "opt_float", 0.7, "Gates (PASS2)", "sir", "global",
          "Empty disables the gate."),
    Lever("STIM_CHANNEL_MIN_MEDIAN_CORR", "Min channel median corr", "float", 0.5, "Gates (PASS2)", "sir", "global", ""),
    Lever("STIM_CHANNEL_MIN_VALID_FRAC", "Min channel valid fraction", "float", 0.3, "Gates (PASS2)", "sir", "global", ""),
    Lever("sir_hard_min_valid_frac", "Hard min valid fraction", "float", 0.0, "Gates (PASS2)", "sir", "kwarg", ""),
    Lever("STIM_ONSET_MAX_DEV_S", "Max onset deviation (s)", "float", 0.003, "Gates (PASS2)", "sir", "global", ""),
    Lever("STIM_EPOCH_ARTIFACT_CORR_REJECTION", "Reject epochs correlating with artifact", "bool", False,
          "Gates (PASS2)", "sir", "global", ""),
    Lever("STIM_EPOCH_ARTIFACT_ABS_CORR_THR", "Artifact |corr| threshold", "float", 0.5, "Gates (PASS2)", "sir", "global", ""),

    Lever("SIR_P1_DOMINANT", "P1 = dominant deflection", "bool", False, "Markers", "sir", "global",
          "Instead of the template's first marker. For responses that lead with a small "
          "artifact-ish bump before the true larger deflection."),
    Lever("SIR_P1_DOMINANT_FRAC", "Dominant fraction", "float", 0.85, "Markers", "sir", "global",
          "P1 = earliest peak whose |amp| >= this fraction of the window max."),
    Lever("SIR_FORCE_P2_WIN_MS", "Forced-P1 P2 half-window (ms)", "float", 999.0, "Markers", "sir", "global",
          "Keeps the green P2 adjacent to a forced red P1 instead of jumping to a large late component."),
]

# --------------------------------------------------------------------------- #
# StartStop mode
# --------------------------------------------------------------------------- #
STARTSTOP: list[Lever] = [
    Lever("STARTSTOP_EPOCH_TMIN", "Epoch start (s)", "float", -0.1, "Windows", "startstop", "global", ""),
    Lever("STARTSTOP_EPOCH_TMAX", "Epoch end (s)", "float", 0.1, "Windows", "startstop", "global", ""),
    Lever("STARTSTOP_BASELINE_TMIN", "Baseline start (s)", "float", -0.05, "Windows", "startstop", "global", ""),
    Lever("STARTSTOP_BASELINE_TMAX", "Baseline end (s)", "float", -0.02, "Windows", "startstop", "global", ""),
    Lever("STARTSTOP_SIM_TMIN", "Similarity window start (s)", "float", -0.02, "Windows", "startstop", "global", ""),
    Lever("STARTSTOP_SIM_TMAX", "Similarity window end (s)", "float", 0.02, "Windows", "startstop", "global", ""),

    Lever("STARTSTOP_TM_SCORE_THR", "Min match score", "float", 0.6, "Template matching", "startstop", "global",
          "Kept moderate on purpose: the per-channel inter-trial gate rejects channels firing "
          "on irregular bursts, so a lower detection threshold improves recall."),
    Lever("STARTSTOP_TM_TEMPLATE_CORR_THR", "Min template correlation", "float", 0.6,
          "Template matching", "startstop", "global", ""),
    Lever("STARTSTOP_TM_SCALES", "Time-scale factors", "floats", (0.95, 1.0, 1.05, 1.1, 1.2),
          "Template matching", "startstop", "global", "Comma-separated."),
    Lever("STARTSTOP_TM_MATCH_TIGHT_WINDOW", "Tight match window", "bool", True,
          "Template matching", "startstop", "global",
          "Crop each template variant to onset..P2 (+pad). Off = correlate the full template, "
          "whose flat tails overlap neighbours in dense trains and kill the correlation."),
    Lever("STARTSTOP_TM_MATCH_WINDOW_PAD_MS", "Tight window pad (ms)", "float", 15.0,
          "Template matching", "startstop", "global", ""),
    Lever("STARTSTOP_MIN_DIST_MS", "Refractory between events (ms)", "float", 50.0,
          "Template matching", "startstop", "global", ""),
    Lever("STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS", "Min distance between score peaks (ms)", "float", 50.0,
          "Template matching", "startstop", "global", ""),
    Lever("STARTSTOP_TM_MATCH_PEAK_PROMINENCE", "Min score-peak prominence", "float", 0.02,
          "Template matching", "startstop", "global", "0 disables."),
    Lever("STARTSTOP_TM_MAX_MATCHES_PER_CHANNEL", "Max matches per channel", "int", 0,
          "Template matching", "startstop", "global", "0 = no cap."),
    Lever("STARTSTOP_TM_TOP_K_CANDIDATES", "Global top-K candidates", "int", 0,
          "Template matching", "startstop", "global", "0 = keep all."),

    Lever("STARTSTOP_LEAKAGE_CORR_REJECTION", "Leakage rejection", "bool", True, "Gates", "startstop", "global",
          "Drop a detection whose +/-40 ms window around P1 correlates >0.7 with more than 3 "
          "other channels. NOTE: those three numbers are hardcoded in pipeline.py, not settings."),
    Lever("STARTSTOP_CHANNEL_MIN_VALID_FRAC", "Min channel valid fraction", "float", 0.7, "Gates", "startstop", "global", ""),
    Lever("STARTSTOP_CHANNEL_MIN_MEDIAN_CORR", "Min channel median corr", "float", 0.7, "Gates", "startstop", "global", ""),
    Lever("STARTSTOP_CHANNEL_MIN_INTERTRIAL_CORR", "Min channel inter-trial corr", "opt_float", 0.6,
          "Gates", "startstop", "global",
          "A real response is stereotyped across trials; a channel firing on irregular EMG "
          "bursts is self-inconsistent. Empty disables."),
    Lever("STARTSTOP_EPOCH_MIN_INTERTRIAL_CORR", "Min epoch inter-trial corr", "opt_float", 0.6,
          "Gates", "startstop", "global", "Drops individual inconsistent detections. Empty disables."),
    Lever("STARTSTOP_MIN_DETECTIONS_PER_CHANNEL", "Min detections per channel", "int", 5, "Gates", "startstop", "global",
          "Fewer than this in a condition -> discard them all. 0 disables."),
    Lever("STARTSTOP_TM_NO_POSTHOC_CHECKS", "Skip post-hoc amplitude/PTP wipes", "bool", True,
          "Gates", "startstop", "global", ""),

    Lever("STARTSTOP_SAVE_DETECTION_MARKERS", "Save annotated .fif of detections", "bool", True,
          "Output", "startstop", "global",
          "Writes 'Detections raw/<cond>_detections_raw.fif' — the file the epoch browser reads."),
    Lever("STARTSTOP_DETECTION_MARKER_MERGE_MS", "Merge markers within (ms)", "float", 15.0,
          "Output", "startstop", "global",
          "Detections on different channels inside this window become one marker, e.g. 'ECR L+TR R'."),
]

# --------------------------------------------------------------------------- #
# Spontaneous EMG (runs inside StartStop mode only)
# --------------------------------------------------------------------------- #
SPONTANEOUS: list[Lever] = [
    Lever("SPONTANEOUS_EMG_ANALYSIS", "Run spontaneous-EMG analysis", "bool", True,
          "Spontaneous", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_HP_FREQ", "High-pass (Hz)", "opt_float", 20.0, "Spontaneous", "startstop", "global",
          "Applied INSIDE this analysis only — the detection path stays notch-only because a "
          "high-pass distorts the stereotyped peak shapes."),
    # These three are function defaults of _run_spontaneous_emg_analysis and are never passed
    # by the caller: neither a module-global patch nor a kwarg reaches them. We rebind the
    # function's __defaults__ instead.
    Lever("SPONTANEOUS_EMG_WINDOW_MS", "RMS window (ms)", "float", 200.0, "Spontaneous", "startstop", "default",
          "Non-overlapping windows for the RMS / mean-amplitude table.", owner="_run_spontaneous_emg_analysis"),
    Lever("SPONTANEOUS_EMG_ENV_HOP_MS", "Envelope hop (ms)", "float", 25.0, "Spontaneous", "startstop", "default",
          "Step of the smooth RMS envelope.", owner="_run_spontaneous_emg_analysis"),
    Lever("SPONTANEOUS_EMG_UV_SCALE", "uV scale", "float", 1e6, "Spontaneous", "startstop", "default",
          "Reporting unit only.", owner="_run_spontaneous_emg_analysis"),

    Lever("SPONTANEOUS_EMG_BURST_DETECTION", "Detect bursts", "bool", True, "Bursts", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_BURST_MIN_SNR", "Min burst SNR (x baseline)", "float", 3.0, "Bursts", "startstop", "global",
          "Scale-invariant: a flat channel has peak ~ baseline."),
    Lever("SPONTANEOUS_EMG_BURST_MIN_PEAK_UV", "Min burst peak (uV)", "float", 60.0, "Bursts", "startstop", "global",
          "Absolute floor. This is what separates a genuinely active muscle from a "
          "low-amplitude noisy channel with a good SNR ratio. Tune per gain/montage."),
    Lever("SPONTANEOUS_EMG_BURST_BASELINE_PCT", "Baseline percentile", "float", 20.0, "Bursts", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_BURST_PEAK_PCT", "Peak percentile", "float", 99.0, "Bursts", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_BURST_THRESH_FRAC", "Extent threshold fraction", "float", 0.30, "Bursts", "startstop", "global",
          "Threshold = baseline + frac * (peak - baseline). Defines burst EXTENT."),
    Lever("SPONTANEOUS_EMG_BURST_KEEP_FRAC", "Keep (hysteresis) fraction", "float", 0.50, "Bursts", "startstop", "global",
          "A burst is kept only if its peak reaches this fraction. Must be >= extent fraction."),
    Lever("SPONTANEOUS_EMG_BURST_MIN_MS", "Min burst duration (ms)", "float", 300.0, "Bursts", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_BURST_MERGE_MS", "Merge gaps shorter than (ms)", "float", 300.0, "Bursts", "startstop", "global", ""),
    Lever("SPONTANEOUS_EMG_ACTIVE_MAD_K", "Active-channel MAD k", "opt_float", 3.0, "Bursts", "startstop", "global",
          "A burst-less channel whose mean RMS exceeds median + k*MAD still gets its envelope "
          "exported. Empty disables."),
]

ALL_LEVERS: list[Lever] = SHARED + SIR + STARTSTOP + SPONTANEOUS
BY_NAME: dict[str, Lever] = {lv.name: lv for lv in ALL_LEVERS}

# Imported by pipeline.py but never read — exposing them as controls would be a lie.
DEAD_CONSTANTS = ("STARTSTOP_ONSET_TMIN", "STARTSTOP_ONSET_TMAX", "STARTSTOP_RESP_TMAX")


def levers_for(mode: str) -> list[Lever]:
    """Levers relevant to 'sir' or 'startstop'."""
    return [lv for lv in ALL_LEVERS if lv.mode in (mode, "both")]


def defaults_for(mode: str) -> dict[str, Any]:
    return {lv.name: lv.default for lv in levers_for(mode)}

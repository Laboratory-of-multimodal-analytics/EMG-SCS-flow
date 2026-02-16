"""Main EMG pipeline logic (ported from the notebook)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import loadmat
from tqdm import tqdm

from .annotations import create_annotation_crops, extract_start_stop_segments
from .constants import (
    ARTCHAN,
    ARTIFACT_REREF,
    CAR_REREF,
    LATERAL_CAR_REREF,
    MAT_DIVIDE_BY_1000,
    BASELINE_TMAX,
    BASELINE_TMIN,
    EPOCH_TMAX,
    EPOCH_TMIN,
    LOWPASS_CUTOFF_HZ,
    MIN_VALID_EPOCHS,
    RAW_BANDPASS_H_FREQ,
    RAW_BANDPASS_L_FREQ,
    RESP_TMAX,
    RESP_TMIN,
    STIM_PROM_PRESTIM_TMAX,
    STIM_PROM_PRESTIM_TMIN,
    STIM_PROM_BASELINE_TMAX,
    STIM_PROM_BASELINE_TMIN,
    STIM_EPOCH_ARTIFACT_ABS_CORR_THR,
    STIM_EPOCH_ARTIFACT_CORR_REJECTION,
    STARTSTOP_MIN_DIST_MS,
    STARTSTOP_MODE,
    STARTSTOP_LEAKAGE_CORR_REJECTION,
    STARTSTOP_TM_NO_POSTHOC_CHECKS,
    STARTSTOP_CHANNEL_MIN_MEDIAN_CORR,
    STARTSTOP_CHANNEL_MIN_VALID_FRAC,
    STARTSTOP_MIN_DETECTIONS_PER_CHANNEL,
    STARTSTOP_RMS_K,
    STARTSTOP_RMS_WIN_MS,
    STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS,
    STARTSTOP_TM_MATCH_PEAK_PROMINENCE,
    STARTSTOP_TM_MAX_MATCHES_PER_CHANNEL,
    STARTSTOP_TM_SCALES,
    STARTSTOP_TM_SCORE_THR,
    STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE,
    STARTSTOP_TM_TEMPLATE_CORR_THR,
    STARTSTOP_TM_TEMPLATE_SFREQ,
    STARTSTOP_TM_TOP_K_CANDIDATES,
    STARTSTOP_EPOCH_TMIN,
    STARTSTOP_EPOCH_TMAX,
    STARTSTOP_BASELINE_TMIN,
    STARTSTOP_BASELINE_TMAX,
    STARTSTOP_ONSET_TMIN,
    STARTSTOP_ONSET_TMAX,
    STARTSTOP_RESP_TMAX,
    STARTSTOP_SIM_TMIN,
    STARTSTOP_SIM_TMAX,
    THRESH,
    get_prominence_k,
)
from .detection import (
    add_extra_peak_to_p1,
    detect_onset_near_template,
    detect_onset_rectified,
    detect_peak_in_window,
    detect_template_peaks,
    find_extra_p1_peak,
    pick_epoch_value_near_latency,
)
from .io_utils import build_output_dirs, list_crop_files
from .plotting import (
    amp_to_number,
    plot_boxplots,
    plot_epochs_panel,
    plot_grouped_by_condition,
    plot_grouped_by_amplitude,
    plot_template_with_markers,
)


GIB_CHARS = set(":>=;@<#")


def _cyrillic_score(text: str) -> int:
    return sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")


def _is_gibberish(text: str) -> bool:
    return any(ch in GIB_CHARS for ch in text)


def _has_internal_plus_minus(tok: str) -> bool:
    return "+" in tok[1:] or "-" in tok[1:]


def _numeric_token_to_keep_ascii(tok: str) -> bool:
    if re.fullmatch(r"\d{2,}", tok):
        return True
    if tok == "1":
        return True
    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?(?:Hz|ms|mA|uA)", tok, re.IGNORECASE):
        return True
    if ("," in tok) or ("." in tok) or ("e" in tok.lower()):
        if re.fullmatch(r"[+-]?[0-9.,eE+-]+", tok):
            return True
    if _has_internal_plus_minus(tok):
        if re.fullmatch(r"[+-]?\d+(?:[+-]\d+)+[+-]?", tok):
            return True
    return False


def _latin_token_to_keep_ascii(tok: str) -> bool:
    if any(ch in GIB_CHARS for ch in tok):
        return False
    if re.search(r"[a-z]", tok):
        return True
    if re.fullmatch(r"[A-Z]{2,}", tok):
        return True
    return False


def _token_to_keep_ascii(tok: str) -> bool:
    return _numeric_token_to_keep_ascii(tok) or _latin_token_to_keep_ascii(tok)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        if _token_to_keep_ascii(tok):
            spans.append(m.span())
    return spans


def _in_spans(i: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= i < b:
            return True
        if i < a:
            return False
    return False


def _decode_7bit_iso8859_5(text: str) -> str:
    protected = _protected_spans(text)
    out = bytearray()
    for i, ch in enumerate(text):
        o = ord(ch)
        if ch in {" ", "\t", "\r", "\n"} or _in_spans(i, protected):
            out.append(o)
        else:
            out.append((o + 0x80) & 0xFF)
    return out.decode("iso8859_5", errors="replace")


def _normalize_text(value) -> str:
    text = value.decode("latin1", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)
    text = text.strip()
    if _cyrillic_score(text) > 0:
        return text
    if not _is_gibberish(text):
        return text
    restored = _decode_7bit_iso8859_5(text).strip()
    if _cyrillic_score(restored) > _cyrillic_score(text):
        return restored
    return text


def _robust_noise_scale(sig: np.ndarray, mask: np.ndarray) -> float:
    """Estimate noise scale robustly for a masked interval."""
    if np.sum(mask) < 5:
        return np.inf
    x = np.asarray(sig[mask], dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.inf
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 0:
        return np.inf
    return scale


def _choose_silent_prominence_baseline_mask(
    sig: np.ndarray,
    prestim_mask: np.ndarray,
    poststim_mask: np.ndarray,
) -> np.ndarray:
    """Choose the quieter baseline window for prominence estimation."""
    pre_scale = _robust_noise_scale(sig, prestim_mask)
    post_scale = _robust_noise_scale(sig, poststim_mask)
    if np.isinf(pre_scale) and np.isinf(post_scale):
        return prestim_mask
    if post_scale < pre_scale:
        return poststim_mask
    return prestim_mask


def _load_raw_from_mat(mat_path: Path) -> mne.io.Raw:
    mat = loadmat(mat_path)
    data = mat["data"].ravel()
    datastart = mat["datastart"].astype(int)
    dataend = mat["dataend"].astype(int)
    titles = [t.strip() if hasattr(t, "strip") else str(t) for t in mat["titles"].ravel()]
    samplerate = mat["samplerate"]

    rates = np.unique(samplerate[np.isfinite(samplerate) & (samplerate > 0)])
    if rates.size == 0:
        raise ValueError("No valid samplerate values found in MAT file.")
    sfreq = float(np.max(rates))

    block_indices = [i for i in range(samplerate.shape[1]) if np.isclose(samplerate[0, i], sfreq)]

    channel_data = []
    for ch_idx in range(len(titles)):
        parts = []
        for block_idx in block_indices:
            start = datastart[ch_idx, block_idx] - 1
            end = dataend[ch_idx, block_idx] - 1
            if end < start:
                continue
            parts.append(data[start : end + 1])
        channel_data.append(np.concatenate(parts, axis=0))

    data_2d = np.vstack(channel_data)
    if MAT_DIVIDE_BY_1000:
        data_2d = data_2d / 1000.0
    ch_types = ["eeg" for _ in titles]
    info = mne.create_info(ch_names=titles, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    com = mat["com"]
    comtext = mat["comtext"].ravel()
    event_blocks = com[:, 1].astype(int) - 1
    event_samples = com[:, 2].astype(int) - 1
    event_ids = com[:, 4].astype(int)

    block_pos = {block_idx: pos for pos, block_idx in enumerate(block_indices)}
    block_lengths = []
    for block_idx in block_indices:
        start = datastart[0, block_idx] - 1
        end = dataend[0, block_idx] - 1
        block_lengths.append(end - start + 1)
    block_offsets = np.cumsum([0] + block_lengths)

    valid_blocks = np.isfinite(event_blocks) & np.isfinite(event_samples)
    event_blocks = event_blocks[valid_blocks]
    event_samples = event_samples[valid_blocks]
    event_ids = event_ids[valid_blocks]

    onsets = []
    descriptions = []
    for block_idx, sample_idx, event_id in zip(event_blocks, event_samples, event_ids):
        if block_idx not in block_pos:
            continue
        pos = block_pos[block_idx]
        onset = (block_offsets[pos] + sample_idx) / sfreq
        onsets.append(onset)
        descriptions.append(_normalize_text(comtext[event_id - 1]))

    annotations = mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descriptions)
    raw.set_annotations(annotations)
    return raw


def _resolve_art_channels(
    raw: mne.io.BaseRaw,
    artchan_setting,
    fallback_chans: list[str] | None = None,
) -> list[str]:
    art_chans: list[str] = []
    if artchan_setting:
        if isinstance(artchan_setting, (list, tuple, set)):
            art_chans = [ch for ch in artchan_setting if ch in raw.ch_names]
        elif isinstance(artchan_setting, str) and artchan_setting in raw.ch_names:
            art_chans = [artchan_setting]
    if not art_chans:
        art_chans = [ch for ch in raw.ch_names if "art" in ch.lower()]
    if not art_chans and fallback_chans:
        art_chans = [ch for ch in fallback_chans if ch in raw.ch_names]
    if not art_chans:
        raise ValueError("No artifact channel found. Set ARTCHAN or name channel with 'art'.")
    return art_chans


def _get_art_signal(raw: mne.io.BaseRaw, art_chans: list[str]) -> np.ndarray:
    data = raw.get_data(picks=art_chans)
    if data.ndim == 2 and data.shape[0] > 1:
        return data.mean(axis=0)
    return data[0]


def _apply_artifact_reref(raw: mne.io.BaseRaw, art_chans: list[str]) -> None:
    """Subtract artifact-channel signal from all non-artifact channels (done first when used with CAR).
    Artifact channels themselves are left unchanged so they can be used in original form for stimulus-onset detection."""
    if not raw.preload:
        raw.load_data()
    art_signal = _get_art_signal(raw, art_chans)
    art_set = set(art_chans)
    for ch_idx, ch_name in enumerate(raw.ch_names):
        if ch_name in art_set:
            continue
        raw._data[ch_idx, :] = raw._data[ch_idx, :] - art_signal


def _apply_car_reref(raw: mne.io.BaseRaw, art_chans: list[str]) -> None:
    """Subtract common average from each channel; mean is computed over non-artifact channels only.
    Artifact channels are not modified (kept in original form for stimulus-onset detection)."""
    if not raw.preload:
        raw.load_data()
    art_set = set(art_chans)
    ch_indices = [i for i, ch in enumerate(raw.ch_names) if ch not in art_set]
    if not ch_indices:
        return
    if not LATERAL_CAR_REREF:
        mean_sig = raw._data[ch_indices, :].mean(axis=0)
        for ch_idx in ch_indices:
            raw._data[ch_idx, :] = raw._data[ch_idx, :] - mean_sig
        return

    def _channel_side(name: str) -> str | None:
        n = name.lower()
        if "_l" in n or " left" in n or n.endswith(" l"):
            return "L"
        if "_r" in n or " right" in n or n.endswith(" r"):
            return "R"
        return None

    left_idx: list[int] = []
    right_idx: list[int] = []
    other_idx: list[int] = []
    for i, ch in enumerate(raw.ch_names):
        if ch in art_set:
            continue
        side = _channel_side(ch)
        if side == "L":
            left_idx.append(i)
        elif side == "R":
            right_idx.append(i)
        else:
            other_idx.append(i)

    if left_idx:
        mean_left = raw._data[left_idx, :].mean(axis=0)
        for ch_idx in left_idx:
            raw._data[ch_idx, :] = raw._data[ch_idx, :] - mean_left
    if right_idx:
        mean_right = raw._data[right_idx, :].mean(axis=0)
        for ch_idx in right_idx:
            raw._data[ch_idx, :] = raw._data[ch_idx, :] - mean_right
    if other_idx:
        mean_other = raw._data[other_idx, :].mean(axis=0)
        for ch_idx in other_idx:
            raw._data[ch_idx, :] = raw._data[ch_idx, :] - mean_other


def _apply_special_crops(raw: mne.io.BaseRaw, file_name: str) -> None:
    if file_name in ["1+2-_9.fif"]:
        raw.crop(0, raw.times.max() - 10)
    if file_name in ["0+1-_2.fif"]:
        raw.crop(10, raw.times.max())
    if file_name in ["3+4-_4.fif", "8+11-_10.fif"]:
        raw.crop(0, 5)
    if file_name in ["5+6-_4.fif"]:
        raw.crop(0, 20)
    if file_name in ["12+13-_1.fif"]:
        raw.crop(5, raw.times.max())
    if file_name in ["14+15-_10.fif"]:
        raw.crop(2, 10)


def _concat_segments(raw: mne.io.BaseRaw, segments: list[tuple[float, float]]) -> mne.io.BaseRaw | None:
    if not segments:
        return None
    raws = []
    for tmin, tmax in segments:
        if tmax <= tmin:
            continue
        raws.append(raw.copy().crop(tmin=tmin, tmax=tmax))
    if not raws:
        return None
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws, preload=True)


def _resolve_startstop_template_dir() -> Path:
    """
    Resolve STARTSTOP template directory across repository layouts.

    Preferred location for this project:
      <repo>/src/EMG-SCS-flow/templates
    """
    here = Path(__file__).resolve()
    candidates = [
        # templates as sibling of src/
        here.parent.parent / "templates",
        # templates next to pipeline.py (legacy/fallback)
        here.parent / "templates",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    return candidates[0]


def _load_startstop_template_bank(
    template_dir: Path,
    template_native_sfreq: float,
    template_center_sample: int,
) -> list[dict[str, object]]:
    bank: list[dict[str, object]] = []
    for npy_path in sorted(template_dir.glob("template_*.npy")):
        m = re.fullmatch(r"template_(\d+)\.npy", npy_path.name)
        if not m:
            continue
        template_id = m.group(1)
        npz_path = template_dir / f"template_{template_id}_onset_and_peaks.npz"
        if not npz_path.exists():
            continue

        wave = np.asarray(np.load(npy_path), dtype=float).ravel()
        if wave.size < 3:
            continue
        sample_idx = np.arange(wave.size, dtype=float)
        times = (sample_idx - float(template_center_sample)) / float(template_native_sfreq)
        marker_idx = np.load(npz_path)
        markers: dict[str, float] = {}
        for key in ("onset", "peak1", "peak2"):
            if key not in marker_idx:
                markers[key] = np.nan
                continue
            idx_raw = float(np.asarray(marker_idx[key]).reshape(()))
            idx = int(np.clip(round(idx_raw), 0, wave.size - 1))
            markers[key] = float(times[idx])

        bank.append(
            {
                "name": f"template_{template_id}",
                "wave": wave,
                "times": times,
                "markers": markers,
            }
        )
    return bank


def _synthesize_template_on_times(
    template_wave: np.ndarray,
    template_times: np.ndarray,
    target_times: np.ndarray,
    scale: float,
    flip: int,
) -> np.ndarray:
    sampled = np.interp(target_times / scale, template_times, template_wave, left=0.0, right=0.0)
    return float(flip) * sampled


def _build_template_variants_for_matching(
    template_bank: list[dict[str, object]],
    sfreq: float,
    scales: tuple[float, ...],
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for tpl in template_bank:
        template_wave = np.asarray(tpl["wave"], dtype=float)
        template_times = np.asarray(tpl["times"], dtype=float)
        duration = float(template_times[-1] - template_times[0])
        markers = tpl["markers"]
        for scale in scales:
            scaled_duration = duration * float(scale)
            n_samples = int(round(scaled_duration * sfreq)) + 1
            if n_samples < 5:
                continue
            t_var = np.linspace(template_times[0] * scale, template_times[-1] * scale, n_samples)
            base_wave = np.interp(t_var / scale, template_times, template_wave, left=0.0, right=0.0)
            for flip in (1, -1):
                wave = float(flip) * base_wave
                wave = wave - np.mean(wave)
                wave_norm = float(np.linalg.norm(wave))
                if wave_norm <= 0:
                    continue
                variants.append(
                    {
                        "template_name": tpl["name"],
                        "scale": float(scale),
                        "flip": int(flip),
                        "wave": wave,
                        "wave_norm": wave_norm,
                        "center_idx": int(np.argmin(np.abs(t_var))),
                        "marker_times": {
                            "onset": float(markers["onset"] * scale) if np.isfinite(markers["onset"]) else np.nan,
                            "p1": float(markers["peak1"] * scale) if np.isfinite(markers["peak1"]) else np.nan,
                            "p2": float(markers["peak2"] * scale) if np.isfinite(markers["peak2"]) else np.nan,
                        },
                    }
                )
    return variants


def _normalized_match_score(sig: np.ndarray, wave: np.ndarray, wave_norm: float) -> np.ndarray:
    n_sig = sig.size
    n_wave = wave.size
    if n_sig < n_wave:
        return np.array([], dtype=float)
    dots = np.correlate(sig, wave, mode="valid")
    sq = np.square(sig)
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    win_energy = csum[n_wave:] - csum[:-n_wave]
    denom = np.sqrt(np.maximum(win_energy, 1e-30)) * max(wave_norm, 1e-30)
    score = np.zeros_like(dots, dtype=float)
    valid = denom > 0
    score[valid] = dots[valid] / denom[valid]
    score[~np.isfinite(score)] = 0.0
    return score


def _detect_template_anchor_samples(
    data: np.ndarray,
    variants: list[dict[str, object]],
    refractory_dist_samples: int,
    match_peak_min_dist_samples: int,
    score_thr: float,
    template_corr_thr: float,
    max_matches_per_channel: int,
    top_k_candidates: int,
    match_peak_prominence: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    if not variants:
        return np.array([], dtype=int), [], []

    selected_all_channels: list[dict[str, object]] = []
    discarded: list[dict[str, object]] = []
    for ch_idx in range(data.shape[0]):
        sig = data[ch_idx]
        sig_centered = sig - np.mean(sig)
        ch_candidates_all: list[dict[str, object]] = []

        for var in variants:
            score = _normalized_match_score(sig_centered, var["wave"], float(var["wave_norm"]))
            if score.size == 0:
                discarded.append(
                    {
                        "channel_index": int(ch_idx),
                        "template_name": str(var["template_name"]),
                        "scale": float(var["scale"]),
                        "flip": int(var["flip"]),
                        "reason": "signal_shorter_than_template",
                        "score": np.nan,
                        "anchor_sample": np.nan,
                    }
                )
                continue

            find_peaks_kw = dict(
                height=score_thr,
                distance=match_peak_min_dist_samples,
            )
            if match_peak_prominence > 0:
                find_peaks_kw["prominence"] = match_peak_prominence
            peaks, props = find_peaks(score, **find_peaks_kw)
            if len(peaks) == 0:
                continue
            heights = props.get("peak_heights", score[peaks])
            for peak_idx, height in zip(peaks, heights):
                anchor = int(peak_idx + int(var["center_idx"]))
                if 0 <= anchor < sig.size:
                    # Additional local-shape check: Pearson correlation between
                    # the candidate window and the matched template variant.
                    win_len = int(var["wave"].size)
                    start = int(peak_idx)
                    stop = start + win_len
                    if stop > sig.size:
                        discarded.append(
                            {
                                "channel_index": int(ch_idx),
                                "template_name": str(var["template_name"]),
                                "scale": float(var["scale"]),
                                "flip": int(var["flip"]),
                                "reason": "window_out_of_bounds_for_corr",
                                "score": float(height),
                                "anchor_sample": anchor,
                            }
                        )
                        continue
                    sig_win = np.asarray(sig[start:stop], dtype=float)
                    sig_win = sig_win - np.mean(sig_win)
                    var_wave = np.asarray(var["wave"], dtype=float)
                    denom = float(np.linalg.norm(sig_win) * np.linalg.norm(var_wave))
                    if denom <= 0:
                        local_corr = 0.0
                    else:
                        local_corr = float(np.dot(sig_win, var_wave) / denom)
                    if local_corr < float(template_corr_thr):
                        discarded.append(
                            {
                                "channel_index": int(ch_idx),
                                "template_name": str(var["template_name"]),
                                "scale": float(var["scale"]),
                                "flip": int(var["flip"]),
                                "reason": "low_template_corr",
                                "score": float(height),
                                "anchor_sample": anchor,
                                "template_corr": local_corr,
                            }
                        )
                        continue
                    ch_candidates_all.append(
                        {
                            "anchor_sample": anchor,
                            "score": float(height),
                            "template_corr": local_corr,
                            "channel_index": int(ch_idx),
                            "template_name": str(var["template_name"]),
                            "scale": float(var["scale"]),
                            "flip": int(var["flip"]),
                            "marker_times": dict(var["marker_times"]),
                        }
                    )

        if not ch_candidates_all:
            discarded.append(
                {
                    "channel_index": int(ch_idx),
                    "template_name": "",
                    "scale": np.nan,
                    "flip": 1,
                    "reason": "no_peak_above_score_threshold",
                    "score": np.nan,
                    "anchor_sample": np.nan,
                }
            )
            continue

        # Group nearby anchors into quasi-detection windows, then keep only the
        # strongest template/scale/polarity match inside each window.
        ch_candidates_all.sort(key=lambda x: int(x["anchor_sample"]))
        quasi_groups: list[list[dict[str, object]]] = []
        current_group: list[dict[str, object]] = [ch_candidates_all[0]]
        for cand in ch_candidates_all[1:]:
            if int(cand["anchor_sample"]) - int(current_group[-1]["anchor_sample"]) <= match_peak_min_dist_samples:
                current_group.append(cand)
            else:
                quasi_groups.append(current_group)
                current_group = [cand]
        quasi_groups.append(current_group)

        ch_candidates: list[dict[str, object]] = []
        for group in quasi_groups:
            best = max(group, key=lambda x: float(x["score"]))
            ch_candidates.append(best)
            for cand in group:
                if cand is not best:
                    discarded.append({**cand, "reason": "lower_score_in_quasi_window"})

        ch_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
        if max_matches_per_channel > 0 and len(ch_candidates) > max_matches_per_channel:
            for cand in ch_candidates[max_matches_per_channel:]:
                discarded.append({**cand, "reason": "per_channel_top_n_limit"})
        if max_matches_per_channel > 0:
            ch_candidates = ch_candidates[:max_matches_per_channel]

        # Keep channel detections independent: refractory is applied per channel.
        ch_selected: list[dict[str, object]] = []
        for cand in ch_candidates:
            anchor = int(cand["anchor_sample"])
            if all(abs(anchor - int(kept["anchor_sample"])) >= refractory_dist_samples for kept in ch_selected):
                ch_selected.append(cand)
            else:
                discarded.append({**cand, "reason": "refractory_within_channel"})
        selected_all_channels.extend(ch_selected)

    if not selected_all_channels:
        return np.array([], dtype=int), [], discarded

    candidates = sorted(selected_all_channels, key=lambda x: float(x["score"]), reverse=True)
    if top_k_candidates > 0 and len(candidates) > top_k_candidates:
        for cand in candidates[top_k_candidates:]:
            discarded.append({**cand, "reason": "global_top_k_limit"})
    if top_k_candidates > 0:
        candidates = candidates[:top_k_candidates]
    candidates.sort(key=lambda x: int(x["anchor_sample"]))
    selected_anchors = np.asarray([int(s["anchor_sample"]) for s in candidates], dtype=int)
    return selected_anchors, candidates, discarded


def _select_best_template_for_channel(
    channel_template: np.ndarray,
    epoch_times: np.ndarray,
    template_bank: list[dict[str, object]],
    scales: tuple[float, ...],
) -> dict[str, object] | None:
    ch = np.asarray(channel_template, dtype=float)
    ch = ch - np.mean(ch)
    ch_std = float(np.std(ch))
    if ch_std <= 0:
        return None

    best: dict[str, object] | None = None
    best_corr = -np.inf
    for tpl in template_bank:
        template_wave = np.asarray(tpl["wave"], dtype=float)
        template_times = np.asarray(tpl["times"], dtype=float)
        for scale in scales:
            for flip in (1, -1):
                synth = _synthesize_template_on_times(
                    template_wave=template_wave,
                    template_times=template_times,
                    target_times=epoch_times,
                    scale=float(scale),
                    flip=int(flip),
                )
                synth = synth - np.mean(synth)
                synth_std = float(np.std(synth))
                if synth_std <= 0:
                    continue
                corr = float(np.corrcoef(ch, synth)[0, 1])
                if np.isnan(corr):
                    continue
                if corr > best_corr:
                    markers = tpl["markers"]
                    best_corr = corr
                    best = {
                        "template_name": tpl["name"],
                        "scale": float(scale),
                        "flip": int(flip),
                        "corr": corr,
                        "template": _synthesize_template_on_times(
                            template_wave=template_wave,
                            template_times=template_times,
                            target_times=epoch_times,
                            scale=float(scale),
                            flip=int(flip),
                        ),
                        "markers": {
                            "onset": float(markers["onset"] * scale) if np.isfinite(markers["onset"]) else np.nan,
                            "p1": float(markers["peak1"] * scale) if np.isfinite(markers["peak1"]) else np.nan,
                            "p2": float(markers["peak2"] * scale) if np.isfinite(markers["peak2"]) else np.nan,
                        },
                    }
    return best


def _run_startstop_analysis(
    raw: mne.io.BaseRaw,
    startstop_dir: Path,
    art_chans: list[str],
) -> None:
    print("[STARTSTOP] Extracting start/stop segments...", flush=True)
    segments = extract_start_stop_segments(raw)
    if not segments:
        return

    art_set = set(art_chans)
    ch_names = [ch for ch in raw.ch_names if ch not in art_set]
    if not ch_names:
        return

    # Template bank lives as a sibling of src/ in this project.
    template_dir = _resolve_startstop_template_dir()
    if not template_dir.exists():
        print(
            f"[STARTSTOP] Template directory not found: {template_dir}",
            flush=True,
        )
    template_bank = _load_startstop_template_bank(
        template_dir=template_dir,
        template_native_sfreq=float(STARTSTOP_TM_TEMPLATE_SFREQ),
        template_center_sample=int(STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE),
    )
    if not template_bank:
        print("[STARTSTOP] No templates found; skipping STARTSTOP analysis.", flush=True)
        return

    paths = {
        "excel_dir": startstop_dir / "Excel",
        "boxplot_dir": startstop_dir / "Boxplots",
        "plots_grouped_dir": startstop_dir / "Plots grouped by condition",
        "plots_grid_dir": startstop_dir / "Plots with grid and markers",
        "plots_plain_dir": startstop_dir / "Plots without grid and markers",
        "raw_epochs_dir": startstop_dir / "Raw epochs",
        "templates_dir": startstop_dir / "Templates",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    group_store_by_condition = defaultdict(lambda: defaultdict(list))
    grouped_times = None
    all_results_rows: list[dict[str, object]] = []
    template_match_rows: list[dict[str, object]] = []
    template_discard_rows: list[dict[str, object]] = []
    channel_template_rows: list[dict[str, object]] = []

    print("[STARTSTOP] Matching templates and detecting responses...", flush=True)
    for condition, side_segments in tqdm(
        segments.items(),
        desc="STARTSTOP: detect by condition",
        total=len(segments),
    ):
        safe_condition = re.sub(r"[^\w\-\+\. ]", "_", str(condition))
        start_segments = side_segments.get("start", [])
        if not start_segments:
            continue

        start_raw = _concat_segments(raw, start_segments)
        if start_raw is None:
            continue

        excel_dir = paths["excel_dir"]
        boxplot_dir = paths["boxplot_dir"]
        plots_grid_dir = paths["plots_grid_dir"]
        plots_plain_dir = paths["plots_plain_dir"]

        sfreq = float(start_raw.info["sfreq"])
        data = start_raw.get_data(picks=ch_names)
        refractory_dist = max(1, int((STARTSTOP_MIN_DIST_MS / 1000.0) * sfreq))
        match_peak_dist = max(1, int((STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS / 1000.0) * sfreq))
        variants = _build_template_variants_for_matching(
            template_bank=template_bank,
            sfreq=sfreq,
            scales=tuple(float(s) for s in STARTSTOP_TM_SCALES),
        )
        peaks, peak_matches, peak_discarded = _detect_template_anchor_samples(
            data=data,
            variants=variants,
            refractory_dist_samples=refractory_dist,
            match_peak_min_dist_samples=match_peak_dist,
            score_thr=float(STARTSTOP_TM_SCORE_THR),
            template_corr_thr=float(STARTSTOP_TM_TEMPLATE_CORR_THR),
            max_matches_per_channel=int(STARTSTOP_TM_MAX_MATCHES_PER_CHANNEL),
            top_k_candidates=int(STARTSTOP_TM_TOP_K_CANDIDATES),
            match_peak_prominence=float(STARTSTOP_TM_MATCH_PEAK_PROMINENCE),
        )
        # Per condition and per channel: if a channel has fewer than N detections in this condition, discard that channel's detections (others preserved).
        min_det = int(STARTSTOP_MIN_DETECTIONS_PER_CHANNEL)
        if min_det > 0 and peak_matches:
            ch_counts: dict[int, int] = defaultdict(int)
            for m in peak_matches:
                ch_counts[int(m["channel_index"])] += 1
            peak_matches = [m for m in peak_matches if ch_counts[int(m["channel_index"])] >= min_det]
        for m in peak_matches:
            ch_idx = int(m["channel_index"])
            ch_name = ch_names[ch_idx] if 0 <= ch_idx < len(ch_names) else f"ch_{ch_idx}"
            anchor_sample = int(m["anchor_sample"])
            template_match_rows.append(
                {
                    "Condition": str(condition),
                    "Channel": str(ch_name),
                    "Anchor sample": anchor_sample,
                    "Anchor time (s)": float(anchor_sample / sfreq),
                    "Template": str(m["template_name"]),
                    "Scale": float(m["scale"]),
                    "Flip": int(m["flip"]),
                    "Match score": float(m["score"]),
                    "Template corr": float(m.get("template_corr", np.nan)),
                    "Marker onset (s)": float(dict(m.get("marker_times", {})).get("onset", np.nan)),
                    "Marker p1 (s)": float(dict(m.get("marker_times", {})).get("p1", np.nan)),
                    "Marker p2 (s)": float(dict(m.get("marker_times", {})).get("p2", np.nan)),
                }
            )
        for d in peak_discarded:
            ch_idx = int(d["channel_index"])
            ch_name = ch_names[ch_idx] if 0 <= ch_idx < len(ch_names) else f"ch_{ch_idx}"
            anchor_raw = d.get("anchor_sample", np.nan)
            if np.isfinite(anchor_raw):
                anchor_sample = int(anchor_raw)
                anchor_time = float(anchor_sample / sfreq)
            else:
                anchor_sample = np.nan
                anchor_time = np.nan
            template_discard_rows.append(
                {
                    "Condition": str(condition),
                    "Channel": str(ch_name),
                    "Template": str(d.get("template_name", "")),
                    "Scale": float(d.get("scale", np.nan)),
                    "Flip": int(d.get("flip", 1)),
                    "Score": float(d.get("score", np.nan)),
                    "Template corr": float(d.get("template_corr", np.nan)),
                    "Anchor sample": anchor_sample,
                    "Anchor time (s)": anchor_time,
                    "Reason": str(d.get("reason", "unknown")),
                }
            )
        if not peak_matches:
            continue

        # One event per unique anchor time (keep best score per sample to avoid duplicate event times)
        by_sample: dict[int, list[dict[str, object]]] = defaultdict(list)
        for m in peak_matches:
            by_sample[int(m["anchor_sample"])].append(m)
        peak_matches_merged = [max(g, key=lambda x: float(x["score"])) for g in by_sample.values()]
        peak_matches_sorted = sorted(peak_matches_merged, key=lambda m: int(m["anchor_sample"]))
        event_samples = np.asarray([int(m["anchor_sample"]) for m in peak_matches_sorted], dtype=int)
        events = np.column_stack(
            [
                event_samples + start_raw.first_samp,
                np.zeros_like(event_samples, dtype=int),
                np.ones_like(event_samples, dtype=int),
            ]
        )
        epochs = mne.Epochs(
            start_raw,
            events,
            event_id=1,
            tmin=STARTSTOP_EPOCH_TMIN,
            tmax=STARTSTOP_EPOCH_TMAX,
            baseline=None,
            preload=True,
        )

        data_epoched = epochs.get_data()
        times = epochs.times
        grouped_times = times

        baseline_mask = (times >= STARTSTOP_BASELINE_TMIN) & (times <= STARTSTOP_BASELINE_TMAX)

        data_filt = data_epoched

        tmpl_cfg = {}
        markers_cfg = {}
        template_lookup = {str(tpl["name"]): tpl for tpl in template_bank}
        best_match_per_channel: dict[int, dict[str, object]] = {}
        for m in peak_matches_sorted:
            ch_idx = int(m["channel_index"])
            prev = best_match_per_channel.get(ch_idx)
            if prev is None or float(m["score"]) > float(prev["score"]):
                best_match_per_channel[ch_idx] = m

        for ch_idx, ch_name in enumerate(epochs.ch_names):
            if ch_name in art_set:
                continue

            best_match = best_match_per_channel.get(ch_idx)
            if best_match is None:
                # No template match for this channel in this condition: dismiss channel (no data-driven fallback)
                continue

            tpl_name = str(best_match["template_name"])
            tpl = template_lookup.get(tpl_name)
            if tpl is not None:
                tmpl = _synthesize_template_on_times(
                    template_wave=np.asarray(tpl["wave"], dtype=float),
                    template_times=np.asarray(tpl["times"], dtype=float),
                    target_times=times,
                    scale=float(best_match["scale"]),
                    flip=int(best_match["flip"]),
                )
            else:
                # Match referred to missing template in bank; use channel mean for shape only
                tmpl = np.mean(data_filt[:, ch_idx, :], axis=0)

            marker_times = dict(best_match.get("marker_times", {}))
            onset_t = float(marker_times.get("onset", np.nan))
            p1_t = float(marker_times.get("p1", np.nan))
            p2_t = float(marker_times.get("p2", np.nan))
            chosen_template = tpl_name
            chosen_scale = float(best_match["scale"])
            chosen_flip = int(best_match["flip"])

            tmpl_cfg[ch_name] = tmpl
            markers_cfg[ch_name] = dict(onset=onset_t, p1=p1_t, p2=p2_t)
            channel_template_rows.append(
                {
                    "Condition": str(condition),
                    "Channel": str(ch_name),
                    "Chosen template": chosen_template,
                    "Scale": chosen_scale,
                    "Flip": chosen_flip,
                    "Template corr": np.nan,
                    "Marker onset (s)": onset_t,
                    "Marker p1 (s)": p1_t,
                    "Marker p2 (s)": p2_t,
                }
            )

        templates_dir = paths["templates_dir"] / safe_condition
        templates_dir.mkdir(parents=True, exist_ok=True)
        for ch_name, tmpl in tmpl_cfg.items():
            markers = markers_cfg.get(ch_name, {"onset": np.nan, "p1": np.nan, "p2": np.nan})
            safe_ch = re.sub(r"[^\w\-_\. ]", "_", str(ch_name))
            out_path = templates_dir / f"{safe_ch}_template.png"
            plot_template_with_markers(
                times=times,
                template=tmpl,
                markers=markers,
                out_path=out_path,
                title=f"StartStop template: {condition} | {ch_name}",
            )

        results = {key: [] for key in [
            "Configuration",
            "Stim. amplitude",
            "Epoch",
            "Channel",
            "Onset latency",
            "Peak1 latency",
            "Peak2 latency",
            "Peak1 value",
            "Peak2 value",
            "PTP amplitude",
            "Time series",
        ]}

        channel_epoch_results = {ch: [] for ch in epochs.ch_names if ch not in art_set}
        latency_markers = {ch: [] for ch in epochs.ch_names if ch not in art_set}

        for ep in range(len(epochs)):
            origin_ch_idx = int(peak_matches_sorted[ep]["channel_index"])
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                if ch_name in art_set:
                    continue
                if ch_idx != origin_ch_idx:
                    continue

                sig = data_epoched[ep, ch_idx, :]
                sig_f = data_filt[ep, ch_idx, :]

                base = sig_f[baseline_mask]

                t_on_tmpl = markers_cfg[ch_name]["onset"]
                t_p1 = markers_cfg[ch_name]["p1"]
                t_p2 = markers_cfg[ch_name]["p2"]
                tmpl = tmpl_cfg[ch_name]

                if np.isnan(t_on_tmpl):
                    onset_latency = np.nan
                else:
                    onset_latency = detect_onset_near_template(
                        sig_f,
                        times,
                        sfreq,
                        baseline_mask,
                        t_on_tmpl,
                        win_ms=2,
                        k=1,
                        sustain_ms=2,
                    )

                peak1_latency = peak2_latency = np.nan
                peak1_value = peak2_value = np.nan

                if not np.isnan(t_p1):
                    idx_tmpl_p1 = int(np.argmin(np.abs(times - t_p1)))
                    pol1 = np.sign(tmpl[idx_tmpl_p1] - np.mean(tmpl[baseline_mask]))
                    pol1 = +1 if pol1 >= 0 else -1

                    tmpl_p1_val = tmpl[idx_tmpl_p1]
                    peak1_latency, peak1_value = detect_peak_in_window(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        baseline_mask=baseline_mask,
                        t_center=t_p1,
                        win_ms=8.0,
                        polarity=pol1,
                        amp_min_uV=5.0,
                        min_width_ms=0.0,
                        choose="nearest",
                        template_peak_val=tmpl_p1_val,
                        min_rel_to_template=0.1,
                    )
                else:
                    pol1 = +1

                if not np.isnan(t_p2):
                    idx_tmpl_p2 = int(np.argmin(np.abs(times - t_p2)))
                    pol2 = np.sign(tmpl[idx_tmpl_p2] - np.mean(tmpl[baseline_mask]))
                    pol2 = +1 if pol2 >= 0 else -1

                    tmpl_p2_val = tmpl[idx_tmpl_p2]
                    peak2_latency, peak2_value = detect_peak_in_window(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        baseline_mask=baseline_mask,
                        t_center=t_p2,
                        win_ms=8.0,
                        polarity=pol2,
                        amp_min_uV=5.0,
                        min_width_ms=0.0,
                        choose="nearest",
                        template_peak_val=tmpl_p2_val,
                        min_rel_to_template=0.1,
                    )
                else:
                    pol2 = +1

                if not np.isnan(peak1_latency):
                    peak1_latency, peak1_value = pick_epoch_value_near_latency(
                        sig,
                        times,
                        peak1_latency,
                        sfreq,
                        win_ms=1.0,
                        polarity=pol1,
                    )

                if not np.isnan(peak2_latency):
                    peak2_latency, peak2_value = pick_epoch_value_near_latency(
                        sig,
                        times,
                        peak2_latency,
                        sfreq,
                        win_ms=1.0,
                        polarity=pol2,
                    )

                if STARTSTOP_TM_NO_POSTHOC_CHECKS:
                    if np.isnan(peak1_latency) and not np.isnan(t_p1):
                        #find individual peak1 latency and value near template onset
                        peak1_latency, peak1_value = detect_peak_in_window(
                            sig_f=sig_f,
                            times=times,
                            sfreq=sfreq,
                            baseline_mask=baseline_mask,
                            t_center=t_p1,
                            win_ms=8.0,
                            polarity=pol1,
                            amp_min_uV=0,
                            min_width_ms=0.0,
                            choose="nearest",
                            template_peak_val=tmpl_p1_val,
                            min_rel_to_template=0,
                        )
                        # peak1_latency = float(t_p1)
                        # peak1_value = pick_epoch_value_near_latency(
                        #     sig, times, peak1_latency, sfreq, win_ms=1.0, polarity=pol1
                        # )[1]
                    if np.isnan(peak2_latency) and not np.isnan(t_p2):

                        #find individual peak2 latency and value near template onset
                        peak2_latency, peak2_value = detect_peak_in_window(
                            sig_f=sig_f,
                            times=times,
                            sfreq=sfreq,
                            baseline_mask=baseline_mask,
                            t_center=t_p2,
                            win_ms=8.0,
                            polarity=pol2,
                            amp_min_uV=0,
                            min_width_ms=0.0,
                            choose="nearest",
                            template_peak_val=tmpl_p2_val,
                            min_rel_to_template=0,
                        )
                        # peak2_latency = float(t_p2)
                        # peak2_value = pick_epoch_value_near_latency(
                        #     sig, times, peak2_latency, sfreq, win_ms=1.0, polarity=pol2
                        # )[1]

                if (not np.isnan(peak1_latency)) and (not np.isnan(peak2_latency)):
                    if peak2_latency <= peak1_latency:
                        peak2_latency = np.nan
                        peak2_value = np.nan

                if (not np.isnan(peak1_value)) and (not np.isnan(peak2_value)):
                    if np.sign(peak1_value) == np.sign(peak2_value):
                        peak2_latency = np.nan
                        peak2_value = np.nan

                p1_corr_value = peak1_value
                if (
                    (not np.isnan(peak1_latency))
                    and (not np.isnan(peak2_latency))
                    and (not np.isnan(peak1_value))
                    and (not np.isnan(peak2_value))
                ):
                    extra_lat, extra_val = find_extra_p1_peak(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        p1_lat=peak1_latency,
                        p2_lat=peak2_latency,
                        p1_polarity=pol1,
                        p2_hint_lat=np.nan,
                        guard_ms=1.0,
                        hint_ms=4.0,
                        min_width_ms=0.4,
                        amp_min_uV=10.0,
                        choose="dominant",
                    )
                    if not np.isnan(extra_val):
                        p1_corr_value = add_extra_peak_to_p1(peak1_value, pol1, extra_val)

                if (not np.isnan(p1_corr_value)) and (not np.isnan(peak2_value)):
                    ptp_amp = float(np.abs(p1_corr_value - peak2_value))
                else:
                    ptp_amp = np.nan

                if not STARTSTOP_TM_NO_POSTHOC_CHECKS:
                    if (not np.isnan(ptp_amp)) and (ptp_amp < 15e-6):
                        onset_latency = np.nan
                        peak1_latency = np.nan
                        peak2_latency = np.nan
                        peak1_value = np.nan
                        peak2_value = np.nan
                        ptp_amp = np.nan

                    if (not np.isnan(peak1_value)) and (np.abs(peak1_value) <= 5e-6):
                        onset_latency = np.nan
                        peak1_latency = np.nan
                        peak2_latency = np.nan
                        peak1_value = np.nan
                        peak2_value = np.nan
                        ptp_amp = np.nan

                    if np.isnan(peak1_value):
                        onset_latency = np.nan
                        peak1_latency = np.nan
                        peak2_latency = np.nan
                        peak1_value = np.nan
                        peak2_value = np.nan
                        ptp_amp = np.nan

                if STARTSTOP_LEAKAGE_CORR_REJECTION and (not np.isnan(peak1_latency)):
                    leak_win_s = 0.04
                    leak_corr_thr = 0.7
                    leak_min_channels = 3
                    win_mask = np.abs(times - peak1_latency) <= leak_win_s
                    if np.any(win_mask):
                        ref_seg = sig_f[win_mask]
                        ref_seg = ref_seg - np.mean(ref_seg)
                        ref_std = np.std(ref_seg)
                        if ref_std > 0:
                            leak_count = 0
                            for other_idx, other_name in enumerate(epochs.ch_names):
                                if other_name == ch_name or other_name in art_set:
                                    continue
                                other_seg = data_filt[ep, other_idx, win_mask]
                                other_seg = other_seg - np.mean(other_seg)
                                other_std = np.std(other_seg)
                                if other_std == 0:
                                    continue
                                corr = float(np.corrcoef(ref_seg, other_seg)[0, 1])
                                if np.isnan(corr):
                                    continue
                                if corr > leak_corr_thr:
                                    leak_count += 1
                            if leak_count > leak_min_channels:
                                onset_latency = np.nan
                                peak1_latency = np.nan
                                peak2_latency = np.nan
                                peak1_value = np.nan
                                peak2_value = np.nan
                                ptp_amp = np.nan

                channel_epoch_results[ch_name].append(
                    {
                        "ep": ep,
                        "onset": onset_latency,
                        "p1": peak1_latency,
                        "p2": peak2_latency,
                        "pv1": p1_corr_value,
                        "pv2": peak2_value,
                        "ptp": ptp_amp,
                        "sig": sig,
                    }
                )

        corr_min_median = float(STARTSTOP_CHANNEL_MIN_MEDIAN_CORR)
        min_valid_frac = float(STARTSTOP_CHANNEL_MIN_VALID_FRAC)
        for ch in epochs.ch_names:
            if ch in art_set:
                continue
            if ch not in tmpl_cfg:
                continue

            entries = channel_epoch_results[ch]
            tmpl = tmpl_cfg[ch]

            sim_mask = (times >= STARTSTOP_SIM_TMIN) & (times <= STARTSTOP_SIM_TMAX)
            tmpl_seg = tmpl[sim_mask]
            tmpl_seg = tmpl_seg - np.mean(tmpl[baseline_mask])

            corrs = []
            valid_flags = []
            for e in entries:
                ep = e["ep"]
                is_valid = not np.isnan(e["p1"])
                valid_flags.append(is_valid)

                sig_ep = data_filt[ep, epochs.ch_names.index(ch), :]
                seg = sig_ep[sim_mask] - np.mean(sig_ep[baseline_mask])

                if np.std(seg) == 0 or np.std(tmpl_seg) == 0:
                    corrs.append(0.0)
                else:
                    c = float(np.corrcoef(seg, tmpl_seg)[0, 1])
                    if np.isnan(c):
                        c = 0.0
                    corrs.append(c)

            valid_fraction = np.mean(valid_flags) if len(valid_flags) else 0.0
            median_corr = float(np.median(corrs)) if len(corrs) else 0.0

            if (valid_fraction < min_valid_frac) and (median_corr < corr_min_median):
                for e in entries:
                    e["onset"] = np.nan
                    e["p1"] = np.nan
                    e["p2"] = np.nan
                    e["pv1"] = np.nan
                    e["pv2"] = np.nan
                    e["ptp"] = np.nan

            for e in entries:
                latency_markers[ch].append(
                    {"epoch": e["ep"], "onset": e["onset"], "peak1": e["p1"], "peak2": e["p2"]}
                )

            for e in entries:
                results["Configuration"].append(str(condition))
                results["Stim. amplitude"].append("start")
                results["Epoch"].append(e["ep"])
                results["Channel"].append(ch)
                results["Onset latency"].append(e["onset"])
                results["Peak1 latency"].append(e["p1"])
                results["Peak2 latency"].append(e["p2"])
                results["Peak1 value"].append(e["pv1"])
                results["Peak2 value"].append(e["pv2"])
                results["PTP amplitude"].append(e["ptp"])
                results["Time series"].append(e["sig"])

            valid_epochs = [e["ep"] for e in entries if not np.isnan(e["p1"])]
            if valid_epochs:
                for ep_idx in valid_epochs:
                    group_store_by_condition[str(condition)][ch].append(
                        data_epoched[ep_idx, epochs.ch_names.index(ch), :]
                    )

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            all_results_rows.append(df_results)

        epochs_to_plot = {
            ch: [e["ep"] for e in entries if not np.isnan(e["p1"])]
            for ch, entries in channel_epoch_results.items()
        }
        has_any_detection = any(len(eps) > 0 for eps in epochs_to_plot.values())

        p1_by_epoch_channel: dict[int, dict[str, float]] = {}
        for ch, entries in channel_epoch_results.items():
            for e in entries:
                if np.isnan(e["p1"]):
                    continue
                p1_by_epoch_channel.setdefault(e["ep"], {})[ch] = float(e["p1"])

        detected_channels_by_epoch = defaultdict(list)
        for ch, entries in channel_epoch_results.items():
            for e in entries:
                if not np.isnan(e["p1"]):
                    detected_channels_by_epoch[e["ep"]].append(ch)

        if detected_channels_by_epoch and has_any_detection:
            raw_epochs_dir = paths["raw_epochs_dir"] / safe_condition
            raw_epochs_dir.mkdir(parents=True, exist_ok=True)
            epochs_wide = mne.Epochs(
                start_raw,
                events,
                event_id=1,
                tmin=-1.0,
                tmax=1.0,
                baseline=None,
                preload=True,
                picks=ch_names,
            )
            wide_times = epochs_wide.times
            wide_data = epochs_wide.get_data()

            for ep_idx, ch_list in detected_channels_by_epoch.items():
                if ep_idx >= wide_data.shape[0]:
                    continue
                max_abs = None
                for ch in ch_list:
                    p1_t = p1_by_epoch_channel.get(ep_idx, {}).get(ch)
                    if p1_t is None or np.isnan(p1_t):
                        continue
                    ch_idx = ch_names.index(ch)
                    win_mask = np.abs(wide_times - p1_t) <= 0.05
                    if not np.any(win_mask):
                        continue
                    val = float(np.nanmax(np.abs(wide_data[ep_idx, ch_idx, win_mask])))
                    if max_abs is None or val > max_abs:
                        max_abs = val
                if max_abs is None:
                    max_abs = float(np.nanmax(np.abs(wide_data[ep_idx]))) if wide_data.size else 0.0
                scale = (max_abs * 2.5) if max_abs > 0 else None
                safe_chs = [re.sub(r"[^\w\-_\. ]", "_", str(ch)) for ch in ch_list]
                ch_suffix = "+".join(safe_chs)
                if len(ch_suffix) > 80:
                    ch_suffix = f"{ch_suffix[:77]}..."
                out_path = raw_epochs_dir / f"epoch_{ep_idx:03d}__{ch_suffix}.png"
                fig = epochs_wide[ep_idx].plot(
                    picks=ch_names,
                    scalings={"eeg": scale} if scale is not None else None,
                    show=False,
                    title=f"{condition} | epoch {ep_idx} | {', '.join(ch_list)}",
                )
                if isinstance(fig, list):
                    fig = fig[0] if fig else None
                if fig is not None:
                    fig.savefig(out_path)
                    plt.close(fig)

        if has_any_detection:
            plot_epochs_panel(
                data_epoched=data_epoched,
                times=times,
                ch_names=ch_names,
                epochs_ch_names=epochs.ch_names,
                art_chans=art_set,
                latency_markers=latency_markers,
                title=f"{safe_condition}_startstop",
                out_path=plots_grid_dir / f"{safe_condition}_startstop.png",
                show_grid=True,
                show_markers=True,
                epochs_to_plot=epochs_to_plot,
            )

            plot_epochs_panel(
                data_epoched=data_epoched,
                times=times,
                ch_names=ch_names,
                epochs_ch_names=epochs.ch_names,
                art_chans=art_set,
                latency_markers=latency_markers,
                title=f"{safe_condition}_startstop",
                out_path=plots_plain_dir / f"{safe_condition}_startstop.png",
                show_grid=False,
                show_markers=False,
                epochs_to_plot=epochs_to_plot,
            )

    if group_store_by_condition and grouped_times is not None:
        plot_grouped_by_condition(
            group_store_by_condition,
            grouped_times,
            paths["plots_grouped_dir"],
            ch_names,
        )

    print("[STARTSTOP] Writing outputs...", flush=True)
    if template_match_rows:
        pd.DataFrame(template_match_rows).to_csv(
            excel_dir / "STARTSTOP_template_anchor_matches.csv", index=False
        )
    if template_discard_rows:
        pd.DataFrame(template_discard_rows).to_csv(
            excel_dir / "STARTSTOP_template_anchor_discarded.csv", index=False
        )
    if channel_template_rows:
        pd.DataFrame(channel_template_rows).to_csv(
            excel_dir / "STARTSTOP_channel_template_selection.csv", index=False
        )

    if not all_results_rows:
        print("[STARTSTOP] 0 patterns detected", flush=True)
        return

    df_results = pd.concat(all_results_rows, ignore_index=True)
    df_results.to_csv(excel_dir / "Large_dataset_emg_response_metrics.csv", index=False)

    valid_mask = df_results["Peak1 latency"].notna()
    valid_counts = (
        df_results.loc[valid_mask]
        .groupby(["Configuration", "Channel"], dropna=False)
        .size()
        .reset_index(name="n_valid")
    )
    if valid_counts.empty:
        print("[STARTSTOP] 0 patterns detected", flush=True)
    else:
        print("STARTSTOP valid patterns per configuration/channel:")
        for _row in valid_counts.itertuples(index=False):
            print(f"  {_row.Configuration} | {_row.Channel}: {_row.n_valid}")
        detected_channels = sorted(set(valid_counts["Channel"].astype(str)))
        print(f"[STARTSTOP] detected channels: {', '.join(detected_channels)}", flush=True)

    metrics = [
        "Onset latency",
        "Peak1 latency",
        "Peak2 latency",
        "Peak1 value",
        "Peak2 value",
        "PTP amplitude",
    ]

    df = df_results.copy()
    df = df.dropna(subset=metrics, how="all").reset_index(drop=True)
    df["Stim. amplitude"] = df["Stim. amplitude"].astype(str)
    df["Configuration"] = df["Configuration"].astype(str)
    df["Channel"] = df["Channel"].astype(str)
    df["Stim_amp_num"] = df["Stim. amplitude"].apply(amp_to_number)

    group_cols = ["Configuration", "Stim. amplitude", "Stim_amp_num", "Channel"]
    summary_rows = []

    for metric in metrics:
        g = df[group_cols + [metric]].copy()
        g[metric] = pd.to_numeric(g[metric], errors="coerce")
        grouped = g.groupby(group_cols, dropna=False)[metric]

        tmp = grouped.agg(
            n_total="size",
            n_valid=lambda x: x.notna().sum(),
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        ).reset_index()

        tmp["iqr"] = tmp["q75"] - tmp["q25"]
        tmp["Metric"] = metric
        summary_rows.append(tmp)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary[
        [
            "Configuration",
            "Stim. amplitude",
            "Stim_amp_num",
            "Channel",
            "Metric",
            "n_total",
            "n_valid",
            "mean",
            "std",
            "median",
            "q25",
            "q75",
            "iqr",
            "min",
            "max",
        ]
    ]
    summary = summary.sort_values(
        by=["Configuration", "Stim_amp_num", "Stim. amplitude", "Channel", "Metric"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary_csv = excel_dir / "Summary_stats_by_config_amp_channel.csv"
    summary.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(
        excel_dir / "Summary_stats_by_config_amp_channel_by_config.xlsx",
        engine="xlsxwriter",
    ) as writer:
        for config, dfc in summary.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc = dfc.sort_values(
                by=["Stim_amp_num", "Stim. amplitude", "Channel", "Metric"],
                kind="mergesort",
            ).reset_index(drop=True)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(
        excel_dir / "Large_dataset_emg_response_metrics_by_config.xlsx",
        engine="xlsxwriter",
    ) as writer:
        for config, dfc in df_results.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    plot_boxplots(df, boxplot_dir)


def _excel_safe_sheet_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    return name[:31]


def _default_output_root_for_input(edf_path: Path) -> Path:
    # Prefer placing outputs under a project-level results/ sibling of data/,
    # even when input files live in nested data subfolders.
    data_ancestor: Path | None = None
    for parent in edf_path.resolve().parents:
        if parent.name.lower() == "data":
            data_ancestor = parent
            break
    if data_ancestor is not None:
        return data_ancestor.parent / "results" / edf_path.stem
    return edf_path.parent.parent / "results" / edf_path.stem


def run_pipeline(
    edf_path: str | Path,
    output_dir: str | Path | None = None,
    template_mode: str = "per_file_mean",
    startstop_mode: bool | None = None,
) -> Path:
    edf_path = Path(edf_path)
    if output_dir is None:
        output_root = _default_output_root_for_input(edf_path)
    else:
        output_root = Path(output_dir)

    use_startstop = startstop_mode if startstop_mode is not None else STARTSTOP_MODE
    # Keep both flows quiet except explicit stage messages and tqdm.
    old_mne_log_level = mne.set_log_level("ERROR", return_old_level=True)
    if use_startstop:
        print("[STARTSTOP] Preparing data...", flush=True)
    else:
        print("[SIR] Preparing data...", flush=True)
    paths = build_output_dirs(output_root, startstop_mode=use_startstop)
    crops_dir = paths["crops_dir"]
    epochs_dir = paths["epochs_dir"]
    excel_dir = paths["excel_dir"]
    boxplot_dir = paths["boxplot_dir"]
    plots_grid_dir = paths["plots_grid_dir"]
    plots_plain_dir = paths["plots_plain_dir"]
    plots_grouped_dir = paths["plots_grouped_dir"]
    templates_dir = paths["templates_dir"]
    startstop_dir = paths["startstop_dir"]

    if edf_path.suffix.lower() == ".mat":
        raw = _load_raw_from_mat(edf_path)
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=True)

    original_fif_path = paths["data_dir"] / f"{edf_path.stem}_original_raw.fif"
    raw.save(original_fif_path, overwrite=True)
    default_art_chans = _resolve_art_channels(raw, ARTCHAN)
    default_art_set = set(default_art_chans)
    nyq = raw.info["sfreq"] / 2.0
    notch_freqs = [f for f in (50, 100, 150, 200, 250, 300) if f < nyq]
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, method="fir", phase="zero")
    raw.filter(l_freq=RAW_BANDPASS_L_FREQ, h_freq=RAW_BANDPASS_H_FREQ, method="fir", phase="zero")
    # When both are enabled: first subtract artifact channels, then CAR (mean over non-artifact channels only).
    if ARTIFACT_REREF:
        _apply_artifact_reref(raw, default_art_chans)
    if CAR_REREF:
        _apply_car_reref(raw, default_art_chans)

    preproc_path = paths["data_dir"] / f"{edf_path.stem}_preprocessed_raw.fif"
    raw.save(preproc_path, overwrite=True)

    if use_startstop:
        _run_startstop_analysis(raw, startstop_dir, default_art_chans)
        if old_mne_log_level is not None:
            mne.set_log_level(old_mne_log_level)
        return output_root

    print("[SIR] Creating annotation crops...", flush=True)
    create_annotation_crops(raw, crops_dir, overwrite=True)

    file_list = list_crop_files(crops_dir)

    valid_modes = {"per_file_mean", "config_grand_avg", "max_amp_only"}
    if template_mode not in valid_modes:
        if old_mne_log_level is not None:
            mne.set_log_level(old_mne_log_level)
        raise ValueError(f"template_mode must be one of {sorted(valid_modes)}")

    config_max_amp = {}
    if template_mode == "max_amp_only":
        for file_name in file_list:
            configuration = file_name.split("-")[0]
            stim_amp_raw = file_name.split("_")[1].split(".fif")[0]
            amp_num = amp_to_number(stim_amp_raw)
            if np.isnan(amp_num):
                continue
            current = config_max_amp.get(configuration)
            if current is None or amp_num > current:
                config_max_amp[configuration] = amp_num

    config_waveforms = defaultdict(lambda: defaultdict(list))
    config_meta: dict[str, dict[str, np.ndarray | float]] = {}
    config_templates = defaultdict(dict)
    config_template_markers = defaultdict(dict)

    print("[SIR] Finding templates...", flush=True)
    for file_name in file_list:
        file_path = crops_dir / file_name
        if template_mode == "max_amp_only":
            configuration = file_name.split("-")[0]
            stim_amp_raw = file_name.split("_")[1].split(".fif")[0]
            amp_num = amp_to_number(stim_amp_raw)
            max_amp = config_max_amp.get(configuration)
            if max_amp is None or np.isnan(amp_num) or not np.isclose(amp_num, max_amp):
                continue
        raw_file = mne.io.read_raw_fif(file_path, preload=False)

        sfreq = raw_file.info["sfreq"]
        # No additional low-pass in this flow

        art_chans = _resolve_art_channels(raw_file, ARTCHAN, fallback_chans=default_art_chans)
        art_set = set(art_chans)
        art = _get_art_signal(raw_file, art_chans)
        threshold = THRESH * np.std(art)
        peaks, _ = find_peaks(-art, height=threshold, distance=sfreq * 0.1)
        if len(peaks) <= MIN_VALID_EPOCHS:
            continue

        events = np.column_stack(
            [peaks + raw_file.first_samp, np.zeros_like(peaks, dtype=int), np.ones_like(peaks, dtype=int)]
        )
        epochs = mne.Epochs(
            raw_file,
            events,
            event_id=1,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,
            preload=True,
        )

        times = epochs.times
        configuration = file_name.split("-")[0]

        if configuration not in config_meta:
            config_meta[configuration] = {"times": times.copy(), "sfreq": sfreq}
        else:
            if sfreq != config_meta[configuration]["sfreq"] or len(times) != len(config_meta[configuration]["times"]):
                continue

        data_epoched = epochs.get_data()
        data_filt = np.zeros_like(data_epoched)
        for ep in range(len(epochs)):
            for ch_idx in range(len(epochs.ch_names)):
                data_filt[ep, ch_idx, :] = data_epoched[ep, ch_idx, :]

        for ch_idx, ch_name in enumerate(epochs.ch_names):
            if ch_name in art_set:
                continue
            tmpl_file = np.median(data_filt[:, ch_idx, :], axis=0)
            config_waveforms[configuration][ch_name].append(tmpl_file)

    for configuration, ch_dict in config_waveforms.items():
        times = config_meta[configuration]["times"]
        sfreq = float(config_meta[configuration]["sfreq"])

        baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
        prestim_prom_mask = (times >= STIM_PROM_PRESTIM_TMIN) & (times <= STIM_PROM_PRESTIM_TMAX)
        if np.sum(prestim_prom_mask) < 5:
            prestim_prom_mask = baseline_mask
        poststim_prom_mask = (times >= STIM_PROM_BASELINE_TMIN) & (times <= STIM_PROM_BASELINE_TMAX)

        for ch_name, waves in ch_dict.items():
            if len(waves) == 0:
                config_templates[configuration][ch_name] = None
                config_template_markers[configuration][ch_name] = dict(onset=np.nan, p1=np.nan, p2=np.nan)
                continue

            tmpl_global = np.median(np.stack(waves, axis=0), axis=0)

            onset_t = detect_onset_rectified(
                tmpl_global,
                times,
                sfreq,
                baseline_mask=baseline_mask,
                onset_tmin=0.003,
                onset_tmax=0.035,
                k=4,
                sustain_ms=5,
            )

            prom_baseline_mask = _choose_silent_prominence_baseline_mask(
                sig=tmpl_global,
                prestim_mask=prestim_prom_mask,
                poststim_mask=poststim_prom_mask,
            )
            p1_t, p1_v, p2_t, p2_v = detect_template_peaks(
                tmpl_global,
                times,
                sfreq,
                baseline_mask=prom_baseline_mask,
                onset_latency=onset_t,
                resp_tmax=RESP_TMAX,
                min_prom_k=0,
                peak2_max_gap_ms=20.0,
                min_width_ms=0.6,
                amp_min_uV=10,
            )

            config_templates[configuration][ch_name] = tmpl_global
            config_template_markers[configuration][ch_name] = dict(onset=onset_t, p1=p1_t, p2=p2_t)

    for configuration, ch_dict in config_templates.items():
        times = config_meta[configuration]["times"]
        for ch_name, template in ch_dict.items():
            if template is None or ch_name in default_art_set:
                continue
            markers = config_template_markers[configuration].get(
                ch_name, {"onset": np.nan, "p1": np.nan, "p2": np.nan}
            )
            safe_config = re.sub(r"[^\w\-\+\. ]", "_", str(configuration))
            safe_ch = re.sub(r"[^\w\-_\. ]", "_", str(ch_name))
            out_path = templates_dir / safe_config / f"{safe_ch}_template.png"
            plot_template_with_markers(
                times=times,
                template=template,
                markers=markers,
                out_path=out_path,
                title=f"Template: {configuration} | {ch_name}",
            )

    group_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results = {key: [] for key in [
        "Configuration",
        "Stim. amplitude",
        "Epoch",
        "Channel",
        "Onset latency",
        "Peak1 latency",
        "Peak2 latency",
        "Peak1 value",
        "Peak2 value",
        "PTP amplitude",
        "Time series",
    ]}

    print("[SIR] Detecting epochs...", flush=True)
    for file_name in tqdm(file_list, desc="PASS 2: detect epochs"):
        file_path = crops_dir / file_name
        raw_file = mne.io.read_raw_fif(file_path)
        _apply_special_crops(raw_file, file_name)

        sfreq = raw_file.info["sfreq"]

        art_chans = _resolve_art_channels(raw_file, ARTCHAN, fallback_chans=default_art_chans)
        art_set = set(art_chans)
        art = _get_art_signal(raw_file, art_chans)
        threshold = THRESH * np.std(art)
        peaks, _ = find_peaks(-art, height=threshold, distance=sfreq * 0.1)

        if len(peaks) <= MIN_VALID_EPOCHS:
            continue

        events = np.column_stack(
            [peaks + raw_file.first_samp, np.zeros_like(peaks, dtype=int), np.ones_like(peaks, dtype=int)]
        )
        epochs = mne.Epochs(
            raw_file,
            events,
            event_id=1,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,
            preload=True,
        )

        epochs.save(epochs_dir / f"{file_name.split('.fif')[0]}-epo.fif", overwrite=True)

        epochs_array = epochs.get_data()
        channel_names = epochs.ch_names
        times_to_save = epochs.times
        ep_df = pd.DataFrame(
            epochs_array.reshape(epochs_array.shape[0], -1),
            columns=[f"{ch}_{t}" for ch in channel_names for t in times_to_save],
        )
        ep_df.to_csv(epochs_dir / f"{file_name.split('.fif')[0]}-epo.txt", index=False, sep="\t")

        data_epoched = epochs.get_data()
        times = epochs.times

        configuration = file_name.split("-")[0]
        stim_amp_raw = file_name.split("_")[1].split(".fif")[0]

        for ch in epochs.ch_names:
            if ch in art_set:
                continue
            ch_idx = epochs.ch_names.index(ch)
            file_mean = data_epoched[:, ch_idx, :].mean(axis=0)
            group_store[configuration][ch][stim_amp_raw].append(file_mean)

        baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
        prestim_prom_mask = (times >= STIM_PROM_PRESTIM_TMIN) & (times <= STIM_PROM_PRESTIM_TMAX)
        if np.sum(prestim_prom_mask) < 5:
            prestim_prom_mask = baseline_mask
        poststim_prom_mask = (times >= STIM_PROM_BASELINE_TMIN) & (times <= STIM_PROM_BASELINE_TMAX)

        tmpl_cfg = config_templates[configuration]
        markers_cfg = config_template_markers[configuration]

        data_filt = data_epoched

        for ch_idx, ch_name in enumerate(epochs.ch_names):
            if ch_name in art_set:
                continue

            # In max-amp mode, keep config-level templates/markers from PASS 1
            # (built on the strongest amplitude) and do not overwrite per file.
            if template_mode == "max_amp_only":
                continue

            tmpl = np.mean(data_filt[:, ch_idx, :], axis=0)
            onset_t = detect_onset_rectified(
                tmpl,
                times,
                sfreq,
                baseline_mask=baseline_mask,
                onset_tmin=0.003,
                onset_tmax=0.035,
                k=0.8,
                sustain_ms=3,
            )

            prom_k = get_prominence_k(file_name, ch_name)
            prom_baseline_mask = _choose_silent_prominence_baseline_mask(
                sig=tmpl,
                prestim_mask=prestim_prom_mask,
                poststim_mask=poststim_prom_mask,
            )
            p1_t, p1_v, p2_t, p2_v = detect_template_peaks(
                tmpl,
                times,
                sfreq,
                baseline_mask=prom_baseline_mask,
                onset_latency=onset_t,
                resp_tmax=RESP_TMAX,
                min_prom_k=prom_k,
                peak2_max_gap_ms=20.0,
                min_width_ms=0.6,
                amp_min_uV=10,
            )

            tmpl_cfg[ch_name] = tmpl
            markers_cfg[ch_name] = dict(onset=onset_t, p1=p1_t, p2=p2_t)

        channel_epoch_results = {ch: [] for ch in epochs.ch_names if ch not in art_set}
        latency_markers = {ch: [] for ch in epochs.ch_names if ch not in art_set}

        for ep in range(len(epochs)):
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                if ch_name in art_set:
                    continue

                sig = data_epoched[ep, ch_idx, :]
                sig_f = data_filt[ep, ch_idx, :]

                base = sig_f[baseline_mask]
                bstd = base.std()
                if bstd == 0 or np.isnan(bstd):
                    channel_epoch_results[ch_name].append(
                        {
                            "ep": ep,
                            "onset": np.nan,
                            "p1": np.nan,
                            "p2": np.nan,
                            "pv1": np.nan,
                            "pv2": np.nan,
                            "ptp": np.nan,
                            "sig": sig,
                        }
                    )
                    continue

                t_on_tmpl = markers_cfg[ch_name]["onset"]
                t_p1 = markers_cfg[ch_name]["p1"]
                t_p2 = markers_cfg[ch_name]["p2"]
                tmpl = tmpl_cfg[ch_name]

                onset_latency = detect_onset_near_template(
                    sig_f,
                    times,
                    sfreq,
                    baseline_mask,
                    t_on_tmpl,
                    win_ms=2,
                    k=1,
                    sustain_ms=2,
                )

                peak1_latency = peak2_latency = np.nan
                peak1_value = peak2_value = np.nan

                if not np.isnan(t_p1):
                    idx_tmpl_p1 = int(np.argmin(np.abs(times - t_p1)))
                    pol1 = np.sign(tmpl[idx_tmpl_p1] - np.mean(tmpl[baseline_mask]))
                    pol1 = +1 if pol1 >= 0 else -1

                    tmpl_p1_val = tmpl[idx_tmpl_p1]
                    peak1_latency, peak1_value = detect_peak_in_window(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        baseline_mask=baseline_mask,
                        t_center=t_p1,
                        win_ms=5.0,
                        polarity=pol1,
                        amp_min_uV=10.0,
                        min_width_ms=0.4,
                        choose="nearest",
                        template_peak_val=tmpl_p1_val,
                        min_rel_to_template=0.1,
                    )
                else:
                    pol1 = +1

                if not np.isnan(t_p2):
                    idx_tmpl_p2 = int(np.argmin(np.abs(times - t_p2)))
                    pol2 = np.sign(tmpl[idx_tmpl_p2] - np.mean(tmpl[baseline_mask]))
                    pol2 = +1 if pol2 >= 0 else -1

                    tmpl_p2_val = tmpl[idx_tmpl_p2]
                    peak2_latency, peak2_value = detect_peak_in_window(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        baseline_mask=baseline_mask,
                        t_center=t_p2,
                        win_ms=5.0,
                        polarity=pol2,
                        amp_min_uV=10.0,
                        min_width_ms=0.4,
                        choose="nearest",
                        template_peak_val=tmpl_p2_val,
                        min_rel_to_template=0.1,
                    )
                else:
                    pol2 = +1

                if not np.isnan(peak1_latency):
                    peak1_latency, peak1_value = pick_epoch_value_near_latency(
                        sig,
                        times,
                        peak1_latency,
                        sfreq,
                        win_ms=1.0,
                        polarity=pol1,
                    )

                if not np.isnan(peak2_latency):
                    peak2_latency, peak2_value = pick_epoch_value_near_latency(
                        sig,
                        times,
                        peak2_latency,
                        sfreq,
                        win_ms=1.0,
                        polarity=pol2,
                    )

                if (not np.isnan(peak1_latency)) and (not np.isnan(peak2_latency)):
                    if peak2_latency <= peak1_latency:
                        peak2_latency = np.nan
                        peak2_value = np.nan

                if (not np.isnan(peak1_value)) and (not np.isnan(peak2_value)):
                    if np.sign(peak1_value) == np.sign(peak2_value):
                        peak2_latency = np.nan
                        peak2_value = np.nan

                p1_corr_value = peak1_value
                if (
                    (not np.isnan(peak1_latency))
                    and (not np.isnan(peak2_latency))
                    and (not np.isnan(peak1_value))
                    and (not np.isnan(peak2_value))
                ):
                    extra_lat, extra_val = find_extra_p1_peak(
                        sig_f=sig_f,
                        times=times,
                        sfreq=sfreq,
                        p1_lat=peak1_latency,
                        p2_lat=peak2_latency,
                        p1_polarity=pol1,
                        p2_hint_lat=np.nan,
                        guard_ms=1.0,
                        hint_ms=4.0,
                        min_width_ms=0.4,
                        amp_min_uV=10.0,
                        choose="dominant",
                    )
                    if not np.isnan(extra_val):
                        p1_corr_value = add_extra_peak_to_p1(peak1_value, pol1, extra_val)

                if (not np.isnan(p1_corr_value)) and (not np.isnan(peak2_value)):
                    ptp_amp = float(np.abs(p1_corr_value - peak2_value))
                else:
                    ptp_amp = np.nan

                if (not np.isnan(ptp_amp)) and (ptp_amp < 30e-6):
                    onset_latency = np.nan
                    peak1_latency = np.nan
                    peak2_latency = np.nan
                    peak1_value = np.nan
                    peak2_value = np.nan
                    ptp_amp = np.nan

                if (not np.isnan(peak1_value)) and (np.abs(peak1_value) <= 10e-6):
                    onset_latency = np.nan
                    peak1_latency = np.nan
                    peak2_latency = np.nan
                    peak1_value = np.nan
                    peak2_value = np.nan
                    ptp_amp = np.nan

                if np.isnan(peak2_value) or np.isnan(peak1_value):
                    onset_latency = np.nan
                    peak1_latency = np.nan
                    peak2_latency = np.nan
                    peak1_value = np.nan
                    peak2_value = np.nan
                    ptp_amp = np.nan

                # Reject likely artifact-driven detections: if the full epoch
                # on this channel is highly correlated (including opposite
                # polarity) with any artifact channel in the same epoch.
                if STIM_EPOCH_ARTIFACT_CORR_REJECTION and (not np.isnan(peak1_latency)):
                    sig_centered = sig_f - np.mean(sig_f)
                    sig_std = float(np.std(sig_centered))
                    if sig_std > 0:
                        artifact_like = False
                        for art_ch in art_chans:
                            if art_ch not in epochs.ch_names:
                                continue
                            art_idx = epochs.ch_names.index(art_ch)
                            art_sig = np.asarray(data_filt[ep, art_idx, :], dtype=float)
                            art_centered = art_sig - np.mean(art_sig)
                            art_std = float(np.std(art_centered))
                            if art_std <= 0:
                                continue
                            corr = float(np.corrcoef(sig_centered, art_centered)[0, 1])
                            if np.isnan(corr):
                                continue
                            if abs(corr) >= float(STIM_EPOCH_ARTIFACT_ABS_CORR_THR):
                                artifact_like = True
                                break
                        if artifact_like:
                            onset_latency = np.nan
                            peak1_latency = np.nan
                            peak2_latency = np.nan
                            peak1_value = np.nan
                            peak2_value = np.nan
                            ptp_amp = np.nan

                channel_epoch_results[ch_name].append(
                    {
                        "ep": ep,
                        "onset": onset_latency,
                        "p1": peak1_latency,
                        "p2": peak2_latency,
                        "pv1": p1_corr_value,
                        "pv2": peak2_value,
                        "ptp": ptp_amp,
                        "sig": sig,
                    }
                )

        corr_min_median = 0.5
        min_valid_frac = 0.4

        for ch in raw_file.info["ch_names"]:
            if ch in art_set:
                continue

            entries = channel_epoch_results[ch]
            tmpl = tmpl_cfg[ch]

            sim_mask = (times >= 0.003) & (times <= RESP_TMAX)
            tmpl_seg = tmpl[sim_mask]
            tmpl_seg = tmpl_seg - np.mean(tmpl[baseline_mask])

            corrs = []
            valid_flags = []
            for e in entries:
                ep = e["ep"]
                is_valid = not np.isnan(e["p1"])
                valid_flags.append(is_valid)

                sig_ep = data_filt[ep, epochs.ch_names.index(ch), :]
                seg = sig_ep[sim_mask] - np.mean(sig_ep[baseline_mask])

                if np.std(seg) == 0 or np.std(tmpl_seg) == 0:
                    corrs.append(0.0)
                else:
                    c = float(np.corrcoef(seg, tmpl_seg)[0, 1])
                    if np.isnan(c):
                        c = 0.0
                    corrs.append(c)

            valid_fraction = np.mean(valid_flags) if len(valid_flags) else 0.0
            median_corr = float(np.median(corrs)) if len(corrs) else 0.0

            if (valid_fraction < min_valid_frac) and (median_corr < corr_min_median):
                for e in entries:
                    e["onset"] = np.nan
                    e["p1"] = np.nan
                    e["p2"] = np.nan
                    e["pv1"] = np.nan
                    e["pv2"] = np.nan
                    e["ptp"] = np.nan

            for e in entries:
                latency_markers[ch].append(
                    {"epoch": e["ep"], "onset": e["onset"], "peak1": e["p1"], "peak2": e["p2"]}
                )

            for e in entries:
                results["Configuration"].append(file_name.split("-")[0])
                results["Stim. amplitude"].append(file_name.split("_")[1].split(".fif")[0])
                results["Epoch"].append(e["ep"])
                results["Channel"].append(ch)
                results["Onset latency"].append(e["onset"])
                results["Peak1 latency"].append(e["p1"])
                results["Peak2 latency"].append(e["p2"])
                results["Peak1 value"].append(e["pv1"])
                results["Peak2 value"].append(e["pv2"])
                results["PTP amplitude"].append(e["ptp"])
                results["Time series"].append(e["sig"])

        base_name = file_name.split(".fif")[0]
        plot_epochs_panel(
            data_epoched=data_epoched,
            times=times,
            ch_names=raw_file.info["ch_names"],
            epochs_ch_names=epochs.ch_names,
            art_chans=art_set,
            latency_markers=latency_markers,
            title=base_name,
            out_path=plots_grid_dir / f"{base_name}.png",
            show_grid=True,
            show_markers=True,
        )

        plot_epochs_panel(
            data_epoched=data_epoched,
            times=times,
            ch_names=raw_file.info["ch_names"],
            epochs_ch_names=epochs.ch_names,
            art_chans=art_set,
            latency_markers=latency_markers,
            title=base_name,
            out_path=plots_plain_dir / f"{base_name}.png",
            show_grid=False,
            show_markers=False,
        )

    print("[SIR] Writing outputs...", flush=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(excel_dir / "Large_dataset_emg_response_metrics.csv", index=False)

    plot_grouped_by_amplitude(group_store, times, plots_grouped_dir, default_art_set)

    metrics = [
        "Onset latency",
        "Peak1 latency",
        "Peak2 latency",
        "Peak1 value",
        "Peak2 value",
        "PTP amplitude",
    ]

    df = df_results.copy()
    df = df.dropna(subset=metrics, how="all").reset_index(drop=True)
    df["Stim. amplitude"] = df["Stim. amplitude"].astype(str)
    df["Configuration"] = df["Configuration"].astype(str)
    df["Channel"] = df["Channel"].astype(str)
    df["Stim_amp_num"] = df["Stim. amplitude"].apply(amp_to_number)

    group_cols = ["Configuration", "Stim. amplitude", "Stim_amp_num", "Channel"]
    summary_rows = []

    for metric in metrics:
        g = df[group_cols + [metric]].copy()
        g[metric] = pd.to_numeric(g[metric], errors="coerce")
        grouped = g.groupby(group_cols, dropna=False)[metric]

        tmp = grouped.agg(
            n_total="size",
            n_valid=lambda x: x.notna().sum(),
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        ).reset_index()

        tmp["iqr"] = tmp["q75"] - tmp["q25"]
        tmp["Metric"] = metric
        summary_rows.append(tmp)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary[
        [
            "Configuration",
            "Stim. amplitude",
            "Stim_amp_num",
            "Channel",
            "Metric",
            "n_total",
            "n_valid",
            "mean",
            "std",
            "median",
            "q25",
            "q75",
            "iqr",
            "min",
            "max",
        ]
    ]
    summary = summary.sort_values(
        by=["Configuration", "Stim_amp_num", "Stim. amplitude", "Channel", "Metric"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary_csv = excel_dir / "Summary_stats_by_config_amp_channel.csv"
    summary.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(
        excel_dir / "Summary_stats_by_config_amp_channel_by_config.xlsx",
        engine="xlsxwriter",
    ) as writer:
        for config, dfc in summary.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc = dfc.sort_values(
                by=["Stim_amp_num", "Stim. amplitude", "Channel", "Metric"],
                kind="mergesort",
            ).reset_index(drop=True)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(
        excel_dir / "Large_dataset_emg_response_metrics_by_config.xlsx",
        engine="xlsxwriter",
    ) as writer:
        for config, dfc in df_results.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    plot_boxplots(df, boxplot_dir)
    if old_mne_log_level is not None:
        mne.set_log_level(old_mne_log_level)

    return output_root

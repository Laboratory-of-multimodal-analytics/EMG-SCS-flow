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
    SPONTANEOUS_EMG_ANALYSIS,
    SPONTANEOUS_EMG_WINDOW_MS,
    SPONTANEOUS_EMG_ENV_HOP_MS,
    SPONTANEOUS_EMG_UV_SCALE,
    SPONTANEOUS_EMG_HP_FREQ,
    SPONTANEOUS_EMG_BURST_DETECTION,
    SPONTANEOUS_EMG_BURST_MIN_SNR,
    SPONTANEOUS_EMG_BURST_MIN_PEAK_UV,
    SPONTANEOUS_EMG_BURST_BASELINE_PCT,
    SPONTANEOUS_EMG_BURST_PEAK_PCT,
    SPONTANEOUS_EMG_BURST_THRESH_FRAC,
    SPONTANEOUS_EMG_BURST_KEEP_FRAC,
    SPONTANEOUS_EMG_BURST_MIN_MS,
    SPONTANEOUS_EMG_BURST_MERGE_MS,

    SPONTANEOUS_EMG_ACTIVE_MAD_K,

    BASELINE_TMAX,
    BASELINE_TMIN,
    EPOCH_TMAX,
    EPOCH_TMIN,
    MIN_VALID_EPOCHS,
    RAW_NOTCH_FREQ,
    RAW_BANDPASS_H_FREQ,
    RAW_BANDPASS_L_FREQ,
    RESP_TMAX,
    RESP_TMIN,
    SIR_TM_MIN_CORR,
    SIR_TM_SCALES,
    SIR_TEMPLATE_PER_AMPLITUDE,
    STIM_EPOCH_ARTIFACT_ABS_CORR_THR,
    STIM_EPOCH_ARTIFACT_CORR_REJECTION,
    STIM_ONSET_MAX_DEV_S,
    STIM_P1_ABS_MIN_UV,
    STIM_PEAK_AMP_MIN_UV,
    STIM_PTP_MIN_UV,
    STIM_MIN_INTER_TRIAL_CORR,
    STIM_CHANNEL_MIN_MEDIAN_CORR,
    STIM_CHANNEL_MIN_VALID_FRAC,
    STARTSTOP_MIN_DIST_MS,
    STARTSTOP_MODE,
    STARTSTOP_SAVE_DETECTION_MARKERS,
    STARTSTOP_DETECTION_MARKER_MERGE_MS,
    STARTSTOP_LEAKAGE_CORR_REJECTION,
    STARTSTOP_TM_NO_POSTHOC_CHECKS,
    STARTSTOP_CHANNEL_MIN_MEDIAN_CORR,
    STARTSTOP_CHANNEL_MIN_INTERTRIAL_CORR,
    STARTSTOP_EPOCH_MIN_INTERTRIAL_CORR,
    STARTSTOP_CHANNEL_MIN_VALID_FRAC,
    STARTSTOP_MIN_DETECTIONS_PER_CHANNEL,
    STARTSTOP_TM_MATCH_PEAK_MIN_DIST_MS,
    STARTSTOP_TM_MATCH_PEAK_PROMINENCE,
    STARTSTOP_TM_MATCH_TIGHT_WINDOW,
    STARTSTOP_TM_MATCH_WINDOW_PAD_MS,
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
    ART_PEAK_WIDTH_MS,
    ART_PEAK_HEIGHT,
)
from .detection import (
    add_extra_peak_to_p1,
    detect_onset_near_template,
    detect_peak_in_window,
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
    plot_template_overlay_panel,
    plot_spontaneous_overview,
    plot_envelopes_overlay,
    plot_burst_envelopes_by_channel,
    plot_spontaneous_boxplots,
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


def _match_sir_template(
    mean_waveform: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    baseline_mask: np.ndarray,
    resp_tmin: float,
    resp_tmax: float,
    template_bank: list[dict[str, object]],
    scales: tuple[float, ...],
    min_corr: float,
) -> dict[str, object] | None:
    """Match pre-computed templates to a channel's mean stimulus-evoked response.

    For each template variant (template x scale x flip), tries candidate shifts
    that place P1 within [resp_tmin, resp_tmax] and computes Pearson correlation
    in the response window.  Returns the best match with absolute marker times
    and a synthesized template waveform on the epoch time axis, or None if no
    match exceeds *min_corr*.
    """
    bmean = float(np.mean(mean_waveform[baseline_mask]))
    sig = mean_waveform - bmean

    resp_mask = (times >= resp_tmin) & (times <= resp_tmax)
    resp_times = times[resp_mask]
    sig_resp = sig[resp_mask]
    sig_resp_c = sig_resp - float(np.mean(sig_resp))
    if float(np.std(sig_resp_c)) == 0 or len(sig_resp) < 5:
        return None

    best_corr = -np.inf
    best: dict[str, object] | None = None

    for tpl in template_bank:
        wave_raw = np.asarray(tpl["wave"], dtype=float)
        tpl_times = np.asarray(tpl["times"], dtype=float)
        markers = tpl["markers"]
        p1_marker = float(markers.get("peak1", np.nan))
        if np.isnan(p1_marker):
            continue
        onset_marker = float(markers.get("onset", np.nan))
        p2_marker = float(markers.get("peak2", np.nan))

        for scale in scales:
            p1_scaled = p1_marker * scale
            onset_scaled = onset_marker * scale if np.isfinite(onset_marker) else np.nan
            p2_scaled = p2_marker * scale if np.isfinite(p2_marker) else np.nan

            # Anchor range: shifts where P1 lands in [resp_tmin, resp_tmax].
            anchor_tmin = resp_tmin - p1_scaled
            anchor_tmax = resp_tmax - p1_scaled
            n_shifts = max(3, int((anchor_tmax - anchor_tmin) * sfreq) + 1)
            anchor_candidates = np.linspace(anchor_tmin, anchor_tmax, n_shifts)

            for flip in (1, -1):
                for anchor_t in anchor_candidates:
                    # Template times in epoch coordinates.
                    shifted_tpl_times = tpl_times * scale + anchor_t

                    # Interpolate template at response-window time points.
                    tpl_at_resp = np.interp(
                        resp_times, shifted_tpl_times, float(flip) * wave_raw,
                        left=0.0, right=0.0,
                    )
                    tpl_at_resp_c = tpl_at_resp - float(np.mean(tpl_at_resp))
                    if float(np.std(tpl_at_resp_c)) == 0:
                        continue

                    corr = float(np.corrcoef(sig_resp_c, tpl_at_resp_c)[0, 1])
                    if np.isnan(corr) or corr <= best_corr:
                        continue

                    p1_abs = anchor_t + p1_scaled
                    p2_abs = anchor_t + p2_scaled if np.isfinite(p2_scaled) else np.nan
                    onset_abs = anchor_t + onset_scaled if np.isfinite(onset_scaled) else np.nan

                    # Sanity: onset must be within the response window, P2 must follow P1.
                    if np.isfinite(onset_abs) and onset_abs < resp_tmin:
                        onset_abs = np.nan
                    if np.isfinite(p2_abs) and p2_abs <= p1_abs:
                        continue

                    best_corr = corr

                    # Synthesize the full template on the epoch time axis.
                    synth_on_epoch = np.interp(
                        times, shifted_tpl_times, float(flip) * wave_raw,
                        left=0.0, right=0.0,
                    )

                    best = {
                        "name": tpl["name"],
                        "scale": scale,
                        "flip": flip,
                        "corr": best_corr,
                        "template": synth_on_epoch,
                        "onset": onset_abs,
                        "p1": p1_abs,
                        "p2": p2_abs,
                    }

    if best is None or best["corr"] < min_corr:
        return None

    # Scale the synthesized template to the signal's amplitude so that
    # downstream peak detectors can compare template values against the
    # real signal (both in Volts).
    tmpl_resp = best["template"][resp_mask]
    tmpl_resp_c = tmpl_resp - float(np.mean(tmpl_resp))
    denom = float(np.dot(tmpl_resp_c, tmpl_resp_c))
    if denom > 0:
        gain = float(np.dot(sig_resp_c, tmpl_resp_c)) / denom
        best["template"] = best["template"] * gain

    return best


# MATLAB Level-4 element storage-type (P digit) -> numpy dtype / byte size.
_MAT4_DTYPE = {0: "<f8", 1: "<f4", 2: "<i4", 3: "<i2", 4: "<u2", 5: "<u1"}
_MAT4_SIZE = {0: 8, 1: 4, 2: 4, 3: 2, 4: 2, 5: 1}
_MAT4_VARS = (
    "data", "datastart", "titles", "rangemin", "rangemax", "unittext",
    "unittextmap", "blocktimes", "tickrate", "samplerate",
    "firstsampleoffset", "comtext", "com",
)


def _read_labchart_mat4(mat_path) -> mne.io.Raw:
    """Fallback reader for malformed/headerless LabChart 'MATLAB Level 4' exports.

    These files lack the 128-byte MAT-file header and `dataend`, so scipy (and
    pymatreader) only recover `data`/`datastart` and mangle the rest. We locate
    each LabChart variable by its name token, read its v4 header (5 int32:
    type, rows, cols, imagf, namelen) and data directly, reconstruct the channel
    matrix from `data`+`datastart`, and rebuild start/stop annotations from
    `com`+`comtext`. Channel names are not reliably recoverable from this format,
    so generic ch1..chN are used (channel order is preserved).
    """
    raw_bytes = Path(mat_path).read_bytes()
    hdr: dict[str, dict] = {}
    for nm in _MAT4_VARS:
        i = raw_bytes.find(nm.encode() + b"\x00")
        if i < 20:
            continue
        typ, mr, nc, imag, nl = (
            int(x) for x in np.frombuffer(raw_bytes, "<i4", count=5, offset=i - 20)
        )
        hdr[nm] = {"typ": typ, "mr": mr, "nc": nc, "nl": nl, "doff": i + nl}
    hdr_starts = sorted(h["doff"] - 20 - h["nl"] for h in hdr.values())

    def _read(nm):
        h = hdr.get(nm)
        if h is None:
            return None
        P = (h["typ"] % 100) // 10
        dt = _MAT4_DTYPE.get(P, "<f8")
        sz = _MAT4_SIZE.get(P, 8)
        count = h["mr"] * h["nc"]
        # Clamp so an over-stated dim cannot read into the next variable.
        nxt = [o for o in hdr_starts if o >= h["doff"]]
        avail = ((min(nxt) if nxt else len(raw_bytes)) - h["doff"]) // sz
        count = min(count, max(0, avail))
        return np.frombuffer(raw_bytes, dt, count=count, offset=h["doff"]).astype(float), h["mr"], h["nc"]

    if "data" not in hdr or "datastart" not in hdr:
        raise ValueError(f"Cannot recover LabChart MAT structure from {mat_path}")

    data = _read("data")[0]
    ds = _read("datastart")[0]
    n_total = data.size
    starts = sorted({int(round(x)) for x in ds if np.isfinite(x) and 1 <= x <= n_total})
    ends = starts[1:] + [n_total + 1]  # single block, channels contiguous
    chans = [data[s - 1:e - 1] for s, e in zip(starts, ends)]
    min_len = min(len(c) for c in chans)
    chans = [c[:min_len] for c in chans]
    data_2d = np.vstack(chans)

    sfreq = 4000.0
    tick = _read("tickrate")
    if tick is not None and tick[0].size:
        sfreq = float(tick[0][0])
    elif _read("samplerate") is not None:
        sr = _read("samplerate")[0]
        sr = sr[np.isfinite(sr) & (sr > 0)]
        if sr.size:
            sfreq = float(np.max(sr))

    if MAT_DIVIDE_BY_1000:
        data_2d = data_2d / 1000.0

    ch_names = [f"ch{i + 1}" for i in range(data_2d.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    com = _read("com")
    comtext = _read("comtext")
    if com is not None and comtext is not None:
        com_v, cmr, cnc = com
        ct_v, tmr, tnc = comtext
        com_m = com_v.reshape((cnc, cmr)).T if com_v.size == cmr * cnc else None
        ct_m = ct_v.reshape((tnc, tmr)).T if ct_v.size == tmr * tnc else None
        if com_m is not None and ct_m is not None:
            labels = [
                _normalize_text("".join(chr(int(c)) for c in ct_m[r] if 0 < int(c) < 65536)).strip()
                for r in range(ct_m.shape[0])
            ]
            onsets, descs = [], []
            for r in range(com_m.shape[0]):
                sample = com_m[r, 2]
                if not np.isfinite(sample):
                    continue
                cid = int(com_m[r, 4]) if np.isfinite(com_m[r, 4]) else 0
                onsets.append(float(sample) / sfreq)
                descs.append(labels[cid - 1] if 1 <= cid <= len(labels) else str(cid))
            if onsets:
                raw.set_annotations(mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descs))
    return raw


def _load_raw_from_mat(mat_path: Path) -> mne.io.Raw:
    mat = loadmat(mat_path)
    # Malformed/headerless LabChart "MATLAB Level 4" exports lose everything but
    # data/datastart through scipy (no 'dataend'); handle them separately.
    if "dataend" not in mat:
        return _read_labchart_mat4(mat_path)
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
    allow_empty: bool = False,
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
    if not art_chans and not allow_empty:
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
    tight_window: bool = False,
    pad_ms: float = 15.0,
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
            # Optionally crop to the active response window so the flat template
            # tails do not correlate against neighbouring responses in trains.
            if tight_window:
                lo_m = markers.get("onset", np.nan)
                hi_m = markers.get("peak2", np.nan)
                if not np.isfinite(hi_m):
                    hi_m = markers.get("peak1", np.nan)
                if np.isfinite(lo_m) and np.isfinite(hi_m):
                    pad = float(pad_ms) / 1000.0
                    lo = lo_m * float(scale) - pad
                    hi = hi_m * float(scale) + pad
                    crop_mask = (t_var >= lo) & (t_var <= hi)
                    if int(np.count_nonzero(crop_mask)) >= 5:
                        t_var = t_var[crop_mask]
                        base_wave = base_wave[crop_mask]
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


def _rms_amp_windows(x: np.ndarray, sfreq: float, window_ms: float):
    """Non-overlapping windows: per-window RMS, mean |amplitude|, centre time."""
    w = max(1, int(round(window_ms / 1000.0 * sfreq)))
    n = x.size // w
    centers, rms, amp = [], [], []
    for i in range(n):
        seg = x[i * w:(i + 1) * w]
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
        amp.append(float(np.mean(np.abs(seg))))
        centers.append((i * w + w / 2.0) / sfreq)
    return np.asarray(centers), np.asarray(rms), np.asarray(amp)


def _moving_rms_envelope(x: np.ndarray, sfreq: float, window_ms: float, hop_ms: float):
    """Smooth RMS envelope: sliding RMS (window_ms) sampled every hop_ms."""
    w = max(1, int(round(window_ms / 1000.0 * sfreq)))
    hop = max(1, int(round(hop_ms / 1000.0 * sfreq)))
    half = w // 2
    centers, env = [], []
    for c in range(0, x.size, hop):
        a = max(0, c - half)
        b = min(x.size, c + half)
        seg = x[a:b]
        if seg.size == 0:
            continue
        env.append(float(np.sqrt(np.mean(seg ** 2))))
        centers.append(c / sfreq)
    return np.asarray(centers), np.asarray(env)


def _detect_bursts_on_envelope(env_times: np.ndarray, env_uv: np.ndarray):
    """Detect activity bursts on an RMS envelope (µV).

    Returns a list of (t_start, t_end) seconds. Empty when the channel has no
    clear bursts (envelope peak not sufficiently above baseline).
    """
    if env_uv.size < 3:
        return []
    baseline = float(np.percentile(env_uv, SPONTANEOUS_EMG_BURST_BASELINE_PCT))
    peak = float(np.percentile(env_uv, SPONTANEOUS_EMG_BURST_PEAK_PCT))
    base_floor = max(baseline, 1e-9)
    if peak < SPONTANEOUS_EMG_BURST_MIN_PEAK_UV:
        return []  # below absolute amplitude floor — treat as inactive channel
    if peak < SPONTANEOUS_EMG_BURST_MIN_SNR * base_floor:
        return []  # no clear excursion above baseline — no bursts
    thr = baseline + SPONTANEOUS_EMG_BURST_THRESH_FRAC * (peak - baseline)

    above = env_uv > thr
    if not np.any(above):
        return []
    # contiguous above-threshold runs (in envelope-sample units)
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [above.size]
    runs = list(zip(starts, ends))  # [start_idx, end_idx)

    # merge runs separated by a gap shorter than merge_ms
    merge_s = SPONTANEOUS_EMG_BURST_MERGE_MS / 1000.0
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (env_times[s] - env_times[merged[-1][1] - 1]) <= merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_s = SPONTANEOUS_EMG_BURST_MIN_MS / 1000.0
    keep_level = baseline + SPONTANEOUS_EMG_BURST_KEEP_FRAC * (peak - baseline)
    bursts = []
    for s, e in merged:
        t0 = float(env_times[s])
        t1 = float(env_times[min(e, env_times.size) - 1])
        if (t1 - t0) < min_s:
            continue
        seg = env_uv[s:min(e, env_uv.size)]
        # Hysteresis: keep only bursts whose peak clears the higher keep level.
        if seg.size and float(np.max(seg)) >= keep_level:
            bursts.append((t0, t1))
    return bursts


def _run_spontaneous_emg_analysis(
    start_raw: mne.io.BaseRaw,
    ch_names: list[str],
    out_dir: Path,
    condition: str,
    window_ms: float = SPONTANEOUS_EMG_WINDOW_MS,
    hop_ms: float = SPONTANEOUS_EMG_ENV_HOP_MS,
    uv: float = SPONTANEOUS_EMG_UV_SCALE,
) -> None:
    """Doctor-requested spontaneous-EMG metrics for one start-stop segment.

    For every channel: windowed RMS and mean amplitude (in µV) -> Excel
    (Summary + Detailed sheets, per channel), a smooth RMS envelope -> per-channel
    .txt, a raw+envelope overview plot, and an all-channel envelope overlay plot.
    Independent of the single-peak detection.
    """
    if not ch_names:
        return
    sfreq = float(start_raw.info["sfreq"])
    data = start_raw.get_data(picks=ch_names)  # Volts
    # Remove low-frequency baseline drift before RMS/envelope/burst computation
    # (detection preprocessing is left untouched — see SPONTANEOUS_EMG_HP_FREQ).
    if SPONTANEOUS_EMG_HP_FREQ is not None:
        data = mne.filter.filter_data(
            np.ascontiguousarray(data, dtype=float), sfreq,
            l_freq=float(SPONTANEOUS_EMG_HP_FREQ), h_freq=None,
            method="fir", phase="zero", verbose=False,
        )
    times = np.arange(data.shape[1]) / sfreq

    plots_dir = out_dir / "Plots"
    env_dir = out_dir / "Envelopes"
    excel_dir = out_dir / "Excel"
    for d in (plots_dir, env_dir, excel_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Clear stale per-burst/per-channel outputs from previous runs so the folder
    # only reflects the current detection.
    for old in list(env_dir.glob("*.txt")) + list(plots_dir.glob("*.png")):
        old.unlink()

    sig_by_ch: dict[str, np.ndarray] = {}
    env_by_ch: dict[str, np.ndarray] = {}
    env_times = None
    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    bursts_by_ch: dict[str, list[tuple[float, float]]] = {}
    burst_env_by_ch: dict[str, list[dict]] = defaultdict(list)  # per-channel burst envelopes
    burst_rows: list[dict[str, object]] = []

    safe_condition = re.sub(r"[^\w\-\+\. ]", "_", str(condition))

    for ci, ch in enumerate(ch_names):
        x = np.asarray(data[ci], dtype=float)
        sig_by_ch[ch] = x * uv

        centers, rms, amp = _rms_amp_windows(x, sfreq, window_ms)
        rms_uv, amp_uv = rms * uv, amp * uv

        et, env = _moving_rms_envelope(x, sfreq, window_ms, hop_ms)
        env_uv = env * uv
        env_by_ch[ch] = env_uv
        env_times = et
        safe_ch = re.sub(r"[^\w\-_\. ]", "_", str(ch))

        # Burst detection on the envelope. Only channels with bursts get per-burst
        # envelope .txt files and contribute to the burst overlay.
        if SPONTANEOUS_EMG_BURST_DETECTION:
            bursts = _detect_bursts_on_envelope(et, env_uv)
            if bursts:
                bursts_by_ch[ch] = bursts
                for bk, (t0, t1) in enumerate(bursts, start=1):
                    dur = t1 - t0
                    pad = dur / 2.0          # half-burst baseline on each side
                    center = (t0 + t1) / 2.0
                    # Metrics over the burst itself.
                    mb = (et >= t0) & (et <= t1)
                    seg_burst = env_uv[mb]
                    # Plot/txt window: burst + half-length padding, centred so the
                    # burst middle sits at x = 0 (baseline visible symmetrically).
                    mp = (et >= t0 - pad) & (et <= t1 + pad)
                    t_rel = et[mp] - center
                    seg = env_uv[mp]
                    np.savetxt(
                        env_dir / f"{safe_ch}_burst{bk}_rms_envelope.txt",
                        np.column_stack([t_rel, seg]),
                        header="time_from_burst_center_s\trms_uV", delimiter="\t", comments="",
                    )
                    burst_env_by_ch[ch].append({"t_rel": t_rel, "env": seg})
                    burst_rows.append({
                        "Condition": str(condition), "Channel": str(ch), "Burst": bk,
                        "Start_s": float(t0), "End_s": float(t1),
                        "Duration_s": float(dur),
                        "RMS_mean_uV": float(np.mean(seg_burst)) if seg_burst.size else np.nan,
                        "RMS_peak_uV": float(np.max(seg_burst)) if seg_burst.size else np.nan,
                    })

        for wi, (tc, r, a) in enumerate(zip(centers, rms_uv, amp_uv)):
            detailed_rows.append({
                "Condition": str(condition), "Channel": str(ch),
                "Window": wi, "Window_center_s": float(tc),
                "RMS_uV": float(r), "MeanAmp_uV": float(a),
            })

        n = int(rms_uv.size)
        sem_rms = float(np.std(rms_uv, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        sem_amp = float(np.std(amp_uv, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        summary_rows.append({
            "Condition": str(condition), "Channel": str(ch), "n_windows": n,
            "n_bursts": len(bursts_by_ch.get(ch, [])),
            "RMS_mean_uV": float(np.mean(rms_uv)) if n else np.nan,
            "RMS_std_uV": float(np.std(rms_uv, ddof=1)) if n > 1 else np.nan,
            "RMS_sem_uV": sem_rms,
            "Amp_mean_uV": float(np.mean(amp_uv)) if n else np.nan,
            "Amp_std_uV": float(np.std(amp_uv, ddof=1)) if n > 1 else np.nan,
            "Amp_sem_uV": sem_amp,
        })


    # Fallback export: for channels with NO bursts that are significantly more
    # active than the rest (robust MAD threshold on mean RMS), still save the
    # whole-segment RMS envelope to .txt so the active channels are captured.
    if SPONTANEOUS_EMG_ACTIVE_MAD_K is not None and env_times is not None and len(summary_rows) >= 3:
        rms_means = {r["Channel"]: r["RMS_mean_uV"] for r in summary_rows}
        vals = np.array([v for v in rms_means.values() if np.isfinite(v)], dtype=float)
        if vals.size >= 3:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            thr = med + SPONTANEOUS_EMG_ACTIVE_MAD_K * 1.4826 * mad if mad > 0 else med
            for ch in ch_names:
                if ch in bursts_by_ch:
                    continue  # bursts already exported per burst
                x = rms_means.get(ch, np.nan)
                if not (np.isfinite(x) and x > thr and x > med):
                    continue
                safe_ch = re.sub(r"[^\w\-_\. ]", "_", str(ch))
                np.savetxt(
                    env_dir / f"{safe_ch}_fullenvelope_rms.txt",
                    np.column_stack([env_times, env_by_ch[ch]]),
                    header="time_s\trms_uV", delimiter="\t", comments="",
                )


    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    bursts_df = pd.DataFrame(burst_rows)
    xlsx_path = excel_dir / f"Spontaneous_EMG_{safe_condition}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        detailed_df.to_excel(writer, sheet_name="Detailed", index=False)
        if not bursts_df.empty:
            bursts_df.to_excel(writer, sheet_name="Bursts", index=False)
    summary_df.to_csv(excel_dir / f"Spontaneous_EMG_summary_{safe_condition}.csv", index=False)
    detailed_df.to_csv(excel_dir / f"Spontaneous_EMG_detailed_{safe_condition}.csv", index=False)
    if not bursts_df.empty:
        bursts_df.to_csv(excel_dir / f"Spontaneous_EMG_bursts_{safe_condition}.csv", index=False)

    plot_spontaneous_overview(
        times=times, ch_names=ch_names, sig_by_ch=sig_by_ch,
        env_times=env_times if env_times is not None else np.array([]),
        env_by_ch=env_by_ch,
        out_path=plots_dir / "overview_raw_with_envelope.png",
        title=f"Spontaneous EMG — raw + RMS envelope + bursts | {condition}",
        bursts_by_ch=bursts_by_ch,
    )
    plot_envelopes_overlay(
        env_times=env_times if env_times is not None else np.array([]),
        env_by_ch=env_by_ch,
        out_path=plots_dir / "envelopes_overlay_all_channels.png",
        title=f"Spontaneous EMG — full RMS envelopes (all channels) | {condition}",
    )
    plot_burst_envelopes_by_channel(
        bursts_by_ch=dict(burst_env_by_ch),
        out_path=plots_dir / "burst_envelopes_by_channel.png",
        title=f"Spontaneous EMG — burst RMS envelopes per channel (centred, ±½-burst baseline) | {condition}",
    )
    plot_spontaneous_boxplots(
        detailed_df=detailed_df,
        out_path=plots_dir / "boxplots_rms_amplitude.png",
        title=f"Spontaneous EMG — per-window RMS / amplitude by channel | {condition}",
    )


def _run_startstop_analysis(
    raw: mne.io.BaseRaw,
    startstop_dir: Path,
    art_chans: list[str],
    default_condition: str | None = None,
) -> None:
    print("[STARTSTOP] Extracting start/stop segments...", flush=True)
    segments = extract_start_stop_segments(raw, default_condition=default_condition)
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
    channel_qc_rows: list[dict[str, object]] = []
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

        # Doctor-requested spontaneous-EMG metrics (RMS / mean amplitude
        # envelopes), independent of the single-peak detection below.
        if SPONTANEOUS_EMG_ANALYSIS:
            _run_spontaneous_emg_analysis(
                start_raw=start_raw,
                ch_names=ch_names,
                out_dir=startstop_dir / "Spontaneous EMG" / safe_condition,
                condition=str(condition),
            )

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
            tight_window=bool(STARTSTOP_TM_MATCH_TIGHT_WINDOW),
            pad_ms=float(STARTSTOP_TM_MATCH_WINDOW_PAD_MS),
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
        # Drop anchors too close to the segment edges so mne.Epochs does not
        # silently discard epochs — otherwise epoch position no longer matches
        # peak_matches_sorted / event_samples and every downstream per-epoch
        # channel/template assignment is shifted.
        n_seg = int(start_raw.n_times)
        lo_guard = int(np.ceil(abs(STARTSTOP_EPOCH_TMIN) * sfreq)) + 1
        hi_guard = n_seg - int(np.ceil(abs(STARTSTOP_EPOCH_TMAX) * sfreq)) - 1
        peak_matches_sorted = [
            m for m in peak_matches_sorted if lo_guard <= int(m["anchor_sample"]) <= hi_guard
        ]
        if not peak_matches_sorted:
            continue
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
            valid_items: list[tuple[dict, np.ndarray]] = []
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

                if is_valid and np.std(seg) > 0:
                    valid_items.append((e, seg - np.mean(seg)))

            valid_fraction = np.mean(valid_flags) if len(valid_flags) else 0.0
            median_corr = float(np.median(corrs)) if len(corrs) else 0.0

            # Inter-trial self-consistency: median pairwise correlation between
            # the channel's own valid epochs. Real responses repeat the same
            # shape; channels firing on irregular EMG bursts are inconsistent.
            intertrial_corr = np.nan
            corr_mat = None
            if len(valid_items) >= 3:
                corr_mat = np.corrcoef(np.vstack([s for _e, s in valid_items]))
                iu = np.triu_indices(len(valid_items), 1)
                pair_vals = corr_mat[iu]
                pair_vals = pair_vals[np.isfinite(pair_vals)]
                if pair_vals.size:
                    intertrial_corr = float(np.median(pair_vals))

            # Keep the original conservative gate, and add a stricter low-corr
            # guard for borderline channels: moderate valid fraction can still
            # be dominated by mismatched shape when template similarity is very low.
            low_valid_low_corr = (valid_fraction < min_valid_frac) and (median_corr < corr_min_median)
            borderline_valid_very_low_corr = (valid_fraction < 0.6) and (median_corr < 0.2)
            reject_intertrial = (
                STARTSTOP_CHANNEL_MIN_INTERTRIAL_CORR is not None
                and len(valid_items) >= 3
                and np.isfinite(intertrial_corr)
                and intertrial_corr < float(STARTSTOP_CHANNEL_MIN_INTERTRIAL_CORR)
            )
            wiped = bool(low_valid_low_corr or borderline_valid_very_low_corr or reject_intertrial)
            n_epoch_dropped = 0
            if wiped:
                for e in entries:
                    e["onset"] = np.nan
                    e["p1"] = np.nan
                    e["p2"] = np.nan
                    e["pv1"] = np.nan
                    e["pv2"] = np.nan
                    e["ptp"] = np.nan
            elif (
                STARTSTOP_EPOCH_MIN_INTERTRIAL_CORR is not None
                and corr_mat is not None
                and len(valid_items) >= 3
            ):
                # Per-epoch consistency: drop individual detections that do not
                # correlate with the consensus of the others, so only the
                # mutually-consistent pattern survives in every output.
                thr_ep = float(STARTSTOP_EPOCH_MIN_INTERTRIAL_CORR)
                n_items = len(valid_items)
                for i, (e, _seg) in enumerate(valid_items):
                    others = corr_mat[i, np.arange(n_items) != i]
                    others = others[np.isfinite(others)]
                    ep_corr = float(np.median(others)) if others.size else 0.0
                    if ep_corr < thr_ep:
                        e["onset"] = np.nan
                        e["p1"] = np.nan
                        e["p2"] = np.nan
                        e["pv1"] = np.nan
                        e["pv2"] = np.nan
                        e["ptp"] = np.nan
                        n_epoch_dropped += 1
            channel_qc_rows.append(
                {
                    "Condition": str(condition),
                    "Channel": str(ch),
                    "n_epochs": len(entries),
                    "n_valid": int(np.sum(valid_flags)) if len(valid_flags) else 0,
                    "valid_fraction": float(valid_fraction),
                    "median_template_corr": float(median_corr),
                    "intertrial_corr": float(intertrial_corr),
                    "n_intertrial_segs": len(valid_items),
                    "n_epoch_consistency_dropped": int(n_epoch_dropped),
                    "wiped": wiped,
                    "wipe_reason": (
                        "intertrial" if reject_intertrial
                        else "low_valid_low_corr" if low_valid_low_corr
                        else "borderline_low_corr" if borderline_valid_very_low_corr
                        else ""
                    ),
                }
            )

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

        # Save the analyzed segment annotated with one marker per detected peak.
        # Peaks on different channels within the merge window collapse into one
        # marker whose description lists every channel they were found on.
        if STARTSTOP_SAVE_DETECTION_MARKERS and detected_channels_by_epoch:
            merge_tol = float(STARTSTOP_DETECTION_MARKER_MERGE_MS) / 1000.0
            dets: list[tuple[float, str]] = []
            for ep_idx, ch_list in detected_channels_by_epoch.items():
                ev_t = float(event_samples[ep_idx]) / sfreq
                for ch in ch_list:
                    p1 = p1_by_epoch_channel.get(ep_idx, {}).get(ch, np.nan)
                    peak_t = ev_t + (float(p1) if np.isfinite(p1) else 0.0)
                    dets.append((peak_t, ch))
            dets.sort(key=lambda x: x[0])

            onsets: list[float] = []
            descs: list[str] = []
            cur_times: list[float] = []
            cur_chs: list[str] = []
            for t, ch in dets:
                if cur_times and (t - cur_times[-1]) > merge_tol:
                    onsets.append(float(np.mean(cur_times)))
                    descs.append("+".join(sorted(set(cur_chs))))
                    cur_times, cur_chs = [], []
                cur_times.append(t)
                cur_chs.append(ch)
            if cur_times:
                onsets.append(float(np.mean(cur_times)))
                descs.append("+".join(sorted(set(cur_chs))))

            if onsets:
                det_raw = start_raw.copy()
                det_raw.set_annotations(
                    mne.Annotations(
                        onset=onsets,
                        duration=[0.0] * len(onsets),
                        description=descs,
                    )
                )
                det_dir = startstop_dir / "Detections raw"
                det_dir.mkdir(parents=True, exist_ok=True)
                det_raw.save(
                    det_dir / f"{safe_condition}_detections_raw.fif", overwrite=True
                )

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
            # epochs_wide uses a wider window and may drop more boundary events
            # than the narrow epochs, so map narrow epoch index -> wide position
            # via its selection instead of indexing wide_data by ep_idx directly.
            wide_pos_by_orig = {int(o): i for i, o in enumerate(epochs_wide.selection)}

            for ep_idx, ch_list in detected_channels_by_epoch.items():
                wpos = wide_pos_by_orig.get(int(ep_idx))
                if wpos is None:
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
                    val = float(np.nanmax(np.abs(wide_data[wpos, ch_idx, win_mask])))
                    if max_abs is None or val > max_abs:
                        max_abs = val
                if max_abs is None:
                    max_abs = float(np.nanmax(np.abs(wide_data[wpos]))) if wide_data.size else 0.0
                scale = (max_abs * 2.5) if max_abs > 0 else None
                safe_chs = [re.sub(r"[^\w\-_\. ]", "_", str(ch)) for ch in ch_list]
                ch_suffix = "+".join(safe_chs)
                if len(ch_suffix) > 80:
                    ch_suffix = f"{ch_suffix[:77]}..."
                out_path = raw_epochs_dir / f"epoch_{ep_idx:03d}__{ch_suffix}.png"
                fig = epochs_wide[wpos].plot(
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
    if channel_qc_rows:
        pd.DataFrame(channel_qc_rows).to_csv(
            excel_dir / "STARTSTOP_channel_qc.csv", index=False
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
    startstop_mode: bool | None = None,
    art_peak_height: tuple[float, float] | None = ART_PEAK_HEIGHT,
    art_peak_width_ms: tuple[float, float] | None = ART_PEAK_WIDTH_MS,
    art_min_distance_ms: float = 100.0,
    epoch_tmin: float = EPOCH_TMIN,
    epoch_tmax: float = EPOCH_TMAX,
    baseline_tmin: float = BASELINE_TMIN,
    baseline_tmax: float = BASELINE_TMAX,
    resp_tmin: float = RESP_TMIN,
    resp_tmax: float = RESP_TMAX,
    raw_notch_freq: float | None = RAW_NOTCH_FREQ,
    raw_bandpass_l_freq: float | None = RAW_BANDPASS_L_FREQ,
    raw_bandpass_h_freq: float | None = RAW_BANDPASS_H_FREQ,
) -> Path:
    edf_path = Path(edf_path)
    if output_dir is None:
        output_root = _default_output_root_for_input(edf_path)
    else:
        output_root = Path(output_dir)

    use_startstop = startstop_mode if startstop_mode is not None else STARTSTOP_MODE
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

    suffix = edf_path.suffix.lower()
    if suffix == ".mat":
        raw = _load_raw_from_mat(edf_path)
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(edf_path, preload=True)
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=True)

    original_fif_path = paths["data_dir"] / f"{edf_path.stem}_original_raw.fif"
    raw.save(original_fif_path, overwrite=True)
    # StartStop detection does not need a stimulus/artifact channel (template
    # matching runs directly on the EMG). Allow art-less recordings there; SIR
    # still requires one for stimulus-onset detection.
    default_art_chans = _resolve_art_channels(raw, ARTCHAN, allow_empty=use_startstop)
    default_art_set = set(default_art_chans)
    emg_picks = [ch for ch in raw.ch_names if ch not in default_art_set]
    if emg_picks:
        # Notch is applied independently of the band-pass: the full power-line
        # harmonic comb (base .. Nyquist) is removed whenever raw_notch_freq is
        # set, even when the band-pass is disabled.
        if raw_notch_freq is not None:
            nyq = raw.info["sfreq"] / 2.0
            notch_freqs = list(np.arange(float(raw_notch_freq), nyq, float(raw_notch_freq)))
            if notch_freqs:
                raw.notch_filter(freqs=notch_freqs, picks=emg_picks, method="fir", phase="zero")
        if raw_bandpass_l_freq is not None or raw_bandpass_h_freq is not None:
            raw.filter(
                l_freq=raw_bandpass_l_freq, h_freq=raw_bandpass_h_freq,
                picks=emg_picks, method="fir", phase="zero",
            )
    if ARTIFACT_REREF and default_art_chans:
        _apply_artifact_reref(raw, default_art_chans)
    if CAR_REREF and default_art_chans:
        _apply_car_reref(raw, default_art_chans)

    preproc_path = paths["data_dir"] / f"{edf_path.stem}_preprocessed_raw.fif"
    raw.save(preproc_path, overwrite=True)

    if use_startstop:
        _run_startstop_analysis(raw, startstop_dir, default_art_chans, default_condition=edf_path.stem)
        if old_mne_log_level is not None:
            mne.set_log_level(old_mne_log_level)
        return output_root

    # ── SIR: load pre-computed template bank ──
    template_dir = _resolve_startstop_template_dir()
    template_bank = _load_startstop_template_bank(
        template_dir=template_dir,
        template_native_sfreq=float(STARTSTOP_TM_TEMPLATE_SFREQ),
        template_center_sample=int(STARTSTOP_TM_TEMPLATE_CENTER_SAMPLE),
    )
    if not template_bank:
        print("[SIR] No templates found in bank; cannot proceed.", flush=True)
        if old_mne_log_level is not None:
            mne.set_log_level(old_mne_log_level)
        return output_root

    print("[SIR] Creating annotation crops...", flush=True)
    create_annotation_crops(raw, crops_dir, overwrite=True)
    file_list = list_crop_files(crops_dir)

    # ── PASS 1: collect per-config channel means across all amplitudes ──
    # Each entry: (amp_num, amp_str, trial_mean). Keep amp_str so per-amp
    # templates can be looked up by file_name parsing in PASS 2.
    config_waveforms: dict[str, dict[str, list[tuple[float, str, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    config_meta: dict[str, dict[str, np.ndarray | float]] = {}

    print("[SIR] Collecting channel means...", flush=True)
    for file_name in file_list:
        file_path = crops_dir / file_name
        raw_file = mne.io.read_raw_fif(file_path, preload=False)
        sfreq = raw_file.info["sfreq"]

        art_chans = _resolve_art_channels(raw_file, ARTCHAN, fallback_chans=default_art_chans)
        art_set = set(art_chans)
        art = _get_art_signal(raw_file, art_chans)
        min_dist_samples = sfreq * art_min_distance_ms / 1000.0
        if art_peak_height is not None:
            fp_kwargs = {"height": art_peak_height, "distance": min_dist_samples}
            if art_peak_width_ms is not None:
                fp_kwargs["width"] = (
                    art_peak_width_ms[0] / 1000.0 * sfreq,
                    art_peak_width_ms[1] / 1000.0 * sfreq,
                )
            peaks, _ = find_peaks(-art, **fp_kwargs)
        else:
            threshold = THRESH * np.std(art)
            peaks, _ = find_peaks(-art, height=threshold, distance=min_dist_samples)
        if len(peaks) <= MIN_VALID_EPOCHS:
            continue

        events = np.column_stack(
            [peaks + raw_file.first_samp, np.zeros_like(peaks, dtype=int), np.ones_like(peaks, dtype=int)]
        )
        epochs = mne.Epochs(
            raw_file, events, event_id=1,
            tmin=epoch_tmin, tmax=epoch_tmax,
            baseline=None, preload=True,
        )

        times = epochs.times
        configuration = file_name.split("-")[0]

        if configuration not in config_meta:
            config_meta[configuration] = {"times": times.copy(), "sfreq": sfreq}
        else:
            if sfreq != config_meta[configuration]["sfreq"] or len(times) != len(config_meta[configuration]["times"]):
                continue

        data_epoched = epochs.get_data()
        stim_amp_raw = file_name.split("_", 1)[1].split(".fif")[0]
        amp_num = amp_to_number(stim_amp_raw)
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            if ch_name in art_set:
                continue
            config_waveforms[configuration][ch_name].append(
                (amp_num, stim_amp_raw, np.mean(data_epoched[:, ch_idx, :], axis=0))
            )

    # ── Match each config/channel mean to the best template from the bank ──
    # config_templates / config_template_markers: per (config, channel) — fit
    # on the max-amplitude trial-mean. Used as the canonical config-level
    # template for overlays and as fallback in PASS 2.
    config_templates: dict[str, dict[str, np.ndarray | None]] = defaultdict(dict)
    config_template_markers: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    # amp_templates / amp_template_markers: per (config, amp_str, channel),
    # only populated when SIR_TEMPLATE_PER_AMPLITUDE is True.
    amp_templates: dict[tuple[str, str], dict[str, np.ndarray | None]] = defaultdict(dict)
    amp_template_markers: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)

    print("[SIR] Matching templates...", flush=True)
    for configuration, ch_dict in config_waveforms.items():
        times = config_meta[configuration]["times"]
        sfreq = float(config_meta[configuration]["sfreq"])
        baseline_mask = (times >= baseline_tmin) & (times <= baseline_tmax)

        for ch_name, waves in ch_dict.items():
            if len(waves) == 0:
                config_templates[configuration][ch_name] = None
                config_template_markers[configuration][ch_name] = dict(onset=np.nan, p1=np.nan, p2=np.nan)
                continue

            # Fit on the maximum-amplitude file's trial-mean. Low-amp files
            # without responses dilute the shape and shift template markers.
            valid = [(a, s, w) for a, s, w in waves if np.isfinite(a)]
            if valid:
                _, _, tmpl_global = max(valid, key=lambda x: x[0])
            else:
                tmpl_global = np.mean(np.stack([w for _, _, w in waves], axis=0), axis=0)

            match = _match_sir_template(
                mean_waveform=tmpl_global,
                times=times,
                sfreq=sfreq,
                baseline_mask=baseline_mask,
                resp_tmin=resp_tmin,
                resp_tmax=resp_tmax,
                template_bank=template_bank,
                scales=tuple(float(s) for s in SIR_TM_SCALES),
                min_corr=float(SIR_TM_MIN_CORR),
            )

            if match is None:
                config_templates[configuration][ch_name] = None
                config_template_markers[configuration][ch_name] = dict(onset=np.nan, p1=np.nan, p2=np.nan)
            else:
                config_templates[configuration][ch_name] = match["template"]
                config_template_markers[configuration][ch_name] = dict(
                    onset=match["onset"], p1=match["p1"], p2=match["p2"],
                )

            if SIR_TEMPLATE_PER_AMPLITUDE:
                for amp_num, amp_str, wave in waves:
                    m = _match_sir_template(
                        mean_waveform=wave,
                        times=times,
                        sfreq=sfreq,
                        baseline_mask=baseline_mask,
                        resp_tmin=resp_tmin,
                        resp_tmax=resp_tmax,
                        template_bank=template_bank,
                        scales=tuple(float(s) for s in SIR_TM_SCALES),
                        min_corr=float(SIR_TM_MIN_CORR),
                    )
                    key = (configuration, amp_str)
                    if m is None:
                        amp_templates[key][ch_name] = None
                        amp_template_markers[key][ch_name] = dict(onset=np.nan, p1=np.nan, p2=np.nan)
                    else:
                        amp_templates[key][ch_name] = m["template"]
                        amp_template_markers[key][ch_name] = dict(
                            onset=m["onset"], p1=m["p1"], p2=m["p2"],
                        )

    # ── Save template figures ──
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
                times=times, template=template, markers=markers,
                out_path=out_path,
                title=f"Template: {configuration} | {ch_name}",
            )

    # ── Save template overlay panels (mean waveform + template per config) ──
    overlay_dir = paths["stim_results_dir"] / "Template overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for configuration in config_templates:
        times = config_meta[configuration]["times"]
        ch_waves = config_waveforms[configuration]
        ch_means: dict[str, np.ndarray] = {}
        for ch_name, waves in ch_waves.items():
            if not waves:
                continue
            valid = [(a, s, w) for a, s, w in waves if np.isfinite(a)]
            if valid:
                _, _, ch_means[ch_name] = max(valid, key=lambda x: x[0])
            else:
                ch_means[ch_name] = np.mean(np.stack([w for _, _, w in waves], axis=0), axis=0)

        safe_config = re.sub(r"[^\w\-\+\. ]", "_", str(configuration))
        plot_template_overlay_panel(
            times=times,
            ch_names=raw.ch_names,
            art_chans=default_art_set,
            config_waveforms=ch_means,
            config_templates=dict(config_templates[configuration]),
            config_markers=dict(config_template_markers[configuration]),
            title=f"Template overlay: {configuration}",
            out_path=overlay_dir / f"{safe_config}_overlay.png",
        )

    # ── Per-amplitude overlay panels ──
    if SIR_TEMPLATE_PER_AMPLITUDE:
        amp_overlay_dir = paths["stim_results_dir"] / "Template overlays per amplitude"
        amp_overlay_dir.mkdir(parents=True, exist_ok=True)
        for (configuration, amp_str), ch_tmpls in amp_templates.items():
            times = config_meta[configuration]["times"]
            ch_means_amp: dict[str, np.ndarray] = {}
            for ch_name, waves in config_waveforms[configuration].items():
                for a_num, a_str, w in waves:
                    if a_str == amp_str:
                        ch_means_amp[ch_name] = w
                        break
            safe_config = re.sub(r"[^\w\-\+\. ]", "_", str(configuration))
            safe_amp = re.sub(r"[^\w\-\+\.,]", "_", str(amp_str))
            plot_template_overlay_panel(
                times=times,
                ch_names=raw.ch_names,
                art_chans=default_art_set,
                config_waveforms=ch_means_amp,
                config_templates=dict(ch_tmpls),
                config_markers=dict(amp_template_markers[(configuration, amp_str)]),
                title=f"Template overlay: {configuration} amp={amp_str}",
                out_path=amp_overlay_dir / f"{safe_config}_{safe_amp}_overlay.png",
            )

    # ── PASS 2: epoch-level detection driven by template markers ──
    group_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results = {key: [] for key in [
        "Configuration", "Stim. amplitude", "Epoch", "Channel",
        "Onset latency", "Peak1 latency", "Peak2 latency",
        "Peak1 value", "Peak2 value", "PTP amplitude",
        "Inter-trial corr", "Time series",
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
        min_dist_samples = sfreq * art_min_distance_ms / 1000.0
        if art_peak_height is not None:
            fp_kwargs = {"height": art_peak_height, "distance": min_dist_samples}
            if art_peak_width_ms is not None:
                fp_kwargs["width"] = (
                    art_peak_width_ms[0] / 1000.0 * sfreq,
                    art_peak_width_ms[1] / 1000.0 * sfreq,
                )
            peaks, _ = find_peaks(-art, **fp_kwargs)
        else:
            threshold = THRESH * np.std(art)
            peaks, _ = find_peaks(-art, height=threshold, distance=min_dist_samples)

        if len(peaks) <= MIN_VALID_EPOCHS:
            continue

        events = np.column_stack(
            [peaks + raw_file.first_samp, np.zeros_like(peaks, dtype=int), np.ones_like(peaks, dtype=int)]
        )
        epochs = mne.Epochs(
            raw_file, events, event_id=1,
            tmin=epoch_tmin, tmax=epoch_tmax,
            baseline=None, preload=True,
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

        baseline_mask = (times >= baseline_tmin) & (times <= baseline_tmax)

        # Use per-amp templates when enabled, else fall back to config-level.
        if SIR_TEMPLATE_PER_AMPLITUDE and (configuration, stim_amp_raw) in amp_templates:
            tmpl_cfg = dict(amp_templates[(configuration, stim_amp_raw)])
            markers_cfg = dict(amp_template_markers[(configuration, stim_amp_raw)])
        else:
            tmpl_cfg = dict(config_templates.get(configuration, {}))
            markers_cfg = dict(config_template_markers.get(configuration, {}))

        eligible_channels = [
            ch for ch in epochs.ch_names
            if ch not in art_set
            and tmpl_cfg.get(ch) is not None
            and np.isfinite(markers_cfg.get(ch, {}).get("p1", np.nan))
        ]
        eligible_set = set(eligible_channels)

        channel_epoch_results: dict[str, list[dict]] = {ch: [] for ch in eligible_channels}
        latency_markers: dict[str, list[dict]] = {ch: [] for ch in epochs.ch_names if ch not in art_set}

        for ep in range(len(epochs)):
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                if (ch_name in art_set) or (ch_name not in eligible_set):
                    continue

                sig = data_epoched[ep, ch_idx, :]

                base = sig[baseline_mask]
                bstd = base.std()
                if bstd == 0 or np.isnan(bstd):
                    channel_epoch_results[ch_name].append(
                        {"ep": ep, "onset": np.nan, "p1": np.nan, "p2": np.nan,
                         "pv1": np.nan, "pv2": np.nan, "ptp": np.nan, "sig": sig}
                    )
                    continue

                t_on_tmpl = markers_cfg[ch_name]["onset"]
                t_p1 = markers_cfg[ch_name]["p1"]
                t_p2 = markers_cfg[ch_name]["p2"]
                tmpl = tmpl_cfg[ch_name]

                # Onset near template onset
                onset_latency = detect_onset_near_template(
                    sig, times, sfreq, baseline_mask, t_on_tmpl,
                    win_ms=2, k=1, sustain_ms=2,
                ) if np.isfinite(t_on_tmpl) else np.nan

                # Clip onset to template ± max_dev
                if np.isfinite(onset_latency) and np.isfinite(t_on_tmpl):
                    onset_latency = float(np.clip(
                        onset_latency,
                        float(t_on_tmpl) - float(STIM_ONSET_MAX_DEV_S),
                        float(t_on_tmpl) + float(STIM_ONSET_MAX_DEV_S),
                    ))

                peak1_latency = peak2_latency = np.nan
                peak1_value = peak2_value = np.nan

                # P1 detection near template P1
                if not np.isnan(t_p1):
                    idx_tmpl_p1 = int(np.argmin(np.abs(times - t_p1)))
                    pol1 = np.sign(tmpl[idx_tmpl_p1] - np.mean(tmpl[baseline_mask]))
                    pol1 = +1 if pol1 >= 0 else -1
                    tmpl_p1_val = tmpl[idx_tmpl_p1]
                    peak1_latency, peak1_value = detect_peak_in_window(
                        sig_f=sig, times=times, sfreq=sfreq,
                        baseline_mask=baseline_mask, t_center=t_p1,
                        win_ms=5.0, polarity=pol1,
                        amp_min_uV=STIM_PEAK_AMP_MIN_UV, min_width_ms=0.4,
                        choose="nearest",
                        template_peak_val=tmpl_p1_val, min_rel_to_template=0.1,
                        t_min=resp_tmin, t_max=resp_tmax,
                    )
                else:
                    pol1 = +1

                # P2 detection near template P2
                if not np.isnan(t_p2):
                    idx_tmpl_p2 = int(np.argmin(np.abs(times - t_p2)))
                    pol2 = np.sign(tmpl[idx_tmpl_p2] - np.mean(tmpl[baseline_mask]))
                    pol2 = +1 if pol2 >= 0 else -1
                    tmpl_p2_val = tmpl[idx_tmpl_p2]
                    peak2_latency, peak2_value = detect_peak_in_window(
                        sig_f=sig, times=times, sfreq=sfreq,
                        baseline_mask=baseline_mask, t_center=t_p2,
                        win_ms=12.0, polarity=pol2,
                        amp_min_uV=STIM_PEAK_AMP_MIN_UV, min_width_ms=0.4,
                        choose="nearest",
                        template_peak_val=tmpl_p2_val, min_rel_to_template=0.1,
                        t_min=resp_tmin, t_max=resp_tmax,
                    )
                else:
                    pol2 = +1

                # Refine on unfiltered signal
                if not np.isnan(peak1_latency):
                    peak1_latency, peak1_value = pick_epoch_value_near_latency(
                        sig, times, peak1_latency, sfreq, win_ms=1.0, polarity=pol1,
                    )
                if not np.isnan(peak2_latency):
                    peak2_latency, peak2_value = pick_epoch_value_near_latency(
                        sig, times, peak2_latency, sfreq, win_ms=1.0, polarity=pol2,
                    )

                # Ordering / same-sign guards
                if (not np.isnan(peak1_latency)) and (not np.isnan(peak2_latency)):
                    if peak2_latency <= peak1_latency:
                        peak2_latency = peak2_value = np.nan
                if (not np.isnan(peak1_value)) and (not np.isnan(peak2_value)):
                    if np.sign(peak1_value) == np.sign(peak2_value):
                        peak2_latency = peak2_value = np.nan

                # PTP
                if (not np.isnan(peak1_value)) and (not np.isnan(peak2_value)):
                    ptp_amp = float(np.abs(peak1_value - peak2_value))
                else:
                    ptp_amp = np.nan

                # Amplitude / PTP thresholds → wipe
                if (not np.isnan(ptp_amp)) and (ptp_amp < (STIM_PTP_MIN_UV * 1e-6)):
                    onset_latency = peak1_latency = peak2_latency = np.nan
                    peak1_value = peak2_value = ptp_amp = np.nan
                if (not np.isnan(peak1_value)) and (np.abs(peak1_value) <= (STIM_P1_ABS_MIN_UV * 1e-6)):
                    onset_latency = peak1_latency = peak2_latency = np.nan
                    peak1_value = peak2_value = ptp_amp = np.nan
                if np.isnan(peak1_value) and np.isnan(peak2_value):
                    onset_latency = peak1_latency = peak2_latency = np.nan
                    peak1_value = peak2_value = ptp_amp = np.nan

                channel_epoch_results[ch_name].append(
                    {"ep": ep, "onset": onset_latency,
                     "p1": peak1_latency, "p2": peak2_latency,
                     "pv1": peak1_value, "pv2": peak2_value,
                     "ptp": ptp_amp, "sig": sig}
                )

        # ── Channel-level consistency / artifact rejection ──
        corr_min_median = float(STIM_CHANNEL_MIN_MEDIAN_CORR)
        min_valid_frac = float(STIM_CHANNEL_MIN_VALID_FRAC)

        artifact_mean_by_ch: dict[str, np.ndarray] = {}
        for art_ch in art_chans:
            if art_ch not in epochs.ch_names:
                continue
            art_idx = epochs.ch_names.index(art_ch)
            artifact_mean_by_ch[art_ch] = np.asarray(
                np.mean(data_epoched[:, art_idx, :], axis=0), dtype=float,
            )

        channel_inter_trial_corr: dict[str, float] = {}
        for ch in eligible_channels:
            entries = channel_epoch_results[ch]
            tmpl = tmpl_cfg[ch]

            sim_mask = (times >= resp_tmin) & (times <= resp_tmax)
            tmpl_seg = tmpl[sim_mask] - np.mean(tmpl[baseline_mask])

            corrs = []
            valid_flags = []
            ch_idx_for_itc = epochs.ch_names.index(ch)
            ep_resp_segs = []
            for e in entries:
                is_valid = not np.isnan(e["p1"])
                valid_flags.append(is_valid)
                sig_ep = data_epoched[e["ep"], ch_idx_for_itc, :]
                seg = sig_ep[sim_mask] - np.mean(sig_ep[baseline_mask])
                ep_resp_segs.append(seg)
                if np.std(seg) == 0 or np.std(tmpl_seg) == 0:
                    corrs.append(0.0)
                else:
                    c = float(np.corrcoef(seg, tmpl_seg)[0, 1])
                    corrs.append(0.0 if np.isnan(c) else c)

            # Inter-trial consistency: median pairwise Pearson corr among
            # epoch traces in the response window for this channel/file.
            if len(ep_resp_segs) >= 2:
                seg_mat = np.stack(ep_resp_segs, axis=0)
                seg_stds = seg_mat.std(axis=1)
                keep = seg_stds > 0
                if keep.sum() >= 2:
                    cm = np.corrcoef(seg_mat[keep])
                    iu = np.triu_indices(cm.shape[0], k=1)
                    pair_corrs = cm[iu]
                    pair_corrs = pair_corrs[~np.isnan(pair_corrs)]
                    inter_trial_corr = float(np.median(pair_corrs)) if pair_corrs.size else 0.0
                else:
                    inter_trial_corr = 0.0
            else:
                inter_trial_corr = 0.0
            channel_inter_trial_corr[ch] = inter_trial_corr

            valid_fraction = np.mean(valid_flags) if valid_flags else 0.0
            median_corr = float(np.median(corrs)) if corrs else 0.0

            artifact_like_channel = False
            if STIM_EPOCH_ARTIFACT_CORR_REJECTION:
                ch_idx = epochs.ch_names.index(ch)
                ch_mean = np.asarray(np.mean(data_epoched[:, ch_idx, :], axis=0), dtype=float)
                ch_centered = ch_mean - np.mean(ch_mean)
                if float(np.std(ch_centered)) > 0:
                    for _art_ch, art_mean in artifact_mean_by_ch.items():
                        art_centered = art_mean - np.mean(art_mean)
                        if float(np.std(art_centered)) <= 0:
                            continue
                        corr = float(np.corrcoef(ch_centered, art_centered)[0, 1])
                        if not np.isnan(corr) and abs(corr) >= float(STIM_EPOCH_ARTIFACT_ABS_CORR_THR):
                            artifact_like_channel = True
                            break

            low_median_corr = (median_corr < corr_min_median)
            low_valid_low_corr = (valid_fraction < min_valid_frac) and (median_corr < corr_min_median)
            borderline_valid_very_low_corr = (valid_fraction < 0.6) and (median_corr < 0.2)
            low_inter_trial_corr = (
                STIM_MIN_INTER_TRIAL_CORR is not None
                and inter_trial_corr < float(STIM_MIN_INTER_TRIAL_CORR)
            )
            if (
                artifact_like_channel
                or low_median_corr
                or low_valid_low_corr
                or borderline_valid_very_low_corr
                or low_inter_trial_corr
            ):
                for e in entries:
                    e["onset"] = e["p1"] = e["p2"] = np.nan
                    e["pv1"] = e["pv2"] = e["ptp"] = np.nan

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
                results["Inter-trial corr"].append(inter_trial_corr)
                results["Time series"].append(e["sig"])

        base_name = file_name.split(".fif")[0]
        plot_epochs_panel(
            data_epoched=data_epoched, times=times,
            ch_names=raw_file.info["ch_names"], epochs_ch_names=epochs.ch_names,
            art_chans=art_set, latency_markers=latency_markers,
            title=base_name, out_path=plots_grid_dir / f"{base_name}.png",
            show_grid=True, show_markers=True,
        )
        plot_epochs_panel(
            data_epoched=data_epoched, times=times,
            ch_names=raw_file.info["ch_names"], epochs_ch_names=epochs.ch_names,
            art_chans=art_set, latency_markers=latency_markers,
            title=base_name, out_path=plots_plain_dir / f"{base_name}.png",
            show_grid=False, show_markers=False,
        )

    # ── Write outputs ──
    print("[SIR] Writing outputs...", flush=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(excel_dir / "Large_dataset_emg_response_metrics.csv", index=False)

    if group_store:
        plot_grouped_by_amplitude(group_store, times, plots_grouped_dir, default_art_set)

    metrics = [
        "Onset latency", "Peak1 latency", "Peak2 latency",
        "Peak1 value", "Peak2 value", "PTP amplitude",
        "Inter-trial corr",
    ]

    df = df_results.copy()
    detection_metrics = [m for m in metrics if m != "Inter-trial corr"]
    # Keep rows where any detection metric is non-NaN OR ITC is non-trivial,
    # so the summary still surfaces ITC for channels that were wiped.
    keep_mask = (
        df[detection_metrics].notna().any(axis=1)
        | (pd.to_numeric(df["Inter-trial corr"], errors="coerce").fillna(0.0) != 0)
    )
    df = df.loc[keep_mask].reset_index(drop=True)
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
            n_total="size", n_valid=lambda x: x.notna().sum(),
            mean="mean", std="std", median="median",
            q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
            min="min", max="max",
        ).reset_index()
        tmp["iqr"] = tmp["q75"] - tmp["q25"]
        tmp["Metric"] = metric
        summary_rows.append(tmp)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary[[
        "Configuration", "Stim. amplitude", "Stim_amp_num", "Channel", "Metric",
        "n_total", "n_valid", "mean", "std", "median", "q25", "q75", "iqr", "min", "max",
    ]]
    summary = summary.sort_values(
        by=["Configuration", "Stim_amp_num", "Stim. amplitude", "Channel", "Metric"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary.to_csv(excel_dir / "Summary_stats_by_config_amp_channel.csv", index=False)

    with pd.ExcelWriter(
        excel_dir / "Summary_stats_by_config_amp_channel_by_config.xlsx", engine="xlsxwriter",
    ) as writer:
        for config, dfc in summary.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc = dfc.sort_values(
                by=["Stim_amp_num", "Stim. amplitude", "Channel", "Metric"], kind="mergesort",
            ).reset_index(drop=True)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(
        excel_dir / "Large_dataset_emg_response_metrics_by_config.xlsx", engine="xlsxwriter",
    ) as writer:
        for config, dfc in df_results.groupby("Configuration"):
            sheet_name = _excel_safe_sheet_name(config)
            dfc.to_excel(writer, sheet_name=sheet_name, index=False)

    plot_boxplots(df, boxplot_dir)
    if old_mne_log_level is not None:
        mne.set_log_level(old_mne_log_level)

    return output_root

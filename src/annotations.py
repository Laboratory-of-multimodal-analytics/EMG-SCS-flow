"""Annotation-based cropping utilities."""

from __future__ import annotations

import os
import re


MISSING_AMP_LABEL = "unspecified"

CONFIG_RE = re.compile(r"^\s*(?P<cfg>\d+(?:[+-]\d+)+[+-]?(?:-[A-Za-z]+)?)\s*$")
CONFIG_WITH_AMP_RE = re.compile(
    r"^\s*(?P<cfg>\d+(?:[+-]\d+)+[+-]?)\s+(?P<amp>[+-]?\d+(?:[.,]\d+)?)(?:\s*(?:mA|uA|ms|Hz|ma|ua))?\s*$",
    re.IGNORECASE,
)
CONFIG_TOKEN_RE = re.compile(r"(?P<cfg>\d+(?:[+-]\d+)+[+-]?)")
AMP_RE = re.compile(r"^\s*[+-]?\d+(?:[.,/]\d+)?(?:\s*(?:mA|uA|ms|Hz|ma|ua))?\s*$", re.IGNORECASE)
STARTSTOP_RE = re.compile(r"^\s*(start|stop)\W*\s*$", re.IGNORECASE)


def _parse_annotation(desc: str) -> tuple[str | None, str | None]:
    desc = desc.strip()
    if not desc:
        return None, None

    # In STARTSTOP flow, labels like "stim 0+1+6-7-,30Hz,11mA,450mks"
    # should be treated as condition names, not config/amplitude headers.
    if "stim" in desc.lower():
        return None, None

    if "p" in desc and desc.lower() != "p 300mks 2hz":
        cfg = desc
        if cfg.startswith("p"):
            cfg = cfg[1:].strip()
        return cfg.replace(" ", ""), None

    m = CONFIG_WITH_AMP_RE.match(desc)
    if m:
        cfg = m.group("cfg").replace(" ", "")
        amp = m.group("amp")
        return cfg, amp

    m = CONFIG_RE.match(desc)
    if m:
        return m.group("cfg").replace(" ", ""), None

    # Mixed-order config strings: "2HZ, 450mks,0+7-", "0+3-, 450,2hhz"
    # Keep this broad enough for real-world typos while requiring both
    # a config-like token and stimulation-related units.
    m = CONFIG_TOKEN_RE.search(desc.replace(" ", ""))
    if m and re.search(r"(?:hz|hhz|mks|ms|us|ma|ua)\b", desc, re.IGNORECASE):
        return m.group("cfg"), None

    if AMP_RE.match(desc):
        return None, desc.replace(" ", "")

    return None, None


def _next_relevant_index(items: list[tuple[float, str]], start: int) -> int | None:
    for j in range(start + 1, len(items)):
        _cfg, _amp = _parse_annotation(items[j][1])
        if _cfg is not None or _amp is not None:
            return j
    return None


def _normalize_startstop(desc: str) -> str | None:
    if not desc:
        return None
    stripped = desc.strip()
    m = STARTSTOP_RE.match(stripped)
    if m:
        return m.group(1).lower()
    cleaned = re.sub(r"[^\w]+", "", stripped.lower())
    if cleaned in {"start", "stop"}:
        return cleaned
    return None


def extract_start_stop_segments(raw) -> dict[str, dict[str, list[tuple[float, float]]]]:
    ann = raw.annotations
    items = []
    for i in range(len(ann)):
        items.append((float(ann.onset[i]), str(ann.description[i])))
    items.sort(key=lambda x: x[0])

    segments: dict[str, dict[str, list[tuple[float, float]]]] = {}
    segment_meta: dict[str, dict[str, bool]] = {}
    current_condition: str | None = None
    current_onset: float | None = None
    has_markers = False
    last_start: float | None = None
    last_stop: float | None = None

    def _close_open_segment(next_onset: float | None) -> None:
        nonlocal last_start, last_stop, has_markers
        if current_condition is None or current_onset is None:
            return
        if last_start is not None and next_onset is not None and next_onset > last_start:
            segments[current_condition]["start"].append((last_start, next_onset))
            last_start = None
            has_markers = True
            segment_meta[current_condition]["has_explicit_markers"] = True
        if not has_markers and next_onset is not None and next_onset > current_onset:
            segments[current_condition]["start"].append((current_onset, next_onset))
            segment_meta[current_condition]["used_fallback_full_segment"] = True

    for onset, desc in items:
        label = _normalize_startstop(desc)
        if label == "start":
            if current_condition is None:
                continue
            if last_stop is not None and onset > last_stop:
                segments[current_condition]["stop"].append((last_stop, onset))
            last_start = onset
            has_markers = True
            segment_meta[current_condition]["has_explicit_markers"] = True
            continue
        if label == "stop":
            if current_condition is None:
                continue
            if last_start is not None and onset > last_start:
                segments[current_condition]["start"].append((last_start, onset))
                segment_meta[current_condition]["has_explicit_markers"] = True
            last_stop = onset
            last_start = None
            has_markers = True
            segment_meta[current_condition]["has_explicit_markers"] = True
            continue

        cfg, amp = _parse_annotation(desc)
        if cfg is not None or amp is not None:
            _close_open_segment(onset)
            current_condition = None
            current_onset = None
            last_start = None
            last_stop = None
            has_markers = False
            continue

        condition = desc.strip()
        if not condition:
            continue

        _close_open_segment(onset)

        current_condition = condition
        current_onset = onset
        segments.setdefault(current_condition, {"start": [], "stop": []})
        segment_meta.setdefault(
            current_condition,
            {"has_explicit_markers": False, "used_fallback_full_segment": False},
        )
        last_start = None
        last_stop = None
        has_markers = False

    end_time = float(raw.times[-1]) if getattr(raw, "times", None) is not None else None
    _close_open_segment(end_time)

    for condition, parts in segments.items():
        starts = parts.get("start", [])
        if not starts:
            print(f"[STARTSTOP segments] condition={condition}: no start segments", flush=True)
            continue
        meta = segment_meta.get(
            condition, {"has_explicit_markers": False, "used_fallback_full_segment": False}
        )
        if meta.get("used_fallback_full_segment", False) and not meta.get("has_explicit_markers", False):
            for tmin, tmax in starts:
                print(
                    "[STARTSTOP segments] "
                    f"condition={condition}: full segment {tmin:.3f}s -> {tmax:.3f}s",
                    flush=True,
                )
            continue
        for tmin, tmax in starts:
            print(
                "[STARTSTOP segments] "
                f"condition={condition}: start interval {tmin:.3f}s -> {tmax:.3f}s",
                flush=True,
            )

    return segments


def _safe_crop_basename(cfg: str, amp: str, suffix: str) -> str:
    """Build a FIF crop filename without path separators (avoid creating subdirs)."""
    safe_cfg = re.sub(r"[/\\]", "_", str(cfg))
    safe_amp = re.sub(r"[/\\]", "_", str(amp))
    return f"{safe_cfg}_{safe_amp}{suffix}.fif"


def create_annotation_crops(raw, out_dir: str | os.PathLike, overwrite: bool = True) -> None:
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ann = raw.annotations
    items = []
    for i in range(len(ann)):
        items.append((float(ann.onset[i]), str(ann.description[i])))

    items.sort(key=lambda x: x[0])

    sfreq = float(raw.info["sfreq"])
    dt = 1.0 / sfreq

    current_cfg = None
    seen: dict[tuple[str, str], int] = {}

    for idx, (onset, desc) in enumerate(items):
        cfg, amp = _parse_annotation(desc)

        if cfg is not None and amp is not None:
            current_cfg = cfg
        elif cfg is not None:
            current_cfg = cfg
            next_idx = _next_relevant_index(items, idx)
            if next_idx is None:
                next_onset = float(raw.times[-1])
            else:
                next_cfg, next_amp = _parse_annotation(items[next_idx][1])
                if next_cfg is not None:
                    next_onset = float(items[next_idx][0])
                    amp = MISSING_AMP_LABEL
                else:
                    continue
            tmin = onset
            tmax = next_onset - dt
            if tmax <= tmin:
                continue
            key = (current_cfg, amp)
            seen[key] = seen.get(key, 0) + 1
            suffix = f"({seen[key]})" if seen[key] > 1 else ""
            out_name = _safe_crop_basename(current_cfg, amp, suffix)
            out_path = os.path.join(out_dir, out_name)
            os.makedirs(os.path.dirname(out_path) or out_dir, exist_ok=True)
            crop_raw = raw.copy().crop(tmin=tmin, tmax=tmax)
            crop_raw.save(out_path, overwrite=overwrite)
            continue

        if amp is None:
            continue
        if current_cfg is None:
            continue

        next_idx = _next_relevant_index(items, idx)
        if next_idx is None:
            next_onset = float(raw.times[-1])
        else:
            next_onset = float(items[next_idx][0])

        tmin = onset
        tmax = next_onset - dt

        if tmax <= tmin:
            continue

        key = (current_cfg, amp)
        seen[key] = seen.get(key, 0) + 1
        suffix = f"({seen[key]})" if seen[key] > 1 else ""

        out_name = _safe_crop_basename(current_cfg, amp, suffix)
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(os.path.dirname(out_path) or out_dir, exist_ok=True)
        crop_raw = raw.copy().crop(tmin=tmin, tmax=tmax)
        crop_raw.save(out_path, overwrite=overwrite)

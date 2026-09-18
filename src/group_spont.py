"""Group analysis of spontaneous EMG across finished StartStop runs.

The stimulation-induced group (``src.group``) compares recruitment curves. This
module compares what the muscles did on their own inside the start–stop
intervals: the RMS level per channel, the number of bursts, the RMS time course
of a segment and the shape of the bursts themselves.

The unit of analysis is a MEMBER: one condition of one run. A run typically
holds several conditions ("Flex hand left stim", "Flex hand left non stim", …)
and each is its own recording of a state, so each gets its own row. A member
carries two labels, exactly as in the stimulation group:

    subject — who or what the recording is of (patient, animal, task);
    state   — what that recording IS relative to the others (stim / non stim,
              before / after, day 30, …).

Both are guessed from the names and meant to be corrected by hand.

Channels are matched across files by a canonical name: case, spaces, hyphens
and underscores are ignored, and an alias table ("BB R=Biceps R; ...") folds
names that differ but mean the same muscle. A member that lacks a channel, or
detected nothing on it, simply contributes nothing there — the group is built
from whoever has the data, and every panel says how many that was.

Nothing here touches Qt; the tab renders what these functions return.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Members and labels
# --------------------------------------------------------------------------- #
_NON_STIM = re.compile(r"non[\s_-]*stim|no[\s_-]*stim|без\s*стим|\bбс\b|_бс(_|$)|off", re.I)
_STIM = re.compile(r"stim|стим|on\b", re.I)
_TRAIL = re.compile(r"[\s_]*\d+\s*$")


def parse_condition(run_name: str, condition: str) -> tuple[str, str]:
    """``(subject, state)`` guessed from a run's name and a condition's name.

    When the condition name says whether stimulation was on, the state is that
    and the subject is what is left of the name once the stim words and a
    trailing repeat number are removed — "Flex hand left stim 2" becomes subject
    "Flex hand left", state "stim", which is what pairs it with its own
    non-stim recording. Otherwise the run is the subject and the condition is
    the state.
    """
    text = condition.strip()
    if _NON_STIM.search(text):
        state = "non stim"
        rest = _NON_STIM.sub(" ", text)
    elif _STIM.search(text):
        state = "stim"
        rest = _STIM.sub(" ", text)
    else:
        return run_name, text or "—"
    rest = re.sub(r"\s+", " ", _TRAIL.sub("", rest)).strip(" _-")
    # "ext hand right" and "Ext hand right" are the same task typed twice; left apart
    # they would never pair with each other's stim / non stim recording
    rest = rest[:1].upper() + rest[1:]
    return rest or run_name, state


def sort_states(states) -> list[str]:
    order = {"non stim": 0, "before": 0, "control": 0, "stim": 1, "after": 1}
    return sorted({str(s) for s in states}, key=lambda s: (order.get(s, 2), s.lower()))


@dataclass
class SpontMember:
    root: Path
    condition: str
    subject: str
    state: str
    include: bool = True
    #: the labels as read from the names, before any grouping or hand edit
    parsed_subject: str = ""
    parsed_state: str = ""
    #: the user's grouping folder: the first folder under the one added to the tab
    folder: str = ""

    @property
    def label(self) -> str:
        return f"{self.subject} · {self.state}"


def scan_members(folder: Path, max_depth: int = 4) -> list[SpontMember]:
    """Every (run, condition) with spontaneous-EMG outputs under *folder*."""
    from emgflow.results import SpontaneousResults

    folder = Path(folder)
    found: list[SpontMember] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            res = SpontaneousResults(d)
            ok = res.ok
        except Exception:
            ok = False
        if ok:
            from .group import folder_group

            for cond in res.conditions():
                subject, state = parse_condition(d.name, cond)
                found.append(SpontMember(d, cond, subject, state,
                                         parsed_subject=subject, parsed_state=state,
                                         folder=folder_group(d, folder)))
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name in {
                    "Excel", "review", "data", "templates", "Envelopes", "Plots"}:
                continue
            walk(child, depth + 1)

    walk(folder, 0)
    found.sort(key=lambda m: (m.subject, m.state, m.condition))
    return found


#: What a member's group can be read from, in the order offered on screen.
GROUP_MODES = {
    "state": "стимуляция (stim / non stim)",
    "subject": "задача (из названия условия)",
    "subject+state": "задача + стимуляция",
    "condition": "условие целиком",
    "run": "прогон",
    "folder": "папка (подпапка добавленной папки)",
    "folder+state": "папка + стимуляция",
}


def group_of(member: SpontMember, mode: str) -> str:
    """The group label of *member* under grouping *mode* (see ``GROUP_MODES``)."""
    state = member.parsed_state or member.state
    subject = member.parsed_subject or member.subject
    return {
        "state": state,
        "subject": subject,
        "subject+state": f"{subject} · {state}",
        "condition": member.condition,
        "run": member.root.name,
        "folder": member.folder or member.root.parent.name,
        "folder+state": f"{member.folder or member.root.parent.name} · {state}",
    }.get(mode, state)


# --------------------------------------------------------------------------- #
# Channel matching
# --------------------------------------------------------------------------- #
def parse_aliases(text: str) -> dict[str, str]:
    """``"BB R=Biceps R; FCU R = Flexor R"`` → {canonical(alias): canonical(target)}."""
    out: dict[str, str] = {}
    for part in re.split(r"[;\n]", text or ""):
        if "=" not in part:
            continue
        a, b = part.split("=", 1)
        if a.strip() and b.strip():
            out[_key(a)] = b.strip()
    return out


def _key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def canonical_channel(name: str, aliases: dict[str, str] | None = None) -> str:
    """The name a channel is matched by across files."""
    aliases = aliases or {}
    k = _key(name)
    if k in aliases:
        return aliases[k]
    return str(name).strip()


def match_channels(names, aliases: dict[str, str] | None = None) -> dict[str, str]:
    """{raw name: canonical name}; names differing only by case/spacing merge."""
    aliases = aliases or {}
    first: dict[str, str] = {}
    out: dict[str, str] = {}
    for n in names:
        target = canonical_channel(n, aliases)
        key = _key(target)
        first.setdefault(key, target)
        out[str(n)] = first[key]
    return out


# --------------------------------------------------------------------------- #
# Collecting
# --------------------------------------------------------------------------- #
METRICS = {
    "rms_uv": "RMS, µV (среднее по окнам)",
    "amp_uv": "средняя амплитуда, µV",
    "n_bursts": "число вспышек",
    "burst_rms_uv": "RMS вспышек, µV",
    "burst_dur_s": "длительность вспышек, с",
}


def _member_frames(members):
    from emgflow.results import SpontaneousResults

    for m in members:
        if not m.include:
            continue
        try:
            res = SpontaneousResults(Path(m.root))
            if not res.ok:
                continue
            yield m, res
        except Exception:
            continue


def collect_summary(members, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per (member, channel): the per-channel numbers of the pipeline's summary.

    Columns: subject, state, run, condition, channel, rms_uv, amp_uv, n_bursts,
    n_windows, burst_rms_uv, burst_dur_s. A member whose summary lacks a channel
    is simply absent on it.
    """
    rows = []
    for m, res in _member_frames(members):
        s = res.summary(m.condition)
        if s.empty or "Channel" not in s.columns:
            continue
        b = res.bursts(m.condition)
        for _, r in s.iterrows():
            ch = str(r["Channel"])
            bb = b[b["Channel"].astype(str) == ch] if not b.empty and "Channel" in b else pd.DataFrame()
            rows.append({
                "subject": m.subject, "state": m.state, "run": str(m.root),
                "condition": m.condition, "raw_channel": ch,
                "rms_uv": pd.to_numeric(r.get("RMS_mean_uV"), errors="coerce"),
                "amp_uv": pd.to_numeric(r.get("Amp_mean_uV"), errors="coerce"),
                "n_bursts": pd.to_numeric(r.get("n_bursts"), errors="coerce"),
                "n_windows": pd.to_numeric(r.get("n_windows"), errors="coerce"),
                "burst_rms_uv": float(pd.to_numeric(bb["RMS_mean_uV"], errors="coerce").mean())
                if not bb.empty else np.nan,
                "burst_dur_s": float(pd.to_numeric(bb["Duration_s"], errors="coerce").mean())
                if not bb.empty else np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=["subject", "state", "run", "condition", "raw_channel",
                                     "channel"] + list(METRICS))
    out = pd.DataFrame(rows)
    mapping = match_channels(out["raw_channel"].unique(), aliases)
    out["channel"] = out["raw_channel"].map(mapping)
    return out


def collect_timecourses(members, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """RMS per window along each member's segment, all channels, long form.

    Columns: subject, state, run, condition, channel, t, frac, rms_uv. ``t`` is
    seconds from the segment start, ``frac`` its position in [0, 1] — segments
    of different length can only be averaged on the fractional axis.
    """
    frames = []
    for m, res in _member_frames(members):
        p = res.base / m.condition / "Excel" / f"Spontaneous_EMG_detailed_{m.condition}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if d.empty or "Window_center_s" not in d.columns:
            continue
        t = pd.to_numeric(d["Window_center_s"], errors="coerce")
        span = float(t.max()) if len(t) else np.nan
        frames.append(pd.DataFrame({
            "subject": m.subject, "state": m.state, "run": str(m.root),
            "condition": m.condition, "raw_channel": d["Channel"].astype(str),
            "t": t, "frac": t / span if span and span > 0 else t * 0,
            "rms_uv": pd.to_numeric(d["RMS_uV"], errors="coerce"),
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["channel"] = out["raw_channel"].map(match_channels(out["raw_channel"].unique(), aliases))
    return out


def collect_burst_envelopes(members, aliases: dict[str, str] | None = None,
                            n_times: int = 200) -> pd.DataFrame:
    """Each member's MEAN burst envelope per channel, on one time-from-centre grid.

    Mean of means: the bursts of one member are averaged first, so a member with
    ten bursts weighs the same as one with two when the group is averaged next.
    The grid spans the intersection of the envelopes' own windows.
    """
    from emgflow.results import SpontaneousResults

    loaded = []
    for m, res in _member_frames(members):
        for f in res.envelope_files(m.condition):
            if "_burst" not in f.stem:
                continue
            ch = f.stem.split("_burst")[0]
            try:
                t, v = SpontaneousResults.read_envelope(f)
            except Exception:
                continue
            if len(t) < 3:
                continue
            loaded.append((m, ch, np.asarray(t, float), np.asarray(v, float)))
    if not loaded:
        return pd.DataFrame()
    lo = max(float(t[0]) for _, _, t, _ in loaded)
    hi = min(float(t[-1]) for _, _, t, _ in loaded)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.median([t[0] for _, _, t, _ in loaded]))
        hi = float(np.median([t[-1] for _, _, t, _ in loaded]))
    grid = np.linspace(lo, hi, int(n_times))
    per: dict[tuple, list[np.ndarray]] = {}
    for m, ch, t, v in loaded:
        vi = np.interp(grid, t, v)
        vi[(grid < t[0]) | (grid > t[-1])] = np.nan
        per.setdefault((m.subject, m.state, str(m.root), m.condition, ch), []).append(vi)
    rows = []
    for (subject, state, run, cond, ch), stack in per.items():
        mean = np.nanmean(np.vstack(stack), axis=0)
        rows.append(pd.DataFrame({
            "subject": subject, "state": state, "run": run, "condition": cond,
            "raw_channel": ch, "t": grid, "value": mean, "n_bursts": len(stack)}))
    out = pd.concat(rows, ignore_index=True)
    out["channel"] = out["raw_channel"].map(match_channels(out["raw_channel"].unique(), aliases))
    return out


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
NORM_NONE = "none"
NORM_BASELINE = "baseline"
NORM_LABELS = {
    NORM_NONE: "без нормировки (µV)",
    NORM_BASELINE: "на базовое состояние субъекта",
}


def normalisation_factors(summary: pd.DataFrame, baseline_state: str,
                          metric: str = "rms_uv") -> pd.DataFrame:
    """One scale per (subject, channel): the metric in that subject's baseline state.

    Several baseline members of one subject (repeats) are averaged. Per subject,
    never pooled — the µV level of a muscle depends on its electrodes.
    """
    if summary.empty:
        return pd.DataFrame(columns=["subject", "channel", "factor"])
    base = summary[summary["state"] == baseline_state]
    if base.empty:
        return pd.DataFrame(columns=["subject", "channel", "factor"])
    f = base.groupby(["subject", "channel"])[metric].mean().reset_index()
    f = f.rename(columns={metric: "factor"})
    f = f[pd.to_numeric(f["factor"], errors="coerce") > 0]
    return f


def apply_normalisation(table: pd.DataFrame, factors: pd.DataFrame,
                        column: str) -> tuple[pd.DataFrame, list[str]]:
    """Add ``value`` = *column* / factor per (subject, channel); list who had none.

    Counts are never divided: "number of bursts as a fraction of baseline" is
    not what anyone asked for.
    """
    out = table.copy()
    raw = pd.to_numeric(out[column], errors="coerce")
    if factors.empty or column == "n_bursts" or column == "burst_dur_s":
        out["value"] = raw
        out["unit"] = METRICS.get(column, column)
        return out, []
    merged = out.merge(factors, on=["subject", "channel"], how="left")
    out["value"] = raw.to_numpy() / merged["factor"].to_numpy()
    out["unit"] = "доля базового состояния"
    gone = merged.loc[merged["factor"].isna(), ["subject", "channel"]].drop_duplicates()
    return out, sorted(f"{r.subject}/{r.channel}" for r in gone.itertuples())


# --------------------------------------------------------------------------- #
# Averaging and contrasts
# --------------------------------------------------------------------------- #
def mean_timecourse(tc: pd.DataFrame, axis: str = "frac", n: int = 60,
                    value: str = "value", min_n: int = 2) -> pd.DataFrame:
    """Mean ± SE across members at each point of a shared axis, per (state, channel).

    ``axis`` is ``frac`` (position in the segment, 0–1) or ``t`` (seconds).
    Points backed by fewer than *min_n* members are dropped; a single member's
    curve is not a group mean.
    """
    if tc.empty:
        return pd.DataFrame()
    x = pd.to_numeric(tc[axis], errors="coerce")
    lo, hi = float(x.min()), float(x.quantile(0.98) if axis == "t" else x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.DataFrame()
    grid = np.linspace(lo, hi, int(n))
    rows = []
    for (subject, state, run, cond, ch), g in tc.groupby(
            ["subject", "state", "run", "condition", "channel"], sort=False):
        d = g[[axis, value]].dropna().sort_values(axis)
        if len(d) < 2:
            continue
        xs, ys = d[axis].to_numpy(float), d[value].to_numpy(float)
        yi = np.interp(grid, xs, ys)
        yi[(grid < xs[0]) | (grid > xs[-1])] = np.nan
        rows.append(pd.DataFrame({"state": state, "channel": ch, "x": grid, "y": yi}))
    if not rows:
        return pd.DataFrame()
    on_grid = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    out = on_grid.groupby(["state", "channel", "x"])["y"].agg(
        mean="mean", sd="std", n="count").reset_index()
    out = out[out["n"] >= int(min_n)]
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out


def mean_burst_envelope(env: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    """Mean ± SE of the members' mean burst envelopes, per (state, channel)."""
    if env.empty:
        return pd.DataFrame()
    d = env.dropna(subset=["value"])
    out = d.groupby(["state", "channel", "t"])["value"].agg(
        mean="mean", sd="std", n="count").reset_index()
    out = out[out["n"] >= int(min_n)]
    out["se"] = out["sd"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
    return out



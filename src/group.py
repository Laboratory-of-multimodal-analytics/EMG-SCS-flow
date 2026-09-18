"""Group analysis across finished runs.

One recording answers "what did this muscle do at this current". A group answers
"did it change" — across animals, across sessions, before and after. The runs
already on disk hold everything needed for that; what they do not hold is the
one thing that makes them a group: which of them belong together, and what each
one IS relative to the others.

So a group here is a set of runs carrying two labels each:

    subject  — the animal or the patient the run came from
    state    — what that run is a recording OF (implantation, 30 day, control,
               before, after, a stimulation level, ...)

Everything else follows from those two. Averaging is over subjects within a
state. A contrast is state against state, paired inside each subject. A baseline
is simply the state declared to be one, and normalising means dividing every
subject's curves by something measured in ITS OWN baseline run — never by a
group-level number, which would carry one animal's scale into another's.

The labels are guessed from the folder path (see ``parse_labels``) and are meant
to be corrected by hand: the guesses are good on the naming conventions actually
used, and wrong labels are the one error here that no amount of statistics will
reveal.

Nothing in this module touches Qt; the group tab renders what these functions
return, and the same functions can be driven from a script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import asymmetry as A

# --------------------------------------------------------------------------- #
# Runs and their labels
# --------------------------------------------------------------------------- #

#: Spelled six ways across the pig array: implantation, inplantation,
#: implantattion, inplanation. All one state.
_IMPLANT = re.compile(r"i[mn]plant?a?t?t?ion|имплант", re.I)
_CONTROL = re.compile(r"\bcontrol\b|контроль", re.I)
_DAY = re.compile(r"(\d+)\s*(?:day|день|дн)", re.I)
#: "pig 5_2 30 day after SCI", "Pig 6_2 ...", "pig 4_6_120 day after SCI"
_PIG = re.compile(r"pig[\s_]*(\d+[\s_-]\d+)[\s_-]*(.*)", re.I)
#: A patient folder from the Neurosoft tree: "(Ж14) Th11-12"
_BRACKETED = re.compile(r"^\((.+?)\)\s*(.*)$")


def normalise_state(text: str) -> str:
    """A state label from free text, folding the spellings that mean one thing.

    Deliberately conservative: anything it does not recognise is returned
    trimmed but otherwise untouched, so an unfamiliar protocol becomes its own
    state rather than being folded into a neighbour's.
    """
    text = re.sub(r"\s+", " ", str(text)).strip(" _-")
    if not text:
        return "—"
    if _IMPLANT.search(text):
        return "implantation"
    if _CONTROL.search(text):
        return "control"
    m = _DAY.search(text)
    if m:
        return f"{int(m.group(1))} day"
    return text


def _state_order_key(state: str) -> tuple:
    """Order states the way the experiment ran, not alphabetically.

    control and implantation come first because they are the baselines, then the
    day numbers in numeric order — "7 day" before "14 day", which sorting as
    text gets backwards. Anything unrecognised keeps alphabetical order after
    them.
    """
    if state == "control":
        return (0, 0, "")
    if state == "implantation":
        return (1, 0, "")
    m = _DAY.search(state)
    if m:
        return (2, int(m.group(1)), "")
    return (3, 0, state.lower())


def sort_states(states) -> list[str]:
    return sorted({str(s) for s in states}, key=_state_order_key)


def parse_labels(run_root: Path) -> tuple[str, str]:
    """``(subject, state)`` guessed from a run folder's path.

    Three conventions are in use and all three are read here:

    * the pig array — ``.../5_2 (7)/pig 5_2 30 day after SCI``: the animal is in
      the run folder itself, and so is the session;
    * the Neurosoft tree — ``.../(Ж14)/(Ж14) Th11-12``: the patient is the
      bracketed code and the rest names the recording, which is the only thing
      distinguishing that patient's runs from each other;
    * anything else — the parent folder is the subject and the run folder is the
      state, which is what a folder-per-subject layout means.
    """
    run_root = Path(run_root)
    name = run_root.name
    parent = run_root.parent.name

    m = _PIG.match(name)
    if m:
        return re.sub(r"[\s-]", "_", m.group(1)), normalise_state(m.group(2))

    m = _BRACKETED.match(name)
    if m:
        subject, rest = m.group(1).strip(), m.group(2).strip()
        return subject, normalise_state(rest or name)

    m = _BRACKETED.match(parent)
    if m:
        return m.group(1).strip(), normalise_state(name)

    if parent and parent not in {"results", "ready", ""}:
        return parent, normalise_state(name)
    return name, "—"


def folder_group(run_root: Path, base: Path | None = None) -> str:
    """The user's own grouping folder of a run: the first folder under *base*.

    *base* is the folder added to a group tab. Results sorted by hand into
    ``base/Responders/…`` and ``base/Non-responders/…`` get the groups
    "Responders" and "Non-responders", however deep inside those folders they
    sit. A run lying straight in *base* gets *base*'s own name; a run added on its
    own (no *base*) gets the folder it lies in.
    """
    run_root = Path(run_root)
    if base is None:
        return run_root.parent.name
    try:
        parts = run_root.relative_to(Path(base)).parts
    except ValueError:
        return run_root.parent.name
    if not parts:
        return run_root.parent.name
    return parts[0] if len(parts) > 1 else Path(base).name


@dataclass
class GroupRun:
    """One member of a group: a finished run plus what it is relative to the rest."""
    root: Path
    subject: str
    state: str
    include: bool = True
    note: str = ""
    #: everything the name could be read for (Neurosoft runs); see parse_neurosoft
    tags: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.subject} · {self.state}"


def looks_like_run(path: Path) -> bool:
    """True when *path* is the root of a finished stimulation-induced run.

    Judged by the metrics table, in any of the three layouts a run can have:
    flattened into ``results/``, under a mode subfolder, or — for runs made
    before outputs were collected at all — straight in the run folder.
    """
    path = Path(path)
    name = "Large_dataset_emg_response_metrics.csv"
    for base in (path / "results", path / "results" / "Stimulation-induced responses", path):
        if (base / "Excel" / name).is_file():
            return True
    return False


def scan_runs(folder: Path, max_depth: int = 4) -> list[GroupRun]:
    """Every finished run under *folder*, labelled.

    Stops descending as soon as a run root is found: everything below one is its
    own output, and walking those on a network drive costs minutes.
    """
    folder = Path(folder)
    found: list[GroupRun] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if looks_like_run(d):
            subject, state = parse_labels(d)
            found.append(GroupRun(d, subject, state))
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name in {
                    "Excel", "review", "data", "templates", "Stimulus-centered epochs"}:
                continue
            walk(child, depth + 1)

    walk(folder, 0)
    found.sort(key=lambda r: (r.subject, _state_order_key(r.state)))
    return found


# --------------------------------------------------------------------------- #
# Collecting the curves
# --------------------------------------------------------------------------- #
#: What a point of a recruitment curve is worth reporting as.
METRICS = {
    "amp_uv": "Response, µV (PTP / |P1|)",
    "p1_uv": "P1, µV",
    "p1_ms": "P1 latency, ms",
    "onset_ms": "Onset latency, ms",
    "area_uvms": "Area of the response, µV·ms",
}


def collect_points(runs, configs=None, channels=None) -> pd.DataFrame:
    """Every selected run's recruitment points in one long table.

    Columns: ``subject, state, run, config, channel, curve, x_value, x_label,
    amp_uv, p1_uv, p1_ms, onset_ms, n_epochs, sd_uv``.

    A run contributes the configurations asked for, or all of its own when none
    are named. Nothing is merged across configurations: two electrode pairs are
    two different stimulations, and averaging them would report a mixture as a
    measurement.
    """
    from emgflow.results import SIRResults

    frames = []
    for run in runs:
        if not run.include:
            continue
        try:
            res = SIRResults(Path(run.root))
        except Exception:
            continue
        if not res.ok:
            continue
        wanted = [c for c in res.configs if configs is None or c in set(configs)]
        hreflex = res.scenario == "H-reflex"
        for config in wanted:
            pts = hreflex_points(res, config) if hreflex else res.recruitment_points(config)
            if pts is None or pts.empty:
                continue
            pts = pts.copy()
            if channels is not None:
                pts = pts[pts["Channel"].isin(set(channels))]
                if pts.empty:
                    continue
            pts["subject"] = run.subject
            pts["state"] = run.state
            pts["run"] = str(run.root)
            pts["config"] = config
            frames.append(pts.rename(columns={"Channel": "channel"}))
    if not frames:
        return pd.DataFrame(columns=["subject", "state", "run", "config", "channel",
                                     "curve", "x_value", "x_label", "amp_uv"])
    out = pd.concat(frames, ignore_index=True)
    # Points whose label carries no number cannot sit on the current axis. Kept
    # out of the curves rather than plotted at an invented x.
    out["x_value"] = pd.to_numeric(out["x_value"], errors="coerce")
    return out


#: How the two components of an H-reflex recording are named as group channels.
H_SUFFIX, M_SUFFIX = " · H", " · M"


def hreflex_points(res, config: str) -> pd.DataFrame:
    """M-wave and H-reflex of one recording as two separate group channels.

    On these files one curve routinely carries one component and not the
    other, and they grow with current in opposite directions, so folding them
    into one "response" would average the M-wave into the H-reflex. Each
    becomes its own channel — ``ch1 · M`` and ``ch1 · H`` — and everything
    downstream (overlay, mean, waveforms, tables) treats them apart.
    """
    hb = res.hreflex_by_curve(config)
    if hb is None or hb.empty:
        return pd.DataFrame()
    frames = []
    for comp, suffix in (("m", M_SUFFIX), ("h", H_SUFFIX)):
        f = pd.DataFrame({
            "Channel": hb["Channel"].astype(str) + suffix,
            "curve": hb["curve"].astype(int),
            "amp_uv": pd.to_numeric(hb[f"{comp}_amp_uv"], errors="coerce"),
            "p1_uv": pd.to_numeric(hb[f"{comp}_p1_uv"], errors="coerce"),
            "p1_ms": pd.to_numeric(hb[f"{comp}_p1_ms"], errors="coerce"),
            "onset_ms": pd.to_numeric(hb[f"{comp}_onset_ms"], errors="coerce"),
            "p2_ms": pd.to_numeric(hb[f"{comp}_p2_ms"], errors="coerce"),
            "area_uvms": (pd.to_numeric(hb[f"{comp}_area_uvms"], errors="coerce")
                          if f"{comp}_area_uvms" in hb.columns else np.nan),
        })
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["x_value"] = out["curve"].astype(float)
    out["x_label"] = out["curve"].astype(str)
    out["n_epochs"] = 1
    out["sd_uv"] = np.nan
    return out.sort_values(["Channel", "curve"])


def split_component(channel: str) -> tuple[str, str | None]:
    """``"ch1 · H"`` → ``("ch1", "h")``; a plain channel → ``(name, None)``."""
    for suffix, comp in ((H_SUFFIX, "h"), (M_SUFFIX, "m")):
        if channel.endswith(suffix):
            return channel[: -len(suffix)], comp
    return channel, None


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
NORM_NONE = "none"
NORM_BASELINE_MAX = "baseline_max"
NORM_BASELINE_TOPN = "baseline_topn"
NORM_OWN_MAX = "own_max"

NORM_LABELS = {
    NORM_NONE: "без нормировки (µV)",
    NORM_BASELINE_MAX: "на максимум базового состояния",
    NORM_BASELINE_TOPN: "на среднее top-N базового состояния",
    NORM_OWN_MAX: "на собственный максимум прогона",
}


def normalisation_factors(points: pd.DataFrame, baseline_state: str,
                          mode: str = NORM_BASELINE_MAX, top_n: int = 3,
                          metric: str = "amp_uv") -> pd.DataFrame:
    """One scale per (subject, config, channel), measured in that subject's baseline.

    Per subject, never pooled: the absolute µV a muscle produces depends on where
    its electrodes ended up, so one animal's scale says nothing about another's.
    A subject with no baseline run gets no factor and drops out of the normalised
    view — reported rather than silently rescaled by something else.
    """
    if points.empty or mode == NORM_NONE:
        return pd.DataFrame(columns=["subject", "config", "channel", "factor"])

    base = points if mode == NORM_OWN_MAX else points[points["state"] == baseline_state]
    if base.empty:
        return pd.DataFrame(columns=["subject", "config", "channel", "factor"])

    keys = ["subject", "config", "channel"]
    rows = []
    for key, grp in base.groupby(keys, sort=False):
        vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        if mode == NORM_BASELINE_TOPN:
            factor = float(vals.nlargest(min(top_n, len(vals))).mean())
        else:
            factor = float(vals.max())
        if factor > 0:
            rows.append(dict(zip(keys, key)) | {"factor": factor})
    return pd.DataFrame(rows)


def apply_normalisation(points: pd.DataFrame, factors: pd.DataFrame,
                        metric: str = "amp_uv") -> tuple[pd.DataFrame, list[str]]:
    """Add a ``value`` column: *metric* divided by its scale, or the raw metric.

    Returns the table and the list of subjects dropped for having no baseline —
    the caller says so on screen. Latency metrics are never divided: a latency
    is already comparable between recordings, and "% of baseline latency" is not
    a quantity anybody asked for.
    """
    out = points.copy()
    raw = pd.to_numeric(out[metric], errors="coerce")
    if factors.empty or metric.endswith("_ms"):
        out["value"] = raw
        out["unit"] = METRICS.get(metric, metric)
        return out, []
    merged = out.merge(factors, on=["subject", "config", "channel"], how="left")
    out["value"] = raw.to_numpy() / merged["factor"].to_numpy()
    out["unit"] = "доля базового максимума"
    # Per (subject, channel), not per subject: a factor is missing when the
    # baseline session did not record THAT channel on THAT configuration, which
    # in this array is the common case — an animal whose implantation session
    # used other electrode pairs keeps every other state and loses only the
    # scale. Reporting it as "subject has no baseline" would overstate it.
    gone = merged.loc[merged["factor"].isna(), ["subject", "channel"]].drop_duplicates()
    missing = [f"{r.subject}/{r.channel}" for r in gone.itertuples()]
    return out, sorted(missing)


# --------------------------------------------------------------------------- #
# The last curves of the ramp
# --------------------------------------------------------------------------- #
#: How many curves at the end of a recording stand for its maximal response.
LAST_N = 5


# --------------------------------------------------------------------------- #
# Left/right asymmetry on the last curves
# --------------------------------------------------------------------------- #
def asymmetry_last_curves(points: pd.DataFrame, metric: str, n: int = LAST_N,
                          recorded: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Left/right asymmetry of every recording on its last *n* curves.

    Channel pairs and indices are ``src/asymmetry.py``: AI = (R-L)/(R+L) and R/L
    for sizes, R-L in ms for latencies; ch1–4 are taken as left, provisionally.

    Computed on the raw metric, never on a normalised value: normalisation divides
    each channel by its own baseline, which would change the ratio of the sides.
    H-reflex components (``ch1 · M``) are left out — those files record one limb,
    so their channels are not two sides.

    ``recorded`` maps a run to its channels (``asymmetry.recorded_channels``): a
    recorded channel absent from the points had no response at all and counts as
    such. A run missing from it falls back to the channels its points have.

    Returns ``(by_curve, by_recording, kind)`` with the recording's labels on every
    row; ``kind`` is ``"size"`` or ``"latency"``.
    """
    kind = "latency" if metric.endswith("_ms") else "size"
    keys = ["subject", "state", "run", "config"]
    if points.empty or metric not in points.columns:
        return pd.DataFrame(), pd.DataFrame(), kind
    plain = points[points["channel"].map(lambda c: split_component(str(c))[1] is None)]
    curves, summaries = [], []
    for key, grp in plain.groupby(keys, sort=False):
        by_curve, summary = A.recording_asymmetry(grp, {metric: kind}, n,
                                                  recorded=(recorded or {}).get(str(key[2])))
        if summary.empty:
            continue
        labels = dict(zip(keys, key))
        curves.append(by_curve.assign(**labels))
        summaries.append(summary.assign(**labels))
    if not summaries:
        return pd.DataFrame(), pd.DataFrame(), kind
    by_curve = pd.concat(curves, ignore_index=True)
    by_recording = pd.concat(summaries, ignore_index=True)
    first = keys + ["pair", "left", "right", "metric"]
    by_curve = by_curve[first + [c for c in by_curve.columns if c not in first]]
    by_recording = by_recording[first + [c for c in by_recording.columns if c not in first]]
    return by_curve, by_recording, kind


# --------------------------------------------------------------------------- #
# Neurosoft recordings: labels from the recording names
# --------------------------------------------------------------------------- #
#: Where the processed Neurosoft array lives on the lab Drive (two roots).
NEUROSOFT_ROOTS = [
    Path("/Users/dkleeva/Library/CloudStorage/GoogleDrive-lma.lab.fccps@gmail.com/My Drive/"
         "Spinal cord injury/Neurosoft data/Травма спинного мозга"),
    Path("/Users/dkleeva/Library/CloudStorage/GoogleDrive-lma.lab.fccps@gmail.com/My Drive/"
         "Spinal cord injury/Neurosoft data/Контроль"),
]

_LEVEL = re.compile(
    r"(?<![A-Za-zА-Яа-я])(th|t|т|тh|c|с|l|л|д)\s*[-_]?\s*(\d{1,2})\s*[-–—]\s*(th|t|т|c|с|l|л|д)?\s*(\d{1,2})",
    re.I)
_LEFT = re.compile(r"\bлев\w*|\bleft\b|\bL\b|слева|лев\b", re.I)
_RIGHT = re.compile(r"\bправ\w*|\bпр\b|\bright\b|\bR\b|справа", re.I)
_POSITION = re.compile(r"\b(supine|side|upright|prone)\b", re.I)
_POLARITY = re.compile(r"\b(black|red)\b", re.I)
_MUSCLE = {
    "Soleus": re.compile(r"sol\w*|сол\w*", re.I),
    "GM": re.compile(r"\bGM\b|гастр\w*|gastroc\w*", re.I),
    "FCU": re.compile(r"\bFCU\b", re.I),
}
_SCENARIO_WORDS = {
    "Jendrassik": re.compile(r"ендр\w*|jendr\w*|\bJM\b|челюсть|зубы", re.I),
    "Paired stimulation": re.compile(r"2\s*stim|2\s*ст\w*|paired|paried|двойн\w*", re.I),
    "H-reflex": re.compile(r"h[\s-]*reflex|н[\s-]*рефлекс|h\s*refl\w*", re.I),
    "Recruitment": re.compile(r"\bRC\b|rec\s*curve|recr\w*|кр\w*\s*рек\w*|\bKR\b|kr\s*rec", re.I),
}


def _canon_level(m: re.Match) -> str:
    def seg(letter: str | None, default: str) -> str:
        if not letter:
            return default
        letter = letter.lower()
        if letter in ("th", "t", "т", "тh"):
            return "Th"
        if letter in ("c", "с"):
            return "C"
        return "L"
    first = seg(m.group(1), "Th")
    second = seg(m.group(3), first)
    a, b = int(m.group(2)), int(m.group(4))
    if first == "Th" and second == "Th" and a == 12 and b == 1:
        second = "L"                        # "Т12-1" is Th12-L1 with the L dropped
    if second == first:
        return f"{first}{a}-{b}"
    return f"{first}{a}-{second}{b}"


def scenario_of(run_root: Path) -> str | None:
    """Which Neurosoft protocol a run produced, read off its deliverable folder."""
    root = Path(run_root)
    for base in (root / "results", root / "results" / "Stimulation-induced responses", root):
        for folder in ("H-reflex", "Recruitment", "Jendrassik", "Paired stimulation",
                       "Condition test"):
            if (base / folder).is_dir():
                return "Paired stimulation" if folder == "Condition test" else folder
    return None


def parse_neurosoft(run_root: Path) -> dict[str, str]:
    """Tags of one Neurosoft run: subject, cohort, level, scenario, side, muscle, position, polarity.

    The recording names are typed by hand and spelled every way at once
    (``Th11-12``, ``T11-12``, ``т11-12 кр рекр``, ``Т12-Л1 ендр 80мА``), so
    everything here is a tolerant regex and a tag it cannot read is ``—``. The
    scenario is taken from what the pipeline wrote, which is more reliable than
    the name; the name only fills in when no deliverable folder is found.
    """
    root = Path(run_root)
    name = root.name
    path = str(root)
    tags = {"subject": name, "cohort": "SCI", "level": "—", "scenario": "—",
            "side": "—", "muscle": "—", "position": "—", "polarity": "—",
            "folder": folder_group(root)}

    m = _BRACKETED.match(name)
    # The patient code may sit one or two folders up ("(П25)/07092021/07092021 T11-12 kr").
    up = next((_BRACKETED.match(p.name) for p in (root.parent, root.parent.parent)
               if _BRACKETED.match(p.name)), None)
    if m:
        tags["subject"] = m.group(1).strip()
        rest = m.group(2)
    elif up:
        tags["subject"] = up.group(1).strip()
        rest = name
    elif re.match(r"patient\s+\S+", name, re.I):
        pm = re.match(r"(patient\s+\S+)\s*(.*)", name, re.I)
        tags["subject"], rest = pm.group(1), pm.group(2)
    elif "контрол" in path.lower():
        tags["cohort"] = "control"
        rest = re.sub(r"^\s*контроль\s*", "", name, flags=re.I)
        lvl = _LEVEL.search(rest)
        tags["subject"] = (rest[:lvl.start()] if lvl else rest).strip(" _-") or name
        rest = rest[lvl.start():] if lvl else rest
    else:
        rest = name

    lvl = _LEVEL.search(rest)
    if lvl:
        tags["level"] = _canon_level(lvl)
    scen = scenario_of(root)
    if scen is None:
        for key, rx in _SCENARIO_WORDS.items():
            if rx.search(rest):
                scen = key
                break
    tags["scenario"] = scen or "—"
    if _LEFT.search(rest):
        tags["side"] = "left"
    elif _RIGHT.search(rest):
        tags["side"] = "right"
    for muscle, rx in _MUSCLE.items():
        if rx.search(rest):
            tags["muscle"] = muscle
            break
    pos = _POSITION.search(rest)
    if pos:
        tags["position"] = pos.group(1).lower()
    pol = _POLARITY.search(rest)
    if pol:
        tags["polarity"] = pol.group(1).lower()
    return tags


#: What a "state" can be built from, in the order offered on screen.
GROUP_MODES = {
    "level": "уровень стимуляции",
    "cohort": "когорта (SCI / control)",
    "cohort+level": "когорта + уровень",
    "scenario": "сценарий (RC / JM / paired / H)",
    "level+scenario": "уровень + сценарий",
    "side": "сторона",
    "position": "положение тела (контроль)",
    "polarity": "полярность (black / red)",
    "muscle+side": "мышца + сторона (H-рефлекс)",
    "folder": "папка (подпапка добавленной папки)",
    "folder+level": "папка + уровень",
}


def state_from_tags(tags: dict[str, str], mode: str) -> str:
    parts = [tags.get(k, "—") for k in mode.split("+")]
    return " · ".join(parts)


def scan_neurosoft(roots=None, mode: str = "level") -> list["GroupRun"]:
    """Every processed Neurosoft run under the dataset roots, labelled by *mode*."""
    roots = [Path(r) for r in (roots or NEUROSOFT_ROOTS)]
    found: list[GroupRun] = []
    for root in roots:
        if not root.exists():
            continue
        for run in scan_runs(root, max_depth=3):
            tags = parse_neurosoft(run.root)
            tags["folder"] = folder_group(run.root, root)
            found.append(GroupRun(run.root, tags["subject"], state_from_tags(tags, mode),
                                  tags=tags))
    found.sort(key=lambda r: (r.tags.get("cohort", ""), r.subject, r.state))
    return found


# --------------------------------------------------------------------------- #
# What the group tab draws: ramps on one axis, plateau waveforms
# --------------------------------------------------------------------------- #
#: How a recording's ramp is laid on the shared x axis.
ALIGN_END = "end"          # 0 = the last curve, -1 the one before, … (the plateau lines up)
ALIGN_START = "start"      # the curve number as exported


def aligned_ramps(points: pd.DataFrame, align: str = ALIGN_END) -> pd.DataFrame:
    """Add ``x``: each curve's position on the shared axis of the group plot.

    Neurosoft ramps differ in length (20 to 170 curves) and in where they start,
    and the export carries no current. Counted from the END, the last curves — the
    highest intensities, the plateau the "last 5" summary is taken from — line up
    across recordings; counted from the start they line up only if every ramp
    began at the same current, which they did not.
    """
    out = points.copy()
    if out.empty:
        out["x"] = []
        return out
    curve = pd.to_numeric(out["curve"], errors="coerce")
    if align == ALIGN_END:
        last = curve.groupby([out["run"], out["channel"]]).transform("max")
        out["x"] = curve - last
    else:
        out["x"] = curve
    return out


def ramp_bands(points: pd.DataFrame, value: str = "value", group_col: str = "state",
               min_n: int = 3, min_frac: float = 0.3) -> pd.DataFrame:
    """Median and quartiles across recordings at every x, per (group, channel).

    A position is kept only where at least *min_n* recordings and at least *min_frac*
    of the group's recordings on that channel reach: out where only the few longest
    ramps go, the "median" is those few recordings and jumps with each of them.
    """
    d = points.dropna(subset=["x", value])
    if d.empty:
        return pd.DataFrame(columns=[group_col, "channel", "x", "n", "median", "q1", "q3"])
    # one value per recording per position (duplicates would weigh a recording twice)
    d = d.groupby([group_col, "channel", "run", "x"], as_index=False)[value].mean()
    g = d.groupby([group_col, "channel", "x"])[value]
    out = pd.concat({"n": g.count(), "median": g.median(), "q1": g.quantile(0.25),
                     "q3": g.quantile(0.75)}, axis=1).reset_index()
    total = d.groupby([group_col, "channel"])["run"].nunique().rename("_total").reset_index()
    out = out.merge(total, on=[group_col, "channel"], how="left")
    need = np.maximum(int(min_n), np.ceil(float(min_frac) * out["_total"]))
    return out[out["n"] >= need].drop(columns="_total")


def plateau_waveforms(run_root, n: int = LAST_N) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """The mean response of a run's last *n* curves on every channel (µV), and its time axis.

    The waveform counterpart of the "last 5 curves" value: the shape of the response
    at the plateau, one per recording. Read once per run; the tab keeps it.
    """
    from emgflow.results import SIRResults

    res = SIRResults(Path(run_root))
    if not res.ok or not res.configs:
        return None, {}
    times, waves, _ = res.scenario_waves(res.configs[0])
    if times is None or not waves:
        return None, {}
    return (np.asarray(times, dtype=float),
            {ch: np.asarray(w[-n:], dtype=float).mean(axis=0) for ch, w in waves.items() if len(w)})


def waveform_bands(loaded: list[tuple[str, str, np.ndarray, np.ndarray]], n_times: int = 500,
                   min_n: int = 3) -> tuple[np.ndarray, pd.DataFrame, list[tuple[str, str, np.ndarray]]]:
    """Group mean waveform with a 95 % confidence band, per group.

    ``loaded``: ``(group, run, times_s, wave)`` for one channel. The waves are put on
    the intersection of their time windows (Neurosoft exports agree; other datasets
    may not). A group with fewer than *min_n* recordings gets its mean and no band.
    Returns ``(grid, bands, traces)`` with ``traces`` = ``(group, run, wave on grid)``.
    """
    if not loaded:
        return np.array([]), pd.DataFrame(), []
    lo = max(float(t[0]) for _, _, t, _ in loaded)
    hi = min(float(t[-1]) for _, _, t, _ in loaded)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([]), pd.DataFrame(), []
    grid = np.linspace(lo, hi, int(n_times))
    traces = [(g, run, np.interp(grid, t, w)) for g, run, t, w in loaded]
    try:
        from scipy.stats import t as student_t
    except Exception:                                   # noqa: BLE001
        student_t = None
    rows = []
    for group in dict.fromkeys(g for g, _, _ in traces):
        stack = np.vstack([w for g, _, w in traces if g == group])
        k = stack.shape[0]
        mean = stack.mean(axis=0)
        if k >= min_n:
            se = stack.std(axis=0, ddof=1) / np.sqrt(k)
            q = float(student_t.ppf(0.975, k - 1)) if student_t is not None else 1.96
            lo_b, hi_b = mean - q * se, mean + q * se
        else:
            lo_b = hi_b = np.full_like(mean, np.nan)
        rows.append(pd.DataFrame({"group": group, "t": grid, "mean": mean, "lo": lo_b,
                                  "hi": hi_b, "n": k}))
    return grid, pd.concat(rows, ignore_index=True), traces


def plateau_table(points: pd.DataFrame, n: int = LAST_N,
                  value: str = "value") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The last *n* curves of every recording and channel — the same selection as
    ``last_curves``, computed in one pass for the group tab.

    Returns ``(per_recording, curves)``: one row per recording and channel with
    ``first_curve, last_curve, n_values, mean, sd, min, max`` (empty values are not
    averaged; ``n_values`` says how many of the *n* carried a response), and the rows
    of the curves used.
    """
    keys = ["subject", "state", "run", "config", "channel"]
    if points.empty or "curve" not in points.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = points.assign(curve=pd.to_numeric(points["curve"], errors="coerce")).dropna(subset=["curve"])
    d = d.sort_values("curve", kind="stable").drop_duplicates(keys + ["curve"], keep="last")
    tail = d.groupby(keys, sort=False).tail(int(n)).copy()
    tail["_v"] = pd.to_numeric(tail[value], errors="coerce")
    g = tail.groupby(keys, sort=False)
    per = pd.concat({"first_curve": g["curve"].min(), "last_curve": g["curve"].max(),
                     "n_values": g["_v"].count(), "mean": g["_v"].mean(), "sd": g["_v"].std(),
                     "min": g["_v"].min(), "max": g["_v"].max()}, axis=1).reset_index()
    if "unit" in tail.columns:
        per = per.merge(g["unit"].first().reset_index(), on=keys, how="left")
    return per, tail.drop(columns="_v")


# --------------------------------------------------------------------------- #
# Standard stimulation runs (pigs, patients): curves on the current axis
# --------------------------------------------------------------------------- #
#: What a standard run's group can be read from, in the order offered on screen.
SIR_GROUP_MODES = {
    "state": "состояние из имени (control / implantation / N day)",
    "subject": "субъект",
    "run": "запись",
    "folder": "папка (подпапка добавленной папки)",
}


def is_standard_sir_run(run_root) -> bool:
    """A finished SIR run cut per (configuration, amplitude) — not a Neurosoft export."""
    return scenario_of(Path(run_root)) is None


def scan_standard_runs(folder, max_depth: int = 4) -> list[GroupRun]:
    """Every standard SIR run under *folder*, with its labels kept as read from the path."""
    runs = [r for r in scan_runs(Path(folder), max_depth) if is_standard_sir_run(r.root)]
    for r in runs:
        r.tags = {"subject": r.subject, "state": r.state,
                  "folder": folder_group(r.root, Path(folder))}
    return runs


def sir_group_of(run: GroupRun, mode: str) -> str:
    """The group of *run* under grouping *mode* (see ``SIR_GROUP_MODES``)."""
    return {"state": run.tags.get("state", run.state),
            "subject": run.tags.get("subject", run.subject),
            "run": run.root.name,
            "folder": run.tags.get("folder") or folder_group(run.root)}.get(mode, run.state)


def sir_points(res) -> pd.DataFrame:
    """Every configuration's recruitment curve of one amplitude-axis run, in one pass.

    The same numbers as ``SIRResults.recruitment_points`` configuration by
    configuration — the mean over the epochs detected at each amplitude; an
    amplitude where nothing was detected stays as NaN — but from ONE groupby over
    the metrics table instead of one per crop (15 s → about 1 s on the pig array).
    """
    from src.recruitment import amplitude_to_float

    m = res.metrics
    if m.empty:
        return pd.DataFrame()

    def col(name, scale):
        return (pd.to_numeric(m[name], errors="coerce") * scale if name in m.columns
                else pd.Series(np.nan, index=m.index))

    ptp, p1 = col("PTP amplitude", 1e6), col("Peak1 value", 1e6)
    d = pd.DataFrame({
        "config": m["Configuration"].astype(str), "x_label": m["Stim. amplitude"].astype(str),
        "channel": m["Channel"].astype(str), "amp_uv": ptp.where(ptp.notna(), p1.abs()),
        "p1_uv": p1, "p1_ms": col("Peak1 latency", 1e3), "onset_ms": col("Onset latency", 1e3),
        "p2_ms": col("Peak2 latency", 1e3), "area_uvms": col("Response area", 1e9),
    })
    out = d.groupby(["config", "x_label", "channel"], sort=False).agg(
        amp_uv=("amp_uv", "mean"), sd_uv=("amp_uv", "std"), n_epochs=("amp_uv", "count"),
        p1_uv=("p1_uv", "mean"), p1_ms=("p1_ms", "mean"), onset_ms=("onset_ms", "mean"),
        p2_ms=("p2_ms", "mean"), area_uvms=("area_uvms", "mean")).reset_index()
    out["x_value"] = out["x_label"].map(amplitude_to_float)
    position = {(cfg, crop.amp): i for cfg in res.configs
                for i, crop in enumerate(res.config_crops(cfg), start=1)}
    out["curve"] = [position.get((c, a), np.nan) for c, a in zip(out["config"], out["x_label"])]
    return out


def collect_sir_points(runs, progress=None) -> pd.DataFrame:
    """The recruitment points of every standard run, all configurations, one long table.

    Runs whose ramp is only a curve index (Neurosoft exports) are left out: their x
    is not a current. ``progress(i, n)`` is called after each run.
    """
    from emgflow.results import SIRResults
    from src.recruitment import STIM_AXIS_AMPLITUDE

    frames = []
    runs = list(runs)
    for i, run in enumerate(runs, 1):
        try:
            res = SIRResults(Path(run.root))
            if res.ok and res.stim_axis == STIM_AXIS_AMPLITUDE:
                pts = sir_points(res)
                if not pts.empty:
                    pts["subject"], pts["state"], pts["run"] = run.subject, run.state, str(run.root)
                    frames.append(pts)
        except Exception:                               # noqa: BLE001 - that run is skipped
            pass
        if progress is not None:
            progress(i, len(runs))
    if not frames:
        return pd.DataFrame(columns=["subject", "state", "run", "config", "channel", "x_value",
                                     "x_label", "amp_uv"])
    return pd.concat(frames, ignore_index=True)


def curve_summaries(points: pd.DataFrame, value: str = "value",
                    threshold_frac: float = 0.1) -> pd.DataFrame:
    """One row per recording, configuration and channel: what a recruitment curve is compared on.

    * ``max`` — the largest response along the ramp (of *value*). A curve on which
      nothing was detected at any current has ``max`` = 0 and ``responded`` False:
      a silent muscle is a result (spinal shock at 7 days), not a missing value;
    * ``at_max_x`` / ``at_max_label`` — the current of the strongest response (by
      the raw amplitude), and its crop label, to find its epochs;
    * ``at_max_value`` — *value* at that current (for a latency: the latency of the
      strongest response);
    * ``threshold_x`` — the lowest current whose response reaches *threshold_frac*
      of that curve's own maximum; relative, so the same rule works on a 60 µV and a
      1800 µV muscle;
    * ``n_points``, ``x_min``, ``x_max`` — how much of the ramp was measured.
    """
    keys = ["subject", "state", "run", "config", "channel"]
    if points.empty:
        return pd.DataFrame()
    rows = []
    for key, g in points.groupby(keys, sort=False):
        d = g.dropna(subset=["x_value"]).sort_values("x_value")
        if d.empty:
            continue
        x = d["x_value"].to_numpy(float)
        amp = pd.to_numeric(d["amp_uv"], errors="coerce").to_numpy(float)
        val = pd.to_numeric(d[value], errors="coerce").to_numpy(float)
        ok = np.isfinite(amp) & (amp > 0)
        row = dict(zip(keys, key)) | {
            "n_points": int(len(d)), "x_min": float(x.min()), "x_max": float(x.max()),
            "responded": bool(ok.any()), "max": 0.0 if not value.endswith("_ms") else np.nan,
            "at_max_x": np.nan, "at_max_label": None, "at_max_value": np.nan, "threshold_x": np.nan,
        }
        if ok.any():
            i = int(np.argmax(np.where(ok, amp, -np.inf)))
            row["at_max_x"], row["at_max_label"] = float(x[i]), str(d["x_label"].iloc[i])
            row["at_max_value"] = float(val[i]) if np.isfinite(val[i]) else np.nan
            if np.isfinite(val).any():
                row["max"] = float(np.nanmax(val))
            over = np.flatnonzero(ok & (amp >= threshold_frac * amp[i]))
            row["threshold_x"] = float(x[over[0]]) if over.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def current_grid(points: pd.DataFrame, n: int = 60, q: float = 0.02) -> np.ndarray:
    """A shared current axis over the range the selected recordings actually cover.

    Quantiles rather than min/max: one session swept to 145 mA while most stop
    around 14, and on a full-range axis most of the grid would be that one session.
    """
    x = pd.to_numeric(points.get("x_value"), errors="coerce").dropna()
    if x.empty:
        return np.array([])
    lo, hi = float(x.quantile(q)), float(x.quantile(1 - q))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.array([lo])
    return np.linspace(lo, hi, int(n))


def current_bands(points: pd.DataFrame, grid: np.ndarray, value: str = "value",
                  group_col: str = "state", absent_as_zero: bool = True, min_n: int = 3,
                  min_frac: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each recording's curve on the shared current axis, and the group median and quartiles.

    A curve is interpolated inside its own measured range only, never extrapolated.
    With ``absent_as_zero`` (sizes) an amplitude where nothing was detected counts as
    no response; for latencies it is simply not there. A grid current is kept where at
    least *min_n* recordings and *min_frac* of the group's recordings on that channel
    were measured. Returns ``(on_grid, bands)``.
    """
    if points.empty or grid.size == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (group, channel, run), g in points.groupby([group_col, "channel", "run"], sort=False):
        d = g[["x_value", value]].copy()
        d[value] = pd.to_numeric(d[value], errors="coerce")
        d = d.dropna(subset=["x_value"])
        if absent_as_zero:
            d[value] = d[value].fillna(0.0)
        d = d.dropna(subset=[value]).groupby("x_value", as_index=False)[value].mean()
        if len(d) < 2:
            continue
        x, y = d["x_value"].to_numpy(float), d[value].to_numpy(float)
        yi = np.interp(grid, x, y)
        yi[(grid < x[0]) | (grid > x[-1])] = np.nan
        rows.append(pd.DataFrame({group_col: group, "channel": channel, "run": run, "x": grid,
                                  "y": yi}))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    on_grid = pd.concat(rows, ignore_index=True)
    total = on_grid.groupby([group_col, "channel"])["run"].nunique().rename("_total")
    valid = on_grid.dropna(subset=["y"])
    g = valid.groupby([group_col, "channel", "x"])["y"]
    bands = pd.concat({"n": g.count(), "median": g.median(), "q1": g.quantile(0.25),
                       "q3": g.quantile(0.75)}, axis=1).reset_index()
    bands = bands.merge(total.reset_index(), on=[group_col, "channel"], how="left")
    need = np.maximum(int(min_n), np.ceil(float(min_frac) * bands["_total"]))
    return on_grid, bands[bands["n"] >= need].drop(columns="_total")


def peak_waveforms(run_root, config: str, labels: dict[str, str]) -> tuple[np.ndarray | None, dict]:
    """The mean epoch (µV) of every channel's strongest crop in one configuration.

    ``labels`` maps a channel to the amplitude label of its strongest response
    (``curve_summaries``' ``at_max_label``). Each crop is read once.
    """
    from emgflow.results import SIRResults

    res = SIRResults(Path(run_root))
    crops = {c.amp: c for c in res.config_crops(config)}
    cache: dict[str, tuple] = {}
    times, out = None, {}
    for channel, label in labels.items():
        crop = crops.get(str(label))
        if crop is None:
            continue
        if crop.amp not in cache:
            ep = res.load_epochs(crop)
            cache[crop.amp] = (np.asarray(ep.times, float),
                               {n: i for i, n in enumerate(ep.ch_names)},
                               ep.get_data().mean(axis=0) * 1e6)
        t, index, mean = cache[crop.amp]
        if channel in index:
            out[channel] = mean[index[channel]]
            times = t
    return times, out


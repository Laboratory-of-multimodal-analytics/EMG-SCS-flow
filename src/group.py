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


def inventory(points: pd.DataFrame) -> pd.DataFrame:
    """What each (config, channel) is available in — how many subjects and states.

    The thing to read before choosing anything: in the pig array only one
    configuration is present in nearly every session, and a group built on a
    configuration two animals share is a group of two.
    """
    if points.empty:
        return pd.DataFrame()
    g = points.groupby(["config", "channel"])
    return pd.DataFrame({
        "subjects": g["subject"].nunique(),
        "states": g["state"].nunique(),
        "runs": g["run"].nunique(),
        "points": g["x_value"].count(),
    }).reset_index().sort_values(["subjects", "runs"], ascending=False)


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
# Averaging over a common current axis
# --------------------------------------------------------------------------- #
def common_grid(points: pd.DataFrame, n: int = 40, q: float = 0.02) -> np.ndarray:
    """A shared current axis over the range the selected runs actually cover.

    Quantiles rather than min/max: one implantation session swept to 45 mA while
    most sessions stop around 14, and on a full-range axis four fifths of the
    grid is a single animal — which reads as a group mean and is not one.
    """
    x = pd.to_numeric(points.get("x_value"), errors="coerce").dropna()
    if x.empty:
        return np.array([])
    lo, hi = float(x.quantile(q)), float(x.quantile(1 - q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo]) if np.isfinite(lo) else np.array([])
    return np.linspace(lo, hi, int(n))


def interpolate_to_grid(points: pd.DataFrame, grid: np.ndarray,
                        value: str = "value") -> pd.DataFrame:
    """Each run's curve on the shared axis, inside its own measured range only.

    The amplitude sets never coincide between recordings — one session sweeps
    0.1 to 45 mA in 55 steps, the next 4 to 12 in eight — so a group mean needs
    the curves on one axis before it can be taken at all.

    Never extrapolated: outside a curve's own range the grid gets NaN, so a
    subject that stopped at 8 mA contributes nothing above 8 rather than a flat
    continuation, and the n behind every grid point says how many subjects were
    actually measured there.
    """
    if points.empty or grid.size == 0:
        return pd.DataFrame()
    rows = []
    keys = ["subject", "state", "config", "channel"]
    for key, grp in points.groupby(keys, sort=False):
        d = grp[["x_value", value]].dropna()
        d = d.groupby("x_value", as_index=False)[value].mean().sort_values("x_value")
        if len(d) < 2:
            continue
        x = d["x_value"].to_numpy(float)
        y = d[value].to_numpy(float)
        yi = np.interp(grid, x, y)
        yi[(grid < x[0]) | (grid > x[-1])] = np.nan
        frame = pd.DataFrame({"x": grid, value: yi})
        for k, v in zip(keys, key):
            frame[k] = v
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def group_mean(on_grid: pd.DataFrame, value: str = "value",
               min_n: int = 2) -> pd.DataFrame:
    """Mean ± SE across subjects at each grid point, per (state, config, channel).

    Grid points backed by fewer than *min_n* subjects are dropped. Above and
    below the currents most sessions covered there is usually exactly one animal
    left, and drawing its curve as the group's — with no error band, since a
    single value has no spread — is the most misleading thing this view could do.
    """
    if on_grid.empty:
        return pd.DataFrame()
    g = on_grid.groupby(["state", "config", "channel", "x"])[value]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out = out[out["n"] >= int(min_n)]
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out


# --------------------------------------------------------------------------- #
# One number per curve, and contrasts between states
# --------------------------------------------------------------------------- #
SCALARS = {
    "max": "максимум отклика",
    "auc": "площадь под кривой",
    "threshold": "порог (мА)",
    "at_max_x": "ток максимума (мА)",
}


def curve_scalars(points: pd.DataFrame, value: str = "value",
                  threshold_frac: float = 0.1) -> pd.DataFrame:
    """Collapse each subject's curve to the numbers a group is compared on.

    * ``max`` — the plateau the muscle reaches;
    * ``at_max_x`` — the current it reaches it at;
    * ``auc`` — the area, which moves with both size and threshold;
    * ``threshold`` — the first current whose response clears *threshold_frac*
      of that curve's own maximum. Relative to the curve rather than to an
      absolute µV, because the same criterion has to work on a channel whose
      responses are 60 µV and one whose are 1800.
    """
    if points.empty:
        return pd.DataFrame()
    rows = []
    for key, grp in points.groupby(["subject", "state", "config", "channel"], sort=False):
        d = grp[["x_value", value]].dropna().sort_values("x_value")
        if d.empty:
            continue
        x = d["x_value"].to_numpy(float)
        y = d[value].to_numpy(float)
        peak = float(np.nanmax(y))
        over = np.flatnonzero(y >= threshold_frac * peak)
        rows.append({
            "subject": key[0], "state": key[1], "config": key[2], "channel": key[3],
            "max": peak,
            "at_max_x": float(x[int(np.nanargmax(y))]),
            "auc": float(np.trapezoid(y, x)) if len(x) > 1 else np.nan,
            "threshold": float(x[over[0]]) if over.size else np.nan,
            "n_points": int(len(x)),
        })
    return pd.DataFrame(rows)


def contrast(scalars: pd.DataFrame, state_a: str, state_b: str,
             scalar: str = "max") -> pd.DataFrame:
    """Paired comparison of *state_b* against *state_a*, per (config, channel).

    Paired inside the subject: the pairing is the point of a longitudinal design,
    and the between-animal spread is far larger than the change being looked for.
    Wilcoxon signed-rank rather than a t-test — a dozen animals, and response
    sizes are not symmetric around their mean.

    ``subjects`` names exactly who was in the pair. It is the column to read
    first: an effect carried by three animals out of thirteen is a different
    statement from one carried by all of them, and only this says which it is.

    One caveat this cannot fix, only declare: under normalisation to the baseline
    maximum, every baseline curve's ``max`` is 1 by construction. A contrast
    against the baseline state is then a one-sample test of "is the ratio
    different from 1" wearing a paired test's clothes. The p-value is the right
    one for that question, but the baseline column is not a measurement and must
    not be read as one. ``baseline_is_constant`` in the result says when this
    applies.
    """
    if scalars.empty:
        return pd.DataFrame()
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = []
    for (config, channel), grp in scalars.groupby(["config", "channel"], sort=False):
        a = grp[grp["state"] == state_a].set_index("subject")[scalar]
        b = grp[grp["state"] == state_b].set_index("subject")[scalar]
        both = a.index.intersection(b.index)
        pair = pd.DataFrame({"a": a.reindex(both), "b": b.reindex(both)}).dropna()
        if pair.empty:
            continue
        diff = pair["b"] - pair["a"]
        p = np.nan
        # Wilcoxon needs at least one non-zero difference and is not meaningful
        # on a handful of pairs; the n column is what says whether to believe it.
        if wilcoxon is not None and len(pair) >= 5 and float(np.abs(diff).sum()) > 0:
            try:
                p = float(wilcoxon(pair["a"], pair["b"]).pvalue)
            except ValueError:
                p = np.nan
        rows.append({
            "config": config, "channel": channel, "scalar": scalar,
            "n_pairs": int(len(pair)),
            f"{state_a} median": float(pair["a"].median()),
            f"{state_b} median": float(pair["b"].median()),
            "median Δ": float(diff.median()),
            "ratio": float(pair["b"].median() / pair["a"].median())
            if pair["a"].median() else np.nan,
            "p (Wilcoxon)": p,
            "baseline_is_constant": bool(pair["a"].nunique() == 1),
            "subjects": ", ".join(map(str, pair.index)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Waveforms across runs
# --------------------------------------------------------------------------- #
def collect_waveforms(runs, config: str | None, channel: str, at_x: float | None = None,
                      n_times: int = 400) -> tuple[np.ndarray, list[dict]]:
    """Mean response shapes from several runs on one time axis.

    Sampling rates differ across the datasets that end up in one group — 4 kHz
    for the pig recordings, 20 kHz for the Neurosoft exports — and so do the
    epoch windows: a Neurosoft curve starts AT the stimulus and has no
    pre-stimulus data at all. So the shared axis is the INTERSECTION of the
    windows, resampled onto one grid. Anything outside every run's own window is
    not drawn rather than padded.

    *at_x* picks the amplitude to show: the point nearest that current in each
    run. None takes each run's strongest response, which is what "show me what
    this muscle can do" means.
    """
    from emgflow.results import SIRResults

    loaded = []
    for run in runs:
        if not run.include:
            continue
        try:
            res = SIRResults(Path(run.root))
            if not res.ok:
                continue
            # None = the run's own configuration: a Neurosoft export has one.
            cfg = config if config is not None else res.configs[0]
            if cfg not in res.configs:
                continue
            times, waves, labels = res.scenario_waves(cfg)
            raw_channel, comp = split_component(channel)
            if times is None or raw_channel not in waves:
                continue
            if comp is not None:
                # An H/M channel exists only on H-reflex recordings; on any
                # other recording it has no points and the run is skipped.
                pts = hreflex_points(res, cfg) if res.scenario == "H-reflex" \
                    else pd.DataFrame()
            else:
                pts = res.recruitment_points(cfg)
            pts = pts[pts["channel" if "channel" in pts else "Channel"] == channel] \
                if not pts.empty else pts
            if comp is not None and pts.empty:
                continue
        except Exception:
            continue
        stack = waves[raw_channel]
        idx, shown = 0, ""
        if not pts.empty:
            xs = pd.to_numeric(pts["x_value"], errors="coerce").to_numpy(float)
            ys = pd.to_numeric(pts["amp_uv"], errors="coerce").to_numpy(float)
            order = pts["curve"].to_numpy(int) - 1
            if at_x is None:
                if np.isfinite(ys).any():
                    idx = int(order[int(np.nanargmax(ys))])
                    shown = str(pts["x_label"].iloc[int(np.nanargmax(ys))])
            elif np.isfinite(xs).any():
                j = int(np.nanargmin(np.abs(xs - at_x)))
                idx = int(order[j])
                shown = str(pts["x_label"].iloc[j])
        if not 0 <= idx < len(stack):
            idx = 0
        loaded.append({"run": run, "times": np.asarray(times, float),
                       "wave": np.asarray(stack[idx], float), "at": shown})
    if not loaded:
        return np.array([]), []

    lo = max(float(d["times"][0]) for d in loaded)
    hi = min(float(d["times"][-1]) for d in loaded)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), []
    grid = np.linspace(lo, hi, int(n_times))
    for d in loaded:
        d["wave"] = np.interp(grid, d["times"], d["wave"])
        d.pop("times")
    return grid, loaded


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
            "side": "—", "muscle": "—", "position": "—", "polarity": "—"}

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
            found.append(GroupRun(run.root, tags["subject"], state_from_tags(tags, mode),
                                  tags=tags))
    found.sort(key=lambda r: (r.tags.get("cohort", ""), r.subject, r.state))
    return found


def waveform_summary(times: np.ndarray, loaded: list[dict]) -> pd.DataFrame:
    """Mean and 95 % confidence band of the waveforms per state.

    Across runs, not across the epochs of one run: the band says how much the
    response shape varies between recordings of that state. The t-based interval
    needs at least two runs; a lone run gets its own trace and no band.
    """
    if not len(times) or not loaded:
        return pd.DataFrame()
    try:
        from scipy.stats import t as student_t
    except Exception:
        student_t = None
    rows = []
    by_state: dict[str, list[np.ndarray]] = {}
    for d in loaded:
        by_state.setdefault(d["run"].state, []).append(np.asarray(d["wave"], float))
    for state, waves in by_state.items():
        stack = np.vstack(waves)
        n = stack.shape[0]
        mean = np.nanmean(stack, axis=0)
        # Two recordings give one degree of freedom and a t of 12.7: the band
        # would be wider than the plot. Three is the least that says anything.
        if n >= 3:
            se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)
            q = float(student_t.ppf(0.975, n - 1)) if student_t is not None else 1.96
            lo, hi = mean - q * se, mean + q * se
        else:
            lo = hi = mean
        rows.append(pd.DataFrame({"state": state, "t": times, "mean": mean,
                                  "lo": lo, "hi": hi, "n": n}))
    return pd.concat(rows, ignore_index=True)

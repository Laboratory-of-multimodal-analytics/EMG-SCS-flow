"""Curve-group deliverable: Jendrassik manoeuvre and paired stimulation.

A. Militskova asks for the same two things on both protocols: the CURVES
themselves — every curve drawn, not one average over them — and the curves
binned into groups of similar amplitude with a mean and a spread per group.

No recruitment curve is drawn here, by request: a Jendrassik file is one or two
stimulation intensities, at each of which the patient first receives a run of
plain test stimuli and then repeats them while performing the manoeuvre. There
is no intensity ramp to plot against — what varies is whether the response was
facilitated, and the curves fall into a small number of response-amplitude
groups (two, in A. Militskova's own example channel). Those groups, and the
per-curve peak-to-peak behind them, are the deliverable.

Paired stimulation lands here whenever the export does not encode the
inter-stimulus interval. When it DOES (the artifact moves from curve to curve,
as in the TMS-conditioning files) the run goes to src/condition.py instead,
which can group by real ISI.

Runs off the SIR metrics CSV (per-curve P1/P2/PTP) plus the saved epochs, so it
is independent of the detection code and can be re-run on a finished results
folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import TEXT_CURVES_RESP_TMIN
from .io_utils import ensure_dir
from .neurosoft import JENDRASSIK, PAIRED
from .recruitment import (
    channel_group_labels,
    drop_legacy_excel_dir,
    shared_excel_dir,
    _assign_amplitude_groups,
    _grid,
    _metrics_csv,
    _sir_dir,
    _stats,
    _tidy_metrics,
)

#: Default number of response-amplitude groups. Two, not three: the protocol is
#: a baseline block and a manoeuvre block at the same intensity, and that is the
#: split the clinician reads off these files. Override per call if a file was run
#: at two intensities and you want them separated as well.
CURVE_GROUPS = 2

#: Output folder and figure wording per scenario.
_SCENARIO_STYLE = {
    JENDRASSIK: {"folder": "Jendrassik", "tag": "приём Ендрассика", "log": "JENDRASSIK"},
    PAIRED: {"folder": "Paired stimulation", "tag": "парная стимуляция", "log": "PAIRED"},
}

#: One colour per amplitude group, low -> high.
_GROUP_COLORS = ["#4575b4", "#f0a202", "#d73027", "#4d9221", "#7b3294"]


def _curve_span(curves) -> str:
    """Compact "1-8, 15" style summary of which curves fell in a group.

    Worth reading: in a Jendrassik file the groups should line up with the two
    blocks in time (test stimuli first, manoeuvre second), so a group whose
    curves are scattered across the whole run is a sign the split is amplitude
    noise rather than the protocol.
    """
    c = sorted(int(x) for x in curves)
    if not c:
        return ""
    spans, start, prev = [], c[0], c[0]
    for v in c[1:]:
        if v == prev + 1:
            prev = v
            continue
        spans.append((start, prev))
        start = prev = v
    spans.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def _group_colors(n_groups: int) -> list[str]:
    if n_groups <= len(_GROUP_COLORS):
        return _GROUP_COLORS[:n_groups]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]


def _load_epoch_waveforms(output_root: Path):
    """(times_s, {channel: (n_curves, n_samples) µV}) from the saved epochs.

    Returns ``(None, {})`` when no epochs file was written.
    """
    import mne

    epochs_dir = _sir_dir(output_root) / "Stimulus-centered epochs"
    files = sorted(epochs_dir.glob("*-epo.fif")) if epochs_dir.exists() else []
    if not files:
        return None, {}
    ep = mne.read_epochs(files[0], preload=True, verbose=False)
    data = ep.get_data() * 1e6          # volts -> µV
    return ep.times, {ch: data[:, i, :] for i, ch in enumerate(ep.ch_names)}


def run_curve_group_analysis(
    output_root: Path,
    scenario: str = JENDRASSIK,
    n_groups: int | None = None,
) -> Path | None:
    """Build the curve/amplitude-group tables and figures from a finished SIR run.

    ``n_groups=None`` lets each channel's own amplitudes decide how many groups
    it has. That is right for paired stimulation, where nothing fixes the number.
    The Jendrassik protocol does fix it — a block without the manoeuvre and a
    block with it, per intensity — so its caller passes the count; falling back
    to the data there would let a channel whose two blocks happen to overlap
    collapse into one group and lose the comparison the test is for.

    Returns the output directory, or None when there is nothing to report.
    """
    if n_groups is None and scenario == JENDRASSIK:
        n_groups = _jendrassik_groups_from_name(output_root)
    style = _SCENARIO_STYLE.get(scenario, _SCENARIO_STYLE[JENDRASSIK])
    log = style["log"]
    output_root = Path(output_root)
    csv = _metrics_csv(output_root)
    if not csv.exists():
        print(f"[{log}] No SIR metrics CSV found; skipping.", flush=True)
        return None

    tidy, responders = _tidy_metrics(csv)
    if tidy is None:
        print(f"[{log}] No responding channels; skipping.", flush=True)
        return None

    out_dir = ensure_dir(_sir_dir(output_root) / style["folder"])
    excel_dir = ensure_dir(shared_excel_dir(output_root))

    tidy = _assign_amplitude_groups(tidy, responders, n_groups)
    n_seen = max((len(channel_group_labels(tidy, ch)) for ch in responders), default=1)
    labels = [f"G{i + 1}" for i in range(n_seen)]
    colors = _group_colors(n_seen)

    # ── per-curve table (with the group each curve landed in) ──
    out = tidy.copy()
    for c in ["Onset ms", "P1 ms", "P2 ms", "P1 uV", "P2 uV", "PTP uV", "Amplitude uV"]:
        out[c] = out[c].round(3)
    out.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "curves_by_amplitude_group_long.csv", index=False)

    # ── per-group statistics: the means and spreads that were asked for ──
    # Only metrics that actually carry values: on these files P2, PTP and the
    # onset go undetected on most channels (the responses are monophasic), and
    # an all-NaN row is noise in a table meant to be read.
    metrics = [m for m in ["Amplitude uV", "PTP uV", "P1 uV", "P2 uV", "P1 ms", "Onset ms"]
               if tidy[m].notna().any()]
    rows = []
    for ch in responders:
        d = tidy[tidy["Channel"] == ch]
        for grp in labels:
            dg = d[d["Amplitude group"] == grp]
            if dg.empty:
                continue
            for metric in metrics:
                rows.append({
                    "Channel": ch, "Amplitude group": grp,
                    "Curves": _curve_span(dg["Curve"]),
                    "Metric": metric, **_stats(dg[metric]),
                })
    stats = pd.DataFrame(rows)
    # A row with N=0 is a metric that was never detected on that channel (P2 and
    # PTP go undetected wherever the response is monophasic). It states nothing,
    # so it is dropped rather than left as a line of NaNs to scroll past.
    if not stats.empty:
        stats = stats[stats["N"] > 0]
    stats.to_csv(excel_dir / "stats_amplitude_groups.csv", index=False)

    # ── the curves themselves: all channels on one figure ──
    times, waves = _load_epoch_waveforms(output_root)
    drawn = [ch for ch in responders if ch in waves]
    if times is not None and drawn:
        _plot_curves_by_group(times, waves, tidy, drawn, labels, colors,
                              out_dir / "curves_by_amplitude_group.png", style["tag"])
    else:
        print(f"[{log}] No saved epochs found — per-curve figures skipped.", flush=True)

    _plot_group_boxplots(tidy, responders, labels, colors,
                         out_dir / "amplitude_by_group_boxplots.png")

    drop_legacy_excel_dir(out_dir)
    per_ch = sorted({len(channel_group_labels(tidy, ch)) for ch in responders})
    how = (f"{n_groups} amplitude groups" if n_groups is not None
           else f"amplitude groups from the data ({'/'.join(map(str, per_ch))} per channel)")
    print(f"[{log}] {len(responders)} channels, {tidy['Curve'].nunique()} curves, "
          f"{how} -> {out_dir}; tables -> {excel_dir}", flush=True)
    return out_dir


def _jendrassik_groups_from_name(output_root: Path) -> int:
    """Two groups per intensity named in the recording's file name."""
    from .neurosoft import intensities_from_name
    from .pipeline import input_from_run_manifest

    src = input_from_run_manifest(output_root)
    n = len(intensities_from_name(src)) if src is not None else 0
    return 2 * max(n, 1)


def run_jendrassik_analysis(output_root: Path, n_groups: int | None = None):
    """Backwards-compatible entry point for the Jendrassik scenario."""
    return run_curve_group_analysis(output_root, JENDRASSIK, n_groups)


def _plot_curves_by_group(times, waves, tidy, responders, labels, colors, out_path,
                          tag="приём Ендрассика"):
    """All channels on ONE figure: every curve drawn, coloured by its amplitude
    group, with each group's mean (thick) and ±SD band on top.

    One subplot per channel rather than one file per channel — the groups are
    read by comparing them, and comparing them across channels means having them
    side by side.

    Each channel keeps its own y-scale (amplitudes differ several-fold between
    muscles) and that scale is fitted to the post-artifact data: the stimulus
    artifact is two orders of magnitude larger than the response and would
    otherwise flatten every curve onto the zero line.
    """
    times = np.asarray(times)
    t_ms = times * 1e3
    post = times >= TEXT_CURVES_RESP_TMIN

    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch_name in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        ch_waves = waves[ch_name]
        # Curve numbers are 1-based in the tables, rows of ch_waves are 0-based.
        ch_tidy = tidy[tidy["Channel"] == ch_name]
        grp_of = dict(zip(ch_tidy["Curve"], ch_tidy["Amplitude group"]))

        for gi, grp in enumerate(labels):
            rows = [c - 1 for c, g in grp_of.items()
                    if g == grp and 0 < c <= len(ch_waves)]
            if not rows:
                continue
            block = ch_waves[sorted(rows)]
            for w in block:
                ax.plot(t_ms, w, color=colors[gi], lw=0.5, alpha=0.30)
            m = block.mean(axis=0)
            sd = block.std(axis=0, ddof=1) if len(block) > 1 else np.zeros_like(m)
            ax.fill_between(t_ms, m - sd, m + sd, color=colors[gi], alpha=0.18, lw=0)
            ax.plot(t_ms, m, color=colors[gi], lw=1.8,
                    label=f"{grp} (n={len(block)})")

        ax.axhline(0, color="0.7", lw=0.6)
        seg = ch_waves[:, post]
        if seg.size and np.isfinite(seg).any():
            lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(ch_name, fontsize=10, loc="left")
        ax.set_ylabel("мкВ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, title="группа (низкая→высокая)", title_fontsize=7)

    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("мс от стимула")
    fig.suptitle(f"Кривые по группам амплитуды, среднее ± SD ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_group_boxplots(tidy, responders, labels, colors, out_path):
    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch]
        # A channel keeps only the groups its own amplitudes produced.
        labels = channel_group_labels(tidy, ch) or labels
        data = [d.loc[d["Amplitude group"] == g, "Amplitude uV"].dropna().values
                for g in labels]
        bp = ax.boxplot(data, positions=np.arange(len(labels)), widths=0.6,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="k"))
        for box, col in zip(bp["boxes"], colors):
            box.set_facecolor(col)
            box.set_alpha(0.35)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ch, fontsize=10, loc="left")
        ax.set_ylabel("Амплитуда, мкВ")
        ax.grid(alpha=0.3, axis="y")
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Амплитуда ответа по группам (низкая→высокая)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

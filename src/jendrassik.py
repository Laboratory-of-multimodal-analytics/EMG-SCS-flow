"""Curve deliverables for the Jendrassik manoeuvre and for paired stimulation.

A. Militskova asks for the CURVES themselves on both protocols — every curve
drawn, not one average over them — with the curves that belong together summarised
by a mean and a spread. What "belong together" means differs:

* **Jendrassik** (reworked after the meeting of 13 September 2026). The run is a
  few stimuli without the manoeuvre, a few with it, then another intensity; block
  lengths depend on the patient and nothing in the export records them. Curves
  that belong together are similar in amplitude AND next to each other in the
  run, so the run is cut into contiguous blocks (``src/curve_blocks.py``). Each
  block gets its spread, and each channel her verdict: an SD above 30 µV means the
  manoeuvre works; a run that starts silent and then holds a steady level is a
  change of intensity, not an effect of the manoeuvre.
* **Paired stimulation** (unchanged). The curves binned into groups of similar
  amplitude, as many as each channel's own levels show.

No recruitment curve is drawn here, by request: there is no intensity ramp to plot
against.

Paired stimulation lands here whenever the export does not encode the
inter-stimulus interval. When it DOES (the artifact moves from curve to curve, as
in the TMS-conditioning files) the run goes to src/condition.py instead, which can
group by real ISI.

Runs off the SIR metrics CSV (per-curve P1/P2/PTP) plus the saved epochs, so it is
independent of the detection code and can be re-run on a finished results folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import curve_blocks as CB
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

#: Output folder and figure wording per scenario.
_SCENARIO_STYLE = {
    JENDRASSIK: {"folder": "Jendrassik", "tag": "приём Ендрассика", "log": "JENDRASSIK"},
    PAIRED: {"folder": "Paired stimulation", "tag": "парная стимуляция", "log": "PAIRED"},
}

#: One colour per amplitude group, low -> high (paired stimulation).
_GROUP_COLORS = ["#4575b4", "#f0a202", "#d73027", "#4d9221", "#7b3294"]

#: File names of the block deliverable. The Jendrassik scenario and the H-reflex
#: files that were also Jendrassik runs write the same set under their own names;
#: ``legacy_*`` are what the amplitude-group version wrote, removed on a re-run so
#: they cannot sit beside the blocks reporting a grouping that is no longer used.
JENDRASSIK_BLOCK_NAMES = {
    "long": "jendrassik_curves_by_block_long.csv",
    "stats": "jendrassik_block_stats.csv",
    "blocks": "jendrassik_blocks.csv",
    "verdict": "jendrassik_verdict.csv",
    "curves_png": "curves_by_block.png",
    "amp_png": "amplitude_by_block.png",
    "legacy_tables": ("curves_by_amplitude_group_long.csv", "stats_amplitude_groups.csv"),
    "legacy_figures": ("curves_by_amplitude_group.png", "amplitude_by_group_boxplots.png"),
}
HREFLEX_BLOCK_NAMES = {
    "long": "hreflex_jendrassik_by_curve_long.csv",
    "stats": "hreflex_jendrassik_block_stats.csv",
    "blocks": "hreflex_jendrassik_blocks.csv",
    "verdict": "hreflex_jendrassik_verdict.csv",
    "curves_png": "jendrassik_curves_by_H_block.png",
    "amp_png": "jendrassik_H_amplitude_by_block.png",
    "legacy_tables": ("hreflex_jendrassik_group_stats.csv",),
    "legacy_figures": ("jendrassik_curves_by_H_group.png",
                       "jendrassik_H_amplitude_by_group_boxplots.png"),
}


def _curve_span(curves) -> str:
    """Compact "1-8, 15" style summary of which curves fell in a group."""
    return CB.curve_span(curves)


def _group_colors(n_groups: int) -> list[str]:
    if n_groups <= len(_GROUP_COLORS):
        return _GROUP_COLORS[:n_groups]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]


def block_colors(n_blocks: int) -> list:
    """One colour per block number (B1, B2, …); the GUI uses the same palette."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(max(n_blocks, 1))]


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
    """Build the curve tables and figures of a finished Jendrassik or paired run.

    Jendrassik goes to ``run_jendrassik_blocks``. For paired stimulation
    ``n_groups=None`` lets each channel's own amplitudes decide how many groups it
    has — nothing about that protocol fixes the number.

    Returns the output directory, or None when there is nothing to report.
    """
    if scenario == JENDRASSIK:
        return run_jendrassik_blocks(output_root)
    style = _SCENARIO_STYLE.get(scenario, _SCENARIO_STYLE[PAIRED])
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
    for c in ["Onset ms", "P1 ms", "P2 ms", "P1 uV", "P2 uV", "PTP uV", "Amplitude uV",
              "Area uV·ms"]:
        if c in out.columns:
            out[c] = out[c].round(3)
    out.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "curves_by_amplitude_group_long.csv", index=False)

    # ── per-group statistics: the means and spreads that were asked for ──
    # Only metrics that actually carry values: on these files P2, PTP and the
    # onset go undetected on most channels (the responses are monophasic), and
    # an all-NaN row is noise in a table meant to be read.
    metrics = _reported_metrics(tidy)
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


def _reported_metrics(tidy: pd.DataFrame, extra=()) -> list[str]:
    return [m for m in ["Amplitude uV", "Area uV·ms", "PTP uV", "P1 uV", "P2 uV", "P1 ms",
                        "Onset ms", *extra]
            if m in tidy.columns and tidy[m].notna().any()]


# --------------------------------------------------------------------------- #
# Jendrassik: contiguous blocks
# --------------------------------------------------------------------------- #
@dataclass
class BlockResult:
    tidy: pd.DataFrame          # per curve, with "Block" and "Block kind"
    stats: pd.DataFrame         # per channel x block x metric
    blocks: pd.DataFrame        # per channel x block: the spread and the step between blocks
    verdict: pd.DataFrame       # per channel
    blocks_of: dict             # channel -> list[curve_blocks.Block]


def _r(x) -> float:
    return round(float(x), 3) if x is not None and np.isfinite(x) else np.nan


def block_analysis(tidy: pd.DataFrame, responders: list[str], metrics: list[str]) -> BlockResult:
    """Cut every channel's run into contiguous blocks on ``Amplitude uV`` and summarise them."""
    tidy = tidy.copy()
    tidy["Block"] = pd.Series(index=tidy.index, dtype=object)
    tidy["Block kind"] = pd.Series(index=tidy.index, dtype=object)
    thr = CB.SPREAD_THRESHOLD_UV
    stats_rows, block_rows, verdict_rows, blocks_of = [], [], [], {}
    for ch in responders:
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        blocks = CB.find_blocks(d["Amplitude uV"].to_numpy(float), d["Curve"].to_numpy(int))
        blocks_of[ch] = blocks
        kinds = [None] * len(d)
        for b in blocks:
            kinds[b.start:b.stop] = [b.kind] * (b.stop - b.start)
        tidy.loc[d.index, "Block"] = CB.labels_per_curve(blocks, len(d))
        tidy.loc[d.index, "Block kind"] = kinds
        verdict = CB.verdict(blocks)

        previous = np.nan
        for b in blocks:
            span = CB.curve_span(b.curves)
            responding = b.kind == CB.RESPONSE and b.n_response > 0
            block_rows.append({
                "Channel": ch, "Block": b.label, "Kind": b.kind, "Curves": span,
                "N curves": b.stop - b.start, "N responses": b.n_response,
                "Amplitude mean uV": _r(b.mean), "Amplitude SD uV": _r(b.sd),
                "Amplitude CV": _r(b.cv),
                "Range uV": _r(b.values.max() - b.values.min()) if b.n_response else np.nan,
                "Step from previous block uV": _r(b.mean - previous) if responding else np.nan,
                f"SD > {thr:g} uV": ("yes" if b.spread_above else "no") if b.n_response > 1 else "",
                "Channel verdict": verdict,
            })
            if responding:
                previous = b.mean
            if not responding:
                stats_rows.append({"Channel": ch, "Block": b.label, "Kind": b.kind, "Curves": span,
                                   "Metric": "Amplitude uV", "N": 0, "mean": np.nan, "SD": np.nan,
                                   "median": np.nan, "min": np.nan, "max": np.nan})
                continue
            rows_d = d.iloc[b.start:b.stop]
            for metric in metrics:
                stats_rows.append({"Channel": ch, "Block": b.label, "Kind": b.kind, "Curves": span,
                                   "Metric": metric, **_stats(rows_d[metric])})

        with_sd = [b.sd for b in blocks if b.kind == CB.RESPONSE and b.n_response > 1]
        verdict_rows.append({
            "Channel": ch, "Verdict": verdict, "Verdict (ru)": CB.VERDICT_RU[verdict],
            "Blocks": " | ".join(
                f"{CB.curve_span(b.curves)}: "
                + ("no response" if b.kind == CB.SILENT or not b.n_response
                   else f"{b.mean:.0f} ± {b.sd:.0f} uV" if b.n_response > 1 else f"{b.mean:.0f} uV")
                for b in blocks),
            "Max block SD uV": _r(max(with_sd)) if with_sd else np.nan,
            "Threshold SD uV": thr,
        })

    stats = pd.DataFrame(stats_rows)
    if not stats.empty:
        # an all-NaN metric of a responding block says nothing; a silent block's row says "silent"
        stats = stats[(stats["N"] > 0) | (stats["Kind"] == CB.SILENT)]
    return BlockResult(tidy, stats, pd.DataFrame(block_rows), pd.DataFrame(verdict_rows), blocks_of)


def write_block_deliverable(output_root: Path, tidy: pd.DataFrame, responders: list[str],
                            metrics: list[str], names: dict, out_dir: Path, excel_dir: Path,
                            tag: str) -> BlockResult:
    """Tables and figures of the block analysis, under the file names in *names*."""
    res = block_analysis(tidy, responders, metrics)

    out = res.tidy.copy()
    for c in ["Onset ms", "P1 ms", "P2 ms", "P1 uV", "P2 uV", "PTP uV", "Amplitude uV",
              "Area uV·ms", "M amplitude uV", "M P1 ms"]:
        if c in out.columns:
            out[c] = out[c].round(3)
    out.sort_values(["Channel", "Curve"]).to_csv(excel_dir / names["long"], index=False)
    res.stats.to_csv(excel_dir / names["stats"], index=False)
    res.blocks.to_csv(excel_dir / names["blocks"], index=False)
    res.verdict.to_csv(excel_dir / names["verdict"], index=False)
    for name in names["legacy_tables"]:
        (excel_dir / name).unlink(missing_ok=True)
    for name in names["legacy_figures"]:
        (out_dir / name).unlink(missing_ok=True)

    n_blocks = max((len(b) for b in res.blocks_of.values()), default=1)
    labels = [f"B{i + 1}" for i in range(n_blocks)]
    colors = block_colors(n_blocks)
    times, waves = _load_epoch_waveforms(output_root)
    drawn = [ch for ch in responders if ch in waves]
    if times is not None and drawn:
        # only responding blocks get a colour and a mean; a silent block has no response to average
        shown = res.tidy.assign(**{"Amplitude group": res.tidy["Block"].where(
            res.tidy["Block kind"] == CB.RESPONSE)})
        _plot_curves_by_group(times, waves, shown, drawn, labels, colors,
                              out_dir / names["curves_png"], tag,
                              legend_title="блок (по порядку)",
                              title="Кривые по блокам, среднее ± SD")
    _plot_amplitude_by_block(res.tidy, responders, res.blocks_of,
                             out_dir / names["amp_png"], tag)
    return res


def run_jendrassik_blocks(output_root: Path) -> Path | None:
    """The Jendrassik deliverable: contiguous blocks, their spread, and the verdict per channel."""
    output_root = Path(output_root)
    csv = _metrics_csv(output_root)
    if not csv.exists():
        print("[JENDRASSIK] No SIR metrics CSV found; skipping.", flush=True)
        return None
    tidy, responders = _tidy_metrics(csv)
    if tidy is None:
        print("[JENDRASSIK] No responding channels; skipping.", flush=True)
        return None
    out_dir = ensure_dir(_sir_dir(output_root) / "Jendrassik")
    excel_dir = ensure_dir(shared_excel_dir(output_root))
    res = write_block_deliverable(output_root, tidy, responders, _reported_metrics(tidy),
                                  JENDRASSIK_BLOCK_NAMES, out_dir, excel_dir, "приём Ендрассика")
    drop_legacy_excel_dir(out_dir)
    counts = res.verdict["Verdict"].value_counts().to_dict()
    print(f"[JENDRASSIK] {len(responders)} channels, {tidy['Curve'].nunique()} curves, "
          f"contiguous blocks; verdicts {counts} -> {out_dir}; tables -> {excel_dir}", flush=True)
    return out_dir


def run_jendrassik_analysis(output_root: Path, n_groups: int | None = None):
    """Backwards-compatible entry point for the Jendrassik scenario (``n_groups`` is unused)."""
    return run_jendrassik_blocks(output_root)


def _plot_curves_by_group(times, waves, tidy, responders, labels, colors, out_path,
                          tag="приём Ендрассика", legend_title="группа (низкая→высокая)",
                          title="Кривые по группам амплитуды, среднее ± SD"):
    """All channels on ONE figure: every curve drawn, coloured by its group (the
    ``Amplitude group`` column), with each group's mean (thick) and ±SD band on top.

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
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, title=legend_title, title_fontsize=7)

    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("мс от стимула")
    fig.suptitle(f"{title} ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_amplitude_by_block(tidy, responders, blocks_of, out_path, tag="приём Ендрассика"):
    """Amplitude against curve number, each block with its mean and ±SD band.

    The picture A. Militskova described: the curves in run order, the level of
    every block and the scatter around it, and the verdict the scatter gives.
    Silent blocks are shaded grey.
    """
    fig, axes, nrow, ncol = _grid(len(responders))
    colors = block_colors(max((len(b) for b in blocks_of.values()), default=1))
    thr = CB.SPREAD_THRESHOLD_UV
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        curves = d["Curve"].to_numpy(int)
        amps = d["Amplitude uV"].to_numpy(float)
        ax.plot(curves, amps, "-", color="0.8", lw=0.8, zorder=1)
        for k, b in enumerate(blocks_of[ch]):
            x0, x1 = min(b.curves) - 0.45, max(b.curves) + 0.45
            if b.kind == CB.SILENT or not b.n_response:
                ax.axvspan(x0, x1, color="0.92", lw=0, zorder=0)
                ax.text((x0 + x1) / 2, 0.97, "нет ответа", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="0.45")
                continue
            col = colors[k]
            sel = (curves >= min(b.curves)) & (curves <= max(b.curves)) & np.isfinite(amps)
            ax.plot(curves[sel], amps[sel], "o", ms=4, color=col, zorder=3)
            sd = b.sd if np.isfinite(b.sd) else 0.0
            ax.fill_between([x0, x1], b.mean - sd, b.mean + sd, color=col, alpha=0.18, lw=0, zorder=0)
            ax.plot([x0, x1], [b.mean, b.mean], color=col, lw=1.6, zorder=2)
            if b.n_response > 1:
                ax.text((x0 + x1) / 2, b.mean + sd, f"SD {sd:.0f}" + (f" > {thr:g}" if b.spread_above else ""),
                        ha="center", va="bottom", fontsize=7, color=col)
        verdict = CB.verdict(blocks_of[ch])
        ax.set_title(f"{ch} — {CB.VERDICT_RU[verdict]}", fontsize=10, loc="left")
        ax.set_ylabel("мкВ")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("номер кривой")
    fig.suptitle(f"Амплитуда по блокам кривых: среднее ± SD; разброс > {thr:g} мкВ — приём работает ({tag})",
                 fontsize=12)
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

"""Per-curve recruitment analysis for single-shock text ``curves`` exports.

Files like ``Т12-Л1 Ендр`` / ``Т11-12 ендрассик`` are recruitment sweeps: each
curve is one stimulus whose response grows across curves (increasing stimulation
amplitude). The standard SIR run already detects onset/P1/P2 and PTP per epoch;
this module reshapes those per-epoch metrics into recruitment deliverables:

  - a per-curve table (curve number + P1, P2, PTP per channel), long and wide;
  - a recruitment plot (amplitude vs curve number, per channel);
  - box-plots + summary statistics on TWO views the clinician asked for:
      * TOP-N  — only the last N curves (the maximal responses / plateau);
      * GROUPS — curves binned by similar response amplitude (low..high).

Runs off the SIR metrics CSV, so it is independent of the detection code and can
also be invoked stand-alone on an existing results folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import STIMULATION_INDUCED_FOLDER, ensure_dir, find_mode_dir

# last N curves treated as the maximal-response group
RECRUITMENT_TOP_N = 5
# number of amplitude bins for the "similar amplitude" grouping
RECRUITMENT_N_GROUPS = 3


def _sir_dir(output_root: Path) -> Path:
    """Where this run's SIR results live — flattened ``results/`` unless the root
    also holds another analysis and the mode level was kept."""
    return find_mode_dir(output_root, STIMULATION_INDUCED_FOLDER)


def _metrics_csv(output_root: Path) -> Path:
    return _sir_dir(output_root) / "Excel" / "Large_dataset_emg_response_metrics.csv"


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "N": int(s.size),
        "mean": round(float(s.mean()), 3) if s.size else np.nan,
        "SD": round(float(s.std(ddof=1)), 3) if s.size > 1 else 0.0,
        "median": round(float(s.median()), 3) if s.size else np.nan,
        "min": round(float(s.min()), 3) if s.size else np.nan,
        "max": round(float(s.max()), 3) if s.size else np.nan,
    }


def _tidy_metrics(csv: Path):
    """Per-curve metrics in physiological units, plus the responding channels.

    Shared with the Jendrassik deliverable, which needs the same table. Returns
    ``(tidy_df, responders)``, or ``(None, [])`` when nothing responded.
    """
    cols = ["Epoch", "Channel", "Onset latency", "Peak1 latency", "Peak2 latency",
            "Peak1 value", "Peak2 value", "PTP amplitude"]
    df = pd.read_csv(csv, usecols=lambda c: c in cols)
    if df.empty:
        return None, []

    # to physiological units: curve number from 1, latencies in ms, values in uV
    tidy = pd.DataFrame({
        "Curve": df["Epoch"].astype(int) + 1,
        "Channel": df["Channel"].astype(str),
        "Onset ms": df["Onset latency"] * 1e3,
        "P1 ms": df["Peak1 latency"] * 1e3,
        "P2 ms": df["Peak2 latency"] * 1e3,
        "P1 uV": df["Peak1 value"] * 1e6,
        "P2 uV": df["Peak2 value"] * 1e6,
        "PTP uV": df["PTP amplitude"] * 1e6,
    })
    # response amplitude: PTP when biphasic, else |P1| (monophasic channels)
    tidy["Amplitude uV"] = tidy["PTP uV"].where(
        tidy["PTP uV"].notna(), tidy["P1 uV"].abs())

    # keep only responding channels (at least one detected P1), ordered naturally
    responders = [c for c in sorted(tidy["Channel"].unique(),
                                    key=lambda s: (len(s), s))
                  if tidy.loc[tidy["Channel"] == c, "P1 uV"].notna().any()]
    tidy = tidy[tidy["Channel"].isin(responders)].copy()
    if tidy.empty:
        return None, []
    return tidy, responders


def run_recruitment_analysis(
    output_root: Path,
    top_n: int = RECRUITMENT_TOP_N,
    n_groups: int | None = None,
) -> Path | None:
    """Build the recruitment tables/plots from a finished SIR run. Returns the
    output directory, or None if the metrics CSV is missing/empty."""
    output_root = Path(output_root)
    csv = _metrics_csv(output_root)
    if not csv.exists():
        print("[RECRUITMENT] No SIR metrics CSV found; skipping.", flush=True)
        return None

    tidy, responders = _tidy_metrics(csv)
    if tidy is None:
        print("[RECRUITMENT] No responding channels; skipping.", flush=True)
        return None

    out_dir = _sir_dir(output_root) / "Recruitment"
    excel_dir = ensure_dir(out_dir / "Excel")

    # On these files P2 and PTP are routinely undetected (monophasic responses),
    # and an all-NaN table, column or box-plot panel is just noise. Report only
    # the metrics that actually carry values.
    metrics = [m for m in ["Amplitude uV", "PTP uV", "P1 uV", "P2 uV"]
               if tidy[m].notna().any()]

    # ── per-curve tables ──
    tidy_round = tidy.copy()
    for c in ["Onset ms", "P1 ms", "P2 ms", "P1 uV", "P2 uV", "PTP uV", "Amplitude uV"]:
        tidy_round[c] = tidy_round[c].round(3)
    tidy_round.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "recruitment_by_curve_long.csv", index=False)
    for metric in metrics:
        fname = f"recruitment_{metric.split()[0]}_uV_wide.csv"
        (tidy.pivot_table(index="Curve", columns="Channel", values=metric)
         .reindex(columns=responders).round(3)
         .to_csv(excel_dir / fname))

    curves = sorted(tidy["Curve"].unique())

    # ── recruitment plot: amplitude vs curve number, per channel ──
    _plot_recruitment_curves(tidy, responders, out_dir / "recruitment_curves.png")

    # ── TOP-N (last N curves) ──
    top_curves = curves[-top_n:]
    top = tidy[tidy["Curve"].isin(top_curves)]
    _plot_top_boxplots(top, responders, top_curves, metrics,
                       out_dir / f"boxplots_top{top_n}.png")
    top_stats = []
    for ch in responders:
        d = top[top["Channel"] == ch]
        for metric in metrics:
            top_stats.append({"Channel": ch, "Metric": metric,
                              "Curves": f"{top_curves[0]}-{top_curves[-1]}",
                              **_stats(d[metric])})
    pd.DataFrame(top_stats).to_csv(excel_dir / f"stats_top{top_n}.csv", index=False)

    # ── GROUPS by similar amplitude (per channel) ──
    tidy = _assign_amplitude_groups(tidy, responders, n_groups)
    tidy_round["Amplitude group"] = tidy["Amplitude group"].values
    tidy_round.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "recruitment_by_curve_long.csv", index=False)   # rewrite with group col
    _plot_group_boxplots(tidy, responders, n_groups, metrics, out_dir)
    grp_stats = []
    for ch in responders:
        d = tidy[tidy["Channel"] == ch]
        for grp in sorted(d["Amplitude group"].dropna().unique()):
            dg = d[d["Amplitude group"] == grp]
            for metric in metrics:
                grp_stats.append({"Channel": ch, "Amplitude group": grp,
                                  "Metric": metric, **_stats(dg[metric])})
    pd.DataFrame(grp_stats).to_csv(excel_dir / "stats_amplitude_groups.csv", index=False)

    print(f"[RECRUITMENT] {len(responders)} channels, {len(curves)} curves -> {out_dir}",
          flush=True)
    return out_dir


def cluster_amplitudes(values: np.ndarray, k: int) -> np.ndarray:
    """Split *values* into k groups where they actually separate (1-D k-means).

    Quantile bins (``qcut``) would force equal-sized groups, which is wrong for
    the protocol these files come from: a Jendrassik run is a block of test
    stimuli followed by a block with the manoeuvre, and those blocks are not the
    same length — A. Militskova's own example channel splits 24/21, another 36/9.
    Equal terciles would cut straight through both blocks and average the two
    conditions together.

    Returns group indices 0..k-1 ordered low -> high amplitude; NaNs get -1.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, -1, dtype=int)
    ok = np.isfinite(v)
    x = v[ok]
    if x.size == 0:
        return out
    k = max(1, min(int(k), int(np.unique(x).size)))
    if k == 1:
        out[ok] = 0
        return out

    # Lloyd's algorithm on a 1-D array, seeded at evenly spaced quantiles.
    centres = np.quantile(x, np.linspace(0, 1, k * 2 + 1)[1::2])
    for _ in range(50):
        assign = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
        new = np.array([x[assign == j].mean() if np.any(assign == j) else centres[j]
                        for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    # Relabel so group 0 is the lowest-amplitude cluster.
    order = np.argsort(centres)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    out[ok] = remap[assign]
    return out


def _silhouette_1d(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of a 1-D clustering. NaN when there is only one cluster."""
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.nan
    d = np.abs(x[:, None] - x[None, :])
    s = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        own = labels == labels[i]
        n_own = int(own.sum())
        a = (d[i, own].sum() / (n_own - 1)) if n_own > 1 else 0.0
        b = min(d[i, labels == u].mean() for u in uniq if u != labels[i])
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(np.mean(s))


def choose_n_groups(
    values: np.ndarray,
    k_max: int = 5,
    min_silhouette: float = 0.70,
    min_per_group: int = 3,
) -> int:
    """How many amplitude groups this channel's curves actually fall into.

    A. Militskova asked for the curves "grouped by similar amplitude values" and
    named no number, so imposing one is a guess — and the wrong guess is visible:
    forcing three groups onto a channel whose responses form two natural levels
    splits one of them down the middle, and forcing three onto a channel with no
    grouping at all invents structure out of a smooth ramp.

    So the count is read off the data: k-means for each k, scored by silhouette
    (how much better a point fits its own group than the nearest other one).
    Returns 1 — no grouping — when the best score fails ``min_silhouette``,
    which is the honest answer for a channel whose amplitudes rise smoothly.

    Groups smaller than ``min_per_group`` disqualify a k: a "group" of one curve
    has no spread to report and is what an outlier produces.

    The 0.70 cut-off is where the two cases part on synthetic checks: amplitudes
    that really do form levels score 0.76 to 0.95 even when the levels nearly
    touch, while a smooth ramp and a flat scatter both top out at 0.60-0.61 — a
    ramp cut in half looks half-decent to any clustering score, which is exactly
    the invented structure that has to be refused.
    """
    v = np.asarray(values, dtype=float)
    x = v[np.isfinite(v)]
    if x.size < 2 * min_per_group or np.unique(x).size < 2:
        return 1

    best_k, best_score = 1, -np.inf
    for k in range(2, min(int(k_max), int(np.unique(x).size)) + 1):
        idx = cluster_amplitudes(x, k)
        counts = np.bincount(idx[idx >= 0], minlength=k)
        if counts.min() < min_per_group:
            continue
        score = _silhouette_1d(x, idx)
        if np.isfinite(score) and score > best_score:
            best_k, best_score = k, score
    return best_k if best_score >= min_silhouette else 1


def _assign_amplitude_groups(tidy, responders, n_groups):
    """Group each channel's curves by response amplitude (low..high).

    Grouping is per channel: the same curve can land in different groups on
    different muscles, which is the point — the manoeuvre does not facilitate
    every muscle equally.

    ``n_groups=None`` means the count is not known in advance and is read off
    each channel's own amplitudes. That is the case wherever the protocol does
    not fix it: a recruitment ramp and a paired-pulse run have however many
    response levels they have. The Jendrassik protocol DOES fix it — a block
    without the manoeuvre and a block with it, per intensity — so it passes the
    number and this never runs.
    """
    tidy = tidy.copy()
    tidy["Amplitude group"] = pd.Series(index=tidy.index, dtype=object)
    for ch in responders:
        m = (tidy["Channel"] == ch).to_numpy()
        amps = tidy.loc[m, "Amplitude uV"].to_numpy(float)
        k = choose_n_groups(amps) if n_groups is None else int(n_groups)
        labels = [f"G{i + 1}" for i in range(k)]
        idx = cluster_amplitudes(amps, k)
        tidy.loc[m, "Amplitude group"] = [labels[i] if i >= 0 else None for i in idx]
    return tidy


def channel_group_labels(tidy, channel) -> list[str]:
    """The group labels this channel actually got, low → high."""
    got = tidy.loc[tidy["Channel"] == channel, "Amplitude group"].dropna().unique()
    return sorted(got, key=lambda s: int(str(s)[1:]))


def _grid(n, ncol=2):
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.0 * nrow),
                             squeeze=False)
    return fig, axes, nrow, ncol


def _plot_recruitment_curves(tidy, responders, out_path):
    """Response size against curve number — the recruitment curve itself.

    ONE series: peak-to-peak, falling back to |P1| where the response is
    monophasic and there is no second peak to measure against. Both are positive,
    so growth reads upward even on channels whose P1 is a trough.

    P1 and P2 used to be drawn alongside it. They duplicated the same curve (PTP
    is their difference) while dragging the axis into negative values, which
    squashed the recruitment into the top half of the panel. The per-curve tables
    still carry all three if a component is needed on its own.
    """
    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        n_ptp = int(d["PTP uV"].notna().sum())
        n_p1 = int(d["Amplitude uV"].notna().sum()) - n_ptp
        ax.plot(d["Curve"], d["Amplitude uV"], marker="o", ms=3.5, lw=1.3,
                color="tab:blue")
        ax.axhline(0, color="0.7", lw=0.6)
        note = f"PTP: {n_ptp}" + (f", |P1|: {n_p1}" if n_p1 else "")
        ax.set_title(f"{ch}   ({note})", fontsize=10, loc="left")
        ax.set_ylabel("µV"); ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("номер кривой")
    fig.suptitle("Кривые рекрутирования — размах ответа (PTP, либо |P1| без второго пика)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def _plot_top_boxplots(top, responders, top_curves, metrics, out_path):
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(1.1 * len(responders) + 3, 3.0 * len(metrics)),
                             squeeze=False)
    xs = np.arange(len(responders))
    for r, metric in enumerate(metrics):
        ax = axes[r][0]
        data = [top.loc[top["Channel"] == ch, metric].dropna().values for ch in responders]
        ax.boxplot(data, positions=xs, widths=0.6, showfliers=False,
                   medianprops=dict(color="tab:red"))
        ax.set_xticks(xs); ax.set_xticklabels(responders, fontsize=8)
        ax.set_ylabel(metric); ax.grid(alpha=0.3, axis="y")
    axes[-1][0].set_xlabel("канал")
    fig.suptitle(f"Топ-{len(top_curves)} (кривые {top_curves[0]}–{top_curves[-1]}): "
                 f"максимальные ответы", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def _plot_group_boxplots(tidy, responders, n_groups, metrics, out_dir):
    for metric in metrics:
        fname = f"boxplots_by_amplitude_group_{metric.split()[0]}.png"
        fig, axes, nrow, ncol = _grid(len(responders))
        for i, ch in enumerate(responders):
            ax = axes[i // ncol][i % ncol]
            d = tidy[tidy["Channel"] == ch]
            # Each channel keeps its own number of groups — see choose_n_groups.
            labels = channel_group_labels(tidy, ch)
            data = [d.loc[d["Amplitude group"] == g, metric].dropna().values for g in labels]
            ax.boxplot(data, positions=np.arange(len(labels)), widths=0.6,
                       showfliers=False, medianprops=dict(color="tab:red"))
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([f"{g}\n(low→high)" if g == labels[0] else g for g in labels],
                               fontsize=7)
            ax.set_title(f"{ch}   ({len(labels)} гр.)" if len(labels) > 1
                         else f"{ch}   (без группировки)", fontsize=10, loc="left")
            ax.set_ylabel(metric); ax.grid(alpha=0.3, axis="y")
        for j in range(len(responders), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{metric}: группы по амплитуде ответа (низкая→высокая)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=160); plt.close(fig)

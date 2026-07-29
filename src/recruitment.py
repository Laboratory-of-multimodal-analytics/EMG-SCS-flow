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

from .io_utils import STIMULATION_INDUCED_FOLDER

# last N curves treated as the maximal-response group
RECRUITMENT_TOP_N = 5
# number of amplitude bins for the "similar amplitude" grouping
RECRUITMENT_N_GROUPS = 3


def _metrics_csv(output_root: Path) -> Path:
    return (output_root / "results" / STIMULATION_INDUCED_FOLDER
            / "Excel" / "Large_dataset_emg_response_metrics.csv")


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


def run_recruitment_analysis(
    output_root: Path,
    top_n: int = RECRUITMENT_TOP_N,
    n_groups: int = RECRUITMENT_N_GROUPS,
) -> Path | None:
    """Build the recruitment tables/plots from a finished SIR run. Returns the
    output directory, or None if the metrics CSV is missing/empty."""
    output_root = Path(output_root)
    csv = _metrics_csv(output_root)
    if not csv.exists():
        print("[RECRUITMENT] No SIR metrics CSV found; skipping.", flush=True)
        return None

    cols = ["Epoch", "Channel", "Onset latency", "Peak1 latency", "Peak2 latency",
            "Peak1 value", "Peak2 value", "PTP amplitude"]
    df = pd.read_csv(csv, usecols=lambda c: c in cols)
    if df.empty:
        print("[RECRUITMENT] SIR metrics CSV is empty; skipping.", flush=True)
        return None

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
    # recruitment amplitude: PTP when biphasic, else |P1| (monophasic channels)
    tidy["Amplitude uV"] = tidy["PTP uV"].where(
        tidy["PTP uV"].notna(), tidy["P1 uV"].abs())

    # keep only responding channels (at least one detected P1), ordered naturally
    responders = [c for c in sorted(tidy["Channel"].unique(),
                                    key=lambda s: (len(s), s))
                  if tidy.loc[tidy["Channel"] == c, "P1 uV"].notna().any()]
    tidy = tidy[tidy["Channel"].isin(responders)].copy()
    if tidy.empty:
        print("[RECRUITMENT] No responding channels; skipping.", flush=True)
        return None

    out_dir = output_root / "results" / STIMULATION_INDUCED_FOLDER / "Recruitment"
    excel_dir = out_dir / "Excel"
    excel_dir.mkdir(parents=True, exist_ok=True)

    # ── per-curve tables ──
    tidy_round = tidy.copy()
    for c in ["Onset ms", "P1 ms", "P2 ms", "P1 uV", "P2 uV", "PTP uV", "Amplitude uV"]:
        tidy_round[c] = tidy_round[c].round(3)
    tidy_round.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "recruitment_by_curve_long.csv", index=False)
    for metric, fname in [("PTP uV", "recruitment_PTP_uV_wide.csv"),
                          ("P1 uV", "recruitment_P1_uV_wide.csv"),
                          ("P2 uV", "recruitment_P2_uV_wide.csv")]:
        (tidy.pivot_table(index="Curve", columns="Channel", values=metric)
         .reindex(columns=responders).round(3)
         .to_csv(excel_dir / fname))

    curves = sorted(tidy["Curve"].unique())

    # ── recruitment plot: amplitude vs curve number, per channel ──
    _plot_recruitment_curves(tidy, responders, out_dir / "recruitment_curves.png")

    # ── TOP-N (last N curves) ──
    top_curves = curves[-top_n:]
    top = tidy[tidy["Curve"].isin(top_curves)]
    _plot_top_boxplots(top, responders, top_curves,
                       out_dir / "boxplots_top5.png")
    top_stats = []
    for ch in responders:
        d = top[top["Channel"] == ch]
        for metric in ["PTP uV", "P1 uV", "P2 uV"]:
            top_stats.append({"Channel": ch, "Metric": metric,
                              "Curves": f"{top_curves[0]}-{top_curves[-1]}",
                              **_stats(d[metric])})
    pd.DataFrame(top_stats).to_csv(excel_dir / "stats_top5.csv", index=False)

    # ── GROUPS by similar amplitude (per channel) ──
    tidy = _assign_amplitude_groups(tidy, responders, n_groups)
    tidy_round["Amplitude group"] = tidy["Amplitude group"].values
    tidy_round.sort_values(["Channel", "Curve"]).to_csv(
        excel_dir / "recruitment_by_curve_long.csv", index=False)   # rewrite with group col
    _plot_group_boxplots(tidy, responders, n_groups, out_dir)
    grp_stats = []
    for ch in responders:
        d = tidy[tidy["Channel"] == ch]
        for grp in sorted(d["Amplitude group"].dropna().unique()):
            dg = d[d["Amplitude group"] == grp]
            for metric in ["PTP uV", "P1 uV", "P2 uV"]:
                grp_stats.append({"Channel": ch, "Amplitude group": grp,
                                  "Metric": metric, **_stats(dg[metric])})
    pd.DataFrame(grp_stats).to_csv(excel_dir / "stats_amplitude_groups.csv", index=False)

    print(f"[RECRUITMENT] {len(responders)} channels, {len(curves)} curves -> {out_dir}",
          flush=True)
    return out_dir


def _assign_amplitude_groups(tidy, responders, n_groups):
    """Bin each channel's curves into n_groups by response amplitude (low..high)."""
    labels = [f"G{i + 1}" for i in range(n_groups)]
    tidy = tidy.copy()
    tidy["Amplitude group"] = pd.Series(index=tidy.index, dtype=object)
    for ch in responders:
        m = tidy["Channel"] == ch
        amp = tidy.loc[m, "Amplitude uV"]
        if amp.notna().sum() < n_groups:
            tidy.loc[m & amp.notna(), "Amplitude group"] = labels[-1]
            continue
        try:
            binned = pd.qcut(amp, n_groups, labels=labels, duplicates="drop")
        except ValueError:
            binned = pd.cut(amp, n_groups, labels=labels)
        tidy.loc[m, "Amplitude group"] = binned.astype(object)
    return tidy


def _grid(n, ncol=2):
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.0 * nrow),
                             squeeze=False)
    return fig, axes, nrow, ncol


def _plot_recruitment_curves(tidy, responders, out_path):
    fig, axes, nrow, ncol = _grid(len(responders))
    for i, ch in enumerate(responders):
        ax = axes[i // ncol][i % ncol]
        d = tidy[tidy["Channel"] == ch].sort_values("Curve")
        ax.plot(d["Curve"], d["PTP uV"], marker="o", ms=3, lw=1.0, color="k", label="PTP")
        ax.plot(d["Curve"], d["P1 uV"], marker=".", ms=3, lw=0.8, color="tab:red", alpha=0.7, label="P1")
        ax.plot(d["Curve"], d["P2 uV"], marker=".", ms=3, lw=0.8, color="tab:green", alpha=0.7, label="P2")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(ch, fontsize=10, loc="left")
        ax.set_ylabel("µV"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(responders), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("номер кривой")
    fig.suptitle("Кривые рекрутирования (амплитуда vs номер кривой)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def _plot_top_boxplots(top, responders, top_curves, out_path):
    metrics = ["PTP uV", "P1 uV", "P2 uV"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(1.1 * len(responders) + 3, 9),
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


def _plot_group_boxplots(tidy, responders, n_groups, out_dir):
    labels = [f"G{i + 1}" for i in range(n_groups)]
    for metric, fname in [("PTP uV", "boxplots_by_amplitude_group_PTP.png"),
                          ("P1 uV", "boxplots_by_amplitude_group_P1.png"),
                          ("P2 uV", "boxplots_by_amplitude_group_P2.png")]:
        fig, axes, nrow, ncol = _grid(len(responders))
        for i, ch in enumerate(responders):
            ax = axes[i // ncol][i % ncol]
            d = tidy[tidy["Channel"] == ch]
            data = [d.loc[d["Amplitude group"] == g, metric].dropna().values for g in labels]
            ax.boxplot(data, positions=np.arange(n_groups), widths=0.6,
                       showfliers=False, medianprops=dict(color="tab:red"))
            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels([f"{g}\n(low→high)" if g == labels[0] else g for g in labels],
                               fontsize=7)
            ax.set_title(ch, fontsize=10, loc="left")
            ax.set_ylabel(metric); ax.grid(alpha=0.3, axis="y")
        for j in range(len(responders), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{metric}: группы по амплитуде ответа (низкая→высокая)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=160); plt.close(fig)

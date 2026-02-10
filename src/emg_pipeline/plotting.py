"""Plotting utilities for the EMG pipeline."""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

_AMP_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def amp_to_number(amp) -> float:
    if amp is None or (isinstance(amp, float) and np.isnan(amp)):
        return np.nan
    if isinstance(amp, (int, float, np.integer, np.floating)):
        return float(amp)
    s = str(amp)
    m = _AMP_NUM_RE.search(s)
    if not m:
        return np.nan
    return float(m.group(0).replace(",", "."))


def intensity_scaled_color(base_rgba, t: float):
    r, g, b, _a = base_rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_new = 0.80 - 0.55 * t
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
    return (r2, g2, b2)


def plot_epochs_panel(
    data_epoched: np.ndarray,
    times: np.ndarray,
    ch_names: list[str],
    epochs_ch_names: list[str],
    art_chans: set[str],
    latency_markers: dict[str, list[dict[str, float]]],
    title: str,
    out_path: Path,
    show_grid: bool = True,
    show_markers: bool = True,
    epochs_to_plot: dict[str, list[int]] | None = None,
) -> None:
    n_channels = len(ch_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2.5 * n_channels), dpi=500)
    if n_channels == 1:
        axes = [axes]

    for i, ch in enumerate(ch_names):
        ax = axes[i]
        ch_idx = epochs_ch_names.index(ch)
        ch_data = data_epoched[:, ch_idx, :]
        idx_map = None
        if epochs_to_plot is not None:
            keep = epochs_to_plot.get(ch, [])
            if keep:
                idx_map = {ep_idx: i for i, ep_idx in enumerate(keep)}
                ch_data = ch_data[keep, :]
            else:
                ch_data = None

        if ch_data is not None:
            ax.plot(times, ch_data.T * 1e6, color="gray", alpha=0.7, linewidth=0.7)
            avg = ch_data.mean(axis=0)
            ax.plot(times, avg * 1e6, color="black", linewidth=2)

        if show_markers and (ch not in art_chans):
            for entry in latency_markers[ch]:
                ep_idx = entry["epoch"]
                if ch_data is None:
                    continue
                if idx_map is not None:
                    ep_idx = idx_map.get(ep_idx)
                    if ep_idx is None:
                        continue
                onset = entry["onset"]
                p1 = entry["peak1"]
                p2 = entry["peak2"]

                if not np.isnan(onset):
                    onset_idx = np.argmin(np.abs(times - onset))
                    ax.scatter(onset, ch_data[ep_idx, onset_idx] * 1e6, color="blue", s=18)

                if not np.isnan(p1):
                    p1_idx = np.argmin(np.abs(times - p1))
                    ax.scatter(p1, ch_data[ep_idx, p1_idx] * 1e6, color="red", s=18)

                if not np.isnan(p2):
                    p2_idx = np.argmin(np.abs(times - p2))
                    ax.scatter(p2, ch_data[ep_idx, p2_idx] * 1e6, color="green", s=18)

        ax.set_ylabel("Amplitude (mkV)")
        ax.grid(bool(show_grid) if ch_data is not None else False)
        ax.set_title(ch)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_grouped_by_amplitude(
    group_store: dict,
    times: np.ndarray,
    out_dir: Path,
    art_chans: set[str],
) -> None:
    cmap = matplotlib.cm.get_cmap("tab10")

    for config, ch_dict in group_store.items():
        channels = [ch for ch in ch_dict.keys() if ch not in art_chans]
        if not channels:
            continue

        n_ch = len(channels)
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2.5 * n_ch), dpi=500)
        if n_ch == 1:
            axes = [axes]

        for i, ch in enumerate(channels):
            ax = axes[i]
            amp_dict = ch_dict[ch]

            amps_sorted = sorted(
                amp_dict.keys(),
                key=lambda a: (np.isnan(amp_to_number(a)), amp_to_number(a)),
            )

            base = cmap(i % 10)
            nA = len(amps_sorted)
            for j, amp_raw in enumerate(amps_sorted):
                waves = amp_dict[amp_raw]
                if not waves:
                    continue
                mean_wave = np.mean(np.stack(waves, axis=0), axis=0)
                t = 0.0 if nA <= 1 else j / (nA - 1)
                col = intensity_scaled_color(base, t)
                ax.plot(times, mean_wave * 1e6, color=col, linewidth=2, label=str(amp_raw))

            ax.set_title(ch)
            ax.set_ylabel("Amplitude (mkV)")
            ax.grid(False)
            ax.legend(
                loc="upper right",
                fontsize=8,
                frameon=False,
                ncol=2,
                columnspacing=0.8,
                handlelength=1.6,
                handletextpad=0.4,
                borderaxespad=0.2,
            )

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Configuration: {config} — mean responses grouped by Stim. amplitude")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe_config = re.sub(r"[^\w\-_\. ]", "_", str(config))
        out_path = out_dir / f"{safe_config}_grouped_by_amplitude.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)


def plot_grouped_by_condition(
    group_store: dict,
    times: np.ndarray,
    out_dir: Path,
    ch_names: list[str],
) -> None:
    if not group_store:
        return
    if not ch_names:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in sorted(group_store.keys()):
        fig, axes = plt.subplots(len(ch_names), 1, figsize=(10, 2.5 * len(ch_names)), dpi=500)
        if len(ch_names) == 1:
            axes = [axes]

        for i, ch in enumerate(ch_names):
            ax = axes[i]
            waves = group_store[condition].get(ch, [])
            if waves:
                mean_wave = np.mean(np.stack(waves, axis=0), axis=0)
                ax.plot(times, mean_wave * 1e6, color="black", linewidth=2)
            ax.set_title(ch)
            ax.set_ylabel("Amplitude (mkV)")
            ax.grid(False)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Condition: {condition} — mean responses")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe_condition = re.sub(r"[^\w\-_\. ]", "_", str(condition))
        out_path = out_dir / f"{safe_condition}_grouped_by_condition.png"
        plt.savefig(out_path)
        plt.close(fig)


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["Stim. amplitude"] = df["Stim. amplitude"].astype(str)
    df["Configuration"] = df["Configuration"].astype(str)
    df["Channel"] = df["Channel"].astype(str)

    if "Stim_amp_num" not in df.columns:
        df["Stim_amp_num"] = df["Stim. amplitude"].apply(amp_to_number)

    metrics_to_plot = [
        ("Onset latency", "Latency (s)", False),
        ("Peak1 latency", "Latency (s)", False),
        ("Peak2 latency", "Latency (s)", False),
        ("Peak1 value", "Amplitude |V|", True),
        ("Peak2 value", "Amplitude |V|", True),
        ("PTP amplitude", "Amplitude |V|", True),
    ]

    dot_alpha = 0.35
    dot_size = 10
    jitter = 0.12
    rng = np.random.default_rng(0)

    def safe_name(name: str) -> str:
        return re.sub(r"[^\w\-_\. ]", "_", str(name))

    configs = sorted(df["Configuration"].dropna().unique())

    for config in configs:
        dfc = df[df["Configuration"] == config].copy()
        if dfc.empty:
            continue

        channels = sorted(dfc["Channel"].dropna().unique())

        for ch in channels:
            dfcc = dfc[dfc["Channel"] == ch].copy()
            if dfcc.empty:
                continue

            amps_tbl = (
                dfcc[["Stim. amplitude", "Stim_amp_num"]]
                .drop_duplicates()
                .sort_values(by=["Stim_amp_num", "Stim. amplitude"], kind="mergesort")
            )
            amp_labels = amps_tbl["Stim. amplitude"].tolist()

            fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=300)
            axes = axes.ravel()

            for ax, (metric, ylab, take_abs) in zip(axes, metrics_to_plot):
                data = []
                labels = []

                for a in amp_labels:
                    vals = pd.to_numeric(
                        dfcc.loc[dfcc["Stim. amplitude"] == a, metric],
                        errors="coerce",
                    ).dropna().values
                    if vals.size == 0:
                        continue
                    if take_abs:
                        vals = np.abs(vals)
                    data.append(vals)
                    labels.append(a)

                if len(data) == 0:
                    ax.set_title(f"{metric} (no data)")
                    ax.axis("off")
                    continue

                x_pos = np.arange(1, len(data) + 1)
                bp = ax.boxplot(
                    data,
                    positions=x_pos,
                    labels=labels,
                    showfliers=False,
                    whis=(5, 95),
                    patch_artist=True,
                    widths=0.55,
                )

                for box in bp["boxes"]:
                    box.set_alpha(0.25)
                    box.set_linewidth(1.0)
                for part in ("whiskers", "caps", "medians"):
                    for obj in bp[part]:
                        obj.set_linewidth(1.0)

                for i, vals in enumerate(data):
                    xi = x_pos[i]
                    jitter_vals = rng.uniform(-jitter, jitter, size=len(vals))
                    ax.scatter(
                        np.full_like(vals, xi, dtype=float) + jitter_vals,
                        vals,
                        s=dot_size,
                        alpha=dot_alpha,
                    )

                ax.set_title(metric)
                ax.set_ylabel(ylab)
                ax.tick_params(axis="x", labelrotation=90)
                ax.grid(False)

            for k in range(len(metrics_to_plot), len(axes)):
                axes[k].axis("off")

            plt.suptitle(
                f"Boxplots + samples by Stim. amplitude — Configuration: {config} | Channel: {ch}"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = output_dir / f"{safe_name(config)}__{safe_name(ch)}_boxplots_samples.png"
            plt.savefig(out_path)
            plt.close(fig)


def plot_template_with_markers(
    times: np.ndarray,
    template: np.ndarray,
    markers: dict[str, float],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=500)
    ax.plot(times, template * 1e6, color="black", linewidth=1.5)

    onset = markers.get("onset", np.nan)
    p1 = markers.get("p1", np.nan)
    p2 = markers.get("p2", np.nan)

    if not np.isnan(onset):
        onset_idx = int(np.argmin(np.abs(times - onset)))
        ax.scatter(onset, template[onset_idx] * 1e6, color="blue", s=30, label="Onset")
    if not np.isnan(p1):
        p1_idx = int(np.argmin(np.abs(times - p1)))
        ax.scatter(p1, template[p1_idx] * 1e6, color="red", s=30, label="P1")
    if not np.isnan(p2):
        p2_idx = int(np.argmin(np.abs(times - p2)))
        ax.scatter(p2, template[p2_idx] * 1e6, color="green", s=30, label="P2")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mkV)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_single_epoch_panel(
    times: np.ndarray,
    data: np.ndarray,
    ch_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    n_channels = len(ch_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2.5 * n_channels), dpi=500)
    if n_channels == 1:
        axes = [axes]

    for i, ch in enumerate(ch_names):
        ax = axes[i]
        ax.plot(times, data[i] * 1e6, color="black", linewidth=1)
        ax.set_ylabel("Amplitude (mkV)")
        ax.set_title(ch)
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_startstop_detection_pair(
    start_data: np.ndarray | None,
    start_env: np.ndarray | None,
    start_times: np.ndarray | None,
    stop_data: np.ndarray | None,
    stop_env: np.ndarray | None,
    stop_times: np.ndarray | None,
    ch_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    n_ch = len(ch_names)
    fig, axes = plt.subplots(n_ch, 2, figsize=(12, 2.4 * n_ch), dpi=500, sharex="col")
    if n_ch == 1:
        axes = np.array([axes])

    for i, ch in enumerate(ch_names):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        if start_data is not None and start_env is not None and start_times is not None:
            ax_l.plot(start_times, start_data[i] * 1e6, color="black", linewidth=1.0)
            ax_l.plot(start_times, start_env[i] * 1e6, color="red", linewidth=1.0, alpha=0.8)
        ax_l.set_ylabel(ch)
        ax_l.grid(True)

        if stop_data is not None and stop_env is not None and stop_times is not None:
            ax_r.plot(stop_times, stop_data[i] * 1e6, color="black", linewidth=1.0)
            ax_r.plot(stop_times, stop_env[i] * 1e6, color="red", linewidth=1.0, alpha=0.8)
        ax_r.grid(True)

        if i == 0:
            ax_l.set_title("Start")
            ax_r.set_title("Stop")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_startstop_detection_columns(
    ch_names: list[str],
    detections: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    if not detections or not ch_names:
        return

    n_det = len(detections)
    n_ch = len(ch_names)
    fig, axes = plt.subplots(n_ch, n_det, figsize=(3.2 * n_det, 2.2 * n_ch), dpi=500, sharex="col")
    if n_det == 1 and n_ch == 1:
        axes = np.array([[axes]])
    elif n_det == 1:
        axes = axes.reshape(n_ch, 1)
    elif n_ch == 1:
        axes = axes.reshape(1, n_det)

    for j, (times, data, envs, highlight) in enumerate(detections):
        for i, ch in enumerate(ch_names):
            ax = axes[i, j]
            is_hi = bool(highlight[i])
            color = "black" if is_hi else "gray"
            alpha = 1.0 if is_hi else 0.5
            ax.plot(times, data[i] * 1e6, color=color, linewidth=1.0, alpha=alpha)
            ax.plot(times, envs[i] * 1e6, color="red", linewidth=1.0, alpha=0.8 if is_hi else 0.4)
            if j == 0:
                ax.set_ylabel(ch)
            if i == 0:
                ax.set_title(f"Det {j + 1}")
            ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

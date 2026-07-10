#!/usr/bin/env python3
"""Quantify and annotate sharpness of BackImage ramp-unit activation maps."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/"
    "ramp_unit_image_maps_top6_img6_v1"
)
DEFAULT_CACHE_NAME = "backimage_contour_axis_rr100_ramp_unit_image_maps.npz"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--selection-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hf-cutoff-cycles-per-px", type=float, default=0.18)
    parser.add_argument(
        "--annotate-metric",
        choices=(
            "map_ssi_bits_per_spike",
            "hf_power_fraction_z",
            "grad_rms_z",
            "laplacian_rms_z",
            "effective_area_fraction",
        ),
        default="hf_power_fraction_z",
    )
    parser.add_argument("--map-vmin-percentile", type=float, default=1.0)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def metric_description(metric: str) -> str:
    if metric == "map_ssi_bits_per_spike":
        return "SSI is computed directly from the displayed mean activation map"
    if metric in {"hf_power_fraction_z", "grad_rms_z", "laplacian_rms_z"}:
        return "metric is amplitude-normalized; higher means more fine spatial structure"
    if metric == "effective_area_fraction":
        return "higher means a broader activation footprint"
    return metric


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def zscore_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(image, dtype=np.float64)
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std <= EPS:
        return np.zeros_like(x, dtype=np.float64), mean, std
    return (x - mean) / std, mean, std


def high_frequency_power_fraction(image: np.ndarray, cutoff_cycles_per_px: float) -> float:
    x = np.asarray(image, dtype=np.float64)
    x = x - float(np.nanmean(x))
    power = np.abs(np.fft.fft2(x)) ** 2
    total = float(np.nansum(power))
    if not np.isfinite(total) or total <= EPS:
        return 0.0
    fy = np.fft.fftfreq(x.shape[0])
    fx = np.fft.fftfreq(x.shape[1])
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = radius >= float(cutoff_cycles_per_px)
    return float(np.nansum(power[mask]) / total)


def map_ssi_bits_per_spike(image: np.ndarray) -> float:
    rate = np.asarray(image, dtype=np.float64)
    if np.nanmin(rate) < -1e-7:
        raise ValueError(f"Activation map contains negative values; min={float(np.nanmin(rate)):.6g}")
    rate = np.maximum(rate, 0.0)
    mean_rate = float(np.nanmean(rate))
    if not np.isfinite(mean_rate) or mean_rate <= EPS:
        return 0.0
    gain = rate / mean_rate
    return float(np.nanmean(gain * np.log2(gain + EPS)))


def effective_area_fraction(image: np.ndarray) -> float:
    x = np.asarray(image, dtype=np.float64)
    baseline = float(np.nanpercentile(x, 10.0))
    w = np.clip(x - baseline, 0.0, None)
    denom = float(np.nansum(w * w))
    if not np.isfinite(denom) or denom <= EPS:
        return 0.0
    effective_area_px = float(np.nansum(w) ** 2 / denom)
    return effective_area_px / float(w.size)


def sharpness_metrics(image: np.ndarray, *, cutoff_cycles_per_px: float) -> dict[str, float]:
    z, mean, std = zscore_image(image)
    dy, dx = np.gradient(z)
    grad_rms = float(np.sqrt(np.nanmean(dx * dx + dy * dy)))
    lap = (
        z[:-2, 1:-1]
        + z[2:, 1:-1]
        + z[1:-1, :-2]
        + z[1:-1, 2:]
        - 4.0 * z[1:-1, 1:-1]
    )
    lap_rms = float(np.sqrt(np.nanmean(lap * lap))) if lap.size else 0.0
    return {
        "map_mean": mean,
        "map_std": std,
        "map_min": float(np.nanmin(image)),
        "map_max": float(np.nanmax(image)),
        "map_ssi_bits_per_spike": map_ssi_bits_per_spike(image),
        "grad_rms_z": grad_rms,
        "laplacian_rms_z": lap_rms,
        "hf_power_fraction_z": high_frequency_power_fraction(z, cutoff_cycles_per_px),
        "effective_area_fraction": effective_area_fraction(image),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_table(
    payload: dict[str, np.ndarray],
    selected: pd.DataFrame,
    *,
    cutoff_cycles_per_px: float,
) -> pd.DataFrame:
    maps = np.asarray(payload["unit_maps"], dtype=np.float32)
    movie_indices = [int(v) for v in np.asarray(payload["movie_indices"], dtype=int)]
    unit_indices = [int(v) for v in np.asarray(payload["unit_indices"], dtype=int)]
    condition_ids = np.asarray(payload["condition_id"]).astype(str)
    condition_labels = np.asarray(payload["condition_label"]).astype(str)
    across_scales = np.asarray(payload["condition_across_scale"], dtype=float)
    movie_pos = {movie_idx: idx for idx, movie_idx in enumerate(movie_indices)}
    unit_pos = {unit_idx: idx for idx, unit_idx in enumerate(unit_indices)}
    selected_lookup = {
        (int(row.unit_index), int(row.movie_index)): row._asdict()
        for row in selected.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for unit in unit_indices:
        unit_rows = selected[selected["unit_index"].astype(int) == int(unit)].sort_values("image_rank_for_unit")
        for selection_row in unit_rows.itertuples(index=False):
            movie_idx = int(selection_row.movie_index)
            meta = selected_lookup[(int(unit), movie_idx)]
            for condition_pos, condition_id in enumerate(condition_ids):
                image = maps[movie_pos[movie_idx], condition_pos, unit_pos[int(unit)]]
                metrics = sharpness_metrics(image, cutoff_cycles_per_px=cutoff_cycles_per_px)
                rows.append(
                    {
                        "unit_index": int(unit),
                        "unit_label": f"u{int(unit):03d}",
                        "image_rank_for_unit": int(meta["image_rank_for_unit"]),
                        "movie_index": movie_idx,
                        "trial_id": int(meta["trial_id"]),
                        "source_row": int(meta["source_row"]),
                        "ramp_score_0_to_3": float(meta["ssi_peak_minus_base"]),
                        "condition_index": int(condition_pos),
                        "condition_id": str(condition_id),
                        "condition_label": str(condition_labels[condition_pos]),
                        "across_scale": float(across_scales[condition_pos]),
                        "hf_cutoff_cycles_per_px": float(cutoff_cycles_per_px),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_summary(out_dir: Path, metrics: pd.DataFrame, *, metric: str, dpi: int) -> Path:
    units = sorted(metrics["unit_index"].unique())
    fig, axes = plt.subplots(len(units), 2, figsize=(9.6, 1.8 * len(units)), sharex=True)
    if len(units) == 1:
        axes = np.asarray([axes])
    for row_idx, unit in enumerate(units):
        sub = metrics[metrics["unit_index"] == unit]
        grouped = sub.groupby("across_scale", sort=True)
        x = np.asarray(list(grouped.groups.keys()), dtype=float)
        sharp_mean = grouped[metric].mean().to_numpy(dtype=float)
        sharp_sem = grouped[metric].sem().fillna(0.0).to_numpy(dtype=float)
        mean_rate = grouped["map_mean"].mean().to_numpy(dtype=float)
        mean_sem = grouped["map_mean"].sem().fillna(0.0).to_numpy(dtype=float)
        ax = axes[row_idx, 0]
        ax.plot(x, sharp_mean, marker="o", linewidth=1.4, color="#2f6fbb")
        ax.fill_between(x, sharp_mean - sharp_sem, sharp_mean + sharp_sem, color="#2f6fbb", alpha=0.18, linewidth=0)
        ax.set_ylabel(f"u{int(unit):03d}", rotation=0, ha="right", va="center", labelpad=26)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax2 = axes[row_idx, 1]
        ax2.plot(x, mean_rate, marker="o", linewidth=1.4, color="#a24c3d")
        ax2.fill_between(x, mean_rate - mean_sem, mean_rate + mean_sem, color="#a24c3d", alpha=0.18, linewidth=0)
        ax2.grid(True, alpha=0.25, linewidth=0.5)
    axes[0, 0].set_title(f"{metric}; mean +/- SEM across selected images", fontsize=10)
    axes[0, 1].set_title("map mean brightness; mean +/- SEM", fontsize=10)
    axes[-1, 0].set_xlabel("across-contour motion scale; along=1")
    axes[-1, 1].set_xlabel("across-contour motion scale; along=1")
    fig.tight_layout()
    path = out_dir / f"ramp_unit_{metric}_and_brightness_summary.png"
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_annotated_sheets(
    out_dir: Path,
    payload: dict[str, np.ndarray],
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    metric: str,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    dpi: int,
) -> list[Path]:
    maps = np.asarray(payload["unit_maps"], dtype=np.float32)
    movie_indices = [int(v) for v in np.asarray(payload["movie_indices"], dtype=int)]
    unit_indices = [int(v) for v in np.asarray(payload["unit_indices"], dtype=int)]
    labels = np.asarray(payload["condition_label"]).astype(str)
    movie_pos_by_index = {int(movie_idx): pos for pos, movie_idx in enumerate(movie_indices)}
    unit_pos_by_index = {int(unit): pos for pos, unit in enumerate(unit_indices)}
    metric_lookup = {
        (int(row.unit_index), int(row.movie_index), str(row.condition_id)): float(getattr(row, metric))
        for row in metrics.itertuples(index=False)
    }
    sheet_dir = out_dir / f"ramp_unit_activation_maps_annotated_{metric}"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for unit in unit_indices:
        rows_for_unit = (
            selected[selected["unit_index"].astype(int) == int(unit)]
            .sort_values("image_rank_for_unit")
            .to_dict("records")
        )
        n_rows = len(rows_for_unit)
        n_cols = len(labels)
        fig = plt.figure(figsize=(max(10.2, 1.18 * n_cols + 2.2), max(4.0, 1.18 * n_rows + 1.55)))
        gs = fig.add_gridspec(
            nrows=n_rows + 1,
            ncols=n_cols + 1,
            width_ratios=[1.65, *([1.0] * n_cols)],
            height_ratios=[0.28, *([1.0] * n_rows)],
            hspace=0.045,
            wspace=0.045,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        ax0.text(1.0, 0.2, "image", ha="right", va="center", fontsize=7, color="#555555")
        for c, label in enumerate(labels, start=1):
            ax = fig.add_subplot(gs[0, c])
            ax.axis("off")
            ax.text(0.5, 0.2, str(label), ha="center", va="center", fontsize=7, color="#555555")
        unit_pos = unit_pos_by_index[int(unit)]
        row_images = [maps[movie_pos_by_index[int(row["movie_index"])], :, unit_pos] for row in rows_for_unit]
        vmin, vmax = image_scale(
            [img for row_stack in row_images for img in row_stack],
            float(map_vmin_percentile),
            float(map_vmax_percentile),
        )
        for r, row in enumerate(rows_for_unit, start=1):
            movie_idx = int(row["movie_index"])
            label_ax = fig.add_subplot(gs[r, 0])
            label_ax.axis("off")
            label_ax.text(
                1.0,
                0.5,
                f"rank {int(row['image_rank_for_unit'])}\nmovie {movie_idx} src {int(row['source_row'])}\nramp {float(row['ssi_peak_minus_base']):.3g}",
                ha="right",
                va="center",
                fontsize=6.4,
                color="#444444",
            )
            movie_pos = movie_pos_by_index[movie_idx]
            for c, condition_id in enumerate(np.asarray(payload["condition_id"]).astype(str)):
                ax = fig.add_subplot(gs[r, c + 1])
                ax.imshow(maps[movie_pos, c, unit_pos], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                value = metric_lookup[(int(unit), movie_idx, str(condition_id))]
                ax.text(
                    0.03,
                    0.06,
                    f"{value:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.55)
                    spine.set_edgecolor("#666666")
        fig.suptitle(
            f"BackImage RR100 unit u{int(unit):03d}: activation maps annotated with {metric}\n"
            f"{metric_description(metric)}",
            fontsize=10.5,
            y=0.995,
        )
        png = sheet_dir / f"backimage_rr100_unit_u{int(unit):03d}_{metric}_annotated_maps.png"
        pdf = sheet_dir / f"backimage_rr100_unit_u{int(unit):03d}_{metric}_annotated_maps.pdf"
        fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        paths.extend([png, pdf])
    return paths


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cache_path = Path(args.cache_path) if args.cache_path else run_dir / "cache" / DEFAULT_CACHE_NAME
    selection_csv = Path(args.selection_csv) if args.selection_csv else run_dir / "ramping_unit_selected_images.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "ramp_unit_activation_map_sharpness"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_npz(cache_path)
    selected = pd.read_csv(selection_csv)
    metrics = build_metric_table(payload, selected, cutoff_cycles_per_px=float(args.hf_cutoff_cycles_per_px))
    csv_path = out_dir / "ramp_unit_activation_map_sharpness_metrics.csv"
    metrics.to_csv(csv_path, index=False)

    by_unit = (
        metrics.groupby("unit_label")
        .apply(
            lambda df: pd.Series(
                {
                    "hf_0x": df[df["across_scale"] == 0.0]["hf_power_fraction_z"].mean(),
                    "hf_1x": df[df["across_scale"] == 1.0]["hf_power_fraction_z"].mean(),
                    "hf_3x": df[df["across_scale"] == 3.0]["hf_power_fraction_z"].mean(),
                    "hf_3_minus_1": df[df["across_scale"] == 3.0]["hf_power_fraction_z"].mean()
                    - df[df["across_scale"] == 1.0]["hf_power_fraction_z"].mean(),
                    "map_ssi_0x": df[df["across_scale"] == 0.0]["map_ssi_bits_per_spike"].mean(),
                    "map_ssi_1x": df[df["across_scale"] == 1.0]["map_ssi_bits_per_spike"].mean(),
                    "map_ssi_3x": df[df["across_scale"] == 3.0]["map_ssi_bits_per_spike"].mean(),
                    "map_ssi_3_minus_1": df[df["across_scale"] == 3.0]["map_ssi_bits_per_spike"].mean()
                    - df[df["across_scale"] == 1.0]["map_ssi_bits_per_spike"].mean(),
                    "mean_1x": df[df["across_scale"] == 1.0]["map_mean"].mean(),
                    "mean_3x": df[df["across_scale"] == 3.0]["map_mean"].mean(),
                    "mean_3_minus_1": df[df["across_scale"] == 3.0]["map_mean"].mean()
                    - df[df["across_scale"] == 1.0]["map_mean"].mean(),
                }
            )
        )
        .reset_index()
    )
    by_unit.to_csv(out_dir / "ramp_unit_sharpness_summary_by_unit.csv", index=False)
    plot_metric_summary(out_dir, metrics, metric=str(args.annotate_metric), dpi=int(args.dpi))
    paths = plot_annotated_sheets(
        out_dir,
        payload,
        selected,
        metrics,
        metric=str(args.annotate_metric),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        dpi=int(args.dpi),
    )
    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote unit summary: {out_dir / 'ramp_unit_sharpness_summary_by_unit.csv'}")
    print(f"Wrote {len(paths)} annotated sheet files under {out_dir / f'ramp_unit_activation_maps_annotated_{args.annotate_metric}'}")


if __name__ == "__main__":
    main()

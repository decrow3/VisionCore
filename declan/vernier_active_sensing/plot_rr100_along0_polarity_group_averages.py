#!/usr/bin/env python3
"""Plot positive/negative RR100 unit polarity group averages along the along=0 line."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi import (
    CACHE_SCHEMA_VERSION as ENDPOINT_CACHE_SCHEMA_VERSION,
    condition_sequence,
    image_scale,
    summarize_units,
)
from declan.vernier_active_sensing.plot_rr100_real_trace_along0_unit_ssi import (
    CACHE_SCHEMA_VERSION as REAL_TRACE_CACHE_SCHEMA_VERSION,
    DEFAULT_OUT_DIR as DEFAULT_REAL_TRACE_UNIT_DIR,
)
from declan.vernier_active_sensing.run_rr100_endpoint_history_scale_grid import (
    DEFAULT_OUT_DIR as DEFAULT_ENDPOINT_SCALE_GRID_DIR,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import DEFAULT_SCALES


DEFAULT_ENDPOINT_UNIT_DIR = DEFAULT_ENDPOINT_SCALE_GRID_DIR / "unit_ssi_along0_diagnostics"
EPS = 1e-8
MIN_CACHE_SCHEMA_VERSION_BY_MODE = {
    "endpoint": ENDPOINT_CACHE_SCHEMA_VERSION,
    "real_trace": REAL_TRACE_CACHE_SCHEMA_VERSION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["both", "real_trace", "endpoint"],
        default="both",
        help="Which cached diagnostic directory to postprocess.",
    )
    parser.add_argument("--real-trace-dir", type=Path, default=DEFAULT_REAL_TRACE_UNIT_DIR)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT_UNIT_DIR)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--real-trace-max-frames", type=int, default=60)
    parser.add_argument("--low-percentile", type=float, default=5.0)
    parser.add_argument("--high-percentile", type=float, default=95.0)
    parser.add_argument(
        "--min-static-ssi-bits",
        type=float,
        default=0.0,
        help="Drop units whose static SSI is below this bits/spike floor before averaging group curves/maps.",
    )
    parser.add_argument(
        "--include-all-group",
        action="store_true",
        help="Also plot the retained-unit all group alongside positive and negative groups.",
    )
    parser.add_argument(
        "--curve-metric",
        choices=["ratio", "absolute"],
        default="ratio",
        help="Plot geometric-mean SSI ratios or arithmetic mean absolute unit SSI.",
    )
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_scale_list(text: str) -> list[float]:
    scales = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not scales:
        raise ValueError("At least one scale is required.")
    return scales


def scale_token(scale: float) -> str:
    return f"{float(scale):g}".replace(".", "p")


def cache_path(
    *,
    mode: str,
    out_dir: Path,
    condition: str,
    fd_step_arcmin: float,
    real_trace_max_frames: int,
) -> Path:
    if mode == "endpoint":
        return out_dir / "cache" / f"rr100_endpoint_along0_unit_ssi_{condition}_fd{float(fd_step_arcmin):.4f}arcmin.npz"
    if mode == "real_trace":
        return (
            out_dir
            / "cache"
            / f"rr100_real_trace_along0_unit_ssi_{condition}_frames{int(real_trace_max_frames)}_fd{float(fd_step_arcmin):.4f}arcmin.npz"
        )
    raise ValueError(f"Unknown mode: {mode}")


def _cache_identity_for_consistency_check(data: Any, *, mode: str, path: Path) -> dict[str, Any]:
    """Parse and validate the per-condition cache identity, dropping the per-condition key.

    Callers compare the returned dicts across conditions to catch a run whose
    caches were computed under mixed code versions or mixed CLI args (e.g. a
    partially recomputed along=0 scale line).
    """
    if "cache_identity_json" not in data.files:
        raise ValueError(
            f"Cache predates identity tracking and cannot be verified fresh: {path}\n"
            f"Rerun the {mode} along=0 unit diagnostic with --force."
        )
    identity = json.loads(str(np.asarray(data["cache_identity_json"]).ravel()[0]))
    min_schema_version = MIN_CACHE_SCHEMA_VERSION_BY_MODE[mode]
    schema_version = identity.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < min_schema_version:
        raise ValueError(
            f"Cache schema_version={schema_version!r} is older than the current "
            f"{mode} along=0 diagnostic (schema_version={min_schema_version}): {path}\n"
            f"Rerun the {mode} along=0 unit diagnostic with --force."
        )
    identity.pop("condition", None)
    return identity


def load_stats_by_condition(
    *,
    mode: str,
    out_dir: Path,
    rows: list[dict[str, Any]],
    fd_step_arcmin: float,
    real_trace_max_frames: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    reference_condition: str | None = None
    for row in rows:
        condition = str(row["condition"])
        path = cache_path(
            mode=mode,
            out_dir=out_dir,
            condition=condition,
            fd_step_arcmin=fd_step_arcmin,
            real_trace_max_frames=real_trace_max_frames,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing unit SSI cache for {condition}: {path}\n"
                f"Run the {mode} along=0 unit diagnostic first."
            )
        with np.load(path) as data:
            missing = {"unit_bits_per_trace", "unit_mean_rate_per_trace", "population_bits_per_trace", "mean_rate_map"} - set(data.files)
            if missing:
                raise ValueError(f"Cache is missing required arrays {sorted(missing)}: {path}")
            identity = _cache_identity_for_consistency_check(data, mode=mode, path=path)
            if reference_identity is None:
                reference_identity = identity
                reference_condition = condition
            elif identity != reference_identity:
                raise ValueError(
                    f"Cache identity mismatch between conditions {reference_condition!r} and {condition!r} "
                    f"under {out_dir}; these caches were computed with different code/args and cannot be "
                    f"combined into one along=0 line. Rerun the {mode} along=0 unit diagnostic with --force."
                )
            out[condition] = {
                "unit_bits_per_trace": np.asarray(data["unit_bits_per_trace"], dtype=np.float32),
                "unit_mean_rate_per_trace": np.asarray(data["unit_mean_rate_per_trace"], dtype=np.float32),
                "population_bits_per_trace": np.asarray(data["population_bits_per_trace"], dtype=np.float32),
                "mean_rate_map": np.asarray(data["mean_rate_map"], dtype=np.float32),
            }
    return out


def classify_unit_polarity(
    static_maps: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> pd.DataFrame:
    maps = np.asarray(static_maps, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"Expected static maps shaped (unit, H, W), got {maps.shape}")
    rows: list[dict[str, Any]] = []
    for unit_index, unit_map in enumerate(maps):
        finite = np.asarray(unit_map, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            median = high = low = pos_strength = neg_strength = float("nan")
            polarity = "unknown"
        else:
            median = float(np.nanmedian(finite))
            high = float(np.nanpercentile(finite, float(high_percentile)))
            low = float(np.nanpercentile(finite, float(low_percentile)))
            pos_strength = high - median
            neg_strength = median - low
            polarity = "positive" if pos_strength >= neg_strength else "negative"
        rows.append(
            {
                "unit_index": int(unit_index),
                "polarity": polarity,
                "static_map_median": median,
                "static_map_high_percentile": high,
                "static_map_low_percentile": low,
                "positive_strength": pos_strength,
                "negative_strength": neg_strength,
                "polarity_score_positive_minus_negative": pos_strength - neg_strength,
                "polarity_rule": f"positive if p{float(high_percentile):g}-median >= median-p{float(low_percentile):g}",
            }
        )
    return pd.DataFrame(rows)


def mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=axis)
    n = arr.shape[axis]
    if n > 1:
        sem = np.nanstd(arr, axis=axis, ddof=1) / math.sqrt(float(n))
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def normalized_group_maps(
    *,
    stats_by_condition: dict[str, dict[str, Any]],
    map_rows: list[dict[str, Any]],
    unit_indices: list[int],
    map_vmin_percentile: float,
    map_vmax_percentile: float,
) -> np.ndarray:
    if not unit_indices:
        first = np.asarray(stats_by_condition[str(map_rows[0]["condition"])]["mean_rate_map"], dtype=np.float32)
        return np.full((len(map_rows), first.shape[1], first.shape[2]), np.nan, dtype=np.float32)
    normed_by_unit: list[np.ndarray] = []
    for unit_index in unit_indices:
        images = [
            np.asarray(stats_by_condition[str(row["condition"])]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
            for row in map_rows
        ]
        vmin, vmax = image_scale(images, float(map_vmin_percentile), float(map_vmax_percentile))
        denom = max(float(vmax - vmin), EPS)
        normed = [np.clip((np.asarray(image, dtype=np.float32) - vmin) / denom, 0.0, 1.0) for image in images]
        normed_by_unit.append(np.asarray(normed, dtype=np.float32))
    return np.nanmean(np.asarray(normed_by_unit, dtype=np.float32), axis=0)


def absolute_unit_ssi_matrices(
    *,
    stats_by_condition: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonstatic_rows = [row for row in rows if not bool(row["is_static_baseline"])]
    static = np.asarray(stats_by_condition["static_center"]["unit_bits_per_trace"], dtype=np.float64)
    n_units = int(static.shape[1])
    unit_abs = np.full((n_units, len(nonstatic_rows)), np.nan, dtype=np.float64)
    population_abs = np.full(len(nonstatic_rows), np.nan, dtype=np.float64)
    x = np.full(len(nonstatic_rows), np.nan, dtype=np.float64)
    for idx, row in enumerate(nonstatic_rows):
        condition = str(row["condition"])
        unit_abs[:, idx] = np.nanmean(
            np.asarray(stats_by_condition[condition]["unit_bits_per_trace"], dtype=np.float64),
            axis=0,
        )
        population_abs[idx] = float(
            np.nanmean(np.asarray(stats_by_condition[condition]["population_bits_per_trace"], dtype=np.float64))
        )
        x[idx] = float(row["across_scale"])
    return x, unit_abs, population_abs


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def draw_group_figure(
    *,
    mode: str,
    out_dir: Path,
    output_stem: str,
    rows: list[dict[str, Any]],
    unit_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    polarity_df: pd.DataFrame,
    stats_by_condition: dict[str, dict[str, Any]],
    min_static_ssi_bits: float,
    include_all_group: bool,
    curve_metric: str,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    dpi: int,
) -> tuple[Path, pd.DataFrame]:
    map_rows = rows
    x = np.asarray(diagnostics["across_values"], dtype=np.float64)
    unit_log2 = np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float64)
    population_log2 = np.log2(np.asarray(diagnostics["population_ratio"], dtype=np.float64))
    abs_x, unit_abs, population_abs = absolute_unit_ssi_matrices(stats_by_condition=stats_by_condition, rows=rows)
    if not np.allclose(x, abs_x, equal_nan=True):
        raise RuntimeError("Absolute SSI x values do not match ratio diagnostics.")
    metric = str(curve_metric)

    if float(min_static_ssi_bits) <= 0.0:
        retained_df = polarity_df.copy()
    else:
        retained_df = polarity_df[
            pd.to_numeric(polarity_df["static_unit_ssi_bits_per_spike_mean"], errors="coerce").ge(
                float(min_static_ssi_bits)
            )
        ].copy()
    all_units = retained_df["unit_index"].astype(int).tolist()
    positive_units = retained_df[retained_df["polarity"].eq("positive")]["unit_index"].astype(int).tolist()
    negative_units = retained_df[retained_df["polarity"].eq("negative")]["unit_index"].astype(int).tolist()
    group_specs: list[tuple[str, list[int], str]] = []
    if bool(include_all_group):
        group_specs.append(("all", all_units, "#111111"))
    group_specs.extend(
        [
            ("positive", positive_units, "#4a4a4a"),
            ("negative", negative_units, "#9a9a9a"),
        ]
    )

    group_rows: list[dict[str, Any]] = []
    group_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_maps: dict[str, np.ndarray] = {}
    filtered_units = float(min_static_ssi_bits) > 0.0

    def display_group_name(group_name: str) -> str:
        if filtered_units and group_name == "all":
            return "all retained"
        return group_name

    for group_name, unit_indices, _color in group_specs:
        if metric == "absolute":
            group_matrix = unit_abs[unit_indices, :] if unit_indices else np.full((0, unit_abs.shape[1]), np.nan)
        else:
            group_matrix = unit_log2[unit_indices, :] if unit_indices else np.full((0, unit_log2.shape[1]), np.nan)
        group_mean, group_sem = mean_sem(group_matrix, axis=0) if unit_indices else (
            np.full(group_matrix.shape[1], np.nan),
            np.full(group_matrix.shape[1], np.nan),
        )
        group_curves[group_name] = (group_mean, group_sem)
        group_maps[group_name] = normalized_group_maps(
            stats_by_condition=stats_by_condition,
            map_rows=map_rows,
            unit_indices=unit_indices,
            map_vmin_percentile=map_vmin_percentile,
            map_vmax_percentile=map_vmax_percentile,
        )
        for idx, across in enumerate(x):
            row_out = {
                "mode": mode,
                "polarity": group_name,
                "curve_metric": metric,
                "n_units": int(len(unit_indices)),
                "min_static_ssi_bits": float(min_static_ssi_bits),
                "across_scale": float(across),
                "along_scale": 0.0,
            }
            if metric == "absolute":
                row_out.update(
                    {
                        "mean_unit_ssi_bits_per_spike": float(group_mean[idx]),
                        "sem_unit_ssi_bits_per_spike": float(group_sem[idx]),
                        "population_ssi_bits_per_spike": float(population_abs[idx]),
                    }
                )
            else:
                row_out.update(
                    {
                        "mean_unit_log2_ssi_vs_static": float(group_mean[idx]),
                        "geometric_mean_ssi_vs_static": float(2.0 ** group_mean[idx]),
                        "sem_unit_log2_ssi_vs_static": float(group_sem[idx]),
                        "population_log2_ssi_vs_static": float(population_log2[idx]),
                    }
                )
            group_rows.append(row_out)

    mode_label = "real-trace" if mode == "real_trace" else "endpoint-history"
    png = out_dir / f"{output_stem}_group_averages.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_width = max(12.6, 1.08 * (len(map_rows) + 1))
    fig_height = max(6.8, 4.25 + 1.25 * len(group_specs))
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(dpi))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[2.35, max(2.15, 0.98 * len(group_specs))],
        hspace=0.24,
    )
    ax = fig.add_subplot(outer[0, 0])
    if metric == "absolute":
        ax.plot(
            x,
            population_abs,
            color="#111111",
            marker=".",
            linewidth=1.2,
            markersize=3.2,
            linestyle=":",
            label="full population SSI",
        )
    elif float(min_static_ssi_bits) <= 0.0 and not bool(include_all_group):
        ax.plot(x, population_log2, color="black", marker="o", linewidth=2.2, markersize=4.2, label="population")
    elif float(min_static_ssi_bits) > 0.0:
        ax.plot(
            x,
            population_log2,
            color="#111111",
            marker=".",
            linewidth=1.2,
            markersize=3.2,
            linestyle=":",
            label="full population ref.",
        )
    for group_name, unit_indices, color in group_specs:
        mean, sem = group_curves[group_name]
        label = f"{display_group_name(group_name)} units (n={len(unit_indices)})"
        ax.plot(x, mean, marker="o", linewidth=1.9, markersize=4.0, color=color, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0.0)
    if metric == "absolute":
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    else:
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    if metric == "absolute":
        ax.set_ylabel("mean unit SSI (bits/spike)")
        ax.set_title("Retained-unit mean absolute SSI curves")
    else:
        ax.set_ylabel("mean unit log2 SSI / own static SSI")
        ax.set_title("Retained-unit geometric-mean SSI-ratio curves")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.0, loc="best")

    map_grid = outer[1].subgridspec(
        nrows=len(group_specs) + 1,
        ncols=len(map_rows) + 1,
        height_ratios=[0.28, *([1.0] * len(group_specs))],
        width_ratios=[1.0, *([1.0] * len(map_rows))],
        hspace=0.08,
        wspace=0.04,
    )
    header = fig.add_subplot(map_grid[0, 0])
    header.axis("off")
    header.text(0.98, 0.5, "group", ha="right", va="center", fontsize=7.0, color="#555555")
    for col_idx, row in enumerate(map_rows):
        ax_head = fig.add_subplot(map_grid[0, col_idx + 1])
        ax_head.axis("off")
        label = "static" if bool(row["is_static_baseline"]) else f"{float(row['across_scale']):g}x"
        ax_head.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold" if (not bool(row["is_static_baseline"]) and np.isclose(float(row["across_scale"]), 1.0)) else "normal",
            color="#333333",
        )
    for row_idx, (group_name, unit_indices, color) in enumerate(group_specs):
        ax_label = fig.add_subplot(map_grid[row_idx + 1, 0])
        ax_label.axis("off")
        ax_label.text(
            0.98,
            0.55,
            f"{display_group_name(group_name)}\nn={len(unit_indices)}",
            ha="right",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=color,
        )
        maps = group_maps[group_name]
        for col_idx in range(len(map_rows)):
            ax_map = fig.add_subplot(map_grid[row_idx + 1, col_idx + 1])
            ax_map.imshow(maps[col_idx], origin="lower", cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            for spine in ax_map.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.55)

    fig.suptitle(
        f"RR100 {mode_label} along=0 polarity-group averages\n"
        "polarity from static map dominant contrast; "
        f"retaining static SSI >= {float(min_static_ssi_bits):g} bits/spike",
        fontsize=11.5,
        y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.06)
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pd.DataFrame(group_rows)


def run_one(args: argparse.Namespace, mode: str) -> None:
    out_dir = Path(args.real_trace_dir if mode == "real_trace" else args.endpoint_dir)
    across_scales = parse_scale_list(args.across_scales)
    rows = condition_sequence(across_scales, float(args.along_scale))
    stats_by_condition = load_stats_by_condition(
        mode=mode,
        out_dir=out_dir,
        rows=rows,
        fd_step_arcmin=float(args.fd_step_arcmin),
        real_trace_max_frames=int(args.real_trace_max_frames),
    )
    unit_df, _top_df, diagnostics = summarize_units(stats_by_condition, rows)
    polarity_df = classify_unit_polarity(
        np.asarray(stats_by_condition["static_center"]["mean_rate_map"], dtype=np.float32),
        low_percentile=float(args.low_percentile),
        high_percentile=float(args.high_percentile),
    )
    unit_table = unit_df[unit_df["condition"].eq("static_center")][
        ["unit_index", "static_unit_ssi_bits_per_spike_mean", "static_unit_mean_rate_mean"]
    ].drop_duplicates()
    polarity_df = polarity_df.merge(unit_table, on="unit_index", how="left")
    if float(args.min_static_ssi_bits) <= 0.0:
        polarity_df["retained_by_static_ssi_floor"] = True
    else:
        polarity_df["retained_by_static_ssi_floor"] = pd.to_numeric(
            polarity_df["static_unit_ssi_bits_per_spike_mean"], errors="coerce"
        ).ge(float(args.min_static_ssi_bits))
    prefix = f"rr100_{mode}_along0_polarity"
    output_stem = prefix
    if str(args.curve_metric) == "absolute":
        output_stem += "_absolute_ssi"
    if float(args.min_static_ssi_bits) > 0.0:
        output_stem += f"_static_ssi_ge_{scale_token(float(args.min_static_ssi_bits))}"
    png, group_df = draw_group_figure(
        mode=mode,
        out_dir=out_dir,
        output_stem=output_stem,
        rows=rows,
        unit_df=unit_df,
        diagnostics=diagnostics,
        polarity_df=polarity_df,
        stats_by_condition=stats_by_condition,
        min_static_ssi_bits=float(args.min_static_ssi_bits),
        include_all_group=bool(args.include_all_group),
        curve_metric=str(args.curve_metric),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        dpi=int(args.dpi),
    )

    unit_csv = out_dir / f"{output_stem}_unit_table.csv"
    group_csv = out_dir / f"{output_stem}_group_summary.csv"
    manifest = out_dir / f"{output_stem}_group_average_manifest.json"
    polarity_df.to_csv(unit_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    payload = {
        "analysis": f"{output_stem}_group_averages",
        "mode": mode,
        "out_dir": out_dir,
        "figure_png": png,
        "unit_table_csv": unit_csv,
        "group_summary_csv": group_csv,
        "min_static_ssi_bits": float(args.min_static_ssi_bits),
        "include_all_group": bool(args.include_all_group),
        "curve_metric": str(args.curve_metric),
        "classification_reference": "static_center mean_rate_map",
        "classification_rule": f"positive if p{float(args.high_percentile):g}-median >= median-p{float(args.low_percentile):g}",
        "positive_unit_count": int(polarity_df["polarity"].eq("positive").sum()),
        "negative_unit_count": int(polarity_df["polarity"].eq("negative").sum()),
        "retained_unit_count": int(polarity_df["retained_by_static_ssi_floor"].sum()),
        "retained_positive_unit_count": int(
            polarity_df["polarity"].eq("positive").multiply(polarity_df["retained_by_static_ssi_floor"]).sum()
        ),
        "retained_negative_unit_count": int(
            polarity_df["polarity"].eq("negative").multiply(polarity_df["retained_by_static_ssi_floor"]).sum()
        ),
        "map_average_contract": "per-unit percentile-normalized maps averaged within polarity group",
        "map_colormap": "gray_monotonic",
    }
    manifest.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {mode} polarity group figure: {png}", flush=True)
    print(f"Wrote {mode} polarity unit table: {unit_csv}", flush=True)
    print(f"Wrote {mode} polarity group summary: {group_csv}", flush=True)
    print(
        polarity_df["polarity"].value_counts().reindex(["positive", "negative"], fill_value=0).to_string(),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    modes = ["real_trace", "endpoint"] if str(args.mode) == "both" else [str(args.mode)]
    for mode in modes:
        run_one(args, mode)


if __name__ == "__main__":
    main()

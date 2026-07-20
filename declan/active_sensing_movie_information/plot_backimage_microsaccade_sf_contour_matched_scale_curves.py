#!/usr/bin/env python3
"""Plot low- vs high-SF contour-matched unit curves for microsaccade snippets."""

from __future__ import annotations

import argparse
import json
import math
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_full_patched_v2"
)
DEFAULT_CACHE = DEFAULT_RUN_DIR / "cache/backimage_contour_axis_rr100_spatial_ssi_cache.npz"
DEFAULT_INVENTORY = DEFAULT_RUN_DIR / "movie_condition_inventory.csv"
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "sf_group_contour_matched_low_high_dynamic_log_gaussian_marginal_low0p05_high0p5"
GROUP_ORDER = ["low_sf", "high_sf"]
GROUP_COLORS = {"low_sf": "#1f77b4", "high_sf": "#d62728"}
GROUP_LABELS = {"low_sf": "low SF", "high_sf": "high SF"}
SSI_KEY = "unit_time_resolved_bits_per_movie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sf-groups", default="low_sf,high_sf")
    parser.add_argument("--alignment-angle-deg", type=float, default=22.5)
    parser.add_argument("--min-orientation-selectivity", type=float, default=0.05)
    parser.add_argument("--min-matched-movies-per-unit", type=int, default=1)
    parser.add_argument(
        "--axis-coordinate-frame",
        choices=("gaze", "image"),
        default="gaze",
        help="Frame for movie_condition_inventory.axis_deg. The contour sweep stores gaze-frame axes.",
    )
    parser.add_argument("--reference-condition-id", default="")
    parser.add_argument("--show-unit-curves", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def orientation_axis_180(angle_deg: float | np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(angle_deg, dtype=np.float64), 180.0)


def angle_180_distance(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64) + 90.0) % 180.0) - 90.0)


def contour_axis_to_image_frame(axis_deg: np.ndarray, coordinate_frame: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64)
    if str(coordinate_frame) == "gaze":
        axis = -axis
    return orientation_axis_180(axis)


def sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def condition_table(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    n_conditions = int(condition_id.size)
    along = np.asarray(stats["along_scale"], dtype=np.float64)
    across = np.asarray(stats["across_scale"], dtype=np.float64)
    motion = np.asarray(stats.get("motion_scale", np.maximum(np.abs(along), np.abs(across))), dtype=np.float64)
    return pd.DataFrame(
        {
            "condition_index": np.arange(n_conditions, dtype=int),
            "condition_id": condition_id,
            "condition_label": np.asarray(stats["condition_label"]).astype(str),
            "along_scale": along,
            "across_scale": across,
            "motion_scale": motion,
            "is_static_baseline": np.asarray(stats["is_static_baseline"], dtype=bool),
        }
    )


def reference_condition_index(conditions: pd.DataFrame, requested: str) -> int:
    requested = str(requested).strip()
    if requested:
        matches = conditions.index[conditions["condition_id"].astype(str).eq(requested)].to_numpy(dtype=int)
        if not len(matches):
            raise ValueError(f"reference condition {requested!r} was not found.")
        return int(matches[0])
    static = conditions.index[conditions["is_static_baseline"].astype(bool)].to_numpy(dtype=int)
    if len(static):
        return int(static[0])
    scale0 = conditions.index[np.isclose(conditions["motion_scale"].to_numpy(dtype=float), 0.0)].to_numpy(dtype=int)
    if len(scale0):
        return int(scale0[0])
    raise ValueError("Could not identify a static or 0x reference condition.")


def load_movie_table(
    inventory_csv: Path,
    stats: dict[str, np.ndarray],
    *,
    axis_coordinate_frame: str,
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_csv)
    required = {"movie_index", "condition_index", "trial_id", "source_row", "session", "trial_idx", "axis_deg"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValueError(f"{inventory_csv} is missing columns: {missing}")
    movies = inventory.sort_values(["movie_index", "condition_index"]).drop_duplicates("movie_index", keep="first").copy()
    n_movies = int(np.asarray(stats["movie_source_row"]).size)
    movies = movies.set_index("movie_index", drop=False).reindex(np.arange(n_movies))
    if movies["source_row"].isna().any():
        raise ValueError("movie_condition_inventory.csv is missing at least one cached movie_index.")
    cache_source = np.asarray(stats["movie_source_row"], dtype=int)
    inv_source = movies["source_row"].to_numpy(dtype=int)
    if not np.array_equal(cache_source, inv_source):
        raise ValueError("Cache movie_source_row order does not match movie_condition_inventory.csv.")
    movies["axis_deg"] = pd.to_numeric(movies["axis_deg"], errors="coerce")
    movies["contour_axis_image_deg"] = contour_axis_to_image_frame(
        movies["axis_deg"].to_numpy(dtype=float),
        axis_coordinate_frame,
    )
    movies["axis_coordinate_frame"] = str(axis_coordinate_frame)
    return movies.reset_index(drop=True)


def load_unit_table(sf_groups_csv: Path, sf_groups: list[str], n_units: int) -> pd.DataFrame:
    units = pd.read_csv(sf_groups_csv)
    required = {
        "unit_index",
        "unit_label",
        "sf_group",
        "sf_group_label",
        "sf_group_definition",
        "sf_split_metric",
        "sf_rank_low_to_high",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
    }
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"{sf_groups_csv} is missing columns: {missing}")
    units = units.copy()
    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))].copy()
    units = units[units["sf_group"].astype(str).isin(sf_groups)].copy()
    units["prior_preferred_orientation_deg"] = pd.to_numeric(units["prior_preferred_orientation_deg"], errors="coerce")
    units["preferred_orientation_image_deg"] = orientation_axis_180(
        units["prior_preferred_orientation_deg"].to_numpy(dtype=float)
    )
    units["prior_orientation_selectivity_index"] = pd.to_numeric(
        units["prior_orientation_selectivity_index"],
        errors="coerce",
    )
    units["sf_split_metric"] = pd.to_numeric(units["sf_split_metric"], errors="coerce")
    units["sf_rank_low_to_high"] = pd.to_numeric(units["sf_rank_low_to_high"], errors="coerce")
    return units.sort_values(["sf_group", "sf_rank_low_to_high", "unit_index"]).reset_index(drop=True)


def unit_selection_table(
    units: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    sf_groups: list[str],
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
) -> pd.DataFrame:
    movie_axes = movies["contour_axis_image_deg"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        preferred = float(unit.preferred_orientation_image_deg)
        osi = float(unit.prior_orientation_selectivity_index)
        orientation_ok = np.isfinite(preferred) and np.isfinite(osi) and osi >= float(min_orientation_selectivity)
        if orientation_ok:
            deltas = angle_180_distance(preferred, movie_axes)
            matched = np.isfinite(deltas) & (deltas <= float(alignment_angle_deg))
        else:
            deltas = np.full(movie_axes.shape, np.nan, dtype=np.float64)
            matched = np.zeros(movie_axes.shape, dtype=bool)
        rows.append(
            {
                "unit_index": int(unit.unit_index),
                "unit_label": str(unit.unit_label),
                "sf_group": str(unit.sf_group),
                "sf_group_label": str(unit.sf_group_label),
                "sf_group_definition": str(unit.sf_group_definition),
                "sf_split_metric": float(unit.sf_split_metric),
                "sf_rank_low_to_high": float(unit.sf_rank_low_to_high),
                "prior_preferred_orientation_deg": float(unit.prior_preferred_orientation_deg),
                "preferred_orientation_image_deg": preferred,
                "prior_orientation_selectivity_index": osi,
                "passes_orientation_selectivity": bool(orientation_ok),
                "n_matched_movies": int(np.sum(matched)),
                "fraction_matched_movies": float(np.mean(matched)) if matched.size else float("nan"),
                "mean_delta_from_contour_deg": float(np.nanmean(deltas[matched])) if np.any(matched) else float("nan"),
                "median_delta_from_contour_deg": float(np.nanmedian(deltas[matched])) if np.any(matched) else float("nan"),
                "matched_movie_indices": " ".join(str(int(v)) for v in np.flatnonzero(matched)),
            }
        )
    selection = pd.DataFrame(rows)
    if selection.empty:
        raise ValueError(f"No units found for sf_groups={sf_groups}.")
    return selection


def build_unit_curves(
    stats: dict[str, np.ndarray],
    conditions: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    min_matched_movies_per_unit: int,
    reference_index: int,
) -> pd.DataFrame:
    bits = np.asarray(stats[SSI_KEY], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    n_conditions, n_movies, n_units = bits.shape
    if n_conditions != conditions.shape[0]:
        raise ValueError("Cache condition count does not match condition table.")
    rows: list[dict[str, Any]] = []
    for unit in selection.itertuples(index=False):
        matched = np.fromstring(str(unit.matched_movie_indices), sep=" ", dtype=int)
        if matched.size < int(min_matched_movies_per_unit):
            continue
        unit_index = int(unit.unit_index)
        if unit_index < 0 or unit_index >= n_units:
            continue
        if matched.size and (int(np.nanmin(matched)) < 0 or int(np.nanmax(matched)) >= n_movies):
            raise ValueError(f"Matched movie index out of range for unit {unit_index}.")
        reference_bits = float(np.nanmean(bits[reference_index, matched, unit_index]))
        reference_rate = float(np.nanmean(rates[reference_index, matched, unit_index]))
        reference_spikes = float(np.nanmean(spikes[reference_index, matched, unit_index]))
        for condition in conditions.itertuples(index=False):
            cidx = int(condition.condition_index)
            mean_bits = float(np.nanmean(bits[cidx, matched, unit_index]))
            rows.append(
                {
                    "unit_index": unit_index,
                    "unit_label": str(unit.unit_label),
                    "sf_group": str(unit.sf_group),
                    "sf_group_label": str(unit.sf_group_label),
                    "sf_group_definition": str(unit.sf_group_definition),
                    "sf_split_metric": float(unit.sf_split_metric),
                    "sf_rank_low_to_high": float(unit.sf_rank_low_to_high),
                    "prior_preferred_orientation_deg": float(unit.prior_preferred_orientation_deg),
                    "preferred_orientation_image_deg": float(unit.preferred_orientation_image_deg),
                    "prior_orientation_selectivity_index": float(unit.prior_orientation_selectivity_index),
                    "n_matched_movies": int(unit.n_matched_movies),
                    "fraction_matched_movies": float(unit.fraction_matched_movies),
                    "mean_delta_from_contour_deg": float(unit.mean_delta_from_contour_deg),
                    "median_delta_from_contour_deg": float(unit.median_delta_from_contour_deg),
                    "condition_index": cidx,
                    "condition_id": str(condition.condition_id),
                    "condition_label": str(condition.condition_label),
                    "along_scale": float(condition.along_scale),
                    "across_scale": float(condition.across_scale),
                    "motion_scale": float(condition.motion_scale),
                    "is_static_baseline": bool(condition.is_static_baseline),
                    "unit_contour_matched_ssi_bits_per_spike": mean_bits,
                    "unit_contour_matched_ssi_at_reference": reference_bits,
                    "unit_contour_matched_ssi_delta_vs_reference": mean_bits - reference_bits,
                    "unit_contour_matched_mean_rate": float(np.nanmean(rates[cidx, matched, unit_index])),
                    "unit_contour_matched_mean_rate_at_reference": reference_rate,
                    "unit_contour_matched_expected_spikes": float(np.nanmean(spikes[cidx, matched, unit_index])),
                    "unit_contour_matched_expected_spikes_at_reference": reference_spikes,
                }
            )
    curves = pd.DataFrame(rows)
    if curves.empty:
        raise ValueError("No contour-matched unit curves survived the support filters.")
    return curves.sort_values(["sf_group", "unit_index", "motion_scale"]).reset_index(drop=True)


def summarize(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_cols = [
        "unit_contour_matched_ssi_bits_per_spike",
        "unit_contour_matched_ssi_delta_vs_reference",
        "unit_contour_matched_mean_rate",
    ]
    for (sf_group, motion_scale), sub in curves.groupby(["sf_group", "motion_scale"], sort=True):
        for col in value_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            rows.append(
                {
                    "sf_group": str(sf_group),
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "sf_group_definition": str(sub["sf_group_definition"].iloc[0]),
                    "motion_scale": float(motion_scale),
                    "value_name": col,
                    "n_units": int(sub["unit_index"].nunique()),
                    "n_finite": int(finite.sum()),
                    "mean": float(np.nanmean(values)),
                    "sem": sem(values),
                    "median": float(np.nanmedian(values)),
                    "mean_matched_movies_per_unit": float(np.nanmean(sub["n_matched_movies"].to_numpy(dtype=float))),
                    "median_matched_movies_per_unit": float(np.nanmedian(sub["n_matched_movies"].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def plot(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
    min_matched_movies_per_unit: int,
    axis_coordinate_frame: str,
    dpi: int,
    show_unit_curves: bool,
) -> tuple[Path, Path]:
    definition = str(curves["sf_group_definition"].dropna().iloc[0]) if curves["sf_group_definition"].notna().any() else ""
    if "low_sf <= 0.05" in definition and "high_sf >= 0.5" in definition:
        definition_text = "low SF <= 0.05 cpd vs high SF >= 0.5 cpd"
    else:
        definition_text = definition
    panels = [
        (
            "unit_contour_matched_ssi_bits_per_spike",
            "SSI (bits/spike)",
            "Absolute SSI on matched windows",
        ),
        (
            "unit_contour_matched_ssi_delta_vs_reference",
            "SSI minus 0x (bits/spike)",
            "Scale modulation",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    for ax, (value_name, ylabel, title) in zip(axes, panels, strict=True):
        for sf_group in GROUP_ORDER:
            if sf_group not in set(curves["sf_group"].astype(str)):
                continue
            color = GROUP_COLORS.get(sf_group, "0.2")
            unit_sub = curves[curves["sf_group"].astype(str).eq(sf_group)]
            if show_unit_curves:
                for _, per_unit in unit_sub.groupby("unit_index", sort=False):
                    per_unit = per_unit.sort_values("motion_scale")
                    ax.plot(
                        per_unit["motion_scale"].to_numpy(dtype=float),
                        per_unit[value_name].to_numpy(dtype=float),
                        color=color,
                        alpha=0.13,
                        linewidth=0.8,
                        zorder=1,
                    )
            mean_sub = summary[
                summary["sf_group"].astype(str).eq(sf_group)
                & summary["value_name"].astype(str).eq(value_name)
            ].sort_values("motion_scale")
            if mean_sub.empty:
                continue
            x = mean_sub["motion_scale"].to_numpy(dtype=float)
            y = mean_sub["mean"].to_numpy(dtype=float)
            e = mean_sub["sem"].to_numpy(dtype=float)
            label = f"{GROUP_LABELS.get(sf_group, sf_group)} (n={int(mean_sub['n_units'].iloc[0])})"
            ax.plot(x, y, marker="o", linewidth=2.3, markersize=4.8, color=color, label=label, zorder=4)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0)
        if value_name == "unit_contour_matched_ssi_delta_vs_reference":
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("microsaccade trace scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Contour-matched unit-window pairs: microsaccade trace scale setting\n"
        f"{definition_text}; axis={axis_coordinate_frame}; align <= {alignment_angle_deg:g} deg; "
        f"OSI >= {min_orientation_selectivity:g}; min windows/unit = {min_matched_movies_per_unit}",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    png = out_dir / "backimage_microsaccade_sf_group_contour_matched_low_high_scale_curves.png"
    pdf = out_dir / "backimage_microsaccade_sf_group_contour_matched_low_high_scale_curves.pdf"
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sf_groups = parse_csv_list(str(args.sf_groups))
    stats = load_cache(args.cache)
    bits = np.asarray(stats[SSI_KEY])
    conditions = condition_table(stats)
    reference_index = reference_condition_index(conditions, str(args.reference_condition_id))
    movies = load_movie_table(args.inventory_csv, stats, axis_coordinate_frame=str(args.axis_coordinate_frame))
    units = load_unit_table(args.sf_groups_csv, sf_groups, int(bits.shape[2]))
    selection = unit_selection_table(
        units,
        movies,
        sf_groups=sf_groups,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
    )
    curves = build_unit_curves(
        stats,
        conditions,
        selection,
        min_matched_movies_per_unit=int(args.min_matched_movies_per_unit),
        reference_index=reference_index,
    )
    summary = summarize(curves)

    selection_csv = args.out_dir / "microsaccade_sf_group_contour_matched_unit_selection.csv"
    curves_csv = args.out_dir / "microsaccade_sf_group_contour_matched_unit_curves.csv"
    summary_csv = args.out_dir / "microsaccade_sf_group_contour_matched_summary.csv"
    selection.to_csv(selection_csv, index=False)
    curves.to_csv(curves_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    png, pdf = plot(
        curves,
        summary,
        args.out_dir,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
        min_matched_movies_per_unit=int(args.min_matched_movies_per_unit),
        axis_coordinate_frame=str(args.axis_coordinate_frame),
        dpi=int(args.dpi),
        show_unit_curves=bool(args.show_unit_curves),
    )
    reference_condition_id = str(conditions.loc[reference_index, "condition_id"])
    write_json(
        args.out_dir / "summary.json",
        {
            "analysis": "backimage_microsaccade_sf_group_contour_matched_scale_curves",
            "cache": args.cache,
            "inventory_csv": args.inventory_csv,
            "sf_groups_csv": args.sf_groups_csv,
            "out_dir": args.out_dir,
            "ssi_metric_key": SSI_KEY,
            "sf_groups": sf_groups,
            "axis_coordinate_frame": str(args.axis_coordinate_frame),
            "axis_frame_contract": (
                "If axis_coordinate_frame is gaze, axis_deg is converted to image-array coordinates as -axis_deg mod 180 "
                "before comparison to prior_preferred_orientation_deg."
            ),
            "alignment_angle_deg": float(args.alignment_angle_deg),
            "min_orientation_selectivity": float(args.min_orientation_selectivity),
            "min_matched_movies_per_unit": int(args.min_matched_movies_per_unit),
            "reference_condition_id": reference_condition_id,
            "n_movies": int(bits.shape[1]),
            "n_conditions": int(bits.shape[0]),
            "n_units_total": int(bits.shape[2]),
            "n_units_by_group_after_matching": curves.groupby("sf_group")["unit_index"].nunique().to_dict(),
            "matched_movies_by_group": (
                curves.drop_duplicates(["sf_group", "unit_index"])
                .groupby("sf_group")["n_matched_movies"]
                .agg(["min", "median", "max"])
                .to_dict(orient="index")
            ),
            "outputs": {
                "selection": selection_csv,
                "unit_curves": curves_csv,
                "summary": summary_csv,
                "png": png,
                "pdf": pdf,
            },
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(
        summary[summary["value_name"].eq("unit_contour_matched_ssi_delta_vs_reference")]
        .sort_values(["sf_group", "motion_scale"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

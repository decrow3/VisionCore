#!/usr/bin/env python3
"""Example-unit diagnostics for SF x contour-alignment RR100 SSI plots.

This is a lightweight post hoc script: it reads the cached contour-axis SSI
arrays and the strict SF-tuning table, then selects a small set of units that
help explain the continuous-weight absolute population plots.  The examples are
selected by weighted static leverage and static-to-2x modulation for pure
across-contour motion with along=0, then visualized across the four sweep views.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_backimage_rr100_sf_contour_alignment_population_ssi import (
    ALIGNMENT_GROUPS,
    EPS,
    axial_alignment_score,
    cache_path,
    condition_frame,
    contour_axis_to_image_frame,
    json_ready,
    load_movie_frame,
    load_npz_without_identity,
    load_unit_table,
    metric_array,
    metric_label,
    orientation_axis_180,
    parse_csv_list,
    sf_label,
    write_json,
)


DEFAULT_CONTOUR_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_plus_primary_across_sweep_merged_v1"
)
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_example_units_dynamic_log_gaussian_marginal_low0p05_high0p5_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contour-run-dir", type=Path, default=DEFAULT_CONTOUR_RUN_DIR)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ssi-metric", choices=("time_resolved", "mean_map"), default="time_resolved")
    parser.add_argument("--sf-groups", type=str, default="low_sf,high_sf")
    parser.add_argument("--min-orientation-selectivity", type=float, default=0.05)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


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


def unit_weight_tables(
    units: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    min_orientation_selectivity: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_movies = int(movies.shape[0])
    n_units = int(units["unit_index"].max()) + 1
    weights = {group: np.zeros((n_movies, n_units), dtype=np.float64) for group in ALIGNMENT_GROUPS}
    signed = np.full((n_movies, n_units), np.nan, dtype=np.float64)
    unit_meta = units.set_index("unit_index", drop=False)
    preferred = unit_meta["preferred_orientation_image_deg"].to_dict()
    osi = unit_meta["prior_orientation_selectivity_index"].to_dict()
    for movie in movies.itertuples(index=False):
        contour = float(movie.contour_axis_image_deg)
        for unit_idx in unit_meta.index.to_numpy(dtype=int):
            pref = float(preferred[int(unit_idx)])
            selectivity = float(osi[int(unit_idx)])
            if not np.isfinite(pref) or not np.isfinite(selectivity) or selectivity < float(min_orientation_selectivity):
                continue
            score = float(axial_alignment_score(np.asarray([pref], dtype=float), contour)[0])
            signed[int(movie.movie_index), int(unit_idx)] = score
            weights["contour_aligned"][int(movie.movie_index), int(unit_idx)] = max(score, 0.0)
            weights["contour_orthogonal"][int(movie.movie_index), int(unit_idx)] = max(-score, 0.0)
    return weights, signed


def condition_index(conditions: pd.DataFrame, condition_id: str) -> int:
    matches = conditions[conditions["condition_id"].astype(str) == str(condition_id)]
    if matches.empty:
        raise ValueError(f"Missing condition {condition_id!r}")
    return int(matches.iloc[0]["condition_index"])


def static_baseline_row(conditions: pd.DataFrame) -> pd.Series:
    if "is_static_baseline" in conditions.columns:
        raw_static = conditions["is_static_baseline"]
        if raw_static.dtype == bool:
            static_mask = raw_static.to_numpy(dtype=bool)
        else:
            static_mask = raw_static.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy(dtype=bool)
        static = conditions[static_mask].copy()
        if not static.empty:
            return static.sort_values("condition_index").iloc[0]
    for condition_id in ("static_along0_across0", "along0_across0"):
        matches = conditions[conditions["condition_id"].astype(str) == condition_id]
        if not matches.empty:
            return matches.iloc[0]
    scale_match = conditions[
        np.isclose(conditions["along_scale"].to_numpy(dtype=float), 0.0)
        & np.isclose(conditions["across_scale"].to_numpy(dtype=float), 0.0)
    ]
    if not scale_match.empty:
        return scale_match.sort_values("condition_index").iloc[0]
    raise ValueError("Missing static baseline condition.")


def static_baseline_index(conditions: pd.DataFrame) -> int:
    return int(static_baseline_row(conditions)["condition_index"])


def static_baseline_condition_id(conditions: pd.DataFrame) -> str:
    return str(static_baseline_row(conditions)["condition_id"])


def condition_subset(
    conditions: pd.DataFrame,
    *,
    sweep_axis: str,
    fixed_along_scale: float | None = None,
    fixed_across_scale: float | None = None,
) -> pd.DataFrame:
    keep = np.ones(int(conditions.shape[0]), dtype=bool)
    if fixed_along_scale is not None:
        keep &= np.isclose(conditions["along_scale"].to_numpy(dtype=float), float(fixed_along_scale))
    if fixed_across_scale is not None:
        keep &= np.isclose(conditions["across_scale"].to_numpy(dtype=float), float(fixed_across_scale))
    out = conditions.loc[keep].copy()
    if sweep_axis == "across":
        out["_x"] = out["across_scale"].to_numpy(dtype=float)
    elif sweep_axis == "along":
        out["_x"] = out["along_scale"].to_numpy(dtype=float)
    else:
        raise ValueError(sweep_axis)
    return out.sort_values(["_x", "condition_index"]).reset_index(drop=True)


def curve_for_unit(
    bits: np.ndarray,
    spikes: np.ndarray,
    unit_idx: int,
    condition_rows: pd.DataFrame,
    *,
    movie_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    denoms: list[float] = []
    if movie_weight is None:
        movie_weight = np.ones(bits.shape[1], dtype=np.float64)
    for _, row in condition_rows.iterrows():
        cidx = int(row["condition_index"])
        unit_spikes = spikes[cidx, :, int(unit_idx)] * movie_weight
        unit_bits = bits[cidx, :, int(unit_idx)]
        denom = float(np.nansum(unit_spikes))
        numer = float(np.nansum(unit_bits * unit_spikes))
        xs.append(float(row["_x"]))
        ys.append(numer / max(denom, EPS) if denom > EPS else float("nan"))
        denoms.append(denom)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(denoms, dtype=float)


def role_candidates(
    bits: np.ndarray,
    spikes: np.ndarray,
    units: pd.DataFrame,
    conditions: pd.DataFrame,
    weights: dict[str, np.ndarray],
) -> pd.DataFrame:
    static_idx = static_baseline_index(conditions)
    endpoint_idx = condition_index(conditions, "along0_across2")
    rows: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        unit_idx = int(unit.unit_index)
        for alignment_group in ALIGNMENT_GROUPS:
            movie_weight = weights[alignment_group][:, unit_idx]
            static_spikes = spikes[static_idx, :, unit_idx] * movie_weight
            endpoint_spikes = spikes[endpoint_idx, :, unit_idx] * movie_weight
            static_den = float(np.nansum(static_spikes))
            endpoint_den = float(np.nansum(endpoint_spikes))
            static_info = float(np.nansum(bits[static_idx, :, unit_idx] * static_spikes))
            endpoint_info = float(np.nansum(bits[endpoint_idx, :, unit_idx] * endpoint_spikes))
            static_bits = static_info / max(static_den, EPS) if static_den > EPS else float("nan")
            endpoint_bits = endpoint_info / max(endpoint_den, EPS) if endpoint_den > EPS else float("nan")
            rows.append(
                {
                    "unit_index": unit_idx,
                    "unit_label": str(unit.unit_label),
                    "sf_group": str(unit.sf_group),
                    "alignment_group": alignment_group,
                    "static_weighted_information": static_info,
                    "endpoint_weighted_information": endpoint_info,
                    "static_weighted_bits_per_spike": static_bits,
                    "endpoint_weighted_bits_per_spike": endpoint_bits,
                    "endpoint_minus_static_bits_per_spike": endpoint_bits - static_bits,
                    "static_weighted_expected_spikes": static_den,
                    "endpoint_weighted_expected_spikes": endpoint_den,
                    "mean_alignment_weight": float(np.nanmean(movie_weight)),
                    "fraction_positive_alignment_weight": float(np.nanmean(movie_weight > EPS)),
                    "prior_preferred_orientation_deg": float(unit.prior_preferred_orientation_deg),
                    "prior_orientation_selectivity_index": float(unit.prior_orientation_selectivity_index),
                    "sf_split_metric": float(unit.sf_split_metric),
                }
            )
    return pd.DataFrame(rows)


def choose_examples(candidates: pd.DataFrame, *, n_examples: int) -> list[dict[str, Any]]:
    specs = [
        (
            "high_sf_orthogonal_static_leverage",
            (candidates["sf_group"] == "high_sf") & (candidates["alignment_group"] == "contour_orthogonal"),
            "static_weighted_information",
            False,
        ),
        (
            "high_sf_aligned_static_leverage",
            (candidates["sf_group"] == "high_sf") & (candidates["alignment_group"] == "contour_aligned"),
            "static_weighted_information",
            False,
        ),
        (
            "high_sf_motion_fragile",
            candidates["sf_group"] == "high_sf",
            "endpoint_minus_static_bits_per_spike",
            True,
        ),
        (
            "low_sf_aligned_motion_gain",
            (candidates["sf_group"] == "low_sf") & (candidates["alignment_group"] == "contour_aligned"),
            "endpoint_minus_static_bits_per_spike",
            False,
        ),
        (
            "low_sf_orthogonal_motion_gain",
            (candidates["sf_group"] == "low_sf") & (candidates["alignment_group"] == "contour_orthogonal"),
            "endpoint_minus_static_bits_per_spike",
            False,
        ),
    ]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for role, mask, sort_col, ascending in specs:
        pool = candidates.loc[mask & np.isfinite(candidates[sort_col].to_numpy(dtype=float))].copy()
        pool = pool[pool["static_weighted_expected_spikes"].to_numpy(dtype=float) > 0.05]
        pool = pool.sort_values(sort_col, ascending=ascending)
        for row in pool.to_dict(orient="records"):
            unit_idx = int(row["unit_index"])
            if unit_idx in used:
                continue
            row["selection_role"] = role
            selected.append(row)
            used.add(unit_idx)
            break
        if len(selected) >= int(n_examples):
            break
    return selected[: int(n_examples)]


def plot_example_curves(
    out_dir: Path,
    bits: np.ndarray,
    spikes: np.ndarray,
    conditions: pd.DataFrame,
    weights: dict[str, np.ndarray],
    examples: list[dict[str, Any]],
    *,
    metric_name: str,
    dpi: int,
) -> tuple[Path, Path]:
    sweep_specs = [
        ("across", 0.0, None, "across scale\nalong=0"),
        ("across", 1.0, None, "across scale\nalong=1"),
        ("along", None, 0.0, "along scale\nacross=0"),
        ("along", None, 1.0, "along scale\nacross=1"),
    ]
    fig, axes = plt.subplots(
        len(examples),
        len(sweep_specs),
        figsize=(13.6, 2.25 * len(examples)),
        sharey="row",
        constrained_layout=True,
    )
    if len(examples) == 1:
        axes = np.asarray([axes])
    for row_idx, example in enumerate(examples):
        unit_idx = int(example["unit_index"])
        for col_idx, (sweep_axis, fixed_along, fixed_across, title) in enumerate(sweep_specs):
            ax = axes[row_idx, col_idx]
            condition_rows = condition_subset(
                conditions,
                sweep_axis=sweep_axis,
                fixed_along_scale=fixed_along,
                fixed_across_scale=fixed_across,
            )
            x, y_all, _ = curve_for_unit(bits, spikes, unit_idx, condition_rows)
            ax.plot(x, y_all, color="0.2", marker="o", linewidth=1.8, markersize=3.5, label="all fixations")
            for alignment_group, color in [("contour_aligned", "#168a96"), ("contour_orthogonal", "#c06b2d")]:
                _, y_weighted, den = curve_for_unit(
                    bits,
                    spikes,
                    unit_idx,
                    condition_rows,
                    movie_weight=weights[alignment_group][:, unit_idx],
                )
                linestyle = "-" if np.nanmax(den) > EPS else "--"
                ax.plot(
                    x,
                    y_weighted,
                    color=color,
                    marker="o",
                    linewidth=1.55,
                    markersize=3.2,
                    linestyle=linestyle,
                    label=alignment_group.replace("contour_", ""),
                )
            ax.axvline(1.0, color="0.65", linestyle=":", linewidth=0.9)
            ax.grid(True, color="0.9", linewidth=0.65)
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                role = str(example["selection_role"]).replace("_", " ")
                ylabel = (
                    f"{example['unit_label']}  {sf_label(str(example['sf_group']))}\n"
                    f"{role}\n"
                    f"SF={float(example['sf_split_metric']):.3g} cpd; "
                    f"pref={float(example['prior_preferred_orientation_deg']):.1f} deg"
                )
                ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=86, fontsize=7.3)
            if row_idx == len(examples) - 1:
                ax.set_xlabel("scale")
            if row_idx == 0 and col_idx == len(sweep_specs) - 1:
                ax.legend(frameon=False, fontsize=7.2, loc="best")
    fig.suptitle(
        "BackImage RR100 example units behind continuous-weight SF/alignment curves\n"
        f"{metric_name}; each unit is shown all-fixation and with per-fixation alignment weights",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_sf_contour_alignment_example_unit_curves.png"
    pdf = out_dir / "backimage_rr100_sf_contour_alignment_example_unit_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_alignment_weight_distributions(
    out_dir: Path,
    signed_alignment: np.ndarray,
    examples: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(examples), 1, figsize=(7.2, 1.65 * len(examples)), sharex=True, constrained_layout=True)
    if len(examples) == 1:
        axes = np.asarray([axes])
    bins = np.linspace(-1.0, 1.0, 25)
    for ax, example in zip(axes, examples, strict=True):
        unit_idx = int(example["unit_index"])
        values = signed_alignment[:, unit_idx]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, color="0.35", alpha=0.72)
        ax.axvline(0.0, color="0.2", linestyle="--", linewidth=0.8)
        ax.axvspan(0.0, 1.0, color="#168a96", alpha=0.08)
        ax.axvspan(-1.0, 0.0, color="#c06b2d", alpha=0.08)
        ax.set_ylabel(str(example["unit_label"]), rotation=0, ha="right", va="center", labelpad=28)
        ax.grid(True, axis="y", color="0.9", linewidth=0.65)
    axes[-1].set_xlabel("signed contour-alignment score: cos(2 * orientation delta)")
    fig.suptitle("How each example unit moves between aligned and orthogonal weighted pools", fontsize=11.5)
    png = out_dir / "backimage_rr100_sf_contour_alignment_example_unit_alignment_weights.png"
    pdf = out_dir / "backimage_rr100_sf_contour_alignment_example_unit_alignment_weights.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def high_sf_spaghetti_rows(
    bits: np.ndarray,
    spikes: np.ndarray,
    conditions: pd.DataFrame,
    weights: dict[str, np.ndarray],
    units: pd.DataFrame,
) -> list[dict[str, Any]]:
    sweep_specs = [
        ("across_along0", "across", 0.0, None),
        ("across_along1", "across", 1.0, None),
        ("along_across0", "along", None, 0.0),
        ("along_across1", "along", None, 1.0),
    ]
    high_units = units[units["sf_group"].astype(str) == "high_sf"].copy()
    rows: list[dict[str, Any]] = []
    for unit in high_units.itertuples(index=False):
        unit_idx = int(unit.unit_index)
        for sweep_name, sweep_axis, fixed_along, fixed_across in sweep_specs:
            condition_rows = condition_subset(
                conditions,
                sweep_axis=sweep_axis,
                fixed_along_scale=fixed_along,
                fixed_across_scale=fixed_across,
            )
            x, y_all, denom_all = curve_for_unit(bits, spikes, unit_idx, condition_rows)
            _, y_aligned, denom_aligned = curve_for_unit(
                bits,
                spikes,
                unit_idx,
                condition_rows,
                movie_weight=weights["contour_aligned"][:, unit_idx],
            )
            _, y_orthogonal, denom_orthogonal = curve_for_unit(
                bits,
                spikes,
                unit_idx,
                condition_rows,
                movie_weight=weights["contour_orthogonal"][:, unit_idx],
            )
            for idx, condition in enumerate(condition_rows.itertuples(index=False)):
                rows.append(
                    {
                        "unit_index": unit_idx,
                        "unit_label": str(unit.unit_label),
                        "sf_group": str(unit.sf_group),
                        "sf_split_metric": float(unit.sf_split_metric),
                        "prior_preferred_orientation_deg": float(unit.prior_preferred_orientation_deg),
                        "prior_orientation_selectivity_index": float(unit.prior_orientation_selectivity_index),
                        "sweep_view": sweep_name,
                        "sweep_axis": sweep_axis,
                        "fixed_along_scale": fixed_along,
                        "fixed_across_scale": fixed_across,
                        "condition_id": str(condition.condition_id),
                        "condition_index": int(condition.condition_index),
                        "x_scale": float(x[idx]),
                        "all_fixations_bits_per_spike": float(y_all[idx]),
                        "aligned_weighted_bits_per_spike": float(y_aligned[idx]),
                        "orthogonal_weighted_bits_per_spike": float(y_orthogonal[idx]),
                        "aligned_minus_orthogonal_bits_per_spike": float(y_aligned[idx] - y_orthogonal[idx]),
                        "all_fixations_expected_spikes": float(denom_all[idx]),
                        "aligned_weighted_expected_spikes": float(denom_aligned[idx]),
                        "orthogonal_weighted_expected_spikes": float(denom_orthogonal[idx]),
                    }
                )
    return rows


def plot_high_sf_spaghetti(
    out_dir: Path,
    spaghetti_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    dpi: int,
) -> tuple[Path, Path, Path, Path]:
    df = pd.DataFrame(spaghetti_rows)
    sweep_order = ["across_along0", "across_along1", "along_across0", "along_across1"]
    sweep_titles = {
        "across_along0": "across scale\nalong=0",
        "across_along1": "across scale\nalong=1",
        "along_across0": "along scale\nacross=0",
        "along_across1": "along scale\nacross=1",
    }
    units_total = sorted(int(v) for v in df["unit_index"].unique())
    usable_units: list[int] = []
    for unit_idx in units_total:
        unit_sub = df[df["unit_index"].astype(int) == int(unit_idx)]
        if (
            np.isfinite(unit_sub["aligned_weighted_bits_per_spike"].to_numpy(dtype=float)).any()
            and np.isfinite(unit_sub["orthogonal_weighted_bits_per_spike"].to_numpy(dtype=float)).any()
        ):
            usable_units.append(int(unit_idx))
    units = usable_units

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True)
    for ax, sweep_view in zip(axes.ravel(), sweep_order, strict=True):
        sub = df[df["sweep_view"].astype(str) == sweep_view].copy()
        for unit_idx in units:
            unit_sub = sub[sub["unit_index"].astype(int) == int(unit_idx)].sort_values("x_scale")
            if unit_sub.empty:
                continue
            x = unit_sub["x_scale"].to_numpy(dtype=float)
            ax.plot(
                x,
                unit_sub["aligned_weighted_bits_per_spike"].to_numpy(dtype=float),
                color="#168a96",
                alpha=0.24,
                linewidth=1.1,
            )
            ax.plot(
                x,
                unit_sub["orthogonal_weighted_bits_per_spike"].to_numpy(dtype=float),
                color="#c06b2d",
                alpha=0.24,
                linewidth=1.1,
            )
        aligned_mean = sub.groupby("x_scale", sort=True)["aligned_weighted_bits_per_spike"].median()
        orth_mean = sub.groupby("x_scale", sort=True)["orthogonal_weighted_bits_per_spike"].median()
        ax.plot(
            aligned_mean.index.to_numpy(dtype=float),
            aligned_mean.to_numpy(dtype=float),
            color="#168a96",
            marker="o",
            linewidth=2.4,
            markersize=4.0,
            label="aligned median",
        )
        ax.plot(
            orth_mean.index.to_numpy(dtype=float),
            orth_mean.to_numpy(dtype=float),
            color="#c06b2d",
            marker="o",
            linewidth=2.4,
            markersize=4.0,
            label="orthogonal median",
        )
        ax.axvline(1.0, color="0.65", linestyle=":", linewidth=0.9)
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(sweep_titles[sweep_view], fontsize=10)
        ax.set_xlabel("scale")
        ax.set_ylabel("bits/spike")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "High-SF unit spaghetti: aligned vs orthogonal weighted absolute SSI\n"
        f"{metric_name}; {len(units)}/{len(units_total)} high-SF units have finite alignment-weighted curves",
        fontsize=12,
    )
    raw_png = out_dir / "backimage_rr100_high_sf_unit_spaghetti_weighted_alignment_curves.png"
    raw_pdf = out_dir / "backimage_rr100_high_sf_unit_spaghetti_weighted_alignment_curves.pdf"
    fig.savefig(raw_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(raw_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True)
    for ax, sweep_view in zip(axes.ravel(), sweep_order, strict=True):
        sub = df[df["sweep_view"].astype(str) == sweep_view].copy()
        for unit_idx in units:
            unit_sub = sub[sub["unit_index"].astype(int) == int(unit_idx)].sort_values("x_scale")
            if unit_sub.empty:
                continue
            ax.plot(
                unit_sub["x_scale"].to_numpy(dtype=float),
                unit_sub["aligned_minus_orthogonal_bits_per_spike"].to_numpy(dtype=float),
                color="0.25",
                alpha=0.24,
                linewidth=1.1,
            )
        median_delta = sub.groupby("x_scale", sort=True)["aligned_minus_orthogonal_bits_per_spike"].median()
        q25 = sub.groupby("x_scale", sort=True)["aligned_minus_orthogonal_bits_per_spike"].quantile(0.25)
        q75 = sub.groupby("x_scale", sort=True)["aligned_minus_orthogonal_bits_per_spike"].quantile(0.75)
        x = median_delta.index.to_numpy(dtype=float)
        ax.fill_between(x, q25.to_numpy(dtype=float), q75.to_numpy(dtype=float), color="0.25", alpha=0.15, linewidth=0.0)
        ax.plot(x, median_delta.to_numpy(dtype=float), color="0.1", marker="o", linewidth=2.3, markersize=4.0)
        frac_positive = sub.groupby("x_scale", sort=True)["aligned_minus_orthogonal_bits_per_spike"].apply(
            lambda values: float(np.nanmean(np.asarray(values, dtype=float) > 0.0))
        )
        for xpos, frac in frac_positive.items():
            ax.text(float(xpos), ax.get_ylim()[1], f"{frac:.2f}", ha="center", va="top", fontsize=6.8, color="0.25")
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        ax.axvline(1.0, color="0.65", linestyle=":", linewidth=0.9)
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(sweep_titles[sweep_view], fontsize=10)
        ax.set_xlabel("scale")
        ax.set_ylabel("aligned - orthogonal\nbits/spike")
    fig.suptitle(
        "High-SF unit spaghetti: aligned-minus-orthogonal per unit\n"
        "numbers at top are fraction of high-SF units with aligned > orthogonal",
        fontsize=12,
    )
    delta_png = out_dir / "backimage_rr100_high_sf_unit_spaghetti_aligned_minus_orthogonal.png"
    delta_pdf = out_dir / "backimage_rr100_high_sf_unit_spaghetti_aligned_minus_orthogonal.pdf"
    fig.savefig(delta_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(delta_pdf, bbox_inches="tight")
    plt.close(fig)
    return raw_png, raw_pdf, delta_png, delta_pdf


def key_condition_ids(conditions: pd.DataFrame) -> list[tuple[str, str]]:
    static_condition_id = static_baseline_condition_id(conditions)
    candidates = [
        ("static", static_condition_id),
        ("across end\nalong=0", "along0_across2"),
        ("across end\nalong=1", "along1_across3"),
        ("along end\nacross=0", "along2_across0"),
        ("along end\nacross=1", "along2_across1"),
    ]
    available = set(conditions["condition_id"].astype(str))
    return [(label, condition_id) for label, condition_id in candidates if condition_id in available]


def high_sf_contribution_rows(
    bits: np.ndarray,
    spikes: np.ndarray,
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    weights: dict[str, np.ndarray],
    units: pd.DataFrame,
    condition_labels: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    condition_by_id = {
        str(row.condition_id): int(row.condition_index)
        for row in conditions.itertuples(index=False)
    }
    movie_axes = movies.set_index("movie_index")["contour_axis_image_deg"].to_dict()
    high_units = units[units["sf_group"].astype(str) == "high_sf"].copy()
    rows: list[dict[str, Any]] = []
    for condition_label, condition_id in condition_labels:
        cidx = int(condition_by_id[str(condition_id)])
        for unit in high_units.itertuples(index=False):
            unit_idx = int(unit.unit_index)
            preferred = float(unit.preferred_orientation_image_deg)
            for movie_idx in range(bits.shape[1]):
                contour = float(movie_axes[int(movie_idx)])
                unit_bits = float(bits[cidx, movie_idx, unit_idx])
                unit_spikes = float(spikes[cidx, movie_idx, unit_idx])
                for alignment_group in ALIGNMENT_GROUPS:
                    weight = float(weights[alignment_group][movie_idx, unit_idx])
                    if not np.isfinite(weight) or weight <= EPS:
                        continue
                    weighted_spikes = unit_spikes * weight
                    information = unit_bits * weighted_spikes
                    rows.append(
                        {
                            "condition_label_short": condition_label,
                            "condition_id": condition_id,
                            "condition_index": cidx,
                            "movie_index": int(movie_idx),
                            "contour_axis_image_deg": contour,
                            "unit_index": unit_idx,
                            "unit_label": str(unit.unit_label),
                            "preferred_orientation_image_deg": preferred,
                            "prior_preferred_orientation_deg": float(unit.prior_preferred_orientation_deg),
                            "prior_orientation_selectivity_index": float(unit.prior_orientation_selectivity_index),
                            "sf_split_metric": float(unit.sf_split_metric),
                            "alignment_group": alignment_group,
                            "alignment_weight": weight,
                            "unit_bits_per_spike": unit_bits,
                            "unit_expected_spikes": unit_spikes,
                            "weighted_expected_spikes": weighted_spikes,
                            "weighted_information": information,
                        }
                    )
    return rows


def axial_hist(values: np.ndarray, weights: np.ndarray | None, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(np.asarray(values, dtype=float) % 180.0, bins=bins, weights=weights)
    total = float(np.nansum(hist))
    return hist / total if total > EPS else hist.astype(float)


def plot_high_sf_axis_mass(
    out_dir: Path,
    contribution_rows: list[dict[str, Any]],
    movies: pd.DataFrame,
    condition_labels: list[tuple[str, str]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    df = pd.DataFrame(contribution_rows)
    bins = np.linspace(0.0, 180.0, 13)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(
        3,
        len(condition_labels),
        figsize=(3.65 * len(condition_labels), 7.8),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    if len(condition_labels) == 1:
        axes = axes[:, None]
    unweighted = axial_hist(movies["contour_axis_image_deg"].to_numpy(dtype=float), None, bins)
    for col_idx, (condition_label, condition_id) in enumerate(condition_labels):
        axes[0, col_idx].bar(centers, unweighted, width=np.diff(bins) * 0.86, color="0.55", alpha=0.62)
        axes[0, col_idx].set_title(condition_label, fontsize=9)
        axes[0, col_idx].set_ylabel("fixation\nfraction" if col_idx == 0 else "")
        condition = df[df["condition_id"].astype(str) == str(condition_id)]
        for row_idx, weight_col in enumerate(["weighted_expected_spikes", "weighted_information"], start=1):
            ax = axes[row_idx, col_idx]
            for alignment_group, color, label in [
                ("contour_aligned", "#168a96", "aligned"),
                ("contour_orthogonal", "#c06b2d", "orthogonal"),
            ]:
                sub = condition[condition["alignment_group"].astype(str) == alignment_group]
                hist = axial_hist(
                    sub["contour_axis_image_deg"].to_numpy(dtype=float),
                    sub[weight_col].to_numpy(dtype=float),
                    bins,
                )
                ax.plot(centers, hist, marker="o", linewidth=1.8, markersize=3.5, color=color, label=label)
                total = float(np.nansum(sub[weight_col].to_numpy(dtype=float)))
                ax.text(
                    0.02,
                    0.93 - 0.12 * (alignment_group == "contour_orthogonal"),
                    f"{label} total={total:.2f}",
                    transform=ax.transAxes,
                    fontsize=6.6,
                    color=color,
                    ha="left",
                    va="top",
                )
            ax.set_ylabel(
                ("expected-spike\nmass fraction" if weight_col == "weighted_expected_spikes" else "information\nmass fraction")
                if col_idx == 0
                else ""
            )
            if row_idx == 1 and col_idx == len(condition_labels) - 1:
                ax.legend(frameon=False, fontsize=7)
        for ax in axes[:, col_idx]:
            ax.grid(True, color="0.9", linewidth=0.7)
            ax.set_xlim(0.0, 180.0)
            ax.set_xticks([0, 45, 90, 135, 180])
    for ax in axes[-1]:
        ax.set_xlabel("contour axis in image frame (deg)")
    fig.suptitle(
        "High-SF contour-axis anisotropy: where denominator and information mass come from\n"
        "histograms are normalized within each condition/alignment pool",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_high_sf_contour_axis_weighted_mass.png"
    pdf = out_dir / "backimage_rr100_high_sf_contour_axis_weighted_mass.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_high_sf_unit_leverage(
    out_dir: Path,
    contribution_rows: list[dict[str, Any]],
    condition_labels: list[tuple[str, str]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    df = pd.DataFrame(contribution_rows)
    fig, axes = plt.subplots(
        len(condition_labels),
        2,
        figsize=(12.8, 2.45 * len(condition_labels)),
        constrained_layout=True,
    )
    if len(condition_labels) == 1:
        axes = np.asarray([axes])
    colors = {"contour_aligned": "#168a96", "contour_orthogonal": "#c06b2d"}
    for row_idx, (condition_label, condition_id) in enumerate(condition_labels):
        condition = df[df["condition_id"].astype(str) == str(condition_id)].copy()
        unit_total = (
            condition.groupby(["unit_index", "unit_label"], sort=True)["weighted_information"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
        )
        unit_order = [idx[0] for idx in unit_total.index]
        labels = [idx[1] for idx in unit_total.index]
        for col_idx, value_col in enumerate(["weighted_expected_spikes", "weighted_information"]):
            ax = axes[row_idx, col_idx]
            agg = (
                condition[condition["unit_index"].astype(int).isin(unit_order)]
                .groupby(["unit_index", "alignment_group"], sort=True)[value_col]
                .sum()
                .reset_index()
            )
            x = np.arange(len(unit_order), dtype=float)
            width = 0.38
            for offset, alignment_group in [(-width / 2, "contour_aligned"), (width / 2, "contour_orthogonal")]:
                vals = []
                for unit_idx in unit_order:
                    sub = agg[(agg["unit_index"].astype(int) == int(unit_idx)) & (agg["alignment_group"].astype(str) == alignment_group)]
                    vals.append(float(sub[value_col].iloc[0]) if not sub.empty else 0.0)
                ax.bar(x + offset, vals, width=width, color=colors[alignment_group], alpha=0.78, label=alignment_group.replace("contour_", ""))
            ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
            ax.grid(True, axis="y", color="0.9", linewidth=0.7)
            if row_idx == 0:
                ax.set_title("weighted expected spikes" if value_col == "weighted_expected_spikes" else "weighted information")
            if col_idx == 0:
                ax.set_ylabel(condition_label, rotation=0, ha="right", va="center", labelpad=52, fontsize=8)
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "High-SF unit leverage by condition: top units by total information in each panel\n"
        "orthogonal can sit higher when high-information units receive more orthogonal denominator mass",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_high_sf_unit_leverage_by_condition.png"
    pdf = out_dir / "backimage_rr100_high_sf_unit_leverage_by_condition.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_high_sf_axis_pref_heatmaps(
    out_dir: Path,
    contribution_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    df = pd.DataFrame(contribution_rows)
    available_condition_ids = set(df["condition_id"].astype(str))
    static_ids = df.loc[
        df["condition_label_short"].astype(str) == "static",
        "condition_id",
    ].astype(str).unique()
    condition_ids = list(static_ids[:1])
    if "along0_across2" in available_condition_ids:
        condition_ids.append("along0_across2")
    condition_titles = {condition_id: "static" for condition_id in condition_ids[:1]}
    condition_titles["along0_across2"] = "across end, along=0"
    available = [condition_id for condition_id in condition_ids if condition_id in available_condition_ids]
    if not available:
        raise ValueError("No static or across-end conditions available for heatmaps.")
    bins = np.linspace(0.0, 180.0, 13)
    fig, axes = plt.subplots(
        len(available),
        2,
        figsize=(9.8, 4.15 * len(available)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(available) == 1:
        axes = np.asarray([axes])
    for row_idx, condition_id in enumerate(available):
        for col_idx, alignment_group in enumerate(ALIGNMENT_GROUPS):
            ax = axes[row_idx, col_idx]
            sub = df[(df["condition_id"].astype(str) == condition_id) & (df["alignment_group"].astype(str) == alignment_group)]
            hist, _, _ = np.histogram2d(
                sub["contour_axis_image_deg"].to_numpy(dtype=float) % 180.0,
                sub["preferred_orientation_image_deg"].to_numpy(dtype=float) % 180.0,
                bins=(bins, bins),
                weights=sub["weighted_information"].to_numpy(dtype=float),
            )
            total = float(np.nansum(hist))
            hist = hist / total if total > EPS else hist
            im = ax.imshow(
                hist.T,
                origin="lower",
                extent=(0, 180, 0, 180),
                aspect="auto",
                cmap="magma",
                vmin=0.0,
            )
            ax.set_title(f"{condition_titles.get(condition_id, condition_id)}\n{alignment_group.replace('contour_', '')}", fontsize=9)
            ax.set_xlabel("contour axis (deg)")
            if col_idx == 0:
                ax.set_ylabel("unit preferred orientation (deg)")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_yticks([0, 45, 90, 135, 180])
            fig.colorbar(im, ax=ax, shrink=0.78, label="fraction of information mass")
    fig.suptitle("High-SF information mass over contour axis x unit preferred orientation", fontsize=12)
    png = out_dir / "backimage_rr100_high_sf_axis_by_preferred_orientation_info_heatmaps.png"
    pdf = out_dir / "backimage_rr100_high_sf_axis_by_preferred_orientation_info_heatmaps.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_npz_without_identity(cache_path(Path(args.contour_run_dir)))
    bits, metric_key = metric_array(stats, str(args.ssi_metric))
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    conditions = condition_frame(stats)
    movies = load_movie_frame(Path(args.contour_run_dir), stats, "gaze")
    sf_groups = parse_csv_list(str(args.sf_groups))
    units = load_unit_table(Path(args.sf_groups_csv), sf_groups, int(bits.shape[2]))
    weights, signed_alignment = unit_weight_tables(
        units,
        movies,
        min_orientation_selectivity=float(args.min_orientation_selectivity),
    )
    candidates = role_candidates(bits, spikes, units, conditions, weights)
    examples = choose_examples(candidates, n_examples=int(args.n_examples))
    if not examples:
        raise ValueError("No example units could be selected.")

    candidate_csv = out_dir / "example_unit_candidate_scores.csv"
    selected_csv = out_dir / "selected_example_units.csv"
    high_sf_spaghetti_csv = out_dir / "high_sf_unit_spaghetti_weighted_alignment_curves.csv"
    high_sf_contribution_csv = out_dir / "high_sf_axis_unit_contribution_rows.csv"
    write_csv(candidate_csv, candidates.to_dict(orient="records"))
    write_csv(selected_csv, examples)
    high_sf_rows = high_sf_spaghetti_rows(bits, spikes, conditions, weights, units)
    write_csv(high_sf_spaghetti_csv, high_sf_rows)
    condition_labels = key_condition_ids(conditions)
    high_sf_contribution = high_sf_contribution_rows(
        bits,
        spikes,
        conditions,
        movies,
        weights,
        units,
        condition_labels,
    )
    write_csv(high_sf_contribution_csv, high_sf_contribution)

    metric_name = metric_label(str(args.ssi_metric))
    curve_png, curve_pdf = plot_example_curves(
        out_dir,
        bits,
        spikes,
        conditions,
        weights,
        examples,
        metric_name=metric_name,
        dpi=int(args.dpi),
    )
    weight_png, weight_pdf = plot_alignment_weight_distributions(
        out_dir,
        signed_alignment,
        examples,
        dpi=int(args.dpi),
    )
    spaghetti_png, spaghetti_pdf, spaghetti_delta_png, spaghetti_delta_pdf = plot_high_sf_spaghetti(
        out_dir,
        high_sf_rows,
        metric_name=metric_name,
        dpi=int(args.dpi),
    )
    axis_mass_png, axis_mass_pdf = plot_high_sf_axis_mass(
        out_dir,
        high_sf_contribution,
        movies,
        condition_labels,
        dpi=int(args.dpi),
    )
    unit_leverage_png, unit_leverage_pdf = plot_high_sf_unit_leverage(
        out_dir,
        high_sf_contribution,
        condition_labels,
        dpi=int(args.dpi),
    )
    heatmap_png, heatmap_pdf = plot_high_sf_axis_pref_heatmaps(
        out_dir,
        high_sf_contribution,
        dpi=int(args.dpi),
    )
    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_sf_contour_alignment_example_units",
            "contour_run_dir": Path(args.contour_run_dir),
            "sf_groups_csv": Path(args.sf_groups_csv),
            "out_dir": out_dir,
            "ssi_metric": str(args.ssi_metric),
            "ssi_metric_cache_key": metric_key,
            "sf_groups": sf_groups,
            "min_orientation_selectivity": float(args.min_orientation_selectivity),
            "selection_contract": (
                "Examples are selected from weighted static leverage and static-to-2x modulation for "
                "the pure across-contour sweep, along=0. Curves then show the same selected units across "
                "all four sweep views using all-fixation, contour-aligned weighted, and orthogonal weighted summaries."
            ),
            "selected_examples": examples,
            "outputs": {
                "candidate_scores_csv": candidate_csv,
                "selected_examples_csv": selected_csv,
                "example_curves_png": curve_png,
                "example_curves_pdf": curve_pdf,
                "alignment_weight_hist_png": weight_png,
                "alignment_weight_hist_pdf": weight_pdf,
                "high_sf_spaghetti_csv": high_sf_spaghetti_csv,
                "high_sf_spaghetti_png": spaghetti_png,
                "high_sf_spaghetti_pdf": spaghetti_pdf,
                "high_sf_spaghetti_delta_png": spaghetti_delta_png,
                "high_sf_spaghetti_delta_pdf": spaghetti_delta_pdf,
                "high_sf_contribution_rows_csv": high_sf_contribution_csv,
                "high_sf_axis_mass_png": axis_mass_png,
                "high_sf_axis_mass_pdf": axis_mass_pdf,
                "high_sf_unit_leverage_png": unit_leverage_png,
                "high_sf_unit_leverage_pdf": unit_leverage_pdf,
                "high_sf_axis_pref_heatmap_png": heatmap_png,
                "high_sf_axis_pref_heatmap_pdf": heatmap_pdf,
            },
        },
    )
    print(f"Wrote {selected_csv}")
    print(f"Wrote {curve_png}")
    print(f"Wrote {weight_png}")
    print(f"Wrote {spaghetti_png}")
    print(f"Wrote {spaghetti_delta_png}")
    print(f"Wrote {axis_mass_png}")
    print(f"Wrote {unit_leverage_png}")
    print(f"Wrote {heatmap_png}")


if __name__ == "__main__":
    main()

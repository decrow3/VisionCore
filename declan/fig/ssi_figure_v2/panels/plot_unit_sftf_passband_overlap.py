#!/usr/bin/env python3
"""Unit SF/TF passband overlap with full-trace retinal power spectra."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.interpolate import RegularGridInterpolator

from plot_eye_movement_joint_sftf_power import db, spatial_edges_from_centers, temporal_edges
from plot_eye_movement_power_spectrum_shift import ROOT


PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DENSE_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
SSI_MATRIX_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged"
)
SPECTRUM_DIR = PANEL_DIR / "high_coherence_full_trace_sftf_components"

DEFAULT_FITS_CSV = DENSE_DIR / "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
DEFAULT_POINTS_CSV = DENSE_DIR / "cycle_valid_dense_sf_tf_points.csv"
DEFAULT_UNIT_TABLE_CSV = SSI_MATRIX_DIR / "unit_feature_table.csv"
DEFAULT_POWER_DETAIL_CSV = SPECTRUM_DIR / "high_coherence_full_trace_sftf_detail.csv"
DEFAULT_IMAGE_SAMPLE_CSV = SPECTRUM_DIR / "high_coherence_full_trace_image_sample.csv"
DEFAULT_OUT_DIR = PANEL_DIR / "unit_sftf_passband_overlap"

MODE_ORDER = ("full_2d", "along_only", "across_only")
MODE_LABELS = {
    "full_2d": "full motion",
    "along_only": "contour-parallel",
    "across_only": "contour-normal",
}
MODE_COLORS = {
    "full_2d": "#222222",
    "along_only": "#2B6CB0",
    "across_only": "#B24A3B",
}
FINAL_GROUP_ORDER = ("low_mid_sf", "high_sf")
FINAL_GROUP_LABELS = {
    "low_mid_sf": "low+middle SF",
    "high_sf": "high SF",
}
FINAL_GROUP_COLORS = {
    "low_mid_sf": "#0072B2",
    "high_sf": "#D55E00",
}
RELATION_ORDER = ("aligned", "oblique", "orthogonal")
RELATION_LABELS = {
    "aligned": "aligned",
    "oblique": "oblique",
    "orthogonal": "orthogonal",
}
RELATION_COLORS = {
    "aligned": "#009E73",
    "oblique": "#CC79A7",
    "orthogonal": "#D55E00",
}
SF_BANDS = (
    ("retinal_low_sf", "0.25-2 cpd", 0.25, 2.0),
    ("retinal_mid_sf", "2-8 cpd", 2.0, 8.0),
    ("retinal_high_sf", "8+ cpd", 8.0, math.inf),
)
TF_BANDS = (
    ("slow_tf", "1-8 Hz", 0.0, 8.0),
    ("mid_tf", "8-30 Hz", 8.0, 30.0),
    ("fast_tf", "30+ Hz", 30.0, math.inf),
)
EPS = 1e-300
GAUSSIAN_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
SF_TICKS = np.asarray([0.25, 0.5, 1, 2, 4, 8, 16], dtype=float)
TF_TICKS = np.asarray([1, 3, 10, 30, 60], dtype=float)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def axis_delta_deg(axis_a: np.ndarray | pd.Series | float, axis_b: np.ndarray | pd.Series | float) -> np.ndarray:
    a = np.asarray(axis_a, dtype=np.float64)
    b = np.asarray(axis_b, dtype=np.float64)
    return np.abs((a - b + 90.0) % 180.0 - 90.0)


def setup_log_axes(ax: plt.Axes, *, x_log2: bool = False, y_log2: bool = False) -> None:
    if x_log2:
        ax.set_xlim(float(np.log2(0.23)), float(np.log2(17.5)))
        ax.set_xticks(np.log2(SF_TICKS), [f"{v:g}" for v in SF_TICKS])
    else:
        ax.set_xscale("log")
        ax.set_xlim(0.23, 17.5)
        ax.set_xticks(SF_TICKS, [f"{v:g}" for v in SF_TICKS])
    if y_log2:
        ax.set_ylim(float(np.log2(0.8)), float(np.log2(64.0)))
        ax.set_yticks(np.log2(TF_TICKS), [f"{v:g}" for v in TF_TICKS])
    else:
        ax.set_yscale("log")
        ax.set_ylim(0.8, 64.0)
        ax.set_yticks(TF_TICKS, [f"{v:g}" for v in TF_TICKS])
    ax.grid(True, color="0.90", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")


def final_sf_group(sf_group: pd.Series) -> pd.Series:
    values = sf_group.astype(str)
    return pd.Series(
        np.where(values.isin(["low_sf", "middle_sf"]), "low_mid_sf", np.where(values.eq("high_sf"), "high_sf", "unknown")),
        index=sf_group.index,
    )


def _finite_fit_mask(frame: pd.DataFrame) -> pd.Series:
    cols = ["fit_pref_log2_sf", "fit_pref_log2_tf", "fit_sigma_sf_octaves", "fit_sigma_tf_octaves"]
    mask = frame["fit_ok"].astype(bool)
    for col in cols:
        mask &= np.isfinite(pd.to_numeric(frame[col], errors="coerce"))
    mask &= pd.to_numeric(frame["fit_sigma_sf_octaves"], errors="coerce").gt(0)
    mask &= pd.to_numeric(frame["fit_sigma_tf_octaves"], errors="coerce").gt(0)
    return mask


def load_atlas(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = pd.read_csv(args.fits_csv)
    unit_table = pd.read_csv(args.unit_table_csv)
    fit_rename = {
        "sf_group": "dense_sf_group",
        "sf_group_label": "dense_sf_group_label",
    }
    unit_rename = {
        "sf_group": "marginal_sf_group",
        "sf_group_label": "marginal_sf_group_label",
    }
    fits = fits.rename(columns=fit_rename)
    unit_cols = [
        "unit_index",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "sf_split_metric",
        "sf_rank_low_to_high",
        "sf_group",
        "sf_group_label",
        "sf_group_definition",
        "sf_split_metric_name",
        "sf_split_metric_column",
        "dynamic_log_gaussian_marginal_sf_cpd",
        "dynamic_log_gaussian_marginal_fwhm_octaves",
        "dynamic_log_gaussian_marginal_r2",
        "dynamic_log_gaussian_marginal_fit_ok",
    ]
    unit_cols = [col for col in unit_cols if col in unit_table.columns]
    unit_table = unit_table[unit_cols].rename(columns=unit_rename)
    atlas = fits.merge(unit_table, on="unit_index", how="left", validate="one_to_one")
    atlas["sf_group"] = atlas["marginal_sf_group"].fillna(atlas["dense_sf_group"]).astype(str)
    atlas["sf_group_label"] = atlas["marginal_sf_group_label"].fillna(atlas["dense_sf_group_label"]).astype(str)
    atlas["final_sf_group"] = final_sf_group(atlas["sf_group"])
    atlas["final_sf_group_label"] = atlas["final_sf_group"].map(FINAL_GROUP_LABELS).fillna(atlas["final_sf_group"])
    atlas["fit_is_edge"] = atlas["fit_edge_sf"].astype(bool) | atlas["fit_edge_tf"].astype(bool)
    atlas["fit_valid_for_overlap"] = _finite_fit_mask(atlas)
    atlas["orientation_tuned"] = (
        np.isfinite(pd.to_numeric(atlas["prior_preferred_orientation_deg"], errors="coerce"))
        & pd.to_numeric(atlas["prior_orientation_selectivity_index"], errors="coerce").ge(float(args.min_osi))
    )
    atlas["fit_pref_sf_center_band"] = pd.cut(
        pd.to_numeric(atlas["fit_pref_sf_cpd"], errors="coerce"),
        bins=[0.0, 0.75, 2.0, 8.0, math.inf],
        labels=["sub-0.75 cpd", "0.75-2 cpd", "2-8 cpd", "8+ cpd"],
        include_lowest=True,
    ).astype(str)
    atlas["fit_pref_tf_center_band"] = pd.cut(
        pd.to_numeric(atlas["fit_pref_tf_hz"], errors="coerce"),
        bins=[0.0, 4.0, 12.0, 30.0, math.inf],
        labels=["sub-4 Hz", "4-12 Hz", "12-30 Hz", "30+ Hz"],
        include_lowest=True,
    ).astype(str)
    points = pd.read_csv(args.points_csv)
    return atlas, points


def load_power_grid(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detail = pd.read_csv(args.power_detail_csv)
    image_sample = pd.read_csv(args.image_sample_csv)
    if int(args.max_windows) > 0:
        keep_images = sorted(detail["image_pos"].drop_duplicates().astype(int).to_list())[: int(args.max_windows)]
        detail = detail[detail["image_pos"].isin(keep_images)].copy()
        image_sample = image_sample[image_sample["image_pos"].isin(keep_images)].copy()
    if detail.empty:
        raise RuntimeError("No retinal power rows after filtering.")
    sf = np.asarray(sorted(detail["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(detail["temporal_frequency_hz"].unique()), dtype=np.float64)
    columns = pd.MultiIndex.from_product([tf, sf], names=["temporal_frequency_hz", "spatial_frequency_cpd"])
    meta_cols = [
        "image_pos",
        "image_index",
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "motion_mode",
        "motion_mode_label",
    ]
    pivot = detail.pivot_table(
        index=meta_cols,
        columns=["temporal_frequency_hz", "spatial_frequency_cpd"],
        values="modulation_power_mean",
        aggfunc="mean",
    ).reindex(columns=columns)
    power_meta = pivot.index.to_frame(index=False)
    image_extra_cols = [
        "image_pos",
        "image_index",
        "trace_duration_s",
        "rms_full_arcmin",
        "rms_along_arcmin",
        "rms_across_arcmin",
        "path_full_arcmin",
        "path_along_arcmin",
        "path_across_arcmin",
    ]
    image_extra_cols = [col for col in image_extra_cols if col in image_sample.columns]
    if {"image_pos", "image_index"}.issubset(set(image_extra_cols)):
        image_extra = image_sample[image_extra_cols].drop_duplicates(["image_pos", "image_index"])
        power_meta = power_meta.merge(image_extra, on=["image_pos", "image_index"], how="left", validate="many_to_one")
    power_values = np.nan_to_num(pivot.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    tf_grid, sf_grid = np.meshgrid(tf, sf, indexing="ij")
    flat_tf = tf_grid.ravel()
    flat_sf = sf_grid.ravel()
    return detail, power_meta, power_values, sf, tf, flat_sf, flat_tf


def fit_weights(atlas: pd.DataFrame, flat_sf: np.ndarray, flat_tf: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(atlas), flat_sf.size), dtype=np.float64)
    log_sf = np.log2(np.asarray(flat_sf, dtype=np.float64))
    log_tf = np.log2(np.asarray(flat_tf, dtype=np.float64))
    for row_i, row in atlas.reset_index(drop=True).iterrows():
        if not bool(row["fit_valid_for_overlap"]):
            continue
        sx = float(row["fit_sigma_sf_octaves"])
        sy = float(row["fit_sigma_tf_octaves"])
        if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
            fwhm_x = float(row.get("fit_fwhm_sf_octaves", np.nan))
            fwhm_y = float(row.get("fit_fwhm_tf_octaves", np.nan))
            sx = fwhm_x / GAUSSIAN_FWHM_FACTOR
            sy = fwhm_y / GAUSSIAN_FWHM_FACTOR
        if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
            continue
        dx = (log_sf - float(row["fit_pref_log2_sf"])) / max(sx, 1e-6)
        dy = (log_tf - float(row["fit_pref_log2_tf"])) / max(sy, 1e-6)
        weights[row_i, :] = np.exp(-0.5 * (dx * dx + dy * dy))
    return weights


def empirical_weights(points: pd.DataFrame, atlas: pd.DataFrame, flat_sf: np.ndarray, flat_tf: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(atlas), flat_sf.size), dtype=np.float64)
    query = np.column_stack([np.log2(flat_tf), np.log2(flat_sf)])
    for row_i, unit_index in enumerate(atlas["unit_index"].astype(int).to_numpy()):
        sub = points[points["unit_index"].astype(int).eq(int(unit_index))].copy()
        if sub.empty:
            continue
        source_sf = np.asarray(sorted(sub["spatial_cpd"].unique()), dtype=np.float64)
        source_tf = np.asarray(sorted(sub["temporal_hz"].unique()), dtype=np.float64)
        grid = (
            sub.pivot_table(
                index="temporal_hz",
                columns="spatial_cpd",
                values="response_amp_rms_mean",
                aggfunc="mean",
            )
            .reindex(index=source_tf, columns=source_sf)
            .to_numpy(dtype=np.float64)
        )
        grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
        grid = np.clip(grid, 0.0, None)
        peak = float(np.nanmax(grid)) if grid.size else 0.0
        if not math.isfinite(peak) or peak <= 0:
            continue
        grid = grid / peak
        interp = RegularGridInterpolator(
            (np.log2(source_tf), np.log2(source_sf)),
            grid,
            bounds_error=False,
            fill_value=0.0,
        )
        weights[row_i, :] = np.clip(interp(query), 0.0, None)
    return weights


def _weighted_log_centroid(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    if not np.any(good):
        return float("nan")
    return float(2.0 ** np.average(np.log2(values[good]), weights=weights[good]))


def _band_fractions(
    weights: np.ndarray,
    flat_sf: np.ndarray,
    flat_tf: np.ndarray,
    *,
    prefix: str,
) -> dict[str, Any]:
    total = float(np.nansum(weights))
    out: dict[str, Any] = {f"{prefix}_weight_sum": total}
    if not math.isfinite(total) or total <= 0:
        for band_id, _label, _lo, _hi in SF_BANDS:
            out[f"{prefix}_frac_{band_id}"] = float("nan")
        for band_id, _label, _lo, _hi in TF_BANDS:
            out[f"{prefix}_frac_{band_id}"] = float("nan")
        out[f"{prefix}_dominant_sf_band"] = "missing"
        out[f"{prefix}_dominant_tf_band"] = "missing"
        out[f"{prefix}_sf_centroid_cpd"] = float("nan")
        out[f"{prefix}_tf_centroid_hz"] = float("nan")
        out[f"{prefix}_fwhm_grid_fraction"] = float("nan")
        return out
    sf_fracs: dict[str, float] = {}
    for band_id, _label, lo, hi in SF_BANDS:
        mask = (flat_sf >= lo) & (flat_sf < hi)
        frac = float(np.nansum(weights[mask]) / total)
        out[f"{prefix}_frac_{band_id}"] = frac
        sf_fracs[band_id] = frac
    tf_fracs: dict[str, float] = {}
    for band_id, _label, lo, hi in TF_BANDS:
        mask = (flat_tf >= lo) & (flat_tf < hi)
        frac = float(np.nansum(weights[mask]) / total)
        out[f"{prefix}_frac_{band_id}"] = frac
        tf_fracs[band_id] = frac
    out[f"{prefix}_dominant_sf_band"] = max(sf_fracs, key=sf_fracs.get)
    out[f"{prefix}_dominant_tf_band"] = max(tf_fracs, key=tf_fracs.get)
    out[f"{prefix}_sf_centroid_cpd"] = _weighted_log_centroid(flat_sf, weights)
    out[f"{prefix}_tf_centroid_hz"] = _weighted_log_centroid(flat_tf, weights)
    out[f"{prefix}_fwhm_grid_fraction"] = float(np.mean(weights >= 0.5))
    return out


def add_passband_metrics(
    atlas: pd.DataFrame,
    fit_w: np.ndarray,
    empirical_w: np.ndarray,
    flat_sf: np.ndarray,
    flat_tf: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in atlas.reset_index(drop=True).iterrows():
        metrics = row.to_dict()
        metrics.update(_band_fractions(fit_w[idx], flat_sf, flat_tf, prefix="fit_soft"))
        metrics.update(_band_fractions(empirical_w[idx], flat_sf, flat_tf, prefix="empirical_soft"))
        metrics["fit_passband_regime"] = (
            f"{metrics['fit_soft_dominant_sf_band']} / {metrics['fit_soft_dominant_tf_band']}"
        )
        metrics["empirical_passband_regime"] = (
            f"{metrics['empirical_soft_dominant_sf_band']} / {metrics['empirical_soft_dominant_tf_band']}"
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _overlap_matrix(power_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = np.nansum(weights, axis=1)
    valid = np.isfinite(denom) & (denom > 0)
    out = np.full((power_values.shape[0], weights.shape[0]), np.nan, dtype=np.float64)
    if np.any(valid):
        out[:, valid] = power_values @ weights[valid].T / denom[valid][None, :]
    return out


def _long_overlap(
    values: np.ndarray,
    power_meta: pd.DataFrame,
    atlas: pd.DataFrame,
    *,
    weighting: str,
) -> pd.DataFrame:
    n_power, n_units = values.shape
    meta = power_meta.iloc[np.repeat(np.arange(n_power), n_units)].reset_index(drop=True)
    unit_cols = [
        "unit_index",
        "unit_label",
        "sf_group",
        "sf_group_label",
        "final_sf_group",
        "final_sf_group_label",
        "dense_sf_group",
        "fit_valid_for_overlap",
        "fit_is_edge",
        "fit_r2",
        "fit_pref_sf_cpd",
        "fit_pref_tf_hz",
        "fit_pref_speed_dps",
        "fit_fwhm_sf_octaves",
        "fit_fwhm_tf_octaves",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "orientation_tuned",
        "sf_split_metric",
        "fit_passband_regime",
        "empirical_passband_regime",
    ]
    unit_cols = [col for col in unit_cols if col in atlas.columns]
    units = atlas[unit_cols].iloc[np.tile(np.arange(n_units), n_power)].reset_index(drop=True)
    out = pd.concat([meta, units], axis=1)
    out["overlap_weighting"] = weighting
    out["overlap_power"] = values.ravel()
    out["overlap_power_db"] = db(out["overlap_power"].to_numpy(dtype=np.float64))
    return out


def add_overlap_ratios(
    overlap: pd.DataFrame,
    *,
    match_max_deg: float,
    orthogonal_min_deg: float,
) -> pd.DataFrame:
    id_cols = ["overlap_weighting", "unit_index", "image_pos"]
    pivot = overlap.pivot_table(index=id_cols, columns="motion_mode", values="overlap_power", aggfunc="mean")
    pivot = pivot.reset_index()
    full = np.asarray(pivot.get("full_2d", np.nan), dtype=np.float64)
    along = np.asarray(pivot.get("along_only", np.nan), dtype=np.float64)
    across = np.asarray(pivot.get("across_only", np.nan), dtype=np.float64)
    pivot["along_over_full"] = along / np.maximum(full, EPS)
    pivot["across_over_full"] = across / np.maximum(full, EPS)
    pivot["across_over_along"] = across / np.maximum(along, EPS)
    pivot["along_minus_full_db"] = db(pivot["along_over_full"].to_numpy(dtype=np.float64))
    pivot["across_minus_full_db"] = db(pivot["across_over_full"].to_numpy(dtype=np.float64))
    pivot["across_minus_along_db"] = db(pivot["across_over_along"].to_numpy(dtype=np.float64))
    ratio_cols = id_cols + [
        "along_over_full",
        "across_over_full",
        "across_over_along",
        "along_minus_full_db",
        "across_minus_full_db",
        "across_minus_along_db",
    ]
    out = overlap.merge(pivot[ratio_cols], on=id_cols, how="left", validate="many_to_one")
    mode = out["motion_mode"].astype(str)
    out["component_over_full"] = np.select(
        [mode.eq("full_2d"), mode.eq("along_only"), mode.eq("across_only")],
        [1.0, out["along_over_full"], out["across_over_full"]],
        default=np.nan,
    )
    out["component_minus_full_db"] = np.select(
        [mode.eq("full_2d"), mode.eq("along_only"), mode.eq("across_only")],
        [0.0, out["along_minus_full_db"], out["across_minus_full_db"]],
        default=np.nan,
    )
    out["axis_delta_deg"] = axis_delta_deg(out["image_edge_axis_deg"], out["prior_preferred_orientation_deg"])
    tuned = out["orientation_tuned"].astype(bool) & np.isfinite(out["axis_delta_deg"].to_numpy(dtype=np.float64))
    relation = np.full(len(out), "untuned_or_missing", dtype=object)
    relation[tuned & out["axis_delta_deg"].le(float(match_max_deg)).to_numpy(dtype=bool)] = "aligned"
    relation[
        tuned
        & out["axis_delta_deg"].gt(float(match_max_deg)).to_numpy(dtype=bool)
        & out["axis_delta_deg"].lt(float(orthogonal_min_deg)).to_numpy(dtype=bool)
    ] = "oblique"
    relation[tuned & out["axis_delta_deg"].ge(float(orthogonal_min_deg)).to_numpy(dtype=bool)] = "orthogonal"
    out["contour_relation"] = relation
    return out


def compute_overlaps(
    atlas: pd.DataFrame,
    power_meta: pd.DataFrame,
    power_values: np.ndarray,
    fit_w: np.ndarray,
    empirical_w: np.ndarray,
    *,
    match_max_deg: float,
    orthogonal_min_deg: float,
) -> pd.DataFrame:
    methods = {
        "fit_soft": fit_w,
        "fit_fwhm": (fit_w >= 0.5).astype(np.float64),
        "empirical_soft": empirical_w,
        "empirical_tophalf": (empirical_w >= 0.5).astype(np.float64),
    }
    frames = []
    for weighting, weights in methods.items():
        frames.append(_long_overlap(_overlap_matrix(power_values, weights), power_meta, atlas, weighting=weighting))
    return add_overlap_ratios(
        pd.concat(frames, ignore_index=True, sort=False),
        match_max_deg=match_max_deg,
        orthogonal_min_deg=orthogonal_min_deg,
    )


POPULATIONS = (
    ("low_mid_sf_all", "low+middle SF", "final_sf_group == 'low_mid_sf'"),
    ("high_sf_all", "high SF", "final_sf_group == 'high_sf'"),
    ("high_sf_tuned_all", "high SF tuned", "final_sf_group == 'high_sf' and orientation_tuned"),
    ("high_sf_aligned", "high SF aligned", "final_sf_group == 'high_sf' and contour_relation == 'aligned'"),
    ("high_sf_oblique", "high SF oblique", "final_sf_group == 'high_sf' and contour_relation == 'oblique'"),
    ("high_sf_orthogonal", "high SF orthogonal", "final_sf_group == 'high_sf' and contour_relation == 'orthogonal'"),
)


def _population_mask(frame: pd.DataFrame, key: str) -> pd.Series:
    high = frame["final_sf_group"].astype(str).eq("high_sf")
    low_mid = frame["final_sf_group"].astype(str).eq("low_mid_sf")
    tuned = frame["orientation_tuned"].astype(bool)
    relation = frame["contour_relation"].astype(str)
    if key == "low_mid_sf_all":
        return low_mid
    if key == "high_sf_all":
        return high
    if key == "high_sf_tuned_all":
        return high & tuned
    if key == "high_sf_aligned":
        return high & relation.eq("aligned")
    if key == "high_sf_oblique":
        return high & relation.eq("oblique")
    if key == "high_sf_orthogonal":
        return high & relation.eq("orthogonal")
    raise KeyError(key)


def _finite_median(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def _finite_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else float("nan")


def summarize_overlap(overlap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    pair = overlap[overlap["motion_mode"].eq("full_2d")].copy()
    for key, label, _expr in POPULATIONS:
        pop_mask = _population_mask(overlap, key)
        pair_mask = _population_mask(pair, key)
        for weighting in sorted(overlap["overlap_weighting"].unique()):
            pair_sub = pair[pair_mask & pair["overlap_weighting"].eq(weighting)].copy()
            contrast_rows.append(
                {
                    "population_key": key,
                    "population_label": label,
                    "overlap_weighting": weighting,
                    "n_units": int(pair_sub["unit_index"].nunique()),
                    "n_windows": int(pair_sub["image_pos"].nunique()),
                    "n_unit_window_pairs": int(pair_sub[["unit_index", "image_pos"]].drop_duplicates().shape[0]),
                    "along_over_full_median": _finite_median(pair_sub["along_over_full"]),
                    "across_over_full_median": _finite_median(pair_sub["across_over_full"]),
                    "across_over_along_median": _finite_median(pair_sub["across_over_along"]),
                    "along_minus_full_db_median": _finite_median(pair_sub["along_minus_full_db"]),
                    "across_minus_full_db_median": _finite_median(pair_sub["across_minus_full_db"]),
                    "across_minus_along_db_median": _finite_median(pair_sub["across_minus_along_db"]),
                    "along_over_full_mean": _finite_mean(pair_sub["along_over_full"]),
                    "across_over_full_mean": _finite_mean(pair_sub["across_over_full"]),
                    "across_minus_along_db_mean": _finite_mean(pair_sub["across_minus_along_db"]),
                }
            )
            for mode in MODE_ORDER:
                sub = overlap[
                    pop_mask
                    & overlap["overlap_weighting"].eq(weighting)
                    & overlap["motion_mode"].astype(str).eq(mode)
                ].copy()
                rows.append(
                    {
                        "population_key": key,
                        "population_label": label,
                        "overlap_weighting": weighting,
                        "motion_mode": mode,
                        "motion_mode_label": MODE_LABELS[mode],
                        "n_units": int(sub["unit_index"].nunique()),
                        "n_windows": int(sub["image_pos"].nunique()),
                        "n_unit_window_pairs": int(sub[["unit_index", "image_pos"]].drop_duplicates().shape[0]),
                        "overlap_power_db_median": _finite_median(sub["overlap_power_db"]),
                        "overlap_power_db_mean": _finite_mean(sub["overlap_power_db"]),
                        "component_over_full_median": _finite_median(sub["component_over_full"]),
                        "component_over_full_mean": _finite_mean(sub["component_over_full"]),
                        "component_minus_full_db_median": _finite_median(sub["component_minus_full_db"]),
                        "component_minus_full_db_mean": _finite_mean(sub["component_minus_full_db"]),
                        "across_minus_along_db_median": _finite_median(sub["across_minus_along_db"]),
                        "axis_delta_deg_median": _finite_median(sub["axis_delta_deg"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows)


def summarize_passbands(atlas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in FINAL_GROUP_ORDER:
        sub = atlas[atlas["final_sf_group"].eq(group)].copy()
        for subset, use in (("all_fit", sub), ("interior_fit", sub[~sub["fit_is_edge"]])):
            row: dict[str, Any] = {
                "final_sf_group": group,
                "final_sf_group_label": FINAL_GROUP_LABELS[group],
                "subset": subset,
                "n_units": int(use["unit_index"].nunique()),
                "fit_edge_fraction": float(sub["fit_is_edge"].mean()) if not sub.empty else float("nan"),
                "fit_pref_sf_cpd_median": _finite_median(use["fit_pref_sf_cpd"]),
                "fit_pref_tf_hz_median": _finite_median(use["fit_pref_tf_hz"]),
                "fit_pref_speed_dps_median": _finite_median(use["fit_pref_speed_dps"]),
                "fit_fwhm_sf_octaves_median": _finite_median(use["fit_fwhm_sf_octaves"]),
                "fit_fwhm_tf_octaves_median": _finite_median(use["fit_fwhm_tf_octaves"]),
                "fit_soft_sf_centroid_cpd_median": _finite_median(use["fit_soft_sf_centroid_cpd"]),
                "fit_soft_tf_centroid_hz_median": _finite_median(use["fit_soft_tf_centroid_hz"]),
            }
            for band_id, _label, _lo, _hi in SF_BANDS:
                row[f"fit_soft_frac_{band_id}_mean"] = _finite_mean(use[f"fit_soft_frac_{band_id}"])
            for band_id, _label, _lo, _hi in TF_BANDS:
                row[f"fit_soft_frac_{band_id}_mean"] = _finite_mean(use[f"fit_soft_frac_{band_id}"])
            rows.append(row)
    return pd.DataFrame(rows)


DOSE_FAMILIES = (
    (
        "component_rms",
        "component RMS excursion",
        "arcmin",
        {"along_only": "rms_along_arcmin", "across_only": "rms_across_arcmin"},
    ),
    (
        "component_path",
        "component path length",
        "arcmin",
        {"along_only": "path_along_arcmin", "across_only": "path_across_arcmin"},
    ),
)


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
    if np.unique(edges).size < edges.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return np.asarray([], dtype=np.float64)
        edges = np.linspace(lo, hi, int(n_bins) + 1)
    span = max(float(edges[-1] - edges[0]), 1e-9)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.full(values.shape, -1, dtype=int)
    bins = np.searchsorted(edges, values, side="right") - 1
    bins[(values < edges[0]) | (values > edges[-1]) | ~np.isfinite(values)] = -1
    return np.clip(bins, -1, edges.size - 2).astype(int)


def summarize_motion_dose(overlap: pd.DataFrame, *, n_bins: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population_key, population_label, _expr in POPULATIONS:
        pop_mask = _population_mask(overlap, population_key)
        for weighting in sorted(overlap["overlap_weighting"].unique()):
            weight_mask = overlap["overlap_weighting"].eq(weighting)
            for family_key, family_label, family_unit, component_cols in DOSE_FAMILIES:
                pooled_values: list[np.ndarray] = []
                for mode, col in component_cols.items():
                    sub = overlap[pop_mask & weight_mask & overlap["motion_mode"].eq(mode)]
                    if col in sub.columns:
                        pooled_values.append(pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float))
                if not pooled_values:
                    continue
                edges = _quantile_edges(np.concatenate(pooled_values), n_bins)
                if edges.size < 2:
                    continue
                for mode, col in component_cols.items():
                    sub = overlap[pop_mask & weight_mask & overlap["motion_mode"].eq(mode)].copy()
                    if col not in sub.columns:
                        continue
                    values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                    bins = _assign_bins(values, edges)
                    for bin_index in range(edges.size - 1):
                        bin_sub = sub[bins == bin_index]
                        rows.append(
                            {
                                "population_key": population_key,
                                "population_label": population_label,
                                "overlap_weighting": weighting,
                                "dose_family": family_key,
                                "dose_family_label": family_label,
                                "dose_unit": family_unit,
                                "motion_mode": mode,
                                "motion_mode_label": MODE_LABELS[mode],
                                "dose_bin": int(bin_index),
                                "dose_bin_min": float(edges[bin_index]),
                                "dose_bin_max": float(edges[bin_index + 1]),
                                "dose_median": _finite_median(values[bins == bin_index]),
                                "n_units": int(bin_sub["unit_index"].nunique()),
                                "n_windows": int(bin_sub["image_pos"].nunique()),
                                "n_unit_window_pairs": int(
                                    bin_sub[["unit_index", "image_pos"]].drop_duplicates().shape[0]
                                ),
                                "component_over_full_median": _finite_median(bin_sub["component_over_full"]),
                                "component_minus_full_db_median": _finite_median(bin_sub["component_minus_full_db"]),
                                "overlap_power_db_median": _finite_median(bin_sub["overlap_power_db"]),
                            }
                        )
    return pd.DataFrame(rows)


def plot_fit_ellipses(ax: plt.Axes, atlas: pd.DataFrame) -> None:
    use = atlas[atlas["fit_valid_for_overlap"].astype(bool)].copy()
    for group in FINAL_GROUP_ORDER:
        sub = use[use["final_sf_group"].eq(group)].copy()
        color = FINAL_GROUP_COLORS[group]
        for row in sub.itertuples(index=False):
            ell = Ellipse(
                (float(row.fit_pref_log2_sf), float(row.fit_pref_log2_tf)),
                width=float(row.fit_fwhm_sf_octaves),
                height=float(row.fit_fwhm_tf_octaves),
                angle=0.0,
                fill=False,
                edgecolor=color,
                linewidth=0.75 if bool(row.fit_is_edge) else 0.95,
                linestyle="--" if bool(row.fit_is_edge) else "-",
                alpha=0.17 if bool(row.fit_is_edge) else 0.31,
            )
            ax.add_patch(ell)
        ax.scatter(
            sub["fit_pref_log2_sf"],
            sub["fit_pref_log2_tf"],
            color=color,
            s=np.where(sub["fit_is_edge"], 11, 18),
            alpha=np.where(sub["fit_is_edge"], 0.45, 0.72),
            linewidth=0,
            label=f"{FINAL_GROUP_LABELS[group]} (n={sub['unit_index'].nunique()})",
        )
    setup_log_axes(ax, x_log2=True, y_log2=True)
    ax.set_title("A. fitted passband centers and FWHM", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower left")


def plot_group_weight_contours(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    fit_w: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
) -> None:
    x, y = np.meshgrid(sf, tf, indexing="xy")
    for group in FINAL_GROUP_ORDER:
        idx = np.flatnonzero(atlas["final_sf_group"].astype(str).eq(group).to_numpy(dtype=bool))
        if idx.size == 0:
            continue
        mean_w = np.nanmean(fit_w[idx], axis=0).reshape(tf.size, sf.size)
        peak = float(np.nanmax(mean_w))
        if peak > 0:
            mean_w = mean_w / peak
        ax.contour(
            x,
            y,
            mean_w,
            levels=[0.25, 0.5, 0.75],
            colors=[FINAL_GROUP_COLORS[group]],
            linewidths=[1.0, 1.55, 2.1],
            alpha=0.9,
        )
    setup_log_axes(ax)
    ax.set_title("B. group-average fitted passband weight", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], color=FINAL_GROUP_COLORS[group], lw=1.8, label=FINAL_GROUP_LABELS[group])
        for group in FINAL_GROUP_ORDER
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")


def _stacked_fraction_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    bands: tuple[tuple[str, str, float, float], ...],
    prefix: str,
    colors: list[str],
    title: str,
) -> None:
    use = summary[summary["subset"].eq("all_fit")].set_index("final_sf_group")
    x = np.arange(len(FINAL_GROUP_ORDER))
    bottom = np.zeros(len(FINAL_GROUP_ORDER), dtype=np.float64)
    for color, (band_id, label, _lo, _hi) in zip(colors, bands, strict=True):
        vals = np.asarray(
            [float(use.loc[group, f"{prefix}_frac_{band_id}_mean"]) if group in use.index else np.nan for group in FINAL_GROUP_ORDER],
            dtype=np.float64,
        )
        ax.bar(x, vals, bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=0.7, label=label)
        bottom += np.nan_to_num(vals, nan=0.0)
    ax.set_xticks(x, [FINAL_GROUP_LABELS[group] for group in FINAL_GROUP_ORDER])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of passband weight")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def plot_passband_atlas(
    atlas: pd.DataFrame,
    passband_summary: pd.DataFrame,
    fit_w: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 8.25), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.90, wspace=0.30, hspace=0.35)
    plot_fit_ellipses(axes[0, 0], atlas)
    plot_group_weight_contours(axes[0, 1], atlas, fit_w, sf, tf)
    _stacked_fraction_bars(
        axes[1, 0],
        passband_summary,
        bands=SF_BANDS,
        prefix="fit_soft",
        colors=["#6BAED6", "#74C476", "#E6550D"],
        title="C. fitted passband SF mass on retinal grid",
    )
    _stacked_fraction_bars(
        axes[1, 1],
        passband_summary,
        bands=TF_BANDS,
        prefix="fit_soft",
        colors=["#9ECAE1", "#A1D99B", "#FC9272"],
        title="D. fitted passband TF mass on retinal grid",
    )
    fig.suptitle("Unit SF/TF passband atlas for overlap analysis", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.02,
        0.94,
        "Soft passbands use the dense 2D log-Gaussian fits; dashed ellipses are fits whose SF or TF center hit a sampled edge.",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    path = out_dir / "unit_sftf_passband_atlas.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_passband_atlas.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_passband_atlas.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlap_component_ratios(contrast: pd.DataFrame, out_dir: Path) -> Path:
    populations = [
        "low_mid_sf_all",
        "high_sf_all",
        "high_sf_aligned",
        "high_sf_oblique",
        "high_sf_orthogonal",
    ]
    pop_labels = {
        "low_mid_sf_all": "low+mid",
        "high_sf_all": "high",
        "high_sf_aligned": "aligned",
        "high_sf_oblique": "oblique",
        "high_sf_orthogonal": "orthogonal",
    }
    methods = ["fit_soft", "fit_fwhm"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.80, wspace=0.30)
    x = np.arange(len(populations), dtype=np.float64)
    width = 0.34
    for method, offset, alpha in (("fit_soft", -width / 2, 0.95), ("fit_fwhm", width / 2, 0.68)):
        sub = contrast[contrast["overlap_weighting"].eq(method)].set_index("population_key")
        along = np.asarray([sub.loc[p, "along_over_full_median"] if p in sub.index else np.nan for p in populations], dtype=float)
        across = np.asarray([sub.loc[p, "across_over_full_median"] if p in sub.index else np.nan for p in populations], dtype=float)
        axes[0].bar(x + offset, along, width=width, color=MODE_COLORS["along_only"], alpha=alpha, label=method)
        axes[1].bar(x + offset, across, width=width, color=MODE_COLORS["across_only"], alpha=alpha, label=method)
    for ax, title in zip(
        axes[:2],
        ["A. parallel-only overlap / full", "B. normal-only overlap / full"],
        strict=True,
    ):
        ax.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9)
        ax.set_xticks(x, [pop_labels[p] for p in populations], rotation=30, ha="right")
        ax.set_ylabel("median component/full")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    for method, marker in (("fit_soft", "o"), ("fit_fwhm", "s")):
        sub = contrast[contrast["overlap_weighting"].eq(method)].set_index("population_key")
        y = np.asarray([sub.loc[p, "across_minus_along_db_median"] if p in sub.index else np.nan for p in populations], dtype=float)
        axes[2].plot(x, y, marker=marker, linewidth=1.8, label=method)
    axes[2].axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[2].set_xticks(x, [pop_labels[p] for p in populations], rotation=30, ha="right")
    axes[2].set_ylabel("normal minus parallel overlap (dB)")
    axes[2].set_title("C. directional spectral advantage", loc="left", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Unit-weighted retinal power overlap by SSI population", x=0.02, ha="left", fontsize=12, fontweight="bold")
    path = out_dir / "unit_sftf_overlap_component_ratios.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_component_ratios.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_component_ratios.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlap_scatter(overlap: pd.DataFrame, out_dir: Path) -> Path:
    pair = overlap[(overlap["motion_mode"].eq("full_2d")) & (overlap["overlap_weighting"].eq("fit_soft"))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.80, wspace=0.30)
    for group in FINAL_GROUP_ORDER:
        sub = pair[pair["final_sf_group"].eq(group)]
        axes[0].scatter(
            sub["fit_pref_tf_hz"],
            sub["across_minus_along_db"],
            s=12,
            alpha=0.24,
            color=FINAL_GROUP_COLORS[group],
            linewidth=0,
            label=FINAL_GROUP_LABELS[group],
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
    axes[0].axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("fitted preferred TF (Hz)")
    axes[0].set_ylabel("normal minus parallel overlap (dB)")
    axes[0].set_title("A. unit-window overlap by TF preference", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)
    high = pair[pair["final_sf_group"].eq("high_sf") & pair["orientation_tuned"].astype(bool)].copy()
    for relation in RELATION_ORDER:
        sub = high[high["contour_relation"].eq(relation)]
        axes[1].scatter(
            sub["axis_delta_deg"],
            sub["across_minus_along_db"],
            s=15,
            alpha=0.35,
            color=RELATION_COLORS[relation],
            linewidth=0,
            label=RELATION_LABELS[relation],
        )
    axes[1].axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[1].axvline(15.0, color="0.65", linestyle="--", linewidth=0.8)
    axes[1].axvline(67.5, color="0.65", linestyle="--", linewidth=0.8)
    axes[1].set_xlim(0, 90)
    axes[1].set_xlabel("unit-contour axis delta (deg)")
    axes[1].set_ylabel("normal minus parallel overlap (dB)")
    axes[1].set_title("B. high-SF relation to contour axis", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per unit-window spectral overlap diagnostics", x=0.02, ha="left", fontsize=12, fontweight="bold")
    path = out_dir / "unit_sftf_overlap_passband_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_passband_scatter.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_passband_scatter.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_motion_dose_curves(dose_summary: pd.DataFrame, out_dir: Path) -> Path:
    populations = ("high_sf_all", "high_sf_aligned")
    dose_families = ("component_rms", "component_path")
    family_titles = {
        "component_rms": "component RMS excursion",
        "component_path": "component path length",
    }
    population_titles = {
        "high_sf_all": "high SF all",
        "high_sf_aligned": "high SF aligned",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.86, wspace=0.28, hspace=0.34)
    for row_i, population in enumerate(populations):
        for col_i, dose_family in enumerate(dose_families):
            ax = axes[row_i, col_i]
            for mode in ("along_only", "across_only"):
                sub = dose_summary[
                    dose_summary["population_key"].eq(population)
                    & dose_summary["dose_family"].eq(dose_family)
                    & dose_summary["overlap_weighting"].eq("fit_soft")
                    & dose_summary["motion_mode"].eq(mode)
                ].sort_values("dose_median")
                if sub.empty:
                    continue
                ax.plot(
                    sub["dose_median"],
                    sub["component_over_full_median"],
                    color=MODE_COLORS[mode],
                    marker="o" if mode == "across_only" else "s",
                    linewidth=1.9,
                    markersize=4.6,
                    label=MODE_LABELS[mode],
                )
            ax.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.04)
            ax.set_xlabel(f"{family_titles[dose_family]} (arcmin)")
            ax.set_ylabel("median component/full")
            ax.set_title(
                f"{chr(65 + row_i * 2 + col_i)}. {population_titles[population]}",
                loc="left",
                fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(
        "Unit-weighted spectral overlap along SSI motion-dose axes",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.90,
        "Dose bins are quantiles pooled across the parallel and normal projected component values within each population.",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    path = out_dir / "unit_sftf_overlap_motion_dose_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_motion_dose_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_motion_dose_curves.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmaps_with_passbands(
    detail: pd.DataFrame,
    atlas: pd.DataFrame,
    fit_w: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
    out_dir: Path,
) -> Path:
    summary = (
        detail.groupby(["motion_mode", "spatial_frequency_cpd", "temporal_frequency_hz"], sort=False)[
            "modulation_power_mean"
        ]
        .mean()
        .reset_index()
    )
    heat_values = db(summary["modulation_power_mean"].to_numpy(dtype=np.float64))
    finite = heat_values[np.isfinite(heat_values)]
    vmin = float(np.percentile(finite, 5.0))
    vmax = float(np.percentile(finite, 98.0))
    x, y = np.meshgrid(sf, tf, indexing="xy")
    group_contours: dict[str, np.ndarray] = {}
    for group in FINAL_GROUP_ORDER:
        idx = np.flatnonzero(atlas["final_sf_group"].astype(str).eq(group).to_numpy(dtype=bool))
        mean_w = np.nanmean(fit_w[idx], axis=0).reshape(tf.size, sf.size)
        peak = float(np.nanmax(mean_w))
        group_contours[group] = mean_w / peak if peak > 0 else mean_w
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.18, top=0.76, wspace=0.28)
    mesh = None
    for ax, mode in zip(axes, MODE_ORDER, strict=True):
        sub = summary[summary["motion_mode"].eq(mode)]
        mat = (
            sub.pivot_table(index="temporal_frequency_hz", columns="spatial_frequency_cpd", values="modulation_power_mean")
            .reindex(index=tf, columns=sf)
            .to_numpy(dtype=np.float64)
        )
        mesh = ax.pcolormesh(
            spatial_edges_from_centers(sf),
            temporal_edges(tf),
            db(mat),
            shading="auto",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )
        for group in FINAL_GROUP_ORDER:
            ax.contour(
                x,
                y,
                group_contours[group],
                levels=[0.5],
                colors=[FINAL_GROUP_COLORS[group]],
                linewidths=1.5,
            )
        setup_log_axes(ax)
        ax.set_title(MODE_LABELS[mode], loc="left", fontweight="bold", color=MODE_COLORS[mode])
    assert mesh is not None
    cax = fig.add_axes([0.895, 0.23, 0.014, 0.47])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("retinal modulation power (dB)")
    handles = [
        Line2D([0], [0], color=FINAL_GROUP_COLORS[group], lw=1.7, label=f"{FINAL_GROUP_LABELS[group]} mean weight 0.5")
        for group in FINAL_GROUP_ORDER
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        "Full-trace retinal spectra with group-average passband contours",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    path = out_dir / "unit_sftf_overlap_heatmaps_with_passbands.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_heatmaps_with_passbands.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "unit_sftf_overlap_heatmaps_with_passbands.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    atlas: pd.DataFrame,
    passband_summary: pd.DataFrame,
    contrast: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    fit_soft = contrast[contrast["overlap_weighting"].eq("fit_soft")].set_index("population_key")
    lines = [
        "# Unit SF/TF Passband Overlap",
        "",
        f"- Passband atlas: `{_relative(paths['atlas_png'])}`",
        f"- Component overlap summary: `{_relative(paths['ratio_png'])}`",
        f"- Scatter diagnostics: `{_relative(paths['scatter_png'])}`",
        f"- Heatmaps with passbands: `{_relative(paths['heatmap_png'])}`",
        f"- Motion-dose curves: `{_relative(paths['dose_png'])}`",
        "",
        "## Inputs",
        "",
        f"- Dense 2D fits: `{_relative(Path(args.fits_csv))}`",
        f"- Dense sampled responses: `{_relative(Path(args.points_csv))}`",
        f"- SSI unit metadata: `{_relative(Path(args.unit_table_csv))}`",
        f"- Full-trace retinal spectra: `{_relative(Path(args.power_detail_csv))}`",
        "",
        "## Working Definition",
        "",
        "For each unit and each window/component, overlap is the retinal modulation power averaged through a unit passband. `fit_soft` uses the full fitted 2D log-Gaussian; `fit_fwhm` uses only grid cells inside the fitted half-max contour. Component ratios are computed within each unit-window relative to full 2D motion.",
        "",
        "## Quick Readout",
        "",
        f"- Fitted units included: {int(atlas['unit_index'].nunique())}; high-SF fitted units: {int(atlas[atlas['final_sf_group'].eq('high_sf')]['unit_index'].nunique())}.",
    ]
    for key, label, _expr in POPULATIONS:
        if key not in fit_soft.index:
            continue
        row = fit_soft.loc[key]
        lines.append(
            f"- {label}: normal/full {float(row['across_over_full_median']):.3g}, "
            f"parallel/full {float(row['along_over_full_median']):.3g}, "
            f"normal-parallel {float(row['across_minus_along_db_median']):.2f} dB "
            f"(n={int(row['n_unit_window_pairs'])} unit-window pairs)."
        )
    lines.extend(["", "## Passband Summary", ""])
    for row in passband_summary[passband_summary["subset"].eq("all_fit")].itertuples(index=False):
        lines.append(
            f"- {row.final_sf_group_label}: n={int(row.n_units)}, median fitted center "
            f"{float(row.fit_pref_sf_cpd_median):.3g} cpd / {float(row.fit_pref_tf_hz_median):.3g} Hz; "
            f"median passband centroid {float(row.fit_soft_sf_centroid_cpd_median):.3g} cpd / "
            f"{float(row.fit_soft_tf_centroid_hz_median):.3g} Hz."
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_raw, points = load_atlas(args)
    detail, power_meta, power_values, sf, tf, flat_sf, flat_tf = load_power_grid(args)
    fit_w = fit_weights(atlas_raw, flat_sf, flat_tf)
    empirical_w = empirical_weights(points, atlas_raw, flat_sf, flat_tf)
    atlas = add_passband_metrics(atlas_raw, fit_w, empirical_w, flat_sf, flat_tf)
    overlap = compute_overlaps(
        atlas,
        power_meta,
        power_values,
        fit_w,
        empirical_w,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
    )
    population_summary, contrast = summarize_overlap(overlap)
    passband_summary = summarize_passbands(atlas)
    dose_summary = summarize_motion_dose(overlap)

    atlas_csv = out_dir / "unit_sftf_passband_atlas.csv"
    overlap_csv = out_dir / "unit_sftf_passband_overlap_by_window.csv"
    population_csv = out_dir / "unit_sftf_passband_overlap_population_summary.csv"
    contrast_csv = out_dir / "unit_sftf_passband_overlap_component_contrasts.csv"
    passband_csv = out_dir / "unit_sftf_passband_group_summary.csv"
    dose_csv = out_dir / "unit_sftf_passband_overlap_motion_dose_summary.csv"
    atlas.to_csv(atlas_csv, index=False)
    overlap.to_csv(overlap_csv, index=False)
    population_summary.to_csv(population_csv, index=False)
    contrast.to_csv(contrast_csv, index=False)
    passband_summary.to_csv(passband_csv, index=False)
    dose_summary.to_csv(dose_csv, index=False)

    paths = {
        "atlas_csv": atlas_csv,
        "overlap_csv": overlap_csv,
        "population_csv": population_csv,
        "contrast_csv": contrast_csv,
        "passband_csv": passband_csv,
        "dose_csv": dose_csv,
        "atlas_png": plot_passband_atlas(atlas, passband_summary, fit_w, sf, tf, out_dir),
        "ratio_png": plot_overlap_component_ratios(contrast, out_dir),
        "scatter_png": plot_overlap_scatter(overlap, out_dir),
        "dose_png": plot_motion_dose_curves(dose_summary, out_dir),
        "heatmap_png": plot_heatmaps_with_passbands(detail, atlas, fit_w, sf, tf, out_dir),
        "provenance_json": out_dir / "unit_sftf_passband_overlap_provenance.json",
    }
    provenance = {
        "fits_csv": Path(args.fits_csv),
        "points_csv": Path(args.points_csv),
        "unit_table_csv": Path(args.unit_table_csv),
        "power_detail_csv": Path(args.power_detail_csv),
        "image_sample_csv": Path(args.image_sample_csv),
        "out_dir": out_dir,
        "n_units": int(atlas["unit_index"].nunique()),
        "n_power_rows": int(power_meta.shape[0]),
        "n_windows": int(power_meta["image_pos"].nunique()),
        "n_sf_bins": int(sf.size),
        "n_tf_bins": int(tf.size),
        "min_osi": float(args.min_osi),
        "match_max_deg": float(args.match_max_deg),
        "orthogonal_min_deg": float(args.orthogonal_min_deg),
        "dose_families": [family[0] for family in DOSE_FAMILIES],
        "mode_order": MODE_ORDER,
        "population_definitions": [
            {"population_key": key, "population_label": label, "filter": expr} for key, label, expr in POPULATIONS
        ],
    }
    _write_json(paths["provenance_json"], provenance)
    write_readme(out_dir, args, atlas, passband_summary, contrast, paths)
    paths["readme"] = out_dir / "README.md"
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-csv", type=Path, default=DEFAULT_FITS_CSV)
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--unit-table-csv", type=Path, default=DEFAULT_UNIT_TABLE_CSV)
    parser.add_argument("--power-detail-csv", type=Path, default=DEFAULT_POWER_DETAIL_CSV)
    parser.add_argument("--image-sample-csv", type=Path, default=DEFAULT_IMAGE_SAMPLE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-osi", type=float, default=0.05)
    parser.add_argument("--match-max-deg", type=float, default=15.0)
    parser.add_argument("--orthogonal-min-deg", type=float, default=67.5)
    parser.add_argument("--max-windows", type=int, default=0, help="Optional smoke-test limit on image_pos values.")
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()

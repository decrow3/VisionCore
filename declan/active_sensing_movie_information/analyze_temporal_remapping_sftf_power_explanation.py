#!/usr/bin/env python3
"""Dumb SF/TF power-shift explanation for temporal-remapping SSI changes.

This is intentionally first-order.  It asks whether a unit's SSI increase is
large when eye motion maps image power near that unit's preferred spatial
frequency onto temporal frequencies near the unit's fitted TF preference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_DENSE_FIT_CSV = ROOT / (
    "outputs/active_sensing_movie_information/backimage_rr100_dense_sf_tf_speed_pref_groups_v1/"
    "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
)
DEFAULT_PARAMETRIC_MODEL_CSV = ROOT / (
    "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/"
    "rr100_sf_tf_parametric_models.csv"
)
EPS = 1e-12
SF_BANDS = (
    ("image_power_0_2_cpd_fraction", 0.0, 2.0, "0-2 cpd"),
    ("image_power_2_4_cpd_fraction", 2.0, 4.0, "2-4 cpd"),
    ("image_power_4_8_cpd_fraction", 4.0, 8.0, "4-8 cpd"),
    ("image_power_8plus_cpd_fraction", 8.0, math.inf, "8+ cpd"),
)
MEASURE_COLUMNS = [
    "unit_ssi_delta_absolute",
    "unit_sf_power_abs",
    "tf_match_fixed",
    "sftf_matched_power",
    "rms_across_velocity_deg_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dense-fit-csv", type=Path, default=DEFAULT_DENSE_FIT_CSV)
    parser.add_argument("--parametric-model-csv", type=Path, default=DEFAULT_PARAMETRIC_MODEL_CSV)
    parser.add_argument(
        "--preference-source",
        choices=("legacy", "parametric"),
        default="legacy",
        help=(
            "legacy keeps the original retiming preferred SF plus dense fitted TF. "
            "parametric replaces preferred SF/TF with the canonical RR100 parametric model table."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tf-match-sigma-octaves", type=float, default=1.0)
    parser.add_argument("--include-tf-edge-fits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-unit-rows", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


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


def sf_band_for_values(sf_cpd: pd.Series) -> pd.DataFrame:
    sf = pd.to_numeric(sf_cpd, errors="coerce").to_numpy(dtype=float)
    labels = np.full(sf.shape, "", dtype=object)
    columns = np.full(sf.shape, "", dtype=object)
    for column, lo, hi, label in SF_BANDS:
        mask = np.isfinite(sf) & (sf >= lo) & (sf < hi)
        labels[mask] = label
        columns[mask] = column
    return pd.DataFrame({"sf_power_band": labels, "sf_power_column": columns}, index=sf_cpd.index)


def sf_group_for_values(sf_cpd: pd.Series) -> pd.DataFrame:
    """Assign the coarse SF groups used by the parametric-preference figures."""
    sf = pd.to_numeric(sf_cpd, errors="coerce").to_numpy(dtype=float)
    groups = np.full(sf.shape, "", dtype=object)
    labels = np.full(sf.shape, "", dtype=object)
    specs = (
        ("low_sf", 0.0, 2.0, "Low SF"),
        ("middle_sf", 2.0, 4.0, "Middle SF"),
        ("high_sf", 4.0, math.inf, "High SF"),
    )
    for group, lo, hi, label in specs:
        mask = np.isfinite(sf) & (sf >= lo) & (sf < hi)
        groups[mask] = group
        labels[mask] = label
    return pd.DataFrame({"sf_group": groups, "sf_group_label": labels}, index=sf_cpd.index)


def load_parametric_fit_table(parametric_model_csv: Path, *, include_tf_edge_fits: bool) -> pd.DataFrame:
    param = pd.read_csv(parametric_model_csv)
    required = {
        "rr100_index",
        "model_valid",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "sf_fit_r2",
        "tf_fit_r2",
        "joint_parametric_surface_r2",
    }
    missing = sorted(required.difference(param.columns))
    if missing:
        raise ValueError(f"Parametric model CSV is missing required columns: {missing}")
    param = param[param["model_valid"].astype(bool)].copy()
    param["unit_index"] = pd.to_numeric(param["rr100_index"], errors="coerce").astype(int)
    support_min = pd.to_numeric(param.get("tf_fit_support_min_hz", 0.5), errors="coerce")
    support_max = pd.to_numeric(param.get("tf_fit_support_max_hz", 32.0), errors="coerce")
    pref_tf = pd.to_numeric(param["preferred_tf_hz"], errors="coerce")
    param["fit_edge_tf"] = np.isclose(pref_tf, support_min) | np.isclose(pref_tf, support_max)
    support_sf_min = pd.to_numeric(param.get("sf_fit_support_min_cpd", 1.0), errors="coerce")
    support_sf_max = pd.to_numeric(param.get("sf_fit_support_max_cpd", 11.313708), errors="coerce")
    pref_sf = pd.to_numeric(param["preferred_sf_cpd"], errors="coerce")
    param["fit_edge_sf"] = np.isclose(pref_sf, support_sf_min) | np.isclose(pref_sf, support_sf_max)
    if not include_tf_edge_fits:
        param = param[~param["fit_edge_tf"].astype(bool)].copy()
    out = pd.DataFrame(
        {
            "unit_index": param["unit_index"].astype(int),
            "parametric_model_valid": param["model_valid"].astype(bool),
            "fit_pref_sf_cpd": pd.to_numeric(param["preferred_sf_cpd"], errors="coerce"),
            "fit_pref_tf_hz": pd.to_numeric(param["preferred_tf_hz"], errors="coerce"),
            "fit_pref_speed_dps": (
                pd.to_numeric(param["preferred_tf_hz"], errors="coerce")
                / np.maximum(pd.to_numeric(param["preferred_sf_cpd"], errors="coerce"), EPS)
            ),
            "fit_sigma_tf_octaves": pd.to_numeric(param.get("tf_sigma_octaves", np.nan), errors="coerce"),
            "fit_fwhm_tf_octaves": pd.to_numeric(param.get("tf_fwhm_octaves", np.nan), errors="coerce"),
            "fit_r2": pd.to_numeric(param["joint_parametric_surface_r2"], errors="coerce"),
            "fit_edge_tf": param["fit_edge_tf"].astype(bool),
            "fit_edge_sf": param["fit_edge_sf"].astype(bool),
            "observed_peak_tf_hz": pd.to_numeric(param.get("tf_sampled_preferred_hz", param["preferred_tf_hz"]), errors="coerce"),
            "sf_fit_r2": pd.to_numeric(param["sf_fit_r2"], errors="coerce"),
            "tf_fit_r2": pd.to_numeric(param["tf_fit_r2"], errors="coerce"),
            "joint_parametric_surface_r2": pd.to_numeric(param["joint_parametric_surface_r2"], errors="coerce"),
        }
    )
    groups = sf_group_for_values(out["fit_pref_sf_cpd"])
    out["parametric_sf_group"] = groups["sf_group"]
    out["parametric_sf_group_label"] = groups["sf_group_label"]
    return out


def take_unit_sf_power(rows: pd.DataFrame) -> np.ndarray:
    values = np.full(rows.shape[0], np.nan, dtype=float)
    for column, _lo, _hi, _label in SF_BANDS:
        mask = rows["sf_power_column"].eq(column).to_numpy()
        if mask.any() and column in rows.columns:
            values[mask] = pd.to_numeric(rows.loc[mask, column], errors="coerce").to_numpy(dtype=float)
    return values


def finite_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    mask = np.ones(out.shape[0], dtype=bool)
    for column in columns:
        mask &= np.isfinite(pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float))
    return out.loc[mask].copy()


def standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not math.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def ols_fit(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[mask]
    x = x[mask]
    if y.size < 3:
        return {"n_rows": int(y.size), "r2": float("nan"), "corr": float("nan"), "intercept": float("nan")}
    design = np.column_stack([np.ones(y.size), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    if x.shape[1] == 1 and np.nanstd(x[:, 0]) > 0 and np.nanstd(y) > 0:
        corr = float(np.corrcoef(x[:, 0], y)[0, 1])
    else:
        corr = float("nan")
    out = {"n_rows": int(y.size), "r2": float(r2), "corr": corr, "intercept": float(coef[0])}
    for idx, value in enumerate(coef[1:]):
        out[f"slope_{idx}"] = float(value)
    return out


def centered_within(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby(group_cols, dropna=False, sort=False)
    for column in value_cols:
        values = pd.to_numeric(out[column], errors="coerce")
        out[f"{column}_within"] = values - group[column].transform("mean")
    return out


def mean_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False, sort=False)[MEASURE_COLUMNS].mean().reset_index()


def centered_measure_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    centered = centered_within(df, group_cols=group_cols, value_cols=MEASURE_COLUMNS)
    return pd.DataFrame({column: centered[f"{column}_within"] for column in MEASURE_COLUMNS})


def load_analysis_table(
    run_dir: Path,
    dense_fit_csv: Path,
    *,
    include_tf_edge_fits: bool,
    sigma_octaves: float,
    preference_source: str = "legacy",
    parametric_model_csv: Path = DEFAULT_PARAMETRIC_MODEL_CSV,
) -> pd.DataFrame:
    unit_cols = [
        "image_index",
        "trace_index",
        "condition_index",
        "condition_id",
        "condition_group",
        "traversal_frames",
        "traversal_duration_ms",
        "timing_placement",
        "retiming_profile",
        "unit_index",
        "sf_group",
        "preferred_sf_cpd",
        "rms_across_velocity_deg_s",
        "characteristic_motion_tf_hz",
        "unit_ssi_delta_absolute",
        "unit_ssi_bits_per_spike",
    ]
    image_cols = [
        "image_index",
        "image_patch_rms_contrast",
        "image_patch_std",
        "image_power_0_2_cpd_fraction",
        "image_power_2_4_cpd_fraction",
        "image_power_4_8_cpd_fraction",
        "image_power_8plus_cpd_fraction",
    ]
    rows = pd.read_csv(run_dir / "retiming_unit_observations.csv", usecols=unit_cols)
    images = pd.read_csv(run_dir / "image_feature_table.csv", usecols=image_cols)
    rows = rows.merge(images, on="image_index", how="left")
    rows["legacy_preferred_sf_cpd"] = rows["preferred_sf_cpd"]
    rows["legacy_sf_group"] = rows["sf_group"]
    rows["legacy_characteristic_motion_tf_hz"] = rows["characteristic_motion_tf_hz"]

    if str(preference_source) == "parametric":
        fit = load_parametric_fit_table(Path(parametric_model_csv), include_tf_edge_fits=include_tf_edge_fits)
        rows = rows.merge(fit, on="unit_index", how="inner")
        rows["preferred_sf_cpd"] = pd.to_numeric(rows["fit_pref_sf_cpd"], errors="coerce")
        rows["sf_group"] = rows["parametric_sf_group"]
        rows["sf_group_label"] = rows["parametric_sf_group_label"]
        across = pd.to_numeric(rows["rms_across_velocity_deg_s"], errors="coerce").to_numpy(dtype=float)
        pref_sf = pd.to_numeric(rows["preferred_sf_cpd"], errors="coerce").to_numpy(dtype=float)
        rows["characteristic_motion_tf_hz"] = across * pref_sf
        rows["preferred_sf_source_column"] = "rr100_parametric_preferred_sf_cpd"
    elif str(preference_source) == "legacy":
        dense = pd.read_csv(dense_fit_csv)
        dense = dense[dense["fit_ok"].astype(bool)].copy()
        if not include_tf_edge_fits:
            dense = dense[~dense["fit_edge_tf"].astype(bool)].copy()
        dense_cols = [
            "unit_index",
            "fit_pref_tf_hz",
            "fit_pref_sf_cpd",
            "fit_pref_speed_dps",
            "fit_sigma_tf_octaves",
            "fit_fwhm_tf_octaves",
            "fit_r2",
            "fit_edge_tf",
            "observed_peak_tf_hz",
        ]
        rows = rows.merge(dense[dense_cols], on="unit_index", how="inner")
    else:
        raise ValueError(f"Unknown preference_source={preference_source!r}")
    rows["preference_source"] = str(preference_source)
    bands = sf_band_for_values(rows["preferred_sf_cpd"])
    rows = pd.concat([rows, bands], axis=1)
    rows["unit_sf_power_fraction"] = take_unit_sf_power(rows)
    contrast = pd.to_numeric(rows["image_patch_rms_contrast"], errors="coerce").to_numpy(dtype=float)
    rows["unit_sf_power_abs"] = rows["unit_sf_power_fraction"].to_numpy(dtype=float) * contrast * contrast
    projected_tf = pd.to_numeric(rows["characteristic_motion_tf_hz"], errors="coerce").to_numpy(dtype=float)
    tf_pref = pd.to_numeric(rows["fit_pref_tf_hz"], errors="coerce").to_numpy(dtype=float)
    valid_tf = (projected_tf > 0.0) & (tf_pref > 0.0) & np.isfinite(projected_tf) & np.isfinite(tf_pref)
    log_distance = np.full(rows.shape[0], np.nan, dtype=float)
    log_distance[valid_tf] = np.log2(projected_tf[valid_tf] / tf_pref[valid_tf])
    rows["projected_vs_pref_tf_log2_distance"] = log_distance
    rows["tf_match_fixed"] = np.exp(-0.5 * (log_distance / float(sigma_octaves)) ** 2)
    rows.loc[~valid_tf, "tf_match_fixed"] = 0.0
    rows["sftf_matched_power"] = rows["unit_sf_power_abs"] * rows["tf_match_fixed"]
    rows["log_projected_tf_hz"] = np.log2(np.maximum(projected_tf, EPS))
    rows["log_pref_tf_hz"] = np.log2(np.maximum(tf_pref, EPS))
    return rows


def model_rows(df: pd.DataFrame, *, label: str) -> list[dict[str, Any]]:
    specs = [
        ("sf_power_only", ["unit_sf_power_abs"]),
        ("tf_match_only", ["tf_match_fixed"]),
        ("sf_power_x_tf_match", ["sftf_matched_power"]),
        ("sf_power_plus_tf_match", ["unit_sf_power_abs", "tf_match_fixed"]),
        ("velocity_only", ["rms_across_velocity_deg_s"]),
    ]
    rows: list[dict[str, Any]] = []
    work = finite_frame(df, ["unit_ssi_delta_absolute", "unit_sf_power_abs", "tf_match_fixed", "sftf_matched_power", "rms_across_velocity_deg_s"])
    y = pd.to_numeric(work["unit_ssi_delta_absolute"], errors="coerce").to_numpy(dtype=float)
    for name, predictors in specs:
        x = np.column_stack([standardize(pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)) for col in predictors])
        row = {"analysis_scale": label, "model": name, "predictors": "+".join(predictors)}
        row.update(ols_fit(y, x))
        rows.append(row)
    return rows


def normal_vs_static_rows(df: pd.DataFrame, out_dir: Path) -> list[dict[str, Any]]:
    normal = df[df["condition_group"].eq("original")].copy()
    if normal.empty:
        return []
    rows: list[dict[str, Any]] = []
    rows.extend(model_rows(normal, label="normal_vs_static_unit_observations"))
    rows.extend(model_rows(centered_measure_table(normal, ["unit_index"]), label="normal_vs_static_within_unit"))

    movie_means = mean_table(normal, ["image_index", "trace_index"])
    movie_means.to_csv(out_dir / "sftf_power_explanation_normal_vs_static_movie_means.csv", index=False)
    rows.extend(model_rows(movie_means, label="normal_vs_static_movie_means"))

    unit_means = mean_table(normal, ["unit_index"])
    unit_means.to_csv(out_dir / "sftf_power_explanation_normal_vs_static_unit_means.csv", index=False)
    rows.extend(model_rows(unit_means, label="normal_vs_static_unit_means"))
    return rows


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "condition_id",
        "condition_group",
        "retiming_profile",
        "timing_placement",
        "traversal_frames",
        "traversal_duration_ms",
    ]
    rows = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        row = {col: val for col, val in zip(group_cols, key, strict=True)}
        for column in ["unit_ssi_delta_absolute", "unit_sf_power_abs", "tf_match_fixed", "sftf_matched_power", "rms_across_velocity_deg_s"]:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            rows.append
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_sem"] = float(np.nanstd(values, ddof=1) / math.sqrt(np.isfinite(values).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def unit_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["unit_index", "condition_id", "condition_group", "retiming_profile", "timing_placement", "traversal_frames"]
    return (
        df.groupby(group_cols, dropna=False, sort=False)[
            ["unit_ssi_delta_absolute", "unit_sf_power_abs", "tf_match_fixed", "sftf_matched_power", "rms_across_velocity_deg_s"]
        ]
        .mean()
        .reset_index()
    )


def write_model_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    retiming = df[df["condition_group"].eq("retiming")].copy()
    rows = []
    rows.extend(normal_vs_static_rows(df, out_dir))
    rows.extend(model_rows(retiming, label="unit_observations_raw"))
    within = centered_measure_table(retiming, ["unit_index", "image_index", "trace_index"])
    rows.extend(model_rows(within, label="within_unit_image_trace"))

    unit_condition = unit_condition_summary(retiming)
    rows.extend(model_rows(unit_condition, label="unit_condition_means"))

    cond = condition_summary(retiming)
    cond_model = cond.rename(
        columns={
            "unit_ssi_delta_absolute_mean": "unit_ssi_delta_absolute",
            "unit_sf_power_abs_mean": "unit_sf_power_abs",
            "tf_match_fixed_mean": "tf_match_fixed",
            "sftf_matched_power_mean": "sftf_matched_power",
            "rms_across_velocity_deg_s_mean": "rms_across_velocity_deg_s",
        }
    )
    rows.extend(model_rows(cond_model, label="condition_means"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sftf_power_explanation_model_summary.csv", index=False)
    cond.to_csv(out_dir / "sftf_power_explanation_condition_summary.csv", index=False)
    return summary


def plot_model_steps(out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.axis("off")
    lines = [
        "Dumb SF/TF power-shift model",
        "",
        "1. Each image has spatial-frequency power: P_image(SF).",
        "2. Each unit has a preferred spatial frequency SF_u and temporal preference TF_u.",
        "3. Eye motion with velocity v moves image power at SF_u to TF_landing = SF_u x v.",
        "4. Matched power = P_image(SF_u) x Gaussian distance(TF_landing, TF_u).",
        "5. First test: normal motion vs counterfactually stabilized.",
        "6. Secondary test: do artificial retimings follow the same spectral ordering?",
    ]
    ax.text(0.02, 0.94, "\n".join(lines), va="top", ha="left", fontsize=13)
    path = out_dir / "sftf_power_explanation_01_dumb_model_steps.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_named_r2_summary(
    model_summary: pd.DataFrame,
    out_dir: Path,
    *,
    scales: list[str],
    title: str,
    filename: str,
) -> str:
    keep = model_summary[model_summary["model"].isin(["sf_power_only", "tf_match_only", "sf_power_x_tf_match", "velocity_only"])].copy()
    model_order = ["velocity_only", "sf_power_only", "tf_match_only", "sf_power_x_tf_match"]
    colors = {
        "velocity_only": "#8c8c8c",
        "sf_power_only": "#4c78a8",
        "tf_match_only": "#f58518",
        "sf_power_x_tf_match": "#54a24b",
    }
    fig, axes = plt.subplots(1, len(scales), figsize=(3.8 * len(scales), 4.0), sharey=True)
    axes_arr = np.atleast_1d(axes)
    max_r2 = float(np.nanmax(keep[keep["analysis_scale"].isin(scales)]["r2"])) if not keep.empty else 0.05
    for ax, scale in zip(axes_arr, scales, strict=True):
        part = keep[keep["analysis_scale"].eq(scale)].set_index("model")
        values = [float(part.loc[name, "r2"]) if name in part.index else np.nan for name in model_order]
        ax.bar(np.arange(len(model_order)), values, color=[colors[name] for name in model_order], alpha=0.88)
        ax.set_title(scale.replace("_", "\n"), fontsize=10)
        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels([name.replace("_", "\n") for name in model_order], rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", color="#e7e7e7", lw=0.7)
        ax.set_ylim(0.0, max(0.05, max_r2 * 1.15))
    axes_arr[0].set_ylabel("R2")
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_normal_vs_static_r2_summary(model_summary: pd.DataFrame, out_dir: Path) -> str:
    return plot_named_r2_summary(
        model_summary,
        out_dir,
        scales=[
            "normal_vs_static_unit_observations",
            "normal_vs_static_within_unit",
            "normal_vs_static_movie_means",
            "normal_vs_static_unit_means",
        ],
        title="First question: normal motion vs counterfactually stabilized",
        filename="sftf_power_explanation_02_normal_vs_static_r2_summary.png",
    )


def plot_r2_summary(model_summary: pd.DataFrame, out_dir: Path) -> str:
    return plot_named_r2_summary(
        model_summary,
        out_dir,
        scales=["unit_observations_raw", "within_unit_image_trace", "unit_condition_means", "condition_means"],
        title="Secondary question: how much does the retiming-condition predictor explain?",
        filename="sftf_power_explanation_04_retiming_r2_summary.png",
    )


def plot_normal_vs_static_scatter(df: pd.DataFrame, out_dir: Path) -> str:
    normal = df[df["condition_group"].eq("original")].copy()
    movie = mean_table(normal, ["image_index", "trace_index"])
    unit = mean_table(normal, ["unit_index"])
    panels = [
        ("Movie means", movie, "image x trace mean over units"),
        ("Unit means", unit, "unit mean over images x traces"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, (title, table, subtitle) in zip(axes, panels, strict=True):
        x = table["sftf_matched_power"].to_numpy(dtype=float)
        y = table["unit_ssi_delta_absolute"].to_numpy(dtype=float)
        fit = ols_fit(y, standardize(x)[:, None])
        ax.scatter(x, y, s=32, alpha=0.78, color="#4c78a8")
        if np.isfinite(x).sum() >= 3:
            idx = np.argsort(x)
            pred = fit["intercept"] + fit["slope_0"] * standardize(x)
            ax.plot(x[idx], pred[idx], color="#222222", lw=1.2)
        ax.set_title(f"{title}\n{subtitle}; R2={fit['r2']:.3f}", fontsize=10)
        ax.set_xlabel("matched SF/TF power proxy")
        ax.set_ylabel("SSI delta: normal motion - stabilized")
        ax.grid(True, color="#e7e7e7", lw=0.7)
    fig.suptitle("First check: does matched SF/TF power explain normal-vs-static SSI?", y=1.02)
    fig.tight_layout()
    path = out_dir / "sftf_power_explanation_03_normal_vs_static_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_condition_scatter(df: pd.DataFrame, out_dir: Path) -> str:
    cond = condition_summary(df[df["condition_group"].eq("retiming")].copy())
    x = cond["sftf_matched_power_mean"].to_numpy(dtype=float)
    y = cond["unit_ssi_delta_absolute_mean"].to_numpy(dtype=float)
    fit = ols_fit(y, standardize(x)[:, None])
    pred = fit["intercept"] + fit["slope_0"] * standardize(x)
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    for (profile, placement), group in cond.groupby(["retiming_profile", "timing_placement"], dropna=False):
        ax.scatter(
            group["sftf_matched_power_mean"],
            group["unit_ssi_delta_absolute_mean"],
            s=50,
            label=f"{profile}/{placement}",
            alpha=0.88,
        )
    idx = np.argsort(x)
    ax.plot(x[idx], pred[idx], color="#222222", lw=1.2, label=f"linear fit R2={fit['r2']:.3f}")
    ax.set_xlabel("mean matched SF/TF power proxy")
    ax.set_ylabel("mean unit SSI delta vs static")
    ax.set_title("Condition means: observed SSI vs dumb spectral predictor")
    ax.grid(True, color="#e7e7e7", lw=0.7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "sftf_power_explanation_05_retiming_condition_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_within_scatter(df: pd.DataFrame, out_dir: Path) -> str:
    retiming = df[df["condition_group"].eq("retiming")].copy()
    centered = centered_within(
        retiming,
        group_cols=["unit_index", "image_index", "trace_index"],
        value_cols=["unit_ssi_delta_absolute", "sftf_matched_power"],
    )
    x = centered["sftf_matched_power_within"].to_numpy(dtype=float)
    y = centered["unit_ssi_delta_absolute_within"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    fit = ols_fit(y, standardize(x)[:, None])
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis")
    xx = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
    yy = fit["intercept"] + fit["slope_0"] * ((xx - np.nanmean(x)) / max(np.nanstd(x), EPS))
    ax.plot(xx, yy, color="white", lw=1.6, label=f"linear fit R2={fit['r2']:.4f}")
    ax.axhline(0.0, color="#eeeeee", lw=0.8)
    ax.axvline(0.0, color="#eeeeee", lw=0.8)
    ax.set_xlabel("matched power proxy, centered within unit x image x trace")
    ax.set_ylabel("unit SSI delta, centered within unit x image x trace")
    ax.set_title("Strict check: does changing retiming move SSI as predicted?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "sftf_power_explanation_06_retiming_within_family_hexbin.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_tf_landing(df: pd.DataFrame, out_dir: Path) -> str:
    retiming = df[df["condition_group"].eq("retiming")].copy()
    unit_condition = unit_condition_summary(retiming)
    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    sc = ax.scatter(
        unit_condition["rms_across_velocity_deg_s"],
        unit_condition["tf_match_fixed"],
        c=unit_condition["unit_ssi_delta_absolute"],
        s=18,
        alpha=0.72,
        cmap="magma",
    )
    ax.set_xlabel("RMS across-contour velocity (deg/s)")
    ax.set_ylabel("TF match at unit preferred SF")
    ax.set_title("Do retimed velocities land near unit TF preferences?")
    ax.grid(True, color="#e7e7e7", lw=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("mean unit SSI delta vs static")
    fig.tight_layout()
    path = out_dir / "sftf_power_explanation_07_retiming_tf_landing_map.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "sftf_power_explanation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_analysis_table(
        run_dir,
        Path(args.dense_fit_csv),
        include_tf_edge_fits=bool(args.include_tf_edge_fits),
        sigma_octaves=float(args.tf_match_sigma_octaves),
        preference_source=str(args.preference_source),
        parametric_model_csv=Path(args.parametric_model_csv),
    )
    unit_rows_path = out_dir / "sftf_power_explanation_unit_rows.csv"
    if bool(args.write_unit_rows):
        rows.to_csv(unit_rows_path, index=False)
    model_summary = write_model_summary(rows, out_dir)
    figures = [
        plot_model_steps(out_dir),
        plot_normal_vs_static_r2_summary(model_summary, out_dir),
        plot_normal_vs_static_scatter(rows, out_dir),
        plot_r2_summary(model_summary, out_dir),
        plot_condition_scatter(rows, out_dir),
        plot_within_scatter(rows, out_dir),
        plot_tf_landing(rows, out_dir),
    ]
    write_json(
        out_dir / "sftf_power_explanation_summary.json",
        {
            "analysis": "temporal_remapping_sftf_power_explanation",
            "run_dir": run_dir,
            "dense_fit_csv": Path(args.dense_fit_csv),
            "parametric_model_csv": Path(args.parametric_model_csv),
            "preference_source": str(args.preference_source),
            "out_dir": out_dir,
            "tf_match_sigma_octaves": float(args.tf_match_sigma_octaves),
            "include_tf_edge_fits": bool(args.include_tf_edge_fits),
            "n_rows": int(rows.shape[0]),
            "n_units": int(rows["unit_index"].nunique()),
            "n_images": int(rows["image_index"].nunique()),
            "n_traces": int(rows["trace_index"].nunique()),
            "n_conditions": int(rows["condition_id"].nunique()),
            "figures": figures,
            "outputs": {
                "unit_rows": unit_rows_path if bool(args.write_unit_rows) else None,
                "model_summary": out_dir / "sftf_power_explanation_model_summary.csv",
                "condition_summary": out_dir / "sftf_power_explanation_condition_summary.csv",
                "normal_vs_static_movie_means": out_dir / "sftf_power_explanation_normal_vs_static_movie_means.csv",
                "normal_vs_static_unit_means": out_dir / "sftf_power_explanation_normal_vs_static_unit_means.csv",
            },
            "contract": (
                "First-order SF/TF predictor: image power in the coarse band containing each unit's preferred SF "
                "times a fixed one-octave Gaussian match between SF_u * across-contour velocity and the unit's "
                "dense grating TF preference. The primary diagnostic is normal natural motion versus the "
                "counterfactually stabilized movie; retiming-condition summaries are secondary descriptive checks. "
                "Fits are descriptive OLS diagnostics, not a trained encoding model."
            ),
        },
    )
    print(f"wrote SF/TF power explanation diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

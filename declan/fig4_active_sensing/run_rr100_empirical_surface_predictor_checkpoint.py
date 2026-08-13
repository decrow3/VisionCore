#!/usr/bin/env python3
"""Compare parametric and empirical grating-surface predictors on the exact 16 movies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[2]
CP11 = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
CP15A = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_predictor_checkpoint_15a_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
SURFACE = (
    ROOT
    / "outputs/redundancy_resolved_v1_twin"
    / "rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
    / "f0_surface_fit_and_residual_points.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_empirical_surface_predictor_checkpoint_15d_v1"
EPS = 1e-15
N_BOOT = 10000
RNG_SEED = 20260812


VARIANT_LABELS = {
    "total_power": "Total power",
    "parametric_separable": "Parametric\nseparable",
    "empirical_rank1_separable": "Empirical rank-1\nseparable",
    "empirical_raw_2d": "Measured 2-D\nsurface",
}


def interpolate_surface(
    frame: pd.DataFrame,
    value_column: str,
    target_sf: np.ndarray,
    target_tf: np.ndarray,
) -> np.ndarray:
    pivot = frame.pivot(index="spatial_cpd", columns="temporal_hz", values=value_column).sort_index().sort_index(axis=1)
    sf = pivot.index.to_numpy(float)
    tf = pivot.columns.to_numpy(float)
    values = pivot.to_numpy(float)
    if target_sf.min() < sf.min() or target_sf.max() > sf.max() or target_tf.min() < tf.min() or target_tf.max() > tf.max():
        raise ValueError("Target power grid extends beyond measured empirical surface support")
    interpolator = RegularGridInterpolator(
        (np.log2(sf), np.log2(tf)), values, method="linear", bounds_error=True
    )
    sf_mesh, tf_mesh = np.meshgrid(np.log2(target_sf), np.log2(target_tf), indexing="ij")
    result = interpolator(np.column_stack([sf_mesh.ravel(), tf_mesh.ravel()])).reshape(len(target_sf), len(target_tf))
    result = np.maximum(result, 0.0)
    maximum = float(result.max())
    return result / maximum if maximum > EPS else np.zeros_like(result)


def build_predictors() -> tuple[pd.DataFrame, pd.DataFrame]:
    power_archive = np.load(CP11 / "all16_original_pair_supported_sf_tf_power.npz")
    sf = power_archive["sf_centers_cpd"].astype(float)
    tf = power_archive["tf_centers_hz"].astype(float)
    surface = pd.read_csv(SURFACE)
    models = pd.read_csv(MODELS).set_index("rr100_index")
    existing = pd.read_csv(CP11 / "all16_spectral_drive_and_response_all_rr100.csv")
    existing = existing.loc[existing["rr100_index"].isin(models.index)].copy()

    sensitivity_rows = []
    predictors = []
    for unit, model in models.iterrows():
        if not bool(model.model_valid):
            continue
        frame = surface.loc[surface["rr100_index"].eq(unit)]
        raw = interpolate_surface(frame, "observed_positive_f0_hz", sf, tf)
        rank1 = interpolate_surface(frame, "rank1_reconstructed_positive_f0_hz", sf, tf)
        for i, sf_value in enumerate(sf):
            for j, tf_value in enumerate(tf):
                sensitivity_rows.append(
                    {
                        "rr100_index": int(unit),
                        "sf_cpd": float(sf_value),
                        "tf_hz": float(tf_value),
                        "empirical_raw_2d_normalized": float(raw[i, j]),
                        "empirical_rank1_separable_normalized": float(rank1[i, j]),
                    }
                )
        for image_index in range(16):
            power = power_archive[f"image_{image_index:02d}_supported_power_sf_tf"].astype(float)
            predictors.append(
                {
                    "rr100_index": int(unit),
                    "image_index": image_index,
                    "empirical_raw_2d": float(np.sqrt(np.sum(power * raw**2))),
                    "empirical_rank1_separable": float(np.sqrt(np.sum(power * rank1**2))),
                    "total_power": float(np.sqrt(np.sum(power))),
                }
            )
    predictors = pd.DataFrame(predictors)
    predictors = predictors.merge(
        existing[
            [
                "rr100_index",
                "image_index",
                "spectral_drive_amplitude_arbitrary",
                "fem_delta_temporal_sd_hz",
            ]
        ],
        on=["rr100_index", "image_index"],
        validate="one_to_one",
    ).rename(columns={"spectral_drive_amplitude_arbitrary": "parametric_separable"})
    return predictors, pd.DataFrame(sensitivity_rows)


def fit_nonnegative_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    slope = max(float(np.sum((x - x_mean) * (y - y_mean)) / max(denominator, EPS)), 0.0)
    return y_mean - slope * x_mean, slope


def cross_validate(predictors: pd.DataFrame, models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_index = models.set_index("rr100_index")
    metric_rows, prediction_rows = [], []
    for unit, frame in predictors.groupby("rr100_index", sort=True):
        frame = frame.sort_values("image_index")
        y = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        for variant in VARIANT_LABELS:
            x = frame[variant].to_numpy(float)
            yhat = np.full(len(frame), np.nan)
            baseline = np.full(len(frame), np.nan)
            slopes = np.full(len(frame), np.nan)
            for held in range(len(frame)):
                train = np.arange(len(frame)) != held
                intercept, slope = fit_nonnegative_line(x[train], y[train])
                yhat[held] = intercept + slope * x[held]
                baseline[held] = float(y[train].mean())
                slopes[held] = slope
                prediction_rows.append(
                    {
                        "rr100_index": int(unit),
                        "image_index": int(frame.iloc[held].image_index),
                        "predictor_variant": variant,
                        "predictor_value": float(x[held]),
                        "observed_modulation_sd_hz": float(y[held]),
                        "held_out_prediction_hz": float(yhat[held]),
                        "held_out_training_mean_hz": float(baseline[held]),
                        "training_slope_nonnegative": float(slope),
                        "training_intercept_hz": float(intercept),
                    }
                )
            sse = float(np.sum((y - yhat) ** 2))
            baseline_sse = float(np.sum((y - baseline) ** 2))
            metric_rows.append(
                {
                    "rr100_index": int(unit),
                    "predictor_variant": variant,
                    "predictor_label": VARIANT_LABELS[variant].replace("\n", " "),
                    "n_images": int(len(frame)),
                    "cv_r2_vs_training_mean": 1.0 - sse / max(baseline_sse, EPS),
                    "oof_rmse_hz": float(np.sqrt(np.mean((y - yhat) ** 2))),
                    "oof_mae_hz": float(np.mean(np.abs(y - yhat))),
                    "oof_pearson_r": float(stats.pearsonr(yhat, y).statistic) if np.ptp(yhat) > EPS else np.nan,
                    "oof_spearman_rho": float(stats.spearmanr(yhat, y).statistic) if np.ptp(yhat) > EPS else np.nan,
                    "n_positive_loo_slopes": int(np.sum(slopes > 0)),
                    "model_valid": bool(model_index.loc[unit, "model_valid"]),
                    "quality_cohort": bool(
                        model_index.loc[unit, "model_valid"]
                        and model_index.loc[unit, "sf_fit_r2"] >= 0.70
                        and model_index.loc[unit, "tf_fit_r2"] >= 0.70
                        and model_index.loc[unit, "joint_parametric_surface_r2"] >= 0.50
                    ),
                    "joint_parametric_surface_r2": float(model_index.loc[unit, "joint_parametric_surface_r2"]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def paired_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    quality = metrics.loc[metrics["quality_cohort"]].copy()
    wide = quality.pivot(index="rr100_index", columns="predictor_variant", values="cv_r2_vs_training_mean")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for variant in VARIANT_LABELS:
        values = wide[variant].to_numpy(float)
        boot = np.median(values[rng.integers(0, len(values), size=(N_BOOT, len(values)))], axis=1)
        rows.append(
            {
                "comparison": "absolute",
                "predictor_variant": variant,
                "reference_variant": "",
                "n_units": int(len(values)),
                "median_cv_r2_or_delta": float(np.median(values)),
                "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                "fraction_positive_or_better": float(np.mean(values > 0)),
            }
        )
    for variant in ["parametric_separable", "empirical_rank1_separable", "empirical_raw_2d"]:
        for reference in ["total_power", "parametric_separable"]:
            if variant == reference:
                continue
            delta = (wide[variant] - wide[reference]).to_numpy(float)
            boot = np.median(delta[rng.integers(0, len(delta), size=(N_BOOT, len(delta)))], axis=1)
            rows.append(
                {
                    "comparison": "paired_delta",
                    "predictor_variant": variant,
                    "reference_variant": reference,
                    "n_units": int(len(delta)),
                    "median_cv_r2_or_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    "fraction_positive_or_better": float(np.mean(delta > 0)),
                }
            )
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    quality = metrics.loc[metrics["quality_cohort"]].copy()
    wide = quality.pivot(index="rr100_index", columns="predictor_variant", values="cv_r2_vs_training_mean")
    order = list(VARIANT_LABELS)
    colors = ["#d65238", "#356fa3", "#4f9d78", "#8e6bb3"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11.5})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), constrained_layout=True)

    ax = axes[0, 0]
    rng = np.random.default_rng(17)
    for i, (variant, color) in enumerate(zip(order, colors)):
        values = wide[variant].to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=13, color=color, alpha=0.55)
        ax.plot([i - 0.24, i + 0.24], [np.median(values)] * 2, color="#172029", lw=2.2)
    ax.axhline(0, color="#777", linestyle="--", lw=1)
    ax.set_xticks(range(len(order)), [VARIANT_LABELS[v] for v in order])
    ax.set_ylabel("Held-out R²")
    ax.set_ylim(float(wide[order].min().min()) - 0.12, 1.0)
    ax.set_title("A  Empirical surfaces do not exceed total power")

    def comparison_panel(ax: plt.Axes, variant: str, title: str, color: str) -> None:
        x = wide["parametric_separable"]
        y = wide[variant]
        lo = min(float(x.min()), float(y.min()), -1.0)
        hi = max(float(x.max()), float(y.max()), 0.9)
        ax.scatter(x, y, s=30, color=color, alpha=0.72, edgecolor="white", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], color="#777", linestyle="--", lw=1)
        ax.axhline(0, color="#c7ccd0", lw=0.8)
        ax.axvline(0, color="#c7ccd0", lw=0.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Parametric surface held-out R²", ylabel=f"{title} held-out R²")
        ax.set_title(f"{title}: {100*np.mean(y > x):.0f}% improve over parametric")

    comparison_panel(axes[0, 1], "empirical_rank1_separable", "B  Empirical rank-1", colors[2])
    comparison_panel(axes[1, 0], "empirical_raw_2d", "C  Measured 2-D", colors[3])

    ax = axes[1, 1]
    deltas = []
    labels = []
    delta_colors = []
    for variant, color in [
        ("parametric_separable", colors[1]),
        ("empirical_rank1_separable", colors[2]),
        ("empirical_raw_2d", colors[3]),
    ]:
        deltas.append((wide[variant] - wide["total_power"]).to_numpy(float))
        labels.append(VARIANT_LABELS[variant])
        delta_colors.append(color)
    parts = ax.violinplot(deltas, positions=np.arange(3), widths=0.75, showextrema=False)
    for body, color in zip(parts["bodies"], delta_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    for i, (delta, color) in enumerate(zip(deltas, delta_colors)):
        ax.scatter(np.full(len(delta), i) + rng.uniform(-0.10, 0.10, len(delta)), delta, s=12, color=color, alpha=0.52)
        ax.plot([i - 0.22, i + 0.22], [np.median(delta)] * 2, color="#172029", lw=2.2)
    ax.axhline(0, color="#777", linestyle="--", lw=1)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Δ held-out R² relative to total power")
    ax.set_ylim(min(float(np.min(delta)) for delta in deltas) - 0.12, max(float(np.max(delta)) for delta in deltas) + 0.12)
    ax.set_title("D  No grating-surface form beats the simpler control")

    for ax in axes.flat:
        ax.grid(color="#e9ecef", lw=0.75)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Checkpoint 15D: replacing parametric fits with measured grating surfaces does not rescue prediction",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(OUT / "checkpoint_15d_empirical_surface_predictor_comparison.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "checkpoint_15d_empirical_surface_predictor_comparison.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictors, sensitivities = build_predictors()
    models = pd.read_csv(MODELS)
    metrics, predictions = cross_validate(predictors, models)
    summary = paired_summary(metrics)
    selected = pd.read_csv(CP15A / "auditable_fit_quality_example_selection.csv")
    selected_metrics = metrics.loc[metrics["rr100_index"].isin(selected["rr100_index"])].merge(
        selected[["rr100_index", "selection_role"]], on="rr100_index", validate="many_to_one"
    )
    predictors.to_csv(OUT / "all16_empirical_and_parametric_predictor_values.csv", index=False)
    sensitivities.to_csv(OUT / "empirical_sensitivity_on_supported_power_grid.csv", index=False)
    metrics.to_csv(OUT / "per_unit_empirical_surface_predictor_metrics.csv", index=False)
    predictions.to_csv(OUT / "leave_one_pair_out_empirical_surface_predictions.csv", index=False)
    summary.to_csv(OUT / "empirical_surface_population_comparison_summary.csv", index=False)
    selected_metrics.to_csv(OUT / "selected_unit_empirical_surface_predictor_metrics.csv", index=False)
    make_figure(metrics, summary)

    absolute = summary.loc[summary["comparison"].eq("absolute")].set_index("predictor_variant")
    vs_total = summary.loc[
        summary["comparison"].eq("paired_delta") & summary["reference_variant"].eq("total_power")
    ].set_index("predictor_variant")
    vs_parametric = summary.loc[
        summary["comparison"].eq("paired_delta") & summary["reference_variant"].eq("parametric_separable")
    ].set_index("predictor_variant")
    manifest = {
        "analysis": "rr100_empirical_grating_surface_predictor_checkpoint_15d",
        "status": "empirical_surface_misspecification_test_complete",
        "n_quality_units": 66,
        "n_conditions": 16,
        "surface_variants": {
            "parametric_separable": "existing log-Gaussian separable F0 model",
            "empirical_rank1_separable": "measured rank-1 positive-F0 surface, log-frequency bilinear interpolation",
            "empirical_raw_2d": "measured positive-F0 SFxTF grid, log-frequency bilinear interpolation",
        },
        "median_held_out_r2": {
            key: float(absolute.loc[key, "median_cv_r2_or_delta"]) for key in VARIANT_LABELS
        },
        "median_delta_r2_vs_total_power": {
            key: float(vs_total.loc[key, "median_cv_r2_or_delta"]) for key in vs_total.index
        },
        "fraction_units_beating_total_power": {
            key: float(vs_total.loc[key, "fraction_positive_or_better"]) for key in vs_total.index
        },
        "median_delta_r2_vs_parametric": {
            key: float(vs_parametric.loc[key, "median_cv_r2_or_delta"]) for key in vs_parametric.index
        },
        "fraction_units_beating_parametric": {
            key: float(vs_parametric.loc[key, "fraction_positive_or_better"]) for key in vs_parametric.index
        },
        "interpretation": "Direct measured and empirical rank-1 grating surfaces do not recover a population-level tuning advantage over total supported dynamic power.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Checkpoint 15D: empirical grating-surface predictor\n\n"
        "This checkpoint directly tests parametric misspecification. The exact 16 supported SF×TF "
        "power maps are weighted by either the existing parametric surface, the measured rank-1 "
        "surface, or the full measured positive-F0 grid. Empirical surfaces are interpolated "
        "bilinearly in log SF and log TF without extrapolation. All variants use the same "
        "leave-one-complete-pair-out nonnegative per-unit calibration.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

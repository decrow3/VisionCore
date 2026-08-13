#!/usr/bin/env python3
"""Population checkpoint for fit quality versus predictor improvement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
CP11 = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_population_checkpoint_15c_v1"
RNG_SEED = 20260812
N_BOOT = 10000


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return np.nan
    result = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
    return float(result.statistic)


def build_per_unit() -> pd.DataFrame:
    variants = pd.read_csv(CP11 / "predictor_variant_per_unit_explainability.csv")
    primary = variants.loc[
        variants["quality_cohort"].astype(bool)
        & variants["predictor_variant"].eq("primary_unit_specific_amplitude")
    ].set_index("rr100_index")
    wide = variants.loc[variants["quality_cohort"].astype(bool)].pivot(
        index="rr100_index",
        columns="predictor_variant",
        values="cv_r2_vs_train_mean_baseline",
    )
    predictions = pd.read_csv(CP11 / "predictor_variant_leave_one_image_out_predictions.csv")
    predictions = predictions.loc[
        predictions["rr100_index"].isin(primary.index)
        & predictions["predictor_variant"].isin(
            ["primary_unit_specific_amplitude", "total_power_amplitude_no_unit_tuning"]
        )
    ].copy()
    profile = predictions.pivot_table(
        index=["rr100_index", "held_out_image_index", "observed_modulation_sd_hz"],
        columns="predictor_variant",
        values="held_out_predicted_modulation_sd_hz",
        aggfunc="first",
    ).reset_index()
    rows = []
    for unit, frame in profile.groupby("rr100_index", sort=True):
        y = frame["observed_modulation_sd_hz"].to_numpy(float)
        unit_pred = frame["primary_unit_specific_amplitude"].to_numpy(float)
        total_pred = frame["total_power_amplitude_no_unit_tuning"].to_numpy(float)
        unit_error = y - unit_pred
        total_error = y - total_pred
        rows.append(
            {
                "rr100_index": int(unit),
                "delta_cv_r2_unit_minus_total": float(
                    wide.loc[unit, "primary_unit_specific_amplitude"]
                    - wide.loc[unit, "total_power_amplitude_no_unit_tuning"]
                ),
                "delta_rmse_hz_total_minus_unit": float(
                    np.sqrt(np.mean(total_error**2)) - np.sqrt(np.mean(unit_error**2))
                ),
                "delta_mae_hz_total_minus_unit": float(
                    np.mean(np.abs(total_error)) - np.mean(np.abs(unit_error))
                ),
                "delta_sse_hz2_total_minus_unit": float(np.sum(total_error**2) - np.sum(unit_error**2)),
                "delta_pearson_unit_minus_total": safe_correlation(y, unit_pred, "pearson")
                - safe_correlation(y, total_pred, "pearson"),
                "delta_spearman_unit_minus_total": safe_correlation(y, unit_pred, "spearman")
                - safe_correlation(y, total_pred, "spearman"),
                "response_modulation_sd_across_images_hz": float(np.std(y)),
            }
        )
    metrics = pd.DataFrame(rows).set_index("rr100_index")
    models = pd.read_csv(MODELS).set_index("rr100_index")
    model_columns = [
        "joint_parametric_surface_r2",
        "sf_fit_r2",
        "tf_fit_r2",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "sf_fwhm_octaves",
        "tf_fwhm_octaves",
        "joint_rank1_gain_f0_hz",
        "joint_parametric_surface_rmse_hz",
        "recorded_sf_curve_r_full_support",
    ]
    result = metrics.join(models[model_columns], validate="one_to_one").reset_index()
    result["log10_response_sd"] = np.log10(result["response_modulation_sd_across_images_hz"])
    result["log2_preferred_sf"] = np.log2(result["preferred_sf_cpd"])
    result["log2_preferred_tf"] = np.log2(result["preferred_tf_hz"])
    result["log10_joint_gain"] = np.log10(result["joint_rank1_gain_f0_hz"].clip(lower=1e-12))
    return result


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    index = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    medians = np.median(values[index], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def threshold_summary(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        frame = table.loc[table["joint_parametric_surface_r2"] >= threshold]
        if len(frame) < 3:
            continue
        delta = frame["delta_cv_r2_unit_minus_total"].to_numpy(float)
        rmse = frame["delta_rmse_hz_total_minus_unit"].to_numpy(float)
        delta_ci = bootstrap_median(delta, rng)
        rmse_ci = bootstrap_median(rmse, rng)
        rows.append(
            {
                "joint_fit_r2_minimum": threshold,
                "n_units": int(len(frame)),
                "median_delta_cv_r2": float(np.median(delta)),
                "median_delta_cv_r2_ci95_low": delta_ci[0],
                "median_delta_cv_r2_ci95_high": delta_ci[1],
                "fraction_delta_cv_r2_positive": float(np.mean(delta > 0)),
                "median_delta_rmse_hz_total_minus_unit": float(np.median(rmse)),
                "median_delta_rmse_ci95_low": rmse_ci[0],
                "median_delta_rmse_ci95_high": rmse_ci[1],
                "fraction_delta_rmse_positive": float(np.mean(rmse > 0)),
            }
        )
    return pd.DataFrame(rows)


def quartile_summary(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame["joint_fit_quality_quartile"] = pd.qcut(
        frame["joint_parametric_surface_r2"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    return (
        frame.groupby("joint_fit_quality_quartile", observed=True)
        .agg(
            n_units=("rr100_index", "size"),
            joint_fit_r2_median=("joint_parametric_surface_r2", "median"),
            delta_cv_r2_median=("delta_cv_r2_unit_minus_total", "median"),
            delta_rmse_hz_median=("delta_rmse_hz_total_minus_unit", "median"),
            delta_spearman_median=("delta_spearman_unit_minus_total", "median"),
        )
        .reset_index()
    )


def bootstrap_spearman(table: pd.DataFrame, outcome: str, rng: np.random.Generator) -> dict[str, float]:
    x = table["joint_parametric_surface_r2"].to_numpy(float)
    y = table[outcome].to_numpy(float)
    observed = float(stats.spearmanr(x, y).statistic)
    values = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(table), len(table))
        values[b] = stats.spearmanr(x[idx], y[idx]).statistic
    return {
        "spearman_rho": observed,
        "bootstrap_ci95_low": float(np.nanquantile(values, 0.025)),
        "bootstrap_ci95_high": float(np.nanquantile(values, 0.975)),
        "two_sided_spearman_p": float(stats.spearmanr(x, y).pvalue),
    }


def standardized_design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(float)
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)


def joint_fit_coefficient(
    table: pd.DataFrame, outcome: str, adjusted: bool, rng: np.random.Generator
) -> dict[str, float]:
    covariates = [
        "joint_parametric_surface_r2",
        "log10_response_sd",
        "log2_preferred_sf",
        "log2_preferred_tf",
        "sf_fwhm_octaves",
        "tf_fwhm_octaves",
    ]
    columns = covariates if adjusted else ["joint_parametric_surface_r2"]
    frame = table.dropna(subset=columns + [outcome]).copy()
    x = standardized_design(frame, columns)
    y_raw = frame[outcome].to_numpy(float)
    y = (y_raw - y_raw.mean()) / y_raw.std(ddof=0)
    design = np.column_stack([np.ones(len(frame)), x])
    coefficient = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
    boot = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(frame), len(frame))
        xb = x[idx]
        yb = y[idx]
        fit = np.linalg.lstsq(np.column_stack([np.ones(len(idx)), xb]), yb, rcond=None)[0]
        boot[b] = fit[1]
    return {
        "adjusted": adjusted,
        "n_units": int(len(frame)),
        "standardized_joint_fit_coefficient": coefficient,
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_fraction_positive": float(np.mean(boot > 0)),
        "design_condition_number": float(np.linalg.cond(design)),
        "covariates": "+".join(columns),
    }


def association_tables(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = {
        "delta_cv_r2_unit_minus_total": "Δ held-out R²",
        "delta_rmse_hz_total_minus_unit": "Δ RMSE (positive favors tuning)",
        "delta_mae_hz_total_minus_unit": "Δ MAE (positive favors tuning)",
        "delta_spearman_unit_minus_total": "Δ Spearman ordering",
        "delta_pearson_unit_minus_total": "Δ Pearson ordering",
    }
    rng = np.random.default_rng(RNG_SEED + 1)
    associations, coefficients = [], []
    for outcome, label in outcomes.items():
        association = bootstrap_spearman(table, outcome, rng)
        associations.append({"outcome": outcome, "outcome_label": label, **association})
        for adjusted in [False, True]:
            coefficient = joint_fit_coefficient(table, outcome, adjusted, rng)
            coefficients.append({"outcome": outcome, "outcome_label": label, **coefficient})
    return pd.DataFrame(associations), pd.DataFrame(coefficients)


def make_figure(
    table: pd.DataFrame,
    thresholds: pd.DataFrame,
    quartiles: pd.DataFrame,
    associations: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11.5})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.3), constrained_layout=True)

    ax = axes[0, 0]
    ax.errorbar(
        thresholds["joint_fit_r2_minimum"],
        thresholds["median_delta_cv_r2"],
        yerr=np.vstack(
            [
                thresholds["median_delta_cv_r2"] - thresholds["median_delta_cv_r2_ci95_low"],
                thresholds["median_delta_cv_r2_ci95_high"] - thresholds["median_delta_cv_r2"],
            ]
        ),
        marker="o",
        color="#356fa3",
        capsize=3,
    )
    ax.axhline(0, color="#777", linestyle="--", linewidth=1)
    for row in thresholds.itertuples(index=False):
        ax.text(
            row.joint_fit_r2_minimum,
            row.median_delta_cv_r2 + 0.012,
            f"n={row.n_units}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set(
        xlabel="Minimum joint grating-fit R² retained",
        ylabel="Median Δ held-out R²",
        title="A  Stricter fit thresholds do not rescue tuning",
    )

    ax = axes[0, 1]
    x = np.arange(len(quartiles))
    ax.bar(x - 0.18, quartiles["delta_cv_r2_median"], width=0.36, color="#356fa3", label="Δ held-out R²")
    scale = 100.0
    ax.bar(
        x + 0.18,
        quartiles["delta_rmse_hz_median"] * scale,
        width=0.36,
        color="#d65238",
        label="100× ΔRMSE (Hz)",
    )
    ax.axhline(0, color="#777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"{q}\nfit median {r:.2f}" for q, r in zip(quartiles.joint_fit_quality_quartile, quartiles.joint_fit_r2_median)])
    ax.set_ylabel("Positive values favor unit SF×TF")
    ax.set_title("B  The best-fit quartile is not selectively improved")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    order = np.arange(len(associations))
    ax.errorbar(
        associations["spearman_rho"],
        order,
        xerr=np.vstack(
            [
                associations["spearman_rho"] - associations["bootstrap_ci95_low"],
                associations["bootstrap_ci95_high"] - associations["spearman_rho"],
            ]
        ),
        fmt="o",
        color="#5b6f7f",
        capsize=3,
    )
    ax.axvline(0, color="#777", linestyle="--", linewidth=1)
    ax.set_yticks(order, associations["outcome_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with joint-fit R²")
    ax.set_title("C  No improvement metric increases with fit quality")

    ax = axes[1, 1]
    unadjusted = coefficients.loc[~coefficients["adjusted"]].reset_index(drop=True)
    adjusted = coefficients.loc[coefficients["adjusted"]].reset_index(drop=True)
    for data, offset, color, label in [
        (unadjusted, -0.10, "#8ca9bf", "unadjusted"),
        (adjusted, 0.10, "#8e6bb3", "adjusted"),
    ]:
        ax.errorbar(
            data["standardized_joint_fit_coefficient"],
            order + offset,
            xerr=np.vstack(
                [
                    data["standardized_joint_fit_coefficient"] - data["bootstrap_ci95_low"],
                    data["bootstrap_ci95_high"] - data["standardized_joint_fit_coefficient"],
                ]
            ),
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
    ax.axvline(0, color="#777", linestyle="--", linewidth=1)
    ax.set_yticks(order, associations["outcome_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient for joint-fit R²")
    ax.set_title("D  Covariate adjustment does not reveal a positive effect")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(color="#e9ecef", linewidth=0.75)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Checkpoint 15C: population tests do not support grating-fit error as the main suppressor",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(OUT / "checkpoint_15c_fit_quality_population_tests.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "checkpoint_15c_fit_quality_population_tests.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_per_unit()
    thresholds = threshold_summary(table)
    quartiles = quartile_summary(table)
    associations, coefficients = association_tables(table)
    table.to_csv(OUT / "per_unit_fit_quality_and_stable_improvement_metrics.csv", index=False)
    thresholds.to_csv(OUT / "fit_quality_threshold_summary.csv", index=False)
    quartiles.to_csv(OUT / "fit_quality_quartile_summary.csv", index=False)
    associations.to_csv(OUT / "fit_quality_metric_associations.csv", index=False)
    coefficients.to_csv(OUT / "fit_quality_covariate_models.csv", index=False)
    make_figure(table, thresholds, quartiles, associations, coefficients)

    primary = associations.loc[associations["outcome"].eq("delta_cv_r2_unit_minus_total")].iloc[0]
    adjusted_primary = coefficients.loc[
        coefficients["outcome"].eq("delta_cv_r2_unit_minus_total") & coefficients["adjusted"]
    ].iloc[0]
    best_quartile = quartiles.loc[quartiles["joint_fit_quality_quartile"].eq("Q4")].iloc[0]
    manifest = {
        "analysis": "rr100_fit_quality_population_checkpoint_15c",
        "status": "population_fit_quality_diagnostic_complete",
        "n_units": int(len(table)),
        "n_bootstrap": N_BOOT,
        "primary_spearman_rho": float(primary.spearman_rho),
        "primary_spearman_p": float(primary.two_sided_spearman_p),
        "primary_spearman_bootstrap_ci95": [
            float(primary.bootstrap_ci95_low),
            float(primary.bootstrap_ci95_high),
        ],
        "adjusted_standardized_joint_fit_coefficient": float(
            adjusted_primary.standardized_joint_fit_coefficient
        ),
        "adjusted_coefficient_bootstrap_ci95": [
            float(adjusted_primary.bootstrap_ci95_low),
            float(adjusted_primary.bootstrap_ci95_high),
        ],
        "best_fit_quartile_median_delta_cv_r2": float(best_quartile.delta_cv_r2_median),
        "best_fit_quartile_median_delta_rmse_hz": float(best_quartile.delta_rmse_hz_median),
        "interpretation": "No tested improvement metric increases with joint grating-fit quality; stricter gates and covariate adjustment do not rescue the unit-specific predictor.",
        "next_checkpoint": "direct empirical and rank1 grating-surface predictors",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Checkpoint 15C: population fit-quality diagnostic\n\n"
        "This checkpoint follows the raw and selected-unit map checks. It evaluates cumulative "
        "fit-quality thresholds, equal-count quality quartiles, stable error and ordering metrics, "
        "and standardized regression with response magnitude, preferred SF/TF, and tuning "
        "bandwidth covariates. Positive differences always favor unit-specific SF×TF overlap.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

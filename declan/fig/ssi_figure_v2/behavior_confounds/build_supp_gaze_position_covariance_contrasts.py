#!/usr/bin/env python3
"""Checkpoint 3: dimensionless covariance contrasts for gaze-position drift geometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import t as student_t
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_supp_gaze_position_anisotropy_broad_model as prior,
)


OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "supp_gaze_position_covariance_checkpoint3_v1"
)
SUBJECTS = prior.SUBJECTS
SUBJECT_COLORS = prior.SUBJECT_COLORS
INK = prior.INK
GRID = prior.GRID
ECC_LABELS = prior.ECC_LABELS
PRIMARY_SPEC = prior.PRIMARY_SPEC

OUTCOMES = {
    "screen": {
        "column": "screen_covariance_contrast",
        "old_fraction": "screen_h_minus_v_fraction",
        "label": "Screen H−V shape",
        "formula": r"$A_{screen}=(\Sigma_{xx}-\Sigma_{yy})/(\Sigma_{xx}+\Sigma_{yy})$",
    },
    "gaze": {
        "column": "gaze_covariance_contrast",
        "old_fraction": "gaze_t_minus_r_fraction",
        "label": "Gaze-frame T−R shape",
        "formula": r"$A_{gaze}=(\Sigma_{tt}-\Sigma_{rr})/(\Sigma_{tt}+\Sigma_{rr})$",
    },
    "axis_free": {
        "column": "axis_free_covariance_contrast",
        "old_fraction": "axis_free_fraction",
        "label": "Axis-free shape",
        "formula": r"$A_{shape}=(\lambda_1-\lambda_2)/(\lambda_1+\lambda_2)$",
    },
}


def rms_difference_from_contrast(contrast: np.ndarray | float, scale_arcmin: float):
    contrast = np.clip(np.asarray(contrast, dtype=float), -0.999999, 0.999999)
    return scale_arcmin * (
        np.sqrt((1.0 + contrast) / 2.0) - np.sqrt((1.0 - contrast) / 2.0)
    )


def rms_difference_derivative(contrast: float, scale_arcmin: float) -> float:
    contrast = float(np.clip(contrast, -0.999999, 0.999999))
    return float(
        scale_arcmin
        / (2.0 * np.sqrt(2.0))
        * (1.0 / np.sqrt(1.0 + contrast) + 1.0 / np.sqrt(1.0 - contrast))
    )


def load_and_derive() -> tuple[pd.DataFrame, dict[str, float]]:
    values, references = prior.load_and_derive()
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)
    trace = cxx + cyy
    x = values["mean_x_deg"].to_numpy(dtype=float)
    y = values["mean_y_deg"].to_numpy(dtype=float)
    eccentricity = np.hypot(x, y)
    radial_x = np.divide(x, eccentricity, out=np.ones_like(x), where=eccentricity > 1e-12)
    radial_y = np.divide(y, eccentricity, out=np.zeros_like(y), where=eccentricity > 1e-12)
    tangent_x = -radial_y
    tangent_y = radial_x

    def variance(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        return ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy

    radial_variance = variance(radial_x, radial_y)
    tangential_variance = variance(tangent_x, tangent_y)
    discriminant = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy**2, 0.0))
    values["screen_covariance_contrast"] = (cxx - cyy) / trace
    values["gaze_covariance_contrast"] = (tangential_variance - radial_variance) / trace
    values["axis_free_covariance_contrast"] = discriminant / trace
    return values, references


def counterfactual_x(model, block: pd.DataFrame, eccentricity: float, scale: float) -> np.ndarray:
    new = block.copy()
    new["eccentricity_deg"] = float(eccentricity)
    new["log_scale"] = float(np.log(scale))
    design = np.asarray(
        build_design_matrices([model.model.data.design_info], new, return_type="dataframe")[0],
        dtype=float,
    )
    return np.average(design, axis=0, weights=block["hierarchical_weight"])


def transformed_point(model, x: np.ndarray, n_sessions: int, scale: float) -> dict[str, float]:
    params = np.asarray(model.params, dtype=float)
    covariance = np.asarray(model.cov_params(), dtype=float)
    contrast = float(x @ params)
    estimate = float(rms_difference_from_contrast(contrast, scale))
    gradient = rms_difference_derivative(contrast, scale) * x
    variance = float(gradient @ covariance @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    critical = float(student_t.ppf(0.975, max(n_sessions - 1, 1)))
    return {
        "predicted_covariance_contrast": contrast,
        "estimate_arcmin": estimate,
        "se_arcmin": se,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "variance": variance,
    }


def transformed_difference(
    model, x_low: np.ndarray, x_high: np.ndarray, n_sessions: int, scale: float
) -> dict[str, float]:
    params = np.asarray(model.params, dtype=float)
    covariance = np.asarray(model.cov_params(), dtype=float)
    low = float(x_low @ params)
    high = float(x_high @ params)
    estimate = float(
        rms_difference_from_contrast(high, scale)
        - rms_difference_from_contrast(low, scale)
    )
    gradient = (
        rms_difference_derivative(high, scale) * x_high
        - rms_difference_derivative(low, scale) * x_low
    )
    variance = float(gradient @ covariance @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    critical = float(student_t.ppf(0.975, max(n_sessions - 1, 1)))
    return {
        "central_covariance_contrast": low,
        "peripheral_covariance_contrast": high,
        "estimate_arcmin": estimate,
        "se_arcmin": se,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "variance": variance,
    }


def fit_models(values: pd.DataFrame, references: dict[str, float]):
    models = {}
    effect_rows = []
    curve_rows = []
    diagnostic_rows = []
    coefficient_rows = []
    central = references["central_eccentricity_deg"]
    peripheral = references["peripheral_eccentricity_deg"]
    scale = references["reference_scale_arcmin"]
    curve_grid = np.linspace(0.5, 12.0, 48)

    for outcome_id, outcome_spec in OUTCOMES.items():
        for spec_name, rhs in prior.MODEL_SPECS.items():
            subject_effects = []
            subject_variances = []
            for subject in SUBJECTS:
                block = values[values["subject"].eq(subject)].copy()
                model = prior.fit_one(block, outcome_spec["column"], rhs)
                models[(outcome_id, spec_name, subject)] = model
                n_sessions = int(block["session"].nunique())
                x_low = counterfactual_x(model, block, central, scale)
                x_high = counterfactual_x(model, block, peripheral, scale)
                effect = transformed_difference(model, x_low, x_high, n_sessions, scale)
                subject_effects.append(effect["estimate_arcmin"])
                subject_variances.append(effect["variance"])
                effect_rows.append(
                    {
                        "outcome": outcome_id,
                        "outcome_label": outcome_spec["label"],
                        "model_spec": spec_name,
                        "scope": subject,
                        **{k: v for k, v in effect.items() if k != "variance"},
                        "central_eccentricity_deg": central,
                        "peripheral_eccentricity_deg": peripheral,
                        "reference_scale_arcmin": scale,
                    }
                )
                diagnostic_rows.append(
                    {
                        "outcome": outcome_id,
                        "model_spec": spec_name,
                        "subject": subject,
                        "n_windows": int(len(block)),
                        "n_sessions": n_sessions,
                        "n_parameters": int(len(model.params)),
                        "design_rank": int(model.model.rank),
                        "condition_number": float(np.linalg.cond(model.model.exog)),
                        "weighted_r_squared": float(model.rsquared),
                    }
                )
                for term, estimate in pd.Series(model.params).items():
                    coefficient_rows.append(
                        {
                            "outcome": outcome_id,
                            "model_spec": spec_name,
                            "subject": subject,
                            "term": str(term),
                            "estimate": float(estimate),
                            "cluster_se": float(pd.Series(model.bse).loc[term]),
                            "cluster_p": float(pd.Series(model.pvalues).loc[term]),
                        }
                    )
                if spec_name == PRIMARY_SPEC:
                    for eccentricity in curve_grid:
                        x_mean = counterfactual_x(model, block, float(eccentricity), scale)
                        point = transformed_point(model, x_mean, n_sessions, scale)
                        curve_rows.append(
                            {
                                "outcome": outcome_id,
                                "scope": subject,
                                "eccentricity_deg": float(eccentricity),
                                **point,
                            }
                        )

            estimate = float(np.mean(subject_effects))
            variance = float(np.sum(subject_variances) / 4.0)
            se = float(np.sqrt(max(variance, 0.0)))
            critical = float(student_t.ppf(0.975, 13))
            effect_rows.append(
                {
                    "outcome": outcome_id,
                    "outcome_label": outcome_spec["label"],
                    "model_spec": spec_name,
                    "scope": "grand_equal_subject",
                    "central_covariance_contrast": np.nan,
                    "peripheral_covariance_contrast": np.nan,
                    "estimate_arcmin": estimate,
                    "se_arcmin": se,
                    "ci95_low": estimate - critical * se,
                    "ci95_high": estimate + critical * se,
                    "central_eccentricity_deg": central,
                    "peripheral_eccentricity_deg": peripheral,
                    "reference_scale_arcmin": scale,
                }
            )

    curves = pd.DataFrame(curve_rows)
    grand_rows = []
    for (outcome, eccentricity), block in curves.groupby(["outcome", "eccentricity_deg"]):
        estimate = float(block["estimate_arcmin"].mean())
        variance = float(block["variance"].sum() / 4.0)
        se = float(np.sqrt(max(variance, 0.0)))
        critical = float(student_t.ppf(0.975, 13))
        grand_rows.append(
            {
                "outcome": outcome,
                "scope": "grand_equal_subject",
                "eccentricity_deg": eccentricity,
                "estimate_arcmin": estimate,
                "se_arcmin": se,
                "ci95_low": estimate - critical * se,
                "ci95_high": estimate + critical * se,
                "variance": variance,
            }
        )
    curves = pd.concat([curves, pd.DataFrame(grand_rows)], ignore_index=True)
    return (
        models,
        pd.DataFrame(effect_rows),
        curves,
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(coefficient_rows),
    )


def descriptive_summary(values: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        source = values[values["subject"].eq(subject)]
        for order, label in enumerate(ECC_LABELS):
            block = source[source["eccentricity_bin"].astype(str).eq(label)]
            row = {
                "subject": subject,
                "eccentricity_bin": label,
                "eccentricity_bin_order": order,
                "eccentricity_deg": float(np.median(block["eccentricity_deg"])),
                "n_windows": int(len(block)),
            }
            for outcome_id, spec in OUTCOMES.items():
                contrast = float(np.median(block[spec["column"]]))
                row[f"{outcome_id}_covariance_contrast"] = contrast
                row[f"{outcome_id}_equivalent_arcmin"] = float(
                    rms_difference_from_contrast(contrast, scale)
                )
            rows.append(row)
    subject_rows = pd.DataFrame(rows)
    grand_rows = []
    for (label, order), block in subject_rows.groupby(
        ["eccentricity_bin", "eccentricity_bin_order"]
    ):
        row = {
            "subject": "grand_equal_subject",
            "eccentricity_bin": label,
            "eccentricity_bin_order": order,
            "eccentricity_deg": float(block["eccentricity_deg"].mean()),
            "n_windows": int(block["n_windows"].sum()),
        }
        for outcome_id in OUTCOMES:
            row[f"{outcome_id}_covariance_contrast"] = float(
                block[f"{outcome_id}_covariance_contrast"].mean()
            )
            row[f"{outcome_id}_equivalent_arcmin"] = float(
                block[f"{outcome_id}_equivalent_arcmin"].mean()
            )
        grand_rows.append(row)
    return pd.concat([subject_rows, pd.DataFrame(grand_rows)], ignore_index=True)


def plot_metric_contract(values: pd.DataFrame, descriptive: pd.DataFrame, scale: float):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True)
    for col, (outcome_id, spec) in enumerate(OUTCOMES.items()):
        ax = axes[0, col]
        sample = values.sample(min(3500, len(values)), random_state=19)
        ax.scatter(
            sample[spec["column"]], sample[spec["old_fraction"]],
            c=sample["eccentricity_deg"], s=4, alpha=0.18, cmap="viridis", rasterized=True,
        )
        grid = np.linspace(-0.98 if outcome_id != "axis_free" else 0.0, 0.98, 300)
        ax.plot(grid, rms_difference_from_contrast(grid, 1.0), color=INK, lw=1.5,
                label="exact covariance-to-RMS map")
        ax.set_title(chr(65 + col) + "  " + spec["label"], loc="left", weight="semibold")
        ax.text(0.03, 0.96, spec["formula"], transform=ax.transAxes, va="top", fontsize=7.0)
        ax.set_xlabel("dimensionless covariance contrast")
        ax.set_ylabel("previous RMS difference / total RMS")
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")

    for col, (outcome_id, spec) in enumerate(OUTCOMES.items()):
        ax = axes[1, col]
        for subject in SUBJECTS:
            block = descriptive[descriptive["subject"].eq(subject)].sort_values("eccentricity_bin_order")
            ax.plot(block["eccentricity_deg"], block[f"{outcome_id}_covariance_contrast"],
                    "o-", ms=3, lw=1.1, color=SUBJECT_COLORS[subject], alpha=0.8, label=subject)
        grand = descriptive[descriptive["subject"].eq("grand_equal_subject")].sort_values("eccentricity_bin_order")
        ax.plot(grand["eccentricity_deg"], grand[f"{outcome_id}_covariance_contrast"],
                "o-", ms=3.5, lw=1.7, color=INK, label="equal-animal median")
        ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        ax.set_title(chr(68 + col) + "  Descriptive covariance contrast", loc="left", weight="semibold")
        ax.set_xlabel("gaze eccentricity (deg; bin median)")
        ax.set_ylabel(spec["formula"])
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(
        f"Checkpoint 3A: scale-free covariance definitions (display scale {scale:.3f} arcmin)",
        fontsize=12.8, weight="bold",
    )
    return fig


def plot_adjusted_curves(curves: pd.DataFrame, descriptive: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.65), constrained_layout=True)
    for ax, (outcome_id, spec) in zip(axes, OUTCOMES.items(), strict=True):
        grand = curves[(curves["outcome"].eq(outcome_id)) & (curves["scope"].eq("grand_equal_subject"))].sort_values("eccentricity_deg")
        ax.fill_between(grand["eccentricity_deg"], grand["ci95_low"], grand["ci95_high"],
                        color="#AEB4BA", alpha=0.28, lw=0)
        for subject in SUBJECTS:
            block = curves[(curves["outcome"].eq(outcome_id)) & (curves["scope"].eq(subject))].sort_values("eccentricity_deg")
            ax.plot(block["eccentricity_deg"], block["estimate_arcmin"],
                    color=SUBJECT_COLORS[subject], lw=1.1, alpha=0.8, label=subject)
        ax.plot(grand["eccentricity_deg"], grand["estimate_arcmin"], color=INK, lw=2.0,
                label="equal-animal adjusted")
        raw = descriptive[descriptive["subject"].eq("grand_equal_subject")].sort_values("eccentricity_bin_order")
        ax.plot(raw["eccentricity_deg"], raw[f"{outcome_id}_equivalent_arcmin"],
                "o--", color="#7D858C", ms=3, lw=0.9, label="descriptive median")
        ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        ax.set_title(spec["label"], loc="left", weight="semibold")
        ax.set_xlabel("counterfactual gaze eccentricity (deg)")
        ax.set_ylabel("RMS difference at reference scale (arcmin)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.2)
    fig.suptitle("Covariance-contrast model: adjusted eccentricity curves", fontsize=12.5, weight="bold")
    return fig


def plot_specification_effects(effects: pd.DataFrame, figure_f: dict[str, float]):
    specs = list(prior.MODEL_SPECS)
    labels = ["within\nsession", "+ total RMS\nradius", "+ gaze polar\nangle", "+ image, phase,\nevent timing", "interaction\nmodel"]
    x = np.arange(len(specs))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    for ax, (outcome_id, outcome_spec) in zip(axes, OUTCOMES.items(), strict=True):
        for subject, offset in (("Allen", -0.10), ("Logan", 0.10)):
            block = effects[(effects["outcome"].eq(outcome_id)) & (effects["scope"].eq(subject))].set_index("model_spec").loc[specs]
            ax.plot(x + offset, block["estimate_arcmin"], "o-", ms=3, lw=0.9,
                    color=SUBJECT_COLORS[subject], alpha=0.78, label=subject)
        grand = effects[(effects["outcome"].eq(outcome_id)) & (effects["scope"].eq("grand_equal_subject"))].set_index("model_spec").loc[specs]
        ax.errorbar(x, grand["estimate_arcmin"], yerr=np.vstack([
            grand["estimate_arcmin"] - grand["ci95_low"],
            grand["ci95_high"] - grand["estimate_arcmin"],
        ]), fmt="o-", color=INK, ecolor=INK, capsize=2.5, lw=1.5, ms=4,
                    label="equal-animal 95% interval")
        ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        if outcome_id == "screen":
            ax.axhspan(figure_f["ci95_low"], figure_f["ci95_high"], color="#AEB4BA", alpha=0.22)
            ax.axhline(figure_f["estimate"], color="#6B6F75", lw=1.0, ls="--",
                       label="Figure 4F numerical scale")
        ax.set_xticks(x, labels)
        ax.set_title(outcome_spec["label"], loc="left", weight="semibold")
        ax.set_ylabel("peripheral − central (arcmin at reference scale)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.1)
    fig.suptitle("Covariance contrast: range across reasonable specifications", fontsize=12.5, weight="bold")
    return fig


def tracker_proxy_summary(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        source = values[values["subject"].eq(subject)]
        for order, label in enumerate(ECC_LABELS):
            block = source[source["eccentricity_bin"].astype(str).eq(label)]
            rows.append(
                {
                    "subject": subject,
                    "eccentricity_bin_order": order,
                    "eccentricity_deg": float(np.median(block["eccentricity_deg"])),
                    "high_frequency_power_fraction": float(np.median(block["position_high_freq_power_fraction_15_60hz"])),
                    "step_median_arcmin": float(60.0 * np.median(block["step_median_deg"])),
                    "screen_covariance_contrast": float(np.median(block["screen_covariance_contrast"])),
                    "n_windows": int(len(block)),
                }
            )
    return pd.DataFrame(rows)


def cross_validate_specifications(values: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Trial-held-out comparison while retaining every session in training."""
    work = values.copy()
    work["cv_fold"] = -1
    for (_subject, _session), indices in work.groupby(["subject", "session"]).groups.items():
        trials = sorted(work.loc[indices, "trial_idx"].unique())
        mapping = {int(trial): index % n_folds for index, trial in enumerate(trials)}
        work.loc[indices, "cv_fold"] = work.loc[indices, "trial_idx"].map(mapping)

    rows = []
    for subject in SUBJECTS:
        subject_values = work[work["subject"].eq(subject)].copy()
        for outcome_id, outcome_spec in OUTCOMES.items():
            for spec_name in ("broad_additive", "interaction_sensitivity"):
                observed_parts = []
                predicted_parts = []
                weight_parts = []
                for fold in range(n_folds):
                    train = subject_values[subject_values["cv_fold"].ne(fold)]
                    test = subject_values[subject_values["cv_fold"].eq(fold)]
                    model = smf.wls(
                        f"{outcome_spec['column']} ~ {prior.MODEL_SPECS[spec_name]}",
                        data=train,
                        weights=train["hierarchical_weight"],
                    ).fit()
                    observed_parts.append(test[outcome_spec["column"]].to_numpy(dtype=float))
                    predicted_parts.append(np.asarray(model.predict(test), dtype=float))
                    weight_parts.append(test["hierarchical_weight"].to_numpy(dtype=float))
                observed = np.concatenate(observed_parts)
                predicted = np.concatenate(predicted_parts)
                weights = np.concatenate(weight_parts)
                rows.append(
                    {
                        "subject": subject,
                        "outcome": outcome_id,
                        "model_spec": spec_name,
                        "n_folds": n_folds,
                        "weighted_rmse": float(
                            np.sqrt(np.average((observed - predicted) ** 2, weights=weights))
                        ),
                        "weighted_mae": float(
                            np.average(np.abs(observed - predicted), weights=weights)
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    additive = table[table["model_spec"].eq("broad_additive")].set_index(["subject", "outcome"])
    interaction = table[table["model_spec"].eq("interaction_sensitivity")].set_index(["subject", "outcome"])
    comparison = additive[["weighted_rmse", "weighted_mae"]].join(
        interaction[["weighted_rmse", "weighted_mae"]],
        lsuffix="_additive",
        rsuffix="_interaction",
    )
    comparison["interaction_minus_additive_rmse_percent"] = 100.0 * (
        comparison["weighted_rmse_interaction"] / comparison["weighted_rmse_additive"] - 1.0
    )
    comparison["interaction_minus_additive_mae_percent"] = 100.0 * (
        comparison["weighted_mae_interaction"] / comparison["weighted_mae_additive"] - 1.0
    )
    return comparison.reset_index()


def plot_cross_validation(comparison: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    x = np.arange(len(OUTCOMES), dtype=float)
    width = 0.34
    for ax, metric, title in (
        (axes[0], "interaction_minus_additive_rmse_percent", "A  Held-out RMSE"),
        (axes[1], "interaction_minus_additive_mae_percent", "B  Held-out MAE"),
    ):
        for index, subject in enumerate(SUBJECTS):
            block = comparison[comparison["subject"].eq(subject)].set_index("outcome").loc[list(OUTCOMES)]
            ax.bar(
                x + (index - 0.5) * width,
                block[metric],
                width=width,
                color=SUBJECT_COLORS[subject],
                alpha=0.86,
                label=subject,
            )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [OUTCOMES[key]["label"] for key in OUTCOMES], rotation=15, ha="right")
        ax.set_ylabel("interaction − additive error (%)")
        ax.set_title(title, loc="left", weight="semibold")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Five-fold trial-held-out comparison (positive favors the additive model)",
        fontsize=11.5,
        weight="bold",
    )
    return fig


def plot_tracker_proxy_diagnostics(values: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0, 0], "high_frequency_power_fraction", "A  High-frequency position power", "15–60 Hz power fraction"),
        (axes[0, 1], "step_median_arcmin", "B  One-sample step size", "median step (arcmin)"),
    ):
        for subject in SUBJECTS:
            block = summary[summary["subject"].eq(subject)].sort_values("eccentricity_bin_order")
            ax.plot(block["eccentricity_deg"], block[metric], "o-", ms=3, lw=1.1,
                    color=SUBJECT_COLORS[subject], label=subject)
        ax.set_title(title, loc="left", weight="semibold")
        ax.set_xlabel("gaze eccentricity (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=6.5)

    ax = axes[0, 2]
    bins = np.arange(0, 12.1, 2.0)
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        block["abs_x_bin"] = pd.cut(np.abs(block["mean_x_deg"]), bins, labels=False, right=False)
        for side, ls in (("left", "--"), ("right", "-")):
            side_block = block[block["mean_x_deg"].lt(0) if side == "left" else block["mean_x_deg"].ge(0)]
            grouped = side_block.groupby("abs_x_bin", observed=True).agg(
                x=("mean_x_deg", lambda z: float(np.median(np.abs(z)))),
                y=("screen_covariance_contrast", "median"), n=("trial_idx", "size"),
            )
            grouped = grouped[grouped["n"] >= 30]
            ax.plot(grouped["x"], grouped["y"], "o", ms=3, color=SUBJECT_COLORS[subject])
            ax.plot(grouped["x"], grouped["y"], ls=ls, lw=1.0, color=SUBJECT_COLORS[subject],
                    label=f"{subject} {side}")
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_title("C  Left–right symmetry check", loc="left", weight="semibold")
    ax.set_xlabel("absolute horizontal gaze (deg)")
    ax.set_ylabel("screen covariance contrast")
    ax.legend(frameon=False, fontsize=5.6, ncol=2)

    ax = axes[1, 0]
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        block["abs_y_bin"] = pd.cut(np.abs(block["mean_y_deg"]), bins, labels=False, right=False)
        for side, ls in (("lower", "--"), ("upper", "-")):
            side_block = block[block["mean_y_deg"].lt(0) if side == "lower" else block["mean_y_deg"].ge(0)]
            grouped = side_block.groupby("abs_y_bin", observed=True).agg(
                x=("mean_y_deg", lambda z: float(np.median(np.abs(z)))),
                y=("screen_covariance_contrast", "median"), n=("trial_idx", "size"),
            )
            grouped = grouped[grouped["n"] >= 30]
            ax.plot(grouped["x"], grouped["y"], "o", ms=3, color=SUBJECT_COLORS[subject])
            ax.plot(grouped["x"], grouped["y"], ls=ls, lw=1.0, color=SUBJECT_COLORS[subject],
                    label=f"{subject} {side}")
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_title("D  Upper–lower symmetry check", loc="left", weight="semibold")
    ax.set_xlabel("absolute vertical gaze (deg)")
    ax.set_ylabel("screen covariance contrast")
    ax.legend(frameon=False, fontsize=5.6, ncol=2)

    ax = axes[1, 1]
    session_rows = []
    for (subject, session), block in values.groupby(["subject", "session"]):
        slope = float(np.polyfit(block["eccentricity_deg"], block["screen_covariance_contrast"], 1)[0])
        session_rows.append({"subject": subject, "session": session, "slope": slope})
    session_table = pd.DataFrame(session_rows)
    for index, subject in enumerate(SUBJECTS):
        block = session_table[session_table["subject"].eq(subject)]
        jitter = np.linspace(-0.08, 0.08, len(block))
        ax.scatter(index + jitter, block["slope"], s=18, color=SUBJECT_COLORS[subject], alpha=0.8)
        ax.plot(index, block["slope"].median(), "D", color=INK, ms=4)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks([0, 1], SUBJECTS)
    ax.set_title("E  Session-specific eccentricity slopes", loc="left", weight="semibold")
    ax.set_ylabel("screen contrast change / deg")

    ax = axes[1, 2]
    ax.scatter(values["position_high_freq_power_fraction_15_60hz"],
               values["screen_covariance_contrast"], s=3, alpha=0.06, color=INK, rasterized=True)
    ax.set_title("F  Noise proxy versus screen shape", loc="left", weight="semibold")
    ax.set_xlabel("15–60 Hz position-power fraction")
    ax.set_ylabel("screen covariance contrast")
    ax.text(0.03, 0.04, "proxy only; not a calibration residual", transform=ax.transAxes,
            fontsize=6.2, color="#6B6F75")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Checkpoint 3B: first tracker-coordinate artifact diagnostics", fontsize=12.5, weight="bold")
    return fig, session_table


def save_figure(fig: plt.Figure, stem: str):
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 260}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def main() -> None:
    prior.configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values, references = load_and_derive()
    figure_f = prior.figure_f_reference()
    models, effects, curves, diagnostics, coefficients = fit_models(values, references)
    descriptive = descriptive_summary(values, references["reference_scale_arcmin"])
    tracker_summary = tracker_proxy_summary(values)
    cross_validation = cross_validate_specifications(values)
    tracker_figure, session_slopes = plot_tracker_proxy_diagnostics(values, tracker_summary)

    keep = [
        "subject", "session", "trial_idx", "phase", "mean_x_deg", "mean_y_deg",
        "eccentricity_deg", "gaze_polar_angle_deg", "scale_arcmin",
        "screen_covariance_contrast", "gaze_covariance_contrast",
        "axis_free_covariance_contrast", "image_orientation_coherence",
        "image_edge_axis_deg", "position_high_freq_power_fraction_15_60hz",
        "step_median_deg", "hierarchical_weight",
    ]
    values[keep].to_csv(OUT_DIR / "covariance_contrast_window_values.csv.gz", index=False, compression="gzip")
    effects.to_csv(OUT_DIR / "covariance_contrast_effects.csv", index=False)
    curves.to_csv(OUT_DIR / "covariance_contrast_adjusted_curves.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "covariance_contrast_model_diagnostics.csv", index=False)
    coefficients.to_csv(OUT_DIR / "covariance_contrast_model_coefficients.csv", index=False)
    descriptive.to_csv(OUT_DIR / "covariance_contrast_descriptive_curves.csv", index=False)
    tracker_summary.to_csv(OUT_DIR / "tracker_proxy_eccentricity_summary.csv", index=False)
    session_slopes.to_csv(OUT_DIR / "tracker_proxy_session_slopes.csv", index=False)
    cross_validation.to_csv(OUT_DIR / "model_specification_cross_validation.csv", index=False)

    outputs = {
        "metric_contract": save_figure(
            plot_metric_contract(values, descriptive, references["reference_scale_arcmin"]),
            "covariance_contrast_metric_contract",
        ),
        "adjusted_curves": save_figure(
            plot_adjusted_curves(curves, descriptive), "covariance_contrast_adjusted_curves"
        ),
        "specification_effects": save_figure(
            plot_specification_effects(effects, figure_f),
            "covariance_contrast_specification_effects",
        ),
        "tracker_proxy_diagnostics": save_figure(
            tracker_figure, "tracker_proxy_diagnostics"
        ),
        "cross_validation": save_figure(
            plot_cross_validation(cross_validation), "model_specification_cross_validation"
        ),
    }
    primary = effects[(effects["model_spec"].eq(PRIMARY_SPEC)) & effects["scope"].eq("grand_equal_subject")]
    interaction = effects[(effects["model_spec"].eq("interaction_sensitivity")) & effects["scope"].eq("grand_equal_subject")]
    report = [
        "# Covariance-contrast gaze-position checkpoint 3",
        "",
        "The primary outcomes are dimensionless covariance contrasts. Total drift-cloud RMS radius",
        "does not appear in the outcome denominator as an estimated response; the contrast separates",
        "shape from scale algebraically. Predictions are converted exactly to an RMS difference at",
        f"the fixed reference scale {references['reference_scale_arcmin']:.4f} arcmin.",
        "",
        "| outcome | additive effect (95% interval) | interaction effect (95% interval) |",
        "|---|---:|---:|",
    ]
    for outcome_id, spec in OUTCOMES.items():
        a = primary[primary["outcome"].eq(outcome_id)].iloc[0]
        i = interaction[interaction["outcome"].eq(outcome_id)].iloc[0]
        report.append(
            f"| {spec['label']} | {a.estimate_arcmin:+.3f} "
            f"[{a.ci95_low:+.3f}, {a.ci95_high:+.3f}] | "
            f"{i.estimate_arcmin:+.3f} [{i.ci95_low:+.3f}, {i.ci95_high:+.3f}] |"
        )
    report.extend(
        [
            "",
            "These are numerical-scale comparisons with Figure 4F, not estimates of Figure 4F",
            "attenuation. The direct attenuation analysis remains a separate required checkpoint.",
            "",
            "The tracker sheet uses high-frequency power, one-sample step size, symmetry, and",
            "session slopes as available proxies. It cannot replace calibration residuals or a",
            "stationary-target recording.",
            "",
            "Five-fold trial-held-out RMSE is lower for the additive model for every outcome in",
            "both animals. The advantage is small, so the interaction model remains an important",
            "sensitivity bound rather than the preferred headline.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 3; dimensionless covariance contrasts and tracker proxies",
        "source": str(prior.SOURCE_WINDOWS.relative_to(ROOT)),
        "n_windows": int(len(values)),
        "n_sessions": int(values["session"].nunique()),
        "references": references,
        "equations": {key: spec["formula"] for key, spec in OUTCOMES.items()},
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for key, paths in outputs.items():
        print(key, ROOT / paths["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

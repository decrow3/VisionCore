#!/usr/bin/env python3
"""Build the empirical RR100 SF x TF x direction tensor and fit diagnostics.

The empirical tensor is the primary routing object.  Smoothly weighted angular
profiles are derived summaries.  Conventional parametric models are fitted
only to the previously selected map-first example units and are used to test
whether the observed tensor is approximately separable and smoothly tuned.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from declan.fig4_active_sensing.analyze_rr100_cached_orientation_direction_tuning_checkpoint import (
    LEGACY,
    NATIVE,
    prepare_native,
    vector_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / (
    "outputs/fig4_active_sensing/rr100_cached_orientation_direction_tuning_checkpoint_v1/"
    "selected_unit_roles.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v2_smooth"
PRIMARY_WEIGHT_EXPONENT = 2.0
WEIGHT_EXPONENTS = (1.0, 2.0, 3.0)
N_FOLDS = 5


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def circular_delta(left: float, right: float, period: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    return float(abs((left - right + 0.5 * period) % period - 0.5 * period))


def build_tensor(points: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    units = np.sort(points["rr100_index"].unique().astype(int))
    sf = np.sort(points["spatial_cpd"].unique().astype(float))
    tf = np.sort(points["temporal_hz"].unique().astype(float))
    directions = np.sort(points["motion_direction_image_deg"].unique().astype(float))
    shape = (len(units), len(sf), len(tf), len(directions))
    signed = np.full(shape, np.nan, dtype=np.float32)
    unit_pos = {value: i for i, value in enumerate(units)}
    sf_pos = {value: i for i, value in enumerate(sf)}
    tf_pos = {value: i for i, value in enumerate(tf)}
    direction_pos = {value: i for i, value in enumerate(directions)}
    for row in points.itertuples(index=False):
        index = (
            unit_pos[int(row.rr100_index)],
            sf_pos[float(row.spatial_cpd)],
            tf_pos[float(row.temporal_hz)],
            direction_pos[float(row.motion_direction_image_deg)],
        )
        if np.isfinite(signed[index]):
            raise ValueError(f"Duplicate tensor cell {index}")
        signed[index] = float(row.signed_f0_hz)
    if not np.isfinite(signed).all():
        raise ValueError("Empirical joint tensor is incomplete")
    positive = np.maximum(signed, 0.0)
    sensitivity = positive.sum(axis=-1)
    peak = sensitivity.max(axis=(1, 2), keepdims=True)
    relative = np.divide(sensitivity, peak, out=np.zeros_like(sensitivity), where=peak > 1e-12)
    conditional_direction = np.divide(
        positive,
        sensitivity[..., None],
        out=np.zeros_like(positive),
        where=sensitivity[..., None] > 1e-12,
    )
    smooth_weights: dict[float, np.ndarray] = {}
    smooth_profiles: dict[float, np.ndarray] = {}
    for exponent in WEIGHT_EXPONENTS:
        unnormalized = sensitivity.astype(np.float64) ** float(exponent)
        denominator = unnormalized.sum(axis=(1, 2), keepdims=True)
        weights = np.divide(
            unnormalized,
            denominator,
            out=np.zeros_like(unnormalized),
            where=denominator > 1e-12,
        )
        smooth_weights[exponent] = weights
        smooth_profiles[exponent] = np.einsum("ust,ustd->ud", weights, conditional_direction)
    preferred_profile = smooth_profiles[PRIMARY_WEIGHT_EXPONENT]
    marginal_profile = positive.sum(axis=(1, 2))

    rows: list[dict[str, object]] = []
    for position, unit in enumerate(units):
        direction, dsi = vector_metrics(directions, preferred_profile[position], 1)
        normal_axis, osi = vector_metrics(directions, preferred_profile[position], 2)
        peak_index = np.unravel_index(int(np.argmax(sensitivity[position])), sensitivity[position].shape)
        rows.append(
            {
                "rr100_index": int(unit),
                "preferred_sf_cpd": float(sf[peak_index[0]]),
                "preferred_tf_hz": float(tf[peak_index[1]]),
                "primary_smooth_weight_exponent": PRIMARY_WEIGHT_EXPONENT,
                "smooth_weighted_motion_direction_image_deg": direction,
                "smooth_weighted_direction_vector_strength": dsi,
                "smooth_weighted_bar_orientation_image_deg": (normal_axis - 90.0) % 180.0,
                "smooth_weighted_orientation_vector_strength": osi,
                "maximum_direction_summed_positive_f0_hz": float(sensitivity[position].max()),
                "responsive_positive_f0": bool(peak[position, 0, 0] > 1e-12),
            }
        )
        for exponent in WEIGHT_EXPONENTS:
            exponent_label = f"alpha{int(exponent)}"
            exponent_direction, exponent_dsi = vector_metrics(
                directions, smooth_profiles[exponent][position], 1
            )
            exponent_normal, exponent_osi = vector_metrics(
                directions, smooth_profiles[exponent][position], 2
            )
            rows[-1].update(
                {
                    f"{exponent_label}_motion_direction_image_deg": exponent_direction,
                    f"{exponent_label}_direction_vector_strength": exponent_dsi,
                    f"{exponent_label}_bar_orientation_image_deg": (exponent_normal - 90.0) % 180.0,
                    f"{exponent_label}_orientation_vector_strength": exponent_osi,
                }
            )
        for left, right in ((1, 2), (2, 3), (1, 3)):
            rows[-1][f"alpha{left}_vs_alpha{right}_direction_delta_deg"] = circular_delta(
                float(rows[-1][f"alpha{left}_motion_direction_image_deg"]),
                float(rows[-1][f"alpha{right}_motion_direction_image_deg"]),
                360.0,
            )
            rows[-1][f"alpha{left}_vs_alpha{right}_orientation_delta_deg"] = circular_delta(
                float(rows[-1][f"alpha{left}_bar_orientation_image_deg"]),
                float(rows[-1][f"alpha{right}_bar_orientation_image_deg"]),
                180.0,
            )
    arrays = {
        "rr100_index": units.astype(np.int64),
        "spatial_cpd": sf.astype(np.float64),
        "temporal_hz": tf.astype(np.float64),
        "motion_direction_image_deg": directions.astype(np.float64),
        "signed_f0_hz": signed,
        "positive_f0_hz": positive.astype(np.float32),
        "direction_summed_positive_f0_hz": sensitivity.astype(np.float32),
        "relative_sf_tf_sensitivity": relative.astype(np.float32),
        "conditional_direction_distribution": conditional_direction.astype(np.float32),
        "smooth_sftf_weight_alpha1": smooth_weights[1.0].astype(np.float32),
        "smooth_sftf_weight_alpha2": smooth_weights[2.0].astype(np.float32),
        "smooth_sftf_weight_alpha3": smooth_weights[3.0].astype(np.float32),
        "smooth_direction_profile_alpha1": smooth_profiles[1.0].astype(np.float32),
        "smooth_direction_profile_alpha2": smooth_profiles[2.0].astype(np.float32),
        "smooth_direction_profile_alpha3": smooth_profiles[3.0].astype(np.float32),
        "full_marginal_direction_profile_hz": marginal_profile.astype(np.float32),
    }
    return arrays, pd.DataFrame(rows)


def predict_model(parameters: np.ndarray, x: np.ndarray, interaction: bool) -> np.ndarray:
    log_sf, log_tf, theta = x.T
    baseline, amplitude, mu_sf, log_sigma_sf, mu_tf, log_sigma_tf = parameters[:6]
    sigma_sf = np.exp(log_sigma_sf)
    sigma_tf = np.exp(log_sigma_tf)
    envelope = np.exp(
        -0.5 * ((log_sf - mu_sf) / sigma_sf) ** 2
        -0.5 * ((log_tf - mu_tf) / sigma_tf) ** 2
    )
    basis = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2.0 * theta), np.sin(2.0 * theta)])
    angular = basis @ parameters[6:10]
    if interaction:
        centered_sf = log_sf - mu_sf
        centered_tf = log_tf - mu_tf
        angular += centered_sf * (basis @ parameters[10:14])
        angular += centered_tf * (basis @ parameters[14:18])
    angular_gain = np.exp(np.clip(angular, -8.0, 8.0))
    return baseline + amplitude * envelope * angular_gain


def initial_and_bounds(x: np.ndarray, y: np.ndarray, interaction: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.maximum(y, 0.0) + 1e-6
    log_sf, log_tf = x[:, 0], x[:, 1]
    mu_sf = float(np.average(log_sf, weights=weights))
    mu_tf = float(np.average(log_tf, weights=weights))
    sigma_sf = max(float(np.sqrt(np.average((log_sf - mu_sf) ** 2, weights=weights))), 0.5)
    sigma_tf = max(float(np.sqrt(np.average((log_tf - mu_tf) ** 2, weights=weights))), 0.5)
    maximum = max(float(np.max(y)), 1e-3)
    initial = np.asarray([0.0, maximum, mu_sf, np.log(sigma_sf), mu_tf, np.log(sigma_tf)] + [0.0] * (12 if interaction else 4))
    lower = np.asarray(
        [0.0, 0.0, log_sf.min() - 1.0, np.log(0.15), log_tf.min() - 1.0, np.log(0.15)]
        + [-4.0] * 4
        + ([-2.0] * 8 if interaction else [])
    )
    upper = np.asarray(
        [maximum * 2.0 + 1.0, maximum * 10.0 + 1.0, log_sf.max() + 1.0, np.log(6.0), log_tf.max() + 1.0, np.log(6.0)]
        + [4.0] * 4
        + ([2.0] * 8 if interaction else [])
    )
    initial = np.minimum(np.maximum(initial, lower + 1e-8), upper - 1e-8)
    return initial, lower, upper


def fit_model(x: np.ndarray, y: np.ndarray, interaction: bool) -> np.ndarray:
    initial, lower, upper = initial_and_bounds(x, y, interaction)
    scale = max(float(np.std(y)), 0.1)
    result = least_squares(
        lambda parameters: predict_model(parameters, x, interaction) - y,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=scale,
        max_nfev=1600,
    )
    if not result.success and result.optimality > 1e-3:
        raise RuntimeError(f"Parametric fit failed: {result.message}")
    return result.x


def score(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    residual = y - prediction
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 1e-12 else float("nan")
    correlation = float(np.corrcoef(y, prediction)[0, 1]) if np.std(y) > 1e-12 and np.std(prediction) > 1e-12 else float("nan")
    return r2, correlation, float(np.sqrt(np.mean(residual**2)))


def selected_parametric_fits(
    arrays: dict[str, np.ndarray], selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    units = arrays["rr100_index"]
    sf = arrays["spatial_cpd"]
    tf = arrays["temporal_hz"]
    directions = arrays["motion_direction_image_deg"]
    ss, tt, dd = np.meshgrid(np.arange(len(sf)), np.arange(len(tf)), np.arange(len(directions)), indexing="ij")
    x = np.column_stack(
        [np.log2(sf[ss.ravel()]), np.log2(tf[tt.ravel()]), np.deg2rad(directions[dd.ravel()])]
    )
    folds = (ss.ravel() + 2 * tt.ravel() + 3 * dd.ravel()) % N_FOLDS
    metrics_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    payload: dict[int, dict[str, np.ndarray]] = {}
    parameter_names = [
        "baseline_hz", "amplitude_hz", "mu_log2_sf", "log_sigma_log2_sf", "mu_log2_tf", "log_sigma_log2_tf",
        "cos_direction", "sin_direction", "cos_orientation", "sin_orientation",
        "sf_x_cos_direction", "sf_x_sin_direction", "sf_x_cos_orientation", "sf_x_sin_orientation",
        "tf_x_cos_direction", "tf_x_sin_direction", "tf_x_cos_orientation", "tf_x_sin_orientation",
    ]
    for selection in selected.itertuples(index=False):
        unit = int(selection.rr100_index)
        position = int(np.flatnonzero(units == unit)[0])
        y = arrays["positive_f0_hz"][position].astype(float).ravel()
        oof: dict[str, np.ndarray] = {}
        full_predictions: dict[str, np.ndarray] = {}
        for name, interaction in (("separable_expected_form", False), ("frequency_dependent_angle", True)):
            if float(np.max(y)) <= 1e-8:
                oof[name] = np.full_like(y, np.nan)
                full_predictions[name] = np.full(
                    (len(sf), len(tf), len(directions)), np.nan, dtype=float
                )
                metrics_rows.append(
                    {
                        "rr100_index": unit,
                        "selection_role": str(selection.selection_role),
                        "model": name,
                        "fit_attempted": False,
                        "fit_exclusion_reason": "no_positive_above_blank_response",
                        "n_parameters": 18 if interaction else 10,
                        "fivefold_condition_cv_r2": np.nan,
                        "fivefold_condition_cv_correlation": np.nan,
                        "fivefold_condition_cv_rmse_hz": np.nan,
                        "full_fit_r2_diagnostic_only": np.nan,
                        "full_fit_correlation_diagnostic_only": np.nan,
                        "full_fit_rmse_hz_diagnostic_only": np.nan,
                    }
                )
                continue
            prediction = np.full_like(y, np.nan)
            for fold in range(N_FOLDS):
                train = folds != fold
                test = ~train
                parameters = fit_model(x[train], y[train], interaction)
                prediction[test] = predict_model(parameters, x[test], interaction)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Incomplete OOF predictions for unit {unit}, {name}")
            full_parameters = fit_model(x, y, interaction)
            full_prediction = predict_model(full_parameters, x, interaction)
            oof[name] = prediction
            full_predictions[name] = full_prediction.reshape(len(sf), len(tf), len(directions))
            cv_r2, cv_r, cv_rmse = score(y, prediction)
            full_r2, full_r, full_rmse = score(y, full_prediction)
            metrics_rows.append(
                {
                    "rr100_index": unit,
                    "selection_role": str(selection.selection_role),
                    "model": name,
                    "fit_attempted": True,
                    "fit_exclusion_reason": "",
                    "n_parameters": int(len(full_parameters)),
                    "fivefold_condition_cv_r2": cv_r2,
                    "fivefold_condition_cv_correlation": cv_r,
                    "fivefold_condition_cv_rmse_hz": cv_rmse,
                    "full_fit_r2_diagnostic_only": full_r2,
                    "full_fit_correlation_diagnostic_only": full_r,
                    "full_fit_rmse_hz_diagnostic_only": full_rmse,
                }
            )
            for parameter_name, value in zip(parameter_names, full_parameters):
                parameter_rows.append(
                    {"rr100_index": unit, "model": name, "parameter": parameter_name, "value": float(value)}
                )
        frame = pd.DataFrame(
            {
                "rr100_index": unit,
                "spatial_cpd": sf[ss.ravel()],
                "temporal_hz": tf[tt.ravel()],
                "motion_direction_image_deg": directions[dd.ravel()],
                "fold": folds,
                "observed_positive_f0_hz": y,
                "separable_expected_form_oof_hz": oof["separable_expected_form"],
                "frequency_dependent_angle_oof_hz": oof["frequency_dependent_angle"],
            }
        )
        prediction_rows.append(frame)
        payload[unit] = {
            "observed": y.reshape(len(sf), len(tf), len(directions)),
            "separable": full_predictions["separable_expected_form"],
            "interaction": full_predictions["frequency_dependent_angle"],
        }
    metrics = pd.DataFrame(metrics_rows)
    wide = metrics.pivot(index=["rr100_index", "selection_role"], columns="model", values="fivefold_condition_cv_r2").reset_index()
    wide["interaction_minus_separable_cv_r2"] = wide["frequency_dependent_angle"] - wide["separable_expected_form"]
    metrics = metrics.merge(wide[["rr100_index", "interaction_minus_separable_cv_r2"]], on="rr100_index", validate="many_to_one")
    return metrics, pd.DataFrame(parameter_rows), pd.concat(prediction_rows, ignore_index=True), payload


def normalized(values: np.ndarray) -> np.ndarray:
    maximum = max(float(np.nanmax(values)), 1e-12)
    return np.asarray(values, float) / maximum


def plot_checkpoint(
    arrays: dict[str, np.ndarray], selected: pd.DataFrame, metrics: pd.DataFrame, payload: dict[int, dict[str, np.ndarray]]
) -> None:
    sf = arrays["spatial_cpd"]
    tf = arrays["temporal_hz"]
    directions = arrays["motion_direction_image_deg"]
    units = arrays["rr100_index"]
    fig = plt.figure(figsize=(17.5, 3.0 * len(selected)), constrained_layout=True)
    grid = fig.add_gridspec(len(selected), 5, width_ratios=[1.1, 1.0, 1.15, 1.15, 1.15])
    for row_number, selection in enumerate(selected.itertuples(index=False)):
        unit = int(selection.rr100_index)
        position = int(np.flatnonzero(units == unit)[0])
        observed = payload[unit]["observed"]
        separable = payload[unit]["separable"]
        interaction = payload[unit]["interaction"]
        sensitivity = arrays["relative_sf_tf_sensitivity"][position]
        smooth_weight = arrays["smooth_sftf_weight_alpha2"][position]
        preferred_tf_index = int(np.unravel_index(np.argmax(sensitivity), sensitivity.shape)[1])

        ax = fig.add_subplot(grid[row_number, 0])
        displayed_weight = smooth_weight / max(float(smooth_weight.max()), 1e-12)
        image = ax.imshow(displayed_weight, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(tf)), [f"{value:g}" for value in tf], rotation=45, ha="right")
        ax.set_yticks(range(len(sf)), [f"{value:g}" for value in sf])
        ax.set(xlabel="TF (Hz)", ylabel="SF (cpd)")
        ax.set_title(f"RR100 {unit}: {str(selection.selection_role).replace('_', ' ')}\nsmooth SF–TF weight, α=2", loc="left", fontsize=9, fontweight="bold")
        if row_number == 0:
            fig.colorbar(image, ax=ax, fraction=0.045, label="relative smooth weight")

        ax = fig.add_subplot(grid[row_number, 1], projection="polar")
        closed = np.r_[np.deg2rad(directions), 2.0 * math.pi]
        observed_profile = arrays["smooth_direction_profile_alpha2"][position]
        weights = arrays["smooth_sftf_weight_alpha2"][position]

        def fitted_profile(values: np.ndarray) -> np.ndarray:
            if not np.isfinite(values).any():
                return np.full(len(directions), np.nan, dtype=float)
            total = np.nansum(values, axis=-1, keepdims=True)
            conditional = np.divide(
                values,
                total,
                out=np.zeros_like(values),
                where=total > 1e-12,
            )
            return np.einsum("st,std->d", weights, conditional)

        sep_profile = fitted_profile(separable)
        int_profile = fitted_profile(interaction)
        for values, color, style, label, width in (
            (arrays["smooth_direction_profile_alpha1"][position], "0.55", "--", "empirical α=1", 1.2),
            (observed_profile, "black", "-", "empirical α=2", 2.2),
            (arrays["smooth_direction_profile_alpha3"][position], "#009E73", "-.", "empirical α=3", 1.2),
            (sep_profile, "#3B6FB6", "--", "separable", 1.5),
            (int_profile, "#D55E00", ":", "frequency-dependent angle", 1.7),
        ):
            if not np.isfinite(values).any():
                continue
            values = normalized(values)
            ax.plot(closed, np.r_[values, values[0]], color=color, ls=style, lw=width, label=label)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_title("smooth-weighted direction profile", fontsize=9)
        if row_number == 0:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=7)

        maximum = max(float(observed[:, preferred_tf_index].max()), 1e-12)
        for column, values, title in (
            (2, observed, "empirical"),
            (3, separable, "expected-form fit"),
            (4, interaction, "frequency-dependent-angle fit"),
        ):
            ax = fig.add_subplot(grid[row_number, column])
            if not np.isfinite(values).any():
                ax.set_facecolor("0.94")
                ax.text(0.5, 0.5, "not fit\nno positive above-blank response", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
                continue
            matrix = values[:, preferred_tf_index, :].T / maximum
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=max(1.0, float(np.nanmax(matrix))))
            ax.set_xticks(range(len(sf)), [f"{value:g}" for value in sf], rotation=45, ha="right")
            ax.set_yticks(range(len(directions)), [f"{value:g}°" for value in directions])
            ax.set(xlabel="SF (cpd)", ylabel="motion direction")
            unit_metrics = metrics[metrics["rr100_index"].eq(unit)].set_index("model")
            if column == 3:
                cv = float(unit_metrics.loc["separable_expected_form", "fivefold_condition_cv_r2"])
                title += f"\nCV R²={cv:.2f}"
            elif column == 4:
                cv = float(unit_metrics.loc["frequency_dependent_angle", "fivefold_condition_cv_r2"])
                delta = float(unit_metrics["interaction_minus_separable_cv_r2"].iloc[0])
                title += f"\nCV R²={cv:.2f}, Δ={delta:+.2f}"
            else:
                title += f" at TF={tf[preferred_tf_index]:g} Hz"
            ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
            if column == 4:
                fig.colorbar(image, ax=ax, fraction=0.045, label="observed-peak normalized response")
    fig.suptitle(
        "RR100 joint SF–TF–direction tuning: empirical tensor is primary; parametric fits are diagnostics\n"
        "Angular summary uses smooth SF–TF weights S² with no hard cutoff; CV holds out stimulus conditions",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT / "selected_joint_tensor_smooth_weighting_parametric_fit_checkpoint.png", dpi=210, bbox_inches="tight")
    fig.savefig(OUT / "selected_joint_tensor_smooth_weighting_parametric_fit_checkpoint.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, _, _ = prepare_native()
    arrays, summary = build_tensor(points)
    selected = pd.read_csv(SELECTION)
    metrics, parameters, predictions, payload = selected_parametric_fits(arrays, selected)
    plot_checkpoint(arrays, selected, metrics, payload)

    np.savez_compressed(OUT / "rr100_empirical_joint_sftf_direction_tuning.npz", **arrays)
    summary.to_csv(OUT / "smooth_weighted_orientation_direction_summary.csv", index=False)
    metrics.to_csv(OUT / "selected_parametric_model_metrics.csv", index=False)
    parameters.to_csv(OUT / "selected_parametric_model_parameters.csv", index=False)
    predictions.to_csv(OUT / "selected_parametric_oof_predictions.csv", index=False)
    selected.to_csv(OUT / "selected_unit_roles.csv", index=False)

    comparison = metrics.drop_duplicates("rr100_index").set_index("rr100_index")["interaction_minus_separable_cv_r2"]
    positive = arrays["positive_f0_hz"].astype(np.float64)
    marginal = positive.sum(axis=(1, 2))
    marginal = np.divide(
        marginal,
        marginal.sum(axis=-1, keepdims=True),
        out=np.zeros_like(marginal),
        where=marginal.sum(axis=-1, keepdims=True) > 1e-12,
    )
    alpha1_identity_error = float(
        np.max(np.abs(arrays["smooth_direction_profile_alpha1"].astype(np.float64) - marginal))
    )
    manifest = {
        "analysis": "rr100_joint_sftf_direction_tuning_smooth_weighting_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "empirical_all_unit_tensor_complete_selected_unit_parametric_checkpoint_complete_population_fit_not_run",
        "primary_object": "full empirical positive and signed F0 tensor over SF x |TF| x eight motion directions",
        "derived_summary": (
            "For each SF x |TF| cell, normalize the positive direction profile to q(direction|SF,TF); "
            "then average q using smooth weights proportional to S(SF,TF)^alpha with no hard cutoff. "
            f"Primary alpha={PRIMARY_WEIGHT_EXPONENT:g}; saved controls alpha={WEIGHT_EXPONENTS}."
        ),
        "parametric_role": "diagnostic only; models do not replace the empirical routing tensor",
        "smooth_weighting_contract": {
            "cell_sensitivity": "S(SF,TF)=sum_direction max(signed_above_blank_F0,0)",
            "conditional_angular_profile": "q(direction|SF,TF)=positive_F0/S for S>0",
            "aggregate": "P_alpha(direction)=sum_SF,TF normalized[S^alpha] * q(direction|SF,TF)",
            "primary_alpha": PRIMARY_WEIGHT_EXPONENT,
            "saved_sensitivity_alphas": list(WEIGHT_EXPONENTS),
            "hard_cutoff": None,
            "alpha1_max_abs_error_vs_ordinary_response_marginal": alpha1_identity_error,
        },
        "model_hierarchy": {
            "separable_expected_form": "log-Gaussian SF x log-Gaussian TF envelope times first/second circular harmonics",
            "frequency_dependent_angle": "expected-form model plus log-SF and log-TF interactions on both circular harmonics",
            "validation": f"deterministic {N_FOLDS}-fold held-out condition prediction",
        },
        "tensor_shape": list(arrays["positive_f0_hz"].shape),
        "selected_interaction_minus_separable_cv_r2": {str(int(k)): float(v) for k, v in comparison.items()},
        "sources": {
            "native_signed_tf_summary": file_identity(NATIVE),
            "legacy_static_probe_context_only": file_identity(LEGACY),
            "map_first_selection": file_identity(SELECTION),
        },
        "next_gate": "Inspect selected smooth-weighted profiles and alpha sensitivity before fitting the parametric hierarchy to all units or replacing any Figure 4 orientation field.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

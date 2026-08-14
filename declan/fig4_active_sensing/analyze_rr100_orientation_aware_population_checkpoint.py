#!/usr/bin/env python3
"""Population test of direct-F0 radial and orientation-aware spectral routing.

The orientation-aware model is strictly nested inside the radial model: its
orientation factor has mean one at every SFxTF bin.  This script uses the three
currently complete balanced production rounds and makes no neural-model calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from declan.fig4_active_sensing.spectral_cache_contract import (
    validate_artifact_not_superseded,
    validated_spectral_cache_from_environment,
)


ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
RESPONSES = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_002_n003"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_population_checkpoint_v1"
SPLIT_SEEDS = (2718, 31415, 16180)
N_FOLDS = 5
EPS = np.finfo(float).tiny

OUTCOME_LABELS = {
    "activation_rms_hz": "activation magnitude\nRMS[FEM − stabilized]",
    "activation_mean_abs_hz": "activation magnitude\nmean |FEM − stabilized|",
    "delta_mean_rate_hz": "signed mean-rate change",
    "delta_expected_spikes": "expected-spike change",
    "delta_information_bits_spikes": "information-numerator change",
    "delta_ssi_bits_per_spike": "SSI change (bits/spike)",
}
MODEL_ORDER = [
    "global",
    "radial",
    "orientation",
    "radial_plus_orientation",
    "global_plus_radial",
    "global_plus_radial_plus_orientation",
]
MODEL_LABELS = {
    "global": "global\npower",
    "radial": "direct-F0\nradial",
    "orientation": "direct-F0\norientation",
    "radial_plus_orientation": "radial +\norientation term",
    "global_plus_radial": "global +\nradial",
    "global_plus_radial_plus_orientation": "global + radial +\norientation term",
}
MODEL_COLORS = {
    "global": "#7F7F7F",
    "radial": "#0072B2",
    "orientation": "#D55E00",
    "radial_plus_orientation": "#CC79A7",
    "global_plus_radial": "#56B4E9",
    "global_plus_radial_plus_orientation": "#E69F00",
}


def assign_identity_folds(identities: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(identities)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {int(identity): int(index % N_FOLDS) for index, identity in enumerate(shuffled)}
    return np.asarray([mapping[int(identity)] for identity in identities], dtype=int)


def design(columns: list[np.ndarray]) -> np.ndarray:
    if not columns:
        raise ValueError("At least one feature is required")
    return np.column_stack([np.ones(len(columns[0]))] + columns)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = float(np.nanmean(train))
    scale = float(np.nanstd(train))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (train - center) / scale, (test - center) / scale


def crossed_predictions(
    y: np.ndarray,
    image_fold: np.ndarray,
    trace_fold: np.ndarray,
    feature_sets: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    predictions = {name: np.full(len(y), np.nan, dtype=float) for name in feature_sets}
    for image_group in range(N_FOLDS):
        for trace_group in range(N_FOLDS):
            test = (image_fold == image_group) & (trace_fold == trace_group)
            train = (image_fold != image_group) & (trace_fold != trace_group)
            if not np.any(test):
                continue
            for name, columns in feature_sets.items():
                train_columns: list[np.ndarray] = []
                test_columns: list[np.ndarray] = []
                for column in columns:
                    train_column, test_column = standardize(column[train], column[test])
                    train_columns.append(train_column)
                    test_columns.append(test_column)
                x_train = design(train_columns)
                x_test = design(test_columns)
                valid = np.isfinite(y[train]) & np.all(np.isfinite(x_train), axis=1)
                if valid.sum() <= x_train.shape[1] + 2:
                    continue
                beta = np.linalg.lstsq(x_train[valid], y[train][valid], rcond=None)[0]
                predictions[name][test] = x_test @ beta
    return predictions


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y) & np.isfinite(prediction)
    if valid.sum() < 3:
        return {"cv_r2": np.nan, "cv_correlation": np.nan, "cv_mae": np.nan, "n_predictions": int(valid.sum())}
    residual = float(np.sum((y[valid] - prediction[valid]) ** 2))
    total = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    correlation = (
        float(np.corrcoef(y[valid], prediction[valid])[0, 1])
        if np.std(y[valid]) > 0 and np.std(prediction[valid]) > 0
        else np.nan
    )
    return {
        "cv_r2": 1.0 - residual / total if total > 0 else np.nan,
        "cv_correlation": correlation,
        "cv_mae": float(np.mean(np.abs(y[valid] - prediction[valid]))),
        "n_predictions": int(valid.sum()),
    }


def clustered_median_interval(
    values: pd.Series,
    sessions: pd.Series,
    seed: int,
    n_bootstrap: int = 5000,
) -> tuple[float, float, float]:
    frame = pd.DataFrame({"value": values, "session": sessions}).dropna()
    session_names = frame.session.unique()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    groups = {session: frame.loc[frame.session.eq(session), "value"].to_numpy(float) for session in session_names}
    for index in range(n_bootstrap):
        sampled = rng.choice(session_names, size=len(session_names), replace=True)
        draws[index] = np.median(np.concatenate([groups[session] for session in sampled]))
    return float(np.median(frame.value)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_predictors_and_outcomes(spectral_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    with np.load(TUNING / "orientation_aware_f0_tuning_and_routing.npz", allow_pickle=False) as data:
        tuning = {key: np.asarray(data[key]) for key in data.files}
    with np.load(spectral_dir / "condition_spectra.npz", allow_pickle=False) as data:
        spectral = {key: np.asarray(data[key]) for key in data.files}
    condition = pd.read_csv(RESPONSES / "condition_index.csv")
    if not np.array_equal(condition.matrix_row_index.to_numpy(), spectral["matrix_row_index"]):
        raise ValueError("Spectral and response conditions are not row-aligned")

    sf_all = 0.5 * (spectral["sf_edges_cpd"][:-1] + spectral["sf_edges_cpd"][1:])
    sf_mask = (sf_all >= tuning["movie_sf_cpd"].min()) & (sf_all <= tuning["movie_sf_cpd"].max())
    tf_mask = (spectral["tf_hz"] >= tuning["movie_tf_hz"].min()) & (spectral["tf_hz"] <= tuning["movie_tf_hz"].max())
    sf = sf_all[sf_mask]
    tf = spectral["tf_hz"][tf_mask]
    if not np.allclose(sf, tuning["movie_sf_cpd"]) or not np.allclose(tf, tuning["movie_tf_hz"]):
        raise ValueError("Movie frequency axes do not match the validated tuning query axes")
    radial_power = spectral["radial_power"][:, tf_mask][:, :, sf_mask].astype(float)
    orientation_power = spectral["orientation_power"][:, tf_mask][:, :, sf_mask, :].astype(float)
    if not np.allclose(radial_power, orientation_power.sum(axis=-1), rtol=2e-5, atol=1e-3):
        raise ValueError("Orientation power does not sum to radial power")

    weights = tuning["orientation_aware_f0_weight"].astype(float)
    radial_weight = tuning["smoothed_radial_f0_weight"].astype(float)
    global_power = radial_power.sum(axis=(1, 2))
    radial_drive_all = np.einsum("ctf,utf->cu", radial_power, radial_weight)
    orientation_drive_all = np.einsum("ctfo,utfo->cu", orientation_power, weights)
    orientation_delta_all = orientation_drive_all - radial_drive_all

    cohort = pd.read_csv(TUNING / "orientation_tuning_fit_quality_and_movie_overlap.csv")
    cohort = cohort[
        cohort.recorded_validation_pass.fillna(False)
        & cohort.responsive_positive_f0_flag.fillna(False)
    ].sort_values("rr100_index").reset_index(drop=True)
    fit_summary = pd.read_csv(
        ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_extended_tf_f0_analysis_v1/extended_f0_fit_unit_summary.csv"
    )[["rr100_index", "session"]]
    cohort = cohort.merge(fit_summary, on="rr100_index", how="left", validate="one_to_one")
    units = cohort.rr100_index.to_numpy(int)

    predictors = {
        "global_power": global_power,
        "radial_drive": radial_drive_all[:, units],
        "orientation_drive": orientation_drive_all[:, units],
        "orientation_delta": orientation_delta_all[:, units],
    }
    moving_rate = np.load(RESPONSES / "moving_mean_rate_hz.npy")[:, units].astype(float)
    moving_spikes = np.load(RESPONSES / "moving_expected_spikes.npy")[:, units].astype(float)
    moving_information = np.load(RESPONSES / "moving_information_numerator_bits_spikes.npy")[:, units].astype(float)
    moving_ssi = np.load(RESPONSES / "moving_movie_ssi_bits_per_spike.npy")[:, units].astype(float)
    baseline = np.load(RESPONSES / "stabilized_by_image_sufficient_statistics.npz")
    image = condition.image_index.to_numpy(int)
    baseline_rate = baseline["mean_rate_hz"][image][:, units].astype(float)
    baseline_spikes = baseline["expected_spikes"][image][:, units].astype(float)
    baseline_information = baseline["information_numerator_bits_spikes"][image][:, units].astype(float)
    baseline_ssi = baseline["movie_ssi_bits_per_spike"][image][:, units].astype(float)
    outcomes = {
        "activation_rms_hz": np.load(RESPONSES / "moving_temporal_rms_delta_from_stabilized_hz.npy")[:, units].astype(float),
        "activation_mean_abs_hz": np.load(RESPONSES / "moving_temporal_mean_abs_delta_from_stabilized_hz.npy")[:, units].astype(float),
        "delta_mean_rate_hz": moving_rate - baseline_rate,
        "delta_expected_spikes": moving_spikes - baseline_spikes,
        "delta_information_bits_spikes": moving_information - baseline_information,
        "delta_ssi_bits_per_spike": moving_ssi - baseline_ssi,
        "moving_mean_rate_hz": moving_rate,
        "stabilized_mean_rate_hz": baseline_rate,
    }
    return predictors, outcomes, cohort, condition


def predictor_feature_sets(predictors: dict[str, np.ndarray], unit_position: int) -> dict[str, list[np.ndarray]]:
    global_power = predictors["global_power"]
    radial = predictors["radial_drive"][:, unit_position]
    orientation = predictors["orientation_drive"][:, unit_position]
    delta = predictors["orientation_delta"][:, unit_position]
    return {
        "global": [global_power],
        "radial": [radial],
        "orientation": [orientation],
        "radial_plus_orientation": [radial, delta],
        "global_plus_radial": [global_power, radial],
        "global_plus_radial_plus_orientation": [global_power, radial, delta],
    }


def run_outcome_models(
    predictors: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
    cohort: pd.DataFrame,
    condition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    split_rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], list[np.ndarray]] = {
        (outcome, model): [] for outcome in OUTCOME_LABELS for model in MODEL_ORDER
    }
    image = condition.image_index.to_numpy(int)
    trace = condition.trace_index.to_numpy(int)
    for split_index, seed in enumerate(SPLIT_SEEDS):
        image_fold = assign_identity_folds(image, seed)
        trace_fold = assign_identity_folds(trace, seed + 1000003)
        split_predictions = {
            key: np.full((len(condition), len(cohort)), np.nan, dtype=float) for key in predictions
        }
        for unit_position, unit in enumerate(cohort.rr100_index.to_numpy(int)):
            feature_sets = predictor_feature_sets(predictors, unit_position)
            for outcome_name in OUTCOME_LABELS:
                y = outcomes[outcome_name][:, unit_position]
                unit_predictions = crossed_predictions(y, image_fold, trace_fold, feature_sets)
                for model_name, prediction in unit_predictions.items():
                    split_predictions[(outcome_name, model_name)][:, unit_position] = prediction
                    split_rows.append(
                        {
                            "split_index": split_index,
                            "split_seed": seed,
                            "rr100_index": unit,
                            "outcome": outcome_name,
                            "model": model_name,
                            **score(y, prediction),
                        }
                    )
        for key, value in split_predictions.items():
            predictions[key].append(value)

    averaged_predictions = {key: np.nanmean(np.stack(value, axis=0), axis=0) for key, value in predictions.items()}
    split_table = pd.DataFrame(split_rows)
    aggregate_rows: list[dict[str, object]] = []
    for unit_position, unit in enumerate(cohort.rr100_index.to_numpy(int)):
        for outcome_name in OUTCOME_LABELS:
            y = outcomes[outcome_name][:, unit_position]
            for model_name in MODEL_ORDER:
                repeat = split_table[
                    split_table.rr100_index.eq(unit)
                    & split_table.outcome.eq(outcome_name)
                    & split_table.model.eq(model_name)
                ]
                ensemble_score = score(y, averaged_predictions[(outcome_name, model_name)][:, unit_position])
                aggregate_rows.append(
                    {
                        "rr100_index": unit,
                        "outcome": outcome_name,
                        "model": model_name,
                        "median_split_cv_r2": float(repeat.cv_r2.median()),
                        "minimum_split_cv_r2": float(repeat.cv_r2.min()),
                        "maximum_split_cv_r2": float(repeat.cv_r2.max()),
                        "median_split_cv_correlation": float(repeat.cv_correlation.median()),
                        "ensemble_cv_r2": ensemble_score["cv_r2"],
                        "ensemble_cv_correlation": ensemble_score["cv_correlation"],
                        "ensemble_cv_mae": ensemble_score["cv_mae"],
                    }
                )
    return split_table, pd.DataFrame(aggregate_rows), averaged_predictions


def run_transform_sensitivity(
    predictors: dict[str, np.ndarray],
    outcome: np.ndarray,
    cohort: pd.DataFrame,
    condition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image = condition.image_index.to_numpy(int)
    trace = condition.trace_index.to_numpy(int)
    for split_index, seed in enumerate(SPLIT_SEEDS):
        image_fold = assign_identity_folds(image, seed)
        trace_fold = assign_identity_folds(trace, seed + 1000003)
        for unit_position, unit in enumerate(cohort.rr100_index.to_numpy(int)):
            raw_features = {
                "global": predictors["global_power"],
                "radial": predictors["radial_drive"][:, unit_position],
                "orientation": predictors["orientation_drive"][:, unit_position],
            }
            for transform in ("linear_power", "sqrt_power", "log_power"):
                transformed: dict[str, np.ndarray] = {}
                for name, values in raw_features.items():
                    if transform == "linear_power":
                        transformed[name] = values
                    elif transform == "sqrt_power":
                        transformed[name] = np.sqrt(np.maximum(values, 0.0))
                    else:
                        scale = max(float(np.nanmedian(np.maximum(values, 0.0))), EPS)
                        transformed[name] = np.log1p(np.maximum(values, 0.0) / scale)
                predictions = crossed_predictions(
                    outcome[:, unit_position], image_fold, trace_fold, {name: [values] for name, values in transformed.items()}
                )
                for model, prediction in predictions.items():
                    rows.append(
                        {
                            "split_index": split_index,
                            "split_seed": seed,
                            "rr100_index": unit,
                            "transform": transform,
                            "model": model,
                            **score(outcome[:, unit_position], prediction),
                        }
                    )
    return pd.DataFrame(rows)


def run_gain_models(
    predictors: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
    cohort: pd.DataFrame,
    condition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image = condition.image_index.to_numpy(int)
    trace = condition.trace_index.to_numpy(int)
    for split_index, seed in enumerate(SPLIT_SEEDS):
        image_fold = assign_identity_folds(image, seed)
        trace_fold = assign_identity_folds(trace, seed + 1000003)
        for unit_position, unit in enumerate(cohort.rr100_index.to_numpy(int)):
            y = outcomes["moving_mean_rate_hz"][:, unit_position]
            baseline = outcomes["stabilized_mean_rate_hz"][:, unit_position]
            feature_sets: dict[str, list[np.ndarray]] = {"baseline_only": [baseline]}
            for prefix, feature in (
                ("global", predictors["global_power"]),
                ("radial", predictors["radial_drive"][:, unit_position]),
                ("orientation", predictors["orientation_drive"][:, unit_position]),
            ):
                interaction = baseline * feature
                feature_sets[f"{prefix}_additive"] = [baseline, feature]
                feature_sets[f"{prefix}_multiplicative"] = [baseline, interaction]
                feature_sets[f"{prefix}_both"] = [baseline, feature, interaction]
            predictions = crossed_predictions(y, image_fold, trace_fold, feature_sets)
            baseline_r2 = float(score(y, predictions["baseline_only"])["cv_r2"])
            for model, prediction in predictions.items():
                values = score(y, prediction)
                rows.append(
                    {
                        "split_index": split_index,
                        "split_seed": seed,
                        "rr100_index": unit,
                        "model": model,
                        **values,
                        "delta_r2_over_baseline": float(values["cv_r2"]) - baseline_r2,
                    }
                )
    return pd.DataFrame(rows)


def build_increment_table(scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    pivot = scores.pivot(index=["rr100_index", "outcome"], columns="model", values="median_split_cv_r2").reset_index()
    pivot["radial_minus_global"] = pivot.radial - pivot["global"]
    pivot["orientation_minus_radial"] = pivot.orientation - pivot.radial
    pivot["nested_orientation_over_radial"] = pivot.radial_plus_orientation - pivot.radial
    pivot["nested_orientation_over_global_plus_radial"] = (
        pivot.global_plus_radial_plus_orientation - pivot.global_plus_radial
    )
    return pivot.merge(
        cohort[["rr100_index", "session", "orientation_vector_strength", "harmonic_cv_r2", "chosen_orientation_model_cv_r2"]],
        on="rr100_index",
        how="left",
        validate="many_to_one",
    )


def population_summaries(
    scores: pd.DataFrame,
    increments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary = (
        scores.groupby(["outcome", "model"], as_index=False)
        .agg(
            n_units=("rr100_index", "size"),
            median_split_cv_r2=("median_split_cv_r2", "median"),
            median_ensemble_cv_r2=("ensemble_cv_r2", "median"),
            median_split_cv_correlation=("median_split_cv_correlation", "median"),
        )
    )
    rows: list[dict[str, object]] = []
    metrics = [
        "radial_minus_global",
        "orientation_minus_radial",
        "nested_orientation_over_radial",
        "nested_orientation_over_global_plus_radial",
    ]
    for outcome_index, outcome in enumerate(OUTCOME_LABELS):
        subset = increments[increments.outcome.eq(outcome)]
        for metric_index, metric in enumerate(metrics):
            median, low, high = clustered_median_interval(
                subset[metric], subset.session, seed=9000 + 101 * outcome_index + metric_index
            )
            rows.append(
                {
                    "outcome": outcome,
                    "increment": metric,
                    "n_units": int(len(subset)),
                    "median_delta_cv_r2": median,
                    "session_cluster_bootstrap_ci_low": low,
                    "session_cluster_bootstrap_ci_high": high,
                    "fraction_units_positive": float((subset[metric] > 0).mean()),
                }
            )
    return model_summary, pd.DataFrame(rows)


def select_examples(increments: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    activation = increments[increments.outcome.eq("activation_rms_hz")].copy()
    ssi = increments[increments.outcome.eq("delta_ssi_bits_per_spike")][
        ["rr100_index", "nested_orientation_over_global_plus_radial"]
    ].rename(columns={"nested_orientation_over_global_plus_radial": "ssi_orientation_increment"})
    activation = activation.merge(ssi, on="rr100_index", how="left", validate="one_to_one")
    selected: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, frame: pd.DataFrame, column: str, largest: bool, criterion: str) -> None:
        frame = frame[~frame.rr100_index.isin(used)].dropna(subset=[column])
        row = frame.loc[frame[column].idxmax() if largest else frame[column].idxmin()].copy()
        row["selection_role"] = role
        row["selection_metric"] = column
        row["selection_value"] = float(row[column])
        row["selection_criterion"] = criterion
        selected.append(row)
        used.add(int(row.rr100_index))

    add(
        "orientation improves activation prediction",
        activation,
        "nested_orientation_over_global_plus_radial",
        True,
        "largest median-split activation ΔR2 from adding orientation to global+radial",
    )
    add(
        "orientation worsens activation prediction",
        activation,
        "nested_orientation_over_global_plus_radial",
        False,
        "smallest median-split activation ΔR2 from adding orientation to global+radial",
    )
    activation["activation_minus_ssi_increment"] = (
        activation.nested_orientation_over_global_plus_radial - activation.ssi_orientation_increment
    )
    activation_benefit_pool = activation[
        activation.nested_orientation_over_global_plus_radial.gt(0)
        & activation.ssi_orientation_increment.le(0)
    ]
    if activation_benefit_pool.empty:
        activation_benefit_pool = activation
    add(
        "activation benefit without SSI benefit",
        activation_benefit_pool,
        "activation_minus_ssi_increment",
        True,
        "largest activation orientation increment minus SSI orientation increment",
    )
    activation["ssi_minus_activation_increment"] = (
        activation.ssi_orientation_increment - activation.nested_orientation_over_global_plus_radial
    )
    ssi_benefit_pool = activation[
        activation.ssi_orientation_increment.gt(0)
        & activation.nested_orientation_over_global_plus_radial.le(0)
    ]
    if ssi_benefit_pool.empty:
        ssi_benefit_pool = activation
    add(
        "SSI benefit without activation benefit",
        ssi_benefit_pool,
        "ssi_minus_activation_increment",
        True,
        "largest SSI orientation increment minus activation orientation increment",
    )
    result = pd.DataFrame(selected).merge(
        cohort[
            [
                "rr100_index",
                "preferred_sf_cpd",
                "extended_tf_center_frequency",
                "preferred_orientation_deg",
                "recorded_sf_curve_r_full_support",
            ]
        ],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    )
    return result


def predictor_geometry_table(predictors: dict[str, np.ndarray], cohort: pd.DataFrame) -> pd.DataFrame:
    global_power = predictors["global_power"]
    radial = predictors["radial_drive"]
    orientation = predictors["orientation_drive"]
    delta = predictors["orientation_delta"]
    rows: list[dict[str, object]] = []
    for unit_position, unit in enumerate(cohort.rr100_index.to_numpy(int)):
        ratio = orientation[:, unit_position] / np.maximum(radial[:, unit_position], EPS)
        rows.append(
            {
                "rr100_index": unit,
                "global_radial_correlation": float(np.corrcoef(global_power, radial[:, unit_position])[0, 1]),
                "radial_orientation_correlation": float(np.corrcoef(radial[:, unit_position], orientation[:, unit_position])[0, 1]),
                "orientation_delta_sd_over_radial_sd": float(
                    np.std(delta[:, unit_position]) / max(float(np.std(radial[:, unit_position])), 1e-15)
                ),
                "orientation_over_radial_ratio_median": float(np.median(ratio)),
                "orientation_over_radial_ratio_iqr": float(np.subtract(*np.percentile(ratio, [75, 25]))),
                "orientation_over_radial_ratio_p05": float(np.percentile(ratio, 5)),
                "orientation_over_radial_ratio_p95": float(np.percentile(ratio, 95)),
            }
        )
    return pd.DataFrame(rows).merge(
        cohort[["rr100_index", "orientation_vector_strength", "chosen_orientation_model_cv_r2"]],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    )


def make_predictor_geometry_figure(geometry: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    bins = np.linspace(0.6, 1.0, 18)
    axes[0].hist(geometry.global_radial_correlation, bins=bins, alpha=0.65, color="#7F7F7F", label="global vs radial")
    axes[0].hist(geometry.radial_orientation_correlation, bins=bins, alpha=0.65, color="#D55E00", label="radial vs orientation")
    axes[0].set(xlabel="condition-wise predictor correlation", ylabel="units", title="Predictors share a strong common component")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(geometry.orientation_delta_sd_over_radial_sd, bins=18, color="#0072B2", alpha=0.8)
    axes[1].axvline(geometry.orientation_delta_sd_over_radial_sd.median(), color="black", lw=2)
    axes[1].set(xlabel="SD[orientation − radial] / SD[radial]", ylabel="units", title="But orientation adds substantial variation")

    axes[2].hist(geometry.orientation_over_radial_ratio_iqr, bins=18, color="#E69F00", alpha=0.8)
    axes[2].axvline(geometry.orientation_over_radial_ratio_iqr.median(), color="black", lw=2)
    axes[2].set(xlabel="IQR of condition-wise orientation/radial ratio", ylabel="units", title="Alignment changes across conditions")

    axes[3].scatter(
        geometry.orientation_vector_strength,
        geometry.orientation_delta_sd_over_radial_sd,
        c=geometry.radial_orientation_correlation,
        cmap="viridis",
        s=52,
        edgecolor="white",
    )
    axes[3].set(
        xlabel="grating orientation vector strength",
        ylabel="relative orientation-correction variation",
        title="Correction is not only an OSI proxy",
    )
    fig.suptitle(
        "The orientation-aware predictor is correlated with radial drive but is not numerically redundant",
        fontsize=14,
        weight="bold",
    )
    return fig


def make_population_figure(scores: pd.DataFrame, increment_summary: pd.DataFrame) -> plt.Figure:
    outcomes = ["activation_rms_hz", "delta_mean_rate_hz", "delta_information_bits_spikes", "delta_ssi_bits_per_spike"]
    shown_models = ["global", "radial", "orientation", "global_plus_radial_plus_orientation"]
    fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.95])
    for column, outcome in enumerate(outcomes):
        axis = fig.add_subplot(grid[0, column])
        subset = scores[scores.outcome.eq(outcome)].pivot(index="rr100_index", columns="model", values="median_split_cv_r2")
        for _, row in subset.iterrows():
            axis.plot(range(len(shown_models)), row[shown_models], color="0.82", lw=0.8, alpha=0.75)
        medians = subset[shown_models].median(axis=0)
        axis.plot(range(len(shown_models)), medians, color="black", lw=2.5, marker="o", ms=6)
        axis.scatter(
            range(len(shown_models)),
            medians,
            c=[MODEL_COLORS[model] for model in shown_models],
            s=55,
            zorder=3,
        )
        axis.axhline(0, color="0.35", ls="--", lw=1)
        axis.set_xticks(range(len(shown_models)), [MODEL_LABELS[model] for model in shown_models], fontsize=8)
        axis.set(ylabel="median across split seeds\nheld-out $R^2$" if column == 0 else "", title=OUTCOME_LABELS[outcome])

    increments = [
        ("radial_minus_global", "radial − global", "#0072B2"),
        ("nested_orientation_over_radial", "+orientation over radial", "#D55E00"),
        ("nested_orientation_over_global_plus_radial", "+orientation over global+radial", "#E69F00"),
    ]
    for column, outcome in enumerate(outcomes):
        axis = fig.add_subplot(grid[1, column])
        subset = increment_summary[increment_summary.outcome.eq(outcome)].set_index("increment")
        for position, (metric, label, color) in enumerate(increments):
            row = subset.loc[metric]
            axis.errorbar(
                position,
                row.median_delta_cv_r2,
                yerr=[[row.median_delta_cv_r2 - row.session_cluster_bootstrap_ci_low], [row.session_cluster_bootstrap_ci_high - row.median_delta_cv_r2]],
                fmt="o",
                color=color,
                capsize=4,
                ms=7,
            )
            axis.text(position, row.session_cluster_bootstrap_ci_low, f"{100*row.fraction_units_positive:.0f}% >0", ha="center", va="top", fontsize=8)
        axis.axhline(0, color="0.35", ls="--", lw=1)
        axis.set_xticks(range(len(increments)), [label for _, label, _ in increments], rotation=20, ha="right", fontsize=8)
        axis.set(ylabel="paired Δ held-out $R^2$" if column == 0 else "", title="Incremental information")
    fig.suptitle(
        "Direct-F0 spectral routing tested on unseen images and unseen corrected eye traces\n"
        "Thin lines are units; intervals resample recording sessions",
        fontsize=15,
        weight="bold",
    )
    return fig


def make_outcome_increment_figure(increments: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    metric = "nested_orientation_over_global_plus_radial"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    positions = np.arange(len(OUTCOME_LABELS))
    for position, outcome in enumerate(OUTCOME_LABELS):
        values = increments.loc[increments.outcome.eq(outcome), metric].to_numpy(float)
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(len(values))
        axes[0].scatter(position + jitter, values, s=18, alpha=0.65, color="#D55E00")
        row = summary[summary.outcome.eq(outcome) & summary.increment.eq(metric)].iloc[0]
        axes[0].errorbar(
            position,
            row.median_delta_cv_r2,
            yerr=[[row.median_delta_cv_r2 - row.session_cluster_bootstrap_ci_low], [row.session_cluster_bootstrap_ci_high - row.median_delta_cv_r2]],
            fmt="o",
            color="black",
            capsize=4,
            ms=6,
        )
    axes[0].axhline(0, color="0.35", ls="--")
    axes[0].set_xticks(positions, [OUTCOME_LABELS[outcome] for outcome in OUTCOME_LABELS], rotation=25, ha="right", fontsize=8)
    axes[0].set(ylabel="orientation increment over global+radial\nheld-out Δ$R^2$", title="Does orientation help each outcome?")

    activation = increments[increments.outcome.eq("activation_rms_hz")]
    axes[1].scatter(
        activation.orientation_vector_strength,
        activation[metric],
        c=activation.chosen_orientation_model_cv_r2,
        cmap="viridis",
        s=48,
        edgecolor="white",
    )
    axes[1].axhline(0, color="0.35", ls="--")
    axes[1].set(
        xlabel="grating orientation vector strength",
        ylabel="activation orientation increment Δ$R^2$",
        title="Is benefit larger for orientation-selective units?",
    )

    pivot = increments.pivot(index="rr100_index", columns="outcome", values=metric)
    axes[2].scatter(
        pivot.activation_rms_hz,
        pivot.delta_ssi_bits_per_spike,
        s=48,
        alpha=0.8,
        color="#0072B2",
        edgecolor="white",
    )
    axes[2].axhline(0, color="0.35", ls="--")
    axes[2].axvline(0, color="0.35", ls="--")
    axes[2].set(
        xlabel="activation orientation increment Δ$R^2$",
        ylabel="SSI orientation increment Δ$R^2$",
        title="Activation and SSI need not agree",
    )
    fig.suptitle("Orientation-specific information is evaluated separately for response magnitude and spatial information", fontsize=14, weight="bold")
    return fig


def binned_residual(axis: plt.Axes, x: np.ndarray, residual: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(residual)
    x = x[valid]
    residual = residual[valid]
    edges = np.quantile(x, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    centers: list[float] = []
    means: list[float] = []
    errors: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (x >= low) & (x <= high if high == edges[-1] else x < high)
        if mask.sum() < 3:
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(residual[mask])))
        errors.append(float(np.std(residual[mask], ddof=1) / np.sqrt(mask.sum())))
    axis.errorbar(centers, means, yerr=errors, marker="o", color="#D55E00", capsize=3)
    axis.axhline(0, color="0.4", ls="--")


def make_example_figure(
    selected: pd.DataFrame,
    cohort: pd.DataFrame,
    outcomes: dict[str, np.ndarray],
    predictors: dict[str, np.ndarray],
    averaged_predictions: dict[tuple[str, str], np.ndarray],
) -> plt.Figure:
    fig, axes = plt.subplots(len(selected), 4, figsize=(17, 3.4 * len(selected)), constrained_layout=True)
    unit_positions = {int(unit): index for index, unit in enumerate(cohort.rr100_index)}
    for row_index, row in enumerate(selected.itertuples(index=False)):
        unit_position = unit_positions[int(row.rr100_index)]
        y = outcomes["activation_rms_hz"][:, unit_position]
        radial_prediction = averaged_predictions[("activation_rms_hz", "global_plus_radial")][:, unit_position]
        oriented_prediction = averaged_predictions[("activation_rms_hz", "global_plus_radial_plus_orientation")][:, unit_position]
        axis = axes[row_index, 0]
        axis.scatter(y, radial_prediction, s=8, alpha=0.25, color="#0072B2", label="global+radial")
        axis.scatter(y, oriented_prediction, s=8, alpha=0.25, color="#D55E00", label="+orientation")
        limit = max(float(np.nanmax(y)), float(np.nanmax(radial_prediction)), float(np.nanmax(oriented_prediction)), 1e-12)
        axis.plot([0, limit], [0, limit], color="0.55", ls="--")
        axis.set(xlabel="observed activation RMS (Hz)", ylabel="held-out prediction (Hz)", title="Activation prediction")
        axis.legend(frameon=False, fontsize=7)

        axis = axes[row_index, 1]
        orientation_delta = predictors["orientation_delta"][:, unit_position]
        orientation_delta = (orientation_delta - orientation_delta.mean()) / max(float(orientation_delta.std()), 1e-12)
        binned_residual(axis, orientation_delta, y - radial_prediction)
        axis.set(
            xlabel="orientation correction (z-scored)",
            ylabel="activation residual after global+radial (Hz)",
            title="Does the correction track residual response?",
        )

        ssi_y = outcomes["delta_ssi_bits_per_spike"][:, unit_position]
        ssi_radial = averaged_predictions[("delta_ssi_bits_per_spike", "global_plus_radial")][:, unit_position]
        ssi_oriented = averaged_predictions[("delta_ssi_bits_per_spike", "global_plus_radial_plus_orientation")][:, unit_position]
        axis = axes[row_index, 2]
        axis.scatter(ssi_y, ssi_radial, s=8, alpha=0.25, color="#0072B2", label="global+radial")
        axis.scatter(ssi_y, ssi_oriented, s=8, alpha=0.25, color="#D55E00", label="+orientation")
        axis.axhline(0, color="0.6", lw=0.8)
        axis.axvline(0, color="0.6", lw=0.8)
        axis.set(xlabel="observed SSI change", ylabel="held-out prediction", title="SSI prediction kept separate")

        axis = axes[row_index, 3]
        axis.axis("off")
        axis.text(0.02, 0.95, row.selection_role, va="top", fontsize=12, weight="bold")
        axis.text(
            0.02,
            0.76,
            f"RR100 {int(row.rr100_index)}\n"
            f"activation orientation ΔR² = {row.nested_orientation_over_global_plus_radial:+.3f}\n"
            f"SSI orientation ΔR² = {row.ssi_orientation_increment:+.3f}\n"
            f"preferred SF = {row.preferred_sf_cpd:.2f} cpd\n"
            f"preferred TF = {row.extended_tf_center_frequency:.1f} Hz\n"
            f"orientation selectivity = {row.orientation_vector_strength:.2f}\n"
            f"recorded-SF validation r = {row.recorded_sf_curve_r_full_support:.2f}",
            va="top",
            fontsize=10,
            linespacing=1.4,
        )
    fig.suptitle(
        "Audibly selected positive, negative, and dissociation cases\n"
        "Selection uses held-out prediction results; all 3,000 points remain visible",
        fontsize=15,
        weight="bold",
    )
    return fig


def make_gain_figure(gain: pd.DataFrame) -> plt.Figure:
    models = [
        "global_additive", "global_multiplicative", "global_both",
        "radial_additive", "radial_multiplicative", "radial_both",
        "orientation_additive", "orientation_multiplicative", "orientation_both",
    ]
    labels = [
        "global\nadd", "global\nmult", "global\nboth",
        "radial\nadd", "radial\nmult", "radial\nboth",
        "orientation\nadd", "orientation\nmult", "orientation\nboth",
    ]
    unit = gain.groupby(["rr100_index", "model"], as_index=False).delta_r2_over_baseline.median()
    pivot = unit.pivot(index="rr100_index", columns="model", values="delta_r2_over_baseline")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for _, row in pivot.iterrows():
        axes[0].plot(range(len(models)), row[models], color="0.82", lw=0.8)
    median = pivot[models].median(axis=0)
    axes[0].plot(range(len(models)), median, color="black", lw=2.5, marker="o")
    axes[0].axhline(0, color="0.35", ls="--")
    axes[0].set_xticks(range(len(models)), labels, fontsize=8)
    axes[0].set(ylabel="Δ held-out $R^2$ over stabilized-rate baseline", title="Conditional moving-rate prediction")

    summary = []
    for prefix in ("global", "radial", "orientation"):
        additive = pivot[f"{prefix}_additive"]
        multiplicative = pivot[f"{prefix}_multiplicative"]
        both = pivot[f"{prefix}_both"]
        summary.append((prefix, float(np.median(additive)), float(np.median(multiplicative)), float(np.median(both))))
    axes[1].axis("off")
    text = "Median ΔR² over baseline-only\n\n" + "\n".join(
        f"{prefix:11s} additive {additive:+.3f}   multiplicative {multiplicative:+.3f}   both {both:+.3f}"
        for prefix, additive, multiplicative, both in summary
    )
    text += (
        "\n\nThis is a conditional rate model: the stabilized response\n"
        "for the held-out image is supplied as a covariate.\n"
        "It is not a stimulus-only prediction and is not applied to SSI."
    )
    axes[1].text(0.02, 0.96, text, va="top", family="monospace", fontsize=10.5, linespacing=1.45)
    fig.suptitle("Additive versus multiplicative appearance is tested only for mean firing rate", fontsize=14, weight="bold")
    return fig


def main() -> None:
    spectral_dir = validated_spectral_cache_from_environment()
    validate_artifact_not_superseded(TUNING, label="orientation tuning")
    OUT.mkdir(parents=True, exist_ok=True)
    predictors, outcomes, cohort, condition = load_predictors_and_outcomes(spectral_dir)
    split_scores, scores, averaged_predictions = run_outcome_models(predictors, outcomes, cohort, condition)
    increments = build_increment_table(scores, cohort)
    model_summary, increment_summary = population_summaries(scores, increments)
    transform_sensitivity = run_transform_sensitivity(
        predictors, outcomes["activation_rms_hz"], cohort, condition
    )
    gain = run_gain_models(predictors, outcomes, cohort, condition)
    selected = select_examples(increments, cohort)
    geometry = predictor_geometry_table(predictors, cohort)

    split_scores.to_csv(OUT / "per_split_unit_outcome_model_scores.csv", index=False)
    scores.to_csv(OUT / "unit_outcome_model_scores.csv", index=False)
    increments.to_csv(OUT / "unit_outcome_increment_scores.csv", index=False)
    model_summary.to_csv(OUT / "population_model_summary.csv", index=False)
    increment_summary.to_csv(OUT / "population_increment_summary.csv", index=False)
    transform_sensitivity.to_csv(OUT / "activation_transform_sensitivity.csv", index=False)
    gain.to_csv(OUT / "conditional_rate_additive_multiplicative_scores.csv", index=False)
    selected.to_csv(OUT / "selected_population_units.csv", index=False)
    geometry.to_csv(OUT / "predictor_geometry_by_unit.csv", index=False)
    cohort.to_csv(OUT / "population_unit_cohort.csv", index=False)

    prediction_array = np.stack(
        [averaged_predictions[(outcome, model)] for outcome in OUTCOME_LABELS for model in MODEL_ORDER], axis=0
    )
    np.savez_compressed(
        OUT / "population_predictors_outcomes_and_oof_predictions.npz",
        rr100_index=cohort.rr100_index.to_numpy(int),
        image_index=condition.image_index.to_numpy(int),
        trace_index=condition.trace_index.to_numpy(int),
        global_power=predictors["global_power"].astype(np.float32),
        radial_drive=predictors["radial_drive"].astype(np.float32),
        orientation_drive=predictors["orientation_drive"].astype(np.float32),
        orientation_delta=predictors["orientation_delta"].astype(np.float32),
        outcome_names=np.asarray(list(OUTCOME_LABELS), dtype="U48"),
        outcome_values=np.stack([outcomes[outcome] for outcome in OUTCOME_LABELS], axis=0).astype(np.float32),
        model_names=np.asarray(MODEL_ORDER, dtype="U48"),
        averaged_oof_predictions=prediction_array.astype(np.float32),
    )

    figures = [
        ("00_predictor_geometry", make_predictor_geometry_figure(geometry)),
        ("01_population_nested_prediction", make_population_figure(scores, increment_summary)),
        ("02_orientation_increment_by_outcome", make_outcome_increment_figure(increments, increment_summary)),
        ("03_selected_unit_dissociations", make_example_figure(selected, cohort, outcomes, predictors, averaged_predictions)),
        ("04_conditional_rate_gain_form", make_gain_figure(gain)),
    ]
    for name, figure in figures:
        figure.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
        figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    with PdfPages(OUT / "orientation_aware_population_checkpoint.pdf") as pdf:
        for _, figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
    plt.close("all")

    primary = increment_summary[
        increment_summary.outcome.eq("activation_rms_hz")
        & increment_summary.increment.isin(
            ["radial_minus_global", "nested_orientation_over_global_plus_radial"]
        )
    ].set_index("increment")
    transform_population = (
        transform_sensitivity.groupby(["transform", "model"], as_index=False)
        .agg(median_cv_r2=("cv_r2", "median"), median_cv_correlation=("cv_correlation", "median"))
    )
    transform_population.to_csv(OUT / "activation_transform_population_summary.csv", index=False)
    gain_population = (
        gain.groupby("model", as_index=False)
        .agg(
            median_cv_r2=("cv_r2", "median"),
            median_delta_r2_over_baseline=("delta_r2_over_baseline", "median"),
            fraction_positive_delta=("delta_r2_over_baseline", lambda values: float((values > 0).mean())),
        )
    )
    gain_population.to_csv(OUT / "conditional_rate_gain_population_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "orientation_aware_population_checkpoint_complete",
        "scope": {
            "conditions": int(len(condition)),
            "complete_balanced_rounds": sorted(condition.round_index.unique().astype(int).tolist()),
            "images": int(condition.image_index.nunique()),
            "traces": int(condition.trace_index.nunique()),
            "recorded_sf_validated_responsive_units": int(len(cohort)),
        },
        "cross_validation": {
            "contract": "5x5 crossed identity folds; training excludes both the held-out image group and held-out trace group",
            "split_seeds": list(SPLIT_SEEDS),
            "unit_score": "median held-out score across three independently randomized identity-fold assignments",
            "population_interval": "session-cluster bootstrap of the paired unit-level median-split ΔR2",
        },
        "predictors": {
            "global": "sum P over identical supported SFxTF bins",
            "radial": "sum P * direct positive-F0 radial weight",
            "orientation": "sum P * direct positive-F0 SFxorientationxTF weight",
            "orientation_nesting": "orientation factor has mean one at every SFxTF bin; nested models add orientation_drive-radial_drive",
            "primary_transform": "linear power/drive; square-root and log sensitivity controls are saved separately",
            "f0_squared": False,
        },
        "primary_activation_result": {
            "radial_minus_global_median_delta_r2": float(primary.loc["radial_minus_global", "median_delta_cv_r2"]),
            "radial_minus_global_cluster_ci": [
                float(primary.loc["radial_minus_global", "session_cluster_bootstrap_ci_low"]),
                float(primary.loc["radial_minus_global", "session_cluster_bootstrap_ci_high"]),
            ],
            "orientation_over_global_plus_radial_median_delta_r2": float(
                primary.loc["nested_orientation_over_global_plus_radial", "median_delta_cv_r2"]
            ),
            "orientation_over_global_plus_radial_cluster_ci": [
                float(primary.loc["nested_orientation_over_global_plus_radial", "session_cluster_bootstrap_ci_low"]),
                float(primary.loc["nested_orientation_over_global_plus_radial", "session_cluster_bootstrap_ci_high"]),
            ],
        },
        "predictor_distinctness": {
            "median_global_radial_correlation": float(geometry.global_radial_correlation.median()),
            "median_radial_orientation_correlation": float(geometry.radial_orientation_correlation.median()),
            "median_orientation_delta_sd_over_radial_sd": float(
                geometry.orientation_delta_sd_over_radial_sd.median()
            ),
            "median_orientation_over_radial_ratio_iqr": float(
                geometry.orientation_over_radial_ratio_iqr.median()
            ),
        },
        "guardrails": [
            "This is an interim three-round balanced bank, not the predeclared 50,000-movie half-bank endpoint.",
            "SSI is the promoted movie-level bits/spike endpoint and is analyzed as a signed difference, not a ratio.",
            "Mean rate, expected spikes, information numerator, and SSI remain separate outcomes.",
            "Additive/multiplicative language is restricted to a conditional moving-rate model that supplies stabilized rate as a covariate.",
            "Opposite drift directions remain folded in the primary orientation tuning tensor.",
        ],
        "artifacts": {
            "multipage_pdf": str((OUT / "orientation_aware_population_checkpoint.pdf").relative_to(ROOT)),
            "selected_units": str((OUT / "selected_population_units.csv").relative_to(ROOT)),
            "population_increment_summary": str((OUT / "population_increment_summary.csv").relative_to(ROOT)),
            "arrays": str((OUT / "population_predictors_outcomes_and_oof_predictions.npz").relative_to(ROOT)),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

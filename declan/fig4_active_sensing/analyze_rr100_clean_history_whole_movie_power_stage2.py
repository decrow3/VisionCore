#!/usr/bin/env python3
"""Stage 2 development test of whole-movie spectral power against RR100 outcomes.

This script consumes only the validated clean-history spectral cache and the
grating-only tuning export. It freezes image and trace identities for a future
test bank, then evaluates scalar predictors on the remaining identities with
crossed image-and-trace folds. Unequal condition counts are corrected by giving
each image equal total weight.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from declan.fig4_active_sensing.spectral_cache_contract import (
    sha256,
    validate_artifact_not_superseded,
    validate_grating_only_tuning,
    validate_spectral_cache,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECTRAL = ROOT / (
    "outputs/fig4_active_sensing/rr100_clean_history_spectral_cache_rounds000_026_n027_v1"
)
DEFAULT_RESPONSE = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/"
    "rounds_000_026_n027_clean_history_snapshot_v1"
)
DEFAULT_TUNING = ROOT / (
    "outputs/fig4_active_sensing/rr100_grating_only_orientation_tuning_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_clean_history_whole_movie_power_stage2_v1"
)
N_FOLDS = 5
MODEL_LABELS = {
    "whole_movie_supported_dynamic_power": "whole-movie supported\ndynamic power",
    "spatial_temporal_direct_f0_power": "spatial × temporal\ndirect-F0 power",
    "spatial_orientation_temporal_direct_f0_power": "spatial × orientation × temporal\ndirect-F0 power",
    "squared_spatial_temporal_tuning_power": "squared spatial × temporal\ntuning power",
    "simple_image_and_dynamic_energy_controls": "image and dynamic-energy\ncontrols",
    "radial_power_plus_image_controls": "spatial × temporal power\nplus image controls",
    "oriented_power_plus_image_controls": "orientation-aware power\nplus image controls",
}
MODEL_ORDER = list(MODEL_LABELS)
OUTCOME_LABELS = {
    "activation_rms_hz": "response-modulation magnitude\nroot mean square (Hz)",
    "activation_mean_abs_hz": "response-modulation magnitude\nmean absolute change (Hz)",
    "delta_mean_rate_hz": "signed mean-rate change (Hz)",
    "delta_expected_spikes": "expected-spike change",
    "delta_information_bits_spikes": "information-numerator change",
    "delta_ssi_bits_per_spike": "SSI change (bits/spike)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-dir", type=Path, default=DEFAULT_SPECTRAL)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--reserved-fraction", type=float, default=0.2)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def freeze_identity_split(
    image: np.ndarray, trace: np.ndarray, seed: int, reserved_fraction: float
) -> tuple[np.ndarray, pd.DataFrame]:
    if not 0.0 < reserved_fraction < 0.5:
        raise ValueError("reserved_fraction must lie between zero and one half")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    development = np.ones(len(image), dtype=bool)
    for kind, values, offset in (("image", image, 0), ("trace", trace, 104729)):
        unique = np.unique(values)
        local_rng = np.random.default_rng(seed + offset)
        shuffled = unique.copy()
        local_rng.shuffle(shuffled)
        n_reserved = max(1, int(np.ceil(reserved_fraction * len(unique))))
        reserved = set(shuffled[:n_reserved].astype(int))
        development &= ~np.isin(values, list(reserved))
        for value in unique:
            rows.append({
                "identity_type": kind,
                "identity": int(value),
                "split": "reserved_final_test" if int(value) in reserved else "development",
                "selection_seed": int(seed + offset),
            })
    return development, pd.DataFrame(rows)


def assign_folds(identities: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(identities)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {int(value): int(index % N_FOLDS) for index, value in enumerate(shuffled)}
    return np.asarray([mapping[int(value)] for value in identities], dtype=int)


def build_grating_weights(
    tuning: dict[str, np.ndarray], sf: np.ndarray, tf: np.ndarray, fourier_orientation_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate clean held-out grating predictions onto the movie grid."""
    source_sf = tuning["measured_sf_cpd"].astype(float)
    source_tf = tuning["measured_tf_hz"].astype(float)
    source_orientation = tuning["measured_grating_orientation_deg"].astype(float)
    positive = np.maximum(tuning["heldout_chosen_prediction_f0_hz"].astype(float), 0.0)
    query_sf, query_tf = np.meshgrid(np.log2(sf), np.log2(tf), indexing="ij")
    query = np.column_stack((query_sf.ravel(), query_tf.ravel()))
    interpolated = np.empty((len(positive), len(sf), len(tf), len(source_orientation)), dtype=float)
    for unit in range(len(positive)):
        for orientation in range(len(source_orientation)):
            function = RegularGridInterpolator(
                (np.log2(source_sf), np.log2(source_tf)),
                positive[unit, :, :, orientation],
                bounds_error=True,
            )
            interpolated[unit, :, :, orientation] = function(query).reshape(len(sf), len(tf))
    interpolated = interpolated.transpose(0, 2, 1, 3)
    theta = np.deg2rad(source_orientation)
    source_design = np.column_stack((np.ones(len(theta)), np.cos(2 * theta), np.sin(2 * theta)))
    coefficients = np.einsum("ko,utso->utsk", np.linalg.pinv(source_design), interpolated)
    grating_orientation = np.mod(90.0 - fourier_orientation_deg, 180.0)
    target_theta = np.deg2rad(grating_orientation)
    target_design = np.column_stack(
        (np.ones(len(target_theta)), np.cos(2 * target_theta), np.sin(2 * target_theta))
    )
    raw_oriented = np.maximum(np.einsum("utsk,ok->utso", coefficients, target_design), 0.0)
    radial = np.maximum(interpolated.mean(axis=-1), 0.0)
    raw_mean = raw_oriented.mean(axis=-1, keepdims=True)
    factor = np.divide(raw_oriented, raw_mean, out=np.ones_like(raw_oriented), where=raw_mean > 1e-12)
    oriented = radial[..., None] * factor
    if not np.allclose(oriented.mean(axis=-1), radial, rtol=1e-9, atol=1e-10):
        raise ValueError("Orientation-aware weights do not preserve the radial direct-F0 marginal")
    return radial, oriented


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def image_equal_weights(image: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(image, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse]
    return weights / np.mean(weights)


def weighted_fit_predict(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    *,
    nonnegative_primary: bool,
) -> tuple[np.ndarray, float]:
    train_z, test_z = standardize(train_x, test_x)
    x_train = np.column_stack((np.ones(len(train_z)), train_z))
    x_test = np.column_stack((np.ones(len(test_z)), test_z))
    root_weight = np.sqrt(train_weight)
    coefficients = np.linalg.lstsq(x_train * root_weight[:, None], train_y * root_weight, rcond=None)[0]
    if nonnegative_primary and coefficients[1] < 0:
        if x_train.shape[1] == 2:
            coefficients = np.asarray([np.average(train_y, weights=train_weight), 0.0])
        else:
            reduced = np.delete(x_train, 1, axis=1)
            reduced_coefficients = np.linalg.lstsq(
                reduced * root_weight[:, None], train_y * root_weight, rcond=None
            )[0]
            coefficients = np.insert(reduced_coefficients, 1, 0.0)
    return x_test @ coefficients, float(coefficients[1])


def score(y: np.ndarray, prediction: np.ndarray, image: np.ndarray) -> dict[str, float]:
    weight = image_equal_weights(image)
    mean = float(np.average(y, weights=weight))
    denominator = float(np.sum(weight * (y - mean) ** 2))
    numerator = float(np.sum(weight * (y - prediction) ** 2))
    correlation = float(np.corrcoef(y, prediction)[0, 1]) if np.std(prediction) > 0 else np.nan
    return {
        "image_balanced_cv_r2": 1.0 - numerator / denominator if denominator > 0 else np.nan,
        "cv_correlation": correlation,
        "image_balanced_mae": float(np.average(np.abs(y - prediction), weights=weight)),
    }


def crossed_predict_many(
    features: np.ndarray,
    outcomes: dict[str, np.ndarray],
    image: np.ndarray,
    trace: np.ndarray,
    image_fold: np.ndarray,
    trace_fold: np.ndarray,
    *,
    nonnegative_outcomes: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    first_outcome = next(iter(outcomes.values()))
    n_units = first_outcome.shape[1]
    predictions = {name: np.full_like(values, np.nan, dtype=float) for name, values in outcomes.items()}
    slopes = {
        name: np.full((N_FOLDS, N_FOLDS, n_units), np.nan, dtype=float) for name in outcomes
    }
    for image_group in range(N_FOLDS):
        for trace_group in range(N_FOLDS):
            test = (image_fold == image_group) & (trace_fold == trace_group)
            train = (image_fold != image_group) & (trace_fold != trace_group)
            if not np.any(test):
                continue
            train_weight = image_equal_weights(image[train])
            if features.ndim == 2:
                expanded = np.broadcast_to(
                    features[:, None, :], (len(features), n_units, features.shape[1])
                )
            else:
                expanded = features
            train_features = expanded[train]
            test_features = expanded[test]
            center = np.nanmean(train_features, axis=0)
            scale = np.nanstd(train_features, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
            train_z = (train_features - center[None, :, :]) / scale[None, :, :]
            test_z = (test_features - center[None, :, :]) / scale[None, :, :]
            x_train = np.concatenate(
                (np.ones((len(train_z), n_units, 1)), train_z), axis=2
            )
            x_test = np.concatenate(
                (np.ones((len(test_z), n_units, 1)), test_z), axis=2
            )
            cross_product = np.einsum(
                "nup,n,nuq->upq", x_train, train_weight, x_train, optimize=True
            )
            inverse_cross_product = np.linalg.pinv(cross_product)
            for outcome_name, outcome_values in outcomes.items():
                cross_target = np.einsum(
                    "nup,n,nu->up", x_train, train_weight, outcome_values[train], optimize=True
                )
                coefficients = np.einsum(
                    "upq,uq->up", inverse_cross_product, cross_target, optimize=True
                )
                if outcome_name not in nonnegative_outcomes:
                    predictions[outcome_name][test] = np.einsum(
                        "nup,up->nu", x_test, coefficients, optimize=True
                    )
                    slopes[outcome_name][image_group, trace_group] = coefficients[:, 1]
                    continue
                constrained = coefficients[:, 1] < 0
                if np.any(constrained):
                    reduced = np.delete(x_train[:, constrained, :], 1, axis=2)
                    reduced_cross_product = np.einsum(
                        "nup,n,nuq->upq", reduced, train_weight, reduced, optimize=True
                    )
                    reduced_cross_target = np.einsum(
                        "nup,n,nu->up", reduced, train_weight,
                        outcome_values[train][:, constrained], optimize=True
                    )
                    reduced_coefficients = np.einsum(
                        "upq,uq->up", np.linalg.pinv(reduced_cross_product), reduced_cross_target,
                        optimize=True,
                    )
                    coefficients[constrained] = np.insert(reduced_coefficients, 1, 0.0, axis=1)
                predictions[outcome_name][test] = np.einsum(
                    "nup,up->nu", x_test, coefficients, optimize=True
                )
                slopes[outcome_name][image_group, trace_group] = coefficients[:, 1]
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Crossed folds did not predict every development condition and outcome")
    return predictions, slopes


def session_balanced_summary(scores: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (outcome, model), group in scores.groupby(["outcome", "model"], sort=False):
        per_unit = group.groupby(["rr100_index", "session"], as_index=False).agg(
            image_balanced_cv_r2=("image_balanced_cv_r2", "mean"),
            cv_correlation=("cv_correlation", "mean"),
            image_balanced_mae=("image_balanced_mae", "mean"),
        )
        sessions = sorted(per_unit.session.unique())
        point = per_unit.groupby("session").image_balanced_cv_r2.mean().mean()
        draws = np.empty(n_bootstrap, dtype=float)
        by_session = {session: per_unit.loc[per_unit.session.eq(session)] for session in sessions}
        for draw in range(n_bootstrap):
            sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True)
            session_means = []
            for session in sampled_sessions:
                values = by_session[session].image_balanced_cv_r2.to_numpy(float)
                session_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
            draws[draw] = float(np.mean(session_means))
        rows.append({
            "outcome": outcome,
            "model": model,
            "n_units": int(len(per_unit)),
            "n_sessions": int(len(sessions)),
            "session_balanced_mean_image_balanced_cv_r2": float(point),
            "hierarchical_bootstrap_low": float(np.quantile(draws, 0.025)),
            "hierarchical_bootstrap_high": float(np.quantile(draws, 0.975)),
            "session_balanced_mean_cv_correlation": float(per_unit.groupby("session").cv_correlation.mean().mean()),
            "median_unit_image_balanced_cv_r2": float(per_unit.image_balanced_cv_r2.median()),
            "fraction_units_positive_cv_r2": float(np.mean(per_unit.image_balanced_cv_r2 > 0)),
        })
    return pd.DataFrame(rows)


def paired_model_comparisons(scores: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    comparisons = {
        "orientation contribution beyond spatial and temporal tuning": (
            "spatial_orientation_temporal_direct_f0_power", "spatial_temporal_direct_f0_power"
        ),
        "orientation-aware power beyond image and energy controls": (
            "oriented_power_plus_image_controls", "simple_image_and_dynamic_energy_controls"
        ),
        "spatial and temporal power beyond image and energy controls": (
            "radial_power_plus_image_controls", "simple_image_and_dynamic_energy_controls"
        ),
        "spatial and temporal tuning versus total supported dynamic power": (
            "spatial_temporal_direct_f0_power", "whole_movie_supported_dynamic_power"
        ),
    }
    per_unit = scores.groupby(
        ["outcome", "rr100_index", "session", "model"], as_index=False
    ).image_balanced_cv_r2.mean()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for outcome in OUTCOME_LABELS:
        outcome_frame = per_unit.loc[per_unit.outcome.eq(outcome)]
        pivot = outcome_frame.pivot(
            index=["rr100_index", "session"], columns="model", values="image_balanced_cv_r2"
        ).reset_index()
        for label, (full_model, reference_model) in comparisons.items():
            frame = pivot[["rr100_index", "session", full_model, reference_model]].dropna().copy()
            frame["difference"] = frame[full_model] - frame[reference_model]
            sessions = sorted(frame.session.unique())
            grouped = {session: frame.loc[frame.session.eq(session), "difference"].to_numpy(float) for session in sessions}
            point = float(np.mean([np.mean(grouped[session]) for session in sessions]))
            draws = np.empty(n_bootstrap, dtype=float)
            for draw in range(n_bootstrap):
                sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True)
                session_means = []
                for session in sampled_sessions:
                    values = grouped[session]
                    session_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
                draws[draw] = float(np.mean(session_means))
            rows.append({
                "outcome": outcome, "comparison": label,
                "full_model": full_model, "reference_model": reference_model,
                "session_balanced_mean_cv_r2_difference": point,
                "hierarchical_bootstrap_low": float(np.quantile(draws, 0.025)),
                "hierarchical_bootstrap_high": float(np.quantile(draws, 0.975)),
                "bootstrap_fraction_positive": float(np.mean(draws > 0)),
                "fraction_units_positive": float(np.mean(frame.difference > 0)),
                "n_units": int(len(frame)), "n_sessions": int(len(sessions)),
            })
    return pd.DataFrame(rows)


def plot_summary(
    scores: pd.DataFrame, population: pd.DataFrame, comparisons: pd.DataFrame,
    condition: pd.DataFrame, out: Path
) -> None:
    primary = scores.loc[scores.outcome.eq("activation_rms_hz")].groupby(
        ["rr100_index", "session", "model"], as_index=False
    ).image_balanced_cv_r2.mean()
    pivot = primary.pivot(index="rr100_index", columns="model", values="image_balanced_cv_r2")
    summary = population.loc[population.outcome.eq("activation_rms_hz")].set_index("model").loc[MODEL_ORDER]
    figure, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    counts = condition.groupby("image_index").size()
    axes[0, 0].bar(counts.index, counts.values, color="#0072B2")
    axes[0, 0].axhline(counts.mean(), color="0.25", ls="--", label="development-image mean")
    axes[0, 0].set(
        xlabel="development image identity", ylabel="number of development conditions",
        title="Development conditions per image\n(equal-image weights correct this imbalance)",
    )
    axes[0, 0].legend(frameon=False)

    x = np.arange(len(MODEL_ORDER))
    point = summary.session_balanced_mean_image_balanced_cv_r2.to_numpy(float)
    low = summary.hierarchical_bootstrap_low.to_numpy(float)
    high = summary.hierarchical_bootstrap_high.to_numpy(float)
    axes[0, 1].errorbar(x, point, yerr=np.vstack((point - low, high - point)), fmt="o", color="#D55E00")
    axes[0, 1].axhline(0, color="0.35", ls="--")
    axes[0, 1].set_xticks(x, [MODEL_LABELS[value] for value in MODEL_ORDER], rotation=35, ha="right")
    axes[0, 1].set(
        ylabel="session-balanced mean held-out R²",
        title="Held-out prediction of RMS response modulation\n(unseen images and eye traces)",
    )

    radial = "spatial_temporal_direct_f0_power"
    oriented = "spatial_orientation_temporal_direct_f0_power"
    limits = [float(min(pivot[radial].min(), pivot[oriented].min())), float(max(pivot[radial].max(), pivot[oriented].max()))]
    axes[0, 2].scatter(pivot[radial], pivot[oriented], color="#009E73", alpha=0.8)
    axes[0, 2].plot(limits, limits, color="0.4", ls="--")
    axes[0, 2].set(
        xlabel="spatial × temporal direct-F0 held-out R²",
        ylabel="orientation-aware direct-F0 held-out R²",
        title="Orientation-aware versus orientation-collapsed power\n(61 validated units)",
    )

    outcome_models = population[population.model.eq("spatial_orientation_temporal_direct_f0_power")].copy()
    axes[1, 0].bar(
        np.arange(len(outcome_models)), outcome_models.session_balanced_mean_image_balanced_cv_r2,
        color=["#D55E00" if name == "activation_rms_hz" else "#7F7F7F" for name in outcome_models.outcome],
    )
    axes[1, 0].axhline(0, color="0.35", ls="--")
    axes[1, 0].set_xticks(
        np.arange(len(outcome_models)), [OUTCOME_LABELS[name] for name in outcome_models.outcome],
        rotation=35, ha="right",
    )
    axes[1, 0].set(
        ylabel="session-balanced mean held-out R²",
        title="Held-out prediction from orientation-aware power\n(each neural summary analyzed separately)",
    )

    differences = pivot[oriented] - pivot[radial]
    orientation_comparison = comparisons.loc[
        comparisons.outcome.eq("activation_rms_hz")
        & comparisons.comparison.eq("orientation contribution beyond spatial and temporal tuning")
    ].iloc[0]
    axes[1, 1].hist(differences, bins=15, color="#CC79A7", alpha=0.9)
    axes[1, 1].axvline(0, color="0.35", ls="--")
    axes[1, 1].set(
        xlabel="orientation-aware minus spatial × temporal held-out R²",
        ylabel="units",
        title=(
            "Orientation contribution to RMS prediction\n"
            f"session-balanced ΔR²={orientation_comparison.session_balanced_mean_cv_r2_difference:+.3f}; "
            f"95% interval [{orientation_comparison.hierarchical_bootstrap_low:+.3f}, "
            f"{orientation_comparison.hierarchical_bootstrap_high:+.3f}]"
        ),
    )

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.98,
        "Interpretation guardrails\n\n"
        "• All displayed predictions are out of fold for both image and eye-trace identity.\n\n"
        "• Each image has equal total weight despite unequal condition counts.\n\n"
        "• Reserved final-test images and traces were not fitted, scored, or used for this figure.\n\n"
        "• These are whole-movie scalar tests. They cannot explain where an activation map sharpens or why SSI changes.",
        va="top", fontsize=11, wrap=True,
    )
    figure.suptitle(
        "Development Stage 2: can whole-movie Fourier power predict FEM-driven neural changes?",
        fontsize=16, weight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spectral_validation = validate_spectral_cache(args.spectral_dir, require_rounds=27)
    tuning_path = validate_artifact_not_superseded(
        args.tuning_dir / "grating_only_orientation_tuning.npz", label="grating-only tuning"
    )
    tuning_validation = validate_grating_only_tuning(tuning_path)
    validate_artifact_not_superseded(args.response_dir, label="clean-history response snapshot")

    with np.load(args.spectral_dir / "condition_spectra.npz", allow_pickle=False) as archive:
        spectral = {key: np.asarray(archive[key]) for key in archive.files}
    condition = pd.read_csv(args.response_dir / "condition_index.csv").sort_values("matrix_row_index").reset_index(drop=True)
    for key in ("matrix_row_index", "image_index", "trace_index", "round_index"):
        if not np.array_equal(condition[key].to_numpy(int), spectral[key].astype(int)):
            raise ValueError(f"Response and spectral rows disagree for {key}")
    with np.load(tuning_path, allow_pickle=False) as archive:
        tuning = {key: np.asarray(archive[key]) for key in archive.files}
    quality = pd.read_csv(args.tuning_dir / "grating_tuning_fit_and_recorded_validation.csv")
    cohort = quality.loc[quality.recorded_validation_pass.astype(bool)].sort_values("rr100_index").copy()
    if len(cohort) != 61:
        raise ValueError(f"Expected 61 recorded-spatial-frequency-validated units, found {len(cohort)}")
    session_table = pd.read_csv(
        ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_extended_tf_f0_analysis_v1/extended_f0_fit_unit_summary.csv"
    )[["rr100_index", "session"]]
    cohort = cohort.merge(session_table, on="rr100_index", how="left", validate="one_to_one")
    units = cohort.rr100_index.to_numpy(int)

    image_all = condition.image_index.to_numpy(int)
    trace_all = condition.trace_index.to_numpy(int)
    development_mask, split_table = freeze_identity_split(
        image_all, trace_all, args.seed, args.reserved_fraction
    )
    split_table.to_csv(args.out_dir / "frozen_image_and_trace_identity_split.csv", index=False)
    condition = condition.loc[development_mask].reset_index(drop=True)
    selected_rows = condition.matrix_row_index.to_numpy(int)
    image = condition.image_index.to_numpy(int)
    trace = condition.trace_index.to_numpy(int)

    sf_all = 0.5 * (spectral["sf_edges_cpd"][:-1] + spectral["sf_edges_cpd"][1:])
    sf_mask = (
        spectral["spatial_frequency_has_support"].astype(bool)
        & (sf_all >= float(tuning["measured_sf_cpd"].min()))
        & (sf_all <= float(tuning["measured_sf_cpd"].max()))
    )
    tf_mask = (
        (spectral["tf_hz"] >= float(tuning["measured_tf_hz"].min()))
        & (spectral["tf_hz"] <= float(tuning["measured_tf_hz"].max()))
    )
    sf = sf_all[sf_mask]
    tf = spectral["tf_hz"][tf_mask]
    fourier_orientation = 0.5 * (
        spectral["orientation_edges_deg"][:-1] + spectral["orientation_edges_deg"][1:]
    )
    radial_weight_all, oriented_weight_all = build_grating_weights(tuning, sf, tf, fourier_orientation)
    radial_weight = radial_weight_all[units]
    oriented_weight = oriented_weight_all[units]
    radial_power = spectral["radial_power"][selected_rows][:, tf_mask][:, :, sf_mask].astype(float)
    oriented_power = spectral["orientation_power"][selected_rows][:, tf_mask][:, :, sf_mask, :].astype(float)
    if not np.allclose(oriented_power.sum(axis=-1), radial_power, rtol=2e-5, atol=1e-3):
        raise ValueError("Orientation-resolved power does not sum to radial power")
    global_amplitude = np.sqrt(np.maximum(radial_power.sum(axis=(1, 2)), 0.0))
    radial_drive = np.einsum("ctf,utf->cu", radial_power, radial_weight)
    oriented_drive = np.einsum("ctfo,utfo->cu", oriented_power, oriented_weight)
    squared_drive = np.sqrt(np.maximum(np.einsum("ctf,utf->cu", radial_power, radial_weight**2), 0.0))

    with np.load(args.spectral_dir / "stabilized_input_predictors_by_image.npz", allow_pickle=False) as archive:
        baseline_input = {key: np.asarray(archive[key]) for key in archive.files}
    baseline_position = {int(value): index for index, value in enumerate(baseline_input["image_index"])}
    baseline_rows = np.asarray([baseline_position[int(value)] for value in image], dtype=int)
    static = baseline_input["static_mean_sd_rms_contrast"][baseline_rows].astype(float)
    common_controls = np.column_stack((global_amplitude, static))
    radial_features = radial_drive[:, :, None]
    oriented_features = oriented_drive[:, :, None]
    squared_features = squared_drive[:, :, None]
    global_features = global_amplitude[:, None]
    control_features = common_controls
    radial_control_features = np.concatenate(
        (radial_features, np.broadcast_to(common_controls[:, None, :], (len(condition), len(units), common_controls.shape[1]))),
        axis=2,
    )
    oriented_control_features = np.concatenate(
        (oriented_features, np.broadcast_to(common_controls[:, None, :], (len(condition), len(units), common_controls.shape[1]))),
        axis=2,
    )
    features = {
        "whole_movie_supported_dynamic_power": global_features,
        "spatial_temporal_direct_f0_power": radial_features,
        "spatial_orientation_temporal_direct_f0_power": oriented_features,
        "squared_spatial_temporal_tuning_power": squared_features,
        "simple_image_and_dynamic_energy_controls": control_features,
        "radial_power_plus_image_controls": radial_control_features,
        "oriented_power_plus_image_controls": oriented_control_features,
    }

    moving_rate = np.load(args.response_dir / "moving_mean_rate_hz.npy", mmap_mode="r")[selected_rows][:, units].astype(float)
    moving_spikes = np.load(args.response_dir / "moving_expected_spikes.npy", mmap_mode="r")[selected_rows][:, units].astype(float)
    moving_information = np.load(args.response_dir / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")[selected_rows][:, units].astype(float)
    moving_ssi = np.load(args.response_dir / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")[selected_rows][:, units].astype(float)
    with np.load(args.response_dir / "stabilized_by_image_sufficient_statistics.npz", allow_pickle=False) as archive:
        response_baseline = {key: np.asarray(archive[key]) for key in archive.files}
    response_baseline_position = {int(value): index for index, value in enumerate(response_baseline["image_index"])}
    response_baseline_rows = np.asarray([response_baseline_position[int(value)] for value in image], dtype=int)
    baseline_rate = response_baseline["mean_rate_hz"][response_baseline_rows][:, units].astype(float)
    baseline_spikes = response_baseline["expected_spikes"][response_baseline_rows][:, units].astype(float)
    baseline_information = response_baseline["information_numerator_bits_spikes"][response_baseline_rows][:, units].astype(float)
    baseline_ssi = response_baseline["movie_ssi_bits_per_spike"][response_baseline_rows][:, units].astype(float)
    outcomes = {
        "activation_rms_hz": np.load(
            args.response_dir / "moving_temporal_rms_delta_from_stabilized_hz.npy", mmap_mode="r"
        )[selected_rows][:, units].astype(float),
        "activation_mean_abs_hz": np.load(
            args.response_dir / "moving_temporal_mean_abs_delta_from_stabilized_hz.npy", mmap_mode="r"
        )[selected_rows][:, units].astype(float),
        "delta_mean_rate_hz": moving_rate - baseline_rate,
        "delta_expected_spikes": moving_spikes - baseline_spikes,
        "delta_information_bits_spikes": moving_information - baseline_information,
        "delta_ssi_bits_per_spike": moving_ssi - baseline_ssi,
    }

    score_rows: list[dict[str, object]] = []
    predictions_to_save: dict[str, np.ndarray] = {}
    for split_index, split_seed in enumerate((args.seed + 11, args.seed + 101, args.seed + 1009)):
        image_fold = assign_folds(image, split_seed)
        trace_fold = assign_folds(trace, split_seed + 1000003)
        for model_name in MODEL_ORDER:
            model_predictions, model_slopes = crossed_predict_many(
                features[model_name], outcomes, image, trace, image_fold, trace_fold,
                nonnegative_outcomes={"activation_rms_hz", "activation_mean_abs_hz"},
            )
            for outcome_name, outcome in outcomes.items():
                prediction = model_predictions[outcome_name]
                outcome_slopes = model_slopes[outcome_name]
                if split_index == 0:
                    predictions_to_save[f"{outcome_name}__{model_name}"] = prediction.astype(np.float32)
                for unit_position, unit in enumerate(units):
                    metrics = score(outcome[:, unit_position], prediction[:, unit_position], image)
                    score_rows.append({
                        "split_index": int(split_index), "split_seed": int(split_seed),
                        "rr100_index": int(unit), "session": cohort.iloc[unit_position].session,
                        "outcome": outcome_name, "model": model_name,
                        "median_primary_standardized_slope": float(np.nanmedian(outcome_slopes[:, :, unit_position])),
                        **metrics,
                    })
    scores = pd.DataFrame(score_rows)
    population = session_balanced_summary(scores, args.n_bootstrap, args.seed + 5003)
    comparisons = paired_model_comparisons(scores, args.n_bootstrap, args.seed + 7001)
    scores.to_csv(args.out_dir / "unit_level_crossed_identity_prediction_scores.csv", index=False)
    population.to_csv(args.out_dir / "session_balanced_population_prediction_summary.csv", index=False)
    comparisons.to_csv(args.out_dir / "paired_model_comparison_summary.csv", index=False)
    cohort.to_csv(args.out_dir / "recorded_spatial_frequency_validated_unit_cohort.csv", index=False)
    condition.to_csv(args.out_dir / "development_condition_index.csv", index=False)
    np.savez_compressed(
        args.out_dir / "development_predictors_and_first_split_predictions.npz",
        matrix_row_index=selected_rows,
        image_index=image,
        trace_index=trace,
        rr100_index=units,
        spatial_frequency_cycles_per_degree=sf,
        temporal_frequency_hz=tf,
        fourier_orientation_deg=fourier_orientation,
        global_supported_dynamic_power_amplitude=global_amplitude.astype(np.float32),
        radial_direct_f0_power=radial_drive.astype(np.float32),
        oriented_direct_f0_power=oriented_drive.astype(np.float32),
        squared_tuning_power_amplitude=squared_drive.astype(np.float32),
        static_image_controls=static.astype(np.float32),
        **predictions_to_save,
    )
    figure_base = args.out_dir / "stage2_whole_movie_power_prediction_summary"
    plot_summary(scores, population, comparisons, condition, figure_base)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_development_stage2_whole_movie_scalar_prediction_complete",
        "tier": "engineering_development_only_not_confirmatory",
        "scope": {
            "development_conditions": int(len(condition)),
            "development_images": int(condition.image_index.nunique()),
            "development_traces": int(condition.trace_index.nunique()),
            "reserved_images": int((split_table.identity_type.eq("image") & split_table.split.eq("reserved_final_test")).sum()),
            "reserved_traces": int((split_table.identity_type.eq("trace") & split_table.split.eq("reserved_final_test")).sum()),
            "units": int(len(units)), "sessions": int(cohort.session.nunique()),
        },
        "contracts": {
            "final_test_reservation": "identities labeled reserved_final_test are excluded from all fitting, scoring, and figures",
            "cross_validation": "three deterministic repetitions of 5x5 crossed image-and-trace folds; training excludes both test identity groups",
            "image_imbalance_control": "each development image has equal total weight in fitting and R2/MAE scoring",
            "primary_predictor": "sum of whole-movie SFxFourier-orientationxTF power times clean held-out direct-positive-F0 grating prediction",
            "orientation_conversion": "Fourier wavevector orientation maps to grating-bar orientation as (90 degrees - wavevector) modulo 180",
            "unsupported_frequency_bins": "excluded using the raw cache support mask; visualization smoothing is not consumed",
            "magnitude_slope": "primary feature constrained nonnegative for RMS and mean-absolute response-modulation outcomes only",
            "signed_outcomes": "signed slopes allowed; direct nonnegative-power prediction is diagnostic rather than a sign-generating mechanism",
            "mechanistic_limit": "whole-movie scalars cannot explain activation-map location or map-derived SSI",
        },
        "validation": {
            "spectral_cache": spectral_validation,
            "grating_only_tuning": tuning_validation,
            "orientation_power_reproduces_radial": True,
            "orientation_weights_preserve_radial_marginal": True,
            "all_development_rows_predicted": True,
        },
        "sources": {
            "spectral_arrays": identity(args.spectral_dir / "condition_spectra.npz"),
            "response_conditions": identity(args.response_dir / "condition_index.csv"),
            "grating_only_tuning": identity(tuning_path),
            "runner": identity(Path(__file__)),
        },
        "artifacts": {
            "figure": figure_base.with_suffix(".pdf").name,
            "identity_split": "frozen_image_and_trace_identity_split.csv",
            "unit_scores": "unit_level_crossed_identity_prediction_scores.csv",
            "population_summary": "session_balanced_population_prediction_summary.csv",
            "paired_model_comparisons": "paired_model_comparison_summary.csv",
            "predictors_and_predictions": "development_predictors_and_first_split_predictions.npz",
        },
        "next_checkpoint": "inspect the Stage 2 scalar result, then begin Stage 3 receptive-field and spatial-coordinate calibration without opening reserved identities",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "primary": population.loc[population.outcome.eq("activation_rms_hz")].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()

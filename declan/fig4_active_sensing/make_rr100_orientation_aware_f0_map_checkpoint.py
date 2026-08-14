#!/usr/bin/env python3
"""Build the map-first orientation-aware F0 routing checkpoint.

This checkpoint deliberately stops before population prediction.  It builds a
smooth, first-orientation-harmonic approximation to the measured positive-F0
grating tensor, validates that approximation on held-out SFxTF cells, and then
shows one fixed retinal movie filtered through several audibly selected units.

The routing weight is F0 itself, not F0 squared.  Under an energy-model reading,
F0 is already proportional to contrast power passed by the unit.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold

from declan.fig4_active_sensing.spectral_cache_contract import (
    validate_artifact_not_superseded,
    validated_spectral_cache_from_environment,
)


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
OLD_NATIVE = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1"
NEW = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_native_extended_tf_32_60_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
INPUT_CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_routing_input_checkpoint_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"

SF_GRID = np.asarray([1, 2**0.5, 2, 2**1.5, 4, 2**2.5, 8, 2**3.5], dtype=float)
OLD_TF = np.asarray([0.5, 2**-0.5, 1, 2**0.5, 2, 2**1.5, 4, 2**2.5, 8, 2**3.5, 16, 2**4.5, 32], dtype=float)
EXT_TF = np.asarray([34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56], dtype=float)
TF_GRID = np.concatenate([OLD_TF, EXT_TF])
GRATING_ORIENTATIONS = np.asarray([0.0, 45.0, 90.0, 135.0])
EPS = np.finfo(float).tiny


def centered_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-float(gamma) * squared)


def solve_kernel_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    kernel = rbf_kernel(train_x, train_x, gamma)
    kernel.flat[:: len(kernel) + 1] += float(alpha)
    coefficients = np.linalg.solve(kernel, train_y)
    return rbf_kernel(query_x, train_x, gamma) @ coefficients


def collect_native_folded(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows.drop_duplicates(["session", "condition_id", "rr100_index"], keep="last").copy()
    rows = rows.drop(columns=["blank_rate_hz", "mean_rate_above_blank_hz"], errors="ignore")
    blank = (
        rows[rows.condition_kind.eq("gray_blank")]
        .groupby(["session", "rr100_index"], as_index=False)
        .mean_rate_hz.mean()
        .rename(columns={"mean_rate_hz": "blank_rate_hz"})
    )
    dynamic = rows[rows.condition_kind.eq("drifting_grating")].merge(
        blank, on=["session", "rr100_index"], how="left", validate="many_to_one"
    )
    dynamic["signed_f0_hz"] = dynamic.mean_rate_hz - dynamic.blank_rate_hz
    dynamic["temporal_hz"] = dynamic.signed_temporal_hz.abs()
    phase_folded = (
        dynamic.groupby(
            ["session", "rr100_index", "orientation_deg", "spatial_cpd", "temporal_hz", "signed_temporal_hz"],
            as_index=False,
        )
        .signed_f0_hz.mean()
    )
    return (
        phase_folded.groupby(
            ["session", "rr100_index", "orientation_deg", "spatial_cpd", "temporal_hz"],
            as_index=False,
        )
        .signed_f0_hz.mean()
    )


def build_measured_tensor() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    old = collect_native_folded(OLD_NATIVE / "native_condition_unit_summary.csv")
    sf_supported = np.any(
        np.isclose(old.spatial_cpd.to_numpy(float)[:, None], SF_GRID[None, :]), axis=1
    )
    tf_supported = np.any(
        np.isclose(old.temporal_hz.to_numpy(float)[:, None], OLD_TF[None, :]), axis=1
    )
    old = old[sf_supported & tf_supported][
        ["rr100_index", "spatial_cpd", "temporal_hz", "orientation_deg", "signed_f0_hz"]
    ].copy()
    new = collect_native_folded(NEW / "native_condition_unit_summary.csv")
    new = new[new.temporal_hz.isin(EXT_TF)][
        ["rr100_index", "spatial_cpd", "temporal_hz", "orientation_deg", "signed_f0_hz"]
    ]
    points = pd.concat([old, new], ignore_index=True)
    summarized = pd.read_csv(OLD / "direction_folded_signed_f0_points.csv")
    preferred = summarized[["rr100_index", "preferred_orientation_deg"]].drop_duplicates()
    raw_preferred = old.merge(preferred, on="rr100_index", how="inner", validate="many_to_one")
    raw_preferred = raw_preferred[
        np.isclose(raw_preferred.orientation_deg, raw_preferred.preferred_orientation_deg)
    ]
    reproduction = raw_preferred.merge(
        summarized[["rr100_index", "spatial_cpd", "temporal_hz", "signed_f0_hz"]],
        on=["rr100_index", "spatial_cpd", "temporal_hz"],
        suffixes=("_raw", "_summary"),
        validate="one_to_one",
    )
    source_audit = {
        "preferred_orientation_points_compared": int(len(reproduction)),
        "maximum_absolute_raw_vs_summary_f0_difference_hz": float(
            np.max(np.abs(reproduction.signed_f0_hz_raw - reproduction.signed_f0_hz_summary))
        ),
    }
    units = np.sort(points.rr100_index.unique().astype(int))
    if not np.array_equal(units, np.arange(100)):
        raise ValueError("Expected complete RR100 identity axis")
    signed = np.full((len(units), len(SF_GRID), len(TF_GRID), len(GRATING_ORIENTATIONS)), np.nan)
    unit_position = {unit: index for index, unit in enumerate(units)}
    for row in points.itertuples(index=False):
        si = int(np.flatnonzero(np.isclose(SF_GRID, float(row.spatial_cpd)))[0])
        ti = int(np.flatnonzero(np.isclose(TF_GRID, float(row.temporal_hz)))[0])
        oi = int(np.flatnonzero(np.isclose(GRATING_ORIENTATIONS, float(row.orientation_deg)))[0])
        signed[unit_position[int(row.rr100_index)], si, ti, oi] = float(row.signed_f0_hz)
    if not np.isfinite(signed).all():
        raise ValueError("The measured SFxorientationxTF tensor is incomplete")
    return units, signed, np.maximum(signed, 0.0), source_audit


def cell_coordinates(sf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    sf_log, tf_log = np.meshgrid(np.log2(sf), np.log2(tf), indexing="ij")
    coordinates = np.column_stack([sf_log.ravel(), tf_log.ravel()])
    low = np.asarray([np.log2(SF_GRID.min()), np.log2(TF_GRID.min())])
    high = np.asarray([np.log2(SF_GRID.max()), np.log2(TF_GRID.max())])
    span = np.maximum(high - low, 1e-12)
    return 2.0 * (coordinates - low) / span - 1.0


def project_first_orientation_harmonic(values: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(GRATING_ORIENTATIONS)
    design = np.column_stack([np.ones(len(theta)), np.cos(2 * theta), np.sin(2 * theta)])
    inverse = np.linalg.pinv(design)
    return np.einsum("ko,uco->uck", inverse, values)


def reconstruct_harmonic(coefficients: np.ndarray, orientations_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(orientations_deg, dtype=float))
    design = np.column_stack([np.ones(len(theta)), np.cos(2 * theta), np.sin(2 * theta)])
    return np.maximum(np.einsum("uck,ok->uco", coefficients, design), 0.0)


def cross_validate_tuning(
    positive: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_units, n_sf, n_tf, n_orientation = positive.shape
    flat = positive.reshape(n_units, n_sf * n_tf, n_orientation)
    harmonic_targets = project_first_orientation_harmonic(flat)
    candidates = [(gamma, alpha) for gamma in (0.5, 1.0, 2.0, 4.0) for alpha in (1e-3, 1e-2, 1e-1)]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2718).split(coordinates))
    harmonic_predictions = np.empty((len(candidates),) + flat.shape, dtype=float)
    separable_predictions = np.empty_like(harmonic_predictions)

    for candidate_index, (gamma, alpha) in enumerate(candidates):
        for train, test in folds:
            target = harmonic_targets[:, train, :].transpose(1, 0, 2).reshape(len(train), -1)
            predicted = solve_kernel_ridge(coordinates[train], target, coordinates[test], gamma, alpha)
            predicted = predicted.reshape(len(test), n_units, 3).transpose(1, 0, 2)
            harmonic_predictions[candidate_index][:, test, :] = reconstruct_harmonic(
                predicted, GRATING_ORIENTATIONS
            )

            mean_prediction = np.maximum(predicted[:, :, 0], 0.0)
            orientation_marginal = flat[:, train, :].mean(axis=1)
            orientation_factor = orientation_marginal / np.maximum(
                orientation_marginal.mean(axis=1, keepdims=True), 1e-12
            )
            separable_predictions[candidate_index][:, test, :] = (
                mean_prediction[:, :, None] * orientation_factor[:, None, :]
            )

    harmonic_r2 = np.empty((len(candidates), n_units), dtype=float)
    separable_r2 = np.empty_like(harmonic_r2)
    for candidate_index in range(len(candidates)):
        for unit in range(n_units):
            harmonic_r2[candidate_index, unit] = centered_r2(
                flat[unit], harmonic_predictions[candidate_index, unit]
            )
            separable_r2[candidate_index, unit] = centered_r2(
                flat[unit], separable_predictions[candidate_index, unit]
            )
    harmonic_scores = np.where(np.isfinite(harmonic_r2), harmonic_r2, -np.inf)
    separable_scores = np.where(np.isfinite(separable_r2), separable_r2, -np.inf)
    best_harmonic = np.argmax(harmonic_scores, axis=0)
    best_separable = np.argmax(separable_scores, axis=0)
    selected_harmonic = np.stack(
        [harmonic_predictions[best_harmonic[unit], unit] for unit in range(n_units)], axis=0
    )
    selected_separable = np.stack(
        [separable_predictions[best_separable[unit], unit] for unit in range(n_units)], axis=0
    )
    rows = []
    for unit in range(n_units):
        hg, ha = candidates[int(best_harmonic[unit])]
        sg, sa = candidates[int(best_separable[unit])]
        harmonic_value = harmonic_r2[best_harmonic[unit], unit]
        separable_value = separable_r2[best_separable[unit], unit]
        rows.append(
            {
                "rr100_index": unit,
                "harmonic_cv_r2": float(harmonic_value),
                "separable_cv_r2": float(separable_value),
                "harmonic_minus_separable_cv_r2": float(
                    harmonic_value - separable_value
                ),
                "harmonic_gamma": float(hg),
                "harmonic_alpha": float(ha),
                "separable_gamma": float(sg),
                "separable_alpha": float(sa),
                "chosen_orientation_model": "harmonic_interaction" if harmonic_value >= separable_value else "separable_orientation",
                "chosen_orientation_model_cv_r2": float(max(harmonic_value, separable_value)),
            }
        )
    return (
        pd.DataFrame(rows),
        best_harmonic,
        best_separable,
        selected_harmonic,
        selected_separable,
        np.asarray(candidates),
    )


def fit_harmonic_surfaces(
    positive: np.ndarray,
    coordinates: np.ndarray,
    query_coordinates: np.ndarray,
    best_candidate: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_units = positive.shape[0]
    flat = positive.reshape(n_units, -1, positive.shape[-1])
    targets = project_first_orientation_harmonic(flat)
    grid_coefficients = np.empty((n_units, len(coordinates), 3), dtype=float)
    query_coefficients = np.empty((n_units, len(query_coordinates), 3), dtype=float)
    for candidate_index in np.unique(best_candidate):
        units = np.flatnonzero(best_candidate == candidate_index)
        gamma, alpha = candidates[int(candidate_index)]
        target = targets[units].transpose(1, 0, 2).reshape(len(coordinates), -1)
        grid = solve_kernel_ridge(coordinates, target, coordinates, float(gamma), float(alpha))
        query = solve_kernel_ridge(coordinates, target, query_coordinates, float(gamma), float(alpha))
        grid_coefficients[units] = grid.reshape(len(coordinates), len(units), 3).transpose(1, 0, 2)
        query_coefficients[units] = query.reshape(len(query_coordinates), len(units), 3).transpose(1, 0, 2)
    return grid_coefficients, query_coefficients


def circular_distance_180(a: np.ndarray, b: float) -> np.ndarray:
    difference = np.abs(np.asarray(a, dtype=float) - float(b)) % 180.0
    return np.minimum(difference, 180.0 - difference)


def collapse_to_four_grating_channels(values: np.ndarray, orientation_edges: np.ndarray) -> np.ndarray:
    centers = 0.5 * (orientation_edges[:-1] + orientation_edges[1:])
    wavevector_targets = (90.0 - GRATING_ORIENTATIONS) % 180.0
    distances = np.stack([circular_distance_180(centers, target) for target in wavevector_targets], axis=1)
    minimum = distances.min(axis=1, keepdims=True)
    weights = np.isclose(distances, minimum).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("...o,oc->...c", values, weights)


def cosine_similarity_rows(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float).ravel()
    candidates = np.asarray(candidates, dtype=float).reshape(len(candidates), -1)
    numerator = candidates @ reference
    denominator = np.linalg.norm(candidates, axis=1) * max(np.linalg.norm(reference), 1e-15)
    return numerator / np.maximum(denominator, 1e-15)


def select_units(metrics: pd.DataFrame, radial_maps: np.ndarray) -> pd.DataFrame:
    eligible = metrics[
        metrics.recorded_validation_pass.fillna(False)
        & metrics.responsive_positive_f0_flag
        & metrics.chosen_orientation_model_cv_r2.ge(0.0)
    ].copy()
    if len(eligible) < 8:
        eligible = metrics[metrics.recorded_validation_pass.fillna(False) & metrics.responsive_positive_f0_flag].copy()
    used: set[int] = set()
    selected: list[pd.Series] = []

    high_radial = eligible[eligible.radial_drive >= eligible.radial_drive.median()]
    high_osi = high_radial[high_radial.orientation_vector_strength >= high_radial.orientation_vector_strength.median()]
    matched_pool = high_osi if len(high_osi) else high_radial
    matched = matched_pool.loc[matched_pool.orientation_alignment_ratio.idxmax()].copy()
    matched["selection_role"] = "orientation-aligned spectral match"
    matched["selection_criterion"] = (
        "largest orientation/radial drive ratio among above-median radial-drive and orientation-selective eligible units"
    )
    selected.append(matched)
    used.add(int(matched.rr100_index))

    available = eligible[~eligible.rr100_index.isin(used)].copy()
    reference_position = int(matched.tensor_position)
    available["radial_map_cosine_similarity_to_match"] = cosine_similarity_rows(
        radial_maps[reference_position], radial_maps[available.tensor_position.astype(int).to_numpy()]
    )
    matched_drive = float(matched.radial_drive)
    mismatch_pool = available[
        available.radial_map_cosine_similarity_to_match.ge(0.8)
        & available.radial_drive.between(0.5 * matched_drive, 2.0 * matched_drive)
    ]
    if mismatch_pool.empty:
        mismatch_pool = available.nlargest(min(20, len(available)), "radial_map_cosine_similarity_to_match")
    mismatch = mismatch_pool.loc[mismatch_pool.orientation_alignment_ratio.idxmin()].copy()
    mismatch["selection_role"] = "similar radial passband, orientation mismatch"
    mismatch["selection_criterion"] = (
        "lowest orientation/radial drive ratio among units with radial-map cosine similarity >=0.8 and radial magnitude within 0.5–2x of the matched unit"
    )
    selected.append(mismatch)
    used.add(int(mismatch.rr100_index))

    available = eligible[~eligible.rr100_index.isin(used)]
    broad_pool = available[available.radial_drive >= eligible.radial_drive.median()]
    quality_broad_pool = broad_pool[
        broad_pool.chosen_orientation_model_cv_r2.ge(0.75)
        & broad_pool.recorded_sf_curve_r_full_support.ge(0.6)
    ]
    if not quality_broad_pool.empty:
        broad_pool = quality_broad_pool
    broad = broad_pool.loc[broad_pool.broad_control_flatness_score.idxmin()].copy()
    broad["selection_role"] = "broad-orientation control"
    broad["selection_criterion"] = (
        "smallest sum of passband-weighted local orientation modulation and absolute log2 orientation/radial drive ratio among above-median-drive units with chosen grating CV R2>=0.75 and recorded-SF r>=0.6"
    )
    selected.append(broad)
    used.add(int(broad.rr100_index))

    available = eligible[~eligible.rr100_index.isin(used)]
    osi_threshold = eligible.orientation_vector_strength.quantile(0.75)
    weak_pool = available[available.orientation_vector_strength >= osi_threshold]
    weak = weak_pool.loc[weak_pool.radial_drive.idxmin()].copy()
    weak["selection_role"] = "orientation-selective, weak spectral support"
    weak["selection_criterion"] = "smallest radial drive among top-quartile orientation-selective eligible units"
    selected.append(weak)

    result = pd.DataFrame(selected)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    if "radial_map_cosine_similarity_to_match" not in result:
        result["radial_map_cosine_similarity_to_match"] = np.nan
    return result


def relative_db(values: np.ndarray, maximum: float | None = None, floor_db: float = -45.0) -> np.ndarray:
    if maximum is None:
        maximum = float(np.nanmax(values))
    maximum = max(float(maximum), EPS)
    return np.maximum(10.0 * np.log10(np.maximum(values / maximum, EPS)), floor_db)


def format_frequency_ticks(axis: plt.Axes, sf: np.ndarray, tf: np.ndarray) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(float(sf.min()), float(sf.max()))
    axis.set_ylim(float(tf.min()), float(tf.max()))
    axis.set_xlabel("SF (cpd)")


def make_validation_page(
    selected: pd.DataFrame,
    positive: np.ndarray,
    oof: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(len(selected), 5, figsize=(17, 3.15 * len(selected)), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        position = int(row.tensor_position)
        observed = positive[position]
        predicted = oof[position].reshape(len(SF_GRID), len(TF_GRID), 4)
        vmax = max(float(observed.max()), 1e-12)
        for orientation_index, orientation in enumerate(GRATING_ORIENTATIONS):
            axis = axes[row_index, orientation_index]
            shown = observed[:, :, orientation_index].T
            image = axis.pcolormesh(
                SF_GRID,
                TF_GRID,
                shown,
                shading="nearest",
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            format_frequency_ticks(axis, SF_GRID, TF_GRID)
            axis.set_ylabel("TF (Hz)" if orientation_index == 0 else "")
            axis.set_title(f"{orientation:.0f}° grating F0")
            if orientation_index == 3:
                fig.colorbar(image, ax=axis, label="F0 above blank (Hz)", shrink=0.8)
        axis = axes[row_index, 4]
        measured = observed.ravel()
        heldout = predicted.ravel()
        axis.scatter(measured, heldout, s=9, alpha=0.35, color="#0072B2")
        limit = max(float(measured.max()), float(heldout.max()), 1e-12)
        axis.plot([0, limit], [0, limit], ls="--", color="0.55")
        axis.set(
            xlabel="measured F0 (Hz)",
            ylabel="held-out prediction (Hz)",
            title=(
                f"{row.selection_role}\nRR100 {int(row.rr100_index)} · chosen {row.chosen_orientation_model.replace('_', ' ')}\n"
                f"held-out $R^2$={row.chosen_orientation_model_cv_r2:.2f}"
            ),
        )
    fig.suptitle(
        "Measured fixed-retina F0 tuning and held-out grating-only model validation\n"
        "Each row has its own F0 color scale; 60 Hz is excluded",
        fontsize=15,
        weight="bold",
    )
    return fig


def make_routing_page(
    selected: pd.DataFrame,
    movie_channels: np.ndarray,
    contribution_channels: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
) -> plt.Figure:
    n_rows = len(selected) + 1
    fig = plt.figure(figsize=(18, 3.0 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, 1, 1, 1.12])
    input_max = float(movie_channels.max())
    for orientation_index, orientation in enumerate(GRATING_ORIENTATIONS):
        axis = fig.add_subplot(grid[0, orientation_index])
        image = axis.pcolormesh(
            sf,
            tf,
            relative_db(movie_channels[:, :, orientation_index], input_max),
            shading="nearest",
            cmap="magma",
            vmin=-45,
            vmax=0,
        )
        format_frequency_ticks(axis, sf, tf)
        axis.set_ylabel("TF (Hz)" if orientation_index == 0 else "")
        axis.set_title(f"Input: {orientation:.0f}° grating-axis channel")
    fig.colorbar(image, ax=[fig.axes[index] for index in range(4)], label="input power (dB, common scale)", shrink=0.75)
    axis = fig.add_subplot(grid[0, 4])
    input_fraction = movie_channels.sum(axis=(0, 1))
    input_fraction /= max(float(input_fraction.sum()), EPS)
    axis.bar(["0°", "45°", "90°", "135°"], 100 * input_fraction, color=["#0072B2", "#E69F00", "#009E73", "#D55E00"])
    axis.set(ylabel="supported dynamic power (%)", title="Same movie for every unit")

    for row_index, row in enumerate(selected.itertuples(index=False), start=1):
        position = int(row.tensor_position)
        contributions = contribution_channels[position]
        row_max = max(float(contributions.max()), EPS)
        for orientation_index, orientation in enumerate(GRATING_ORIENTATIONS):
            axis = fig.add_subplot(grid[row_index, orientation_index])
            image = axis.pcolormesh(
                sf,
                tf,
                relative_db(contributions[:, :, orientation_index], row_max),
                shading="nearest",
                cmap="magma",
                vmin=-45,
                vmax=0,
            )
            format_frequency_ticks(axis, sf, tf)
            axis.set_ylabel("TF (Hz)" if orientation_index == 0 else "")
            axis.set_title(f"Power accepted through {orientation:.0f}° channel")
        fig.colorbar(
            image,
            ax=[fig.axes[-4], fig.axes[-3], fig.axes[-2], fig.axes[-1]],
            label="accepted power (dB, row scale)",
            shrink=0.75,
        )
        axis = fig.add_subplot(grid[row_index, 4])
        radial = float(row.radial_drive)
        oriented = float(row.orientation_aware_drive)
        axis.bar(["SF×TF\nonly", "+ orientation"], [radial, oriented], color=["0.62", "#D55E00"])
        axis.set_yticks([])
        axis.set_title(
            f"{row.selection_role}\nRR100 {int(row.rr100_index)} · ratio={row.orientation_alignment_ratio:.2f}\n"
            f"OSI={row.orientation_vector_strength:.2f}"
        )
        axis.text(
            0.5,
            -0.25,
            "Direct F0 weighting; absolute scale is arbitrary",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    fig.suptitle(
        "One retinal movie routed through four measured SF×orientation×TF filters\n"
        "Nothing is rotated: columns denote image/Fourier orientation channels matched to each grating axis",
        fontsize=15,
        weight="bold",
    )
    return fig


def make_nested_comparison_page(
    selected: pd.DataFrame,
    radial_maps: np.ndarray,
    oriented_maps: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 3.2 * len(selected)), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        position = int(row.tensor_position)
        radial = radial_maps[position]
        oriented = oriented_maps[position]
        maximum = max(float(radial.max()), float(oriented.max()), EPS)
        for column, values, title in (
            (0, radial, "SF×TF-only accepted power"),
            (1, oriented, "orientation-aware accepted power"),
        ):
            axis = axes[row_index, column]
            image = axis.pcolormesh(
                sf, tf, relative_db(values, maximum), shading="nearest", cmap="magma", vmin=-45, vmax=0
            )
            format_frequency_ticks(axis, sf, tf)
            axis.set_ylabel("TF (Hz)" if column == 0 else "")
            axis.set_title(title)
        fig.colorbar(image, ax=[axes[row_index, 0], axes[row_index, 1]], label="accepted power (dB, shared row scale)")
        difference = oriented - radial
        difference_scale = max(float(np.max(np.abs(difference))), EPS)
        axis = axes[row_index, 2]
        image = axis.pcolormesh(
            sf,
            tf,
            difference,
            shading="nearest",
            cmap="coolwarm",
            vmin=-difference_scale,
            vmax=difference_scale,
        )
        format_frequency_ticks(axis, sf, tf)
        axis.set_title("orientation contribution\n(aware − SF×TF only)")
        fig.colorbar(image, ax=axis, label="signed accepted power (a.u.)")
        axis = axes[row_index, 3]
        axis.axis("off")
        axis.text(0.03, 0.88, row.selection_role, fontsize=12, weight="bold", va="top")
        axis.text(
            0.03,
            0.68,
            f"RR100 {int(row.rr100_index)}\n"
            f"orientation-aware / radial = {row.orientation_alignment_ratio:.2f}\n"
            f"orientation vector strength = {row.orientation_vector_strength:.2f}\n"
            f"chosen held-out $R^2$ = {row.chosen_orientation_model_cv_r2:.2f}\n"
            f"recorded-SF validation r = {row.recorded_sf_curve_r_full_support:.2f}",
            fontsize=10.5,
            va="top",
            linespacing=1.45,
        )
    fig.suptitle(
        "Orientation is a strictly nested correction to the same SF×TF passband\n"
        "The orientation factor is normalized to mean one at every SF×TF bin",
        fontsize=15,
        weight="bold",
    )
    return fig


def main() -> None:
    spectral = validated_spectral_cache_from_environment()
    validate_artifact_not_superseded(INPUT_CHECKPOINT, label="orientation-routing input checkpoint")
    OUT.mkdir(parents=True, exist_ok=True)
    units, signed, positive, source_audit = build_measured_tensor()
    coordinates = cell_coordinates(SF_GRID, TF_GRID)
    (
        fit_quality,
        best_harmonic,
        best_separable,
        oof_harmonic,
        oof_separable,
        candidates,
    ) = cross_validate_tuning(positive, coordinates)

    with np.load(spectral / "condition_spectra.npz", allow_pickle=False) as data:
        oriented_all = np.asarray(data["orientation_power"], dtype=float)
        radial_all = np.asarray(data["radial_power"], dtype=float)
        sf_edges = np.asarray(data["sf_edges_cpd"], dtype=float)
        movie_tf_all = np.asarray(data["tf_hz"], dtype=float)
        orientation_edges = np.asarray(data["orientation_edges_deg"], dtype=float)
        image_ids = np.asarray(data["image_index"], dtype=int)
        trace_ids = np.asarray(data["trace_index"], dtype=int)
        round_ids = np.asarray(data["round_index"], dtype=int)
    selected_input = pd.read_csv(INPUT_CHECKPOINT / "selected_input_condition.csv").iloc[0]
    condition_row = int(selected_input.matrix_row_index)
    movie_sf_all = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    sf_mask = (movie_sf_all >= SF_GRID.min()) & (movie_sf_all <= SF_GRID.max())
    tf_mask = (movie_tf_all > 0) & (movie_tf_all <= EXT_TF.max())
    movie_sf = movie_sf_all[sf_mask]
    movie_tf = movie_tf_all[tf_mask]
    movie_power = oriented_all[condition_row][tf_mask][:, sf_mask, :]
    radial_power = radial_all[condition_row][tf_mask][:, sf_mask]
    if not np.allclose(movie_power.sum(axis=-1), radial_power, rtol=2e-5, atol=1e-3):
        raise ValueError("Orientation bins do not sum back to radial power")

    query_coordinates = cell_coordinates(movie_sf, movie_tf)
    _, query_coefficients = fit_harmonic_surfaces(
        positive, coordinates, query_coordinates, best_harmonic, candidates
    )
    _, separable_query_coefficients = fit_harmonic_surfaces(
        positive, coordinates, query_coordinates, best_separable, candidates
    )
    n_units = len(units)
    n_tf = len(movie_tf)
    n_sf = len(movie_sf)
    coefficients = query_coefficients.reshape(n_units, n_sf, n_tf, 3).transpose(0, 2, 1, 3)
    wavevector_centers = 0.5 * (orientation_edges[:-1] + orientation_edges[1:])
    grating_centers = (90.0 - wavevector_centers) % 180.0
    theta = np.deg2rad(grating_centers)
    harmonic_raw_weight = np.maximum(
        coefficients[..., 0, None]
        + coefficients[..., 1, None] * np.cos(2 * theta)
        + coefficients[..., 2, None] * np.sin(2 * theta),
        0.0,
    )
    harmonic_wbar = np.maximum(coefficients[..., 0], 0.0)
    raw_mean = harmonic_raw_weight.mean(axis=-1, keepdims=True)
    harmonic_orientation_factor = np.divide(
        harmonic_raw_weight,
        raw_mean,
        out=np.ones_like(harmonic_raw_weight),
        where=raw_mean > 1e-12,
    )
    harmonic_weights = harmonic_wbar[..., None] * harmonic_orientation_factor

    separable_coefficients = separable_query_coefficients.reshape(n_units, n_sf, n_tf, 3).transpose(0, 2, 1, 3)
    separable_wbar = np.maximum(separable_coefficients[..., 0], 0.0)
    orientation_marginal = positive.mean(axis=(1, 2))
    separable_orientation_factor_four = orientation_marginal / np.maximum(
        orientation_marginal.mean(axis=1, keepdims=True), 1e-12
    )
    separable_orientation_coefficients = project_first_orientation_harmonic(
        separable_orientation_factor_four[:, None, :]
    )
    separable_orientation_factor = reconstruct_harmonic(
        separable_orientation_coefficients, grating_centers
    )[:, 0, :]
    separable_orientation_factor /= np.maximum(
        separable_orientation_factor.mean(axis=1, keepdims=True), 1e-12
    )
    separable_weights = separable_wbar[..., None] * separable_orientation_factor[:, None, None, :]
    choose_harmonic = fit_quality.chosen_orientation_model.eq("harmonic_interaction").to_numpy()
    wbar = np.where(choose_harmonic[:, None, None], harmonic_wbar, separable_wbar)
    orientation_factor = np.where(
        choose_harmonic[:, None, None, None],
        harmonic_orientation_factor,
        separable_orientation_factor[:, None, None, :],
    )
    weights = np.where(choose_harmonic[:, None, None, None], harmonic_weights, separable_weights)
    if not np.allclose(weights.mean(axis=-1), wbar, rtol=1e-8, atol=1e-10):
        raise ValueError("Orientation normalization failed to preserve the SFxTF marginal")

    radial_maps = radial_power[None, :, :] * wbar
    oriented_contributions = movie_power[None, :, :, :] * weights
    oriented_maps = oriented_contributions.sum(axis=-1)
    radial_drive = radial_maps.sum(axis=(1, 2))
    orientation_drive = oriented_maps.sum(axis=(1, 2))

    orientation_phase = np.exp(2j * np.deg2rad(GRATING_ORIENTATIONS))
    orientation_vector_strength = np.abs(orientation_marginal @ orientation_phase) / np.maximum(
        orientation_marginal.sum(axis=1), 1e-12
    )
    preferred_orientation = GRATING_ORIENTATIONS[np.argmax(orientation_marginal, axis=1)]
    assignments = pd.read_csv(ASSIGNMENTS)
    fit_summary = pd.read_csv(
        ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_extended_tf_f0_analysis_v1/extended_f0_fit_unit_summary.csv"
    )
    metrics = fit_quality.merge(
        assignments[
            [
                "rr100_index",
                "recorded_validation_pass",
                "recorded_sf_curve_r_full_support",
                "sf_outer_third",
                "preferred_sf_cpd",
            ]
        ],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    ).merge(
        fit_summary[["rr100_index", "responsive_positive_f0_flag", "extended_tf_center_frequency"]],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    )
    metrics["tensor_position"] = np.arange(len(metrics))
    metrics["orientation_vector_strength"] = orientation_vector_strength
    metrics["preferred_orientation_deg"] = preferred_orientation
    metrics["maximum_positive_f0_hz"] = positive.max(axis=(1, 2, 3))
    metrics["suppressive_cell_fraction"] = (signed < 0).mean(axis=(1, 2, 3))
    metrics["radial_drive"] = radial_drive
    metrics["orientation_aware_drive"] = orientation_drive
    metrics["orientation_alignment_ratio"] = orientation_drive / np.maximum(radial_drive, EPS)
    local_orientation_variance = np.mean((orientation_factor - 1.0) ** 2, axis=-1)
    metrics["passband_weighted_orientation_modulation_rms"] = np.sqrt(
        np.sum(radial_maps * local_orientation_variance, axis=(1, 2))
        / np.maximum(radial_maps.sum(axis=(1, 2)), 1e-12)
    )
    metrics["broad_control_flatness_score"] = (
        metrics.passband_weighted_orientation_modulation_rms
        + np.abs(np.log2(np.maximum(metrics.orientation_alignment_ratio, 1e-12)))
    )
    metrics.to_csv(OUT / "orientation_tuning_fit_quality_and_movie_overlap.csv", index=False)

    selected = select_units(metrics, radial_maps)
    selected.insert(0, "condition_matrix_row", condition_row)
    selected.insert(1, "image_index", int(image_ids[condition_row]))
    selected.insert(2, "trace_index", int(trace_ids[condition_row]))
    selected.insert(3, "round_index", int(round_ids[condition_row]))
    selected.to_csv(OUT / "selected_units.csv", index=False)

    movie_channels = collapse_to_four_grating_channels(movie_power, orientation_edges)
    contribution_channels = collapse_to_four_grating_channels(oriented_contributions, orientation_edges)
    oof_chosen = np.where(
        choose_harmonic[:, None, None], oof_harmonic, oof_separable
    )
    validation_figure = make_validation_page(selected, positive, oof_chosen)
    routing_figure = make_routing_page(
        selected, movie_channels, contribution_channels, movie_sf, movie_tf
    )
    nested_figure = make_nested_comparison_page(
        selected, radial_maps, oriented_maps, movie_sf, movie_tf
    )
    for name, figure in (
        ("01_measured_f0_and_heldout_validation", validation_figure),
        ("02_same_movie_through_four_units", routing_figure),
        ("03_nested_orientation_correction", nested_figure),
    ):
        figure.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
        figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    with PdfPages(OUT / "orientation_aware_f0_map_checkpoint.pdf") as pdf:
        for figure in (validation_figure, routing_figure, nested_figure):
            pdf.savefig(figure, bbox_inches="tight")
    plt.close("all")

    np.savez_compressed(
        OUT / "orientation_aware_f0_tuning_and_routing.npz",
        rr100_index=units,
        measured_sf_cpd=SF_GRID,
        measured_tf_hz=TF_GRID,
        measured_grating_orientation_deg=GRATING_ORIENTATIONS,
        measured_signed_f0_hz=signed.astype(np.float32),
        measured_positive_f0_hz=positive.astype(np.float32),
        heldout_harmonic_prediction_f0_hz=oof_harmonic.reshape(positive.shape).astype(np.float32),
        heldout_separable_prediction_f0_hz=oof_separable.reshape(positive.shape).astype(np.float32),
        heldout_chosen_prediction_f0_hz=oof_chosen.reshape(positive.shape).astype(np.float32),
        movie_sf_cpd=movie_sf,
        movie_tf_hz=movie_tf,
        movie_fourier_orientation_deg=wavevector_centers,
        movie_power=movie_power.astype(np.float32),
        smoothed_radial_f0_weight=wbar.astype(np.float32),
        normalized_orientation_factor=orientation_factor.astype(np.float32),
        orientation_aware_f0_weight=weights.astype(np.float32),
        chosen_orientation_model=fit_quality.chosen_orientation_model.to_numpy(dtype="U24"),
        radial_accepted_power_map=radial_maps.astype(np.float32),
        orientation_aware_accepted_power_map=oriented_maps.astype(np.float32),
    )

    validated = metrics[metrics.recorded_validation_pass.fillna(False) & metrics.responsive_positive_f0_flag]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_orientation_aware_f0_checkpoint_complete",
        "scope": {
            "units_with_complete_grating_tensor": int(len(units)),
            "recorded_sf_validated_responsive_units": int(len(validated)),
            "fixed_movie_condition_row": condition_row,
            "image_index": int(image_ids[condition_row]),
            "trace_index": int(trace_ids[condition_row]),
        },
        "tuning_contract": {
            "response": "direction-folded, phase-averaged F0 above blank; negative values retained in audit tensor and clipped to zero for excitatory routing",
            "support": "8 SFs from 1 to 11.3137 cpd; 25 fitted TFs from 0.5 to 56 Hz",
            "controls_excluded": "new repeated 32-Hz plane is replication-only; 60 Hz is Nyquist-only",
            "orientation_model": "per-unit grating-only selection between (a) a smooth SFxTF surface times one global orientation profile and (b) smooth SFxTF-varying mean/cos(2theta)/sin(2theta) surfaces",
            "validation": "five-fold held-out SFxTF cells; all four orientations of a cell are held out together",
            "source_reproduction_audit": source_audit,
        },
        "routing_contract": {
            "primary_equation": "sum P(SF,theta,TF) * W_F0(SF,theta,TF)",
            "important_correction": "F0 is used directly and is not squared",
            "radial_nesting": "orientation factor is normalized to mean one at each SFxTF bin, preserving the same smoothed radial F0 marginal",
            "movie_orientation_bins_used_in_calculation": 12,
            "four_channel_panels": "visualization-only power-preserving collapse to 0/45/90/135 degree grating-axis channels",
            "rotation": "neither image nor eye trace nor unit is rotated; Fourier wavevector theta_k maps to grating-bar theta_g=(90-theta_k) mod 180",
        },
        "selection_guardrail": "example units selected from grating tuning plus the fixed movie power only; natural-movie model responses and SSI were not consulted",
        "fit_summary_validated_units": {
            "median_harmonic_cv_r2": float(validated.harmonic_cv_r2.median()),
            "median_separable_cv_r2": float(validated.separable_cv_r2.median()),
            "median_harmonic_minus_separable_cv_r2": float(validated.harmonic_minus_separable_cv_r2.median()),
            "n_harmonic_interaction_selected": int(validated.chosen_orientation_model.eq("harmonic_interaction").sum()),
            "n_separable_orientation_selected": int(validated.chosen_orientation_model.eq("separable_orientation").sum()),
        },
        "unsupported_yet": [
            "whether orientation-aware routed power predicts frozen-model activation",
            "whether it predicts SSI or information numerator",
            "whether it improves crossed image-and-trace generalization over the corrected direct-F0 radial model",
            "direction-selective routing, because this primary tensor folds opposite drift directions",
        ],
        "artifacts": {
            "multipage_pdf": str((OUT / "orientation_aware_f0_map_checkpoint.pdf").relative_to(ROOT)),
            "selected_units": str((OUT / "selected_units.csv").relative_to(ROOT)),
            "fit_quality": str((OUT / "orientation_tuning_fit_quality_and_movie_overlap.csv").relative_to(ROOT)),
            "arrays": str((OUT / "orientation_aware_f0_tuning_and_routing.npz").relative_to(ROOT)),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

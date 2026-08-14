#!/usr/bin/env python3
"""Distribution-constrained 3-D phase control for the Stage 2A map input.

This is deliberately separate from the completed unconstrained Stage 2A v1
factorial.  It uses independent 3-D IAAFT projections at the FEM and stabilized
power levels: every saved surrogate ends with an exact Fourier-amplitude
projection, while the iterative rank projection makes its value distribution
closely match the corresponding original cube.  Because the two IAAFT phase
fields are optimized independently, this is not described as a shared-phase
factorial.

The runner stops at input validation, recorded-grating generic-degradation
calibration, and one small raw-map checkpoint.  It never opens final-test
identities and never performs population inference.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew, spearmanr, wasserstein_distance

from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig4_active_sensing.run_rr100_map_support_amplitude_phase_factorial_stage2a import (
    amplitude_relative_error,
    file_identity,
    hermitian_relative_error,
    map_metrics,
    phase_unit,
    tile_energy,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
SOURCE_V1 = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_v1"
SOURCE_METHOD_AUDIT = ROOT / (
    "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_method_audit_v1"
)
GRATING_SOURCE = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_retinal_power_input_checkpoint_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_map_support_stage2a_distribution_constrained_phase_v1"
)
SEEDS = (20260901, 20260902, 20260903)
N_ITERATIONS = 64
PATCH_SIZE = 15
CANONICAL_LOW, CANONICAL_HIGH = -127.0 / 255.0, 128.0 / 255.0
RELATIVE_SUPPORT_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
MAP_SEED = SEEDS[0]
CONDITION_LABELS = {
    "stabilized_original": "original stabilized",
    "stabilized_power_iaaft_phase": "stabilized power + IAAFT phase",
    "fem_original": "original FEM",
    "fem_power_iaaft_phase": "FEM power + IAAFT phase",
    "stabilized_power_fem_phase": "stabilized power + FEM phase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v1", type=Path, default=SOURCE_V1)
    parser.add_argument("--method-audit-v1", type=Path, default=SOURCE_METHOD_AUDIT)
    parser.add_argument("--grating-source", type=Path, default=GRATING_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def rank_match(values: np.ndarray, sorted_target: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values).ravel(), kind="stable")
    matched = np.empty(order.size, dtype=np.float64)
    matched[order] = sorted_target
    return matched.reshape(values.shape)


def iaaft_3d(
    target: np.ndarray, seed: int, iterations: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return an amplitude-exact, distribution-constrained 3-D surrogate."""
    reference = np.asarray(target, dtype=np.float64)
    amplitude = np.abs(np.fft.fftn(reference))
    sorted_target = np.sort(reference.ravel())
    random_spectrum = np.fft.fftn(np.random.default_rng(int(seed)).standard_normal(reference.shape))
    phase = phase_unit(random_spectrum)
    rows: list[dict[str, float | int]] = []
    checkpoints = {1, 2, 4, 8, 16, 32, int(iterations)}
    for iteration in range(1, int(iterations) + 1):
        amplitude_exact = np.fft.ifftn(amplitude * phase).real
        distribution_exact = rank_match(amplitude_exact, sorted_target)
        phase = phase_unit(np.fft.fftn(distribution_exact))
        if iteration in checkpoints:
            candidate = np.fft.ifftn(amplitude * phase).real
            rows.append({
                "iteration": iteration,
                "histogram_wasserstein_distance": float(
                    np.mean(np.abs(np.sort(candidate.ravel()) - sorted_target))
                ),
                "maximum_value_error_after_sorting": float(
                    np.max(np.abs(np.sort(candidate.ravel()) - sorted_target))
                ),
                "fraction_outside_canonical_range": float(
                    np.mean((candidate < CANONICAL_LOW) | (candidate > CANONICAL_HIGH))
                ),
            })
    final_complex = np.fft.ifftn(amplitude * phase)
    maximum_imaginary = float(np.max(np.abs(final_complex.imag)))
    candidate = final_complex.real.astype(np.float32)
    convergence = pd.DataFrame(rows)
    convergence["seed"] = int(seed)
    convergence["maximum_imaginary_reconstruction_residual"] = maximum_imaginary
    return candidate, convergence


def safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    if rank:
        return float(spearmanr(x, y).statistic)
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / max(denominator, np.finfo(float).tiny))


def patch_statistic(cube: np.ndarray, statistic: str) -> np.ndarray:
    values = np.asarray(cube, dtype=np.float64)
    usable_y = values.shape[1] - values.shape[1] % PATCH_SIZE
    usable_x = values.shape[2] - values.shape[2] % PATCH_SIZE
    output: list[float] = []
    for frame in values:
        for y in range(0, usable_y, PATCH_SIZE):
            for x in range(0, usable_x, PATCH_SIZE):
                patch = frame[y:y + PATCH_SIZE, x:x + PATCH_SIZE].ravel()
                if statistic == "kurtosis":
                    value = kurtosis(patch, fisher=True, bias=False)
                elif statistic == "standard_deviation":
                    value = np.std(patch)
                else:
                    raise ValueError(statistic)
                if np.isfinite(value):
                    output.append(float(value))
    return np.asarray(output, dtype=np.float64)


def phase_coherence_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = np.fft.fftn(np.asarray(reference, dtype=np.float64))
    right = np.fft.fftn(np.asarray(candidate, dtype=np.float64))
    valid = (np.abs(left) > 1e-10) & (np.abs(right) > 1e-10)
    delta = phase_unit(right[valid]) * np.conj(phase_unit(left[valid]))
    weights = np.abs(left[valid]) ** 2
    frame_values: list[float] = []
    for left_frame, right_frame in zip(reference, candidate, strict=True):
        a = np.fft.fft2(np.asarray(left_frame, dtype=np.float64))
        b = np.fft.fft2(np.asarray(right_frame, dtype=np.float64))
        keep = (np.abs(a) > 1e-10) & (np.abs(b) > 1e-10)
        if np.any(keep):
            frame_values.append(
                float(np.abs(np.mean(phase_unit(b[keep]) * np.conj(phase_unit(a[keep])))))
            )
    mean_framewise = float(np.mean(frame_values)) if frame_values else float("nan")
    maximum_framewise = float(np.max(frame_values)) if frame_values else float("nan")
    return {
        "global_unweighted_phase_coherence": float(np.abs(np.mean(delta))),
        "global_energy_weighted_phase_coherence": float(
            np.abs(np.sum(weights * delta) / max(float(weights.sum()), np.finfo(float).tiny))
        ),
        "mean_framewise_2d_phase_coherence": mean_framewise,
        "maximum_framewise_2d_phase_coherence": maximum_framewise,
    }


def distribution_audit(
    reference: np.ndarray, candidate: np.ndarray, *, seed: int, condition: str, source_kind: str
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    amplitude = np.abs(np.fft.fftn(ref))
    spectrum = np.fft.fftn(test)
    ref_patch_kurtosis = patch_statistic(ref, "kurtosis")
    test_patch_kurtosis = patch_statistic(test, "kurtosis")
    ref_patch_std = patch_statistic(ref, "standard_deviation")
    test_patch_std = patch_statistic(test, "standard_deviation")
    ref_spatial_gradient = np.concatenate((np.diff(ref, axis=1).ravel(), np.diff(ref, axis=2).ravel()))
    test_spatial_gradient = np.concatenate((np.diff(test, axis=1).ravel(), np.diff(test, axis=2).ravel()))
    ref_temporal_gradient = np.diff(ref, axis=0).ravel()
    test_temporal_gradient = np.diff(test, axis=0).ravel()
    ref_tile = tile_energy(ref)
    test_tile = tile_energy(test)
    global_kurtosis_reference = float(kurtosis(ref.ravel(), fisher=True, bias=False))
    global_kurtosis_candidate = float(kurtosis(test.ravel(), fisher=True, bias=False))
    metrics = {
        "seed": int(seed),
        "condition": condition,
        "source_kind": source_kind,
        "fourier_amplitude_relative_error": amplitude_relative_error(amplitude, test),
        "reconstructed_spectrum_hermitian_relative_error": hermitian_relative_error(spectrum),
        "histogram_wasserstein_distance": float(wasserstein_distance(ref.ravel(), test.ravel())),
        "maximum_sorted_value_error": float(np.max(np.abs(np.sort(ref.ravel()) - np.sort(test.ravel())))),
        "minimum": float(test.min()),
        "maximum": float(test.max()),
        "fraction_outside_canonical_input_range": float(
            np.mean((test < CANONICAL_LOW) | (test > CANONICAL_HIGH))
        ),
        "mean_difference": float(test.mean() - ref.mean()),
        "standard_deviation_ratio": float(test.std() / max(float(ref.std()), np.finfo(float).tiny)),
        "global_skew_reference": float(skew(ref.ravel(), bias=False)),
        "global_skew_candidate": float(skew(test.ravel(), bias=False)),
        "global_excess_kurtosis_reference": global_kurtosis_reference,
        "global_excess_kurtosis_candidate": global_kurtosis_candidate,
        "absolute_global_excess_kurtosis_difference": abs(
            global_kurtosis_candidate - global_kurtosis_reference
        ),
        "patch_kurtosis_wasserstein_distance": float(
            wasserstein_distance(ref_patch_kurtosis, test_patch_kurtosis)
        ),
        "patch_standard_deviation_wasserstein_distance": float(
            wasserstein_distance(ref_patch_std, test_patch_std)
        ),
        "spatial_gradient_wasserstein_over_reference_std": float(
            wasserstein_distance(ref_spatial_gradient, test_spatial_gradient)
            / max(float(np.std(ref_spatial_gradient)), np.finfo(float).tiny)
        ),
        "temporal_gradient_wasserstein_over_reference_std": float(
            wasserstein_distance(ref_temporal_gradient, test_temporal_gradient)
            / max(float(np.std(ref_temporal_gradient)), np.finfo(float).tiny)
        ),
        "tile_energy_correlation_with_reference": safe_correlation(ref_tile, test_tile),
        "tile_energy_total_variation_from_reference": float(0.5 * np.abs(ref_tile - test_tile).sum()),
        **phase_coherence_metrics(ref, test),
    }
    metrics["gate_exact_amplitude"] = metrics["fourier_amplitude_relative_error"] <= 1e-6
    metrics["gate_input_range"] = metrics["fraction_outside_canonical_input_range"] <= 1e-4
    metrics["gate_histogram"] = metrics["histogram_wasserstein_distance"] <= 1e-3
    metrics["gate_global_higher_order"] = metrics["absolute_global_excess_kurtosis_difference"] <= 0.02
    metrics["gate_patch_higher_order"] = metrics["patch_kurtosis_wasserstein_distance"] <= 0.5
    metrics["gate_spatial_gradient_distribution"] = (
        metrics["spatial_gradient_wasserstein_over_reference_std"] <= 0.15
    )
    metrics["gate_temporal_gradient_distribution"] = (
        metrics["temporal_gradient_wasserstein_over_reference_std"] <= 0.15
    )
    metrics["gate_phase_destroyed"] = (
        metrics["global_unweighted_phase_coherence"] <= 0.05
        and metrics["global_energy_weighted_phase_coherence"] <= 0.10
    )
    gate_columns = [key for key in metrics if key.startswith("gate_")]
    metrics["distribution_and_phase_gate_pass"] = bool(all(bool(metrics[key]) for key in gate_columns))
    return metrics


def phase_source_support_audit(conditions: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    targets = {
        "FEM": np.asarray(conditions[(SEEDS[0], "fem_original")], dtype=np.float64),
        "stabilized": np.asarray(conditions[(SEEDS[0], "stabilized_original")], dtype=np.float64),
    }
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sources = {
            "FEM original": targets["FEM"],
            "stabilized original": targets["stabilized"],
            "FEM IAAFT": conditions[(seed, "fem_power_iaaft_phase")],
            "stabilized IAAFT": conditions[(seed, "stabilized_power_iaaft_phase")],
        }
        for source_name, source_cube in sources.items():
            source_amplitude = np.abs(np.fft.fftn(np.asarray(source_cube, dtype=np.float64)))
            for target_name, target_cube in targets.items():
                target_amplitude = np.abs(np.fft.fftn(target_cube))
                used = (
                    (source_name == "FEM IAAFT" and target_name == "FEM")
                    or (source_name == "stabilized IAAFT" and target_name == "stabilized")
                    or (source_name == "FEM original" and target_name in {"FEM", "stabilized"})
                    or (source_name == "stabilized original" and target_name == "stabilized")
                )
                for threshold in RELATIVE_SUPPORT_THRESHOLDS:
                    absolute = max(1e-12, threshold * float(source_amplitude.max()))
                    invalid = source_amplitude <= absolute
                    rows.append({
                        "seed": int(seed),
                        "source_phase": source_name,
                        "target_amplitude": target_name,
                        "pair_used_in_checkpoint": bool(used),
                        "relative_source_amplitude_threshold": float(threshold),
                        "invalid_source_phase_bin_fraction": float(invalid.mean()),
                        "target_spectral_energy_fraction_in_invalid_source_bins": float(
                            np.square(target_amplitude[invalid]).sum()
                            / max(float(np.square(target_amplitude).sum()), np.finfo(float).tiny)
                        ),
                    })
    return pd.DataFrame(rows)


def load_natural_conditions(
    source: Path, iterations: int
) -> tuple[dict[tuple[int, str], np.ndarray], pd.DataFrame, dict[str, int]]:
    with np.load(source / "factorial_input_cubes.npz", allow_pickle=False) as archive:
        fem = np.asarray(archive["fem_original"], dtype=np.float32)
        stabilized = np.asarray(archive["stabilized_original"], dtype=np.float32)
        stabilized_fem_phase = np.asarray(archive["stabilized_power_fem_phase"], dtype=np.float32)
        identity = {
            "image_index": int(archive["image_index"]),
            "trace_index": int(archive["trace_index"]),
            "scored_frame": int(archive["scored_frame"]),
        }
    conditions: dict[tuple[int, str], np.ndarray] = {}
    convergence_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        fem_surrogate, fem_convergence = iaaft_3d(fem, seed, iterations)
        stabilized_surrogate, stabilized_convergence = iaaft_3d(stabilized, seed + 1000, iterations)
        fem_convergence["condition"] = "fem_power_iaaft_phase"
        stabilized_convergence["condition"] = "stabilized_power_iaaft_phase"
        convergence_frames.extend((fem_convergence, stabilized_convergence))
        conditions[(seed, "fem_original")] = fem
        conditions[(seed, "stabilized_original")] = stabilized
        conditions[(seed, "stabilized_power_fem_phase")] = stabilized_fem_phase
        conditions[(seed, "fem_power_iaaft_phase")] = fem_surrogate
        conditions[(seed, "stabilized_power_iaaft_phase")] = stabilized_surrogate
    return conditions, pd.concat(convergence_frames, ignore_index=True), identity


def natural_input_audits(conditions: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(distribution_audit(
            conditions[(seed, "fem_original")],
            conditions[(seed, "fem_power_iaaft_phase")],
            seed=seed,
            condition="fem_power_iaaft_phase",
            source_kind="natural map-support cube",
        ))
        rows.append(distribution_audit(
            conditions[(seed, "stabilized_original")],
            conditions[(seed, "stabilized_power_iaaft_phase")],
            seed=seed,
            condition="stabilized_power_iaaft_phase",
            source_kind="natural map-support cube",
        ))
    return pd.DataFrame(rows)


def make_input_movie(conditions: dict[tuple[int, str], np.ndarray], path: Path, dpi: int) -> None:
    seed = MAP_SEED
    fem = conditions[(seed, "fem_original")]
    fem_surrogate = conditions[(seed, "fem_power_iaaft_phase")]
    stabilized = conditions[(seed, "stabilized_original")]
    stabilized_surrogate = conditions[(seed, "stabilized_power_iaaft_phase")]
    pairs = ((fem, fem_surrogate, "FEM"), (stabilized, stabilized_surrogate, "stabilized"))
    difference_limit = max(
        float(np.quantile(np.abs(fem_surrogate - fem), 0.995)),
        float(np.quantile(np.abs(stabilized_surrogate - stabilized), 0.995)),
        1e-6,
    )
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    images = []
    for row, (original, surrogate, label) in enumerate(pairs):
        images.extend([
            axes[row, 0].imshow(original[0], cmap="gray", vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower"),
            axes[row, 1].imshow(surrogate[0], cmap="gray", vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower"),
            axes[row, 2].imshow(
                surrogate[0] - original[0], cmap="RdBu_r", origin="lower",
                norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0.0, vmax=difference_limit),
            ),
        ])
        axes[row, 0].set_ylabel(label)
    for column, title in enumerate(("original", "distribution-constrained phase", "surrogate − original")):
        axes[0, column].set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    title = figure.suptitle("3-D IAAFT history cube: frame 1/32")

    def update(frame: int):
        index = 0
        for original, surrogate, _ in pairs:
            for values in (original[frame], surrogate[frame], surrogate[frame] - original[frame]):
                images[index].set_data(values)
                index += 1
        title.set_text(f"3-D IAAFT history cube: frame {frame + 1}/32; seed {seed}")
        return [*images, title]

    writer = animation.FFMpegWriter(fps=12, bitrate=2200)
    animation.FuncAnimation(figure, update, frames=fem.shape[0], interval=80, blit=False).save(
        path, writer=writer, dpi=dpi
    )
    plt.close(figure)


def plot_input_contact_sheet(conditions: dict[tuple[int, str], np.ndarray], path: Path, dpi: int) -> None:
    seed = MAP_SEED
    frames = (0, 8, 16, 24, 31)
    names = ("fem_original", "fem_power_iaaft_phase", "stabilized_original", "stabilized_power_iaaft_phase")
    figure, axes = plt.subplots(len(names), len(frames), figsize=(15, 10), constrained_layout=True)
    for row, name in enumerate(names):
        cube = conditions[(seed, name)]
        for column, frame in enumerate(frames):
            axes[row, column].imshow(
                cube[frame], cmap="gray", vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"history frame {frame + 1}")
        axes[row, 0].set_ylabel(CONDITION_LABELS[name])
    figure.suptitle("Distribution-constrained phase control: input frames before neural scoring", weight="bold")
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_input_audit(audit: pd.DataFrame, path: Path, dpi: int) -> None:
    columns = (
        ("histogram_wasserstein_distance", "histogram Wasserstein"),
        ("fraction_outside_canonical_input_range", "fraction outside range"),
        ("patch_kurtosis_wasserstein_distance", "patch-kurtosis Wasserstein"),
        ("spatial_gradient_wasserstein_over_reference_std", "spatial-gradient distance / SD"),
        ("global_energy_weighted_phase_coherence", "energy-weighted phase coherence"),
        ("tile_energy_total_variation_from_reference", "tiled-energy total variation"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    labels = [f"{row.condition.replace('_power_iaaft_phase', '')}\nseed {row.seed}" for row in audit.itertuples()]
    colors = ["#0072B2" if row.condition.startswith("fem") else "#E69F00" for row in audit.itertuples()]
    for axis, (column, title) in zip(axes.ravel(), columns, strict=True):
        axis.bar(np.arange(len(audit)), audit[column], color=colors)
        axis.set_xticks(np.arange(len(audit)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(title)
    figure.suptitle(
        "Stage 2A distribution-constrained input audit\nexact global power and marginal distribution do not imply preserved local energy",
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def load_grating_cubes(source: Path) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    selected = pd.read_csv(source / "selected_input_windows.csv")
    cubes: dict[int, np.ndarray] = {}
    with np.load(source / "selected_retinal_movies_and_power.npz", allow_pickle=False) as archive:
        for row in selected.itertuples(index=False):
            window = int(row.window_index)
            movie = np.asarray(archive[f"window_{window:04d}_movie_uint8"], dtype=np.float32)
            cubes[window] = (movie[-32:] - 127.0) / 255.0
    return cubes, selected


def make_grating_surrogates(
    source: Path, iterations: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame, pd.DataFrame]:
    originals, selected = load_grating_cubes(source)
    surrogates: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    convergence: list[pd.DataFrame] = []
    for ordinal, row in enumerate(selected.itertuples(index=False)):
        window = int(row.window_index)
        seed = 20261000 + ordinal
        surrogate, frame = iaaft_3d(originals[window], seed, iterations)
        frame["condition"] = f"recorded_grating_window_{window:04d}"
        convergence.append(frame)
        surrogates[window] = surrogate
        rows.append(distribution_audit(
            originals[window], surrogate, seed=seed,
            condition=f"recorded_grating_window_{window:04d}",
            source_kind="validated recorded-grating input cube",
        ))
    return originals, surrogates, pd.DataFrame(rows), pd.concat(convergence, ignore_index=True)


def score_recorded_gratings(
    scorer: CanonicalTwinScorer,
    originals: dict[int, np.ndarray],
    surrogates: dict[int, np.ndarray],
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    windows = selected.window_index.astype(int).tolist()
    cubes = np.stack([originals[w] for w in windows] + [surrogates[w] for w in windows])[:, None]
    model = scorer.ctx.model.model
    dataset_index = scorer.model_names.index(str(selected.session.iloc[0]))
    recurrent: list[np.ndarray] = []
    hook = model.recurrent.register_forward_hook(
        lambda _module, _inputs, output: recurrent.append(output.detach().cpu().numpy())
    )
    responses: list[np.ndarray] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for cube in cubes:
            response = model(torch.from_numpy(cube[None]).to(device), dataset_idx=dataset_index)
            responses.append(response.detach().cpu().numpy()[0])
    hook.remove()
    rates = np.stack(responses)
    features = np.concatenate(recurrent, axis=0)
    n = len(windows)
    original_rates, surrogate_rates = rates[:n], rates[n:]
    original_features, surrogate_features = features[:n], features[n:]
    rows: list[dict[str, Any]] = []
    for ordinal, window in enumerate(windows):
        rows.append({
            "window_index": int(window),
            "selection_role": str(selected.iloc[ordinal].selection_role),
            "original_mean_rate": float(original_rates[ordinal].mean()),
            "surrogate_mean_rate": float(surrogate_rates[ordinal].mean()),
            "mean_rate_ratio": float(
                surrogate_rates[ordinal].mean() / max(float(original_rates[ordinal].mean()), np.finfo(float).tiny)
            ),
            "unit_rate_pearson": safe_correlation(original_rates[ordinal], surrogate_rates[ordinal]),
            "unit_rate_spearman": safe_correlation(original_rates[ordinal], surrogate_rates[ordinal], rank=True),
            "recurrent_feature_cosine": cosine_similarity(original_features[ordinal], surrogate_features[ordinal]),
            "recurrent_feature_rms_ratio": float(
                np.sqrt(np.mean(surrogate_features[ordinal] ** 2))
                / max(float(np.sqrt(np.mean(original_features[ordinal] ** 2))), np.finfo(float).tiny)
            ),
            "recurrent_feature_map_pearson": safe_correlation(
                original_features[ordinal], surrogate_features[ordinal]
            ),
        })
    unit_rank = np.asarray([
        safe_correlation(original_rates[:, unit], surrogate_rates[:, unit], rank=True)
        for unit in range(original_rates.shape[1])
    ])
    summary = {
        "session": str(selected.session.iloc[0]),
        "n_windows": n,
        "n_session_units": int(original_rates.shape[1]),
        "overall_mean_rate_ratio": float(surrogate_rates.mean() / max(float(original_rates.mean()), np.finfo(float).tiny)),
        "flattened_rate_pearson": safe_correlation(original_rates, surrogate_rates),
        "flattened_rate_spearman": safe_correlation(original_rates, surrogate_rates, rank=True),
        "median_per_unit_three_window_spearman": float(np.nanmedian(unit_rank)),
        "preferred_window_agreement_fraction": float(
            np.mean(np.argmax(original_rates, axis=0) == np.argmax(surrogate_rates, axis=0))
        ),
        "median_recurrent_feature_rms_ratio": float(
            np.median([row["recurrent_feature_rms_ratio"] for row in rows])
        ),
    }
    summary["gate_rate_scale"] = 0.5 <= summary["overall_mean_rate_ratio"] <= 2.0
    summary["gate_flattened_tuning_rank"] = summary["flattened_rate_spearman"] >= 0.5
    summary["gate_median_unit_tuning_rank"] = summary["median_per_unit_three_window_spearman"] >= 0.5
    summary["gate_recurrent_feature_scale"] = 0.5 <= summary["median_recurrent_feature_rms_ratio"] <= 2.0
    summary["generic_degradation_gate_pass"] = bool(
        summary["gate_rate_scale"]
        and summary["gate_flattened_tuning_rank"]
        and summary["gate_median_unit_tuning_rank"]
        and summary["gate_recurrent_feature_scale"]
    )
    unit_table = pd.DataFrame({
        "session_unit_index": np.arange(original_rates.shape[1]),
        "three_window_tuning_spearman": unit_rank,
        "preferred_window_original": np.argmax(original_rates, axis=0),
        "preferred_window_surrogate": np.argmax(surrogate_rates, axis=0),
    })
    return pd.DataFrame(rows), unit_table, summary


def plot_grating_calibration(rows: pd.DataFrame, summary: dict[str, Any], path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(rows.original_mean_rate, rows.surrogate_mean_rate, s=70)
    limit = max(float(rows[["original_mean_rate", "surrogate_mean_rate"]].to_numpy().max()), 1e-6)
    axes[0].plot([0, limit], [0, limit], linestyle="--", color="0.4")
    axes[0].set(xlabel="original mean rate", ylabel="IAAFT mean rate", title="Rate scale")
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, rows.unit_rate_spearman, width=0.36, label="unit-rate rank")
    axes[1].bar(x + 0.18, rows.recurrent_feature_map_pearson, width=0.36, label="feature-map correlation")
    axes[1].set_xticks(x, rows.selection_role, rotation=30, ha="right")
    axes[1].set_ylim(-1, 1)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Window-level preservation")
    axes[2].axis("off")
    text = "\n".join([
        f"overall rate ratio: {summary['overall_mean_rate_ratio']:.3f}",
        f"flattened rate Spearman: {summary['flattened_rate_spearman']:.3f}",
        f"median unit tuning Spearman: {summary['median_per_unit_three_window_spearman']:.3f}",
        f"preferred-window agreement: {summary['preferred_window_agreement_fraction']:.3f}",
        f"median recurrent RMS ratio: {summary['median_recurrent_feature_rms_ratio']:.3f}",
        f"generic-degradation gate: {'PASS' if summary['generic_degradation_gate_pass'] else 'FAIL'}",
    ])
    axes[2].text(0.02, 0.95, text, va="top", fontsize=12, family="monospace")
    figure.suptitle("Recorded-grating calibration of the IAAFT manipulation", weight="bold")
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def score_natural_maps(
    scorer: CanonicalTwinScorer,
    conditions: dict[tuple[int, str], np.ndarray],
    selected_units: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    names = list(CONDITION_LABELS)
    cubes = np.stack([conditions[(MAP_SEED, name)] for name in names])[:, None]
    frontend: list[np.ndarray] = []
    model = scorer.ctx.model.model
    hook = model.frontend.register_forward_hook(
        lambda _module, _inputs, output: frontend.append(output.detach().cpu().numpy())
    )
    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    full = scorer._compute_rate_map_batched(torch.from_numpy(cubes.astype(np.float32)))
    hook.remove()
    maps = apply_population_view(full, view).clamp_min(0.0).cpu().numpy().astype(np.float32)
    features = np.concatenate(frontend, axis=0)
    units = selected_units.rr100_index.astype(int).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    reference_by_name = {
        "stabilized_original": "stabilized_original",
        "stabilized_power_iaaft_phase": "stabilized_original",
        "fem_original": "fem_original",
        "fem_power_iaaft_phase": "fem_original",
        "stabilized_power_fem_phase": "stabilized_original",
    }
    for condition_index, name in enumerate(names):
        for role_index, unit in enumerate(units):
            metric_rows.append({
                "seed": MAP_SEED,
                "condition": name,
                "condition_label": CONDITION_LABELS[name],
                "rr100_index": int(unit),
                "selection_role": selected_units.iloc[role_index].selection_role,
                **map_metrics(maps[condition_index, unit]),
            })
        reference_index = names.index(reference_by_name[name])
        feature_rows.append({
            "condition": name,
            "reference_condition": reference_by_name[name],
            "frontend_cosine": cosine_similarity(features[reference_index], features[condition_index]),
            "frontend_pearson": safe_correlation(features[reference_index], features[condition_index]),
            "frontend_rms_ratio": float(
                np.sqrt(np.mean(features[condition_index] ** 2))
                / max(float(np.sqrt(np.mean(features[reference_index] ** 2))), np.finfo(float).tiny)
            ),
        })
    return maps[:, units], pd.DataFrame(metric_rows), pd.DataFrame(feature_rows)


def plot_raw_maps(
    maps: np.ndarray, units: pd.DataFrame, metrics: pd.DataFrame, path: Path, dpi: int
) -> None:
    names = list(CONDITION_LABELS)
    values = np.transpose(maps, (1, 0, 2, 3))
    reference_columns = np.asarray([0, 0, 2, 2, 0], dtype=int)
    differences = np.stack([
        values[:, column] - values[:, reference_columns[column]] for column in range(len(names))
    ], axis=1)
    figure, axes = plt.subplots(len(units), 2 * len(names), figsize=(27, 3.0 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units.itertuples(index=False)):
        rate_limit = max(float(np.quantile(values[row], 0.995)), 1e-6)
        diff_limit = max(float(np.quantile(np.abs(differences[row, 1:]), 0.995)), 1e-6)
        for column, name in enumerate(names):
            record = metrics.loc[
                metrics.condition.eq(name) & metrics.rr100_index.eq(int(unit.rr100_index))
            ].iloc[0]
            axes[row, column].imshow(values[row, column], cmap="magma", vmin=0, vmax=rate_limit, origin="lower")
            axes[row, column].set_title(
                f"{CONDITION_LABELS[name]}\n{record.instantaneous_mean_rate_hz:.3f} Hz; "
                f"SSI {record.instantaneous_ssi_bits_per_spike:.3f}", fontsize=7.5
            )
            axes[row, len(names) + column].imshow(
                differences[row, column], cmap="RdBu_r", origin="lower",
                norm=TwoSlopeNorm(vmin=-diff_limit, vcenter=0, vmax=diff_limit),
            )
            axes[row, len(names) + column].set_title(
                f"difference from {CONDITION_LABELS[names[reference_columns[column]]]}", fontsize=7.5
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(
            f"RR100 {int(unit.rr100_index)}\n{unit.selection_role}", fontsize=8
        )
    figure.suptitle(
        f"Stage 2A distribution-constrained raw-map checkpoint; seed {MAP_SEED}\n"
        "per-unit scales shared across conditions; descriptive only",
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conditions, convergence, identity = load_natural_conditions(args.source_v1, int(args.iterations))
    natural_audit = natural_input_audits(conditions)
    phase_support = phase_source_support_audit(conditions)
    convergence.to_csv(args.out_dir / "iaaft_convergence.csv", index=False)
    natural_audit.to_csv(args.out_dir / "natural_input_distribution_phase_audit.csv", index=False)
    phase_support.to_csv(args.out_dir / "symmetric_phase_source_support_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "distribution_constrained_input_cubes.npz",
        seeds=np.asarray(SEEDS, dtype=np.int64),
        image_index=np.asarray(identity["image_index"]),
        trace_index=np.asarray(identity["trace_index"]),
        scored_frame=np.asarray(identity["scored_frame"]),
        fem_original=conditions[(MAP_SEED, "fem_original")],
        stabilized_original=conditions[(MAP_SEED, "stabilized_original")],
        stabilized_power_fem_phase=conditions[(MAP_SEED, "stabilized_power_fem_phase")],
        fem_power_iaaft_phase=np.stack([conditions[(seed, "fem_power_iaaft_phase")] for seed in SEEDS]),
        stabilized_power_iaaft_phase=np.stack([
            conditions[(seed, "stabilized_power_iaaft_phase")] for seed in SEEDS
        ]),
    )
    make_input_movie(conditions, args.out_dir / "01_distribution_constrained_input_movie.mp4", int(args.dpi))
    plot_input_contact_sheet(conditions, args.out_dir / "02_input_contact_sheet", int(args.dpi))
    plot_input_audit(natural_audit, args.out_dir / "03_input_validation", int(args.dpi))

    grating_originals, grating_surrogates, grating_audit, grating_convergence = make_grating_surrogates(
        args.grating_source, int(args.iterations)
    )
    grating_audit.to_csv(args.out_dir / "recorded_grating_input_audit.csv", index=False)
    grating_convergence.to_csv(args.out_dir / "recorded_grating_iaaft_convergence.csv", index=False)
    _, selected_gratings = load_grating_cubes(args.grating_source)

    scorer = CanonicalTwinScorer(
        device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True
    )
    grating_rows, grating_units, grating_summary = score_recorded_gratings(
        scorer, grating_originals, grating_surrogates, selected_gratings
    )
    grating_rows.to_csv(args.out_dir / "recorded_grating_generic_degradation_by_window.csv", index=False)
    grating_units.to_csv(args.out_dir / "recorded_grating_unit_tuning_rank_audit.csv", index=False)
    (args.out_dir / "recorded_grating_generic_degradation_summary.json").write_text(
        json.dumps(grating_summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_grating_calibration(
        grating_rows, grating_summary, args.out_dir / "04_recorded_grating_calibration", int(args.dpi)
    )

    natural_input_gate = bool(natural_audit.distribution_and_phase_gate_pass.all())
    grating_input_gate = bool(grating_audit.distribution_and_phase_gate_pass.all())
    generic_gate = bool(grating_summary["generic_degradation_gate_pass"])
    raw_map_completed = natural_input_gate and grating_input_gate and generic_gate
    model_metadata: dict[str, Any] = {
        "device": str(next(scorer.ctx.model.model.parameters()).device),
        "rr100_version": RR100_MOVIE_MEDOID_VERSION,
    }
    if raw_map_completed:
        units = pd.read_csv(args.source_v1 / "selected_unit_roles_pre_response.csv")
        maps, map_metrics_table, early_features = score_natural_maps(scorer, conditions, units)
        map_metrics_table.to_csv(args.out_dir / "selected_unit_condition_map_metrics.csv", index=False)
        early_features.to_csv(args.out_dir / "natural_early_frozen_representation_audit.csv", index=False)
        np.savez_compressed(
            args.out_dir / "selected_unit_activation_maps.npz",
            rr100_index=units.rr100_index.astype(int).to_numpy(),
            condition_names=np.asarray(list(CONDITION_LABELS), dtype="U64"),
            seed=np.asarray(MAP_SEED),
            activation_maps=maps,
        )
        units.to_csv(args.out_dir / "selected_unit_roles_pre_response.csv", index=False)
        plot_raw_maps(maps, units, map_metrics_table, args.out_dir / "05_raw_activation_maps", int(args.dpi))

    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    status = (
        "input_grating_and_raw_map_checkpoint_complete"
        if raw_map_completed
        else "stopped_before_raw_maps_because_a_predeclared_gate_failed"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_map_support_stage2a_distribution_constrained_phase_v1",
        "status": status,
        "scope": {
            **identity,
            "phase_seeds": list(SEEDS),
            "map_seed": MAP_SEED,
            "iaaft_iterations": int(args.iterations),
            "development_only": True,
            "population_inference": False,
        },
        "construction": {
            "method": "independent full-cube three-dimensional IAAFT at each power level",
            "last_projection": "exact target Fourier amplitude",
            "distribution_projection": "stable rank assignment to the exact corresponding original value multiset",
            "phase_pairing": (
                "FEM and stabilized IAAFT phases are optimized independently; this is not a shared-phase factorial"
            ),
            "reason_shared_phase_not_used": (
                "a paired shared-phase pilot was unstable and produced extreme FEM outliers; it is not an accepted control"
            ),
            "no_clipping": True,
        },
        "gates": {
            "natural_input_distribution_and_phase": natural_input_gate,
            "recorded_grating_input_distribution_and_phase": grating_input_gate,
            "recorded_grating_generic_degradation": generic_gate,
            "raw_map_checkpoint_authorized_and_completed": raw_map_completed,
        },
        "recorded_grating_summary": grating_summary,
        "model": model_metadata,
        "sources": {
            "completed_stage2a_v1_cubes": file_identity(args.source_v1 / "factorial_input_cubes.npz"),
            "completed_stage2a_v1_units": file_identity(args.source_v1 / "selected_unit_roles_pre_response.csv"),
            "completed_stage2a_method_audit": file_identity(args.method_audit_v1 / "manifest.json"),
            "recorded_grating_movies": file_identity(
                args.grating_source / "selected_retinal_movies_and_power.npz"
            ),
            "runner": file_identity(Path(__file__)),
        },
        "artifacts": {
            "movie": "01_distribution_constrained_input_movie.mp4",
            "contact_sheet": "02_input_contact_sheet.pdf",
            "input_validation": "03_input_validation.pdf",
            "grating_calibration": "04_recorded_grating_calibration.pdf",
            "raw_maps": "05_raw_activation_maps.pdf" if raw_map_completed else None,
        },
        "decision_gate": (
            "stop for human inspection; do not add frames, units, seeds, or population inference until the input movie, "
            "distribution audit, recorded-grating calibration, and raw maps are reviewed"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = [
        "# Stage 2A distribution-constrained phase workstream",
        "",
        "This workstream leaves the completed unconstrained Stage 2A v1 runner and artifacts untouched. It uses ",
        "independent full-cube 3-D IAAFT projections at FEM and stabilized power. The last projection preserves ",
        "the exact target Fourier amplitude; iterative rank projections constrain each surrogate to the original ",
        "value distribution. The two phase fields are not shared, so these conditions are not presented as a ",
        "shared-phase factorial.",
        "",
        f"Input distribution/phase gate: **{'PASS' if natural_input_gate else 'FAIL'}**.  ",
        f"Recorded-grating input gate: **{'PASS' if grating_input_gate else 'FAIL'}**.  ",
        f"Recorded-grating generic-degradation gate: **{'PASS' if generic_gate else 'FAIL'}**.  ",
        f"Raw-map checkpoint: **{'completed' if raw_map_completed else 'not run'}**.",
        "",
        "This is a human checkpoint, not a population result. Local-energy redistribution is reported as an ",
        "effect of phase destruction, not treated as a failure of the exact global 3-D amplitude contract. ",
        "Marginal and recorded-grating gate passage is evidence against two specific confounds, not proof that ",
        "the surrogate lies on the natural-movie training distribution.",
        "",
        "See `manifest.json` for provenance and the precise stop rule.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

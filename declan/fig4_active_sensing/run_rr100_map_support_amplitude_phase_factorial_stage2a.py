#!/usr/bin/env python3
"""Stage 2A map-support amplitude-by-phase factorial on one development input."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import render_scored_embedding
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import RR100_MOVIE_MEDOID_VERSION
from declan.fig4_active_sensing.spectral_cache_contract import sha256
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE2 = ROOT / "outputs/fig4_active_sensing/rr100_clean_history_whole_movie_power_stage2_v1"
DEFAULT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
DEFAULT_TUNING = ROOT / "outputs/fig4_active_sensing/rr100_grating_only_orientation_tuning_v1"
DEFAULT_ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_v1"
N_HISTORY = 32
SCORED_FRAME = 20
DT = 1.0 / 120.0
EPS = np.finfo(np.float64).tiny
SEEDS = (20260831, 20260832, 20260833)
CONDITION_LABELS = {
    "stabilized_original": "original stabilized input",
    "fem_original": "original FEM input",
    "fem_power_random_phase": "FEM power with shared random phase",
    "stabilized_power_fem_phase": "stabilized power with FEM phase",
    "stabilized_power_random_phase": "stabilized power with shared random phase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--response-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scored-frame", type=int, default=SCORED_FRAME)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def choose_development_condition(stage2_dir: Path) -> pd.Series:
    condition = pd.read_csv(stage2_dir / "development_condition_index.csv")
    with np.load(stage2_dir / "development_predictors_and_first_split_predictions.npz", allow_pickle=False) as archive:
        row = np.asarray(archive["matrix_row_index"], dtype=int)
        amplitude = np.asarray(archive["global_supported_dynamic_power_amplitude"], dtype=float)
    if not np.array_equal(row, condition.matrix_row_index.to_numpy(int)):
        raise ValueError("Stage-2 predictor rows do not match the development condition table")
    target = float(np.median(amplitude))
    ordinal = int(np.argmin(np.abs(amplitude - target)))
    selected = condition.iloc[ordinal].copy()
    selected["selection_role"] = "median whole-movie supported dynamic-power development condition"
    selected["selection_criterion"] = "minimum absolute distance to the development median power amplitude"
    selected["selection_value"] = float(amplitude[ordinal])
    selected["development_median_power_amplitude"] = target
    return selected


def orientation_vector_strength(tuning_dir: Path) -> pd.DataFrame:
    with np.load(tuning_dir / "grating_only_orientation_tuning.npz", allow_pickle=False) as archive:
        units = np.asarray(archive["rr100_index"], dtype=int)
        orientations = np.asarray(archive["measured_grating_orientation_deg"], dtype=float)
        positive = np.asarray(archive["measured_positive_f0_hz"], dtype=float)
    marginal = positive.mean(axis=(1, 2))
    vector = np.exp(2j * np.deg2rad(orientations))
    strength = np.abs(marginal @ vector) / np.maximum(marginal.sum(axis=1), 1e-12)
    preferred = orientations[np.argmax(marginal, axis=1)]
    return pd.DataFrame({
        "rr100_index": units,
        "grating_orientation_vector_strength": strength,
        "preferred_grating_orientation_deg": preferred,
    })


def select_units(assignments_path: Path, tuning_dir: Path) -> pd.DataFrame:
    assignments = pd.read_csv(assignments_path)
    eligible = assignments.loc[assignments.recorded_validation_pass.astype(bool)].merge(
        orientation_vector_strength(tuning_dir), on="rr100_index", how="left", validate="one_to_one"
    )
    selected: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, frame: pd.DataFrame, score: pd.Series, criterion: str, maximum: bool) -> None:
        available = frame.loc[~frame.rr100_index.astype(int).isin(used)].copy()
        values = score.reindex(available.index)
        index = values.idxmax() if maximum else values.idxmin()
        row = eligible.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(score.loc[index])
        row["selection_is_algorithmic_and_pre_response"] = True
        selected.append(row)
        used.add(int(row.rr100_index))

    for group, label in (("sf_low_half", "low spatial-frequency"), ("sf_high_half", "high spatial-frequency")):
        frame = eligible.loc[eligible.sf_outer_third.eq(group)]
        median = float(frame.preferred_sf_cpd.median())
        add(
            f"{label} representative", frame, -(frame.preferred_sf_cpd - median).abs(),
            "closest to the recorded-validated half's median preferred spatial frequency", True,
        )
        add(
            f"{label} extreme control", frame, frame.preferred_sf_cpd,
            "minimum preferred spatial frequency" if group == "sf_low_half" else "maximum preferred spatial frequency",
            group == "sf_high_half",
        )
    add(
        "strong orientation-selectivity example", eligible,
        eligible.grating_orientation_vector_strength,
        "maximum fixed-retina grating orientation vector strength among unused validated units", True,
    )
    add(
        "weak orientation-selectivity control", eligible,
        eligible.grating_orientation_vector_strength,
        "minimum fixed-retina grating orientation vector strength among unused validated units", False,
    )
    return pd.DataFrame(selected).reset_index(drop=True)


def load_development_input(selection: pd.Series, cache_dir: Path) -> tuple[np.ndarray, float, np.ndarray]:
    image_index = int(selection.image_index)
    trace_index = int(selection.trace_index)
    with np.load(cache_dir / "input_cache/images" / f"image_{image_index:03d}.npz", allow_pickle=False) as archive:
        patch = np.asarray(archive["corrected_patch"], dtype=np.float32)
        ppd = float(archive["patch_ppd"].item())
    with np.load(cache_dir / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as archive:
        trace_ids = np.asarray(archive["trace_index"], dtype=int)
        ordinal = np.flatnonzero(trace_ids == trace_index)
        if len(ordinal) != 1:
            raise ValueError(f"Trace {trace_index} is not unique in the corrected trace cache")
        ordinal = int(ordinal[0])
        trace72 = np.concatenate(
            (np.asarray(archive["history_xy_deg"][ordinal]), np.asarray(archive["score_xy_deg"][ordinal])),
            axis=0,
        ).astype(np.float32)
    return patch, ppd, trace72


def phase_unit(spectrum: np.ndarray) -> np.ndarray:
    magnitude = np.abs(spectrum)
    return np.divide(spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 1e-12)


def inverse_with_audit(magnitude: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, float]:
    complex_cube = np.fft.ifftn(magnitude * phase)
    imaginary = float(np.max(np.abs(complex_cube.imag)))
    return complex_cube.real.astype(np.float32), imaginary


def amplitude_relative_error(reference_amplitude: np.ndarray, cube: np.ndarray) -> float:
    observed = np.abs(np.fft.fftn(np.asarray(cube, dtype=np.float64)))
    return float(np.linalg.norm(observed - reference_amplitude) / max(np.linalg.norm(reference_amplitude), EPS))


def hermitian_relative_error(spectrum: np.ndarray) -> float:
    values = np.asarray(spectrum)
    mirrored = np.conj(values)
    for axis, size in enumerate(values.shape):
        mirrored = np.take(mirrored, (-np.arange(size)) % size, axis=axis)
    return float(np.linalg.norm(values - mirrored) / max(np.linalg.norm(values), EPS))


def phase_coherence(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.fft.fftn(np.asarray(reference, dtype=np.float64))
    right = np.fft.fftn(np.asarray(candidate, dtype=np.float64))
    valid = (np.abs(left) > 1e-10) & (np.abs(right) > 1e-10)
    phase_delta = phase_unit(right[valid]) * np.conj(phase_unit(left[valid]))
    return float(np.abs(np.mean(phase_delta)))


def tile_energy(cube: np.ndarray, tiles: int = 5) -> np.ndarray:
    values = np.asarray(cube, dtype=np.float64)
    y_edges = np.linspace(0, values.shape[1], tiles + 1, dtype=int)
    x_edges = np.linspace(0, values.shape[2], tiles + 1, dtype=int)
    energy = np.empty((tiles, tiles), dtype=float)
    for y in range(tiles):
        for x in range(tiles):
            block = values[:, y_edges[y]:y_edges[y + 1], x_edges[x]:x_edges[x + 1]]
            energy[y, x] = float(np.sum(block**2))
    return energy / max(float(energy.sum()), EPS)


def construct_factorial(fem: np.ndarray, stabilized: np.ndarray) -> tuple[dict[tuple[int, str], np.ndarray], pd.DataFrame, pd.DataFrame]:
    fem_spectrum = np.fft.fftn(np.asarray(fem, dtype=np.float64))
    stabilized_spectrum = np.fft.fftn(np.asarray(stabilized, dtype=np.float64))
    fem_amplitude = np.abs(fem_spectrum)
    stabilized_amplitude = np.abs(stabilized_spectrum)
    fem_phase = phase_unit(fem_spectrum)
    conditions: dict[tuple[int, str], np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    canonical_min, canonical_max = -127.0 / 255.0, 128.0 / 255.0

    def record(seed: int, name: str, cube: np.ndarray, target_amplitude: np.ndarray, imaginary: float) -> None:
        conditions[(seed, name)] = np.asarray(cube, dtype=np.float32)
        reference = fem if name in {"fem_original", "fem_power_random_phase"} else stabilized
        audit_rows.append({
            "seed": int(seed), "condition": name, "condition_label": CONDITION_LABELS[name],
            "target_power_level": "FEM" if target_amplitude is fem_amplitude else "stabilized",
            "fourier_amplitude_relative_error": amplitude_relative_error(target_amplitude, cube),
            "maximum_imaginary_reconstruction_residual": imaginary,
            "reconstructed_spectrum_hermitian_relative_error": hermitian_relative_error(np.fft.fftn(cube)),
            "mean": float(np.mean(cube)), "rms": float(np.sqrt(np.mean(np.asarray(cube, dtype=float) ** 2))),
            "standard_deviation": float(np.std(cube)), "minimum": float(np.min(cube)), "maximum": float(np.max(cube)),
            "fraction_outside_canonical_input_range": float(np.mean((cube < canonical_min) | (cube > canonical_max))),
            "histogram_wasserstein_distance_from_same_power_original": float(
                wasserstein_distance(np.asarray(reference).ravel(), np.asarray(cube).ravel())
            ),
            "fourier_phase_retention_coherence_with_fem": phase_coherence(fem, cube),
            "tile_energy_correlation_with_fem": float(np.corrcoef(tile_energy(fem).ravel(), tile_energy(cube).ravel())[0, 1]),
            "tile_energy_correlation_with_stabilized": float(
                np.corrcoef(tile_energy(stabilized).ravel(), tile_energy(cube).ravel())[0, 1]
            ),
        })
        for y, x in np.ndindex(tile_energy(cube).shape):
            tile_rows.append({"seed": int(seed), "condition": name, "tile_y": y, "tile_x": x,
                              "fraction_of_cube_energy": float(tile_energy(cube)[y, x])})

    stabilized_fem_phase, stabilized_fem_imaginary = inverse_with_audit(stabilized_amplitude, fem_phase)
    for seed in SEEDS:
        rng = np.random.default_rng(int(seed))
        random_spectrum = np.fft.fftn(rng.standard_normal(fem.shape))
        random_phase = phase_unit(random_spectrum)
        random_phase.flat[0] = fem_phase.flat[0]
        fem_random, fem_random_imaginary = inverse_with_audit(fem_amplitude, random_phase)
        stabilized_random, stabilized_random_imaginary = inverse_with_audit(stabilized_amplitude, random_phase)
        record(seed, "stabilized_original", stabilized, stabilized_amplitude, 0.0)
        record(seed, "fem_original", fem, fem_amplitude, 0.0)
        record(seed, "fem_power_random_phase", fem_random, fem_amplitude, fem_random_imaginary)
        record(seed, "stabilized_power_fem_phase", stabilized_fem_phase, stabilized_amplitude, stabilized_fem_imaginary)
        record(seed, "stabilized_power_random_phase", stabilized_random, stabilized_amplitude, stabilized_random_imaginary)
    return conditions, pd.DataFrame(audit_rows), pd.DataFrame(tile_rows)


def unique_condition_stack(conditions: dict[tuple[int, str], np.ndarray]) -> tuple[list[tuple[int, str]], np.ndarray]:
    keys: list[tuple[int, str]] = []
    cubes: list[np.ndarray] = []
    seen: set[tuple[str, int]] = set()
    for seed in SEEDS:
        for name in CONDITION_LABELS:
            identity = (name, 0 if name in {"stabilized_original", "fem_original", "stabilized_power_fem_phase"} else seed)
            if identity in seen:
                continue
            seen.add(identity)
            keys.append((seed, name))
            cubes.append(conditions[(seed, name)])
    return keys, np.stack(cubes)[:, None]


def score_maps(cubes: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, dict[str, object]]:
    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    scorer = CanonicalTwinScorer(device=device, batch_size=batch_size, empty_cache_every_batch=True)
    stimulus = torch.from_numpy(np.asarray(cubes, dtype=np.float32))
    full = scorer._compute_rate_map_batched(stimulus)
    rr100 = apply_population_view(full, view).clamp_min(0.0)
    maps = rr100.detach().cpu().numpy().astype(np.float32, copy=False)
    metadata = {
        "device": str(next(scorer.ctx.model.model.parameters()).device),
        "full_population_map_shape": list(full.shape), "rr100_map_shape": list(maps.shape),
        "rr100_version": RR100_MOVIE_MEDOID_VERSION,
    }
    del stimulus, full, rr100, scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return maps, metadata


def map_metrics(rate_map: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(rate_map, dtype=np.float64), 0.0)
    rate = float(values.mean())
    gain = values / max(rate, EPS)
    ssi = float(np.mean(gain * np.log2(np.maximum(gain, EPS))))
    expected = rate * DT
    numerator = ssi * expected
    return {
        "instantaneous_mean_rate_hz": rate,
        "instantaneous_ssi_bits_per_spike": ssi,
        "expected_spikes_in_frame": expected,
        "information_numerator_bits_spikes_in_frame": numerator,
        "information_rate_bits_per_second": numerator / DT,
    }


def expand_scored_maps(
    keys: list[tuple[int, str]], unique_maps: np.ndarray, conditions: dict[tuple[int, str], np.ndarray]
) -> dict[tuple[int, str], np.ndarray]:
    lookup: dict[tuple[str, int], np.ndarray] = {}
    for key, maps in zip(keys, unique_maps, strict=True):
        seed, name = key
        identity = (name, 0 if name in {"stabilized_original", "fem_original", "stabilized_power_fem_phase"} else seed)
        lookup[identity] = maps
    expanded = {}
    for seed, name in conditions:
        identity = (name, 0 if name in {"stabilized_original", "fem_original", "stabilized_power_fem_phase"} else seed)
        expanded[(seed, name)] = lookup[identity]
    return expanded


def response_tables(
    maps: dict[tuple[int, str], np.ndarray], selected_units: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected = selected_units.rr100_index.to_numpy(int)
    for (seed, condition), all_maps in maps.items():
        for role_ordinal, unit in enumerate(selected):
            rows.append({
                "seed": int(seed), "condition": condition, "condition_label": CONDITION_LABELS[condition],
                "rr100_index": int(unit), "selection_role": selected_units.iloc[role_ordinal].selection_role,
                **map_metrics(all_maps[int(unit)]),
            })
    metrics = pd.DataFrame(rows)
    effects: list[dict[str, Any]] = []
    definitions = {
        "original FEM minus original stabilized": ("fem_original", "stabilized_original"),
        "phase effect at FEM power": ("fem_original", "fem_power_random_phase"),
        "phase effect at stabilized power": ("stabilized_power_fem_phase", "stabilized_power_random_phase"),
        "power effect under FEM phase": ("fem_original", "stabilized_power_fem_phase"),
        "power effect under shared random phase": ("fem_power_random_phase", "stabilized_power_random_phase"),
    }
    measure_columns = [
        "instantaneous_mean_rate_hz", "instantaneous_ssi_bits_per_spike", "expected_spikes_in_frame",
        "information_numerator_bits_spikes_in_frame", "information_rate_bits_per_second",
    ]
    for (seed, unit), frame in metrics.groupby(["seed", "rr100_index"]):
        indexed = frame.set_index("condition")
        role = frame.selection_role.iloc[0]
        effect_values: dict[tuple[str, str], float] = {}
        for label, (left, right) in definitions.items():
            record: dict[str, Any] = {"seed": int(seed), "rr100_index": int(unit), "selection_role": role,
                                      "effect": label, "left_condition": left, "right_condition": right}
            for measure in measure_columns:
                value = float(indexed.loc[left, measure] - indexed.loc[right, measure])
                record[f"difference__{measure}"] = value
                effect_values[(label, measure)] = value
            effects.append(record)
        interaction = {"seed": int(seed), "rr100_index": int(unit), "selection_role": role,
                       "effect": "power-by-phase interaction", "left_condition": "difference of phase effects",
                       "right_condition": "not a single condition"}
        for measure in measure_columns:
            interaction[f"difference__{measure}"] = (
                effect_values[("phase effect at FEM power", measure)]
                - effect_values[("phase effect at stabilized power", measure)]
            )
        effects.append(interaction)
    return metrics, pd.DataFrame(effects)


def plot_inputs(
    conditions: dict[tuple[int, str], np.ndarray], audit: pd.DataFrame, selection: pd.Series,
    trace72: np.ndarray, path: Path, dpi: int,
) -> None:
    names = list(CONDITION_LABELS)
    fig, axes = plt.subplots(len(SEEDS), len(names), figsize=(18, 10), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for column, name in enumerate(names):
            cube = conditions[(seed, name)]
            record = audit.loc[audit.seed.eq(seed) & audit.condition.eq(name)].iloc[0]
            image = axes[row, column].imshow(cube[0], cmap="gray", vmin=-127 / 255, vmax=128 / 255, origin="lower")
            axes[row, column].set_title(
                f"{CONDITION_LABELS[name]}\nrange [{record.minimum:+.2f}, {record.maximum:+.2f}]; "
                f"outside={100*record.fraction_outside_canonical_input_range:.1f}%",
                fontsize=8,
            )
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"phase seed {seed}", fontsize=9)
    fig.colorbar(image, ax=axes, label="normalized twin input value; shared canonical display range", shrink=0.7)
    fig.suptitle(
        f"Stage 2A input checkpoint: complete 32-lag map inputs at scored frame {SCORED_FRAME}\n"
        f"development image {int(selection.image_index)}, trace {int(selection.trace_index)}; exact raw Fourier amplitude; no clipping",
        fontsize=14, weight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_maps(
    maps: dict[tuple[int, str], np.ndarray], units: pd.DataFrame, metrics: pd.DataFrame,
    path: Path, dpi: int, *, global_scale: bool,
) -> None:
    seed = SEEDS[0]
    names = list(CONDITION_LABELS)
    selected = units.rr100_index.to_numpy(int)
    values = np.stack([maps[(seed, name)][selected] for name in names], axis=1)
    differences = values - values[:, :1]
    if global_scale:
        rate_limits = np.full(len(selected), max(float(np.quantile(values, 0.995)), 1e-6))
        difference_limits = np.full(len(selected), max(float(np.quantile(np.abs(differences[:, 1:]), 0.995)), 1e-6))
    else:
        rate_limits = np.maximum(np.quantile(values, 0.995, axis=(1, 2, 3)), 1e-6)
        difference_limits = np.maximum(np.quantile(np.abs(differences[:, 1:]), 0.995, axis=(1, 2, 3)), 1e-6)
    fig, axes = plt.subplots(len(selected), 9, figsize=(27, 3.15 * len(selected)), constrained_layout=True)
    for row, unit in enumerate(selected):
        for column, name in enumerate(names):
            metric = metrics.loc[
                metrics.seed.eq(seed) & metrics.condition.eq(name) & metrics.rr100_index.eq(unit)
            ].iloc[0]
            rate_image = axes[row, column].imshow(
                values[row, column], cmap="magma", vmin=0, vmax=rate_limits[row], origin="lower"
            )
            axes[row, column].set_title(
                f"{CONDITION_LABELS[name]}\nrate={metric.instantaneous_mean_rate_hz:.2f} Hz; "
                f"SSI={metric.instantaneous_ssi_bits_per_spike:.3f}", fontsize=7.5,
            )
        for difference_column, condition_column in enumerate(range(1, len(names)), start=len(names)):
            name = names[condition_column]
            diff_image = axes[row, difference_column].imshow(
                differences[row, condition_column], cmap="RdBu_r", origin="lower",
                norm=TwoSlopeNorm(vmin=-difference_limits[row], vcenter=0, vmax=difference_limits[row]),
            )
            axes[row, difference_column].set_title(f"{CONDITION_LABELS[name]}\nminus original stabilized", fontsize=7.5)
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
        axes[row, 0].set_ylabel(
            f"RR100 {unit}\n{units.iloc[row].selection_role}\n"
            f"preferred SF={units.iloc[row].preferred_sf_cpd:.2f} cycles/degree",
            fontsize=8,
        )
        fig.colorbar(rate_image, ax=axes[row, :5], shrink=0.55, pad=0.004, label="activation rate (Hz)")
        fig.colorbar(diff_image, ax=axes[row, 5:], shrink=0.55, pad=0.004, label="rate difference (Hz)")
    scale_label = "one scale shared across all selected units" if global_scale else "separate scale per unit, shared across conditions"
    fig.suptitle(
        f"Stage 2A raw activation maps: amplitude-by-phase factorial, seed {seed}\n{scale_label}; maps are post-activation firing rates",
        fontsize=15, weight="bold",
    )
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = choose_development_condition(args.stage2_dir)
    split = pd.read_csv(args.stage2_dir / "frozen_image_and_trace_identity_split.csv")
    for kind, value in (("image", int(selection.image_index)), ("trace", int(selection.trace_index))):
        record = split.loc[split.identity_type.eq(kind) & split.identity.eq(value)]
        if len(record) != 1 or record.iloc[0].split != "development":
            raise ValueError(f"Selected {kind} {value} is not a declared development identity")
    pd.DataFrame([selection]).to_csv(args.out_dir / "selected_development_condition.csv", index=False)
    units = select_units(args.assignments, args.tuning_dir)
    units.to_csv(args.out_dir / "selected_unit_roles_pre_response.csv", index=False)

    patch, ppd, trace72 = load_development_input(selection, args.response_cache_dir)
    common = _load_twin_common()
    fem_stim = render_scored_embedding(common, torch, patch, trace72, ppd)
    stabilized_stim = render_scored_embedding(common, torch, patch, np.zeros_like(trace72), ppd)
    frame = int(args.scored_frame)
    if not 0 <= frame < fem_stim.shape[0]:
        raise ValueError(f"scored frame {frame} is outside the 40-frame response window")
    fem_cube = fem_stim[frame, 0].detach().cpu().numpy().astype(np.float32)
    stabilized_cube = stabilized_stim[frame, 0].detach().cpu().numpy().astype(np.float32)
    if fem_cube.shape != (N_HISTORY, 151, 151):
        raise ValueError(f"Expected a 32x151x151 model input cube, got {fem_cube.shape}")

    conditions, input_audit, tile_audit = construct_factorial(fem_cube, stabilized_cube)
    input_audit.to_csv(args.out_dir / "factorial_input_audit.csv", index=False)
    tile_audit.to_csv(args.out_dir / "factorial_tiled_energy_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "factorial_input_cubes.npz",
        seeds=np.asarray(SEEDS, dtype=np.int64), scored_frame=np.asarray(frame),
        image_index=np.asarray(int(selection.image_index)), trace_index=np.asarray(int(selection.trace_index)),
        fem_original=fem_cube, stabilized_original=stabilized_cube,
        fem_power_random_phase=np.stack([conditions[(seed, "fem_power_random_phase")] for seed in SEEDS]),
        stabilized_power_fem_phase=conditions[(SEEDS[0], "stabilized_power_fem_phase")],
        stabilized_power_random_phase=np.stack([conditions[(seed, "stabilized_power_random_phase")] for seed in SEEDS]),
    )
    plot_inputs(
        conditions, input_audit, selection, trace72,
        args.out_dir / "01_factorial_input_cubes", int(args.dpi),
    )

    keys, cube_stack = unique_condition_stack(conditions)
    unique_maps, model_metadata = score_maps(cube_stack, args.device, int(args.batch_size))
    maps = expand_scored_maps(keys, unique_maps, conditions)
    response_metrics, factorial_effects = response_tables(maps, units)
    response_metrics.to_csv(args.out_dir / "selected_unit_condition_map_metrics.csv", index=False)
    factorial_effects.to_csv(args.out_dir / "selected_unit_factorial_effects.csv", index=False)
    selected = units.rr100_index.to_numpy(int)
    np.savez_compressed(
        args.out_dir / "selected_unit_activation_maps.npz",
        rr100_index=selected, seeds=np.asarray(SEEDS),
        condition_names=np.asarray(list(CONDITION_LABELS), dtype="U64"),
        activation_maps=np.stack(
            [[maps[(seed, name)][selected] for name in CONDITION_LABELS] for seed in SEEDS]
        ).astype(np.float32),
    )
    plot_maps(
        maps, units, response_metrics, args.out_dir / "02_activation_maps_global_shared_scale",
        int(args.dpi), global_scale=True,
    )
    plot_maps(
        maps, units, response_metrics, args.out_dir / "03_activation_maps_per_unit_shared_condition_scale",
        int(args.dpi), global_scale=False,
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_map_support_amplitude_by_phase_factorial_stage2a",
        "status": "targeted_development_input_and_raw_map_checkpoint_complete",
        "scope": {
            "image_index": int(selection.image_index), "trace_index": int(selection.trace_index),
            "scored_frame": frame, "phase_seeds": list(SEEDS), "selected_units": selected.tolist(),
        },
        "contracts": {
            "identity_tier": "development identities only; frozen final-test identities unopened",
            "input_tensor": "one exact normalized 32-lag by 151 by 151 model history cube for one scored map",
            "power_equality": "complete unwindowed three-dimensional Fourier amplitude of each raw history cube",
            "factorial": "FEM/stabilized amplitude crossed with FEM/shared-random phase; same random phase within seed",
            "raw_control": "no clipping, rank matching, histogram matching, or input-range projection",
            "target_map": "post-output-activation nonnegative RR100 firing-rate map",
            "ssi": "instantaneous spatial SSI from one raw rate map; not trajectory-averaged movie SSI",
            "selection": "condition selected by median input power; units selected from fixed-retina grating properties before responses",
        },
        "input_validation": {
            "maximum_fourier_amplitude_relative_error": float(input_audit.fourier_amplitude_relative_error.max()),
            "maximum_imaginary_reconstruction_residual": float(input_audit.maximum_imaginary_reconstruction_residual.max()),
            "maximum_hermitian_relative_error": float(input_audit.reconstructed_spectrum_hermitian_relative_error.max()),
            "maximum_outside_canonical_input_fraction": float(input_audit.fraction_outside_canonical_input_range.max()),
            "maximum_histogram_wasserstein_distance": float(input_audit.histogram_wasserstein_distance_from_same_power_original.max()),
        },
        "model": model_metadata,
        "sources": {
            "stage2_manifest": file_identity(args.stage2_dir / "manifest.json"),
            "identity_split": file_identity(args.stage2_dir / "frozen_image_and_trace_identity_split.csv"),
            "trace_cache": file_identity(args.response_cache_dir / "input_cache/corrected_trace_segments.npz"),
            "grating_tuning": file_identity(args.tuning_dir / "grating_only_orientation_tuning.npz"),
            "runner": file_identity(Path(__file__)),
        },
        "artifacts": {
            "input_figure": "01_factorial_input_cubes.pdf",
            "global_scale_map_figure": "02_activation_maps_global_shared_scale.pdf",
            "per_unit_scale_map_figure": "03_activation_maps_per_unit_shared_condition_scale.pdf",
            "input_audit": "factorial_input_audit.csv", "factorial_effects": "selected_unit_factorial_effects.csv",
            "unit_selection": "selected_unit_roles_pre_response.csv",
        },
        "decision_gate": (
            "stop for human inspection; do not expand frames, units, or population summaries until raw input-range, "
            "histogram, activation-map, and SSI effects are judged interpretable"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# Stage 2A map-support amplitude-by-phase factorial\n\n"
        "This targeted development checkpoint crosses FEM versus stabilized raw 3-D Fourier amplitude with "
        "FEM versus shared random phase for one canonical 32-lag model input. It stops at input audits and "
        "raw activation maps for six pre-response-selected units. See `manifest.json` for the exact contract.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

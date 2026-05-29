#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import run_model
from eval.fixrsvp import get_fixrsvp_data


ROOT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")
DEFAULT_OUTPUT_DIR = ROOT_OUTPUT_DIR / "default_run"
RUN_INDEX_FILENAME = "run_index.md"
RUN_OVERVIEW_FILENAME = "step01_run_overview.md"
RUN_INTERPRETATION_FILENAME = "step01_results_interpretation.md"
LEGACY_RUN_OVERVIEW_FILENAME = "README.md"
LEGACY_RUN_INTERPRETATION_FILENAME = "results_so_far.md"
LEGACY_ROOT_INDEX_FILENAMES = (
    LEGACY_RUN_OVERVIEW_FILENAME,
    LEGACY_RUN_INTERPRETATION_FILENAME,
)


@dataclass(frozen=True)
class AnalysisUnit:
    unit_id: str
    image_id: int
    unit_mode: str
    phase_bin: int | None
    radius_bin: int | None
    trial_indices: tuple[int, ...]
    time_indices: tuple[int, ...]
    n_samples: int
    median_radius_deg: float
    p75_radius_deg: float
    p90_radius_deg: float


@dataclass(frozen=True)
class ModelBackendConfig:
    dataset_idx: int
    model_device: str
    unit_mode: str
    n_fixation_phase_bins: int
    n_radius_bins: int
    max_units: int
    max_samples_per_unit: int
    min_backend_samples: int
    n_image_shuffle_matches: int
    jacobian_step_px: float
    jacobian_step_sizes_px: tuple[float, ...]
    displacement_magnitudes_px: tuple[float, ...]
    empirical_displacement_percentiles: tuple[float, ...]
    local_state_keep_fraction: float
    pixels_per_degree: float | None = None
    n_random_null_reps: int = 256
    max_baseline_relative_radius_px: float | None = None
    pairwise_bin_edges_px: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
    model_type: str | None = None
    model_index: int | None = None
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None


class Step01Backend:
    """
    Backend seam for model-response and Jacobian evaluation.

    The current scaffold deliberately stops after data collation, analysis-unit
    definition, and empirical displacement gating. The next edit should attach a
    backend here that takes an AnalysisUnit and returns model responses under
    shifted inputs plus a local Jacobian estimate on the same unit definition.
    """

    def __init__(self, config: ModelBackendConfig | None = None) -> None:
        self.config = config

    def enabled(self) -> bool:
        return self.config is not None

    def describe(self) -> str:
        if not self.enabled():
            return "manifest-only"
        return "lagged-stimulus-finite-difference"

    def run(
        self,
        units: list[AnalysisUnit],
        data: dict,
        dataset_configs_path: str,
        output_dir: Path,
    ) -> dict:
        if not self.enabled():
            return {
                "status": "manifest_only",
                "backend": self.describe(),
                "n_units": len(units),
            }

        config = self.config
        assert config is not None

        load_model_kwargs = {
            "model_type": config.model_type,
            "model_index": config.model_index,
            "cfg_dir_override": dataset_configs_path,
            "device": config.model_device,
        }
        if config.checkpoint_path is not None:
            load_model_kwargs["checkpoint_path"] = config.checkpoint_path
        if config.checkpoint_dir is not None:
            load_model_kwargs["checkpoint_dir"] = config.checkpoint_dir
        model, _model_info = load_model(**load_model_kwargs)
        model.model.eval()

        pixels_per_degree = _resolve_pixels_per_degree(data, config)
        step_displacement_deg = compute_step_displacement_deg(data["eyepos"])

        if not units:
            return {
                "status": "no_units",
                "backend": self.describe(),
                "n_units": 0,
            }

        rng = np.random.default_rng(0)
        candidate_units = list(units)
        if config.max_units < len(candidate_units):
            candidate_units = [candidate_units[idx] for idx in rng.permutation(len(candidate_units))]
        unit_results = []
        jacobians = {}
        fem_covariances = {}
        shifted_resps: dict[str, np.ndarray] = {}
        centered_eye_pxs: dict[str, np.ndarray] = {}
        per_sample_jacobians_dict: dict[str, np.ndarray] = {}  # (N, n_neurons, 2) per unit
        skipped_low_samples = 0
        skipped_radius_filter = 0
        attempted_units = 0

        for unit in candidate_units:
            if len(unit_results) >= config.max_units:
                break
            attempted_units += 1
            sampled = _sample_unit_payload(
                unit=unit,
                data=data,
                max_samples=config.max_samples_per_unit,
                rng=rng,
            )
            if sampled is None:
                continue

            stim, eyepos, sampled_trial_idx, sampled_time_idx = sampled
            if stim.shape[0] < config.min_backend_samples:
                skipped_low_samples += 1
                continue
            baseline_stim, baseline_eye_deg, baseline_idx = _choose_representative_baseline(stim, eyepos)
            pre_filter_n_samples = int(stim.shape[0])
            pre_filter_centered_eye_px = (eyepos - baseline_eye_deg[None, :]) * pixels_per_degree
            stim, eyepos, sampled_trial_idx, sampled_time_idx, baseline_idx = _filter_local_state_neighborhood(
                stim=stim,
                eyepos_deg=eyepos,
                trial_idx=sampled_trial_idx,
                time_idx=sampled_time_idx,
                baseline_stim=baseline_stim,
                baseline_idx=baseline_idx,
                keep_fraction=config.local_state_keep_fraction,
                min_keep=config.min_backend_samples,
            )
            centered_eye_deg = eyepos - baseline_eye_deg[None, :]
            centered_eye_px = centered_eye_deg * pixels_per_degree
            n_samples_post_local_filter = int(stim.shape[0])
            if config.max_baseline_relative_radius_px is not None:
                stim, eyepos, sampled_trial_idx, sampled_time_idx, baseline_idx, centered_eye_px = (
                    _filter_absolute_radius_neighborhood(
                        stim=stim,
                        eyepos_deg=eyepos,
                        trial_idx=sampled_trial_idx,
                        time_idx=sampled_time_idx,
                        baseline_idx=baseline_idx,
                        centered_eye_px=centered_eye_px,
                        max_radius_px=config.max_baseline_relative_radius_px,
                        min_keep=config.min_backend_samples,
                    )
                )
                if stim is None:
                    skipped_radius_filter += 1
                    continue
            # Baseline-relative offsets are what drive Step 1 covariance; use these
            # for the central-mass Step 0 gate so the two analyses are aligned.
            baseline_relative_magnitudes_px = _baseline_relative_displacement_magnitudes_px(
                centered_eye_px=centered_eye_px,
                percentiles=config.empirical_displacement_percentiles,
            )
            # Frame-to-frame step magnitudes kept as a separate diagnostic field.
            step_displacement_magnitudes_px = _empirical_displacement_magnitudes_px_from_values(
                step_displacement_deg[sampled_trial_idx, sampled_time_idx],
                pixels_per_degree=pixels_per_degree,
                percentiles=config.empirical_displacement_percentiles,
            )
            effective_displacement_magnitudes_px = _merge_displacement_magnitudes(
                config.displacement_magnitudes_px,
                baseline_relative_magnitudes_px,
            )

            jacobian_by_step = {
                step_px: _compute_local_jacobian(
                    model=model,
                    stim=baseline_stim,
                    dataset_idx=config.dataset_idx,
                    step_px=step_px,
                )
                for step_px in config.jacobian_step_sizes_px
            }
            jacobian = jacobian_by_step[config.jacobian_step_px]

            baseline_resp = _predict_responses(model, baseline_stim, config.dataset_idx)[0]
            extra_directions: np.ndarray | None = None
            if centered_eye_px.shape[0] >= 3:
                try:
                    cov2 = np.cov(centered_eye_px.T)
                    if np.ndim(cov2) == 2 and np.isfinite(cov2).all():
                        _, evecs = np.linalg.eigh(cov2)
                        extra_directions = evecs.T
                except np.linalg.LinAlgError:
                    pass
            lin_metrics = _evaluate_linearization(
                model=model,
                stim=baseline_stim,
                base_resp=baseline_resp,
                dataset_idx=config.dataset_idx,
                jacobian=jacobian,
                displacement_magnitudes_px=effective_displacement_magnitudes_px,
                empirical_displacement_magnitudes_px=baseline_relative_magnitudes_px,
                extra_directions=extra_directions,
            )

            shifted_resp = _predict_shifted_responses(
                model=model,
                dataset_idx=config.dataset_idx,
                baseline_stim=baseline_stim,
                displacements_px=centered_eye_px,
            )

            cov_result = _compute_covariance_geometry(
                shifted_resp=shifted_resp,
                jacobian=jacobian,
                eye_displacements_px=centered_eye_px,
                n_random_null_reps=config.n_random_null_reps,
            )
            step_stability = _summarize_jacobian_step_stability(jacobian_by_step)
            jacobian_rank = _summarize_jacobian_rank(jacobian)
            baseline_features = _summarize_baseline_features(
                baseline_stim=baseline_stim,
                baseline_resp=baseline_resp,
                centered_eye_px=centered_eye_px,
                pre_filter_centered_eye_px=pre_filter_centered_eye_px,
            )
            replay_metrics = _replay_crosscheck(
                data=data,
                processed_trial_idx=int(sampled_trial_idx[baseline_idx]),
                processed_time_idx=int(sampled_time_idx[baseline_idx]),
                collated_baseline_stim=baseline_stim,
                baseline_resp=baseline_resp,
                model=model,
                dataset_idx=config.dataset_idx,
            )
            unit_record = {
                "unit_id": unit.unit_id,
                "image_id": unit.image_id,
                "phase_bin": -1 if unit.phase_bin is None else unit.phase_bin,
                "radius_bin": -1 if unit.radius_bin is None else unit.radius_bin,
                "n_samples": int(shifted_resp.shape[0]),
                "n_samples_pre_local_filter": pre_filter_n_samples,
                "n_samples_post_local_filter": n_samples_post_local_filter,
                "local_state_retained_fraction": float(shifted_resp.shape[0] / max(pre_filter_n_samples, 1)),
                "pixels_per_degree": float(pixels_per_degree),
                **lin_metrics,
                **step_stability,
                **jacobian_rank,
                **baseline_features,
                **replay_metrics,
                **cov_result,
            }
            for percentile, magnitude in zip(config.empirical_displacement_percentiles, baseline_relative_magnitudes_px):
                unit_record[_empirical_displacement_field_name(percentile)] = float(magnitude)
            for percentile, magnitude in zip(config.empirical_displacement_percentiles, step_displacement_magnitudes_px):
                unit_record[_step_displacement_field_name(percentile)] = float(magnitude)
            unit_results.append(unit_record)
            jacobians[unit.unit_id] = cov_result["J_baseline"]
            fem_covariances[unit.unit_id] = cov_result["cov_model_fem"]
            shifted_resps[unit.unit_id] = shifted_resp
            centered_eye_pxs[unit.unit_id] = centered_eye_px
            per_sample_jacobians_dict[unit.unit_id] = _compute_per_sample_jacobians_batch(
                model=model,
                baseline_stim=baseline_stim,
                centered_eye_px=centered_eye_px,
                dataset_idx=config.dataset_idx,
                step_px=config.jacobian_step_px,
            )

        _attach_image_shuffled_nulls(
            unit_results,
            jacobians,
            fem_covariances,
            n_matches=config.n_image_shuffle_matches,
        )
        pairwise_rows = _compute_pairwise_bin_analysis(
            unit_results=unit_results,
            jacobians=jacobians,
            shifted_resps=shifted_resps,
            centered_eye_pxs=centered_eye_pxs,
            bin_edges_px=config.pairwise_bin_edges_px,
            n_shuffle_matches=config.n_image_shuffle_matches,
            per_sample_jacobians_dict=per_sample_jacobians_dict,
        )
        _write_backend_outputs(output_dir, unit_results)
        _write_pairwise_bin_outputs(output_dir, pairwise_rows)

        valid_align = [row["alignment_A_J"] for row in unit_results if np.isfinite(row["alignment_A_J"])]
        valid_capture = [row["capture_V_J"] for row in unit_results if np.isfinite(row["capture_V_J"])]
        median_empirical_displacement_magnitudes_px = _median_empirical_displacement_magnitudes_px(
            unit_results,
            config.empirical_displacement_percentiles,
        )
        median_step_displacement_magnitudes_px = _median_step_displacement_magnitudes_px(
            unit_results,
            config.empirical_displacement_percentiles,
        )
        return {
            "status": "completed",
            "backend": self.describe(),
            "n_units": len(unit_results),
            "attempted_units": attempted_units,
            "skipped_low_samples": skipped_low_samples,
            "skipped_radius_filter": skipped_radius_filter,
            "max_baseline_relative_radius_px": config.max_baseline_relative_radius_px,
            "min_backend_samples": config.min_backend_samples,
            "pixels_per_degree": float(pixels_per_degree),
            "effective_displacement_magnitudes_px": [
                float(x)
                for x in _merge_displacement_magnitudes(
                    config.displacement_magnitudes_px,
                    tuple(median_empirical_displacement_magnitudes_px),
                )
            ],
            "explicit_displacement_magnitudes_px": [float(x) for x in config.displacement_magnitudes_px],
            "empirical_displacement_percentiles": [float(x) for x in config.empirical_displacement_percentiles],
            "median_empirical_displacement_magnitudes_px": [float(x) for x in median_empirical_displacement_magnitudes_px],
            "median_step_displacement_magnitudes_px": [float(x) for x in median_step_displacement_magnitudes_px],
            "unit_mode": config.unit_mode,
            "n_fixation_phase_bins": config.n_fixation_phase_bins,
            "n_radius_bins": config.n_radius_bins,
            "median_alignment_A_J": float(np.nanmedian(valid_align)) if valid_align else float("nan"),
            "median_capture_V_J": float(np.nanmedian(valid_capture)) if valid_capture else float("nan"),
        }


def _normalize_stim_dims(stim: np.ndarray) -> torch.Tensor:
    stim_tensor = torch.as_tensor(stim, dtype=torch.float32)
    if stim_tensor.dim() == 4:
        stim_tensor = stim_tensor.unsqueeze(1)
    if stim_tensor.dim() != 5:
        raise ValueError(f"Expected stim to be 4D or 5D, got shape {tuple(stim_tensor.shape)}")
    return stim_tensor


def _sample_unit_payload(
    unit: AnalysisUnit,
    data: dict,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray] | None:
    trial_idx = np.asarray(unit.trial_indices, dtype=np.int64)
    time_idx = np.asarray(unit.time_indices, dtype=np.int64)
    stim = data["stim"][trial_idx, time_idx]
    eyepos = data["eyepos"][trial_idx, time_idx]

    finite_mask = np.isfinite(stim).all(axis=tuple(range(1, stim.ndim))) & np.isfinite(eyepos).all(axis=1)
    if not np.any(finite_mask):
        return None

    stim = stim[finite_mask]
    eyepos = eyepos[finite_mask]
    trial_idx = trial_idx[finite_mask]
    time_idx = time_idx[finite_mask]
    if stim.shape[0] > max_samples:
        keep = np.sort(rng.choice(stim.shape[0], size=max_samples, replace=False))
        stim = stim[keep]
        eyepos = eyepos[keep]
        trial_idx = trial_idx[keep]
        time_idx = time_idx[keep]

    return _normalize_stim_dims(stim), eyepos.astype(np.float64), trial_idx, time_idx


def _choose_representative_baseline(stim: torch.Tensor, eyepos_deg: np.ndarray) -> tuple[torch.Tensor, np.ndarray, int]:
    eye_center = np.nanmedian(eyepos_deg, axis=0)
    distances = np.linalg.norm(eyepos_deg - eye_center[None, :], axis=1)
    baseline_idx = int(np.nanargmin(distances))
    return stim[baseline_idx : baseline_idx + 1].clone(), eyepos_deg[baseline_idx].astype(np.float64), baseline_idx


def _filter_local_state_neighborhood(
    stim: torch.Tensor,
    eyepos_deg: np.ndarray,
    trial_idx: np.ndarray,
    time_idx: np.ndarray,
    baseline_stim: torch.Tensor,
    baseline_idx: int,
    keep_fraction: float,
    min_keep: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, int]:
    if keep_fraction >= 1.0 or stim.shape[0] <= min_keep:
        return stim, eyepos_deg, trial_idx, time_idx, baseline_idx

    keep_count = max(min_keep, int(np.ceil(float(keep_fraction) * stim.shape[0])))
    if keep_count >= stim.shape[0]:
        return stim, eyepos_deg, trial_idx, time_idx, baseline_idx

    flat_stim = stim.reshape(stim.shape[0], -1)
    flat_baseline = baseline_stim.reshape(1, -1)
    stim_rms_delta = torch.sqrt(torch.mean((flat_stim - flat_baseline).square(), dim=1))
    keep = np.argsort(stim_rms_delta.detach().cpu().numpy())[:keep_count]
    new_baseline_idx = int(np.where(keep == baseline_idx)[0][0])
    return stim[keep], eyepos_deg[keep], trial_idx[keep], time_idx[keep], new_baseline_idx


def _filter_absolute_radius_neighborhood(
    stim: torch.Tensor,
    eyepos_deg: np.ndarray,
    trial_idx: np.ndarray,
    time_idx: np.ndarray,
    baseline_idx: int,
    centered_eye_px: np.ndarray,
    max_radius_px: float,
    min_keep: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray] | tuple[None, ...]:
    """Keep only samples within max_radius_px of the baseline in absolute eye-position space.

    The baseline sample always passes (its centered_eye_px radius is zero).
    Returns a 6-tuple of None values if fewer than min_keep samples pass.
    """
    radii = np.linalg.norm(centered_eye_px, axis=1)
    mask = radii <= max_radius_px
    mask[baseline_idx] = True  # baseline always included (radius == 0)
    keep_indices = np.flatnonzero(mask)
    if len(keep_indices) < min_keep:
        return None, None, None, None, None, None
    new_baseline_idx = int(np.searchsorted(keep_indices, baseline_idx))
    return (
        stim[keep_indices],
        eyepos_deg[keep_indices],
        trial_idx[keep_indices],
        time_idx[keep_indices],
        new_baseline_idx,
        centered_eye_px[keep_indices],
    )


def _predict_responses(model, stim: torch.Tensor, dataset_idx: int, batch_size: int = 64) -> np.ndarray:
    outputs = []
    for start in range(0, stim.shape[0], batch_size):
        batch = run_model(model, {"stim": stim[start : start + batch_size]}, dataset_idx=dataset_idx)
        outputs.append(batch["rhat"].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _predict_shifted_responses(
    model,
    dataset_idx: int,
    baseline_stim: torch.Tensor,
    displacements_px: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    responses = []
    for start in range(0, displacements_px.shape[0], batch_size):
        chunk = displacements_px[start : start + batch_size]
        tiled_stim = baseline_stim.repeat(chunk.shape[0], 1, 1, 1, 1)
        shifted_stim = _shift_stimulus_batch(tiled_stim, chunk)
        responses.append(_predict_responses(model, shifted_stim, dataset_idx))
    return np.concatenate(responses, axis=0).astype(np.float64)


def _shift_stimulus(stim: torch.Tensor, dx_px: float, dy_px: float) -> torch.Tensor:
    batch, channels, lags, height, width = stim.shape
    merged = stim.reshape(batch, channels * lags, height, width)
    ys = torch.linspace(-1.0, 1.0, height, device=stim.device, dtype=stim.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=stim.device, dtype=stim.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    shift_x = 2.0 * dx_px / max(width - 1, 1)
    shift_y = 2.0 * dy_px / max(height - 1, 1)
    base_grid[..., 0] -= shift_x
    base_grid[..., 1] -= shift_y
    shifted = F.grid_sample(
        merged,
        base_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return shifted.reshape(batch, channels, lags, height, width)


def _shift_stimulus_batch(stim: torch.Tensor, displacements_px: np.ndarray) -> torch.Tensor:
    batch, channels, lags, height, width = stim.shape
    merged = stim.reshape(batch, channels * lags, height, width)
    ys = torch.linspace(-1.0, 1.0, height, device=stim.device, dtype=stim.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=stim.device, dtype=stim.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    displacement_tensor = torch.as_tensor(displacements_px, device=stim.device, dtype=stim.dtype)
    shift_x = 2.0 * displacement_tensor[:, 0] / max(width - 1, 1)
    shift_y = 2.0 * displacement_tensor[:, 1] / max(height - 1, 1)
    base_grid[..., 0] -= shift_x[:, None, None]
    base_grid[..., 1] -= shift_y[:, None, None]
    shifted = F.grid_sample(
        merged,
        base_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return shifted.reshape(batch, channels, lags, height, width)


def _compute_local_jacobian(model, stim: torch.Tensor, dataset_idx: int, step_px: float) -> np.ndarray:
    resp_x_plus = _predict_responses(model, _shift_stimulus(stim, step_px, 0.0), dataset_idx)[0]
    resp_x_minus = _predict_responses(model, _shift_stimulus(stim, -step_px, 0.0), dataset_idx)[0]
    resp_y_plus = _predict_responses(model, _shift_stimulus(stim, 0.0, step_px), dataset_idx)[0]
    resp_y_minus = _predict_responses(model, _shift_stimulus(stim, 0.0, -step_px), dataset_idx)[0]
    jac_x = (resp_x_plus - resp_x_minus) / (2.0 * step_px)
    jac_y = (resp_y_plus - resp_y_minus) / (2.0 * step_px)
    return np.stack((jac_x, jac_y), axis=-1)


def _compute_per_sample_jacobians_batch(
    model,
    baseline_stim: torch.Tensor,
    centered_eye_px: np.ndarray,
    dataset_idx: int,
    step_px: float,
) -> np.ndarray:
    """Compute the image-translation Jacobian at each sample's absolute eye position.

    For sample i whose centered eye position is p_i (px, relative to baseline),
    the stimulus seen by the retina is baseline_stim shifted by p_i.  The local
    Jacobian at that position is estimated by finite differences around p_i:

        J_i = [dr/dx, dr/dy] evaluated at the stimulus shifted to p_i.

    Uses 4 batched calls to _predict_shifted_responses (p_i ± step in x or y),
    keeping cost to 4 × N model forward passes.

    Returns shape (N, n_neurons, 2) — one (n_neurons, 2) Jacobian per sample.
    """
    displacements_xp = centered_eye_px.copy()
    displacements_xp[:, 0] += step_px
    displacements_xm = centered_eye_px.copy()
    displacements_xm[:, 0] -= step_px
    displacements_yp = centered_eye_px.copy()
    displacements_yp[:, 1] += step_px
    displacements_ym = centered_eye_px.copy()
    displacements_ym[:, 1] -= step_px

    resp_xp = _predict_shifted_responses(model, dataset_idx, baseline_stim, displacements_xp)
    resp_xm = _predict_shifted_responses(model, dataset_idx, baseline_stim, displacements_xm)
    resp_yp = _predict_shifted_responses(model, dataset_idx, baseline_stim, displacements_yp)
    resp_ym = _predict_shifted_responses(model, dataset_idx, baseline_stim, displacements_ym)

    jac_x = (resp_xp - resp_xm) / (2.0 * step_px)   # (N, n_neurons)
    jac_y = (resp_yp - resp_ym) / (2.0 * step_px)   # (N, n_neurons)
    return np.stack([jac_x, jac_y], axis=-1)          # (N, n_neurons, 2)


def _displacement_directions() -> tuple[np.ndarray, ...]:
    diag = 1.0 / np.sqrt(2.0)
    return (
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([diag, diag], dtype=np.float64),
        np.array([diag, -diag], dtype=np.float64),
    )


def _safe_r2(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = np.sum((actual - predicted) ** 2, axis=1)
    denom = np.sum(actual ** 2, axis=1)
    out = np.full(actual.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = 1.0 - residual[valid] / denom[valid]
    return out


def _safe_relative_residual(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = np.linalg.norm(actual - predicted, axis=1)
    denom = np.linalg.norm(actual, axis=1)
    out = np.full(actual.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = residual[valid] / denom[valid]
    return out


def _safe_cosine(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    num = np.sum(actual * predicted, axis=1)
    denom = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
    out = np.full(actual.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = num[valid] / denom[valid]
    return out


def _evaluate_linearization(
    model,
    stim: torch.Tensor,
    base_resp: np.ndarray,
    jacobian: np.ndarray,
    dataset_idx: int,
    displacement_magnitudes_px: tuple[float, ...],
    empirical_displacement_magnitudes_px: tuple[float, ...],
    extra_directions: np.ndarray | None = None,
) -> dict:
    all_directions = list(_displacement_directions())
    if extra_directions is not None and extra_directions.shape[0] > 0:
        for i in range(extra_directions.shape[0]):
            d = np.asarray(extra_directions[i], dtype=np.float64)
            norm = float(np.linalg.norm(d))
            if norm > 1e-12:
                all_directions.append(d / norm)

    all_r2 = []
    all_resid = []
    all_cos = []
    per_mag_rows = []

    for magnitude in displacement_magnitudes_px:
        mag_r2 = []
        mag_resid = []
        mag_cos = []
        mag_actual_norm = []
        mag_pred_norm = []
        for direction in all_directions:
            dx, dy = (magnitude * direction).tolist()
            shifted_resp = _predict_responses(model, _shift_stimulus(stim, dx, dy), dataset_idx)[0]
            delta_actual = (shifted_resp - base_resp)[None, :]
            delta_lin = (jacobian @ np.array([dx, dy], dtype=np.float64))[None, :]
            mag_r2.append(_safe_r2(delta_actual, delta_lin))
            mag_resid.append(_safe_relative_residual(delta_actual, delta_lin))
            mag_cos.append(_safe_cosine(delta_actual, delta_lin))
            mag_actual_norm.append(np.linalg.norm(delta_actual, axis=1))
            mag_pred_norm.append(np.linalg.norm(delta_lin, axis=1))

        mag_r2_arr = np.concatenate(mag_r2)
        mag_resid_arr = np.concatenate(mag_resid)
        mag_cos_arr = np.concatenate(mag_cos)
        mag_actual_norm_arr = np.concatenate(mag_actual_norm)
        mag_pred_norm_arr = np.concatenate(mag_pred_norm)
        per_mag_rows.append(
            {
                f"r2_median_px_{magnitude:g}": float(np.nanmedian(mag_r2_arr)),
                f"resid_median_px_{magnitude:g}": float(np.nanmedian(mag_resid_arr)),
                f"cosine_median_px_{magnitude:g}": float(np.nanmedian(mag_cos_arr)),
                f"actual_norm_median_px_{magnitude:g}": float(np.nanmedian(mag_actual_norm_arr)),
                f"pred_norm_median_px_{magnitude:g}": float(np.nanmedian(mag_pred_norm_arr)),
            }
        )
        all_r2.append(mag_r2_arr)
        all_resid.append(mag_resid_arr)
        all_cos.append(mag_cos_arr)

    result = {
        "step0_r2_median": float(np.nanmedian(np.concatenate(all_r2))),
        "step0_resid_median": float(np.nanmedian(np.concatenate(all_resid))),
        "step0_cosine_median": float(np.nanmedian(np.concatenate(all_cos))),
    }
    for row in per_mag_rows:
        result.update(row)
    central_r2 = [
        result[f"r2_median_px_{float(magnitude):g}"]
        for magnitude in empirical_displacement_magnitudes_px
        if f"r2_median_px_{float(magnitude):g}" in result
    ]
    central_resid = [
        result[f"resid_median_px_{float(magnitude):g}"]
        for magnitude in empirical_displacement_magnitudes_px
        if f"resid_median_px_{float(magnitude):g}" in result
    ]
    central_cosine = [
        result[f"cosine_median_px_{float(magnitude):g}"]
        for magnitude in empirical_displacement_magnitudes_px
        if f"cosine_median_px_{float(magnitude):g}" in result
    ]
    result["central_mass_r2_median"] = float(np.nanmedian(central_r2)) if central_r2 else float("nan")
    result["central_mass_resid_median"] = float(np.nanmedian(central_resid)) if central_resid else float("nan")
    result["central_mass_cosine_median"] = float(np.nanmedian(central_cosine)) if central_cosine else float("nan")
    return result


def _alignment_score(U1: np.ndarray, U2: np.ndarray) -> float:
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    singular_values = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))


def _capture_fraction(U: np.ndarray, cov: np.ndarray) -> float:
    Q, _ = np.linalg.qr(U)
    return float(np.trace(Q.T @ cov @ Q) / (np.trace(cov) + 1e-12))


def _top_two_eigenvectors(cov: np.ndarray) -> np.ndarray:
    _, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -2:]


def _compute_covariance_geometry(
    shifted_resp: np.ndarray,
    jacobian: np.ndarray,
    eye_displacements_px: np.ndarray,
    n_random_null_reps: int,
) -> dict:
    cov_model = np.cov(shifted_resp, rowvar=False).astype(np.float64)
    cov_eye = np.cov(eye_displacements_px.T).astype(np.float64)
    U_jac, _ = np.linalg.qr(jacobian)
    U_fem = _top_two_eigenvectors(cov_model)
    pred_cov = jacobian @ cov_eye @ jacobian.T
    random_align, random_capture = _random_subspace_nulls(cov_model, U_fem, n_random_null_reps)
    return {
        "alignment_A_J": _alignment_score(U_jac, U_fem),
        "capture_V_J": _capture_fraction(U_jac, cov_model),
        "trace_cov_model_fem": float(np.trace(cov_model)),
        "jacobian_fro_norm": float(np.linalg.norm(jacobian)),
        "predicted_drive_trace": float(np.trace(pred_cov)),
        "random_subspace_alignment_median": float(np.nanmedian(random_align)),
        "random_subspace_capture_median": float(np.nanmedian(random_capture)),
        "J_baseline": jacobian,
        "U_fem": U_fem,
        "cov_model_fem": cov_model,
    }


def _get_dataset_index(data: dict) -> int:
    dataset = data["dataset"]
    return int(dataset.inds[:, 0].unique().item())


def _get_dataset_stim_lags(data: dict) -> list[int]:
    dataset = data["dataset"]
    stim_lags = dataset.keys_lags.get("stim", 0)
    if isinstance(stim_lags, torch.Tensor):
        return [int(x) for x in stim_lags.detach().cpu().tolist()]
    if isinstance(stim_lags, np.ndarray):
        return [int(x) for x in stim_lags.tolist()]
    if isinstance(stim_lags, (list, tuple)):
        return [int(x) for x in stim_lags]
    return [int(stim_lags)]


def _reconstruct_replay_baseline_stim(
    data: dict,
    processed_trial_idx: int,
    processed_time_idx: int,
) -> torch.Tensor | None:
    trial_ids = data.get("trial_ids")
    if trial_ids is None:
        return None

    dataset = data["dataset"]
    dset_idx = _get_dataset_index(data)
    raw_trial_id = int(trial_ids[processed_trial_idx])
    trial_inds = dataset.dsets[dset_idx].covariates["trial_inds"].numpy()
    psth_inds = dataset.dsets[dset_idx].covariates["psth_inds"].numpy()
    raw_stim = dataset.dsets[dset_idx]["stim"].numpy()

    target_rows = np.flatnonzero((trial_inds == raw_trial_id) & (psth_inds == processed_time_idx))
    if target_rows.size == 0:
        return None
    target_row = int(target_rows[0])

    if raw_stim.ndim >= 5:
        return _normalize_stim_dims(raw_stim[target_row : target_row + 1])

    stim_lags = _get_dataset_stim_lags(data)
    replay = np.full((1, raw_stim.shape[1], len(stim_lags), *raw_stim.shape[2:]), np.nan, dtype=np.float32)
    for lag_idx, lag in enumerate(stim_lags):
        lagged_row = target_row - int(lag)
        if lagged_row < 0:
            continue
        if int(trial_inds[lagged_row]) != raw_trial_id:
            continue
        replay[0, :, lag_idx] = raw_stim[lagged_row]
    if not np.isfinite(replay).all():
        return None
    return torch.as_tensor(replay, dtype=torch.float32)


def _replay_crosscheck(
    data: dict,
    processed_trial_idx: int,
    processed_time_idx: int,
    collated_baseline_stim: torch.Tensor,
    baseline_resp: np.ndarray,
    model,
    dataset_idx: int,
) -> dict:
    replay_stim = _reconstruct_replay_baseline_stim(
        data=data,
        processed_trial_idx=processed_trial_idx,
        processed_time_idx=processed_time_idx,
    )
    if replay_stim is None:
        return {
            "replay_stim_mae": float("nan"),
            "replay_stim_max_abs": float("nan"),
            "replay_resp_l2": float("nan"),
            "replay_resp_corr": float("nan"),
        }

    stim_diff = (replay_stim - collated_baseline_stim).detach().cpu().numpy().astype(np.float64)
    replay_resp = _predict_responses(model, replay_stim, dataset_idx)[0]
    if np.std(replay_resp) < 1e-12 or np.std(baseline_resp) < 1e-12:
        replay_corr = float("nan")
    else:
        replay_corr = float(np.corrcoef(replay_resp, baseline_resp)[0, 1])
    return {
        "replay_stim_mae": float(np.nanmean(np.abs(stim_diff))),
        "replay_stim_max_abs": float(np.nanmax(np.abs(stim_diff))),
        "replay_resp_l2": float(np.linalg.norm(replay_resp - baseline_resp)),
        "replay_resp_corr": replay_corr,
    }


def _random_subspace_nulls(cov_model: np.ndarray, U_fem: np.ndarray, n_reps: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_neurons = cov_model.shape[0]
    alignments = np.empty(n_reps, dtype=np.float64)
    captures = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        U_rand, _ = np.linalg.qr(rng.standard_normal((n_neurons, 2)))
        alignments[rep] = _alignment_score(U_rand, U_fem)
        captures[rep] = _capture_fraction(U_rand, cov_model)
    return alignments, captures


def _summarize_jacobian_step_stability(jacobian_by_step: dict[float, np.ndarray]) -> dict:
    steps = sorted(jacobian_by_step)
    result = {}
    for idx, step in enumerate(steps):
        result[f"jacobian_fro_norm_h_{step:g}"] = float(np.linalg.norm(jacobian_by_step[step]))
        if idx == 0:
            continue
        prev_step = steps[idx - 1]
        U_prev, _ = np.linalg.qr(jacobian_by_step[prev_step])
        U_curr, _ = np.linalg.qr(jacobian_by_step[step])
        result[f"jacobian_step_alignment_{prev_step:g}_to_{step:g}"] = _alignment_score(U_prev, U_curr)
        result[f"jacobian_norm_ratio_{prev_step:g}_to_{step:g}"] = float(
            np.linalg.norm(jacobian_by_step[step]) / (np.linalg.norm(jacobian_by_step[prev_step]) + 1e-12)
        )
    return result


def _summarize_jacobian_rank(jacobian: np.ndarray) -> dict:
    sv = np.linalg.svd(jacobian, compute_uv=False)
    s1 = float(sv[0]) if sv.size >= 1 else 0.0
    s2 = float(sv[1]) if sv.size >= 2 else 0.0
    return {
        "jacobian_singular_1": s1,
        "jacobian_singular_2": s2,
        "jacobian_rank_ratio": float(s2 / (s1 + 1e-12)),
    }


def _baseline_stim_rms(baseline_stim: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(baseline_stim.square())).item())


def _baseline_stim_grad_energy(baseline_stim: torch.Tensor) -> float:
    grad_x = baseline_stim[..., :, 1:] - baseline_stim[..., :, :-1]
    grad_y = baseline_stim[..., 1:, :] - baseline_stim[..., :-1, :]
    mean_grad_x = torch.mean(grad_x.square()) if grad_x.numel() else torch.tensor(0.0, device=baseline_stim.device)
    mean_grad_y = torch.mean(grad_y.square()) if grad_y.numel() else torch.tensor(0.0, device=baseline_stim.device)
    return float((mean_grad_x + mean_grad_y).item())


def _summarize_baseline_features(
    baseline_stim: torch.Tensor,
    baseline_resp: np.ndarray,
    centered_eye_px: np.ndarray,
    pre_filter_centered_eye_px: np.ndarray | None = None,
) -> dict:
    eye_radius_px = np.linalg.norm(centered_eye_px, axis=1)
    result = {
        "baseline_stim_rms": _baseline_stim_rms(baseline_stim),
        "baseline_stim_grad_energy": _baseline_stim_grad_energy(baseline_stim),
        "baseline_resp_mean": float(np.mean(baseline_resp)),
        "baseline_resp_norm": float(np.linalg.norm(baseline_resp)),
        "centered_eye_radius_px_median": float(np.nanmedian(eye_radius_px)),
        "centered_eye_radius_px_p75": float(np.nanpercentile(eye_radius_px, 75)),
    }
    if pre_filter_centered_eye_px is not None:
        pre_radius_px = np.linalg.norm(pre_filter_centered_eye_px, axis=1)
        result["pre_filter_eye_radius_px_median"] = float(np.nanmedian(pre_radius_px))
        result["pre_filter_eye_radius_px_p75"] = float(np.nanpercentile(pre_radius_px, 75))
    return result


def _matched_candidate_ids(unit_results: list[dict], target_row: dict, n_matches: int) -> list[str]:
    feature_names = [
        "jacobian_fro_norm",
        "baseline_stim_rms",
        "baseline_stim_grad_energy",
        "baseline_resp_mean",
        "centered_eye_radius_px_median",
    ]
    feature_matrix = np.array(
        [[row[name] for name in feature_names] for row in unit_results],
        dtype=np.float64,
    )
    center = np.nanmedian(feature_matrix, axis=0)
    scale = np.nanmedian(np.abs(feature_matrix - center[None, :]), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0

    target_features = np.array([target_row[name] for name in feature_names], dtype=np.float64)
    distances = []
    for row in unit_results:
        if row["unit_id"] == target_row["unit_id"]:
            continue
        if row["image_id"] == target_row["image_id"]:
            continue
        row_features = np.array([row[name] for name in feature_names], dtype=np.float64)
        distance = np.linalg.norm((row_features - target_features) / scale)
        distances.append((distance, row["unit_id"]))

    distances.sort(key=lambda item: item[0])
    return [unit_id for _distance, unit_id in distances[:n_matches]]


def _attach_image_shuffled_nulls(
    unit_results: list[dict],
    jacobians: dict[str, np.ndarray],
    fem_covariances: dict[str, np.ndarray],
    n_matches: int,
) -> None:
    for row in unit_results:
        matched_ids = _matched_candidate_ids(unit_results, row, n_matches=n_matches)
        if not matched_ids:
            row["image_shuffle_alignment_median"] = float("nan")
            row["image_shuffle_capture_median"] = float("nan")
            row["image_shuffle_match_count"] = 0
            continue
        shuffled_align = []
        shuffled_capture = []
        cov_model = fem_covariances[row["unit_id"]]
        U_fem = _top_two_eigenvectors(cov_model)
        for other_id in matched_ids:
            J_other = jacobians[other_id]
            U_other, _ = np.linalg.qr(J_other)
            shuffled_align.append(_alignment_score(U_other, U_fem))
            shuffled_capture.append(_capture_fraction(U_other, cov_model))
        row["image_shuffle_alignment_median"] = float(np.nanmedian(shuffled_align))
        row["image_shuffle_capture_median"] = float(np.nanmedian(shuffled_capture))
        row["image_shuffle_match_count"] = int(len(matched_ids))


def _compute_pairwise_bin_metrics(
    shifted_resp: np.ndarray,
    centered_eye_px: np.ndarray,
    jacobian: np.ndarray,
    shuffled_jacobians: list[np.ndarray],
    bin_edges_px: tuple[float, ...],
    per_sample_jacobians: np.ndarray | None = None,
    shuffled_local_data: list[tuple[np.ndarray, np.ndarray]] | None = None,
    min_pairs_linearization: int = 3,
    min_pairs_covariance: int = 8,
) -> list[dict]:
    """Per-distance-bin pairwise linearization and covariance metrics for one unit.

    Computes two sets of metrics in parallel:
      baseline_*  — uses one baseline Jacobian (current behaviour)
      local_*     — uses per-pair midpoint Jacobians averaged from per_sample_jacobians;
                    only populated when per_sample_jacobians is provided

    For V_J local, the shuffled null uses bin-specific mean Jacobians computed
    from each shuffled unit's own (per_sample_jac, eye_px) pair, restricted to
    pairs in the same distance bin. This keeps matched and shuffled subspaces
    symmetric: both are bin-specific averages of local midpoint Jacobians.

    shuffled_local_data: list of (per_sample_jac, eye_px) for each shuffled unit,
        where per_sample_jac is (N_s, n_neurons, 2) and eye_px is (N_s, 2).
    """
    n = shifted_resp.shape[0]
    if n < 2 or len(bin_edges_px) < 2:
        return []

    # Vectorised pair enumeration
    pair_i, pair_j = np.triu_indices(n, k=1)          # (P,), (P,)
    all_dr = (shifted_resp[pair_i] - shifted_resp[pair_j]).astype(np.float64)  # (P, n_neurons)
    all_dp = (centered_eye_px[pair_i] - centered_eye_px[pair_j]).astype(np.float64)  # (P, 2)
    all_dist = np.linalg.norm(all_dp, axis=1)          # (P,)

    # Baseline Jacobian objects
    U_J, _ = np.linalg.qr(jacobian)
    shuffled_Us = [np.linalg.qr(J_s)[0] for J_s in shuffled_jacobians if J_s is not None]

    # Per-sample Jacobian objects (optional)
    has_local = per_sample_jacobians is not None
    has_shuffled_local = shuffled_local_data is not None and len(shuffled_local_data) > 0

    if has_local:
        # Per-pair midpoint Jacobians for the matched unit: (P, n_neurons, 2)
        J_pairs = (
            per_sample_jacobians[pair_i].astype(np.float64)
            + per_sample_jacobians[pair_j].astype(np.float64)
        ) / 2.0

    if has_shuffled_local:
        # Pre-compute per-pair midpoint Jacobians and pairwise distances for each
        # shuffled unit, so we can extract the bin-specific mean J inside the loop.
        # Each entry: (J_pairs_s, dist_s) where J_pairs_s is (P_s, n_neurons, 2).
        shuffled_pair_precomp: list[tuple[np.ndarray, np.ndarray]] = []
        for J_s_ps, eye_s in shuffled_local_data:
            if J_s_ps is None or eye_s is None:
                continue
            n_s = J_s_ps.shape[0]
            si, sj = np.triu_indices(n_s, k=1)
            J_s_pairs = (
                J_s_ps[si].astype(np.float64) + J_s_ps[sj].astype(np.float64)
            ) / 2.0                                      # (P_s, n_neurons, 2)
            dist_s = np.linalg.norm(
                (eye_s[si] - eye_s[sj]).astype(np.float64), axis=1
            )                                            # (P_s,)
            shuffled_pair_precomp.append((J_s_pairs, dist_s))

    local_nan_keys = (
        "r2_lin_local_median", "cosine_local_median", "relative_resid_local_median",
        "capture_V_J_local_matched", "capture_V_J_local_shuffled_median", "capture_V_J_local_delta",
    )

    rows = []
    for k in range(len(bin_edges_px) - 1):
        lo = float(bin_edges_px[k])
        hi = float(bin_edges_px[k + 1])
        mask = (all_dist >= lo) & (all_dist < hi)
        n_pairs = int(mask.sum())

        row: dict = {
            "bin_lo_px": lo,
            "bin_hi_px": hi,
            "bin_mid_px": (lo + hi) * 0.5,
            "n_pairs": n_pairs,
        }

        baseline_nan_keys = (
            "r2_lin_median", "cosine_median", "relative_resid_median",
            "capture_V_J_matched", "capture_V_J_shuffled_median", "capture_V_J_delta",
        )
        if n_pairs < min_pairs_linearization:
            for k_ in baseline_nan_keys:
                row[k_] = float("nan")
            if has_local:
                for k_ in local_nan_keys:
                    row[k_] = float("nan")
            rows.append(row)
            continue

        dr_bin = all_dr[mask]   # (n_pairs_bin, n_neurons)
        dp_bin = all_dp[mask]   # (n_pairs_bin, 2)

        # --- Baseline Jacobian metrics ---
        dr_pred_base = (jacobian @ dp_bin.T).T
        row["r2_lin_median"] = float(np.nanmedian(_safe_r2(dr_bin, dr_pred_base)))
        row["cosine_median"] = float(np.nanmedian(_safe_cosine(dr_bin, dr_pred_base)))
        row["relative_resid_median"] = float(np.nanmedian(_safe_relative_residual(dr_bin, dr_pred_base)))

        cov_dr: np.ndarray | None = None
        if n_pairs >= min_pairs_covariance:
            cov_dr = np.cov(dr_bin, rowvar=False).astype(np.float64)
            row["capture_V_J_matched"] = _capture_fraction(U_J, cov_dr)
            shuf_vals = [_capture_fraction(U_s, cov_dr) for U_s in shuffled_Us]
            row["capture_V_J_shuffled_median"] = float(np.nanmedian(shuf_vals)) if shuf_vals else float("nan")
            row["capture_V_J_delta"] = (
                row["capture_V_J_matched"] - row["capture_V_J_shuffled_median"]
                if np.isfinite(row["capture_V_J_shuffled_median"])
                else float("nan")
            )
        else:
            for k_ in ("capture_V_J_matched", "capture_V_J_shuffled_median", "capture_V_J_delta"):
                row[k_] = float("nan")

        # --- Local (per-pair midpoint) Jacobian metrics ---
        if has_local:
            J_pairs_bin = J_pairs[mask]   # (n_pairs_bin, n_neurons, 2)
            # Per-pair prediction: dr_local[k] = J_pairs_bin[k] @ dp_bin[k]
            dr_pred_local = np.einsum("pnd,pd->pn", J_pairs_bin, dp_bin)
            row["r2_lin_local_median"] = float(np.nanmedian(_safe_r2(dr_bin, dr_pred_local)))
            row["cosine_local_median"] = float(np.nanmedian(_safe_cosine(dr_bin, dr_pred_local)))
            row["relative_resid_local_median"] = float(np.nanmedian(_safe_relative_residual(dr_bin, dr_pred_local)))

            if n_pairs >= min_pairs_covariance:
                if cov_dr is None:
                    cov_dr = np.cov(dr_bin, rowvar=False).astype(np.float64)
                J_bin_avg = np.mean(J_pairs_bin, axis=0)   # (n_neurons, 2): bin-average local J
                U_local, _ = np.linalg.qr(J_bin_avg)
                row["capture_V_J_local_matched"] = _capture_fraction(U_local, cov_dr)

                # Bin-specific shuffled nulls: for each shuffled unit, take mean of
                # its midpoint Jacobians restricted to pairs in this distance bin.
                shuf_local_vals: list[float] = []
                if has_shuffled_local:
                    for J_s_pairs, dist_s in shuffled_pair_precomp:
                        mask_s = (dist_s >= lo) & (dist_s < hi)
                        if mask_s.sum() < min_pairs_covariance:
                            continue
                        J_s_bin_avg = np.mean(J_s_pairs[mask_s], axis=0)  # (n_neurons, 2)
                        U_s_local, _ = np.linalg.qr(J_s_bin_avg)
                        shuf_local_vals.append(_capture_fraction(U_s_local, cov_dr))
                row["capture_V_J_local_shuffled_median"] = (
                    float(np.nanmedian(shuf_local_vals)) if shuf_local_vals else float("nan")
                )
                row["capture_V_J_local_delta"] = (
                    row["capture_V_J_local_matched"] - row["capture_V_J_local_shuffled_median"]
                    if np.isfinite(row["capture_V_J_local_shuffled_median"])
                    else float("nan")
                )
            else:
                for k_ in ("capture_V_J_local_matched", "capture_V_J_local_shuffled_median",
                           "capture_V_J_local_delta"):
                    row[k_] = float("nan")
        else:
            for k_ in local_nan_keys:
                row[k_] = float("nan")

        rows.append(row)

    return rows


def _compute_pairwise_bin_analysis(
    unit_results: list[dict],
    jacobians: dict[str, np.ndarray],
    shifted_resps: dict[str, np.ndarray],
    centered_eye_pxs: dict[str, np.ndarray],
    bin_edges_px: tuple[float, ...],
    n_shuffle_matches: int,
    per_sample_jacobians_dict: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    all_rows: list[dict] = []
    if not unit_results or len(bin_edges_px) < 2:
        return all_rows

    for row in unit_results:
        uid = row["unit_id"]
        if uid not in shifted_resps or uid not in jacobians or uid not in centered_eye_pxs:
            continue
        matched_ids = _matched_candidate_ids(unit_results, row, n_matches=n_shuffle_matches)
        shuffled_J = [jacobians[mid] for mid in matched_ids if mid in jacobians]
        # Local shuffled nulls: (per_sample_jac, eye_px) pairs for each matched unit,
        # so _compute_pairwise_bin_metrics can compute bin-specific mean Jacobians.
        shuffled_local: list[tuple[np.ndarray, np.ndarray]] = []
        if per_sample_jacobians_dict:
            for mid in matched_ids:
                if mid in per_sample_jacobians_dict and mid in centered_eye_pxs:
                    shuffled_local.append(
                        (per_sample_jacobians_dict[mid], centered_eye_pxs[mid])
                    )
        bin_rows = _compute_pairwise_bin_metrics(
            shifted_resp=shifted_resps[uid],
            centered_eye_px=centered_eye_pxs[uid],
            jacobian=jacobians[uid],
            shuffled_jacobians=shuffled_J,
            bin_edges_px=bin_edges_px,
            per_sample_jacobians=per_sample_jacobians_dict.get(uid) if per_sample_jacobians_dict else None,
            shuffled_local_data=shuffled_local if shuffled_local else None,
        )
        for br in bin_rows:
            br["unit_id"] = uid
            br["image_id"] = int(row["image_id"])
        all_rows.extend(bin_rows)
    return all_rows


def _write_pairwise_bin_outputs(output_dir: Path, pairwise_rows: list[dict]) -> None:
    if not pairwise_rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "step01_pairwise_bins.csv"
    scalar_keys = sorted(
        {key for row in pairwise_rows for key, value in row.items() if np.isscalar(value)}
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in pairwise_rows:
            writer.writerow({key: row.get(key, "") for key in scalar_keys})


def _write_backend_outputs(output_dir: Path, unit_results: list[dict]) -> None:
    if not unit_results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "step01_backend_units.csv"
    scalar_keys = sorted(
        {
            key
            for row in unit_results
            for key, value in row.items()
            if np.isscalar(value)
        }
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in unit_results:
            writer.writerow({key: row.get(key, "") for key in scalar_keys})

    np.savez_compressed(
        output_dir / "step01_backend_arrays.npz",
        unit_ids=np.array([row["unit_id"] for row in unit_results], dtype=object),
        image_ids=np.array([row["image_id"] for row in unit_results], dtype=np.int32),
        alignment_A_J=np.array([row["alignment_A_J"] for row in unit_results], dtype=np.float64),
        capture_V_J=np.array([row["capture_V_J"] for row in unit_results], dtype=np.float64),
        image_shuffle_alignment_median=np.array([row["image_shuffle_alignment_median"] for row in unit_results], dtype=np.float64),
        image_shuffle_capture_median=np.array([row["image_shuffle_capture_median"] for row in unit_results], dtype=np.float64),
        random_subspace_alignment_median=np.array([row["random_subspace_alignment_median"] for row in unit_results], dtype=np.float64),
        random_subspace_capture_median=np.array([row["random_subspace_capture_median"] for row in unit_results], dtype=np.float64),
        step0_r2_median=np.array([row["step0_r2_median"] for row in unit_results], dtype=np.float64),
        step0_resid_median=np.array([row["step0_resid_median"] for row in unit_results], dtype=np.float64),
        step0_cosine_median=np.array([row["step0_cosine_median"] for row in unit_results], dtype=np.float64),
    )


def _resolve_pixels_per_degree(data: dict, config: ModelBackendConfig) -> float:
    if config.pixels_per_degree is not None:
        return float(config.pixels_per_degree)
    dataset = data.get("dataset")
    if dataset is None or not getattr(dataset, "dsets", None):
        raise ValueError("Unable to infer pixels-per-degree from dataset; pass --pixels-per-degree")
    ppd = dataset.dsets[0].metadata.get("ppd")
    if ppd is None:
        raise ValueError("Dataset metadata does not expose 'ppd'; pass --pixels-per-degree")
    return float(ppd)


def _valid_xy(eyepos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(eyepos).all(axis=-1)
    x = eyepos[..., 0]
    y = eyepos[..., 1]
    return x, y, finite_mask


def compute_centered_radius_deg(eyepos: np.ndarray) -> np.ndarray:
    x, y, finite_mask = _valid_xy(eyepos)
    centered = np.full_like(eyepos, np.nan, dtype=np.float64)
    for trial_idx in range(eyepos.shape[0]):
        valid = finite_mask[trial_idx]
        if not np.any(valid):
            continue
        xy = eyepos[trial_idx, valid]
        xy_centered = xy - np.nanmedian(xy, axis=0, keepdims=True)
        centered[trial_idx, valid] = xy_centered
    return np.linalg.norm(centered, axis=-1)


def compute_step_displacement_deg(eyepos: np.ndarray) -> np.ndarray:
    steps = np.full(eyepos.shape[:2], np.nan, dtype=np.float64)
    for trial_idx in range(eyepos.shape[0]):
        trial = eyepos[trial_idx]
        finite = np.isfinite(trial).all(axis=-1)
        valid_idx = np.flatnonzero(finite)
        if valid_idx.size < 2:
            continue
        diffs = np.diff(trial[valid_idx], axis=0)
        steps[trial_idx, valid_idx[1:]] = np.linalg.norm(diffs, axis=-1)
    return steps


def _empirical_displacement_magnitudes_px(
    eyepos: np.ndarray,
    *,
    pixels_per_degree: float,
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    if not percentiles:
        return tuple()
    step_deg = compute_step_displacement_deg(eyepos)
    return _empirical_displacement_magnitudes_px_from_values(
        step_deg,
        pixels_per_degree=pixels_per_degree,
        percentiles=percentiles,
    )


def _empirical_displacement_magnitudes_px_from_values(
    values_deg: np.ndarray,
    *,
    pixels_per_degree: float,
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    finite = np.asarray(values_deg, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return tuple()
    values = np.percentile(finite, np.asarray(percentiles, dtype=np.float64)) * float(pixels_per_degree)
    return tuple(float(x) for x in values)


def _baseline_relative_displacement_magnitudes_px(
    centered_eye_px: np.ndarray,
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    """Percentiles of baseline-relative eye-offset radii (pixels).

    These are the displacements actually used to generate Step 1 shifted
    responses, so they are the correct distribution for the central-mass
    Step 0 linearization gate.
    """
    if not percentiles:
        return tuple()
    radii = np.linalg.norm(centered_eye_px, axis=1)
    finite = radii[np.isfinite(radii)]
    if finite.size == 0:
        return tuple()
    return tuple(float(x) for x in np.percentile(finite, np.asarray(percentiles, dtype=np.float64)))


def _merge_displacement_magnitudes(*groups: tuple[float, ...]) -> tuple[float, ...]:
    merged = []
    for group in groups:
        merged.extend(float(x) for x in group if np.isfinite(x) and x > 0)
    if not merged:
        return tuple()
    rounded = {round(value, 6): value for value in merged}
    return tuple(sorted(rounded.values()))


def _central_mass_metric_keys(backend_result: dict, prefix: str, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    keys = []
    for magnitude in backend_result.get("median_empirical_displacement_magnitudes_px", [])[:2]:
        key = f"{prefix}{float(magnitude):g}"
        if key in rows[0]:
            keys.append(key)
    return keys


def _empirical_displacement_field_name(percentile: float) -> str:
    # Stores baseline-relative offset radii — the displacements that drive Step 1 covariance.
    return f"empirical_displacement_px_p{float(percentile):g}"


def _step_displacement_field_name(percentile: float) -> str:
    # Stores frame-to-frame step magnitudes — kept as a diagnostic only.
    return f"step_displacement_px_p{float(percentile):g}"


def _median_empirical_displacement_magnitudes_px(
    backend_units: list[dict],
    percentiles: Iterable[float],
) -> tuple[float, ...]:
    medians = []
    for percentile in percentiles:
        key = _empirical_displacement_field_name(percentile)
        values = [row[key] for row in backend_units if key in row and np.isfinite(row[key])]
        if values:
            medians.append(float(np.nanmedian(values)))
    return tuple(medians)


def _median_step_displacement_magnitudes_px(
    backend_units: list[dict],
    percentiles: Iterable[float],
) -> tuple[float, ...]:
    medians = []
    for percentile in percentiles:
        key = _step_displacement_field_name(percentile)
        values = [row[key] for row in backend_units if key in row and np.isfinite(row[key])]
        if values:
            medians.append(float(np.nanmedian(values)))
    return tuple(medians)


def _central_mass_window_text(backend_units: list[dict], backend_result: dict) -> str:
    medians = backend_result.get("median_empirical_displacement_magnitudes_px", [])[:2]
    if not medians:
        return "n/a"
    return ", ".join(f"{float(value):.3f} px" for value in medians)


def _small_displacement_metric_keys(rows: list[dict], prefix: str, max_magnitude_px: float = 0.25) -> list[str]:
    if not rows:
        return []
    keys = []
    for key in rows[0]:
        if not key.startswith(prefix):
            continue
        try:
            magnitude = float(key.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if magnitude <= max_magnitude_px + 1e-12:
            keys.append(key)
    return sorted(keys, key=lambda item: float(item.rsplit("_", 1)[-1]))


def _median_over_keys(row: dict, keys: list[str]) -> float:
    if not keys:
        return float("nan")
    values = [row[key] for key in keys if key in row and np.isfinite(row[key])]
    if not values:
        return float("nan")
    return float(np.nanmedian(values))


def _step_stability_lines(backend_units: list[dict]) -> list[str]:
    if not backend_units:
        return []
    lines = []
    first_row = backend_units[0]
    alignment_keys = sorted(key for key in first_row if key.startswith("jacobian_step_alignment_"))
    ratio_keys = sorted(key for key in first_row if key.startswith("jacobian_norm_ratio_"))
    for key in alignment_keys:
        values = [row[key] for row in backend_units if np.isfinite(row[key])]
        if values:
            lines.append(f"- median {key.replace('_', ' ')}: {float(np.nanmedian(values)):.6f}")
    for key in ratio_keys:
        values = [row[key] for row in backend_units if np.isfinite(row[key])]
        if values:
            lines.append(f"- median {key.replace('_', ' ')}: {float(np.nanmedian(values)):.6f}")
    return lines


def _classify_step0_regime(row: dict, backend_result: dict) -> str:
    central_r2 = row.get("central_mass_r2_median", float("nan"))
    if np.isfinite(central_r2):
        if central_r2 > 0.5:
            return "good"
        if central_r2 > 0.0:
            return "conditional"
        return "poor"
    if row["step0_r2_median"] > 0.5:
        return "good"
    if row["step0_r2_median"] > 0.0:
        return "conditional"
    return "poor"


def _classify_small_displacement_regime(row: dict) -> str:
    small_r2_keys = _small_displacement_metric_keys([row], "r2_median_px_")
    small_r2 = _median_over_keys(row, small_r2_keys)
    if np.isfinite(small_r2):
        if small_r2 > 0.5:
            return "good"
        if small_r2 > 0.0:
            return "conditional"
        return "poor"
    return "poor"


def _subset_metric_summary(backend_units: list[dict], backend_result: dict) -> list[str]:
    if not backend_units:
        return ["- no backend units evaluated"]
    summaries = []
    for regime in ("good", "conditional", "poor"):
        subset = [row for row in backend_units if _classify_step0_regime(row, backend_result) == regime]
        if not subset:
            summaries.append(f"- {regime}: no units")
            continue
        align = np.array([row["alignment_A_J"] for row in subset], dtype=np.float64)
        align_shuffle = np.array([row["image_shuffle_alignment_median"] for row in subset], dtype=np.float64)
        capture = np.array([row["capture_V_J"] for row in subset], dtype=np.float64)
        capture_shuffle = np.array([row["image_shuffle_capture_median"] for row in subset], dtype=np.float64)
        summaries.append(
            "- {regime}: n={n}, median matched A_J={align:.6f}, median img-shuf A_J={align_shuffle:.6f}, delta A_J={delta_align:.6f}, median matched V_J={capture:.6f}, median img-shuf V_J={capture_shuffle:.6f}, delta V_J={delta_capture:.6f}".format(
                regime=regime,
                n=len(subset),
                align=float(np.nanmedian(align)),
                align_shuffle=float(np.nanmedian(align_shuffle)),
                delta_align=float(np.nanmedian(align - align_shuffle)),
                capture=float(np.nanmedian(capture)),
                capture_shuffle=float(np.nanmedian(capture_shuffle)),
                delta_capture=float(np.nanmedian(capture - capture_shuffle)),
            )
        )
    return summaries


def _small_displacement_subset_summary(backend_units: list[dict]) -> list[str]:
    if not backend_units:
        return ["- no backend units evaluated"]
    summaries = []
    for regime in ("good", "conditional", "poor"):
        subset = [row for row in backend_units if _classify_small_displacement_regime(row) == regime]
        if not subset:
            summaries.append(f"- {regime}: no units")
            continue
        align = np.array([row["alignment_A_J"] for row in subset], dtype=np.float64)
        align_shuffle = np.array([row["image_shuffle_alignment_median"] for row in subset], dtype=np.float64)
        capture = np.array([row["capture_V_J"] for row in subset], dtype=np.float64)
        capture_shuffle = np.array([row["image_shuffle_capture_median"] for row in subset], dtype=np.float64)
        summaries.append(
            "- {regime}: n={n}, median matched A_J={align:.6f}, median img-shuf A_J={align_shuffle:.6f}, delta A_J={delta_align:.6f}, median matched V_J={capture:.6f}, median img-shuf V_J={capture_shuffle:.6f}, delta V_J={delta_capture:.6f}".format(
                regime=regime,
                n=len(subset),
                align=float(np.nanmedian(align)),
                align_shuffle=float(np.nanmedian(align_shuffle)),
                delta_align=float(np.nanmedian(align - align_shuffle)),
                capture=float(np.nanmedian(capture)),
                capture_shuffle=float(np.nanmedian(capture_shuffle)),
                delta_capture=float(np.nanmedian(capture - capture_shuffle)),
            )
        )
    return summaries


def _paired_delta_summary(rows: list[dict], matched_key: str, shuffled_key: str) -> dict:
    if not rows:
        return {
            "n": 0,
            "matched_median": float("nan"),
            "shuffled_median": float("nan"),
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_count": 0,
            "nonnegative_count": 0,
        }

    matched = np.array([row[matched_key] for row in rows], dtype=np.float64)
    shuffled = np.array([row[shuffled_key] for row in rows], dtype=np.float64)
    valid = np.isfinite(matched) & np.isfinite(shuffled)
    matched = matched[valid]
    shuffled = shuffled[valid]
    if matched.size == 0:
        return {
            "n": 0,
            "matched_median": float("nan"),
            "shuffled_median": float("nan"),
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_count": 0,
            "nonnegative_count": 0,
        }

    delta = matched - shuffled
    rng = np.random.default_rng(0)
    boot = np.empty(10000, dtype=np.float64)
    for idx in range(boot.size):
        sample_idx = rng.integers(0, delta.size, size=delta.size)
        boot[idx] = np.nanmedian(delta[sample_idx])
    return {
        "n": int(delta.size),
        "matched_median": float(np.nanmedian(matched)),
        "shuffled_median": float(np.nanmedian(shuffled)),
        "median_delta": float(np.nanmedian(delta)),
        "mean_delta": float(np.nanmean(delta)),
        "ci95_low": float(np.nanpercentile(boot, 2.5)),
        "ci95_high": float(np.nanpercentile(boot, 97.5)),
        "positive_count": int(np.sum(delta > 0.0)),
        "nonnegative_count": int(np.sum(delta >= 0.0)),
    }


def _paired_delta_summary_md(summary: dict, metric_label: str) -> str:
    return (
        f"- {metric_label}: n={summary['n']}, matched median={summary['matched_median']:.6f}, "
        f"img-shuf median={summary['shuffled_median']:.6f}, median delta={summary['median_delta']:.6f}, "
        f"mean delta={summary['mean_delta']:.6f}, bootstrap 95% CI=[{summary['ci95_low']:.6f}, {summary['ci95_high']:.6f}], "
        f"positive units={summary['positive_count']}/{summary['n']}, nonnegative units={summary['nonnegative_count']}/{summary['n']}"
    )


def _paired_delta_summaries(backend_units: list[dict]) -> dict:
    small_good_units = [row for row in backend_units if _classify_small_displacement_regime(row) == "good"]
    return {
        "alignment_all": _paired_delta_summary(backend_units, "alignment_A_J", "image_shuffle_alignment_median"),
        "capture_all": _paired_delta_summary(backend_units, "capture_V_J", "image_shuffle_capture_median"),
        "alignment_small_good": _paired_delta_summary(small_good_units, "alignment_A_J", "image_shuffle_alignment_median"),
        "capture_small_good": _paired_delta_summary(small_good_units, "capture_V_J", "image_shuffle_capture_median"),
    }


def _small_good_focus_lines(backend_units: list[dict]) -> list[str]:
    if not backend_units:
        return ["- no backend units evaluated"]
    paired = _paired_delta_summaries(backend_units)
    lines = []
    if paired["alignment_small_good"]["n"] > 0:
        lines.append(
            "- primary alignment readout: small-displacement-good units show matched-minus-shuffled median $A_J$ delta {delta:.6f} with bootstrap 95% CI [{lo:.6f}, {hi:.6f}] across {n} units.".format(
                delta=paired["alignment_small_good"]["median_delta"],
                lo=paired["alignment_small_good"]["ci95_low"],
                hi=paired["alignment_small_good"]["ci95_high"],
                n=paired["alignment_small_good"]["n"],
            )
        )
    else:
        lines.append("- primary alignment readout: no small-displacement-good units were available.")
    lines.append(
        "- all-unit context: matched-minus-shuffled median $A_J$ delta is {delta:.6f} with bootstrap 95% CI [{lo:.6f}, {hi:.6f}] across {n} units.".format(
            delta=paired["alignment_all"]["median_delta"],
            lo=paired["alignment_all"]["ci95_low"],
            hi=paired["alignment_all"]["ci95_high"],
            n=paired["alignment_all"]["n"],
        )
    )
    return lines


def _backend_unit_table_md(backend_units: list[dict], backend_result: dict) -> str:
    if not backend_units:
        return "No backend units evaluated."
    small_r2_keys = _small_displacement_metric_keys(backend_units, "r2_median_px_")
    small_resid_keys = _small_displacement_metric_keys(backend_units, "resid_median_px_")
    small_cosine_keys = _small_displacement_metric_keys(backend_units, "cosine_median_px_")
    lines = [
        "| Unit | Samples | Step 0 median R2 | Step 0 median resid | Step 0 median cosine | Small-disp R2 | Small-disp resid | Small-disp cosine | Central-mass R2 | Central-mass resid | Central-mass cosine | Alignment A_J | Capture V_J | Img-shuf A_J | Img-shuf V_J |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in backend_units:
        lines.append(
            "| {unit_id} | {n_samples:.0f} | {step0_r2_median:.4f} | {step0_resid_median:.4f} | {step0_cosine_median:.4f} | {small_r2:.4f} | {small_resid:.4f} | {small_cosine:.4f} | {central_r2:.4f} | {central_resid:.4f} | {central_cosine:.4f} | {alignment_A_J:.4f} | {capture_V_J:.4f} | {image_shuffle_alignment_median:.4f} | {image_shuffle_capture_median:.4f} |".format(
                unit_id=row["unit_id"],
                n_samples=row["n_samples"],
                step0_r2_median=row["step0_r2_median"],
                step0_resid_median=row["step0_resid_median"],
                step0_cosine_median=row["step0_cosine_median"],
                small_r2=_median_over_keys(row, small_r2_keys),
                small_resid=_median_over_keys(row, small_resid_keys),
                small_cosine=_median_over_keys(row, small_cosine_keys),
                central_r2=row.get("central_mass_r2_median", float("nan")),
                central_resid=row.get("central_mass_resid_median", float("nan")),
                central_cosine=row.get("central_mass_cosine_median", float("nan")),
                alignment_A_J=row["alignment_A_J"],
                capture_V_J=row["capture_V_J"],
                image_shuffle_alignment_median=row["image_shuffle_alignment_median"],
                image_shuffle_capture_median=row["image_shuffle_capture_median"],
            )
        )
    return "\n".join(lines)


def summarize_percentiles(values: np.ndarray, percentiles: tuple[int, ...] = (50, 75, 90, 95)) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{pct}": float("nan") for pct in percentiles}
    return {f"p{pct}": float(np.percentile(finite, pct)) for pct in percentiles}


def _compute_fixation_phase_bins(usable: np.ndarray, n_phase_bins: int) -> np.ndarray:
    phase_bins = np.full(usable.shape, -1, dtype=np.int64)
    for trial_idx in range(usable.shape[0]):
        valid_idx = np.flatnonzero(usable[trial_idx])
        if valid_idx.size == 0:
            continue
        rel_pos = np.arange(valid_idx.size, dtype=np.float64) / valid_idx.size
        bins = np.clip((rel_pos * n_phase_bins).astype(np.int64), 0, n_phase_bins - 1)
        phase_bins[trial_idx, valid_idx] = bins
    return phase_bins


def _compute_radius_bins(radius_deg: np.ndarray, usable: np.ndarray, n_radius_bins: int) -> np.ndarray:
    radius_bins = np.full(radius_deg.shape, -1, dtype=np.int64)
    finite_radius = radius_deg[usable]
    if finite_radius.size == 0 or n_radius_bins <= 1:
        radius_bins[usable] = 0
        return radius_bins

    quantiles = np.linspace(0.0, 100.0, n_radius_bins + 1)[1:-1]
    edges = np.percentile(finite_radius, quantiles)
    edges = np.unique(edges[np.isfinite(edges)])
    radius_bins[usable] = np.digitize(radius_deg[usable], edges, right=False)
    return radius_bins


def build_image_units(
    image_ids: np.ndarray,
    radius_deg: np.ndarray,
    min_samples: int,
    unit_mode: str,
    n_fixation_phase_bins: int,
    n_radius_bins: int,
) -> list[AnalysisUnit]:
    units: list[AnalysisUnit] = []
    finite_radius = np.isfinite(radius_deg)
    finite_image = image_ids >= 0
    usable = finite_radius & finite_image
    image_values = np.unique(image_ids[usable])
    phase_bins = _compute_fixation_phase_bins(usable, n_fixation_phase_bins)
    radius_bins = _compute_radius_bins(radius_deg, usable, n_radius_bins)
    trial_index_grid = np.broadcast_to(np.arange(image_ids.shape[0], dtype=np.int64)[:, None], image_ids.shape)

    for image_id in image_values:
        if unit_mode == "image":
            groups = [(None, None, usable & (image_ids == image_id))]
        elif unit_mode == "image_phase":
            groups = [
                (phase_bin, None, usable & (image_ids == image_id) & (phase_bins == phase_bin))
                for phase_bin in range(n_fixation_phase_bins)
            ]
        elif unit_mode == "image_phase_radius":
            groups = [
                (
                    phase_bin,
                    radius_bin,
                    usable & (image_ids == image_id) & (phase_bins == phase_bin) & (radius_bins == radius_bin),
                )
                for phase_bin in range(n_fixation_phase_bins)
                for radius_bin in range(n_radius_bins)
            ]
        else:
            raise ValueError(f"Unsupported unit mode: {unit_mode}")

        for group in groups:
            phase_bin, radius_bin, group_mask = group
            sample_trial_idx, time_idx = np.where(group_mask)
            if sample_trial_idx.size < min_samples:
                continue
            radii = radius_deg[sample_trial_idx, time_idx]
            unit_id = f"image_{int(image_id):05d}"
            if phase_bin is not None:
                unit_id = f"{unit_id}_phase_{int(phase_bin):02d}"
            if radius_bin is not None:
                unit_id = f"{unit_id}_radius_{int(radius_bin):02d}"
            units.append(
                AnalysisUnit(
                    unit_id=unit_id,
                    image_id=int(image_id),
                    unit_mode=unit_mode,
                    phase_bin=None if phase_bin is None else int(phase_bin),
                    radius_bin=None if radius_bin is None else int(radius_bin),
                    trial_indices=tuple(int(x) for x in sample_trial_idx.tolist()),
                    time_indices=tuple(int(x) for x in time_idx.tolist()),
                    n_samples=int(sample_trial_idx.size),
                    median_radius_deg=float(np.nanmedian(radii)),
                    p75_radius_deg=float(np.nanpercentile(radii, 75)),
                    p90_radius_deg=float(np.nanpercentile(radii, 90)),
                )
            )

    units.sort(key=lambda unit: (-unit.n_samples, unit.image_id))
    return units


def write_units_csv(path: Path, units: list[AnalysisUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_id",
                "image_id",
                "unit_mode",
                "phase_bin",
                "radius_bin",
                "n_samples",
                "median_radius_deg",
                "p75_radius_deg",
                "p90_radius_deg",
            ],
        )
        writer.writeheader()
        for unit in units:
            writer.writerow(
                {
                    "unit_id": unit.unit_id,
                    "image_id": unit.image_id,
                    "unit_mode": unit.unit_mode,
                    "phase_bin": -1 if unit.phase_bin is None else unit.phase_bin,
                    "radius_bin": -1 if unit.radius_bin is None else unit.radius_bin,
                    "n_samples": unit.n_samples,
                    "median_radius_deg": unit.median_radius_deg,
                    "p75_radius_deg": unit.p75_radius_deg,
                    "p90_radius_deg": unit.p90_radius_deg,
                }
            )


def write_summary_md(
    path: Path,
    *,
    subject: str,
    date: str,
    dataset_configs_path: str,
    radius_summary: dict,
    step_summary: dict,
    n_trials: int,
    n_units: int,
    min_samples: int,
    unit_mode: str,
    backend_name: str,
    backend_result: dict,
    backend_units: list[dict] | None,
    figure_paths: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_shuffle_alignment = float("nan")
    image_shuffle_capture = float("nan")
    random_alignment = float("nan")
    random_capture = float("nan")
    replay_stim_mae = float("nan")
    replay_stim_max_abs = float("nan")
    replay_resp_l2 = float("nan")
    replay_resp_corr = float("nan")
    step0_good_fraction = float("nan")
    step0_conditional_fraction = float("nan")
    central_mass_good_fraction = float("nan")
    small_displacement_good_fraction = float("nan")
    central_mass_window = "n/a"
    small_displacement_window = "n/a"
    stability_lines = ["- none"]
    subset_summary_lines = ["- no backend units evaluated"]
    small_subset_summary_lines = ["- no backend units evaluated"]
    paired_delta_lines = ["- no backend units evaluated"]
    small_good_paired_delta_lines = ["- no small-displacement-good backend units"]
    small_good_focus_lines = ["- no backend units evaluated"]
    backend_unit_table = "No backend units evaluated."
    if backend_units:
        image_shuffle_alignment = float(np.nanmedian([row["image_shuffle_alignment_median"] for row in backend_units]))
        image_shuffle_capture = float(np.nanmedian([row["image_shuffle_capture_median"] for row in backend_units]))
        random_alignment = float(np.nanmedian([row["random_subspace_alignment_median"] for row in backend_units]))
        random_capture = float(np.nanmedian([row["random_subspace_capture_median"] for row in backend_units]))
        replay_stim_mae = float(np.nanmedian([row["replay_stim_mae"] for row in backend_units]))
        replay_stim_max_abs = float(np.nanmedian([row["replay_stim_max_abs"] for row in backend_units]))
        replay_resp_l2 = float(np.nanmedian([row["replay_resp_l2"] for row in backend_units]))
        replay_resp_corr = float(np.nanmedian([row["replay_resp_corr"] for row in backend_units]))
        step0_good_fraction = float(np.mean([row["step0_r2_median"] > 0.5 for row in backend_units]))
        step0_conditional_fraction = float(
            np.mean([(row["step0_r2_median"] > 0.0) and (row["step0_r2_median"] <= 0.5) for row in backend_units])
        )
        small_r2_keys = _small_displacement_metric_keys(backend_units, "r2_median_px_")
        if small_r2_keys:
            small_displacement_window = ", ".join(
                f"{float(key.rsplit('_', 1)[-1]):.3f} px" for key in small_r2_keys
            )
            small_displacement_passes = []
            for row in backend_units:
                value = _median_over_keys(row, small_r2_keys)
                if np.isfinite(value):
                    small_displacement_passes.append(value > 0.5)
            if small_displacement_passes:
                small_displacement_good_fraction = float(np.mean(small_displacement_passes))
        central_mass_window = _central_mass_window_text(backend_units, backend_result)
        central_mass_passes = []
        for row in backend_units:
            value = row.get("central_mass_r2_median", float("nan"))
            if np.isfinite(value):
                central_mass_passes.append(value > 0.5)
        if central_mass_passes:
            central_mass_good_fraction = float(np.mean(central_mass_passes))
        stability_lines = _step_stability_lines(backend_units) or ["- none"]
        subset_summary_lines = _subset_metric_summary(backend_units, backend_result)
        small_subset_summary_lines = _small_displacement_subset_summary(backend_units)
        paired = _paired_delta_summaries(backend_units)
        paired_delta_lines = [
            _paired_delta_summary_md(paired["alignment_all"], "alignment $A_J$ paired delta"),
            _paired_delta_summary_md(paired["capture_all"], "capture $V_J$ paired delta"),
        ]
        if paired["alignment_small_good"]["n"] > 0:
            small_good_paired_delta_lines = [
                _paired_delta_summary_md(paired["alignment_small_good"], "small-displacement-good alignment $A_J$ paired delta"),
                _paired_delta_summary_md(paired["capture_small_good"], "small-displacement-good capture $V_J$ paired delta"),
            ]
        small_good_focus_lines = _small_good_focus_lines(backend_units)
        backend_unit_table = _backend_unit_table_md(backend_units, backend_result)

    figure_lines = "\n".join(f"- {label}: {rel_path}" for label, rel_path in figure_paths.items()) if figure_paths else "- none"
    stability_text = "\n".join(stability_lines)
    subset_summary_text = "\n".join(subset_summary_lines)
    small_subset_summary_text = "\n".join(small_subset_summary_lines)
    paired_delta_text = "\n".join(paired_delta_lines)
    small_good_paired_delta_text = "\n".join(small_good_paired_delta_lines)
    small_good_focus_text = "\n".join(small_good_focus_lines)
    effective_magnitudes = backend_result.get("effective_displacement_magnitudes_px", [])
    effective_magnitude_text = ", ".join(f"{float(value):.3f} px" for value in effective_magnitudes) if effective_magnitudes else "none"
    text = f"""# fixRSVP Step 0/1 scaffold summary

## Scope

- subject: {subject}
- date: {date}
- dataset config: {dataset_configs_path}
- backend: {backend_name}

## Analysis unit

Primary unit for this scaffold is defined by `{unit_mode}` over valid fixRSVP bins with finite eye position.
Units are retained only if they have at least {min_samples} valid samples across trials.

## Data summary

- trials after fixRSVP preprocessing: {n_trials}
- retained image units: {n_units}
- active unit mode: {unit_mode}

## Empirical displacement summaries

Centered eye-position radius in degrees, pooled over valid bins:

- median: {radius_summary['p50']:.6f}
- p75: {radius_summary['p75']:.6f}
- p90: {radius_summary['p90']:.6f}
- p95: {radius_summary['p95']:.6f}

Frame-to-frame eye-step magnitude in degrees, pooled over valid bins:

- median: {step_summary['p50']:.6f}
- p75: {step_summary['p75']:.6f}
- p90: {step_summary['p90']:.6f}
- p95: {step_summary['p95']:.6f}

## Status

This run establishes the fixRSVP analysis-unit manifest and the Step 0 displacement gates.

- backend status: {backend_result['status']}
- backend units evaluated: {backend_result.get('n_units', 0)}
- attempted backend units before post-filter sample gate: {backend_result.get('attempted_units', 0)}
- post-filter minimum backend samples: {backend_result.get('min_backend_samples', 0)}
- units skipped for low post-filter sample count: {backend_result.get('skipped_low_samples', 0)}
- units skipped by absolute-radius filter: {backend_result.get('skipped_radius_filter', 0)}
- absolute-radius filter threshold: {backend_result.get('max_baseline_relative_radius_px') or 'none (all samples retained)'}
- backend pixels/degree: {backend_result.get('pixels_per_degree', float('nan')):.6f}
- Step 0 displacement magnitudes evaluated: {effective_magnitude_text}
- backend median alignment A_J: {backend_result.get('median_alignment_A_J', float('nan')):.6f}
- backend median capture V_J: {backend_result.get('median_capture_V_J', float('nan')):.6f}

## Raw-Row Replay Sanity Check

- median replay stimulus MAE: {replay_stim_mae:.6f}
- median replay stimulus max abs diff: {replay_stim_max_abs:.6f}
- median replay response L2 diff: {replay_resp_l2:.6f}
- median replay response correlation: {replay_resp_corr:.6f}

## Step 0 interpretation

- fraction of backend units with median Step 0 $R^2_{{lin}} > 0.5$: {step0_good_fraction:.6f}
- fraction of backend units in the conditional regime $0 < R^2_{{lin}} \\le 0.5$: {step0_conditional_fraction:.6f}
- fraction of backend units with median Step 0 $R^2_{{lin}} > 0.5$ over small displacements ({small_displacement_window}): {small_displacement_good_fraction:.6f}
- fraction of backend units with median Step 0 $R^2_{{lin}} > 0.5$ over empirical central-mass displacements ({central_mass_window}): {central_mass_good_fraction:.6f}

## Step 1 interpretation

{small_good_focus_text}
- image-shuffled median alignment A_J: {image_shuffle_alignment:.6f}
- image-shuffled median capture V_J: {image_shuffle_capture:.6f}
- random-subspace median alignment A_J: {random_alignment:.6f}
- random-subspace median capture V_J: {random_capture:.6f}

## Step 1 By Step 0 Regime

{subset_summary_text}

## Step 1 By Small-Displacement Regime

{small_subset_summary_text}

## Paired Matched-vs-Shuffled Deltas

{paired_delta_text}

## Small-Displacement-Good Paired Deltas

{small_good_paired_delta_text}

## Finite-Difference Step Stability

{stability_text}

## Backend Unit Table

{backend_unit_table}

## Generated diagnostics

{figure_lines}

Backend interpretation:

- `manifest-only` means only unit manifests and empirical displacement summaries were saved.
- `lagged-stimulus-finite-difference` means a common-translation Jacobian was estimated around one representative lagged-input baseline per unit, and model FEM covariance was computed from controlled shifted copies of that same baseline stimulus.

Current unit definition caveat:

- units are now grouped by `{unit_mode}`, but the model FEM covariance is still computed from one representative baseline snippet per unit and empirical centered eye displacements applied to that fixed baseline.

Current missing pieces relative to the handoff:

- replay is currently a raw-row baseline sanity check rather than a full image-stack-plus-eye-trace replay
- no example stimulus panel with eye covariance ellipse yet
- no empirical neural bridge yet; current outputs are still model-internal Step 0/1 diagnostics
"""
    path.write_text(text)


def write_results_so_far_md(
    path: Path,
    *,
    subject: str,
    date: str,
    dataset_configs_path: str,
    radius_summary: dict,
    step_summary: dict,
    n_trials: int,
    n_units: int,
    unit_mode: str,
    backend_name: str,
    backend_result: dict,
    backend_units: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_magnitudes = backend_result.get("effective_displacement_magnitudes_px", [])
    empirical_magnitudes = backend_result.get("median_empirical_displacement_magnitudes_px", [])
    subset_summary = "\n".join(_subset_metric_summary(backend_units, backend_result)) if backend_units else "- no backend units evaluated"
    small_subset_summary = "\n".join(_small_displacement_subset_summary(backend_units)) if backend_units else "- no backend units evaluated"
    stability_summary = "\n".join(_step_stability_lines(backend_units)) if backend_units else "- none"
    unit_table = _backend_unit_table_md(backend_units, backend_result)
    paired = _paired_delta_summaries(backend_units) if backend_units else _paired_delta_summaries([])
    paired_delta_summary = "\n".join(
        [
            _paired_delta_summary_md(paired["alignment_all"], "alignment $A_J$ paired delta"),
            _paired_delta_summary_md(paired["capture_all"], "capture $V_J$ paired delta"),
        ]
    )
    small_good_focus_summary = "\n".join(_small_good_focus_lines(backend_units)) if backend_units else "- no backend units evaluated"
    small_good_paired_delta_summary = "\n".join(
        [
            _paired_delta_summary_md(paired["alignment_small_good"], "small-displacement-good alignment $A_J$ paired delta"),
            _paired_delta_summary_md(paired["capture_small_good"], "small-displacement-good capture $V_J$ paired delta"),
        ]
    ) if paired["alignment_small_good"]["n"] > 0 else "- no small-displacement-good backend units"

    image_shuffle_alignment = float(np.nanmedian([row["image_shuffle_alignment_median"] for row in backend_units])) if backend_units else float("nan")
    image_shuffle_capture = float(np.nanmedian([row["image_shuffle_capture_median"] for row in backend_units])) if backend_units else float("nan")
    random_alignment = float(np.nanmedian([row["random_subspace_alignment_median"] for row in backend_units])) if backend_units else float("nan")
    random_capture = float(np.nanmedian([row["random_subspace_capture_median"] for row in backend_units])) if backend_units else float("nan")
    replay_stim_mae = float(np.nanmedian([row["replay_stim_mae"] for row in backend_units])) if backend_units else float("nan")
    replay_stim_max_abs = float(np.nanmedian([row["replay_stim_max_abs"] for row in backend_units])) if backend_units else float("nan")
    replay_resp_l2 = float(np.nanmedian([row["replay_resp_l2"] for row in backend_units])) if backend_units else float("nan")
    replay_resp_corr = float(np.nanmedian([row["replay_resp_corr"] for row in backend_units])) if backend_units else float("nan")
    good_fraction = float(np.mean([row["step0_r2_median"] > 0.5 for row in backend_units])) if backend_units else float("nan")
    conditional_fraction = float(np.mean([(row["step0_r2_median"] > 0.0) and (row["step0_r2_median"] <= 0.5) for row in backend_units])) if backend_units else float("nan")
    small_r2_keys = _small_displacement_metric_keys(backend_units, "r2_median_px_")
    small_good_fraction = float("nan")
    if backend_units and small_r2_keys:
        small_pass = []
        for row in backend_units:
            value = _median_over_keys(row, small_r2_keys)
            if np.isfinite(value):
                small_pass.append(value > 0.5)
        if small_pass:
            small_good_fraction = float(np.mean(small_pass))
    central_mass_good_fraction = float("nan")
    if backend_units:
        central_pass = []
        for row in backend_units:
            value = row.get("central_mass_r2_median", float("nan"))
            if np.isfinite(value):
                central_pass.append(value > 0.5)
        if central_pass:
            central_mass_good_fraction = float(np.mean(central_pass))

    results_text = f"""# fixRSVP Jacobian Results So Far

## Scope

This note summarizes the current Allen smoke run for Step 0 and Step 1 of the predictive Jacobian framework.

Current run:

| Field | Value |
| --- | --- |
| Subject | {subject} |
| Date | {date} |
| Dataset config | {dataset_configs_path} |
| Backend | {backend_name} |
| Unit mode | {unit_mode} |
| Backend units evaluated | {backend_result.get('n_units', 0)} |
| Attempted backend units | {backend_result.get('attempted_units', 0)} |
| Post-filter minimum backend samples | {backend_result.get('min_backend_samples', 0)} |
| Units skipped for low post-filter sample count | {backend_result.get('skipped_low_samples', 0)} |
| Retained analysis units | {n_units} |
| Trials after preprocessing | {n_trials} |
| Pixels per degree | {backend_result.get('pixels_per_degree', float('nan')):.6f} |

## Empirical Eye-Movement Summary

| Metric | Median | p75 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Centered eye-position radius (deg) | {radius_summary['p50']:.6f} | {radius_summary['p75']:.6f} | {radius_summary['p90']:.6f} | {radius_summary['p95']:.6f} |
| Frame-to-frame eye-step magnitude (deg) | {step_summary['p50']:.6f} | {step_summary['p75']:.6f} | {step_summary['p90']:.6f} | {step_summary['p95']:.6f} |

Step 0 displacement magnitudes evaluated by the backend:

| Source | Magnitudes |
| --- | --- |
| Explicit plus median unit-local empirical set | {', '.join(f'{float(value):.3f} px' for value in effective_magnitudes) if effective_magnitudes else 'none'} |
| Median unit-local empirical percentile magnitudes | {', '.join(f'{float(value):.3f} px' for value in empirical_magnitudes) if empirical_magnitudes else 'none'} |

## Aggregate Results

### Step 0 linearization gate

| Quantity | Value |
| --- | ---: |
| Fraction of backend units with median $R^2_{{lin}} > 0.5$ | {good_fraction:.6f} |
| Fraction of backend units with $0 < R^2_{{lin}} \\le 0.5$ | {conditional_fraction:.6f} |
| Fraction with median $R^2_{{lin}} > 0.5$ over small displacements | {small_good_fraction:.6f} |
| Fraction with median $R^2_{{lin}} > 0.5$ over empirical central-mass displacements | {central_mass_good_fraction:.6f} |

### Step 1 geometry

| Quantity | Value |
| --- | ---: |
| Median matched alignment $A_J$ | {backend_result.get('median_alignment_A_J', float('nan')):.6f} |
| Median matched capture $V_J$ | {backend_result.get('median_capture_V_J', float('nan')):.6f} |
| Median image-shuffled alignment $A_J$ | {image_shuffle_alignment:.6f} |
| Median image-shuffled capture $V_J$ | {image_shuffle_capture:.6f} |
| Median random-subspace alignment $A_J$ | {random_alignment:.6f} |
| Median random-subspace capture $V_J$ | {random_capture:.6f} |

### Default interpretation focus

{small_good_focus_summary}

### Raw-row replay sanity check

| Quantity | Value |
| --- | ---: |
| Median replay stimulus MAE | {replay_stim_mae:.6f} |
| Median replay stimulus max abs diff | {replay_stim_max_abs:.6f} |
| Median replay response L2 diff | {replay_resp_l2:.6f} |
| Median replay response corr | {replay_resp_corr:.6f} |

## Step 1 By Step 0 Regime

{subset_summary}

## Step 1 By Small-Displacement Regime

{small_subset_summary}

## Paired Matched-vs-Shuffled Deltas

{paired_delta_summary}

## Small-Displacement-Good Paired Deltas

{small_good_paired_delta_summary}

## Finite-Difference Step Stability

{stability_summary}

## Per-Unit Backend Table

{unit_table}

## Interpretation So Far

| Topic | Interpretation |
| --- | --- |
| Step 0 gate | The stronger smoke-test gate now removes the unstable low-sample row and shows that local linearization is not uniformly bad at very small displacements, but it still does not pass the handoff's strong central-mass validation gate. |
| Step 1 signal | The primary positive signal is now the small-displacement-good subset, where matched-minus-shuffled alignment stabilizes; the all-unit signal is secondary context. |
| Main scientific boundary | The empirical central-mass displacement regime is still not well explained by a strict first-order response prediction, so this remains a smoke test rather than a publishable fixRSVP generalization result. |
| Best current reframe | The current evidence is most consistent with a stable local tangent that remains informative about covariance geometry even as finite empirical displacements sample nonlinear structure. |

## Current Caveats

| Caveat | Why it matters |
| --- | --- |
| Analysis unit can still mix nearby local states | Even with image-phase splitting, these units are not yet a full replay-defined local stimulus state and can still blur linearization and covariance specificity. |
| Backend is still lagged-input common-translation | It now has a raw-row baseline replay sanity check, but not a full replayed image-stack-plus-eye-trace pipeline. |
| No empirical neural bridge yet | All current quantities are model-internal. There is no real-data variance or covariance bridge in this output set. |
| Still a small backend sample | This scaled run is stronger than the earlier smoke test, but it is still too small to support a strong distributional claim. |

## Related Outputs

| Artifact | Path |
| --- | --- |
| Run overview | step01_run_overview.md |
| Results interpretation | step01_results_interpretation.md |
| Backend status | backend_status.json |
| Per-unit backend table | step01_backend_units.csv |
| Step 0 histogram | figures/step0_empirical_fem_hist.png |
| Step 0 diagnostics | figures/step0_linearization_diagnostics.png |
| Step 1 alignment and capture | figures/step1_alignment_and_capture.png |
| Step 1 scaling scatter | figures/step1_scaling_scatter.png |
| Step 1 linearization overlay | figures/step1_linearization_overlay.png |
"""
    path.write_text(results_text)


def _cleanup_legacy_markdown_names(directory: Path, legacy_names: Iterable[str]) -> None:
    for legacy_name in legacy_names:
        legacy_path = directory / legacy_name
        if legacy_path.exists() and legacy_path.is_file():
            legacy_path.unlink()


def _run_interpretation_path(run_dir: Path) -> Path:
    return run_dir / RUN_INTERPRETATION_FILENAME


def _run_overview_path(run_dir: Path) -> Path:
    return run_dir / RUN_OVERVIEW_FILENAME


def _discover_run_dirs(root_dir: Path) -> list[Path]:
    run_dirs = []
    if not root_dir.exists():
        return run_dirs
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        has_backend = (child / "backend_status.json").exists()
        has_interpretation = _run_interpretation_path(child).exists() or (child / LEGACY_RUN_INTERPRETATION_FILENAME).exists()
        if has_backend and has_interpretation:
            run_dirs.append(child)
    return run_dirs


def write_run_index_md(root_dir: Path, latest_run_dir: Path | None = None) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = _discover_run_dirs(root_dir)
    latest = latest_run_dir if latest_run_dir is not None else (run_dirs[-1] if run_dirs else None)
    latest_name = latest.name if latest is not None else "none"

    lines = [
        "# fixRSVP Jacobian Run Index",
        "",
        "Top-level files in this directory are an index, not a separate results run.",
        "Run-specific outputs live in subdirectories.",
        "",
        f"- latest recommended run: {latest_name}",
    ]
    if latest is not None:
        lines.extend(
            [
                f"- latest interpretation: {latest.name}/{RUN_INTERPRETATION_FILENAME}",
                f"- latest overview: {latest.name}/{RUN_OVERVIEW_FILENAME}",
                f"- latest backend status: {latest.name}/backend_status.json",
                "",
                "## Available Runs",
                "",
            ]
        )
    else:
        lines.extend(["", "## Available Runs", "", "- none"]) 

    if run_dirs:
        for run_dir in run_dirs:
            lines.append(f"- {run_dir.name}/{RUN_INTERPRETATION_FILENAME}")

    text = "\n".join(lines) + "\n"
    (root_dir / RUN_INDEX_FILENAME).write_text(text)
    _cleanup_legacy_markdown_names(root_dir, LEGACY_ROOT_INDEX_FILENAMES)


def _normalize_output_dir(output_dir: Path) -> Path:
    if output_dir.resolve() == ROOT_OUTPUT_DIR.resolve():
        return DEFAULT_OUTPUT_DIR
    return output_dir


def _read_backend_units_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "unit_id":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _extract_metric_series(rows: list[dict], prefix: str) -> dict[float, np.ndarray]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if key.startswith(prefix)})
    series = {}
    for key in keys:
        magnitude = float(key.rsplit("_", 1)[-1])
        values = []
        for row in rows:
            value = row.get(key, "")
            if value in ("", None):
                values.append(float("nan"))
            else:
                values.append(float(value))
        series[magnitude] = np.array(values, dtype=np.float64)
    return dict(sorted(series.items()))


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.clf()


def write_diagnostic_figures(
    output_dir: Path,
    *,
    step_deg: np.ndarray,
    backend_units: list[dict],
    pixels_per_degree: float | None,
    pairwise_rows: list[dict] | None = None,
) -> dict[str, str]:
    figure_paths: dict[str, str] = {}
    if not backend_units and not np.isfinite(step_deg).any():
        return figure_paths

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return figure_paths

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    empirical_step = step_deg[np.isfinite(step_deg)]
    if empirical_step.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(empirical_step, bins=40, color="#5177a5", alpha=0.8)
        ax.set_title("Empirical FEM Step Distribution")
        ax.set_xlabel("Frame-to-frame eye-step magnitude (deg)")
        ax.set_ylabel("Count")
        if backend_units and pixels_per_degree and math.isfinite(pixels_per_degree) and pixels_per_degree > 0:
            for magnitude_px in sorted(_extract_metric_series(backend_units, "r2_median_px_").keys()):
                ax.axvline(magnitude_px / pixels_per_degree, color="#c44e52", linestyle="--", linewidth=1)
        rel_path = "figures/step0_empirical_fem_hist.png"
        _save_figure(fig, output_dir / rel_path)
        figure_paths["step0 empirical FEM histogram"] = rel_path
        plt.close(fig)

    if backend_units:
        magnitudes = sorted(_extract_metric_series(backend_units, "r2_median_px_").keys())
        small_r2_keys = _small_displacement_metric_keys(backend_units, "r2_median_px_")
        small_resid_keys = _small_displacement_metric_keys(backend_units, "resid_median_px_")
        if magnitudes:
            r2_series = _extract_metric_series(backend_units, "r2_median_px_")
            resid_series = _extract_metric_series(backend_units, "resid_median_px_")
            cosine_series = _extract_metric_series(backend_units, "cosine_median_px_")
            actual_norm_series = _extract_metric_series(backend_units, "actual_norm_median_px_")
            pred_norm_series = _extract_metric_series(backend_units, "pred_norm_median_px_")

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            x = np.array(magnitudes, dtype=np.float64)
            axes[0, 0].plot(x, [np.nanmedian(r2_series[m]) for m in magnitudes], marker="o", color="#2a9d8f")
            axes[0, 0].axhline(0.5, color="#c44e52", linestyle="--", linewidth=1)
            axes[0, 0].set_title("Step 0 Median $R^2_{lin}$")
            axes[0, 0].set_xlabel("Displacement magnitude (model px)")
            axes[0, 0].set_ylabel("Median $R^2_{lin}$")

            axes[0, 1].plot(x, [np.nanmedian(resid_series[m]) for m in magnitudes], marker="o", color="#e9c46a")
            axes[0, 1].axhline(0.5, color="#c44e52", linestyle="--", linewidth=1)
            axes[0, 1].set_title("Step 0 Relative Residual")
            axes[0, 1].set_xlabel("Displacement magnitude (model px)")
            axes[0, 1].set_ylabel("Median relative residual")

            axes[1, 0].plot(x, [np.nanmedian(cosine_series[m]) for m in magnitudes], marker="o", color="#264653")
            axes[1, 0].set_title("Step 0 Cosine Similarity")
            axes[1, 0].set_xlabel("Displacement magnitude (model px)")
            axes[1, 0].set_ylabel("Median cosine")

            example_magnitude = magnitudes[min(len(magnitudes) - 1, 1)]
            actual_example = actual_norm_series[example_magnitude]
            pred_example = pred_norm_series[example_magnitude]
            axes[1, 1].scatter(actual_example, pred_example, color="#5177a5", alpha=0.8)
            diag_max = np.nanmax(np.concatenate([actual_example, pred_example])) if np.size(actual_example) else 1.0
            axes[1, 1].plot([0.0, diag_max], [0.0, diag_max], linestyle="--", color="#c44e52", linewidth=1)
            axes[1, 1].set_title(f"Actual vs Predicted Change Norms ({example_magnitude:g} px)")
            axes[1, 1].set_xlabel("Median actual change norm")
            axes[1, 1].set_ylabel("Median predicted change norm")

            rel_path = "figures/step0_linearization_diagnostics.png"
            _save_figure(fig, output_dir / rel_path)
            figure_paths["step0 linearization diagnostics"] = rel_path
            plt.close(fig)

        alignment = np.array([row["alignment_A_J"] for row in backend_units], dtype=np.float64)
        alignment_shuffle = np.array([row["image_shuffle_alignment_median"] for row in backend_units], dtype=np.float64)
        alignment_random = np.array([row["random_subspace_alignment_median"] for row in backend_units], dtype=np.float64)
        capture = np.array([row["capture_V_J"] for row in backend_units], dtype=np.float64)
        capture_shuffle = np.array([row["image_shuffle_capture_median"] for row in backend_units], dtype=np.float64)
        capture_random = np.array([row["random_subspace_capture_median"] for row in backend_units], dtype=np.float64)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].boxplot([alignment, alignment_shuffle, alignment_random], tick_labels=["matched", "img-shuf", "random"])
        axes[0].set_title("Step 1 Alignment Distributions")
        axes[0].set_ylabel("A_J")
        axes[1].boxplot([capture, capture_shuffle, capture_random], tick_labels=["matched", "img-shuf", "random"])
        axes[1].set_title("Step 1 Capture Distributions")
        axes[1].set_ylabel("V_J")
        rel_path = "figures/step1_alignment_and_capture.png"
        _save_figure(fig, output_dir / rel_path)
        figure_paths["step1 alignment and capture distributions"] = rel_path
        plt.close(fig)

        trace_cov = np.array([row["trace_cov_model_fem"] for row in backend_units], dtype=np.float64)
        predicted_drive = np.array([row["predicted_drive_trace"] for row in backend_units], dtype=np.float64)
        jacobian_norm = np.array([row["jacobian_fro_norm"] for row in backend_units], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(predicted_drive, trace_cov, color="#2a9d8f", alpha=0.8)
        axes[0].set_title(f"Predicted Drive vs FEM Covariance\nr={_safe_corrcoef(predicted_drive, trace_cov):.3f}")
        axes[0].set_xlabel("Predicted drive trace")
        axes[0].set_ylabel("Observed FEM covariance trace")
        axes[1].scatter(jacobian_norm, trace_cov, color="#264653", alpha=0.8)
        axes[1].set_title(f"Jacobian Norm vs FEM Covariance\nr={_safe_corrcoef(jacobian_norm, trace_cov):.3f}")
        axes[1].set_xlabel("Jacobian Frobenius norm")
        axes[1].set_ylabel("Observed FEM covariance trace")
        rel_path = "figures/step1_scaling_scatter.png"
        _save_figure(fig, output_dir / rel_path)
        figure_paths["step1 covariance scaling scatter"] = rel_path
        plt.close(fig)

        step0_r2 = np.array([row["step0_r2_median"] for row in backend_units], dtype=np.float64)
        step0_resid = np.array([row["step0_resid_median"] for row in backend_units], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(step0_r2, alignment, color="#5177a5", alpha=0.8)
        axes[0].axvline(0.5, color="#c44e52", linestyle="--", linewidth=1)
        axes[0].set_title("Alignment vs Step 0 $R^2_{lin}$")
        axes[0].set_xlabel("Step 0 median $R^2_{lin}$")
        axes[0].set_ylabel("A_J")
        axes[1].scatter(step0_resid, capture, color="#e76f51", alpha=0.8)
        axes[1].axvline(0.5, color="#c44e52", linestyle="--", linewidth=1)
        axes[1].set_title("Capture vs Step 0 Residual")
        axes[1].set_xlabel("Step 0 median relative residual")
        axes[1].set_ylabel("V_J")
        rel_path = "figures/step1_linearization_overlay.png"
        _save_figure(fig, output_dir / rel_path)
        figure_paths["step1 linearization overlay"] = rel_path
        plt.close(fig)

        if small_r2_keys and small_resid_keys:
            small_step0_r2 = np.array([_median_over_keys(row, small_r2_keys) for row in backend_units], dtype=np.float64)
            small_step0_resid = np.array([_median_over_keys(row, small_resid_keys) for row in backend_units], dtype=np.float64)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            axes[0].scatter(small_step0_r2, alignment, color="#2a9d8f", alpha=0.8)
            axes[0].axvline(0.5, color="#c44e52", linestyle="--", linewidth=1)
            axes[0].set_title("Alignment vs Small-Disp Step 0 $R^2_{lin}$")
            axes[0].set_xlabel("Small-displacement median $R^2_{lin}$")
            axes[0].set_ylabel("A_J")
            axes[1].scatter(small_step0_resid, capture, color="#264653", alpha=0.8)
            axes[1].axvline(0.5, color="#c44e52", linestyle="--", linewidth=1)
            axes[1].set_title("Capture vs Small-Disp Residual")
            axes[1].set_xlabel("Small-displacement median relative residual")
            axes[1].set_ylabel("V_J")
            rel_path = "figures/step1_small_displacement_overlay.png"
            _save_figure(fig, output_dir / rel_path)
            figure_paths["step1 small-displacement overlay"] = rel_path
            plt.close(fig)

        paired = _paired_delta_summaries(backend_units)
        all_align_delta = np.array(
            [row["alignment_A_J"] - row["image_shuffle_alignment_median"] for row in backend_units],
            dtype=np.float64,
        )
        all_capture_delta = np.array(
            [row["capture_V_J"] - row["image_shuffle_capture_median"] for row in backend_units],
            dtype=np.float64,
        )
        small_good_mask = np.array(
            [_classify_small_displacement_regime(row) == "good" for row in backend_units],
            dtype=bool,
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
        axes[1].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
        axes[0].scatter(np.zeros_like(all_align_delta), all_align_delta, color="#8d99ae", alpha=0.6, label="all units")
        axes[1].scatter(np.zeros_like(all_capture_delta), all_capture_delta, color="#8d99ae", alpha=0.6, label="all units")
        if np.any(small_good_mask):
            axes[0].scatter(
                np.ones(int(np.sum(small_good_mask))),
                all_align_delta[small_good_mask],
                color="#2a9d8f",
                alpha=0.85,
                label="small-disp good",
            )
            axes[1].scatter(
                np.ones(int(np.sum(small_good_mask))),
                all_capture_delta[small_good_mask],
                color="#2a9d8f",
                alpha=0.85,
                label="small-disp good",
            )
        axes[0].set_xticks([0, 1], labels=["all", "small-good"])
        axes[1].set_xticks([0, 1], labels=["all", "small-good"])
        axes[0].set_title(
            "Alignment Paired Delta\nall={all_delta:.3f}, small-good={small_delta:.3f}".format(
                all_delta=paired["alignment_all"]["median_delta"],
                small_delta=paired["alignment_small_good"]["median_delta"],
            )
        )
        axes[1].set_title(
            "Capture Paired Delta\nall={all_delta:.3f}, small-good={small_delta:.3f}".format(
                all_delta=paired["capture_all"]["median_delta"],
                small_delta=paired["capture_small_good"]["median_delta"],
            )
        )
        axes[0].set_ylabel("Matched minus image-shuffled $A_J$")
        axes[1].set_ylabel("Matched minus image-shuffled $V_J$")
        axes[0].legend(frameon=False, loc="best")
        axes[1].legend(frameon=False, loc="best")
        rel_path = "figures/step1_paired_delta_split.png"
        _save_figure(fig, output_dir / rel_path)
        figure_paths["step1 paired delta split"] = rel_path
        plt.close(fig)

        eye_radius_median = np.array(
            [row.get("centered_eye_radius_px_median", np.nan) for row in backend_units],
            dtype=np.float64,
        )
        if np.any(np.isfinite(eye_radius_median)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            axes[0].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
            axes[0].scatter(eye_radius_median, all_align_delta, color="#5177a5", alpha=0.8)
            axes[0].set_xlabel("Median baseline-relative eye radius (model px)")
            axes[0].set_ylabel("Matched minus img-shuffled $A_J$")
            axes[0].set_title("Alignment delta vs displacement radius")
            axes[1].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
            axes[1].scatter(eye_radius_median, all_capture_delta, color="#e76f51", alpha=0.8)
            axes[1].set_xlabel("Median baseline-relative eye radius (model px)")
            axes[1].set_ylabel("Matched minus img-shuffled $V_J$")
            axes[1].set_title("Capture delta vs displacement radius")
            rel_path = "figures/step1_delta_vs_radius.png"
            _save_figure(fig, output_dir / rel_path)
            figure_paths["step1 alignment delta vs displacement radius"] = rel_path
            plt.close(fig)

    if pairwise_rows:
        try:
            bin_mids = sorted({
                float(r["bin_mid_px"]) for r in pairwise_rows
                if np.isfinite(r.get("bin_mid_px", float("nan")))
            })
            if bin_mids:
                def _collect(key: str) -> np.ndarray:
                    by_bin: dict[float, list] = {m: [] for m in bin_mids}
                    for pr in pairwise_rows:
                        m = float(pr.get("bin_mid_px", float("nan")))
                        if np.isfinite(m) and np.isfinite(pr.get(key, float("nan"))):
                            by_bin[m].append(float(pr[key]))
                    return np.array([np.nanmedian(by_bin[m]) if by_bin[m] else np.nan for m in bin_mids])

                x = np.array(bin_mids)
                r2_base = _collect("r2_lin_median")
                cos_base = _collect("cosine_median")
                delta_base = _collect("capture_V_J_delta")
                r2_local = _collect("r2_lin_local_median")
                cos_local = _collect("cosine_local_median")
                delta_local = _collect("capture_V_J_local_delta")
                npairs_med = np.array([
                    int(np.median([int(pr.get("n_pairs", 0)) for pr in pairwise_rows
                                   if float(pr.get("bin_mid_px", float("nan"))) == m]))
                    for m in bin_mids
                ])

                has_local_data = np.any(np.isfinite(r2_local))

                fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
                bw = (x[1] - x[0]) * 0.4 if len(x) > 1 else 0.8

                # Row 0: baseline Jacobian
                axes[0, 0].plot(x, r2_base, marker="o", color="#2a9d8f", label="baseline J")
                axes[0, 0].axhline(0.5, color="#c44e52", linestyle="--", linewidth=1, label="R²=0.5")
                axes[0, 0].axhline(0.0, color="#888888", linestyle="-", linewidth=0.8)
                axes[0, 0].set_ylabel("Median $R^2_{lin}$")
                axes[0, 0].set_title("Linearization — baseline J")
                axes[0, 0].legend(fontsize=8, frameon=False)

                axes[0, 1].plot(x, cos_base, marker="o", color="#5177a5", label="baseline J")
                axes[0, 1].axhline(0.0, color="#888888", linestyle="-", linewidth=0.8)
                axes[0, 1].set_ylabel("Median cosine")
                axes[0, 1].set_title("Cosine similarity — baseline J")

                axes[0, 2].bar(x - bw / 2, delta_base,
                               color=["#2a9d8f" if d > 0 else "#c44e52" for d in np.nan_to_num(delta_base)],
                               alpha=0.75, width=bw, label="baseline J")
                axes[0, 2].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
                for xi, np_ in zip(x, npairs_med):
                    axes[0, 2].text(xi, 0, f"n={np_}", ha="center", va="bottom", fontsize=6, rotation=90)
                axes[0, 2].set_ylabel("Matched − shuffled $V_J$")
                axes[0, 2].set_title("V_J delta — baseline J")

                # Row 1: local (per-sample midpoint) Jacobian
                lc = "#e76f51" if has_local_data else "#aaaaaa"
                axes[1, 0].plot(x, r2_local, marker="s", color=lc,
                                label="local J" if has_local_data else "local J (no data)")
                axes[1, 0].axhline(0.5, color="#c44e52", linestyle="--", linewidth=1, label="R²=0.5")
                axes[1, 0].axhline(0.0, color="#888888", linestyle="-", linewidth=0.8)
                axes[1, 0].set_xlabel("Pairwise eye distance (px)")
                axes[1, 0].set_ylabel("Median $R^2_{lin}$")
                axes[1, 0].set_title("Linearization — local midpoint J")
                axes[1, 0].legend(fontsize=8, frameon=False)
                axes[1, 0].text(
                    0.98, 0.97, "local J at (p_i+p_j)/2",
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=7, style="italic",
                    color="#555555",
                )

                axes[1, 1].plot(x, cos_local, marker="s", color=lc, label="local J")
                axes[1, 1].axhline(0.0, color="#888888", linestyle="-", linewidth=0.8)
                axes[1, 1].set_xlabel("Pairwise eye distance (px)")
                axes[1, 1].set_ylabel("Median cosine")
                axes[1, 1].set_title("Cosine similarity — local midpoint J")

                axes[1, 2].bar(x + bw / 2, delta_local,
                               color=["#e76f51" if d > 0 else "#aaaaaa" for d in np.nan_to_num(delta_local)],
                               alpha=0.75, width=bw, label="local J")
                axes[1, 2].axhline(0.0, color="#c44e52", linestyle="--", linewidth=1)
                axes[1, 2].set_xlabel("Pairwise eye distance (px)")
                axes[1, 2].set_ylabel("Matched − shuffled $V_J$")
                axes[1, 2].set_title("V_J delta — local midpoint J")

                fig.suptitle(
                    "Pairwise analysis: baseline J (row 0) vs local midpoint J (row 1)\n"
                    "Distance-dependent signal expected if local tangent story holds",
                    fontsize=10,
                )
                fig.tight_layout(h_pad=0.3)

                rel_path = "figures/pairwise_bin_diagnostics.png"
                _save_figure(fig, output_dir / rel_path)
                figure_paths["pairwise bin diagnostics"] = rel_path
                plt.close(fig)
        except Exception:
            pass

    return figure_paths


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _build_backend_config(args: argparse.Namespace) -> ModelBackendConfig | None:
    if args.model_type is None and args.checkpoint_path is None:
        return None
    if args.dataset_idx is None:
        raise ValueError("--dataset-idx is required when enabling the model backend")
    jacobian_step_sizes_px = _parse_float_tuple(args.jacobian_step_sizes_px)
    if args.jacobian_step_px not in jacobian_step_sizes_px:
        jacobian_step_sizes_px = tuple(sorted({*jacobian_step_sizes_px, float(args.jacobian_step_px)}))
    return ModelBackendConfig(
        model_type=args.model_type,
        model_index=args.model_index,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        dataset_idx=args.dataset_idx,
        model_device=args.model_device,
        unit_mode=args.unit_mode,
        n_fixation_phase_bins=args.n_fixation_phase_bins,
        n_radius_bins=args.n_radius_bins,
        max_units=args.max_units,
        max_samples_per_unit=args.max_samples_per_unit,
        min_backend_samples=args.min_backend_samples,
        n_image_shuffle_matches=args.n_image_shuffle_matches,
        jacobian_step_px=args.jacobian_step_px,
        jacobian_step_sizes_px=jacobian_step_sizes_px,
        displacement_magnitudes_px=_parse_float_tuple(args.displacement_magnitudes_px),
        empirical_displacement_percentiles=_parse_float_tuple(args.empirical_displacement_percentiles),
        local_state_keep_fraction=args.local_state_keep_fraction,
        max_baseline_relative_radius_px=args.max_baseline_relative_radius_px,
        pairwise_bin_edges_px=_parse_float_tuple(args.pairwise_bin_edges_px),
        pixels_per_degree=args.pixels_per_degree,
        n_random_null_reps=args.n_random_null_reps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold Step 0 and Step 1 fixRSVP Jacobian analyses."
    )
    parser.add_argument("--subject", required=True, help="Session subject name.")
    parser.add_argument("--date", required=True, help="Session date in YYYY-MM-DD format.")
    parser.add_argument(
        "--dataset-configs-path",
        required=True,
        help="Dataset config YAML used for fixRSVP collation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for manifests and summaries.",
    )
    parser.add_argument(
        "--min-samples-per-unit",
        type=int,
        default=100,
        help="Minimum valid bins required for a retained analysis unit.",
    )
    parser.add_argument(
        "--unit-mode",
        choices=("image", "image_phase", "image_phase_radius"),
        default="image",
        help="Primary analysis unit definition: pooled image identity only, image plus relative fixation phase, or image plus fixation phase plus centered-radius bin.",
    )
    parser.add_argument(
        "--n-fixation-phase-bins",
        type=int,
        default=4,
        help="Number of relative fixation-phase bins used when --unit-mode=image_phase.",
    )
    parser.add_argument(
        "--n-radius-bins",
        type=int,
        default=2,
        help="Number of centered-eye-radius quantile bins used when --unit-mode=image_phase_radius.",
    )
    parser.add_argument(
        "--use-cached-data",
        action="store_true",
        help="Use cached fixRSVP collation when available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print fixRSVP preprocessing diagnostics.",
    )
    parser.add_argument("--model-type", help="Model type for load_model().")
    parser.add_argument("--model-index", type=int, help="Model index for load_model().")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory for model discovery.")
    parser.add_argument("--checkpoint-path", help="Explicit checkpoint path.")
    parser.add_argument("--dataset-idx", type=int, help="Dataset index inside the loaded model.")
    parser.add_argument(
        "--model-device",
        default="cpu",
        help="Device used to load and evaluate the model backend.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=8,
        help="Maximum retained analysis units to evaluate with the model backend.",
    )
    parser.add_argument(
        "--max-samples-per-unit",
        type=int,
        default=128,
        help="Maximum stimulus snippets evaluated per retained analysis unit.",
    )
    parser.add_argument(
        "--min-backend-samples",
        type=int,
        default=32,
        help="Minimum usable post-filter samples required for a backend unit after finite-value filtering and subsampling.",
    )
    parser.add_argument(
        "--n-image-shuffle-matches",
        type=int,
        default=8,
        help="Number of nearest matched units used for the image-shuffled null.",
    )
    parser.add_argument(
        "--jacobian-step-px",
        type=float,
        default=0.25,
        help="Primary finite-difference step size in model input pixels.",
    )
    parser.add_argument(
        "--jacobian-step-sizes-px",
        default="0.125,0.25,0.5",
        help="Comma-separated finite-difference step sizes used for Jacobian stability checks.",
    )
    parser.add_argument(
        "--displacement-magnitudes-px",
        default="0.0625,0.125,0.25,0.5",
        help="Comma-separated displacement magnitudes in model pixels for Step 0 checks.",
    )
    parser.add_argument(
        "--empirical-displacement-percentiles",
        default="50,75,90,95",
        help="Comma-separated percentiles of the empirical eye-step distribution to convert into Step 0 test magnitudes in model pixels.",
    )
    parser.add_argument(
        "--pixels-per-degree",
        type=float,
        help="Override pixels-per-degree conversion used to map eye displacements from degrees to model pixels.",
    )
    parser.add_argument(
        "--n-random-null-reps",
        type=int,
        default=256,
        help="Number of random 2D subspace null samples for each unit.",
    )
    parser.add_argument(
        "--local-state-keep-fraction",
        type=float,
        default=1.0,
        help="Fraction of sampled snippets to keep after choosing the representative baseline, ranked by lagged-stimulus proximity to that baseline. Values below 1.0 tighten the local-state neighborhood while preserving at least the backend sample floor.",
    )
    parser.add_argument(
        "--max-baseline-relative-radius-px",
        type=float,
        default=None,
        help="If set, discard all samples whose absolute baseline-relative eye-position offset exceeds this radius (model pixels) after local-state filtering. This restricts the covariance cloud to a tight absolute neighborhood of the baseline eye position.",
    )
    parser.add_argument(
        "--pairwise-bin-edges-px",
        default="0,1,2,3,5,8,12",
        help="Comma-separated bin edges (model pixels) for pairwise eye-distance binning. Pairs with distance in [edges[k], edges[k+1]) form one bin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _normalize_output_dir(Path(args.output_dir))
    root_output_dir = ROOT_OUTPUT_DIR

    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        use_cached_data=args.use_cached_data,
        verbose=args.verbose,
    )

    radius_deg = compute_centered_radius_deg(data["eyepos"])
    step_deg = compute_step_displacement_deg(data["eyepos"])
    radius_summary = summarize_percentiles(radius_deg)
    step_summary = summarize_percentiles(step_deg)
    units = build_image_units(
        image_ids=data["image_ids"],
        radius_deg=radius_deg,
        min_samples=args.min_samples_per_unit,
        unit_mode=args.unit_mode,
        n_fixation_phase_bins=args.n_fixation_phase_bins,
        n_radius_bins=args.n_radius_bins,
    )

    backend = Step01Backend(config=_build_backend_config(args))
    backend_result = backend.run(
        units=units,
        data=data,
        dataset_configs_path=args.dataset_configs_path,
        output_dir=output_dir,
    )
    backend_units = _read_backend_units_csv(output_dir / "step01_backend_units.csv")
    pairwise_rows_from_csv = _read_backend_units_csv(output_dir / "step01_pairwise_bins.csv")
    figure_paths = write_diagnostic_figures(
        output_dir,
        step_deg=step_deg,
        backend_units=backend_units,
        pixels_per_degree=backend_result.get("pixels_per_degree"),
        pairwise_rows=pairwise_rows_from_csv,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "step01_manifest.npz",
        centered_radius_deg=radius_deg,
        step_displacement_deg=step_deg,
        image_ids=data["image_ids"],
        fix_dur=data["fix_dur"],
        radius_p50=radius_summary["p50"],
        radius_p75=radius_summary["p75"],
        radius_p90=radius_summary["p90"],
        radius_p95=radius_summary["p95"],
        step_p50=step_summary["p50"],
        step_p75=step_summary["p75"],
        step_p90=step_summary["p90"],
        step_p95=step_summary["p95"],
        retained_unit_ids=np.array([unit.unit_id for unit in units], dtype=object),
        retained_image_ids=np.array([unit.image_id for unit in units], dtype=np.int32),
        retained_phase_bins=np.array([-1 if unit.phase_bin is None else unit.phase_bin for unit in units], dtype=np.int32),
        retained_radius_bins=np.array([-1 if unit.radius_bin is None else unit.radius_bin for unit in units], dtype=np.int32),
        retained_unit_samples=np.array([unit.n_samples for unit in units], dtype=np.int32),
    )
    write_units_csv(output_dir / "step01_units.csv", units)
    write_summary_md(
        _run_overview_path(output_dir),
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        radius_summary=radius_summary,
        step_summary=step_summary,
        n_trials=int(data["robs"].shape[0]),
        n_units=len(units),
        min_samples=args.min_samples_per_unit,
        unit_mode=args.unit_mode,
        backend_name=backend.describe(),
        backend_result=backend_result,
        backend_units=backend_units,
        figure_paths=figure_paths,
    )
    write_results_so_far_md(
        _run_interpretation_path(output_dir),
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        radius_summary=radius_summary,
        step_summary=step_summary,
        n_trials=int(data["robs"].shape[0]),
        n_units=len(units),
        unit_mode=args.unit_mode,
        backend_name=backend.describe(),
        backend_result=backend_result,
        backend_units=backend_units,
    )
    _cleanup_legacy_markdown_names(
        output_dir,
        (LEGACY_RUN_OVERVIEW_FILENAME, LEGACY_RUN_INTERPRETATION_FILENAME),
    )
    backend_result = {
        **backend_result,
        "paired_delta_summary": _paired_delta_summaries(backend_units),
    }
    (output_dir / "backend_status.json").write_text(json.dumps(backend_result, indent=2))

    write_run_index_md(root_output_dir, latest_run_dir=output_dir)

    print(f"Saved Step 0/1 scaffold outputs to {output_dir}")
    print(f"Retained {len(units)} {args.unit_mode} units with >= {args.min_samples_per_unit} samples")


if __name__ == "__main__":
    main()
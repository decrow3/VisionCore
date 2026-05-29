#!/usr/bin/env python3
"""Figure 4 geometry-bridge audit for fixRSVP sessions.

Initial implementation scope:
- Audit 1 sign/axis confirmation sweep
- Canonical y-flip verification diagnostic
- Model-side centering variants for B_model, FEM_PCs, and J_local
- Ceiling-normalized reporting against empirical split-half bundles

This script is intentionally non-destructive and does not overwrite the main
jacobian_predictive_framework outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.fixrsvp import get_fixrsvp_data
from scripts.fixrsvp_eye_conventions import (
    DEFAULT_STORED_EYE_CONVENTION,
    stored_eyepos_to_eye_norm,
    visual_eye_deg_to_canonical_shift_px,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_model_empirical_alignment import (
    _basis_summary,
    _matched_candidate_ids_local,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_step2 import (
    DEFAULT_JACOBIAN_STEP_PX,
    DEFAULT_N_SPLIT_REPEATS,
    REFERENCE_RATE_HZ,
    _choose_baseline,
    _collect_image_windows,
    _predict_responses,
    _resolve_pixels_per_degree,
    _shift_stimulus_batch,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_steps01 import (
    _paired_delta_summary,
    _reconstruct_replay_baseline_stim,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_translation_chart import (
    _build_global_basis,
    _load_model_for_session,
    compute_grid_responses,
)
from scripts.mcfarland_sim import shift_movie_with_eye


DEFAULT_OUTPUT_DIR = Path("results/figure4_geometry_bridge_audit")
DEFAULT_TRANSFORM_MODES = (
    "xy",
    "negx_y",
    "x_negy",
    "negx_negy",
    "yx",
    "negy_x",
    "y_negx",
    "negy_negx",
)
DEFAULT_CENTERING_MODES = ("current", "mean", "baseline")
DEFAULT_PREDICTOR_MODES = ("emp_xy", "emp_x_negy")
DEFAULT_BASIS_NAMES = ("B_model", "FEM_PCs", "J_local")
DEFAULT_WINDOW_LEVELS = ("image_id", "prev_next")
DEFAULT_HISTORY_HASH_MIN_SAMPLES = 20


class AuditWindow:
    __slots__ = (
        "image_id",
        "trial_indices",
        "time_indices",
        "stim",
        "eyepos_deg",
        "robs_rates",
        "window_level",
        "window_key",
        "history_key",
        "time_bin",
    )

    def __init__(
        self,
        image_id: int,
        trial_indices: np.ndarray,
        time_indices: np.ndarray,
        stim: torch.Tensor,
        eyepos_deg: np.ndarray,
        robs_rates: np.ndarray,
        window_level: str,
        window_key: str,
        history_key: str = "",
        time_bin: int | None = None,
    ) -> None:
        self.image_id = image_id
        self.trial_indices = trial_indices
        self.time_indices = time_indices
        self.stim = stim
        self.eyepos_deg = eyepos_deg
        self.robs_rates = robs_rates
        self.window_level = window_level
        self.window_key = window_key
        self.history_key = history_key
        self.time_bin = time_bin


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def _nanmedian(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmedian(np.asarray(values, dtype=np.float64)))


def _safe_ratio(numer: float, denom: float, min_abs_denom: float = 0.05) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < min_abs_denom:
        return float("nan")
    return float(numer / denom)


def _stable_int_hash(value: str, modulo: int = 2**31) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def _stable_array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8).tobytes()).hexdigest()[:16]


def _collect_audit_windows(
    data: dict[str, Any],
    min_samples: int,
    dt: float,
    window_levels: list[str],
) -> list[AuditWindow]:
    base_windows = _collect_image_windows(data, min_samples=min_samples, dt=dt)
    output_windows: list[AuditWindow] = []
    image_ids_raw = np.asarray(data["image_ids"])

    if "image_id" in window_levels:
        for win in base_windows:
            output_windows.append(AuditWindow(
                image_id=int(win.image_id),
                trial_indices=win.trial_indices,
                time_indices=win.time_indices,
                stim=win.stim,
                eyepos_deg=win.eyepos_deg,
                robs_rates=win.robs_rates,
                window_level="image_id",
                window_key=f"image_{win.image_id}",
            ))

    if "history_hash" in window_levels:
        for win in base_windows:
            history_buckets: dict[str, list[int]] = {}
            stim_np = win.stim.detach().cpu().numpy()
            for idx in range(stim_np.shape[0]):
                history_key = _stable_array_hash(stim_np[idx])
                history_buckets.setdefault(history_key, []).append(idx)
            for history_key, indices in history_buckets.items():
                if len(indices) < DEFAULT_HISTORY_HASH_MIN_SAMPLES:
                    continue
                sel = np.asarray(indices, dtype=np.int64)
                output_windows.append(AuditWindow(
                    image_id=int(win.image_id),
                    trial_indices=win.trial_indices[sel],
                    time_indices=win.time_indices[sel],
                    stim=win.stim[sel].clone(),
                    eyepos_deg=win.eyepos_deg[sel].copy(),
                    robs_rates=win.robs_rates[sel].copy(),
                    window_level="history_hash",
                    window_key=f"image_{win.image_id}|history_{history_key}",
                    history_key=history_key,
                ))

    if "prev_next" in window_levels:
        for win in base_windows:
            context_buckets: dict[tuple[int, int], list[int]] = {}
            for idx, (trial_idx, time_idx) in enumerate(zip(win.trial_indices, win.time_indices)):
                prev_img = int(image_ids_raw[trial_idx, time_idx - 1]) if time_idx - 1 >= 0 and image_ids_raw[trial_idx, time_idx - 1] >= 0 else -1
                next_img = int(image_ids_raw[trial_idx, time_idx + 1]) if time_idx + 1 < image_ids_raw.shape[1] and image_ids_raw[trial_idx, time_idx + 1] >= 0 else -1
                context_buckets.setdefault((prev_img, next_img), []).append(idx)
            for (prev_img, next_img), indices in context_buckets.items():
                if len(indices) < DEFAULT_HISTORY_HASH_MIN_SAMPLES:
                    continue
                sel = np.asarray(indices, dtype=np.int64)
                context_key = f"prev_{prev_img}_next_{next_img}"
                output_windows.append(AuditWindow(
                    image_id=int(win.image_id),
                    trial_indices=win.trial_indices[sel],
                    time_indices=win.time_indices[sel],
                    stim=win.stim[sel].clone(),
                    eyepos_deg=win.eyepos_deg[sel].copy(),
                    robs_rates=win.robs_rates[sel].copy(),
                    window_level="prev_next",
                    window_key=f"image_{win.image_id}|{context_key}",
                    history_key=context_key,
                ))

    return output_windows


def _transform_eye_to_retinal_offsets(eye_offsets_px: np.ndarray, mode: str) -> np.ndarray:
    x = eye_offsets_px[:, 0]
    y = eye_offsets_px[:, 1]

    if mode == "xy":
        out = np.column_stack([x, y])
    elif mode == "negx_y":
        out = np.column_stack([-x, y])
    elif mode == "x_negy":
        out = np.column_stack([x, -y])
    elif mode == "negx_negy":
        out = np.column_stack([-x, -y])
    elif mode == "yx":
        out = np.column_stack([y, x])
    elif mode == "negy_x":
        out = np.column_stack([-y, x])
    elif mode == "y_negx":
        out = np.column_stack([y, -x])
    elif mode == "negy_negx":
        out = np.column_stack([-y, -x])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")
    return np.asarray(out, dtype=np.float64)


def _transform_empirical_predictor(eye_offsets_px: np.ndarray, predictor_mode: str) -> np.ndarray:
    x = eye_offsets_px[:, 0]
    y = eye_offsets_px[:, 1]

    if predictor_mode == "emp_xy":
        out = np.column_stack([x, y])
    elif predictor_mode == "emp_x_negy":
        out = np.column_stack([x, -y])
    else:
        raise ValueError(f"Unknown predictor mode: {predictor_mode}")
    return np.asarray(out, dtype=np.float64)


def _fit_slopes_with_intercept(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones(x.shape[0], dtype=np.float64)])
    if int(np.linalg.matrix_rank(design)) < 2:
        if y.ndim == 1:
            return np.zeros(2, dtype=np.float64)
        return np.zeros((2, y.shape[1]), dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coeffs[:2], dtype=np.float64)


def _fit_empirical_B_and_count(
    robs: np.ndarray,
    eye_predictor_px: np.ndarray,
    min_samples: int,
) -> tuple[np.ndarray, int]:
    n_samples, n_neurons = robs.shape
    if n_samples < min_samples:
        return np.zeros((n_neurons, 2), dtype=np.float64), 0
    B = np.zeros((n_neurons, 2), dtype=np.float64)
    n_valid = 0
    for neuron_idx in range(n_neurons):
        response = robs[:, neuron_idx]
        ok = np.isfinite(response)
        if int(ok.sum()) < min_samples:
            continue
        x = eye_predictor_px[ok]
        y = response[ok]
        response_centered = y - y.mean()
        slopes = _fit_slopes_with_intercept(x, response_centered)
        if not np.isfinite(slopes).all():
            continue
        B[neuron_idx] = slopes
        n_valid += 1
    return B, n_valid


def _fit_empirical_B(robs: np.ndarray, eye_predictor_px: np.ndarray, min_samples: int) -> np.ndarray:
    B, _n_valid = _fit_empirical_B_and_count(robs, eye_predictor_px, min_samples)
    return B


def _generate_empirical_split_bundle(
    robs: np.ndarray,
    eye_predictor_px: np.ndarray,
    trial_indices: np.ndarray,
    min_samples: int,
    n_split_repeats: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    n_samples = robs.shape[0]
    if n_samples < 2 * min_samples:
        return None
    unique_trials = np.unique(trial_indices)
    rng = np.random.default_rng(seed)
    split_a: list[np.ndarray] = []
    split_b: list[np.ndarray] = []

    for _ in range(n_split_repeats):
        if len(unique_trials) >= 2:
            perm_trials = rng.permutation(unique_trials)
            half = len(perm_trials) // 2
            if half == 0 or half == len(perm_trials):
                continue
            trials_a = perm_trials[:half]
            trials_b = perm_trials[half:]
            idx_a = np.where(np.isin(trial_indices, trials_a))[0]
            idx_b = np.where(np.isin(trial_indices, trials_b))[0]
        else:
            perm = rng.permutation(n_samples)
            idx_a, idx_b = perm[: n_samples // 2], perm[n_samples // 2 :]

        if len(idx_a) < min_samples or len(idx_b) < min_samples:
            continue

        B_a, n_valid_a = _fit_empirical_B_and_count(robs[idx_a], eye_predictor_px[idx_a], min_samples)
        B_b, n_valid_b = _fit_empirical_B_and_count(robs[idx_b], eye_predictor_px[idx_b], min_samples)
        if n_valid_a == 0 or n_valid_b == 0:
            continue
        split_a.append(B_a)
        split_b.append(B_b)

    if not split_a:
        return None
    return {
        "B_a": np.asarray(split_a, dtype=np.float64),
        "B_b": np.asarray(split_b, dtype=np.float64),
    }


def _empirical_bundle_reference_B(emp_bundle: dict[str, np.ndarray]) -> np.ndarray:
    B_a = np.asarray(emp_bundle["B_a"], dtype=np.float64)
    B_b = np.asarray(emp_bundle["B_b"], dtype=np.float64)
    combined = np.concatenate([B_a, B_b], axis=0)
    return np.nanmedian(combined, axis=0)


def _window_has_valid_empirical_bundle(
    win,
    centering_modes: list[str],
    predictor_modes: list[str],
    pixels_per_degree: float,
    min_empirical_samples: int,
    n_split_repeats: int,
    session_name: str,
) -> bool:
    for centering_mode in centering_modes:
        for predictor_mode in predictor_modes:
            state = _prepare_window_state(win, centering_mode, pixels_per_degree, predictor_mode)
            bundle = _generate_empirical_split_bundle(
                robs=win.robs_rates,
                eye_predictor_px=state["eye_predictor_px"],
                trial_indices=win.trial_indices,
                min_samples=min_empirical_samples,
                n_split_repeats=n_split_repeats,
                seed=_stable_int_hash(
                    f"{session_name}|image_{win.image_id}|{centering_mode}|{predictor_mode}|prefilter"
                ),
            )
            if bundle is not None:
                return True
    return False


def _vector_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _prepare_window_state(
    win,
    centering_mode: str,
    pixels_per_degree: float,
    predictor_mode: str,
) -> dict[str, Any]:
    if centering_mode == "current":
        baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        baseline_idx = int(np.nanargmin(np.linalg.norm(win.eyepos_deg - baseline_eye_deg[None, :], axis=1)))
        shift_eye_deg = win.eyepos_deg - baseline_eye_deg[None, :]
        raw_eye_offsets_px = shift_eye_deg * pixels_per_degree
        raw_eye_offsets_px = raw_eye_offsets_px - raw_eye_offsets_px.mean(axis=0, keepdims=True)
    elif centering_mode == "mean":
        center_eye_deg = np.nanmean(win.eyepos_deg, axis=0)
        baseline_idx = int(np.nanargmin(np.linalg.norm(win.eyepos_deg - center_eye_deg[None, :], axis=1)))
        baseline_stim = win.stim[baseline_idx: baseline_idx + 1].clone()
        baseline_eye_deg = win.eyepos_deg[baseline_idx].copy()
        shift_eye_deg = win.eyepos_deg - center_eye_deg[None, :]
        raw_eye_offsets_px = shift_eye_deg * pixels_per_degree
        raw_eye_offsets_px = raw_eye_offsets_px - raw_eye_offsets_px.mean(axis=0, keepdims=True)
    elif centering_mode == "baseline":
        baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        baseline_idx = int(np.nanargmin(np.linalg.norm(win.eyepos_deg - baseline_eye_deg[None, :], axis=1)))
        shift_eye_deg = win.eyepos_deg - baseline_eye_deg[None, :]
        raw_eye_offsets_px = shift_eye_deg * pixels_per_degree
    else:
        raise ValueError(f"Unknown centering mode: {centering_mode}")

    eye_predictor_px = _transform_empirical_predictor(raw_eye_offsets_px, predictor_mode)

    return {
        "baseline_stim": baseline_stim,
        "baseline_eye_deg": np.asarray(baseline_eye_deg, dtype=np.float64),
        "baseline_sample_idx": baseline_idx,
        "pixels_per_degree": float(pixels_per_degree),
        "shift_eye_px": np.asarray(raw_eye_offsets_px, dtype=np.float64),
        "eye_predictor_px": np.asarray(eye_predictor_px, dtype=np.float64),
        "predictor_mode": predictor_mode,
        "mean_eye_x": float(np.nanmean(win.eyepos_deg[:, 0])),
        "mean_eye_y": float(np.nanmean(win.eyepos_deg[:, 1])),
    }


def _stimulus_frame_for_display(stim: torch.Tensor) -> np.ndarray:
    arr = stim.detach().cpu().numpy().astype(np.float64)
    frame = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[-1]
    return np.asarray(frame, dtype=np.float64)


def _tensor_comparison_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().cpu().numpy().astype(np.float64)
    cand = candidate.detach().cpu().numpy().astype(np.float64)
    diff = cand - ref
    return {
        "mae": float(np.nanmean(np.abs(diff))),
        "max_abs": float(np.nanmax(np.abs(diff))),
        "corr": _vector_corr(ref, cand),
    }


def _apply_vetted_resampler(
    baseline_stim: torch.Tensor,
    eye_offset_deg: np.ndarray,
    pixels_per_degree: float,
) -> torch.Tensor:
    if baseline_stim.shape[0] != 1:
        raise ValueError(f"Expected singleton baseline stim batch, got {tuple(baseline_stim.shape)}")
    lagged_movie = baseline_stim[0].permute(1, 0, 2, 3).contiguous()
    eye_trace_deg = np.repeat(np.asarray(eye_offset_deg, dtype=np.float32)[None, :], lagged_movie.shape[0], axis=0)
    eye_norm = stored_eyepos_to_eye_norm(
        eye_trace_deg,
        float(pixels_per_degree),
        lagged_movie.shape[-2:],
        stored_convention="visual_xy",
        device=lagged_movie.device,
    )
    shifted = shift_movie_with_eye(
        lagged_movie,
        eye_norm,
        out_size=lagged_movie.shape[-2:],
        center=(0.0, 0.0),
        mode="bilinear",
        padding_mode="zeros",
        scale_factor=1.0,
        align_corners=True,
    )
    return shifted.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def _tensor_equivalence_rows(
    data: dict[str, Any],
    pixels_per_degree: float,
    state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if state is None:
        return []

    rows: list[dict[str, Any]] = []
    replay_stim = _reconstruct_replay_baseline_stim(
        data=data,
        processed_trial_idx=int(state["baseline_processed_trial_idx"]),
        processed_time_idx=int(state["baseline_processed_time_idx"]),
    )
    if replay_stim is not None:
        rows.append({
            "comparison": "collated_vs_dataset_replay",
            "probe_axis": "baseline",
            "transform_mode": "none",
            "window_id": state["window_id"],
            **_tensor_comparison_metrics(replay_stim, state["baseline_stim"]),
        })

    probe_offsets_px = {
        "pos_x": np.array([1.0, 0.0], dtype=np.float64),
        "pos_y": np.array([0.0, 1.0], dtype=np.float64),
    }
    for probe_axis, raw_offset_px in probe_offsets_px.items():
        vetted = _apply_vetted_resampler(
            state["baseline_stim"],
            eye_offset_deg=raw_offset_px / float(pixels_per_degree),
            pixels_per_degree=pixels_per_degree,
        )
        for transform_mode in ("xy", "x_negy"):
            if transform_mode == "x_negy":
                audit_offset_px = visual_eye_deg_to_canonical_shift_px(
                    raw_offset_px[None, :] / float(pixels_per_degree),
                    pixels_per_degree,
                )
            else:
                audit_offset_px = _transform_eye_to_retinal_offsets(raw_offset_px[None, :], transform_mode)
            audit_shifted = _shift_stimulus_batch(state["baseline_stim"].clone(), audit_offset_px)
            rows.append({
                "comparison": "vetted_vs_audit_shift",
                "probe_axis": probe_axis,
                "transform_mode": transform_mode,
                "window_id": state["window_id"],
                **_tensor_comparison_metrics(vetted, audit_shifted),
            })
    return rows


def _write_provenance_summary(output_dir: Path) -> None:
    lines = [
        "Figure 4 audit provenance summary",
        "",
        "Based on the current code read, the audit inputs follow this provenance chain:",
        "",
        "eye provenance:",
        "- win.eyepos_deg comes from _collect_image_windows -> data['eyepos'][tri, ti].",
        "- data['eyepos'] comes from eval.fixrsvp.collate_fixrsvp_data -> dataset.dsets[dset_idx]['eyepos'][ix].numpy().",
        "- eval.fixrsvp does not flip or swap eyepos axes during collation, duplicate removal, fixation filtering, or image-id alignment.",
        "",
        "stim provenance:",
        "- win.stim comes from _collect_image_windows -> data['stim'][tri, ti].",
        "- data['stim'] comes from eval.fixrsvp._build_stim_source, which only lag-stacks dataset.dsets[dset_idx]['stim'] rows across time.",
        "- eval.fixrsvp does not apply eye-dependent spatial resampling while constructing data['stim'].",
        "- multi_basic_120_long_legacy.yaml applies only pixelnorm + unsqueeze to stim and no transform ops to eye_pos itself.",
        "",
        "historical convention note:",
        f"- scripts.fixrsvp_eye_conventions.DEFAULT_STORED_EYE_CONVENTION = {DEFAULT_STORED_EYE_CONVENTION}.",
        "- scripts/spatial_info.py and scripts/check_fixrsvp_counterfactual_stim_match.py now both route stored eyepos through scripts.fixrsvp_eye_conventions.stored_eyepos_to_eye_norm().",
        "- The audit keeps explicit transform sweeps for hypothesis testing, but the vetted resampler path now shares the same convention helper entrypoint.",
    ]
    (output_dir / "provenance_summary.txt").write_text("\n".join(lines) + "\n")


def _write_run_config(output_dir: Path, config: dict[str, Any]) -> None:
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _write_render_sanity_check(
    output_dir: Path,
    state: dict[str, Any] | None,
) -> None:
    if state is None:
        return
    raw_offsets = np.asarray(state["shift_eye_px"], dtype=np.float64)
    positive_y = np.where(np.isfinite(raw_offsets[:, 1]) & (raw_offsets[:, 1] > 0))[0]
    if positive_y.size == 0:
        return
    probe_idx = int(positive_y[np.argmax(raw_offsets[positive_y, 1])])
    positive_eye_offset = raw_offsets[probe_idx]
    xy_shift = _transform_eye_to_retinal_offsets(positive_eye_offset[None, :], "xy")
    canonical_shift = visual_eye_deg_to_canonical_shift_px(
        positive_eye_offset[None, :] / float(state["pixels_per_degree"]),
        state["pixels_per_degree"],
    )

    baseline = _stimulus_frame_for_display(state["baseline_stim"])
    xy_shifted = _stimulus_frame_for_display(
        _shift_stimulus_batch(state["baseline_stim"].clone(), xy_shift)
    )
    canonical_shifted = _stimulus_frame_for_display(
        _shift_stimulus_batch(state["baseline_stim"].clone(), canonical_shift)
    )

    vlim = float(np.nanmax(np.abs([baseline, xy_shifted, canonical_shifted])))
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("baseline", baseline),
        ("xy shift", xy_shifted),
        ("x_negy shift", canonical_shifted),
    ]
    for ax, (title, image) in zip(axes, panels, strict=True):
        ax.imshow(image, cmap="gray", vmin=-vlim, vmax=vlim, origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output_dir / "figures" / "render_sanity_check.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "positive eye_y sanity check",
        f"window_id: {state['window_id']}",
        f"centering_mode: {state['centering_mode']}",
        f"predictor_mode: {state['predictor_mode']}",
        f"probe_sample_index: {probe_idx}",
        f"raw_eye_offset_px: x={positive_eye_offset[0]:.4f}, y={positive_eye_offset[1]:.4f}",
        "positive eye_y should correspond to image content moving: unresolved from metadata alone; inspect rendered panels against the canonical training/simulation path.",
        f"current xy produces shift offsets: x={xy_shift[0, 0]:.4f}, y={xy_shift[0, 1]:.4f}",
        f"x_negy produces shift offsets: x={canonical_shift[0, 0]:.4f}, y={canonical_shift[0, 1]:.4f}",
        "canonical pipeline produces: unavailable in this audit script; compare render_sanity_check.png against the training/simulation path if that renderer is exposed elsewhere.",
    ]
    (output_dir / "render_sanity_check.txt").write_text("\n".join(lines) + "\n")


def _basis_from_model_fem_audit(
    eye_predictor_px: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    if delta.shape[0] < 4:
        return np.zeros((delta.shape[1], 2), dtype=np.float64)
    coeffs = _fit_slopes_with_intercept(eye_predictor_px, delta)
    return np.asarray(coeffs.T, dtype=np.float64)


def _basis_from_fem_cov_pcs_audit(
    delta: np.ndarray,
) -> np.ndarray:
    if delta.shape[0] < 4:
        return np.zeros((delta.shape[1], 2), dtype=np.float64)
    if delta.shape[0] < 2:
        return np.zeros((delta.shape[1], 2), dtype=np.float64)
    centered = delta - delta.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] == 0:
        return np.zeros((delta.shape[1], 2), dtype=np.float64)
    basis = vh[:2].T if vh.shape[0] >= 2 else vh[:1].T
    if basis.shape[1] == 1:
        basis = np.column_stack([basis[:, 0], np.zeros_like(basis[:, 0])])
    return np.asarray(basis, dtype=np.float64)


def _basis_from_jacobian_audit(
    model,
    baseline_stim: torch.Tensor,
    dataset_idx: int,
    jacobian_step_px: float,
    transform_mode: str,
) -> np.ndarray:
    eye_offsets = np.array(
        [
            [jacobian_step_px, 0.0],
            [-jacobian_step_px, 0.0],
            [0.0, jacobian_step_px],
            [0.0, -jacobian_step_px],
        ],
        dtype=np.float64,
    )
    shift_offsets = _transform_eye_to_retinal_offsets(eye_offsets, transform_mode)
    responses = compute_grid_responses(model, baseline_stim, shift_offsets, dataset_idx)
    jac_x = (responses[0] - responses[1]) / (2.0 * jacobian_step_px)
    jac_y = (responses[2] - responses[3]) / (2.0 * jacobian_step_px)
    return np.stack([jac_x, jac_y], axis=-1).astype(np.float64)


def _window_basis_map(
    model,
    win,
    dataset_idx: int,
    pixels_per_degree: float,
    transform_mode: str,
    centering_mode: str,
    predictor_mode: str,
    jacobian_step_px: float,
    min_empirical_samples: int,
) -> dict[str, Any]:
    prepared = _prepare_window_state(win, centering_mode, pixels_per_degree, predictor_mode)
    baseline_resp = _predict_responses(model, prepared["baseline_stim"], dataset_idx)[0]
    shift_offsets_px = _transform_eye_to_retinal_offsets(prepared["shift_eye_px"], transform_mode)
    model_responses = compute_grid_responses(model, prepared["baseline_stim"], shift_offsets_px, dataset_idx)
    delta = model_responses - baseline_resp[None, :]
    b_model = _basis_from_model_fem_audit(prepared["eye_predictor_px"], delta)
    fem_pcs = _basis_from_fem_cov_pcs_audit(delta)
    j_local = _basis_from_jacobian_audit(
        model,
        prepared["baseline_stim"],
        dataset_idx,
        jacobian_step_px,
        transform_mode,
    )
    emp_B = _fit_empirical_B(win.robs_rates, prepared["eye_predictor_px"], min_empirical_samples)
    sigma_eye = np.cov(prepared["eye_predictor_px"].T).astype(np.float64)
    return {
        **prepared,
        "baseline_resp": baseline_resp,
        "shift_offsets_px": shift_offsets_px,
        "model_delta": delta,
        "eye_cov_xx": float(sigma_eye[0, 0]) if sigma_eye.shape == (2, 2) else float("nan"),
        "eye_cov_xy": float(sigma_eye[0, 1]) if sigma_eye.shape == (2, 2) else float("nan"),
        "eye_cov_yy": float(sigma_eye[1, 1]) if sigma_eye.shape == (2, 2) else float("nan"),
        "B_emp_full": emp_B,
        "B_model_basis": b_model,
        "FEM_PCs_basis": fem_pcs,
        "J_local_basis": j_local,
        "model_fem_trace": float(np.trace(np.cov(delta.T))) if int(delta.shape[0]) >= 2 else float("nan"),
        "jacobian_fro_norm": float(np.linalg.norm(j_local, ord="fro")),
        "mean_model_rate": float(np.mean(baseline_resp)),
    }


def _basis_for_shuffled_window(
    model,
    basis_name: str,
    ref_state: dict[str, Any],
    target_state: dict[str, Any],
    dataset_idx: int,
    jacobian_step_px: float,
    transform_mode: str,
) -> np.ndarray:
    if basis_name == "B_model":
        delta = compute_grid_responses(model, ref_state["baseline_stim"], target_state["shift_offsets_px"], dataset_idx) - ref_state["baseline_resp"][None, :]
        return _basis_from_model_fem_audit(
            target_state["eye_predictor_px"],
            delta,
        )
    if basis_name == "FEM_PCs":
        delta = compute_grid_responses(model, ref_state["baseline_stim"], target_state["shift_offsets_px"], dataset_idx) - ref_state["baseline_resp"][None, :]
        return _basis_from_fem_cov_pcs_audit(delta)
    if basis_name == "J_local":
        return _basis_from_jacobian_audit(
            model,
            ref_state["baseline_stim"],
            dataset_idx,
            jacobian_step_px,
            transform_mode,
        )
    raise ValueError(f"Unknown basis name: {basis_name}")


def _summarize_rows(rows: list[dict[str, Any]], session_name: str) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    key_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["window_level"], row["model_object"], row["transform_mode"], row["centering_mode"], row["predictor_mode"])
        key_groups.setdefault(key, []).append(row)

    for (window_level, model_object, transform_mode, centering_mode, predictor_mode), subset in sorted(key_groups.items()):
        paired_2d = [
            {"matched": row["matched_2d"], "shuffled": row["shuffled_2d"]}
            for row in subset
        ]
        paired_top1 = [
            {"matched": row["matched_top1"], "shuffled": row["shuffled_top1"]}
            for row in subset
        ]
        summary_2d = _paired_delta_summary(paired_2d, "matched", "shuffled")
        summary_top1 = _paired_delta_summary(paired_top1, "matched", "shuffled")
        summary_rows.append({
            "session": session_name,
            "window_level": window_level,
            "model_object": model_object,
            "transform_mode": transform_mode,
            "centering_mode": centering_mode,
            "predictor_mode": predictor_mode,
            "mixture_mode": "single_baseline",
            "n_windows": len(subset),
            "median_samples_per_window": _nanmedian([float(row["n_samples"]) for row in subset]),
            "emp_ceiling_2d": _nanmedian([float(row["emp_ceiling_2d"]) for row in subset]),
            "emp_ceiling_top1": _nanmedian([float(row["emp_ceiling_top1"]) for row in subset]),
            "matched_2d": _nanmedian([float(row["matched_2d"]) for row in subset]),
            "shuffled_2d": _nanmedian([float(row["shuffled_2d"]) for row in subset]),
            "delta_2d": _nanmedian([float(row["delta_2d"]) for row in subset]),
            "delta_2d_ci_low": float(summary_2d["ci95_low"]),
            "delta_2d_ci_high": float(summary_2d["ci95_high"]),
            "matched_top1": _nanmedian([float(row["matched_top1"]) for row in subset]),
            "shuffled_top1": _nanmedian([float(row["shuffled_top1"]) for row in subset]),
            "delta_top1": _nanmedian([float(row["delta_top1"]) for row in subset]),
            "delta_top1_ci_low": float(summary_top1["ci95_low"]),
            "delta_top1_ci_high": float(summary_top1["ci95_high"]),
            "delta_over_ceiling_2d": _nanmedian([float(row["delta_over_ceiling_2d"]) for row in subset]),
            "matched_over_ceiling_2d": _nanmedian([float(row["matched_over_ceiling_2d"]) for row in subset]),
            "b_emp_model_x_corr": _nanmedian([float(row["b_emp_model_x_corr"]) for row in subset]),
            "b_emp_model_y_corr": _nanmedian([float(row["b_emp_model_y_corr"]) for row in subset]),
            "notes": f"empirical_target=local_regenerated_split_bundles;window_level={window_level}",
        })
    return summary_rows


def _canonical_vs_current_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = {
        (row["window_level"], row["model_object"], row["centering_mode"], row["predictor_mode"], row["transform_mode"]): row
        for row in summary_rows
    }
    output_rows: list[dict[str, Any]] = []
    keys = sorted({(row["window_level"], row["model_object"], row["centering_mode"], row["predictor_mode"]) for row in summary_rows})
    for window_level, model_object, centering_mode, predictor_mode in keys:
        current = rows_by_key.get((window_level, model_object, centering_mode, predictor_mode, "xy"))
        canonical = rows_by_key.get((window_level, model_object, centering_mode, predictor_mode, "x_negy"))
        if current is None or canonical is None:
            continue
        output_rows.append({
            "session": canonical["session"],
            "window_level": window_level,
            "model_object": model_object,
            "centering_mode": centering_mode,
            "predictor_mode": predictor_mode,
            "delta_2d_current": current["delta_2d"],
            "delta_2d_x_negy": canonical["delta_2d"],
            "delta_top1_current": current["delta_top1"],
            "delta_top1_x_negy": canonical["delta_top1"],
            "matched_2d_current": current["matched_2d"],
            "matched_2d_x_negy": canonical["matched_2d"],
            "matched_top1_current": current["matched_top1"],
            "matched_top1_x_negy": canonical["matched_top1"],
            "b_emp_model_y_corr_current": current["b_emp_model_y_corr"],
            "b_emp_model_y_corr_x_negy": canonical["b_emp_model_y_corr"],
        })
    return output_rows


def _window_ladder_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = {
        (row["window_level"], row["model_object"], row["transform_mode"], row["centering_mode"], row["predictor_mode"]): row
        for row in summary_rows
    }
    output_rows: list[dict[str, Any]] = []
    keys = sorted({(row["model_object"], row["transform_mode"], row["centering_mode"], row["predictor_mode"]) for row in summary_rows})
    for model_object, transform_mode, centering_mode, predictor_mode in keys:
        image_row = rows_by_key.get(("image_id", model_object, transform_mode, centering_mode, predictor_mode))
        prev_next_row = rows_by_key.get(("prev_next", model_object, transform_mode, centering_mode, predictor_mode))
        history_row = rows_by_key.get(("history_hash", model_object, transform_mode, centering_mode, predictor_mode))
        if image_row is None and prev_next_row is None and history_row is None:
            continue
        source_row = history_row if history_row is not None else (prev_next_row if prev_next_row is not None else image_row)
        if source_row is None:
            continue
        output_rows.append({
            "session": source_row["session"],
            "model_object": model_object,
            "transform_mode": transform_mode,
            "centering_mode": centering_mode,
            "predictor_mode": predictor_mode,
            "n_windows_image_id": image_row["n_windows"] if image_row is not None else 0,
            "n_windows_prev_next": prev_next_row["n_windows"] if prev_next_row is not None else 0,
            "n_windows_history_hash": history_row["n_windows"] if history_row is not None else 0,
            "median_samples_image_id": image_row["median_samples_per_window"] if image_row is not None else float("nan"),
            "median_samples_prev_next": prev_next_row["median_samples_per_window"] if prev_next_row is not None else float("nan"),
            "median_samples_history_hash": history_row["median_samples_per_window"] if history_row is not None else float("nan"),
            "matched_2d_image_id": image_row["matched_2d"] if image_row is not None else float("nan"),
            "matched_2d_prev_next": prev_next_row["matched_2d"] if prev_next_row is not None else float("nan"),
            "matched_2d_history_hash": history_row["matched_2d"] if history_row is not None else float("nan"),
            "delta_2d_image_id": image_row["delta_2d"] if image_row is not None else float("nan"),
            "delta_2d_prev_next": prev_next_row["delta_2d"] if prev_next_row is not None else float("nan"),
            "delta_2d_history_hash": history_row["delta_2d"] if history_row is not None else float("nan"),
            "delta_over_ceiling_2d_image_id": image_row["delta_over_ceiling_2d"] if image_row is not None else float("nan"),
            "delta_over_ceiling_2d_prev_next": prev_next_row["delta_over_ceiling_2d"] if prev_next_row is not None else float("nan"),
            "delta_over_ceiling_2d_history_hash": history_row["delta_over_ceiling_2d"] if history_row is not None else float("nan"),
            "delta_2d_prev_next_minus_image": (
                prev_next_row["delta_2d"] - image_row["delta_2d"]
                if image_row is not None and prev_next_row is not None and np.isfinite(image_row["delta_2d"]) and np.isfinite(prev_next_row["delta_2d"])
                else float("nan")
            ),
            "delta_2d_history_minus_image": (
                history_row["delta_2d"] - image_row["delta_2d"]
                if image_row is not None and history_row is not None and np.isfinite(image_row["delta_2d"]) and np.isfinite(history_row["delta_2d"])
                else float("nan")
            ),
        })
    return output_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_union_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    summary_rows: list[dict[str, Any]],
    verification_rows: list[dict[str, Any]],
    tensor_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Figure 4 geometry-bridge audit",
        "",
        "## Scope",
        "- Implements the canonical y-flip audit for model stimulus shifts.",
        "- Separates model stimulus-shift transforms from empirical predictor-frame transforms.",
        "- Runs the full sign/axis grid as a confirmation sweep.",
        "- Regenerates empirical split-half bundles inside the audit using the same predictor frame used for the local empirical checks.",
        "- Computes ceiling-normalized summaries.",
        "",
        "## Current limitations",
        "- Mixture-matched model objects and residual analyses are not implemented yet.",
        "- Window ladder currently includes image_id and previous/next image context levels.",
        "- Exact lagged-stimulus history hashes are currently omitted by default because this fixRSVP slice produced no powered repeated-history windows.",
        "- Figure generation is not implemented yet in this first executable slice.",
        "",
        "## Guardrails",
        "- `x_negy` is the principled transform from the resampler convention; the 8-mode sweep is confirmatory, not a free selection step.",
        "- If a non-canonical transform outperforms `x_negy`, treat that as a metadata or calibration red flag rather than a mode to adopt.",
        "- `J_local` is structurally insensitive to eye-cloud-dependent mixture and window-pooling manipulations because it only sees the local baseline image tangent.",
        "- `mean` should be interpreted as a mean-nearest baseline, not an exact rendered mean-eye retinal state.",
        "- `current` currently means median-nearest baseline stimulus with a mean-centered eye predictor, not purely baseline-centered eye offsets.",
        "",
        "## Command",
        f"- subject/date: {args.subject} {args.date}",
        f"- dataset_configs_path: {args.dataset_configs_path}",
        f"- checkpoint_path: {args.checkpoint_path}",
        f"- split_bundle_path: {args.split_bundle_path or 'none'} (provenance only; local empirical bundles are regenerated in this audit)",
        f"- transform_modes: {args.transform_modes}",
        f"- predictor_modes: {args.predictor_modes}",
        f"- centering_modes: {args.centering_modes}",
        f"- basis_names: {args.basis_names}",
        f"- window_levels: {args.window_levels}",
        "",
        "## Canonical verification",
    ]
    if verification_rows:
        best = sorted(verification_rows, key=lambda row: row.get("n_samples", 0), reverse=True)[0]
        lines.extend([
            f"- verification_window_id: {best['window_id']}",
            f"- predictor_mode: {best['predictor_mode']}",
            f"- empirical_bundle_vs_model_x_corr_current: {best['emp_model_x_corr_current']:.4f}",
            f"- empirical_bundle_vs_model_x_corr_canonical: {best['emp_model_x_corr_canonical']:.4f}",
            f"- empirical_bundle_vs_model_y_corr_current: {best['emp_model_y_corr_current']:.4f}",
            f"- empirical_bundle_vs_model_y_corr_canonical: {best['emp_model_y_corr_canonical']:.4f}",
        ])
        if best["emp_model_y_corr_canonical"] <= best["emp_model_y_corr_current"]:
            lines.append("- warning: canonical y correlation did not improve on the representative window; inspect eye-trace metadata before treating any alternative transform as valid.")
    else:
        lines.append("- verification_window_id: none")
    lines.extend(["", "## Session summary"])
    for row in summary_rows:
        if row["transform_mode"] != "x_negy":
            continue
        lines.extend([
            f"- {row['window_level']} / {row['model_object']} / {row['centering_mode']} / {row['predictor_mode']}: delta_2d={row['delta_2d']:.4f}, delta_top1={row['delta_top1']:.4f}, delta_over_ceiling_2d={row['delta_over_ceiling_2d']:.4f}",
        ])
    ladder_rows = [row for row in _window_ladder_rows(summary_rows) if row["transform_mode"] == "x_negy"]
    if ladder_rows:
        lines.extend(["", "## Window ladder"])
        for row in ladder_rows:
            lines.append(
                f"- {row['model_object']} / {row['centering_mode']} / {row['predictor_mode']}: image_id delta_2d={row['delta_2d_image_id']:.4f}, prev_next delta_2d={row['delta_2d_prev_next']:.4f}, history_hash delta_2d={row['delta_2d_history_hash']:.4f}"
            )
    render_probe_path = path.parent / "figures" / "render_sanity_check.png"
    if render_probe_path.exists():
        lines.extend([
            "",
            "## Render sanity check",
            f"- render_sanity_check_png: {render_probe_path}",
            f"- render_sanity_check_txt: {path.parent / 'render_sanity_check.txt'}",
        ])
    if tensor_rows:
        lines.extend([
            "",
            "## Tensor equivalence",
            f"- tensor_equivalence_csv: {path.parent / 'tensor_equivalence_summary.csv'}",
            f"- provenance_summary_txt: {path.parent / 'provenance_summary.txt'}",
        ])
    path.write_text("\n".join(lines) + "\n")


def run_geometry_bridge_audit(
    subject: str,
    date: str,
    dataset_configs_path: str,
    checkpoint_path: str | None,
    dataset_idx: int,
    split_bundle_path: str | None,
    output_dir: Path,
    min_samples: int,
    n_shuffle_matches: int,
    jacobian_step_px: float,
    rank_ratio_threshold: float,
    transform_modes: list[str],
    centering_modes: list[str],
    basis_names: list[str],
    model_type: str | None,
    model_index: int | None,
    model_device: str,
    max_windows: int | None,
    min_empirical_samples: int,
    n_split_repeats: int,
    predictor_modes: list[str],
    window_levels: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = get_fixrsvp_data(
        subject=subject,
        date=date,
        dataset_configs_path=dataset_configs_path,
        use_cached_data=True,
    )
    dt = float(data.get("dt", 1.0 / REFERENCE_RATE_HZ))
    windows = _collect_audit_windows(data, min_samples=min_samples, dt=dt, window_levels=window_levels)
    pixels_per_degree = _resolve_pixels_per_degree(data)

    model, _info = _load_model_for_session(
        dataset_configs_path=dataset_configs_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        model_index=model_index,
        model_device=model_device,
    )
    model_any: Any = model
    model_any.model.eval()
    session_name = f"{subject}_{date}"
    if dataset_idx is not None and hasattr(model_any, "names"):
        try:
            dataset_idx = model_any.names.index(session_name)
        except ValueError:
            pass

    filtered_windows: list[AuditWindow] = []
    for window_level in window_levels:
        level_windows = [
            win for win in windows
            if win.window_level == window_level and _window_has_valid_empirical_bundle(
                win=win,
                centering_modes=centering_modes,
                predictor_modes=predictor_modes,
                pixels_per_degree=pixels_per_degree,
                min_empirical_samples=min_empirical_samples,
                n_split_repeats=n_split_repeats,
                session_name=session_name,
            )
        ]
        if max_windows is not None:
            level_windows = level_windows[:max_windows]
        filtered_windows.extend(level_windows)
    windows = filtered_windows

    all_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    render_probe_state: dict[str, Any] | None = None

    for window_level in window_levels:
        level_windows = [win for win in windows if win.window_level == window_level]
        if not level_windows:
            continue

        for centering_mode in centering_modes:
            config_states: list[dict[str, Any]] = []
            current_transform_states: dict[str, dict[str, Any]] = {}
            canonical_transform_states: dict[str, dict[str, Any]] = {}

            for win in level_windows:
                window_id = win.window_key
                time_bin = "" if win.time_bin is None else int(win.time_bin)
                for predictor_mode in predictor_modes:
                    prototype_state = _prepare_window_state(win, centering_mode, pixels_per_degree, predictor_mode)
                    emp_bundle = _generate_empirical_split_bundle(
                        robs=win.robs_rates,
                        eye_predictor_px=prototype_state["eye_predictor_px"],
                        trial_indices=win.trial_indices,
                        min_samples=min_empirical_samples,
                        n_split_repeats=n_split_repeats,
                        seed=_stable_int_hash(f"{session_name}|{window_id}|{centering_mode}|{predictor_mode}|emp_bundle"),
                    )
                    if emp_bundle is None:
                        continue
                    b_emp_reference = _empirical_bundle_reference_B(emp_bundle)

                    for transform_mode in transform_modes:
                        state = _window_basis_map(
                            model=model,
                            win=win,
                            dataset_idx=dataset_idx,
                            pixels_per_degree=pixels_per_degree,
                            transform_mode=transform_mode,
                            centering_mode=centering_mode,
                            predictor_mode=predictor_mode,
                            jacobian_step_px=jacobian_step_px,
                            min_empirical_samples=min_empirical_samples,
                        )
                        row_state = {
                            "session": session_name,
                            "image_id": int(win.image_id),
                            "window_level": win.window_level,
                            "window_key": win.window_key,
                            "window_id": window_id,
                            "history_key": win.history_key,
                            "time_bin": time_bin,
                            "n_samples": int(win.eyepos_deg.shape[0]),
                            "baseline_processed_trial_idx": int(win.trial_indices[state["baseline_sample_idx"]]),
                            "baseline_processed_time_idx": int(win.time_indices[state["baseline_sample_idx"]]),
                            "emp_bundle": emp_bundle,
                            "B_emp_reference": b_emp_reference,
                            "robs_rates": win.robs_rates,
                            "transform_mode": transform_mode,
                            "centering_mode": centering_mode,
                            "predictor_mode": predictor_mode,
                            **state,
                        }
                        config_states.append(row_state)
                        verification_key = f"{window_level}|{window_id}|{predictor_mode}"
                        if transform_mode == "xy":
                            current_transform_states[verification_key] = row_state
                            if window_level == "image_id" and centering_mode == "current" and predictor_mode == "emp_xy" and render_probe_state is None:
                                render_probe_state = row_state
                        if transform_mode == "x_negy":
                            canonical_transform_states[verification_key] = row_state

            if not config_states:
                continue

            for predictor_mode in predictor_modes:
                predictor_states = [row for row in config_states if row["predictor_mode"] == predictor_mode]
                if not predictor_states:
                    continue

                for transform_mode in transform_modes:
                    relevant_states = [row for row in predictor_states if row["transform_mode"] == transform_mode]
                    if not relevant_states:
                        continue

                    global_basis_map = {
                        "B_model": _build_global_basis([row["B_model_basis"] for row in relevant_states]),
                        "FEM_PCs": _build_global_basis([row["FEM_PCs_basis"] for row in relevant_states]),
                        "J_local": _build_global_basis([row["J_local_basis"] for row in relevant_states]),
                    }
                    rows_for_matching = [
                        {
                            "window_id": row["window_id"],
                            "image_id": row["image_id"],
                            "jacobian_fro_norm": row["jacobian_fro_norm"],
                            "eye_amplitude_px2": row["eye_cov_xx"] + row["eye_cov_yy"],
                            "mean_model_rate": row["mean_model_rate"],
                        }
                        for row in relevant_states
                    ]
                    state_by_window = {row["window_id"]: row for row in relevant_states}

                    for state in relevant_states:
                        matched_ids = _matched_candidate_ids_local(rows_for_matching, {
                            "window_id": state["window_id"],
                            "image_id": state["image_id"],
                            "jacobian_fro_norm": state["jacobian_fro_norm"],
                            "eye_amplitude_px2": state["eye_cov_xx"] + state["eye_cov_yy"],
                            "mean_model_rate": state["mean_model_rate"],
                        }, n_matches=n_shuffle_matches)

                        matched_basis_map = {
                            "B_model": state["B_model_basis"],
                            "FEM_PCs": state["FEM_PCs_basis"],
                            "J_local": state["J_local_basis"],
                        }

                        for basis_name in basis_names:
                            matched_summary = _basis_summary(
                                matched_basis_map[basis_name],
                                state["emp_bundle"],
                                rank_ratio_threshold,
                            )
                            shuffled_summaries = []
                            for matched_id in matched_ids:
                                ref_state = state_by_window.get(matched_id)
                                if ref_state is None:
                                    continue
                                if matched_id == state["window_id"]:
                                    continue
                                if state["window_level"] == "history_hash" and ref_state.get("history_key") == state.get("history_key"):
                                    continue
                                shuffled_basis = _basis_for_shuffled_window(
                                    model=model,
                                    basis_name=basis_name,
                                    ref_state=ref_state,
                                    target_state=state,
                                    dataset_idx=dataset_idx,
                                    jacobian_step_px=jacobian_step_px,
                                    transform_mode=transform_mode,
                                )
                                shuffled_summaries.append(
                                    _basis_summary(shuffled_basis, state["emp_bundle"], rank_ratio_threshold)
                                )
                            if shuffled_summaries:
                                shuffled_2d = _nanmedian([float(item["align_to_emp_2d"]) for item in shuffled_summaries])
                                shuffled_top1 = _nanmedian([float(item["align_to_emp_top1"]) for item in shuffled_summaries])
                            else:
                                shuffled_2d = float("nan")
                                shuffled_top1 = float("nan")

                            all_rows.append({
                                "session": session_name,
                                "window_key": state["window_key"],
                                "window_id": state["window_id"],
                                "image_id": state["image_id"],
                                "time_bin": state["time_bin"],
                                "history_key": state["history_key"],
                                "n_samples": int(state["n_samples"]),
                                "model_object": basis_name,
                                "transform_mode": transform_mode,
                                "centering_mode": centering_mode,
                                "predictor_mode": predictor_mode,
                                "window_level": state["window_level"],
                                "mixture_mode": "single_baseline",
                                "emp_ceiling_2d": matched_summary["emp_split_alignment_2d"],
                                "emp_ceiling_top1": matched_summary["emp_split_alignment_top1"],
                                "matched_2d": matched_summary["align_to_emp_2d"],
                                "shuffled_2d": shuffled_2d,
                                "delta_2d": matched_summary["align_to_emp_2d"] - shuffled_2d if np.isfinite(shuffled_2d) else float("nan"),
                                "matched_top1": matched_summary["align_to_emp_top1"],
                                "shuffled_top1": shuffled_top1,
                                "delta_top1": matched_summary["align_to_emp_top1"] - shuffled_top1 if np.isfinite(shuffled_top1) else float("nan"),
                                "delta_over_ceiling_2d": _safe_ratio(
                                    matched_summary["align_to_emp_2d"] - shuffled_2d if np.isfinite(shuffled_2d) else float("nan"),
                                    matched_summary["emp_split_alignment_2d"],
                                ),
                                "matched_over_ceiling_2d": _safe_ratio(
                                    matched_summary["align_to_emp_2d"],
                                    matched_summary["emp_split_alignment_2d"],
                                ),
                                "mean_eye_x": state["mean_eye_x"],
                                "mean_eye_y": state["mean_eye_y"],
                                "cov_eye_xx": state["eye_cov_xx"],
                                "cov_eye_xy": state["eye_cov_xy"],
                                "cov_eye_yy": state["eye_cov_yy"],
                                "model_j_norm": state["jacobian_fro_norm"],
                                "model_fem_trace": state["model_fem_trace"],
                                "psth_amp": float(np.nanstd(state["robs_rates"])),
                                "mean_rate": float(np.nanmean(state["robs_rates"])),
                                "valid": True,
                                "failure_reason": "",
                                "global_matched_2d": _basis_summary(global_basis_map[basis_name], state["emp_bundle"], rank_ratio_threshold)["align_to_emp_2d"],
                                "b_emp_model_x_corr": _vector_corr(state["B_emp_reference"][:, 0], state["B_model_basis"][:, 0]) if basis_name == "B_model" else float("nan"),
                                "b_emp_model_y_corr": _vector_corr(state["B_emp_reference"][:, 1], state["B_model_basis"][:, 1]) if basis_name == "B_model" else float("nan"),
                            })

            for verification_key, canonical_state in canonical_transform_states.items():
                current_state = current_transform_states.get(verification_key)
                if current_state is None:
                    continue
                verification_rows.append({
                    "session": session_name,
                    "window_level": canonical_state["window_level"],
                    "window_id": canonical_state["window_id"],
                    "n_samples": int(canonical_state["n_samples"]),
                    "centering_mode": centering_mode,
                    "predictor_mode": canonical_state["predictor_mode"],
                    "emp_model_x_corr_current": _vector_corr(current_state["B_emp_reference"][:, 0], current_state["B_model_basis"][:, 0]),
                    "emp_model_x_corr_canonical": _vector_corr(canonical_state["B_emp_reference"][:, 0], canonical_state["B_model_basis"][:, 0]),
                    "emp_model_y_corr_current": _vector_corr(current_state["B_emp_reference"][:, 1], current_state["B_model_basis"][:, 1]),
                    "emp_model_y_corr_canonical": _vector_corr(canonical_state["B_emp_reference"][:, 1], canonical_state["B_model_basis"][:, 1]),
                    "b_model_2d_current": _basis_summary(current_state["B_model_basis"], current_state["emp_bundle"], rank_ratio_threshold)["align_to_emp_2d"],
                    "b_model_2d_canonical": _basis_summary(canonical_state["B_model_basis"], canonical_state["emp_bundle"], rank_ratio_threshold)["align_to_emp_2d"],
                    "b_model_top1_current": _basis_summary(current_state["B_model_basis"], current_state["emp_bundle"], rank_ratio_threshold)["align_to_emp_top1"],
                    "b_model_top1_canonical": _basis_summary(canonical_state["B_model_basis"], canonical_state["emp_bundle"], rank_ratio_threshold)["align_to_emp_top1"],
                })

    summary_rows = _summarize_rows(all_rows, session_name)
    window_ladder_rows = _window_ladder_rows(summary_rows)
    audit_summary = {
        "session": session_name,
        "n_window_rows": len(all_rows),
        "n_summary_rows": len(summary_rows),
        "window_levels": window_levels,
        "transform_modes": transform_modes,
        "predictor_modes": predictor_modes,
        "centering_modes": centering_modes,
        "basis_names": basis_names,
    }

    _write_csv(output_dir / "audit_by_window.csv", all_rows)
    _write_csv(output_dir / "audit_by_session.csv", summary_rows)
    _write_csv(output_dir / "audit_summary.csv", summary_rows)
    _write_csv(output_dir / "window_definition_ladder_summary.csv", window_ladder_rows)
    _write_csv(output_dir / "canonical_vs_current_summary.csv", _canonical_vs_current_rows(summary_rows))
    _write_csv(
        output_dir / "sign_axis_grid_summary.csv",
        [row for row in summary_rows if row["centering_mode"] == "current"],
    )
    _write_csv(
        output_dir / "ceiling_normalized_summary.csv",
        [
            {
                "session": row["session"],
                "model_object": row["model_object"],
                "window_level": row["window_level"],
                "transform_mode": row["transform_mode"],
                "centering_mode": row["centering_mode"],
                "delta_over_ceiling_2d": row["delta_over_ceiling_2d"],
                "matched_over_ceiling_2d": row["matched_over_ceiling_2d"],
            }
            for row in summary_rows
        ],
    )
    _write_csv(output_dir / "verification_summary.csv", verification_rows)
    (output_dir / "figures").mkdir(exist_ok=True)
    _write_render_sanity_check(output_dir, render_probe_state)
    tensor_rows = _tensor_equivalence_rows(data, pixels_per_degree, render_probe_state)
    _write_csv(output_dir / "tensor_equivalence_summary.csv", tensor_rows)
    _write_run_config(output_dir, {
        "subject": subject,
        "date": date,
        "dataset_configs_path": dataset_configs_path,
        "checkpoint_path": checkpoint_path,
        "dataset_idx": dataset_idx,
        "split_bundle_path": split_bundle_path,
        "output_dir": str(output_dir),
        "min_samples": min_samples,
        "min_empirical_samples": min_empirical_samples,
        "n_shuffle_matches": n_shuffle_matches,
        "jacobian_step_px": jacobian_step_px,
        "rank_ratio_threshold": rank_ratio_threshold,
        "transform_modes": list(transform_modes),
        "predictor_modes": list(predictor_modes),
        "centering_modes": list(centering_modes),
        "basis_names": list(basis_names),
        "window_levels": list(window_levels),
        "model_type": model_type,
        "model_index": model_index,
        "model_device": model_device,
        "max_windows": max_windows,
        "n_split_repeats": n_split_repeats,
    })
    _write_provenance_summary(output_dir)
    _write_readme(output_dir / "README.md", summary_rows, verification_rows, tensor_rows, argparse.Namespace(
        subject=subject,
        date=date,
        dataset_configs_path=dataset_configs_path,
        checkpoint_path=checkpoint_path,
        split_bundle_path=split_bundle_path,
        transform_modes=",".join(transform_modes),
        predictor_modes=",".join(predictor_modes),
        centering_modes=",".join(centering_modes),
        basis_names=",".join(basis_names),
        window_levels=",".join(window_levels),
    ))
    (output_dir / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2) + "\n")
    print(json.dumps(audit_summary, indent=2))
    return audit_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--dataset-configs-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--dataset-idx", type=int, required=True)
    parser.add_argument(
        "--split-bundle-path",
        default=None,
        help="Reference split-bundle provenance path only; this audit regenerates local empirical bundles internally.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-empirical-samples", type=int, default=20)
    parser.add_argument("--n-shuffle-matches", type=int, default=8)
    parser.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    parser.add_argument("--rank-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--transform-modes", default=",".join(DEFAULT_TRANSFORM_MODES))
    parser.add_argument("--predictor-modes", default=",".join(DEFAULT_PREDICTOR_MODES))
    parser.add_argument("--centering-modes", default=",".join(DEFAULT_CENTERING_MODES))
    parser.add_argument("--basis-names", default=",".join(DEFAULT_BASIS_NAMES))
    parser.add_argument("--window-levels", default=",".join(DEFAULT_WINDOW_LEVELS))
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--n-split-repeats", type=int, default=DEFAULT_N_SPLIT_REPEATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_geometry_bridge_audit(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        checkpoint_path=args.checkpoint_path,
        dataset_idx=args.dataset_idx,
        split_bundle_path=args.split_bundle_path,
        output_dir=Path(args.output_dir),
        min_samples=args.min_samples,
        n_shuffle_matches=args.n_shuffle_matches,
        jacobian_step_px=args.jacobian_step_px,
        rank_ratio_threshold=args.rank_ratio_threshold,
        transform_modes=_parse_csv_list(args.transform_modes),
        predictor_modes=_parse_csv_list(args.predictor_modes),
        centering_modes=_parse_csv_list(args.centering_modes),
        basis_names=_parse_csv_list(args.basis_names),
        model_type=args.model_type,
        model_index=args.model_index,
        model_device=args.model_device,
        max_windows=args.max_windows,
        min_empirical_samples=args.min_empirical_samples,
        n_split_repeats=args.n_split_repeats,
        window_levels=_parse_csv_list(args.window_levels),
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
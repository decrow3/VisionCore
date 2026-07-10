#!/usr/bin/env python3
"""Estimate lag-aware Vernier translation geometry from cached runs.

This diagnostic targets the Wu-style correction: for the framewise ConvGRU
path, the response at one output bin depends on a lagged retinal movie, not just
the current eye position.  The script perturbs one lag plane of the embedded
stimulus at a time and measures the local response kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from scripts.temporal_decoding.rate_computation import compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS

from .forward import (
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
    renderer_raw_to_model_pixelnorm,
)
from .joint_observer import build_compact_translation_basis
from .run_vernier_active_sensing import build_spec, parse_csv_float, parse_csv_int, parse_csv_str
from .stimulus import RenderGeometry, render_world, sample_retina_movie


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_label(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(text))


def _rate_cache_path(source_dir: Path, condition: str, fd_step: float) -> Path:
    return source_dir / "cache" / f"rates_{condition}_fd{float(fd_step):.4f}arcmin.npz"


def _instant_cache_path(source_dir: Path, condition: str, trace_idx: int, fd_step: float, eps: float, t: int) -> Path:
    safe = _safe_label(f"{condition}_trace{trace_idx}")
    path = (
        source_dir
        / "cache"
        / "joint_geometry"
        / f"local_translation_jacobian_{safe}_fd{float(fd_step):.4f}arcmin_eps{float(eps):.4f}arcmin_t{int(t)}.npz"
    )
    if path.exists():
        return path
    matches = sorted(
        (source_dir / "cache" / "joint_geometry").glob(
            f"local_translation_jacobian_{safe}_fd{float(fd_step):.4f}arcmin_eps{float(eps):.4f}arcmin_t*.npz"
        )
    )
    if not matches:
        raise FileNotFoundError(f"No instantaneous geometry cache found for {safe} fd={fd_step}")
    return matches[0]


def _padded_eye(reference_trace_deg: np.ndarray, n_lags: int) -> np.ndarray:
    ref = np.asarray(reference_trace_deg, dtype=np.float32)
    return np.concatenate([np.repeat(ref[:1], max(int(n_lags) - 1, 0), axis=0), ref], axis=0)


def build_lag_kernel_for_trace(
    *,
    model: Any,
    readout: Any,
    args: SimpleNamespace,
    out_dir: Path,
    geometry: RenderGeometry,
    condition: str,
    trace_idx: int,
    fd_step_arcmin: float,
    reference_trace_deg: np.ndarray,
    lag_indices: list[int],
) -> dict[str, Any]:
    """Build K[theta, time, lag, unit, xy] for selected lag planes."""
    if str(args.inference_mode) != "framewise":
        raise ValueError("Lag-plane diagnostic currently supports only framewise inference")
    t = int(reference_trace_deg.shape[0])
    eps = float(args.lag_translation_eps_arcmin)
    n_lags = int(N_LAGS)
    lags = [int(lag) for lag in lag_indices]
    invalid_lags = [lag for lag in lags if lag < 0 or lag >= n_lags]
    if invalid_lags:
        raise ValueError(f"Requested lag indices {invalid_lags} are outside [0, {n_lags - 1}]")
    if not lags:
        raise ValueError("At least one lag index is required")
    device = str(args.device or "cpu")
    theta_values = np.asarray([float(fd_step_arcmin), -float(fd_step_arcmin)], dtype=np.float32)
    theta_labels = np.asarray(["plus", "minus"])
    cache_dir = out_dir / "cache" / "lag_geometry"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_label(f"{condition}_trace{trace_idx}")
    cache_path = (
        cache_dir
        / f"lag_translation_kernel_{safe}_fd{float(fd_step_arcmin):.4f}arcmin_eps{eps:.4f}arcmin_lags{'-'.join(map(str, lags))}_t{t}.npz"
    )

    kernels: list[np.ndarray] = []
    mu0_rates: list[np.ndarray] = []
    dxdy = [np.asarray([eps / 60.0, 0.0], dtype=np.float32), np.asarray([0.0, eps / 60.0], dtype=np.float32)]
    for theta in theta_values:
        spec = build_spec(args, float(theta))
        baseline_stim = build_vernier_movie(
            spec,
            reference_trace_deg,
            geometry=geometry,
            n_lags=n_lags,
            device=device,
        )
        baseline_rates = compute_trial_rates(
            model,
            readout,
            baseline_stim,
            batch_size=int(args.batch_size),
            spatial_collapse=str(args.spatial_collapse),
        )[:t]
        mu0_rates.append(baseline_rates.astype(np.float32))
        world = render_world(spec, geometry, device=device)
        padded = _padded_eye(reference_trace_deg, n_lags)
        lag_kernel = np.zeros((t, len(lags), baseline_rates.shape[1], 2), dtype=np.float32)
        for li, lag in enumerate(lags):
            start = n_lags - 1 - int(lag)
            stop = start + t
            for dim, delta in enumerate(dxdy):
                shifted_plus = padded.copy()
                shifted_minus = padded.copy()
                shifted_plus[:, :] += delta[None, :]
                shifted_minus[:, :] -= delta[None, :]
                movie_plus = sample_retina_movie(world, shifted_plus, geometry=geometry, device=device)[0, 0].detach().cpu()
                movie_minus = sample_retina_movie(world, shifted_minus, geometry=geometry, device=device)[0, 0].detach().cpu()
                stim_plus = baseline_stim.clone()
                stim_minus = baseline_stim.clone()
                stim_plus[:, :, lag] = renderer_raw_to_model_pixelnorm(
                    movie_plus[start:stop].unsqueeze(1),
                    max_raw=float(geometry.max_raw),
                )
                stim_minus[:, :, lag] = renderer_raw_to_model_pixelnorm(
                    movie_minus[start:stop].unsqueeze(1),
                    max_raw=float(geometry.max_raw),
                )
                rates_plus = compute_trial_rates(
                    model,
                    readout,
                    stim_plus,
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                )[:t]
                rates_minus = compute_trial_rates(
                    model,
                    readout,
                    stim_minus,
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                )[:t]
                lag_kernel[:, li, :, dim] = ((rates_plus - rates_minus) / (2.0 * eps)).astype(np.float32)
                del stim_plus, stim_minus, rates_plus, rates_minus, movie_plus, movie_minus
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        kernels.append(lag_kernel)

    payload = {
        "path": cache_path,
        "theta_arcmin": theta_values,
        "theta_labels": theta_labels,
        "lag_indices": np.asarray(lags, dtype=np.int32),
        "translation_eps_arcmin": np.asarray([eps], dtype=np.float32),
        "reference_trace_deg": np.asarray(reference_trace_deg, dtype=np.float32),
        "mu0_rates": np.asarray(mu0_rates, dtype=np.float32),
        "lag_kernel_rates_per_arcmin": np.asarray(kernels, dtype=np.float32),
        "stimulus_normalization": np.asarray([STIMULUS_NORMALIZATION]),
    }
    np.savez_compressed(cache_path, **{key: value for key, value in payload.items() if key != "path"})
    return payload


def predict_from_lag_kernel(kernel: np.ndarray, pose_residual_arcmin: np.ndarray, lag_indices: np.ndarray) -> np.ndarray:
    """Predict response residual from a lag kernel and a pose trajectory."""
    k = np.asarray(kernel, dtype=np.float64)
    pose = np.asarray(pose_residual_arcmin, dtype=np.float64)
    lags = np.asarray(lag_indices, dtype=np.int32)
    t = min(k.shape[0], pose.shape[0])
    pred = np.zeros((t, k.shape[2]), dtype=np.float64)
    for li, lag in enumerate(lags):
        source_idx = np.arange(t) - int(lag)
        source_idx[source_idx < 0] = 0
        pred += np.einsum("tud,td->tu", k[:t, li], pose[source_idx])
    return pred


def predict_increment_from_lag_kernel(kernel: np.ndarray, pose_residual_arcmin: np.ndarray, lag_indices: np.ndarray) -> np.ndarray:
    """Predict response residual using recent eye-position increments as the drive."""
    pose = np.asarray(pose_residual_arcmin, dtype=np.float64)
    increments = np.zeros_like(pose)
    if pose.shape[0]:
        increments[0] = pose[0]
        increments[1:] = pose[1:] - pose[:-1]
    return predict_from_lag_kernel(kernel, increments, lag_indices)


def _aligned_arrays(exact: np.ndarray, pred: np.ndarray, shift_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Align exact and predicted arrays for a diagnostic output shift."""
    x = np.asarray(exact)
    y = np.asarray(pred)
    exact_slice, pred_slice = _alignment_slices(x.shape[0], y.shape[0], shift_bins)
    return x[exact_slice], y[pred_slice]


def _alignment_slices(n_exact: int, n_pred: int, shift_bins: int) -> tuple[slice, slice]:
    """Return exact/pred slices for comparing pred shifted relative to exact."""
    shift = int(shift_bins)
    if shift == 0:
        t = min(int(n_exact), int(n_pred))
        return slice(0, t), slice(0, t)
    if shift > 0:
        t = min(int(n_exact) - shift, int(n_pred))
        if t <= 0:
            return slice(0, 0), slice(0, 0)
        return slice(shift, shift + t), slice(0, t)
    offset = -shift
    t = min(int(n_exact), int(n_pred) - offset)
    if t <= 0:
        return slice(0, 0), slice(0, 0)
    return slice(0, t), slice(offset, offset + t)


def _project_time_units(arr: np.ndarray, projector: np.ndarray | None) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if projector is None:
        return values
    return values @ np.asarray(projector, dtype=np.float64)


def _diag_residual_score(
    observed_counts: np.ndarray,
    predicted_counts: np.ndarray,
    covariance_counts: np.ndarray,
    *,
    phi: float,
    epsilon: float = 1e-8,
) -> float:
    """Residual-only diagonal Gaussian score in count space."""
    obs = np.asarray(observed_counts, dtype=np.float64)
    pred = np.asarray(predicted_counts, dtype=np.float64)
    cov_ref = np.asarray(covariance_counts, dtype=np.float64)
    var = np.maximum(float(phi) * np.maximum(cov_ref, 0.0), float(epsilon))
    return float(-0.5 * np.sum(((obs - pred) ** 2) / var))


def _projected_residual_score(
    observed_counts: np.ndarray,
    predicted_counts: np.ndarray,
    covariance_counts: np.ndarray,
    projector: np.ndarray | None,
    *,
    phi: float,
    epsilon: float = 1e-8,
) -> float:
    """Residual-only Gaussian score in full-unit or compact-projected space."""
    obs = np.asarray(observed_counts, dtype=np.float64)
    pred = np.asarray(predicted_counts, dtype=np.float64)
    cov_ref = np.asarray(covariance_counts, dtype=np.float64)
    if projector is None:
        return _diag_residual_score(obs, pred, cov_ref, phi=phi, epsilon=epsilon)
    u = np.asarray(projector, dtype=np.float64)
    residual = (obs - pred) @ u
    var_units = np.maximum(float(phi) * np.maximum(cov_ref, 0.0), float(epsilon))
    cov_z = np.einsum("uk,tu,ul->tkl", u, var_units, u)
    score = 0.0
    eye = np.eye(u.shape[1], dtype=np.float64)
    for ti in range(residual.shape[0]):
        cov = 0.5 * (cov_z[ti] + cov_z[ti].T) + float(epsilon) * eye
        try:
            sol = np.linalg.solve(cov, residual[ti])
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(cov) @ residual[ti]
        score += float(-0.5 * residual[ti] @ sol)
    return float(score)


def diagnostic_rows_for_trace(
    *,
    condition: str,
    fd_step_arcmin: float,
    trace_idx: int,
    plus_rates: np.ndarray,
    minus_rates: np.ndarray,
    pose_trace_deg: np.ndarray,
    lag_payload: dict[str, Any],
    instant_cache: dict[str, np.ndarray] | None,
    bin_seconds: float,
    phi: float,
    time_shift_bins: list[int] | None = None,
    compact_k_list: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lag_kernel = np.asarray(lag_payload["lag_kernel_rates_per_arcmin"], dtype=np.float64)
    mu0 = np.asarray(lag_payload["mu0_rates"], dtype=np.float64)
    ref = np.asarray(lag_payload["reference_trace_deg"], dtype=np.float64)
    lags = np.asarray(lag_payload["lag_indices"], dtype=np.int32)
    pose = (np.asarray(pose_trace_deg[: ref.shape[0]], dtype=np.float64) - ref) * 60.0
    exact_rates = [np.asarray(plus_rates[: ref.shape[0]], dtype=np.float64), np.asarray(minus_rates[: ref.shape[0]], dtype=np.float64)]
    instant = None if instant_cache is None else np.asarray(instant_cache["jacobian_rates_per_arcmin"], dtype=np.float64)
    shifts = [0] if time_shift_bins is None else [int(shift) for shift in time_shift_bins]
    if not shifts:
        shifts = [0]
    mode_predictions: dict[tuple[str, str], list[np.ndarray]] = {
        ("lag_kernel", "position"): [],
        ("lag_kernel", "increment"): [],
    }
    if instant is not None:
        mode_predictions[("instant_chart", "position")] = []
    for ci, label in enumerate(("plus", "minus")):
        exact = exact_rates[ci] - mu0[ci]
        lag_pred = predict_from_lag_kernel(lag_kernel[ci], pose, lags)
        lag_increment_pred = predict_increment_from_lag_kernel(lag_kernel[ci], pose, lags)
        mode_predictions[("lag_kernel", "position")].append(lag_pred)
        mode_predictions[("lag_kernel", "increment")].append(lag_increment_pred)
        pred_pairs = [
            ("lag_kernel", "position", lag_pred),
            ("lag_kernel", "increment", lag_increment_pred),
        ]
        if instant is not None:
            instant_pred = np.einsum("tud,td->tu", instant[ci, : exact.shape[0]], pose[: exact.shape[0]])
            mode_predictions[("instant_chart", "position")].append(instant_pred)
            pred_pairs.append(("instant_chart", "position", instant_pred))
    measurement_spaces: list[tuple[str, int | str, np.ndarray | None]] = [("full_units", "all", None)]
    if compact_k_list and instant is not None:
        for compact_k in compact_k_list:
            if int(compact_k) <= 0:
                continue
            projector = build_compact_translation_basis(instant, compact_k=int(compact_k), control="correct_chart", seed=0)
            measurement_spaces.append((f"compact_k{projector.shape[1]}", int(projector.shape[1]), projector))

    for ci, label in enumerate(("plus", "minus")):
        exact = exact_rates[ci] - mu0[ci]
        for mode_key, preds in mode_predictions.items():
            geometry_mode, drive_mode = mode_key
            pred = preds[ci]
            for measurement_space, compact_k, projector in measurement_spaces:
                exact_proj = _project_time_units(exact, projector)
                pred_proj = _project_time_units(pred, projector)
                for shift in shifts:
                    x_arr, y_arr = _aligned_arrays(exact_proj, pred_proj, shift)
                    x = x_arr.ravel()
                    y = y_arr.ravel()
                    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
                    rows.append(
                        {
                            "condition": condition,
                            "fd_step_arcmin": float(fd_step_arcmin),
                            "trace_index": int(trace_idx),
                            "diagnostic_type": "rate_fidelity",
                            "theta_label": label,
                            "observed_theta_label": label,
                            "scored_theta_label": label,
                            "geometry_mode": geometry_mode,
                            "drive_mode": drive_mode,
                            "measurement_space": measurement_space,
                            "compact_k": compact_k,
                            "time_shift_bins": int(shift),
                            "n_timebins": int(x_arr.shape[0]),
                            "n_units": int(x_arr.shape[1]) if x_arr.ndim == 2 else 0,
                            "lag_indices": ",".join(str(int(lag)) for lag in lags),
                            "corr_exact_vs_pred": corr,
                            "relative_error": float(np.linalg.norm(x - y) / max(float(np.linalg.norm(x)), 1e-12)),
                            "exact_rms": float(np.linalg.norm(x) / np.sqrt(max(x.size, 1))),
                            "pred_rms": float(np.linalg.norm(y) / np.sqrt(max(y.size, 1))),
                            "kernel_cache_path": str(lag_payload["path"]),
                        }
                    )
    labels = ("plus", "minus")
    dt = float(bin_seconds)
    for true_idx, true_label in enumerate(labels):
        for measurement_space, compact_k, projector in measurement_spaces:
            for shift in shifts:
                exact_slice, pred_slice = _alignment_slices(exact_rates[true_idx].shape[0], exact_rates[true_idx].shape[0], shift)
                observed_counts = exact_rates[true_idx][exact_slice] * dt
                exact_scores: dict[str, float] = {}
                zero_scores: dict[str, float] = {}
                model_scores_by_mode: dict[tuple[str, str], dict[str, float]] = {mode: {} for mode in mode_predictions}
                for score_idx, score_label in enumerate(labels):
                    exact_candidate_counts_units = exact_rates[score_idx][exact_slice] * dt
                    zero_candidate_counts_units = mu0[score_idx][exact_slice] * dt
                    exact_score = _projected_residual_score(
                        observed_counts,
                        exact_candidate_counts_units,
                        exact_candidate_counts_units,
                        projector,
                        phi=float(phi),
                    )
                    zero_score = _projected_residual_score(
                        observed_counts,
                        zero_candidate_counts_units,
                        exact_candidate_counts_units,
                        projector,
                        phi=float(phi),
                    )
                    exact_scores[score_label] = exact_score
                    zero_scores[score_label] = zero_score
                    for mode_key, preds in mode_predictions.items():
                        geometry_mode, drive_mode = mode_key
                        pred_counts_units = (mu0[score_idx][exact_slice] + preds[score_idx][pred_slice]) * dt
                        model_score = _projected_residual_score(
                            observed_counts,
                            pred_counts_units,
                            exact_candidate_counts_units,
                            projector,
                            phi=float(phi),
                        )
                        model_scores_by_mode[mode_key][score_label] = model_score
                        rows.append(
                            {
                                "condition": condition,
                                "fd_step_arcmin": float(fd_step_arcmin),
                                "trace_index": int(trace_idx),
                                "diagnostic_type": "likelihood_fidelity",
                                "theta_label": true_label,
                                "observed_theta_label": true_label,
                                "scored_theta_label": score_label,
                                "geometry_mode": geometry_mode,
                                "drive_mode": drive_mode,
                                "measurement_space": measurement_space,
                                "compact_k": compact_k,
                                "time_shift_bins": int(shift),
                                "n_timebins": int(observed_counts.shape[0]),
                                "n_units": int(projector.shape[1]) if projector is not None else int(observed_counts.shape[1]),
                                "lag_indices": ",".join(str(int(lag)) for lag in lags),
                                "exact_known_residual_score": exact_score,
                                "model_known_residual_score": model_score,
                                "zero_eye_residual_score": zero_score,
                                "model_minus_exact_score": float(model_score - exact_score),
                                "model_minus_zero_score": float(model_score - zero_score),
                                "kernel_cache_path": str(lag_payload["path"]),
                            }
                        )
                other_label = "minus" if true_label == "plus" else "plus"
                exact_margin = float(exact_scores[true_label] - exact_scores[other_label])
                zero_margin = float(zero_scores[true_label] - zero_scores[other_label])
                pred_exact = "plus" if exact_scores["plus"] >= exact_scores["minus"] else "minus"
                pred_zero = "plus" if zero_scores["plus"] >= zero_scores["minus"] else "minus"
                exact_vec = np.asarray([exact_scores["plus"], exact_scores["minus"]], dtype=np.float64)
                for mode_key, model_scores in model_scores_by_mode.items():
                    geometry_mode, drive_mode = mode_key
                    model_margin = float(model_scores[true_label] - model_scores[other_label])
                    pred_model = "plus" if model_scores["plus"] >= model_scores["minus"] else "minus"
                    model_vec = np.asarray([model_scores["plus"], model_scores["minus"]], dtype=np.float64)
                    rank_corr = (
                        float(np.corrcoef(exact_vec, model_vec)[0, 1])
                        if np.std(exact_vec) > 1e-12 and np.std(model_vec) > 1e-12
                        else float("nan")
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "fd_step_arcmin": float(fd_step_arcmin),
                            "trace_index": int(trace_idx),
                            "diagnostic_type": "decision_fidelity",
                            "theta_label": true_label,
                            "observed_theta_label": true_label,
                            "geometry_mode": geometry_mode,
                            "drive_mode": drive_mode,
                            "measurement_space": measurement_space,
                            "compact_k": compact_k,
                            "time_shift_bins": int(shift),
                            "n_timebins": int(observed_counts.shape[0]),
                            "n_units": int(projector.shape[1]) if projector is not None else int(observed_counts.shape[1]),
                            "lag_indices": ",".join(str(int(lag)) for lag in lags),
                            "pred_exact_known": pred_exact,
                            "pred_model_known": pred_model,
                            "pred_zero_eye": pred_zero,
                            "model_matches_exact_decision": bool(pred_model == pred_exact),
                            "likelihood_rank_correlation": rank_corr,
                            "exact_known_margin": exact_margin,
                            "model_known_margin": model_margin,
                            "zero_eye_margin": zero_margin,
                            "model_margin_error": float(model_margin - exact_margin),
                            "exact_known_advantage_vs_zero": float(exact_margin - zero_margin),
                            "model_known_advantage_vs_zero": float(model_margin - zero_margin),
                            "kernel_cache_path": str(lag_payload["path"]),
                        }
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=str, default="real_fem")
    parser.add_argument("--fd-steps-arcmin", type=str, default="0.25")
    parser.add_argument("--trace-indices", type=str, default="0")
    parser.add_argument("--lag-indices", type=str, default="0,1,2,4,8,16,31")
    parser.add_argument("--time-shift-bins", type=str, default="0")
    parser.add_argument("--compact-k-list", type=str, default="")
    parser.add_argument("--lag-translation-eps-arcmin", type=float, default=0.25)
    parser.add_argument(
        "--instant-translation-eps-arcmin",
        type=float,
        default=None,
        help="Finite-difference epsilon used by the source instantaneous local-translation cache.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--spatial-collapse", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    manifest = _read_json(source_dir / "vernier_active_sensing_manifest.json")
    source_args = dict(manifest.get("args", {}))
    for key, value in vars(args).items():
        if value is not None:
            source_args[key] = value
    source_args.setdefault("inference_mode", "framewise")
    source_args.setdefault("spatial_collapse", "max")
    source_args["device"] = args.device or source_args.get("device") or "cpu"
    source_args["batch_size"] = int(args.batch_size)
    source_args["lag_translation_eps_arcmin"] = float(args.lag_translation_eps_arcmin)
    run_args = SimpleNamespace(**source_args)
    if str(run_args.inference_mode) != "framewise":
        raise ValueError("Lag geometry diagnostic currently requires framewise source caches")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = parse_csv_str(args.conditions)
    fd_steps = parse_csv_float(args.fd_steps_arcmin)
    trace_indices = parse_csv_int(args.trace_indices)
    lag_indices = parse_csv_int(args.lag_indices)
    time_shift_bins = parse_csv_int(args.time_shift_bins)
    compact_k_list = parse_csv_int(args.compact_k_list) if str(args.compact_k_list).strip() else []
    geometry = RenderGeometry()
    model, readout = load_model_and_readout(str(run_args.device))
    rows: list[dict[str, Any]] = []
    for fd_step in fd_steps:
        for condition in conditions:
            with np.load(_rate_cache_path(source_dir, condition, fd_step), allow_pickle=True) as rate_npz:
                plus = np.asarray(rate_npz["plus"], dtype=np.float32)
                minus = np.asarray(rate_npz["minus"], dtype=np.float32)
                poses = np.asarray(rate_npz["poses"], dtype=np.float32)
                lengths = np.asarray(rate_npz["lengths"], dtype=np.int32)
            for trace_idx in trace_indices:
                t = int(lengths[int(trace_idx)])
                instant_eps_arg = getattr(run_args, "instant_translation_eps_arcmin", None)
                instant_eps = float(
                    instant_eps_arg
                    if instant_eps_arg is not None
                    else getattr(
                        run_args,
                        "joint_translation_eps_arcmin",
                        getattr(run_args, "lag_translation_eps_arcmin", args.lag_translation_eps_arcmin),
                    )
                )
                instant_path = _instant_cache_path(
                    source_dir,
                    condition,
                    int(trace_idx),
                    fd_step,
                    instant_eps,
                    t,
                )
                with np.load(instant_path, allow_pickle=True) as instant_npz:
                    reference = np.asarray(instant_npz["reference_trace_deg"], dtype=np.float32)[:t]
                    instant_cache = {
                        "jacobian_rates_per_arcmin": np.asarray(
                            instant_npz["jacobian_rates_per_arcmin"], dtype=np.float32
                        )[:, :t],
                    }
                print(
                    f"condition={condition} fd={fd_step:g} trace={trace_idx} "
                    f"lags={','.join(map(str, lag_indices))}",
                    flush=True,
                )
                lag_payload = build_lag_kernel_for_trace(
                    model=model,
                    readout=readout,
                    args=run_args,
                    out_dir=out_dir,
                    geometry=geometry,
                    condition=condition,
                    trace_idx=int(trace_idx),
                    fd_step_arcmin=float(fd_step),
                    reference_trace_deg=reference,
                    lag_indices=lag_indices,
                )
                rows.extend(
                    diagnostic_rows_for_trace(
                        condition=condition,
                        fd_step_arcmin=float(fd_step),
                        trace_idx=int(trace_idx),
                        plus_rates=plus[int(trace_idx), :t],
                        minus_rates=minus[int(trace_idx), :t],
                        pose_trace_deg=poses[int(trace_idx), :t],
                        lag_payload=lag_payload,
                        instant_cache=instant_cache,
                        bin_seconds=float(getattr(run_args, "bin_seconds", 1.0)),
                        phi=float(getattr(run_args, "phi", 1.0)),
                        time_shift_bins=time_shift_bins,
                        compact_k_list=compact_k_list,
                    )
                )
    _write_csv(out_dir / "lag_geometry_diagnostic.csv", rows)
    with (out_dir / "lag_geometry_diagnostic_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_dir": str(source_dir),
                "args": _json_ready(vars(args)),
                "instant_translation_eps_arcmin": float(
                    getattr(run_args, "instant_translation_eps_arcmin", None)
                    if getattr(run_args, "instant_translation_eps_arcmin", None) is not None
                    else getattr(
                        run_args,
                        "joint_translation_eps_arcmin",
                        getattr(run_args, "lag_translation_eps_arcmin", args.lag_translation_eps_arcmin),
                    )
                ),
                "n_rows": len(rows),
                "stimulus_normalization": STIMULUS_NORMALIZATION,
                "note": (
                    "Lag-plane Jacobian diagnostic for framewise Vernier observer; "
                    "this is not yet the production joint trajectory filter."
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote lag geometry diagnostic to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

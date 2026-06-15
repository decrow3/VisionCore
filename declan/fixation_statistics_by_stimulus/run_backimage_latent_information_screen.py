"""Minimal BackImage latent-information screen for free-viewing FEMs.

This first-pass runner tests the `I_z(tau)` plan with external feature latents.
For each reliable BackImage window it extracts Gabor-like and DCT latents from
the local image patch, renders matched counterfactual motion templates along a
small set of axes, runs the V1 digital twin, and asks whether the response
decodes the latent better than static or random-axis controls.

The output is intentionally a screen, not the final information estimator:
cross-validated ridge `R2` and per-window cross-validated negative MSE are used
as information proxies. The script reports absolute scores plus deltas versus
static and random-axis matched controls, separately for pose-aware flattened
responses and pose-blind time-averaged responses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_twin_drift_geometry import (
        TwinScorer,
        _clip_patch,
        _cos2,
        _load_twin_common,
        _standardize_uint_like,
        _trace_xy_to_twin_helper_order,
    )
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
        TwinScorer,
        _clip_patch,
        _cos2,
        _load_twin_common,
        _standardize_uint_like,
        _trace_xy_to_twin_helper_order,
    )

try:
    from jake.twininfo.retinal_examples import _padded_even_patch, _patch_to_tensor, _steerable_pyramid

    HAVE_STEERABLE_PYRAMID = True
except Exception:  # pragma: no cover - optional dependency/fallback path
    HAVE_STEERABLE_PYRAMID = False


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_latent_information_screen_pyramid_gabor_dct_n256"
)


@dataclass(frozen=True)
class RunConfig:
    input: str
    out_dir: str
    max_windows: int
    window_manifest: str | None
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    min_duration_s: float
    patch_size_px: int
    min_patch_image_margin_px: float
    latent_crop_px: int
    center_crop_px: int
    local_field_grid: int
    n_timepoints: int
    observed_rms_scale: float
    observed_rms_scales: list[float]
    absolute_rms_arcmin: list[float]
    min_rms_deg: float
    max_rms_deg: float
    max_observed_rms_deg: float | None
    random_axes_per_window: int
    include_fixed_grid: bool
    fixed_grid_step_deg: float
    candidate_groups: list[str]
    observers: list[str]
    latent_names: list[str]
    pca_k_list: list[int]
    ridge_alphas: list[float]
    ridge_alpha_mode: str
    fixed_ridge_alpha: float | None
    outer_folds: int
    inner_folds: int
    n_shuffle_nulls: int
    n_session_bootstrap: int
    population_mode: str
    twin_population_n: int
    twin_batch_size: int
    twin_trace_batch_size: int
    cuda_empty_cache_every_batch: bool
    check_trace_batch_equivalence: bool
    device: str
    progress_every: int
    seed: int


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _progress(message: str) -> None:
    print(f"[backimage-latent-screen] {message}", flush=True)


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_str_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _trace_cache_key(trace: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    arr = np.ascontiguousarray(np.asarray(trace, dtype=np.float32))
    return tuple(arr.shape), arr.tobytes()


def _first_close_match(work: pd.DataFrame, manifest_row: pd.Series) -> int | None:
    mask = np.ones(work.shape[0], dtype=bool)
    if "session" in manifest_row.index and "session" in work.columns:
        mask &= work["session"].astype(str).to_numpy() == str(manifest_row["session"])
    if "trial_idx" in manifest_row.index and "trial_idx" in work.columns:
        mask &= work["trial_idx"].astype(int).to_numpy() == int(manifest_row["trial_idx"])
    float_pairs = (
        ("real_drift_axis_deg", "drift_orientation_deg"),
        ("edge_axis_deg", "image_edge_axis_deg"),
        ("observed_rms_radius_deg", "rms_radius_deg"),
    )
    for manifest_col, work_col in float_pairs:
        if manifest_col in manifest_row.index and work_col in work.columns and pd.notna(manifest_row[manifest_col]):
            mask &= np.isclose(
                work[work_col].astype(float).to_numpy(),
                float(manifest_row[manifest_col]),
                rtol=1e-7,
                atol=1e-7,
                equal_nan=True,
            )
    matches = np.flatnonzero(mask)
    if matches.size == 0:
        return None
    if matches.size > 1 and "window_id" in manifest_row.index and "window_id" in work.columns:
        window_mask = work.iloc[matches]["window_id"].astype(int).to_numpy() == int(manifest_row["window_id"])
        if np.count_nonzero(window_mask) == 1:
            return int(matches[np.flatnonzero(window_mask)[0]])
    if matches.size > 1:
        raise ValueError(
            "Ambiguous --window-manifest row without source_row; multiple rows match "
            f"session={manifest_row.get('session', '')!r}, trial_idx={manifest_row.get('trial_idx', '')!r}."
        )
    return int(matches[0])


def _scale_token(value: float) -> str:
    token = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return token


def _central_crop(image: np.ndarray, size_px: int) -> np.ndarray:
    size_px = int(size_px)
    if size_px % 2 == 0:
        size_px += 1
    half = size_px // 2
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    return image[cy - half : cy + half + 1, cx - half : cx + half + 1]


def _zscore_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    image = image - float(np.nanmean(image))
    sd = float(np.nanstd(image))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(image, dtype=np.float64)
    return image / sd


def _resize_to_square(image: np.ndarray, size_px: int = 64) -> np.ndarray:
    from PIL import Image as PILImage

    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0
    resized = PILImage.fromarray(image.astype(np.float32), mode="F").resize((int(size_px), int(size_px)), resample=2)
    return np.asarray(resized, dtype=np.float64)


def _dct_features(crop: np.ndarray, *, n_freq: int) -> np.ndarray:
    small = _zscore_image(_resize_to_square(crop, size_px=64))
    coeff = fft.dctn(small, type=2, norm="ortho")
    block = coeff[: int(n_freq), : int(n_freq)].copy()
    block[0, 0] = 0.0
    return block.reshape(-1).astype(np.float64)


def _gabor_kernel(size: int, frequency: float, theta_deg: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    half = int(size) // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    theta = np.radians(float(theta_deg))
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(xr**2 + yr**2) / (2.0 * float(sigma) ** 2))
    phase = 2.0 * np.pi * float(frequency) * xr
    even = envelope * np.cos(phase)
    odd = envelope * np.sin(phase)
    even -= float(np.mean(even))
    odd -= float(np.mean(odd))
    even /= float(np.sqrt(np.sum(even * even)) + 1e-12)
    odd /= float(np.sqrt(np.sum(odd * odd)) + 1e-12)
    return even, odd


def _block_means(image: np.ndarray, grid: int) -> np.ndarray:
    h, w = image.shape
    rows = np.array_split(np.arange(h), int(grid))
    cols = np.array_split(np.arange(w), int(grid))
    out = []
    for rr in rows:
        for cc in cols:
            out.append(float(np.nanmean(image[np.ix_(rr, cc)])))
    return np.asarray(out, dtype=np.float64)


def _gabor_features(crop: np.ndarray, *, scope: str, local_grid: int) -> np.ndarray:
    image = _zscore_image(_resize_to_square(crop, size_px=64))
    orientations = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
    freqs = (0.06, 0.11, 0.18)
    feats: list[np.ndarray] = []
    for freq in freqs:
        sigma = max(3.0, 0.55 / float(freq))
        kernel_size = int(min(31, max(9, 2 * round(3.0 * sigma) + 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        for theta in orientations:
            even, odd = _gabor_kernel(kernel_size, freq, theta, sigma=min(sigma, 8.0))
            ev = convolve(image, even, mode="nearest")
            od = convolve(image, odd, mode="nearest")
            amp = np.sqrt(ev * ev + od * od)
            if scope == "center":
                cy, cx = amp.shape[0] // 2, amp.shape[1] // 2
                feats.append(np.asarray([float(ev[cy, cx]), float(od[cy, cx]), float(amp[cy, cx])], dtype=np.float64))
            else:
                feats.extend((_block_means(ev, grid=local_grid), _block_means(od, grid=local_grid), _block_means(amp, grid=local_grid)))
    return np.concatenate(feats).astype(np.float64)


def _tensor_to_numpy(value) -> np.ndarray:
    return value.detach().cpu().squeeze().numpy()


def _pyramid_features(crop: np.ndarray, *, scope: str, local_grid: int, height: int = 4, order: int = 3) -> np.ndarray:
    if not HAVE_STEERABLE_PYRAMID:
        return np.empty(0, dtype=np.float64)
    image = _zscore_image(_resize_to_square(crop, size_px=128)).astype(np.float32)
    patch, _padding = _padded_even_patch(image)
    pyr = _steerable_pyramid(patch.shape, height=int(height), order=int(order))
    coeffs = pyr(_patch_to_tensor(patch))
    feats: list[np.ndarray] = []
    for key, value in coeffs.items():
        arr = _tensor_to_numpy(value)
        if not np.iscomplexobj(arr):
            continue
        # plenoptic 1.4 shape after squeeze is (orient, H, W). Older versions
        # may return a single orientation as (H, W), so normalize the rank.
        if arr.ndim == 2:
            arr = arr[None, :, :]
        for orient_idx in range(arr.shape[0]):
            plane = arr[orient_idx]
            real = np.real(plane)
            imag = np.imag(plane)
            mag = np.abs(plane)
            if scope == "center":
                cy, cx = real.shape[0] // 2, real.shape[1] // 2
                feats.append(
                    np.asarray(
                        [
                            float(real[cy, cx]),
                            float(imag[cy, cx]),
                            float(mag[cy, cx]),
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                feats.extend((_block_means(real, grid=local_grid), _block_means(imag, grid=local_grid), _block_means(mag, grid=local_grid)))
    if not feats:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(feats).astype(np.float64)


def _extract_latents(patch: np.ndarray, *, latent_crop_px: int, center_crop_px: int, local_field_grid: int) -> dict[str, np.ndarray]:
    image = _standardize_uint_like(patch)
    field_crop = _central_crop(image, int(latent_crop_px))
    center_crop = _central_crop(image, int(center_crop_px))
    out = {
        "dct_center": _dct_features(center_crop, n_freq=8),
        "dct_local_field": _dct_features(field_crop, n_freq=8),
        "gabor_center": _gabor_features(center_crop, scope="center", local_grid=int(local_field_grid)),
        "gabor_local_field": _gabor_features(field_crop, scope="local_field", local_grid=int(local_field_grid)),
    }
    if HAVE_STEERABLE_PYRAMID:
        out["pyramid_center"] = _pyramid_features(center_crop, scope="center", local_grid=int(local_field_grid))
        out["pyramid_local_field"] = _pyramid_features(field_crop, scope="local_field", local_grid=int(local_field_grid))
    return {key: value for key, value in out.items() if value.size > 0}



def _axis_trace(axis_deg: float, rms_radius_deg: float, n_timepoints: int) -> np.ndarray:
    theta = np.radians(float(axis_deg))
    amp = float(rms_radius_deg) * np.sqrt(2.0)
    t = np.linspace(0.0, 2.0 * np.pi, int(n_timepoints), endpoint=False)
    trace = amp * np.sin(t)[:, None] * np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)[None, :]
    trace -= trace.mean(axis=0, keepdims=True)
    return trace.astype(np.float32)


def _static_trace(n_timepoints: int) -> np.ndarray:
    return np.zeros((int(n_timepoints), 2), dtype=np.float32)


def _path_length(trace: np.ndarray) -> float:
    if trace.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))


def _trace_rms(trace: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(trace, dtype=np.float64) ** 2, axis=1))))


def _trace_extent(trace: np.ndarray, axis_deg: float) -> float:
    theta = np.radians(float(axis_deg))
    u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    projected = np.asarray(trace, dtype=np.float64) @ u
    return float(np.nanmax(projected) - np.nanmin(projected))


def _observed_rms_deg(row: pd.Series) -> float:
    rms = float(row.get("rms_radius_deg", np.nan))
    if not np.isfinite(rms):
        rms = float(row.get("median_radius_deg", np.nan))
    if not np.isfinite(rms):
        rms = 0.03
    return rms


def _motion_scale_specs(row: pd.Series, args: argparse.Namespace) -> list[dict[str, Any]]:
    observed_rms = _observed_rms_deg(row)
    relative_scales = _parse_float_list(args.observed_rms_scales) if args.observed_rms_scales else [float(args.observed_rms_scale)]
    absolute_arcmin = _parse_float_list(args.absolute_rms_arcmin) if args.absolute_rms_arcmin else []
    specs: list[dict[str, Any]] = []
    for scale in relative_scales:
        raw_rms = float(scale) * observed_rms
        rms = 0.0 if raw_rms <= 0.0 else float(np.clip(raw_rms, float(args.min_rms_deg), float(args.max_rms_deg)))
        specs.append(
            {
                "motion_scale_id": f"rel_{_scale_token(scale)}x",
                "motion_scale_kind": "relative_observed_rms",
                "motion_scale_value": float(scale),
                "motion_scale_label": f"{float(scale):g}x observed RMS",
                "nominal_observed_rms_scale": float(scale),
                "nominal_absolute_rms_deg": float("nan"),
                "raw_rms_radius_deg": float(raw_rms),
                "rms_clipped_low": bool(raw_rms > 0.0 and raw_rms < float(args.min_rms_deg)),
                "rms_clipped_high": bool(raw_rms > float(args.max_rms_deg)),
                "rms_radius_deg": rms,
            }
        )
    for arcmin in absolute_arcmin:
        raw_rms = float(arcmin) / 60.0
        rms = 0.0 if raw_rms <= 0.0 else float(np.clip(raw_rms, float(args.min_rms_deg), float(args.max_rms_deg)))
        specs.append(
            {
                "motion_scale_id": f"abs_{_scale_token(arcmin)}arcmin",
                "motion_scale_kind": "absolute_rms_arcmin",
                "motion_scale_value": float(arcmin),
                "motion_scale_label": f"{float(arcmin):g} arcmin RMS",
                "nominal_observed_rms_scale": float(raw_rms / observed_rms) if observed_rms > 0 else float("nan"),
                "nominal_absolute_rms_deg": float(raw_rms),
                "raw_rms_radius_deg": float(raw_rms),
                "rms_clipped_low": bool(raw_rms > 0.0 and raw_rms < float(args.min_rms_deg)),
                "rms_clipped_high": bool(raw_rms > float(args.max_rms_deg)),
                "rms_radius_deg": rms,
            }
        )
    return specs


def _candidate_specs(row: pd.Series, *, rng: np.random.Generator, args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = set(_parse_str_list(args.candidate_groups))
    if "all" in groups:
        groups = {"static", "real", "edge", "edge_orthogonal", "spectrum", "random", "grid"}
    axis_specs: list[dict[str, Any]] = []
    if "static" in groups:
        axis_specs.append({"candidate": "static", "axis_deg": np.nan, "trace": _static_trace(int(args.n_timepoints))})
    if "edge" in groups:
        axis_specs.append({"candidate": "edge", "axis_deg": float(row["image_edge_axis_deg"])})
    if "edge_orthogonal" in groups:
        axis_specs.append({"candidate": "edge_orthogonal", "axis_deg": float(row["image_edge_axis_deg"]) + 90.0})
    if "real" in groups:
        axis_specs.append({"candidate": "real_drift_axis", "axis_deg": float(row["drift_orientation_deg"])})
    if "spectrum" in groups:
        axis_specs.append({"candidate": "spectrum", "axis_deg": float(row["image_spectrum_orientation_deg"])})
    if "random" in groups:
        for j in range(int(args.random_axes_per_window)):
            axis_specs.append({"candidate": f"random_axis_{j}", "axis_deg": float(rng.uniform(0.0, 180.0))})
    if "grid" in groups and bool(args.include_fixed_grid):
        for axis in np.arange(0.0, 180.0, float(args.fixed_grid_step_deg)):
            axis_specs.append({"candidate": f"grid_{axis:g}", "axis_deg": float(axis)})
    if not axis_specs:
        raise ValueError(f"No candidate axes requested by --candidate-groups={args.candidate_groups!r}")
    specs: list[dict[str, Any]] = []
    for scale in _motion_scale_specs(row, args):
        for axis_spec in axis_specs:
            spec = dict(axis_spec)
            rms = float(scale["rms_radius_deg"])
            if spec["candidate"] == "static":
                trace = _static_trace(int(args.n_timepoints))
            else:
                trace = _axis_trace(float(spec["axis_deg"]), rms, int(args.n_timepoints))
            axis = float(spec["axis_deg"]) if np.isfinite(spec["axis_deg"]) else 0.0
            spec["trace"] = trace
            spec["amplitude_type"] = "RMS radius"
            spec["amplitude_value_deg"] = rms if spec["candidate"] != "static" else 0.0
            spec["observed_rms_scale"] = float(scale["nominal_observed_rms_scale"])
            spec["motion_scale_id"] = str(scale["motion_scale_id"])
            spec["motion_scale_kind"] = str(scale["motion_scale_kind"])
            spec["motion_scale_value"] = float(scale["motion_scale_value"])
            spec["motion_scale_label"] = str(scale["motion_scale_label"])
            spec["nominal_absolute_rms_deg"] = float(scale["nominal_absolute_rms_deg"])
            spec["raw_rms_radius_deg"] = float(scale["raw_rms_radius_deg"])
            spec["rms_clipped_low"] = bool(scale["rms_clipped_low"])
            spec["rms_clipped_high"] = bool(scale["rms_clipped_high"])
            spec["duration_s"] = float(row.get("duration_s", np.nan))
            spec["template_shape"] = "sinusoidal_line" if spec["candidate"] != "static" else "static_center"
            spec["n_timepoints"] = int(args.n_timepoints)
            spec["path_length_deg"] = _path_length(trace)
            spec["rms_radius_deg"] = _trace_rms(trace)
            spec["endpoint_extent_deg"] = _trace_extent(trace, axis)
            specs.append(spec)
    return specs


def _align_response_to_trace(response: np.ndarray, n_timepoints: int) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    n_timepoints = int(n_timepoints)
    if response.shape[0] == n_timepoints:
        return response
    if response.shape[0] == n_timepoints + 1:
        return response[1 : n_timepoints + 1]
    raise ValueError(
        f"Twin response has {response.shape[0]} frames for a {n_timepoints}-sample trace; "
        "expected T or T+1 frames."
    )


def _response_features(response: np.ndarray, *, static_response: np.ndarray | None = None) -> dict[str, np.ndarray]:
    response = np.asarray(response, dtype=np.float32)
    out = {
        "pose_aware_flat": response.reshape(-1).astype(np.float32),
        "pose_blind_mean": np.mean(response, axis=0).astype(np.float32),
    }
    if static_response is not None:
        static_response = np.asarray(static_response, dtype=np.float32)
        if static_response.shape != response.shape:
            raise ValueError(f"Static response shape {static_response.shape} does not match response shape {response.shape}")
        delta = response - static_response
        out.update(
            {
                "pose_aware_delta_flat": delta.reshape(-1).astype(np.float32),
                "pose_blind_delta_mean": np.mean(delta, axis=0).astype(np.float32),
            }
        )
    return out


class CanonicalTwinScorer:
    """Canonical 756-unit shared readout used by the TFTS/NITS analyses."""

    def __init__(self, *, device: str, batch_size: int, empty_cache_every_batch: bool = False):
        import torch

        from declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure import _load_twin_context

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from scripts.spatial_info import compute_rate_map

        dev = "cuda:0" if str(device) == "auto" else str(device)
        self.ctx = _load_twin_context(model_device=dev)
        self.common = _load_twin_common()
        self.batch_size = int(batch_size)
        self.empty_cache_every_batch = bool(empty_cache_every_batch)
        self.torch = torch
        self.device = dev
        self.n_units = int(self.ctx.n_units)
        self.population_source = "canonical_shared_population_readout"
        self.model_family = "tfts_nits_standard_canonical_shared_readout"
        self.model_names = [str(name) for name in self.ctx.model.names]
        self.compute_rate_map = compute_rate_map

    def response(self, patch: np.ndarray, trace: np.ndarray) -> np.ndarray:
        return self.responses(patch, [trace], trace_batch_size=1)[0]

    def _compute_rate_map_batched(self, stim: Any):
        device = next(self.ctx.model.model.parameters()).device
        y_chunks = []
        self.ctx.model.model.eval()
        self.ctx.readout.eval()
        with self.torch.no_grad():
            for t_start in range(0, int(stim.shape[0]), self.batch_size):
                t_end = min(t_start + self.batch_size, int(stim.shape[0]))
                x = stim[t_start:t_end].to(device)
                y_batch = self.compute_rate_map(self.ctx.model, self.ctx.readout, x)
                y_chunks.append(y_batch.cpu())
                del x, y_batch
                if self.empty_cache_every_batch and str(device).startswith("cuda"):
                    self.torch.cuda.empty_cache()
        return self.torch.cat(y_chunks, dim=0)

    def responses(self, patch: np.ndarray, traces: list[np.ndarray], *, trace_batch_size: int = 1) -> list[np.ndarray]:
        if not traces:
            return []
        image = _standardize_uint_like(patch)
        trace_batch_size = max(1, int(trace_batch_size))
        out: list[np.ndarray] = []
        for start in range(0, len(traces), trace_batch_size):
            trace_chunk = traces[start : start + trace_batch_size]
            stims = []
            lengths = []
            for trace in trace_chunk:
                trace = np.asarray(trace, dtype=np.float32)
                full_stack = np.broadcast_to(
                    image[None, :, :],
                    (trace.shape[0] + self.common.N_LAGS + 1, *image.shape),
                ).copy()
                eye = self.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
                stim = self.common.make_counterfactual_stim(
                    full_stack,
                    eye,
                    ppd=self.common.PPD,
                    scale_factor=1.0,
                    n_lags=self.common.N_LAGS,
                    out_size=self.common.OUT_SIZE,
                )
                stims.append((stim - 127.0) / 255.0)
                lengths.append(int(stim.shape[0]))
            rate_map = self._compute_rate_map_batched(self.torch.cat(stims, dim=0))
            rates = rate_map.amax(dim=(-2, -1)).detach().cpu().numpy().astype(np.float32, copy=False)
            offset = 0
            for length in lengths:
                out.append(rates[offset : offset + length])
                offset += length
            del stims, rate_map, rates
        return out


def _make_twin_scorer(args: argparse.Namespace):
    mode = str(args.population_mode)
    if mode == "sampled":
        scorer = TwinScorer(
            device=str(args.device),
            population_n=int(args.twin_population_n),
            batch_size=int(args.twin_batch_size),
            seed=int(args.seed),
        )
        scorer.population_source = "ryan_reliable_unit_grid_sample"
        scorer.model_family = "digital_twin_120_sampled_reliable_grid"
        scorer.model_names = [str(name) for name in scorer.model.names]
        return scorer
    if mode == "canonical":
        return CanonicalTwinScorer(
            device=str(args.device),
            batch_size=int(args.twin_batch_size),
            empty_cache_every_batch=bool(args.cuda_empty_cache_every_batch),
        )
    raise ValueError(f"Unknown population_mode={mode!r}")


def _check_trace_batch_equivalence(
    scorer: Any,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
    n_timepoints: int,
) -> None:
    if not hasattr(scorer, "responses") or int(trace_batch_size) <= 1 or not traces:
        return
    sample = traces[: min(4, len(traces))]
    single = [
        _align_response_to_trace(resp, int(n_timepoints))
        for resp in scorer.responses(patch, sample, trace_batch_size=1)
    ]
    batched = [
        _align_response_to_trace(resp, int(n_timepoints))
        for resp in scorer.responses(patch, sample, trace_batch_size=int(trace_batch_size))
    ]
    max_abs = 0.0
    for one, many in zip(single, batched, strict=True):
        if one.shape != many.shape:
            raise ValueError(f"Trace-batch equivalence failed: response shape {one.shape} != {many.shape}")
        max_abs = max(max_abs, float(np.nanmax(np.abs(one - many))))
    if max_abs > 1e-5:
        raise ValueError(f"Trace-batch equivalence failed: max_abs_diff={max_abs:.6g} > 1e-5")
    _progress(f"trace-batch equivalence passed for {len(sample)} traces; max_abs_diff={max_abs:.3g}")


def _split_outer(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size >= 2:
        n = min(int(n_splits), unique.size)
        if n >= 2:
            return list(GroupKFold(n_splits=n).split(np.zeros(groups.size), groups=groups))
    n = min(int(n_splits), groups.size)
    if n < 2:
        return [(np.arange(groups.size), np.arange(groups.size))]
    return list(KFold(n_splits=n, shuffle=True, random_state=int(seed)).split(np.zeros(groups.size)))


def _standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    sd = np.nanstd(train, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (train - mean) / sd, (test - mean) / sd


def _mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=0)
    valid = ss_tot > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(1.0 - ss_res[valid] / ss_tot[valid]))


def _choose_alpha(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: list[float],
    inner_folds: int,
    seed: int,
) -> float:
    if X.shape[0] < 6:
        return float(alphas[len(alphas) // 2])
    splits = _split_outer(groups, int(inner_folds), int(seed))
    scores = []
    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in splits:
            if np.intersect1d(train_idx, val_idx).size:
                continue
            X_train, X_val = _standardize_train_test(X[train_idx], X[val_idx])
            Y_train, Y_val = _standardize_train_test(Y[train_idx], Y[val_idx])
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(X_train, Y_train)
            fold_scores.append(_mean_r2(Y_val, model.predict(X_val)))
        scores.append(float(np.nanmean(fold_scores)) if fold_scores else float("-inf"))
    return float(alphas[int(np.nanargmax(scores))])


def _cross_validated_decode(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    alphas: list[float],
    alpha_mode: str,
    fixed_alpha: float | None,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    groups = np.asarray(groups)
    n = X.shape[0]
    k_eff = int(min(int(k), Z.shape[1], max(1, n - 2)))
    pred = np.full((n, k_eff), np.nan, dtype=np.float64)
    target = np.full((n, k_eff), np.nan, dtype=np.float64)
    chosen_alphas = []
    fold_r2s = []
    splits = _split_outer(groups, int(outer_folds), int(seed))
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train_raw, X_test = _standardize_train_test(X[train_idx], X[test_idx])
        Z_train_raw, Z_test_raw = _standardize_train_test(Z[train_idx], Z[test_idx])
        pca = PCA(n_components=k_eff, svd_solver="full")
        Y_train = pca.fit_transform(Z_train_raw)
        Y_test = pca.transform(Z_test_raw)
        if alpha_mode == "fixed":
            alpha = float(fixed_alpha) if fixed_alpha is not None else float(alphas[len(alphas) // 2])
        elif alpha_mode == "nested_per_candidate":
            alpha = _choose_alpha(
                X_train_raw,
                Y_train,
                groups[train_idx],
                alphas=alphas,
                inner_folds=int(inner_folds),
                seed=int(seed) + fold + 1,
            )
        else:
            raise ValueError(f"Unknown alpha_mode={alpha_mode!r}")
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_raw, Y_train)
        Y_pred = model.predict(X_test)
        fold_r2s.append(_mean_r2(Y_test, Y_pred))
        pred[test_idx] = Y_pred
        target[test_idx] = Y_test
        chosen_alphas.append(alpha)
    mse = np.mean((target - pred) ** 2, axis=1)
    valid_fold_r2s = [float(v) for v in fold_r2s if np.isfinite(v)]
    return {
        "r2": float(np.mean(valid_fold_r2s)) if valid_fold_r2s else float("nan"),
        "mean_neg_mse": float(np.nanmean(-mse)),
        "per_window_score": -mse,
        "chosen_alpha_median": float(np.nanmedian(chosen_alphas)) if chosen_alphas else float("nan"),
        "ridge_alpha_mode": str(alpha_mode),
        "target_dim": k_eff,
        "r2_method": "mean_outer_fold_r2_in_fold_pca_basis",
    }


def _demean_within_session(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    means = series.groupby(pd.Series(sessions)).transform("mean")
    return (series - means).to_numpy(dtype=np.float64)


def _ols_coef_delta_r2(y: np.ndarray, predictor: np.ndarray, controls: np.ndarray) -> tuple[float, float, float, float]:
    ok = np.isfinite(y) & np.isfinite(predictor) & np.all(np.isfinite(controls), axis=1)
    y = np.asarray(y[ok], dtype=np.float64)
    predictor = np.asarray(predictor[ok], dtype=np.float64)
    controls = np.asarray(controls[ok], dtype=np.float64)
    if y.size <= controls.shape[1] + 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)

    def std_cols(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64).copy()
        for j in range(A.shape[1]):
            A[:, j] = (A[:, j] - np.mean(A[:, j])) / (np.std(A[:, j]) + 1e-12)
        return A

    X0 = np.column_stack([np.ones(y.size), std_cols(controls)])
    pred_col = (predictor - np.mean(predictor)) / (np.std(predictor) + 1e-12)
    X1 = np.column_stack([X0, pred_col])
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)

    def r2(X: np.ndarray, beta: np.ndarray) -> float:
        yh = X @ beta
        ss_res = float(np.sum((y - yh) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_control = r2(X0, beta0)
    r2_full = r2(X1, beta1)
    return float(beta1[-1]), float(r2_full - r2_control), float(r2_full), float(r2_control)


def _shuffle_pvalues(
    y: np.ndarray,
    predictor: np.ndarray,
    controls: np.ndarray,
    sessions: np.ndarray,
    *,
    rng: np.random.Generator,
    n_shuffle: int,
) -> tuple[float, float, float, float]:
    obs_coef, obs_delta, _, _ = _ols_coef_delta_r2(y, predictor, controls)
    if int(n_shuffle) <= 0:
        return obs_coef, obs_delta, float("nan"), float("nan")
    null_coef = np.empty(int(n_shuffle), dtype=np.float64)
    null_delta = np.empty(int(n_shuffle), dtype=np.float64)
    sessions = np.asarray(sessions)
    for j in range(int(n_shuffle)):
        shuf = np.asarray(predictor).copy()
        for sess in np.unique(sessions):
            idx = np.flatnonzero(sessions == sess)
            if idx.size > 1:
                shuf[idx] = rng.permutation(shuf[idx])
        null_coef[j], null_delta[j], _, _ = _ols_coef_delta_r2(y, shuf, controls)
    p_coef = (1.0 + float(np.count_nonzero(null_coef >= obs_coef))) / (float(n_shuffle) + 1.0)
    p_delta = (1.0 + float(np.count_nonzero(null_delta >= obs_delta))) / (float(n_shuffle) + 1.0)
    return obs_coef, obs_delta, p_coef, p_delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument(
        "--window-manifest",
        type=Path,
        default=None,
        help=(
            "Optional analysis_windows.csv from a previous run. When provided, windows are replayed "
            "by window_id in manifest order and --max-windows sampling is skipped."
        ),
    )
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument(
        "--local-field-grid",
        type=int,
        default=8,
        help="Spatial pooling grid for local-field Gabor/pyramid maps. Previous runs effectively used 4.",
    )
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--observed-rms-scale", type=float, default=1.0)
    parser.add_argument(
        "--observed-rms-scales",
        default=None,
        help="Comma-separated relative RMS-radius scale sweep. If omitted, uses --observed-rms-scale.",
    )
    parser.add_argument(
        "--absolute-rms-arcmin",
        default=None,
        help="Optional comma-separated absolute RMS radii in arcmin, appended after relative scales.",
    )
    parser.add_argument("--min-rms-deg", type=float, default=0.0)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument(
        "--max-observed-rms-deg",
        type=float,
        default=None,
        help="Optional window-level filter on observed RMS radius before relative scale construction.",
    )
    parser.add_argument("--random-axes-per-window", type=int, default=1)
    parser.add_argument("--include-fixed-grid", action="store_true")
    parser.add_argument("--fixed-grid-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--candidate-groups",
        default="static,real,edge,edge_orthogonal,spectrum,random,grid",
        help=(
            "Comma-separated candidate groups to render. Supported: all, static, real, edge, "
            "edge_orthogonal, spectrum, random, grid. The grid group still requires --include-fixed-grid."
        ),
    )
    parser.add_argument(
        "--observers",
        default="pose_aware_flat,pose_blind_mean,pose_aware_delta_flat,pose_blind_delta_mean",
        help=(
            "Comma-separated response observers to keep: pose_aware_flat,pose_blind_mean,"
            "pose_aware_delta_flat,pose_blind_delta_mean. Delta observers subtract the static response "
            "for the same window before flattening/averaging."
        ),
    )
    parser.add_argument(
        "--latent-names",
        default=None,
        help="Optional comma-separated latent names to keep, e.g. pyramid_local_field,gabor_local_field,dct_local_field.",
    )
    parser.add_argument("--pca-k-list", default="4,8,16")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument(
        "--ridge-alpha-mode",
        choices=("nested_per_candidate", "fixed"),
        default="nested_per_candidate",
        help=(
            "Ridge alpha policy. nested_per_candidate preserves the original per-candidate nested-CV "
            "choice; fixed uses one alpha for all candidates/folds as a contrast sensitivity check."
        ),
    )
    parser.add_argument(
        "--fixed-ridge-alpha",
        type=float,
        default=None,
        help="Ridge alpha used when --ridge-alpha-mode=fixed. Defaults to the middle value in --ridge-alphas.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-shuffle-nulls", type=int, default=1000)
    parser.add_argument("--n-session-bootstrap", type=int, default=1000)
    parser.add_argument("--population-mode", choices=("canonical", "sampled"), default="canonical")
    parser.add_argument("--twin-population-n", type=int, default=256)
    parser.add_argument("--twin-batch-size", type=int, default=24)
    parser.add_argument(
        "--twin-trace-batch-size",
        type=int,
        default=8,
        help=(
            "Number of candidate traces to concatenate per canonical-twin call. This preserves the "
            "rendered stimuli and outputs but reduces per-trace overhead; lower it if GPU memory is tight."
        ),
    )
    parser.add_argument(
        "--cuda-empty-cache-every-batch",
        action="store_true",
        help="Match the older conservative behavior by calling torch.cuda.empty_cache() after each twin mini-batch.",
    )
    parser.add_argument(
        "--check-trace-batch-equivalence",
        action="store_true",
        help="Before the main loop, verify canonical responses match for trace_batch_size=1 and --twin-trace-batch-size.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _prepare_windows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    df["source_row"] = np.arange(df.shape[0], dtype=int)
    candidate_groups = set(_parse_str_list(args.candidate_groups))
    if "all" in candidate_groups:
        candidate_groups = {"static", "real", "edge", "edge_orthogonal", "spectrum", "random", "grid"}
    required = [
        "session",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "drift_orientation_deg",
        "anisotropy",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "image_patch_distance_to_image_border_px",
    ]
    if "spectrum" in candidate_groups:
        required.append("image_spectrum_orientation_deg")
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "duration_s" not in df.columns:
        df["duration_s"] = df.get("epoch_duration_s", np.nan)
    margin = float(args.min_patch_image_margin_px) if args.min_patch_image_margin_px is not None else float(args.patch_size_px) / 2.0
    keep = (
        np.isfinite(df["drift_orientation_deg"].astype(float))
        & np.isfinite(df["image_edge_axis_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(args.reliable_drift_anisotropy_min))
        & (df["image_orientation_coherence"].astype(float) >= float(args.reliable_image_coherence_min))
        & (df["duration_s"].astype(float) >= float(args.min_duration_s))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= margin)
    )
    if "spectrum" in candidate_groups:
        keep = keep & np.isfinite(df["image_spectrum_orientation_deg"].astype(float))
    if args.max_observed_rms_deg is not None:
        if "rms_radius_deg" not in df.columns:
            raise ValueError("--max-observed-rms-deg requires an input rms_radius_deg column")
        keep = keep & np.isfinite(df["rms_radius_deg"].astype(float)) & (df["rms_radius_deg"].astype(float) <= float(args.max_observed_rms_deg))
    work = df.loc[keep].copy()
    work["window_id"] = np.arange(work.shape[0], dtype=int)
    if args.window_manifest is not None:
        manifest = pd.read_csv(args.window_manifest)
        if "source_row" in manifest.columns:
            requested = manifest["source_row"].astype(int).drop_duplicates().to_list()
            available = set(work["source_row"].astype(int).to_list())
            missing_ids = sorted(set(requested).difference(available))
            if missing_ids:
                preview = ", ".join(str(v) for v in missing_ids[:10])
                suffix = "..." if len(missing_ids) > 10 else ""
                raise ValueError(f"--window-manifest source_row values do not survive current filters: {preview}{suffix}")
            work = work.set_index("source_row", drop=False).loc[requested].reset_index(drop=True)
        else:
            matched_rows: list[int] = []
            missing: list[int] = []
            for manifest_idx, manifest_row in manifest.drop_duplicates().iterrows():
                match = _first_close_match(work, manifest_row)
                if match is None:
                    missing.append(int(manifest_idx))
                else:
                    matched_rows.append(match)
            if missing:
                preview = ", ".join(str(v) for v in missing[:10])
                suffix = "..." if len(missing) > 10 else ""
                raise ValueError(
                    "--window-manifest lacks source_row and some rows could not be matched by "
                    f"session/trial/geometry after current filters: manifest rows {preview}{suffix}"
                )
            work = work.iloc[matched_rows].reset_index(drop=True)
    elif int(args.max_windows) > 0 and work.shape[0] > int(args.max_windows):
        work = work.sample(n=int(args.max_windows), replace=False, random_state=int(args.seed)).sort_values(
            ["session", "trial_idx", "window_id"]
        )
    return work.reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_k_list = _parse_int_list(args.pca_k_list)
    ridge_alphas = _parse_float_list(args.ridge_alphas)
    observed_rms_scales = _parse_float_list(args.observed_rms_scales) if args.observed_rms_scales else [float(args.observed_rms_scale)]
    absolute_rms_arcmin = _parse_float_list(args.absolute_rms_arcmin) if args.absolute_rms_arcmin else []
    latent_name_filter = set(_parse_str_list(args.latent_names))
    candidate_groups = _parse_str_list(args.candidate_groups)
    observers = _parse_str_list(args.observers)
    valid_candidate_groups = {"all", "static", "real", "edge", "edge_orthogonal", "spectrum", "random", "grid"}
    invalid_candidate_groups = sorted(set(candidate_groups).difference(valid_candidate_groups))
    if invalid_candidate_groups:
        raise ValueError(f"Unknown --candidate-groups entries: {invalid_candidate_groups}")
    valid_observers = {"pose_aware_flat", "pose_blind_mean", "pose_aware_delta_flat", "pose_blind_delta_mean"}
    invalid_observers = sorted(set(observers).difference(valid_observers))
    if invalid_observers:
        raise ValueError(f"Unknown --observers entries: {invalid_observers}")
    if not observers:
        raise ValueError("--observers must include at least one observer")
    if int(args.local_field_grid) < 1:
        raise ValueError("--local-field-grid must be >= 1")
    if float(args.min_rms_deg) < 0.0:
        raise ValueError("--min-rms-deg must be >= 0")
    if float(args.max_rms_deg) < float(args.min_rms_deg):
        raise ValueError("--max-rms-deg must be >= --min-rms-deg")
    if int(args.twin_trace_batch_size) < 1:
        raise ValueError("--twin-trace-batch-size must be >= 1")
    if str(args.ridge_alpha_mode) == "fixed" and args.fixed_ridge_alpha is not None and float(args.fixed_ridge_alpha) < 0.0:
        raise ValueError("--fixed-ridge-alpha must be nonnegative")
    min_timepoints = int(_load_twin_common().N_LAGS)
    if int(args.n_timepoints) < min_timepoints:
        raise ValueError(
            f"--n-timepoints must be >= the twin lag count ({min_timepoints}) for Ryan's "
            "counterfactual stimulus helper. Lower values need a dedicated scorer rewrite."
        )
    min_patch_image_margin_px = (
        float(args.min_patch_image_margin_px)
        if args.min_patch_image_margin_px is not None
        else float(args.patch_size_px) / 2.0
    )
    cfg = RunConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        max_windows=int(args.max_windows),
        window_manifest=str(args.window_manifest) if args.window_manifest is not None else None,
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
        patch_size_px=int(args.patch_size_px),
        min_patch_image_margin_px=min_patch_image_margin_px,
        latent_crop_px=int(args.latent_crop_px),
        center_crop_px=int(args.center_crop_px),
        local_field_grid=int(args.local_field_grid),
        n_timepoints=int(args.n_timepoints),
        observed_rms_scale=float(args.observed_rms_scale),
        observed_rms_scales=observed_rms_scales,
        absolute_rms_arcmin=absolute_rms_arcmin,
        min_rms_deg=float(args.min_rms_deg),
        max_rms_deg=float(args.max_rms_deg),
        max_observed_rms_deg=float(args.max_observed_rms_deg) if args.max_observed_rms_deg is not None else None,
        random_axes_per_window=int(args.random_axes_per_window),
        include_fixed_grid=bool(args.include_fixed_grid),
        fixed_grid_step_deg=float(args.fixed_grid_step_deg),
        candidate_groups=candidate_groups,
        observers=observers,
        latent_names=sorted(latent_name_filter),
        pca_k_list=pca_k_list,
        ridge_alphas=ridge_alphas,
        ridge_alpha_mode=str(args.ridge_alpha_mode),
        fixed_ridge_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        n_shuffle_nulls=int(args.n_shuffle_nulls),
        n_session_bootstrap=int(args.n_session_bootstrap),
        population_mode=str(args.population_mode),
        twin_population_n=int(args.twin_population_n),
        twin_batch_size=int(args.twin_batch_size),
        twin_trace_batch_size=int(args.twin_trace_batch_size),
        cuda_empty_cache_every_batch=bool(args.cuda_empty_cache_every_batch),
        check_trace_batch_equivalence=bool(args.check_trace_batch_equivalence),
        device=str(args.device),
        progress_every=int(args.progress_every),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed))
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No reliable BackImage windows survived the screen filters.")
    _progress(
        f"prepared {work.shape[0]} windows from {args.input}; "
        f"steerable_pyramid={'yes' if HAVE_STEERABLE_PYRAMID else 'no'}; output={out_dir}"
    )
    scorer = _make_twin_scorer(args)
    actual_response_units = int(getattr(scorer, "n_units", getattr(getattr(scorer, "population", None), "N", -1)))
    population_source = str(getattr(scorer, "population_source", str(args.population_mode)))
    model_family = str(getattr(scorer, "model_family", "unknown"))
    model_names = list(getattr(scorer, "model_names", []))
    _progress(
        f"loaded twin population_mode={args.population_mode}, requested_n={args.twin_population_n}, "
        f"actual_response_units={actual_response_units}, batch_size={args.twin_batch_size}, "
        f"trace_batch_size={args.twin_trace_batch_size}, device={args.device}"
    )

    window_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    latent_values: dict[str, list[np.ndarray]] = {}
    response_values: dict[tuple[str, str, str], list[np.ndarray]] = {}
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    observer_set = set(observers)
    trace_batch_checked = False

    for i, row in tqdm(work.iterrows(), total=work.shape[0], desc="latent information screen"):
        canvas_key = (str(row["session"]), int(row["trial_idx"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
        latents = _extract_latents(
            patch,
            latent_crop_px=int(args.latent_crop_px),
            center_crop_px=int(args.center_crop_px),
            local_field_grid=int(args.local_field_grid),
        )
        if latent_name_filter:
            latents = {name: value for name, value in latents.items() if name in latent_name_filter}
            if not latents:
                raise ValueError(f"None of --latent-names were available. Requested: {sorted(latent_name_filter)}")
        for name, value in latents.items():
            latent_values.setdefault(name, []).append(value)
        window_rows.append(
            {
                "window_row": int(i),
                "window_id": int(row["window_id"]),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "real_drift_axis_deg": float(row["drift_orientation_deg"]),
                "edge_axis_deg": float(row["image_edge_axis_deg"]),
                "spectrum_axis_deg": float(row["image_spectrum_orientation_deg"]) if "image_spectrum_orientation_deg" in row else np.nan,
                "drift_edge_cos2": float(_cos2(float(row["drift_orientation_deg"]), float(row["image_edge_axis_deg"]))),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "observed_rms_radius_deg": float(row.get("rms_radius_deg", np.nan)),
            }
        )
        cand_rng = np.random.default_rng(int(args.seed) + int(row["window_id"]) * 7919)
        response_cache: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
        needs_delta_observer = bool({"pose_aware_delta_flat", "pose_blind_delta_mean"} & observer_set)
        specs = _candidate_specs(row, rng=cand_rng, args=args)
        traces_to_score: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
        if needs_delta_observer:
            static_trace = _static_trace(int(args.n_timepoints))
            static_key = _trace_cache_key(static_trace)
            traces_to_score[static_key] = static_trace
        for spec in specs:
            traces_to_score.setdefault(_trace_cache_key(spec["trace"]), spec["trace"])
        trace_keys = list(traces_to_score)
        trace_values = [traces_to_score[key] for key in trace_keys]
        if bool(args.check_trace_batch_equivalence) and not trace_batch_checked:
            _check_trace_batch_equivalence(
                scorer,
                patch,
                trace_values,
                trace_batch_size=int(args.twin_trace_batch_size),
                n_timepoints=int(args.n_timepoints),
            )
            trace_batch_checked = True
        if hasattr(scorer, "responses"):
            raw_responses = scorer.responses(
                patch,
                trace_values,
                trace_batch_size=int(args.twin_trace_batch_size),
            )
        else:
            raw_responses = [scorer.response(patch, trace) for trace in trace_values]
        for trace_key, raw_response in zip(trace_keys, raw_responses, strict=True):
            response_cache[trace_key] = _align_response_to_trace(raw_response, int(args.n_timepoints))
        static_response = response_cache.get(_trace_cache_key(_static_trace(int(args.n_timepoints)))) if needs_delta_observer else None
        for spec in specs:
            trace_key = _trace_cache_key(spec["trace"])
            response = response_cache[trace_key]
            feats = {
                observer: vec
                for observer, vec in _response_features(response, static_response=static_response).items()
                if observer in observer_set
            }
            candidate = str(spec["candidate"])
            for observer, vec in feats.items():
                response_values.setdefault((str(spec["motion_scale_id"]), candidate, observer), []).append(vec)
            motion_rows.append(
                {
                    "window_row": int(i),
                    "window_id": int(row["window_id"]),
                    "source_row": int(row["source_row"]),
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    "candidate": candidate,
                    "motion_scale_id": str(spec["motion_scale_id"]),
                    "motion_scale_kind": str(spec["motion_scale_kind"]),
                    "motion_scale_value": float(spec["motion_scale_value"]),
                    "motion_scale_label": str(spec["motion_scale_label"]),
                    "axis_deg": float(spec["axis_deg"]) if np.isfinite(spec["axis_deg"]) else np.nan,
                    "amplitude_type": spec["amplitude_type"],
                    "amplitude_value_deg": float(spec["amplitude_value_deg"]),
                    "observed_rms_scale": float(spec["observed_rms_scale"]),
                    "nominal_absolute_rms_deg": float(spec["nominal_absolute_rms_deg"]),
                    "raw_rms_radius_deg": float(spec["raw_rms_radius_deg"]),
                    "rms_clipped_low": bool(spec["rms_clipped_low"]),
                    "rms_clipped_high": bool(spec["rms_clipped_high"]),
                    "duration_s": float(spec["duration_s"]) if np.isfinite(spec["duration_s"]) else np.nan,
                    "template_shape": spec["template_shape"],
                    "n_timepoints": int(spec["n_timepoints"]),
                    "n_response_frames": int(response.shape[0]),
                    "n_response_units": int(response.shape[1]),
                    "path_length_deg": float(spec["path_length_deg"]),
                    "rms_radius_deg": float(spec["rms_radius_deg"]),
                    "endpoint_extent_deg": float(spec["endpoint_extent_deg"]),
                }
            )
        done = int(i) + 1
        if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
            _progress(
                f"windows {done}/{work.shape[0]}; "
                f"latent_arrays={len(latent_values)}; response_blocks={len(response_values)}"
            )

    _progress("writing window, candidate, latent, and response arrays")
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(out_dir / "analysis_windows.csv", index=False)
    _write_csv(out_dir / "candidate_motion_metadata.csv", motion_rows)
    latent_arrays = {name: np.vstack(values) for name, values in latent_values.items()}
    response_arrays = {key: np.vstack(values) for key, values in response_values.items()}
    np.savez_compressed(out_dir / "latent_feature_arrays.npz", **latent_arrays)
    np.savez_compressed(
        out_dir / "response_feature_arrays.npz",
        **{f"{scale_id}__{candidate}__{observer}": values for (scale_id, candidate, observer), values in response_arrays.items()},
    )

    groups = window_df["session"].to_numpy()
    decode_rows: list[dict[str, Any]] = []
    per_window_rows: list[dict[str, Any]] = []
    total_decode_jobs = len(latent_arrays) * len(response_arrays) * len(pca_k_list)
    decode_job = 0
    _progress(f"starting decode screen with {total_decode_jobs} jobs")
    scale_lookup = {
        str(row["motion_scale_id"]): {
            "motion_scale_kind": str(row["motion_scale_kind"]),
            "motion_scale_value": float(row["motion_scale_value"]),
            "motion_scale_label": str(row["motion_scale_label"]),
            "observed_rms_scale": float(row["observed_rms_scale"]),
            "nominal_absolute_rms_deg": float(row["nominal_absolute_rms_deg"]),
        }
        for row in pd.DataFrame(motion_rows)
        .drop_duplicates("motion_scale_id")
        .to_dict(orient="records")
    }
    for latent_name, Z in latent_arrays.items():
        latent_family, latent_scope = latent_name.split("_", 1)
        for (scale_id, candidate, observer), X in response_arrays.items():
            scale_meta = scale_lookup[str(scale_id)]
            for k in pca_k_list:
                decode_job += 1
                if decode_job == 1 or decode_job == total_decode_jobs or decode_job % 25 == 0:
                    _progress(
                        f"decode {decode_job}/{total_decode_jobs}: "
                        f"latent={latent_name}, candidate={candidate}, observer={observer}, k={k}"
                    )
                result = _cross_validated_decode(
                    X,
                    Z,
                    groups,
                    k=int(k),
                    alphas=ridge_alphas,
                    alpha_mode=str(args.ridge_alpha_mode),
                    fixed_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
                    outer_folds=int(args.outer_folds),
                    inner_folds=int(args.inner_folds),
                    seed=int(args.seed),
                )
                decode_rows.append(
                    {
                        "latent_name": latent_name,
                        "latent_family": latent_family,
                        "latent_scope": latent_scope,
                        "motion_scale_id": str(scale_id),
                        "motion_scale_kind": scale_meta["motion_scale_kind"],
                        "motion_scale_value": scale_meta["motion_scale_value"],
                        "motion_scale_label": scale_meta["motion_scale_label"],
                        "observed_rms_scale": scale_meta["observed_rms_scale"],
                        "nominal_absolute_rms_deg": scale_meta["nominal_absolute_rms_deg"],
                        "candidate": candidate,
                        "observer": observer,
                        "pca_k": int(k),
                        "target_dim": int(result["target_dim"]),
                        "R2_z": float(result["r2"]),
                        "R2_z_method": str(result["r2_method"]),
                        "decode_score_neg_mse": float(result["mean_neg_mse"]),
                        "chosen_alpha_median": float(result["chosen_alpha_median"]),
                        "ridge_alpha_mode": str(result["ridge_alpha_mode"]),
                    }
                )
                scores = result["per_window_score"]
                for idx, score in enumerate(scores):
                    per_window_rows.append(
                        {
                            "window_row": int(idx),
                            "window_id": int(window_df.loc[idx, "window_id"]),
                            "source_row": int(window_df.loc[idx, "source_row"]),
                            "session": str(window_df.loc[idx, "session"]),
                            "latent_name": latent_name,
                            "latent_family": latent_family,
                            "latent_scope": latent_scope,
                            "motion_scale_id": str(scale_id),
                            "motion_scale_kind": scale_meta["motion_scale_kind"],
                            "motion_scale_value": scale_meta["motion_scale_value"],
                            "motion_scale_label": scale_meta["motion_scale_label"],
                            "observed_rms_scale": scale_meta["observed_rms_scale"],
                            "nominal_absolute_rms_deg": scale_meta["nominal_absolute_rms_deg"],
                            "candidate": candidate,
                            "observer": observer,
                            "pca_k": int(k),
                            "decode_score_neg_mse": float(score),
                        }
                    )

    decode_df = pd.DataFrame(decode_rows)
    if not decode_df.empty:
        scale_cols = ["motion_scale_id", "motion_scale_kind", "motion_scale_value", "motion_scale_label"]
        merge_cols = ["latent_name", "observer", "pca_k", *scale_cols]
        static = decode_df[decode_df["candidate"] == "static"][
            [*merge_cols, "R2_z", "decode_score_neg_mse"]
        ].rename(columns={"R2_z": "static_R2_z", "decode_score_neg_mse": "static_decode_score_neg_mse"})
        randoms = decode_df[decode_df["candidate"].str.startswith("random_axis_")]
        random_summary = randoms.groupby(merge_cols, as_index=False)[
            ["R2_z", "decode_score_neg_mse"]
        ].mean().rename(columns={"R2_z": "random_axis_R2_z", "decode_score_neg_mse": "random_axis_decode_score_neg_mse"})
        decode_df = decode_df.merge(static, on=merge_cols, how="left")
        decode_df = decode_df.merge(random_summary, on=merge_cols, how="left")
        decode_df["Delta_R2_z_vs_static"] = decode_df["R2_z"] - decode_df["static_R2_z"]
        decode_df["Delta_R2_z_vs_random_axis"] = decode_df["R2_z"] - decode_df["random_axis_R2_z"]
        decode_df["Delta_score_vs_static"] = decode_df["decode_score_neg_mse"] - decode_df["static_decode_score_neg_mse"]
        decode_df["Delta_score_vs_random_axis"] = decode_df["decode_score_neg_mse"] - decode_df["random_axis_decode_score_neg_mse"]
    decode_df.to_csv(out_dir / "decode_summary_by_candidate.csv", index=False)
    per_window_df = pd.DataFrame(per_window_rows)
    per_window_df.to_csv(out_dir / "decode_score_by_window_candidate.csv", index=False)

    _progress("running alignment-strength regressions and shuffle nulls")
    geom_rows: list[dict[str, Any]] = []
    y = window_df["drift_edge_cos2"].to_numpy(dtype=np.float64)
    sessions = window_df["session"].to_numpy()
    if not per_window_df.empty:
        pivot = per_window_df.pivot_table(
            index="window_row",
            columns=["latent_name", "observer", "pca_k", "motion_scale_id", "candidate"],
            values="decode_score_neg_mse",
            aggfunc="mean",
        )
        for latent_name in latent_arrays:
            for observer in observers:
                for k in pca_k_list:
                    for scale_id in sorted(scale_lookup):
                        key_edge = (latent_name, observer, int(k), scale_id, "edge")
                        key_orth = (latent_name, observer, int(k), scale_id, "edge_orthogonal")
                        if key_edge not in pivot.columns or key_orth not in pivot.columns:
                            continue
                        advantage = (pivot[key_edge] - pivot[key_orth]).reindex(np.arange(window_df.shape[0])).to_numpy(dtype=np.float64)
                        pred_dm = _demean_within_session(advantage, sessions)
                        y_dm = _demean_within_session(y, sessions)
                        controls_dm = np.column_stack([
                            _demean_within_session(window_df["image_orientation_coherence"].to_numpy(dtype=np.float64), sessions)
                        ])
                        coef, delta, p_coef, p_delta = _shuffle_pvalues(
                            y_dm,
                            pred_dm,
                            controls_dm,
                            sessions,
                            rng=rng,
                            n_shuffle=int(args.n_shuffle_nulls),
                        )
                        scale_meta = scale_lookup[str(scale_id)]
                        geom_rows.append(
                            {
                                "latent_name": latent_name,
                                "observer": observer,
                                "pca_k": int(k),
                                "motion_scale_id": str(scale_id),
                                "motion_scale_kind": scale_meta["motion_scale_kind"],
                                "motion_scale_value": scale_meta["motion_scale_value"],
                                "motion_scale_label": scale_meta["motion_scale_label"],
                                "observed_rms_scale": scale_meta["observed_rms_scale"],
                                "nominal_absolute_rms_deg": scale_meta["nominal_absolute_rms_deg"],
                                "predictor": "edge_minus_orth_decode_score",
                                "controls": "within_session_image_orientation_coherence",
                                "within_session_coef": coef,
                                "within_session_incremental_r2": delta,
                                "within_session_shuffle_p_coef_ge": p_coef,
                                "within_session_shuffle_p_incremental_r2_ge": p_delta,
                                "mean_predictor": float(np.nanmean(advantage)),
                            }
                        )
    _write_csv(out_dir / "alignment_strength_prediction_summary.csv", geom_rows)

    _progress("writing figures and summary")
    _plot_decode_summary(out_dir, decode_df)
    _write_summary(
        out_dir,
        cfg,
        decode_df,
        geom_rows,
        actual_response_units=actual_response_units,
        population_source=population_source,
        model_family=model_family,
    )
    _write_json(
        out_dir / "run_metadata.json",
        {
            "config": asdict(cfg),
            "n_input_rows": int(pd.read_csv(args.input, usecols=["session"]).shape[0]),
            "n_reliable_rows": int(work.shape[0]),
            "n_windows": int(window_df.shape[0]),
            "n_motion_rows": int(len(motion_rows)),
            "n_latent_arrays": int(len(latent_arrays)),
            "n_response_arrays": int(len(response_arrays)),
            "motion_scale_ids": sorted(scale_lookup),
            "population_mode": str(args.population_mode),
            "population_source": population_source,
            "model_family": model_family,
            "model_names": model_names,
            "model_dataset_count": int(len(model_names)),
            "actual_response_units": int(actual_response_units),
            "n_canvas_cache_entries": int(len(canvas_cache)),
            "notes": (
                "Ridge R2 and negative MSE are first-pass information proxies. "
                "R2_z is the mean of outer-fold R2 values in each fold's target PCA basis. "
                f"Ridge alpha mode is {cfg.ridge_alpha_mode}. "
                "Trace batching is a canonical-twin optimization; sampled mode falls back to per-trace scoring. "
                "Axis candidates use matched sinusoidal-line templates with amplitude_type=RMS radius. "
                "Twin responses are aligned back to trace length before observer features are built; "
                "delta observers subtract the aligned static response for each window."
            ),
        },
    )
    _progress(f"wrote BackImage latent-information screen to {out_dir}")
    return out_dir


def _plot_decode_summary(out_dir: Path, decode_df: pd.DataFrame) -> None:
    if decode_df.empty:
        return
    key = decode_df[(decode_df["pca_k"] == decode_df["pca_k"].min()) & decode_df["candidate"].isin(["static", "edge", "edge_orthogonal", "real_drift_axis", "spectrum", "random_axis_0"])]
    if key.empty:
        return
    for observer, block in key.groupby("observer"):
        labels = []
        values = []
        for _, row in block.sort_values(["latent_name", "candidate"]).iterrows():
            labels.append(f"{row['latent_name']}\n{row['candidate']}")
            values.append(float(row["Delta_R2_z_vs_static"]))
        fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(values)), 3.8), dpi=150)
        ax.bar(np.arange(len(values)), values, color="#4c78a8")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("Delta R2 vs static")
        ax.set_title(f"BackImage latent screen ({observer})", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_delta_r2_vs_static_{observer}.png", dpi=150)
        plt.close(fig)


def _write_summary(
    out_dir: Path,
    cfg: RunConfig,
    decode_df: pd.DataFrame,
    geom_rows: list[dict[str, Any]],
    *,
    actual_response_units: int,
    population_source: str,
    model_family: str,
) -> None:
    lines = [
        "# BackImage Latent-Information Screen",
        "",
        f"Input: `{cfg.input}`",
        "",
        "## Run",
        "",
        f"- Windows: `{cfg.max_windows}`",
        f"- Window manifest: `{cfg.window_manifest}`",
        f"- Population mode: `{cfg.population_mode}`",
        f"- Population source: `{population_source}`",
        f"- Model family: `{model_family}`",
            f"- Requested sampled population (sampled mode only): `{cfg.twin_population_n}`",
            f"- Actual response units: `{actual_response_units}`",
            f"- Twin frame batch size: `{cfg.twin_batch_size}`; canonical trace batch size: `{cfg.twin_trace_batch_size}`; "
            f"empty-cache every batch: `{cfg.cuda_empty_cache_every_batch}`; trace-batch equivalence check: `{cfg.check_trace_batch_equivalence}`",
            f"- Local-field grid: `{cfg.local_field_grid}x{cfg.local_field_grid}` blocks; Gabor local fields include even, odd, and amplitude maps",
        "- R2_z method: mean outer-fold R2 in each fold's target PCA basis",
        f"- Motion amplitude: `{cfg.observed_rms_scale}x observed RMS radius`, clipped to `[{cfg.min_rms_deg}, {cfg.max_rms_deg}]` deg",
        f"- Max observed RMS filter: `{cfg.max_observed_rms_deg}`",
        f"- Motion scale sweep: relative `{cfg.observed_rms_scales}`, absolute arcmin `{cfg.absolute_rms_arcmin}`",
        f"- Candidate groups: `{cfg.candidate_groups}`",
        f"- Observers: `{cfg.observers}`",
        "- Response alignment: model outputs are cropped back to the requested trace length before observer features are built",
        "- Delta observers subtract the aligned static response for the same window before decoding",
        f"- Random axes per window: `{cfg.random_axes_per_window}`",
        f"- PCA k list: `{cfg.pca_k_list}`",
        f"- Ridge alpha mode: `{cfg.ridge_alpha_mode}`; fixed alpha `{cfg.fixed_ridge_alpha}`",
        "",
        "## Decode Screen",
        "",
    ]
    if not decode_df.empty:
        key = decode_df[
            decode_df["candidate"].isin(["edge", "edge_orthogonal", "real_drift_axis", "spectrum", "random_axis_0"])
        ].copy()
        key = key.sort_values("Delta_R2_z_vs_static", ascending=False).head(12)
        for _, row in key.iterrows():
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` `{row['candidate']}`: "
                f"R2 `{row['R2_z']:.4f}`, dR2 static `{row['Delta_R2_z_vs_static']:.4f}`, "
                f"dR2 random `{row['Delta_R2_z_vs_random_axis']:.4f}`."
            )
    lines.extend(["", "## Alignment-Strength Prediction", ""])
    if geom_rows:
        geom = pd.DataFrame(geom_rows).sort_values("within_session_coef", ascending=False)
        for _, row in geom.head(12).iterrows():
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` `{row['predictor']}`: "
                f"coef `{row['within_session_coef']:.4f}`, dR2 `{row['within_session_incremental_r2']:.4f}`, "
                f"p(coef>=obs) `{row['within_session_shuffle_p_coef_ge']:.4f}`, "
                f"p(dR2>=obs) `{row['within_session_shuffle_p_incremental_r2_ge']:.4f}`."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `analysis_windows.csv`",
            "- `candidate_motion_metadata.csv`",
            "- `latent_feature_arrays.npz`",
            "- `response_feature_arrays.npz`",
            "- `decode_summary_by_candidate.csv`",
            "- `decode_score_by_window_candidate.csv`",
            "- `alignment_strength_prediction_summary.csv`",
            "",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

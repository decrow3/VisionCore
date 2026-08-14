"""Validated lag-zero-only BackImage renderer for input spectral analysis.

This module intentionally has no neural-model import.  It uses Torch only for
the same bilinear ``grid_sample`` operation as the validated counterfactual
helper, avoiding its unnecessary 32-lag tensor for input-only spectra.
"""
from __future__ import annotations

import numpy as np
import torch

from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _trace_xy_to_twin_helper_order,
)

def render_retinal_frames_lag_zero(
    common,
    patch: np.ndarray,
    trace_xy: np.ndarray,
    *,
    ppd: float,
    device: str = "cpu",
    out_size: tuple[int, int] = (51, 51),
) -> torch.Tensor:
    """Return ``(T, H, W)`` frames matching the helper's lag-zero output.

    ``trace_xy`` is a target-centred trajectory in the x/y convention. The
    caller is responsible for the conventional negative sign that converts a
    crop trajectory into retinal image motion. Torch is used only for the same
    bilinear ``grid_sample`` operation as the validated helper; this function
    does not embed lags or import/load a neural model.
    """
    trace = np.asarray(trace_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 1:
        raise ValueError(f"Expected a nonempty (T, 2) trace, got {trace.shape}")
    if not np.isfinite(trace).all():
        raise ValueError("Trace contains non-finite values")
    image = np.asarray(patch, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 51:
        raise ValueError(f"Expected a 2D source patch at least 51 px wide, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Source patch contains non-finite values")
    if not np.isfinite(float(ppd)) or float(ppd) <= 0:
        raise ValueError(f"Invalid ppd={ppd!r}")

    target_device = torch.device(device)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(target_device)
    # ``expand`` avoids materializing T identical 540x540 source images. The
    # downstream grid sampler treats the zero-stride time dimension read-only.
    stack = image_tensor.unsqueeze(0).expand(trace.shape[0], -1, -1)
    # Match the validated helper's gaze-coordinate conversion exactly before
    # its subsequent x/y flip for grid sampling.
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace)).to(target_device)
    eye_norm = common._eye_deg_to_norm(torch.fliplr(eye), float(ppd), tuple(image.shape))
    movie = common._shift_movie_with_eye(
        stack,
        eye_norm,
        out_size=tuple(int(value) for value in out_size),
        center=(0.0, 0.0),
        scale_factor=1.0,
        mode="bilinear",
    )
    expected = (trace.shape[0], int(out_size[0]), int(out_size[1]))
    if tuple(movie.shape) != expected:
        raise AssertionError(f"Unexpected lag-zero movie shape {tuple(movie.shape)}")
    return movie

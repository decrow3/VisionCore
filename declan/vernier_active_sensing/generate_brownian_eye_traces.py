#!/usr/bin/env python3
"""Generate synthetic Brownian eye traces in the Vernier trace-file format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trajectories import DEFAULT_EYE_TRACES_PATH, FRAME_RATE_HZ

COVARIANCE_MODES = ("diagonal_empirical", "full_empirical", "isotropic_scalar")
CENTER_MODES = ("zero_mean", "source_grand_mean", "start_zero")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_valid_samples_and_steps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        traces = np.asarray(npz["traces"], dtype=np.float64)
        durations = np.asarray(npz["durations"], dtype=np.int64)
    samples: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    for trace, duration in zip(traces, durations, strict=True):
        t = int(duration)
        if t <= 0:
            continue
        valid = np.asarray(trace[:t], dtype=np.float64)
        samples.append(valid)
        if t > 1:
            steps.append(np.diff(valid, axis=0))
    if not samples or not steps:
        raise ValueError(f"No valid samples/steps found in {path}")
    return np.concatenate(samples, axis=0), np.concatenate(steps, axis=0)


def estimate_source_covariance(
    path: Path,
    *,
    covariance_mode: str,
    global_scale: float = 1.0,
) -> dict[str, Any]:
    """Estimate source Brownian step covariance in deg^2/frame."""
    samples, steps = _load_valid_samples_and_steps(path)
    raw_cov = np.cov(steps.T)
    if str(covariance_mode) == "diagonal_empirical":
        cov = np.diag(np.diag(raw_cov))
    elif str(covariance_mode) == "full_empirical":
        cov = raw_cov
    elif str(covariance_mode) == "isotropic_scalar":
        scalar = float(np.mean(np.diag(raw_cov)))
        cov = np.eye(2, dtype=np.float64) * scalar
    else:
        raise ValueError(f"Unsupported covariance_mode={covariance_mode!r}; expected {COVARIANCE_MODES}")
    cov = np.asarray(cov, dtype=np.float64) * float(global_scale)
    return {
        "source_sample_mean_deg": np.mean(samples, axis=0),
        "source_step_mean_deg_per_frame": np.mean(steps, axis=0),
        "source_step_cov_deg2_per_frame": raw_cov,
        "synthetic_step_cov_deg2_per_frame": cov,
        "n_source_samples": int(samples.shape[0]),
        "n_source_steps": int(steps.shape[0]),
    }


def diffusion_arcmin2_per_s(step_cov_deg2_per_frame: np.ndarray, *, frame_rate_hz: float) -> np.ndarray:
    """Convert per-frame Brownian step covariance to axis diffusion constants."""
    cov = np.asarray(step_cov_deg2_per_frame, dtype=np.float64)
    return np.diag(cov) * (60.0**2) * float(frame_rate_hz) / 2.0


def generate_brownian_traces(
    *,
    step_cov_deg2_per_frame: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
    center_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Generate centered Brownian traces with shape (trace, frame, xy)."""
    n_traces = int(n_traces)
    n_frames = int(n_frames)
    if n_traces <= 0:
        raise ValueError("n_traces must be positive")
    if n_frames <= 1:
        raise ValueError("n_frames must be at least 2")
    rng = np.random.default_rng(int(seed))
    cov = np.asarray(step_cov_deg2_per_frame, dtype=np.float64)
    steps = rng.multivariate_normal(np.zeros(2, dtype=np.float64), cov, size=(n_traces, n_frames - 1))
    traces = np.zeros((n_traces, n_frames, 2), dtype=np.float64)
    traces[:, 1:, :] = np.cumsum(steps, axis=1)

    mode = str(center_mode)
    if mode == "zero_mean":
        traces = traces - np.mean(traces, axis=1, keepdims=True)
    elif mode == "source_grand_mean":
        if center_deg is None:
            raise ValueError("center_deg is required when center_mode='source_grand_mean'")
        traces = traces - np.mean(traces, axis=1, keepdims=True) + np.asarray(center_deg, dtype=np.float64)[None, None, :]
    elif mode == "start_zero":
        pass
    else:
        raise ValueError(f"Unsupported center_mode={center_mode!r}; expected {CENTER_MODES}")
    return traces.astype(np.float32)


def write_trace_file(
    path: Path,
    traces: np.ndarray,
    *,
    frame_rate_hz: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(traces, dtype=np.float32)
    durations = np.full(arr.shape[0], arr.shape[1], dtype=np.int32)
    steps = np.diff(arr, axis=1)
    rms = np.sqrt(np.mean(np.sum(arr * arr, axis=2), axis=1)).astype(np.float32)
    path_length = np.sum(np.linalg.norm(steps, axis=2), axis=1).astype(np.float32)
    velocity_rms = (
        np.sqrt(np.mean(np.sum((steps * float(frame_rate_hz)) ** 2, axis=2), axis=1)).astype(np.float32)
        if steps.size
        else np.zeros(arr.shape[0], dtype=np.float32)
    )
    sessions = np.asarray([f"synthetic_brownian_{idx:04d}" for idx in range(arr.shape[0])])
    np.savez_compressed(
        path,
        traces=arr,
        durations=durations,
        sessions=sessions,
        rms=rms,
        path_length=path_length,
        velocity_rms=velocity_rms,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    stats = estimate_source_covariance(
        Path(args.source_eye_traces),
        covariance_mode=str(args.covariance_mode),
        global_scale=float(args.global_scale),
    )
    traces = generate_brownian_traces(
        step_cov_deg2_per_frame=np.asarray(stats["synthetic_step_cov_deg2_per_frame"], dtype=np.float64),
        n_traces=int(args.n_traces),
        n_frames=int(args.n_frames),
        seed=int(args.seed),
        center_mode=str(args.center_mode),
        center_deg=np.asarray(stats["source_sample_mean_deg"], dtype=np.float64),
    )
    out_path = Path(args.out_path)
    write_trace_file(out_path, traces, frame_rate_hz=float(args.frame_rate_hz))

    realized_steps = np.diff(traces.astype(np.float64), axis=1).reshape(-1, 2)
    realized_cov = np.cov(realized_steps.T)
    manifest = {
        "source_eye_traces": Path(args.source_eye_traces),
        "out_path": out_path,
        "n_traces": int(args.n_traces),
        "n_frames": int(args.n_frames),
        "seed": int(args.seed),
        "frame_rate_hz": float(args.frame_rate_hz),
        "covariance_mode": str(args.covariance_mode),
        "center_mode": str(args.center_mode),
        "global_scale": float(args.global_scale),
        **stats,
        "synthetic_diffusion_arcmin2_per_s_xy": diffusion_arcmin2_per_s(
            np.asarray(stats["synthetic_step_cov_deg2_per_frame"], dtype=np.float64),
            frame_rate_hz=float(args.frame_rate_hz),
        ),
        "synthetic_diffusion_scalar_arcmin2_per_s": float(
            np.mean(
                diffusion_arcmin2_per_s(
                    np.asarray(stats["synthetic_step_cov_deg2_per_frame"], dtype=np.float64),
                    frame_rate_hz=float(args.frame_rate_hz),
                )
            )
        ),
        "realized_step_cov_deg2_per_frame": realized_cov,
        "realized_diffusion_arcmin2_per_s_xy": diffusion_arcmin2_per_s(
            realized_cov,
            frame_rate_hz=float(args.frame_rate_hz),
        ),
    }
    _write_json(out_path.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eye-traces", type=Path, default=DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--n-traces", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-rate-hz", type=float, default=FRAME_RATE_HZ)
    parser.add_argument("--covariance-mode", choices=COVARIANCE_MODES, default="diagonal_empirical")
    parser.add_argument("--center-mode", choices=CENTER_MODES, default="zero_mean")
    parser.add_argument("--global-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    diffusion = np.asarray(manifest["synthetic_diffusion_arcmin2_per_s_xy"], dtype=np.float64)
    print(
        "Wrote Brownian traces with D_x={:.3f}, D_y={:.3f} arcmin^2/s to {}".format(
            float(diffusion[0]),
            float(diffusion[1]),
            manifest["out_path"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

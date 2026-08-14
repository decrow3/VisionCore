#!/usr/bin/env python3
"""Stream corrected lag-zero retinal movies to SF x TF sufficient statistics.

This runner is input-only. It imports Torch for bilinear grid sampling and FFTs,
but never loads or calls the frozen neural model. Each completed image is an
atomic resumable shard. Spectra are calculated only from the 40 scored frames;
the 32-frame recorded history is validated and retained for later neural
scoring but cannot affect a lag-zero retinal-input spectrum.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
)
from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_bridge_cohort_checkpoint_28_v1"
IMAGE_AUDIT = ROOT / "outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1"
VALIDATION = ROOT / "outputs/fig4_active_sensing/rr100_input_only_renderer_actual_validation_checkpoint_31_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_input_spectral_cache_checkpoint_34_v1"

FRAME_RATE_HZ = 120.0
N_HISTORY = 32
N_SCORE = 40
TF_CORE_MAX_HZ = 32.0
TF_EDGE_HZ = 45.25
SF_FIT_MIN_CPD = 1.0
SF_FIT_MAX_CPD = 11.313708498984761
SF_EDGES_CPD = np.asarray(
    [0.0, 0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0, 5.65685425, 8.0, 11.3137085, 16.0, 22.627417, 32.0],
    dtype=np.float64,
)
ORIENTATION_EDGES_DEG = np.linspace(0.0, 180.0, 13, dtype=np.float64)
SCALAR_NAMES = (
    "total_positive_tf_power",
    "power_le_32_all_sf",
    "power_32_45p25_all_sf",
    "power_45p25_60_all_sf",
    "power_at_60_all_sf",
    "total_positive_tf_power_fitted_sf",
    "power_le_32_fitted_sf",
    "power_32_45p25_fitted_sf",
    "power_45p25_60_fitted_sf",
    "power_at_60_fitted_sf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--image-audit-dir", type=Path, default=IMAGE_AUDIT)
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--max-movies", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def require_validation_gate(validation_dir: Path) -> dict:
    path = validation_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Actual-input renderer validation is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    gates = manifest.get("gates", {})
    if not bool(gates.get("helper_equivalence_pass")):
        raise RuntimeError("Actual-input helper-equivalence gate did not pass")
    if not bool(gates.get("median_exact_input_agreement_pass_ge_0p80")):
        raise RuntimeError("Actual-input saved-frame agreement gate did not pass")
    return manifest


def trace_indices(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    history = np.arange(
        int(row["corrected_history_global_start"]),
        int(row["corrected_history_global_stop_exclusive"]),
        2,
        dtype=int,
    )
    score = np.arange(
        int(row["corrected_scored_global_start"]),
        int(row["corrected_scored_global_stop_exclusive"]),
        2,
        dtype=int,
    )
    if history.shape != (N_HISTORY,) or score.shape != (N_SCORE,):
        raise ValueError(
            f"Trace {row['trace_index']} has history/score lengths {history.shape}/{score.shape}"
        )
    return history, score


def materialize_trace_arrays(traces: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    history_out = np.empty((len(traces), N_HISTORY, 2), dtype=np.float32)
    score_out = np.empty((len(traces), N_SCORE, 2), dtype=np.float32)
    ppd_by_session: dict[str, float] = {}
    for session, session_rows in traces.groupby("session", sort=True):
        cache: dict = {}
        dset = load_dset(str(session), cache)
        crop = corrected_crop_xy_deg(dset)
        ppd_by_session[str(session)] = float(dset.metadata["ppd"])
        for row_index, row in session_rows.iterrows():
            history_indices, score_indices = trace_indices(row)
            if not bool(row["explicit_history_valid"]):
                raise ValueError(f"Trace {row['trace_index']} failed explicit-history validity")
            score = crop[score_indices]
            center = score.mean(axis=0, keepdims=True)
            history_out[int(row_index)] = (crop[history_indices] - center).astype(np.float32)
            score_out[int(row_index)] = (score - center).astype(np.float32)
        del crop, dset
        cache.clear()
        gc.collect()
    return history_out, score_out, ppd_by_session


def spatial_lookup(ppd: float, size: int = 51) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / float(ppd)))
    fx = np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / float(ppd)))
    radial = np.hypot(fx[None, :], fy[:, None])
    orientation = np.mod(np.degrees(np.arctan2(fy[:, None], fx[None, :])), 180.0)
    sf_bin = np.digitize(radial.ravel(), SF_EDGES_CPD, right=False) - 1
    ori_bin = np.digitize(orientation.ravel(), ORIENTATION_EDGES_DEG, right=False) - 1
    sf_bin = np.clip(sf_bin, 0, len(SF_EDGES_CPD) - 2)
    ori_bin = np.clip(ori_bin, 0, len(ORIENTATION_EDGES_DEG) - 2)
    fitted = (radial.ravel() >= SF_FIT_MIN_CPD) & (radial.ravel() <= SF_FIT_MAX_CPD)
    return sf_bin.astype(int), ori_bin.astype(int), fitted


def spectral_statistics(movie: np.ndarray, *, ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(movie, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] != N_SCORE or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Expected a {N_SCORE} x H x H scored movie, got {arr.shape}")
    spatial_size = int(arr.shape[1])
    residual = arr - arr.mean(axis=0, keepdims=True)
    temporal_window = np.hanning(N_SCORE)[:, None, None]
    spatial_window = np.outer(np.hanning(spatial_size), np.hanning(spatial_size))[None, :, :]
    temporal_fft = np.fft.rfft(residual * temporal_window * spatial_window, axis=0)
    spectrum = np.fft.fftshift(np.fft.fft2(temporal_fft, axes=(1, 2)), axes=(1, 2))
    power = np.abs(spectrum) ** 2
    # Restore the negative-temporal-frequency partner for interior rFFT bins.
    temporal_weights = np.ones(power.shape[0], dtype=np.float64)
    temporal_weights[1:-1] = 2.0
    power *= temporal_weights[:, None, None]
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)
    positive = tf_hz > 0
    positive_power = power[positive].reshape(np.count_nonzero(positive), -1)
    positive_tf = tf_hz[positive]
    sf_bin, ori_bin, fitted = spatial_lookup(float(ppd), size=spatial_size)
    n_sf = len(SF_EDGES_CPD) - 1
    n_ori = len(ORIENTATION_EDGES_DEG) - 1
    radial = np.empty((len(positive_tf), n_sf), dtype=np.float32)
    oriented = np.empty((len(positive_tf), n_sf, n_ori), dtype=np.float32)
    joint_bin = sf_bin * n_ori + ori_bin
    for index, values in enumerate(positive_power):
        radial[index] = np.bincount(sf_bin, weights=values, minlength=n_sf).astype(np.float32)
        oriented[index] = np.bincount(
            joint_bin, weights=values, minlength=n_sf * n_ori
        ).reshape(n_sf, n_ori).astype(np.float32)

    core = positive_tf <= TF_CORE_MAX_HZ
    edge = (positive_tf > TF_CORE_MAX_HZ) & (positive_tf <= TF_EDGE_HZ)
    upper = positive_tf > TF_EDGE_HZ
    nyquist = np.isclose(positive_tf, FRAME_RATE_HZ / 2.0)
    scalar = np.asarray(
        [
            positive_power.sum(),
            positive_power[core].sum(),
            positive_power[edge].sum(),
            positive_power[upper].sum(),
            positive_power[nyquist].sum(),
            positive_power[:, fitted].sum(),
            positive_power[core][:, fitted].sum(),
            positive_power[edge][:, fitted].sum(),
            positive_power[upper][:, fitted].sum(),
            positive_power[nyquist][:, fitted].sum(),
        ],
        dtype=np.float64,
    )
    return radial, oriented, scalar


def load_image_patch(image_index: int, image_audit_dir: Path) -> np.ndarray:
    path = image_audit_dir / "partials" / f"image_{int(image_index):03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Corrected image partial is missing: {path}")
    with np.load(path) as data:
        patch = np.asarray(data["corrected_patch"], dtype=np.float32)
    return _standardize_uint_like(patch)


def existing_shard_is_valid(path: Path, trace_indices_expected: np.ndarray) -> bool:
    try:
        with np.load(path) as data:
            return (
                np.array_equal(data["trace_index"], trace_indices_expected)
                and data["radial_power"].shape[0] == len(trace_indices_expected)
                and data["orientation_power"].shape[0] == len(trace_indices_expected)
                and data["scalar_metrics"].shape == (len(trace_indices_expected), len(SCALAR_NAMES))
            )
    except Exception:
        return False


def plot_first_streamed_example(example_path: Path, out_dir: Path) -> None:
    """Make the concrete map-first artifact for the first streamed condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(example_path) as data:
        image_index = int(np.asarray(data["image_index"]).ravel()[0])
        trace_index = int(np.asarray(data["trace_index"]).ravel()[0])
        patch = np.asarray(data["source_patch"], dtype=float)
        trace = np.asarray(data["scored_retinal_trace"], dtype=float)
        movie = np.asarray(data["scored_movie"], dtype=float)
        radial = np.asarray(data["radial_power"], dtype=float)
        oriented = np.asarray(data["orientation_power"], dtype=float)

    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    ori_centers = 0.5 * (ORIENTATION_EDGES_DEG[:-1] + ORIENTATION_EDGES_DEG[1:])
    radial_scale = max(float(radial.max()), np.finfo(float).tiny)
    log_radial = 10.0 * np.log10(np.maximum(radial / radial_scale, 1e-4))
    orientation_tf = oriented.sum(axis=1)
    orientation_scale = max(float(orientation_tf.max()), np.finfo(float).tiny)
    log_orientation = 10.0 * np.log10(np.maximum(orientation_tf / orientation_scale, 1e-4))

    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.8), constrained_layout=True)
    axes[0].imshow(patch, cmap="gray", origin="lower")
    axes[0].set_title("Corrected source patch")
    axes[0].axis("off")
    axes[1].plot(trace[:, 0], trace[:, 1], color="0.2", lw=1.2)
    axes[1].scatter(trace[0, 0], trace[0, 1], s=24, color="#22aa88", label="start")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_xlabel("retinal shift x (deg)")
    axes[1].set_ylabel("retinal shift y (deg)")
    axes[1].set_title("What moves: image on retina")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].imshow(movie.mean(axis=0), cmap="gray", origin="lower")
    axes[2].set_title("Mean of 40 scored frames")
    axes[2].axis("off")
    mesh = axes[3].pcolormesh(sf_centers, tf_hz, log_radial, shading="nearest", cmap="magma", vmin=-40, vmax=0)
    axes[3].set_xscale("log")
    axes[3].axvspan(SF_FIT_MIN_CPD, SF_FIT_MAX_CPD, color="white", alpha=0.08)
    axes[3].axhline(TF_CORE_MAX_HZ, color="cyan", ls="--", lw=1)
    axes[3].set_xlabel("spatial frequency (cpd)")
    axes[3].set_ylabel("temporal frequency (Hz)")
    axes[3].set_title("FEM-created SF x TF power")
    fig.colorbar(mesh, ax=axes[3], label="relative power (dB)")
    mesh_ori = axes[4].pcolormesh(
        ori_centers, tf_hz, log_orientation, shading="nearest", cmap="magma", vmin=-40, vmax=0
    )
    axes[4].axhline(TF_CORE_MAX_HZ, color="cyan", ls="--", lw=1)
    axes[4].set_xlabel("Fourier orientation (deg)")
    axes[4].set_ylabel("temporal frequency (Hz)")
    axes[4].set_title("Orientation x TF power")
    fig.colorbar(mesh_ori, ax=axes[4], label="relative power (dB)")
    fig.suptitle(
        f"Input-only streaming example: image {image_index}, trace {trace_index}\n"
        "Spectra use the 40 scored lag-zero retinal frames; model history is not mixed into input power",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_dir / "first_streamed_spectral_example.png", dpi=180)
    fig.savefig(out_dir / "first_streamed_spectral_example.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_path = args.cohort_dir / "interim49_images.csv"
    trace_path = args.cohort_dir / "interim973_traces.csv"
    images = pd.read_csv(image_path).sort_values("image_index").reset_index(drop=True)
    traces = pd.read_csv(trace_path).sort_values("trace_index").reset_index(drop=True)
    if len(images) != 49 or images["image_index"].nunique() != 49:
        raise ValueError("Expected the frozen 49-image interim cohort")
    if len(traces) != 973 or traces["trace_index"].nunique() != 973:
        raise ValueError("Expected the frozen 973-trace interim cohort")
    if int(args.max_images) > 0:
        images = images.iloc[: int(args.max_images)].copy()
    if int(args.max_traces) > 0:
        traces = traces.iloc[: int(args.max_traces)].copy()
    requested_movies = len(images) * len(traces)
    if int(args.max_movies) > 0:
        requested_movies = min(requested_movies, int(args.max_movies))

    config = {
        "status": "input_only_spectral_cache_configured",
        "n_images": int(len(images)),
        "n_traces": int(len(traces)),
        "n_movies_requested": int(requested_movies),
        "device": str(args.device),
        "frame_rate_hz": FRAME_RATE_HZ,
        "history_frames_validated_not_spectralized": N_HISTORY,
        "scored_frames_spectralized": N_SCORE,
        "tf_hz": np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:].tolist(),
        "sf_edges_cpd": SF_EDGES_CPD.tolist(),
        "fourier_orientation_edges_deg": ORIENTATION_EDGES_DEG.tolist(),
        "temporal_window": "Hann",
        "spatial_window": "separable 2D Hann",
        "neural_model_calls": False,
        "sources": {
            "images": file_identity(image_path),
            "traces": file_identity(trace_path),
            "runner": file_identity(Path(__file__)),
            "renderer": file_identity(
                ROOT / "declan/fig4_active_sensing/input_only_retinal_renderer.py"
            ),
        },
    }
    if bool(args.dry_run):
        print(json.dumps(config, indent=2))
        return

    validation_manifest = require_validation_gate(args.validation_dir)
    if args.out_dir.exists() and not bool(args.resume):
        raise FileExistsError(f"Output exists and --no-resume was requested: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    config["actual_input_validation"] = validation_manifest.get("gates", {})
    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    config["implementation_status"] = "running_or_resumable"
    atomic_json(args.out_dir / "manifest.json", config)

    history, score, ppd_by_session = materialize_trace_arrays(traces)
    np.savez_compressed(
        args.out_dir / "corrected_trace_arrays.npz",
        trace_index=traces["trace_index"].to_numpy(int),
        source_row=traces["source_row"].to_numpy(int),
        history_crop_xy=history,
        score_crop_xy=score,
    )
    common = _load_twin_common()
    trace_ids = traces["trace_index"].to_numpy(int)
    source_ids = traces["source_row"].to_numpy(int)
    total_done = 0
    start_time = time.perf_counter()
    first_example_written = (args.out_dir / "first_streamed_example.npz").exists()

    for image in images.itertuples(index=False):
        remaining = requested_movies - total_done
        if remaining <= 0:
            break
        n_this = min(len(traces), remaining)
        expected_ids = trace_ids[:n_this]
        shard_path = shard_dir / f"image_{int(image.image_index):03d}.npz"
        if shard_path.exists() and existing_shard_is_valid(shard_path, expected_ids):
            total_done += n_this
            print(f"resume: image {image.image_index} ({total_done}/{requested_movies} movies)", flush=True)
            continue
        if shard_path.exists():
            raise RuntimeError(f"Existing shard failed validation; refusing overwrite: {shard_path}")
        patch = load_image_patch(int(image.image_index), args.image_audit_dir)
        ppd = ppd_by_session.get(str(image.session))
        if ppd is None:
            cache: dict = {}
            dset = load_dset(str(image.session), cache)
            ppd = float(dset.metadata["ppd"])
            del dset
            cache.clear()
            gc.collect()
        radial_rows: list[np.ndarray] = []
        orientation_rows: list[np.ndarray] = []
        scalar_rows: list[np.ndarray] = []
        with torch.no_grad():
            for trace_position in range(n_this):
                # Power is calculated from scored lag-zero frames. Recorded
                # history is intentionally excluded because it affects model
                # state, not the retinal movie during the scored interval.
                retinal_score = -score[trace_position]
                movie_tensor = render_retinal_frames_lag_zero(
                    common,
                    patch,
                    retinal_score,
                    ppd=float(ppd),
                    device=str(args.device),
                )
                movie = movie_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                radial, oriented, scalar = spectral_statistics(movie, ppd=float(ppd))
                radial_rows.append(radial)
                orientation_rows.append(oriented)
                scalar_rows.append(scalar)
                if not first_example_written:
                    atomic_npz(
                        args.out_dir / "first_streamed_example.npz",
                        image_index=np.asarray([int(image.image_index)]),
                        trace_index=np.asarray([int(trace_ids[trace_position])]),
                        source_patch=patch,
                        scored_retinal_trace=retinal_score,
                        scored_movie=movie,
                        radial_power=radial,
                        orientation_power=oriented,
                    )
                    first_example_written = True
                del movie_tensor, movie
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
        atomic_npz(
            shard_path,
            image_index=np.asarray([int(image.image_index)], dtype=int),
            image_source_row=np.asarray([int(image.source_row)], dtype=int),
            trace_index=expected_ids,
            trace_source_row=source_ids[:n_this],
            radial_power=np.stack(radial_rows).astype(np.float32),
            orientation_power=np.stack(orientation_rows).astype(np.float32),
            scalar_metrics=np.stack(scalar_rows).astype(np.float64),
        )
        total_done += n_this
        elapsed = time.perf_counter() - start_time
        print(
            f"image {image.image_index}: {total_done}/{requested_movies} movies; "
            f"{elapsed / max(total_done, 1):.3f} s/movie",
            flush=True,
        )

    movie_rows: list[dict[str, object]] = []
    completed_shards = 0
    for image in images.itertuples(index=False):
        shard_path = shard_dir / f"image_{int(image.image_index):03d}.npz"
        if not shard_path.exists():
            continue
        with np.load(shard_path) as data:
            metrics = np.asarray(data["scalar_metrics"], dtype=float)
            shard_trace_ids = np.asarray(data["trace_index"], dtype=int)
            shard_source_ids = np.asarray(data["trace_source_row"], dtype=int)
        completed_shards += 1
        for row_index, (trace_index, trace_source) in enumerate(zip(shard_trace_ids, shard_source_ids, strict=True)):
            record: dict[str, object] = {
                "image_index": int(image.image_index),
                "image_source_row": int(image.source_row),
                "trace_index": int(trace_index),
                "trace_source_row": int(trace_source),
            }
            record.update({name: float(metrics[row_index, column]) for column, name in enumerate(SCALAR_NAMES)})
            total = max(float(record["total_positive_tf_power"]), np.finfo(float).tiny)
            record["fraction_le_32_all_sf"] = float(record["power_le_32_all_sf"]) / total
            record["fraction_32_45p25_all_sf"] = float(record["power_32_45p25_all_sf"]) / total
            record["fraction_45p25_60_all_sf"] = float(record["power_45p25_60_all_sf"]) / total
            movie_rows.append(record)
    movie_table = pd.DataFrame(movie_rows)
    movie_table.to_csv(args.out_dir / "movie_spectral_metrics.csv", index=False)
    example_path = args.out_dir / "first_streamed_example.npz"
    if example_path.exists():
        plot_first_streamed_example(example_path, args.out_dir)

    elapsed = time.perf_counter() - start_time
    full_request = (
        int(args.max_images) == 0 and int(args.max_traces) == 0 and int(args.max_movies) == 0
    )
    complete = len(movie_table) == requested_movies
    final_status = (
        "input_only_spectral_cache_complete"
        if full_request and complete
        else "bounded_streaming_smoke_complete"
        if complete
        else "incomplete_resumable"
    )
    final_manifest = {
        **config,
        "status": final_status,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_status": final_status,
        "counts": {
            "completed_movies": int(len(movie_table)),
            "requested_movies": int(requested_movies),
            "completed_image_shards": int(completed_shards),
        },
        "elapsed_seconds_this_invocation": float(elapsed),
        "seconds_per_new_movie_this_invocation": float(elapsed / max(total_done, 1)),
        "outputs": {
            "movie_metrics": str((args.out_dir / "movie_spectral_metrics.csv").resolve()),
            "shard_dir": str(shard_dir.resolve()),
            "trace_arrays": str((args.out_dir / "corrected_trace_arrays.npz").resolve()),
            "first_example": str((args.out_dir / "first_streamed_example.npz").resolve()),
            "first_example_figure": str(
                (args.out_dir / "first_streamed_spectral_example.pdf").resolve()
            ),
        },
        "guardrail": "Input spectra only; no frozen-twin model or neural response was loaded or scored.",
    }
    atomic_json(args.out_dir / "manifest.json", final_manifest)
    print(json.dumps(final_manifest, indent=2))


if __name__ == "__main__":
    main()

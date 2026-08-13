#!/usr/bin/env python3
"""Audit the eye-trace contract and 32--60 Hz power for the RR100 Figure-4 analysis.

This checkpoint distinguishes four objects that were previously conflated:

1. raw 240-Hz ``eyepos`` samples;
2. the 120-Hz visual sampling used by the frozen model (even-frame decimation);
3. the shifter-corrected crop trajectory stored as ``dpi_pix``;
4. the already-materialized 51x51 BackImage frames used by the model.

Temporal power is computed with a true real-input one-sided FFT, including the
60-Hz Nyquist coefficient.  The script preserves both exact recorded movies
and crossed image x trace reconstructions so claims can be traced to the model
input as well as to the Figure-4 counterfactual construction.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from DataYatesV1.utils.data.datasets import DictDataset

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.analyze_rr100_corrected_figure4_cache import (
    RUN,
    TRACE_CACHE,
    render_with_common,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1"
FRAME_RATE_HZ = 120.0
SOURCE_RATE_HZ = 240.0
N_FRAMES = 32
SF_MIN_CPD = 1.0
SF_MAX_CPD = float(8.0 * np.sqrt(2.0))
TF_CORE_MAX_HZ = 32.0
TF_EDGE_HZ = float(32.0 * np.sqrt(2.0))
EPS = 1e-20
CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_12-03-23_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga1/"
    "epoch=193-val_bps_overall=0.6000.ckpt"
)
TF45_AUDIT = ROOT / "outputs/fig4_active_sensing/rr100_tf45_edge_support_audit_v1/tf45_edge_unit_audit.csv"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def model_aligned_indices(window_start: int, window_stop: int) -> np.ndarray:
    """Return 32 even global source indices matching loader ``stim[::2]`` parity."""
    window_start, window_stop = int(window_start), int(window_stop)
    if window_stop - window_start < 2 * N_FRAMES:
        raise ValueError("Source window is too short for 32 model frames")
    center = (window_start + window_stop) // 2
    start = center - N_FRAMES
    if start % 2:
        start -= 1
    first_even = window_start if window_start % 2 == 0 else window_start + 1
    start = max(first_even, start)
    if start + 2 * N_FRAMES - 1 >= window_stop:
        start = window_stop - 2 * N_FRAMES
        if start % 2:
            start -= 1
    indices = np.arange(start, start + 2 * N_FRAMES, 2, dtype=int)
    if indices.shape != (N_FRAMES,) or indices[0] % 2 or indices[-1] >= window_stop:
        raise AssertionError("Could not produce loader-aligned 120-Hz indices")
    return indices


def corrected_crop_xy_deg(dset: DictDataset) -> np.ndarray:
    """Convert shifter-corrected ``dpi_pix`` to the x/y degree convention of eyepos."""
    dpi_pix = np.asarray(dset["dpi_pix"], dtype=np.float64)
    ppd = float(dset.metadata["ppd"])
    screen_resolution = np.asarray(dset.metadata["screen_resolution"], dtype=float)
    center_pix = np.flipud((screen_resolution + 1.0) / 2.0)
    return np.column_stack(
        [dpi_pix[:, 1] - center_pix[1], -(dpi_pix[:, 0] - center_pix[0])]
    ) / ppd


def centered(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    return trace - np.mean(trace, axis=0, keepdims=True)


def rfft_power(movie: np.ndarray, *, ppd: float, temporal_window: str = "hann") -> dict[str, np.ndarray]:
    """One-sided spatiotemporal power with the even-length Nyquist bin retained."""
    arr = np.asarray(movie, dtype=np.float64)
    if arr.shape[0] != N_FRAMES:
        raise ValueError(f"Expected {N_FRAMES} frames, got {arr.shape}")
    residual = arr - arr.mean(axis=0, keepdims=True)
    if temporal_window == "hann":
        tw = np.hanning(arr.shape[0])
    elif temporal_window == "rectangular":
        tw = np.ones(arr.shape[0])
    else:
        raise ValueError(temporal_window)
    if arr.ndim == 3:
        sw = np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2]))
        weighted = residual * tw[:, None, None] * sw[None, :, :]
        temporal_fft = np.fft.rfft(weighted, axis=0)
        spectrum = np.fft.fftshift(np.fft.fft2(temporal_fft, axes=(1, 2)), axes=(1, 2))
        fy = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=1.0 / float(ppd)))
        fx = np.fft.fftshift(np.fft.fftfreq(arr.shape[2], d=1.0 / float(ppd)))
        radial_sf = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    elif arr.ndim == 2:
        weighted = residual * tw[:, None]
        spectrum = np.fft.rfft(weighted, axis=0)
        radial_sf = np.zeros(arr.shape[1:], dtype=float)
    else:
        raise ValueError(f"Unsupported input shape {arr.shape}")
    power = np.abs(spectrum) ** 2
    # Restore the negative-frequency partner energy for interior rFFT bins.
    weights = np.ones(power.shape[0], dtype=float)
    if power.shape[0] > 2:
        weights[1:-1] = 2.0
    power *= weights.reshape((-1,) + (1,) * (power.ndim - 1))
    tf = np.fft.rfftfreq(arr.shape[0], d=1.0 / FRAME_RATE_HZ)
    return {"power": power, "tf_hz": tf, "radial_sf_cpd": radial_sf}


def power_metrics(decomp: dict[str, np.ndarray]) -> dict[str, float]:
    power = np.asarray(decomp["power"], dtype=float)
    tf = np.asarray(decomp["tf_hz"], dtype=float)
    sf = np.asarray(decomp["radial_sf_cpd"], dtype=float)
    pos = tf > 0
    core = pos & (tf <= TF_CORE_MAX_HZ)
    band_32_45 = (tf > TF_CORE_MAX_HZ) & (tf <= TF_EDGE_HZ)
    band_45_60 = (tf > TF_EDGE_HZ) & (tf <= FRAME_RATE_HZ / 2.0)
    band_32_60 = tf > TF_CORE_MAX_HZ
    nyquist = np.isclose(tf, FRAME_RATE_HZ / 2.0)
    total = float(power[pos].sum())
    out = {
        "total_positive_tf_power": total,
        "power_le_32_all_sf": float(power[core].sum()),
        "power_32_45p25_all_sf": float(power[band_32_45].sum()),
        "power_45p25_60_all_sf": float(power[band_45_60].sum()),
        "power_32_60_all_sf": float(power[band_32_60].sum()),
        "power_at_60_all_sf": float(power[nyquist].sum()),
    }
    if power.ndim == 3:
        sf_ok = (sf >= SF_MIN_CPD) & (sf <= SF_MAX_CPD)
        total_sf = float(power[pos][:, sf_ok].sum())
        out.update({
            "total_positive_tf_power_fitted_sf": total_sf,
            "power_le_32_fitted_sf": float(power[core][:, sf_ok].sum()),
            "power_32_45p25_fitted_sf": float(power[band_32_45][:, sf_ok].sum()),
            "power_45p25_60_fitted_sf": float(power[band_45_60][:, sf_ok].sum()),
            "power_32_60_fitted_sf": float(power[band_32_60][:, sf_ok].sum()),
            "power_at_60_fitted_sf": float(power[nyquist][:, sf_ok].sum()),
        })
    for key, value in list(out.items()):
        if key.startswith("power_") and key.endswith("_all_sf"):
            out["fraction_" + key.removeprefix("power_")] = value / max(total, EPS)
        if key.startswith("power_") and key.endswith("_fitted_sf"):
            out["fraction_" + key.removeprefix("power_") + "_of_fitted_sf"] = value / max(
                out["total_positive_tf_power_fitted_sf"], EPS
            )
            out["fraction_" + key.removeprefix("power_") + "_of_all_power"] = value / max(total, EPS)
    return out


def trace_metrics(trace: np.ndarray) -> dict[str, float]:
    trace = centered(trace)
    decomp = rfft_power(trace, ppd=1.0)
    metrics = power_metrics(decomp)
    steps = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    metrics.update({
        "median_step_arcmin": float(np.median(steps) * 60.0),
        "p95_speed_dps": float(np.percentile(steps * FRAME_RATE_HZ, 95)),
        "path_length_arcmin": float(steps.sum() * 60.0),
        "rms_radius_arcmin": float(np.sqrt(np.mean(np.sum(trace**2, axis=1))) * 60.0),
    })
    return metrics


def load_dset(session: str, cache: dict[str, DictDataset]) -> DictDataset:
    if session not in cache:
        path = Path("/mnt/ssd/YatesMarmoV1/processed") / session / "datasets/backimage.dset"
        cache[session] = DictDataset.load(path)
    return cache[session]


def build_trace_variants(source_rows: pd.DataFrame, trace_source_ids: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    dsets: dict[str, DictDataset] = {}
    current = np.load(TRACE_CACHE)["trace"].astype(np.float64)
    variants: dict[str, list[np.ndarray]] = {
        "current_consecutive_raw_eyepos_labeled_120hz": [],
        "visual_even_decimated_raw_eyepos": [],
        "visual_even_decimated_corrected_crop": [],
        "behavior_pair_averaged_raw_eyepos": [],
        "savgol11_then_even_decimated_corrected_crop": [],
    }
    rows: list[dict[str, Any]] = []
    for trace_index, source_id in enumerate(trace_source_ids.astype(int)):
        row = source_row_by_id(source_rows, int(source_id))
        dset = load_dset(str(row["session"]), dsets)
        indices = model_aligned_indices(int(row["global_start"]), int(row["global_stop"]))
        start = int(indices[0])
        raw64 = np.asarray(dset["eyepos"], dtype=np.float64)[start : start + 64]
        crop_all = corrected_crop_xy_deg(dset)
        crop64 = crop_all[start : start + 64]
        # Filter locally with five context samples on each side when available.
        pad_start = max(0, start - 5)
        pad_stop = min(len(crop_all), start + 64 + 5)
        local = crop_all[pad_start:pad_stop]
        smooth = savgol_filter(local, 11, 3, axis=0)
        offset = start - pad_start
        smooth32 = smooth[offset : offset + 64 : 2]
        item = {
            "current_consecutive_raw_eyepos_labeled_120hz": current[trace_index],
            "visual_even_decimated_raw_eyepos": raw64[::2],
            "visual_even_decimated_corrected_crop": crop64[::2],
            "behavior_pair_averaged_raw_eyepos": raw64.reshape(32, 2, 2).mean(axis=1),
            "savgol11_then_even_decimated_corrected_crop": smooth32,
        }
        for contract, trace in item.items():
            trace = centered(trace)
            variants[contract].append(trace)
            rows.append({
                "trace_index": int(trace_index),
                "source_row": int(source_id),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "source_window_start": int(row["global_start"]),
                "source_window_stop": int(row["global_stop"]),
                "model_raw_start": start,
                "model_raw_stop_exclusive": start + 64,
                "global_decimation_parity": "even",
                "trace_contract": contract,
                **trace_metrics(trace),
            })
    return pd.DataFrame(rows), {k: np.stack(v) for k, v in variants.items()}


def exact_recorded_movie_audit(source_rows: pd.DataFrame, source_ids: list[int]) -> pd.DataFrame:
    dsets: dict[str, DictDataset] = {}
    rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        row = source_row_by_id(source_rows, int(source_id))
        dset = load_dset(str(row["session"]), dsets)
        indices = model_aligned_indices(int(row["global_start"]), int(row["global_stop"]))
        movie = np.asarray(dset["stim"], dtype=np.float64)[indices]
        for window in ("hann", "rectangular"):
            rows.append({
                "source_row": int(source_id),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "movie_contract": "exact_backimage_dset_stim_even_decimated",
                "temporal_window": window,
                **power_metrics(rfft_power(movie, ppd=float(dset.metadata["ppd"]), temporal_window=window)),
            })
    return pd.DataFrame(rows)


def crossed_reconstruction_audit(
    source_rows: pd.DataFrame,
    image_table: pd.DataFrame,
    traces: dict[str, np.ndarray],
) -> pd.DataFrame:
    cache_path = OUT / "crossed_reconstruction_power_metrics.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    common = _load_twin_common()
    rows: list[dict[str, Any]] = []
    canvas_cache: dict = {}
    dset_cache: dict[str, DictDataset] = {}
    contracts = (
        "current_consecutive_raw_eyepos_labeled_120hz",
        "visual_even_decimated_corrected_crop",
    )
    for image_number, image in enumerate(image_table.sort_values("image_index").itertuples(), start=1):
        source = source_row_by_id(source_rows, int(image.source_row))
        old_patch, old_patch_meta = _extract_patch(source, canvas_cache=canvas_cache, patch_size_px=540)
        dset = load_dset(str(source["session"]), dset_cache)
        image_indices = model_aligned_indices(int(source["global_start"]), int(source["global_stop"]))
        image_crop_xy = corrected_crop_xy_deg(dset)[image_indices]
        roi_src = np.asarray(dset.metadata["roi_src"], dtype=float)
        roi_center_yx = (roi_src[:, 0] + roi_src[:, 1] - 1.0) / 2.0
        roi_offset_xy_deg = np.asarray(
            [roi_center_yx[1], -roi_center_yx[0]], dtype=float
        ) / float(dset.metadata["ppd"])
        corrected_source = source.copy()
        corrected_center = image_crop_xy.mean(axis=0) + roi_offset_xy_deg
        corrected_source["mean_x_deg"] = float(corrected_center[0])
        corrected_source["mean_y_deg"] = float(corrected_center[1])
        corrected_patch, corrected_patch_meta = _extract_patch(
            corrected_source, canvas_cache=canvas_cache, patch_size_px=540
        )
        for contract in contracts:
            if contract == "current_consecutive_raw_eyepos_labeled_120hz":
                patch = old_patch
                ppd = float(old_patch_meta["patch_ppd"])
                applied_sign = 1.0
                geometry_contract = "legacy_gaze_centered_patch_and_legacy_trace_sign"
            else:
                patch = corrected_patch
                ppd = float(corrected_patch_meta["patch_ppd"])
                # Positive gaze/crop displacement moves the sampled retinal image in the
                # opposite direction. This sign reproduces saved backimage.dset frames.
                applied_sign = -1.0
                geometry_contract = "rf_crop_centered_patch_and_retinal_translation_sign"
            for trace_index, trace in enumerate(traces[contract]):
                movie = render_with_common(
                    np.asarray(patch, np.float32), applied_sign * trace, ppd=ppd, common=common
                )
                rows.append({
                    "image_index": int(image.image_index),
                    "image_source_row": int(image.source_row),
                    "trace_index": int(trace_index),
                    "trace_contract": contract,
                    "geometry_contract": geometry_contract,
                    **power_metrics(rfft_power(movie, ppd=ppd)),
                })
                del movie
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        gc.collect()
        print(f"crossed reconstruction: image {image_number}/{len(image_table)}", flush=True)
    return pd.DataFrame(rows)


def renderer_validation(source_rows: pd.DataFrame, image_table: pd.DataFrame) -> pd.DataFrame:
    """Compare reconstructed original pairs directly with saved model-input frames."""
    common = _load_twin_common()
    canvas_cache: dict = {}
    dset_cache: dict[str, DictDataset] = {}
    rows: list[dict[str, Any]] = []
    for image in image_table.sort_values("image_index").itertuples():
        source = source_row_by_id(source_rows, int(image.source_row))
        dset = load_dset(str(source["session"]), dset_cache)
        indices = model_aligned_indices(int(source["global_start"]), int(source["global_stop"]))
        crop_xy = corrected_crop_xy_deg(dset)[indices]
        roi_src = np.asarray(dset.metadata["roi_src"], dtype=float)
        roi_center_yx = (roi_src[:, 0] + roi_src[:, 1] - 1.0) / 2.0
        roi_offset_xy_deg = np.asarray(
            [roi_center_yx[1], -roi_center_yx[0]], dtype=float
        ) / float(dset.metadata["ppd"])
        adjusted = source.copy()
        center_xy = crop_xy.mean(axis=0) + roi_offset_xy_deg
        adjusted["mean_x_deg"] = float(center_xy[0])
        adjusted["mean_y_deg"] = float(center_xy[1])
        patch, meta = _extract_patch(adjusted, canvas_cache=canvas_cache, patch_size_px=540)
        reconstructed = render_with_common(
            np.asarray(patch, np.float32), -centered(crop_xy),
            ppd=float(meta["patch_ppd"]), common=common,
        )
        exact = np.asarray(dset["stim"], dtype=np.float32)[indices]
        pixel_r = float(np.corrcoef(reconstructed.ravel(), exact.ravel())[0, 1])
        rec_power = power_metrics(rfft_power(reconstructed, ppd=float(meta["patch_ppd"])))
        exact_power = power_metrics(rfft_power(exact, ppd=float(dset.metadata["ppd"])))
        rows.append({
            "image_index": int(image.image_index),
            "source_row": int(image.source_row),
            "session": str(source["session"]),
            "pixel_r": pixel_r,
            "pixel_mae_uint8_scale": float(np.mean(np.abs(reconstructed - exact))),
            "reconstructed_fraction_32_60_all_sf": rec_power["fraction_32_60_all_sf"],
            "exact_fraction_32_60_all_sf": exact_power["fraction_32_60_all_sf"],
            "reconstructed_fraction_32_60_fitted_sf": rec_power["fraction_32_60_fitted_sf_of_fitted_sf"],
            "exact_fraction_32_60_fitted_sf": exact_power["fraction_32_60_fitted_sf_of_fitted_sf"],
        })
    return pd.DataFrame(rows)


def temporal_frontend_response() -> pd.DataFrame:
    import torch

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    key = "model.frontend.temporal_conv.conv.parametrizations.weight.original"
    weights = checkpoint["state_dict"][key].detach().cpu().numpy()[:, 0, :, 0, 0]
    frequencies = np.linspace(0.0, 60.0, 601)
    time = np.arange(weights.shape[1], dtype=float)
    transfer = np.exp(-2j * np.pi * frequencies[:, None] * time[None, :] / FRAME_RATE_HZ) @ weights.T
    energy = np.sum(np.abs(transfer) ** 2, axis=1)
    energy /= max(float(np.max(energy)), EPS)
    return pd.DataFrame({"temporal_hz": frequencies, "normalized_combined_frontend_energy": energy})


def distribution_summary(frame: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for label, part in frame.groupby(group, sort=False):
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            rows.append({
                group: label,
                "metric": metric,
                "n": int(len(values)),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "mean": float(values.mean()),
            })
    return pd.DataFrame(rows)


def plot_trace_conditioning(metrics: pd.DataFrame, traces: dict[str, np.ndarray]) -> None:
    primary = metrics.loc[metrics["trace_contract"].eq("visual_even_decimated_corrected_crop")].copy()
    median_row = primary.loc[
        (primary["fraction_32_60_all_sf"] - primary["fraction_32_60_all_sf"].median()).abs().idxmin()
    ]
    high_row = primary.loc[primary["fraction_32_60_all_sf"].idxmax()]
    roles = {
        "median corrected trace": int(median_row["trace_index"]),
        "largest corrected high-TF fraction": int(high_row["trace_index"]),
        "largest saved-to-corrected reduction": int(
            metrics.pivot(index="trace_index", columns="trace_contract", values="fraction_32_60_all_sf")
            .assign(delta=lambda d: d["current_consecutive_raw_eyepos_labeled_120hz"] - d["visual_even_decimated_corrected_crop"])
            ["delta"].idxmax()
        ),
    }
    contracts = [
        "current_consecutive_raw_eyepos_labeled_120hz",
        "visual_even_decimated_corrected_crop",
        "savgol11_then_even_decimated_corrected_crop",
    ]
    colors = ["#D55E00", "#0072B2", "#009E73"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.2), constrained_layout=True)
    selections = []
    for row_idx, (role, trace_index) in enumerate(roles.items()):
        selections.append({"selection_role": role, "trace_index": trace_index})
        ax = axes[row_idx, 0]
        for contract, color in zip(contracts, colors):
            tr = traces[contract][trace_index] * 60.0
            ax.plot(tr[:, 0], tr[:, 1], color=color, lw=1.2, alpha=0.9, label=contract.replace("_", " "))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="horizontal position (arcmin)", ylabel="vertical position (arcmin)", title=f"{role} · trace {trace_index}")
        if row_idx == 0:
            ax.legend(fontsize=7, loc="best")
        ax = axes[row_idx, 1]
        t_ms = np.arange(N_FRAMES) / FRAME_RATE_HZ * 1000.0
        for contract, color in zip(contracts, colors):
            tr = traces[contract][trace_index] * 60.0
            ax.plot(t_ms, tr[:, 0], color=color, lw=1.1)
        ax.set(xlabel="time (ms)", ylabel="horizontal position (arcmin)", title="Same samples in time")
        ax = axes[row_idx, 2]
        for contract, color in zip(contracts, colors):
            dec = rfft_power(traces[contract][trace_index], ppd=1.0)
            p = dec["power"].sum(axis=1)
            p /= max(float(p[1:].sum()), EPS)
            ax.plot(dec["tf_hz"][1:], p[1:], marker="o", ms=3, color=color, lw=1.1)
        ax.axvspan(32, 60, color="#E69F00", alpha=0.12)
        ax.axvline(32, color="0.35", ls="--", lw=1)
        ax.set(xlabel="temporal frequency (Hz)", ylabel="fraction of positive-TF position power", title="Trace spectrum (60-Hz bin included)")
    fig.suptitle("Eye-trace conditioning changes the inferred high-frequency motion", fontsize=15)
    fig.savefig(OUT / "trace_conditioning_examples.png", dpi=220)
    fig.savefig(OUT / "trace_conditioning_examples.pdf")
    plt.close(fig)
    pd.DataFrame(selections).to_csv(OUT / "selected_trace_examples.csv", index=False)


def plot_power_summary(exact: pd.DataFrame, crossed: pd.DataFrame, frontend: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4), constrained_layout=True)
    ax = axes[0, 0]
    groups = []
    labels = []
    for contract, label in (
        ("current_consecutive_raw_eyepos_labeled_120hz", "old reconstruction\nwrong source-rate contract"),
        ("visual_even_decimated_corrected_crop", "corrected crossed\nreconstruction"),
    ):
        groups.append(100.0 * crossed.loc[crossed.trace_contract.eq(contract), "fraction_32_60_all_sf"].to_numpy(float))
        labels.append(label)
    groups.append(100.0 * exact.loc[exact.temporal_window.eq("hann"), "fraction_32_60_all_sf"].to_numpy(float))
    labels.append("exact model input\nrecorded pairs")
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#D55E00", "#0072B2", "#009E73"]):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    ax.set(ylabel="power between >32 and 60 Hz (%)", title="A  Correct conditioning reduces—but does not eliminate—high-TF power")

    corrected = crossed.loc[crossed.trace_contract.eq("visual_even_decimated_corrected_crop")]
    ax = axes[0, 1]
    band_cols = ["fraction_le_32_all_sf", "fraction_32_45p25_all_sf", "fraction_45p25_60_all_sf"]
    names = ["≤32", "32–45.25", "45.25–60"]
    med = 100.0 * corrected[band_cols].median().to_numpy(float)
    q1 = 100.0 * corrected[band_cols].quantile(0.25).to_numpy(float)
    q3 = 100.0 * corrected[band_cols].quantile(0.75).to_numpy(float)
    ax.bar(names, med, color=["#56B4E9", "#E69F00", "#CC79A7"])
    ax.errorbar(np.arange(3), med, yerr=np.vstack([med-q1, q3-med]), fmt="none", color="black", capsize=4)
    ax.set(ylabel="fraction of all positive-TF power (%)", title="B  Corrected crossed movies: temporal-band allocation")

    ax = axes[1, 0]
    vals_all = 100.0 * corrected["fraction_32_60_all_sf"].to_numpy(float)
    vals_sf = 100.0 * corrected["fraction_32_60_fitted_sf_of_fitted_sf"].to_numpy(float)
    ax.scatter(vals_all, vals_sf, s=12, alpha=0.35, color="#0072B2", edgecolor="none")
    ax.plot([0, 100], [0, 100], "--", color="0.5", lw=1)
    ax.set(xlabel="32–60 Hz fraction, all spatial frequencies (%)", ylabel="32–60 Hz fraction within 1–11.31 cpd (%)", title="C  High-TF power also lies inside measured SF support")

    ax = axes[1, 1]
    ax.plot(frontend["temporal_hz"], frontend["normalized_combined_frontend_energy"], color="#6A3D9A", lw=2, label="learned 16-frame temporal frontend")
    ax.axvspan(32, 60, color="#E69F00", alpha=0.12, label="currently under-measured band")
    ax.axvline(32, color="0.35", ls="--", lw=1)
    if TF45_AUDIT.exists():
        edge = pd.read_csv(TF45_AUDIT)
        informative = edge.loc[edge["observed_mean_positive_f0_32_hz"] > 0.1]
        ratio = float(informative["observed_45_to_32_mean_ratio"].median())
        ax.text(0.03, 0.05, f"Measured F0 at 45.25 Hz remains\n{ratio:.2f}× the 32-Hz response (median)", transform=ax.transAxes, va="bottom")
    ax.set(xlabel="temporal frequency (Hz)", ylabel="normalized combined filter energy", title="D  The frozen model has nontrivial high-TF sensitivity")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("RR100 eye-trace and temporal-support audit", fontsize=15)
    fig.savefig(OUT / "eye_trace_and_32_60hz_power_audit.png", dpi=220)
    fig.savefig(OUT / "eye_trace_and_32_60hz_power_audit.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    summary = json.loads((RUN / "summary.json").read_text())
    source_csv = Path(summary["source_csv"])
    source_rows = load_source_rows(source_csv)
    image_table = pd.read_csv(RUN / "image_feature_table.csv").sort_values("image_index")
    trace_npz = np.load(TRACE_CACHE)
    trace_ids = trace_npz["trace_source_row"].astype(int)

    trace_table, traces = build_trace_variants(source_rows, trace_ids)
    trace_table.to_csv(OUT / "trace_conditioning_metrics.csv", index=False)
    np.savez_compressed(OUT / "trace_conditioning_arrays.npz", **traces)

    exact_ids = list(dict.fromkeys(image_table["source_row"].astype(int).tolist() + trace_ids.tolist()))
    exact = exact_recorded_movie_audit(source_rows, exact_ids)
    exact.to_csv(OUT / "exact_recorded_model_input_power_metrics.csv", index=False)

    crossed = crossed_reconstruction_audit(source_rows, image_table, traces)
    validation = renderer_validation(source_rows, image_table)
    validation.to_csv(OUT / "renderer_vs_exact_model_input_validation.csv", index=False)
    frontend = temporal_frontend_response()
    frontend.to_csv(OUT / "learned_temporal_frontend_frequency_response.csv", index=False)

    trace_summary = distribution_summary(trace_table, "trace_contract", [
        "fraction_32_60_all_sf", "fraction_at_60_all_sf", "median_step_arcmin", "p95_speed_dps"
    ])
    exact_summary = distribution_summary(exact, "temporal_window", [
        "fraction_32_60_all_sf", "fraction_32_60_fitted_sf_of_fitted_sf",
        "fraction_le_32_fitted_sf_of_all_power", "fraction_32_45p25_fitted_sf_of_all_power",
        "fraction_45p25_60_fitted_sf_of_all_power",
    ])
    crossed_summary = distribution_summary(crossed, "trace_contract", [
        "fraction_32_60_all_sf", "fraction_32_60_fitted_sf_of_fitted_sf",
        "fraction_le_32_fitted_sf_of_all_power", "fraction_32_45p25_fitted_sf_of_all_power",
        "fraction_45p25_60_fitted_sf_of_all_power",
    ])
    trace_summary.to_csv(OUT / "trace_conditioning_summary.csv", index=False)
    exact_summary.to_csv(OUT / "exact_recorded_power_summary.csv", index=False)
    crossed_summary.to_csv(OUT / "crossed_reconstruction_power_summary.csv", index=False)

    plot_trace_conditioning(trace_table, traces)
    plot_power_summary(exact, crossed, frontend)

    def med(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
        return float(frame.loc[mask, column].median())

    corrected_mask = crossed.trace_contract.eq("visual_even_decimated_corrected_crop")
    old_mask = crossed.trace_contract.eq("current_consecutive_raw_eyepos_labeled_120hz")
    exact_hann = exact.temporal_window.eq("hann")
    edge_ratio = None
    if TF45_AUDIT.exists():
        edge = pd.read_csv(TF45_AUDIT)
        edge = edge.loc[edge.observed_mean_positive_f0_32_hz > 0.1]
        edge_ratio = float(edge.observed_45_to_32_mean_ratio.median())
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "primary_contract": (
            "shifter-corrected crop trajectory derived from backimage.dset dpi_pix, sampled at even "
            "240-Hz source indices to match visual stim[::2], with the session roi_src center added to "
            "the base patch and the mean-centered trajectory applied with retinal-translation sign"
        ),
        "critical_provenance_findings": [
            "The previous trace bank took 32 consecutive 240-Hz eyepos samples and labeled them 120 Hz.",
            "The model loader decimates visual stim on global even indices; it pair-averages eyepos only for the behavior covariate.",
            "BackImage stim was cropped with shifter-corrected dpi_pix, whereas eyepos stores the uncorrected gaze covariate.",
            "The legacy counterfactual reconstruction omitted the session RF-crop offset and used the opposite retinal-translation sign.",
            "The prior full FFT discarded the +60-Hz Nyquist coefficient because np.fft.fftfreq represents it at -60 Hz.",
        ],
        "power_results": {
            "old_crossed_median_fraction_32_60_all_sf": med(crossed, old_mask, "fraction_32_60_all_sf"),
            "corrected_crossed_median_fraction_32_60_all_sf": med(crossed, corrected_mask, "fraction_32_60_all_sf"),
            "corrected_crossed_median_fraction_32_60_within_fitted_sf": med(crossed, corrected_mask, "fraction_32_60_fitted_sf_of_fitted_sf"),
            "corrected_crossed_median_fraction_le32_fitted_sf_of_all_power": med(crossed, corrected_mask, "fraction_le_32_fitted_sf_of_all_power"),
            "corrected_crossed_median_fraction_32_45p25_fitted_sf_of_all_power": med(crossed, corrected_mask, "fraction_32_45p25_fitted_sf_of_all_power"),
            "corrected_crossed_median_fraction_45p25_60_fitted_sf_of_all_power": med(crossed, corrected_mask, "fraction_45p25_60_fitted_sf_of_all_power"),
            "exact_recorded_median_fraction_32_60_all_sf": med(exact, exact_hann, "fraction_32_60_all_sf"),
            "exact_recorded_median_fraction_32_60_within_fitted_sf": med(exact, exact_hann, "fraction_32_60_fitted_sf_of_fitted_sf"),
        },
        "renderer_validation": {
            "n_original_image_pairs": int(len(validation)),
            "median_pixel_r_reconstructed_vs_saved_stim": float(validation.pixel_r.median()),
            "minimum_pixel_r_reconstructed_vs_saved_stim": float(validation.pixel_r.min()),
            "median_abs_difference_in_32_60_fraction_all_sf": float(
                np.median(np.abs(
                    validation.reconstructed_fraction_32_60_all_sf
                    - validation.exact_fraction_32_60_all_sf
                ))
            ),
        },
        "model_support_evidence": {
            "sampling_rate_hz": 120.0,
            "nyquist_hz": 60.0,
            "temporal_frontend_kernel_frames": 16,
            "normalized_combined_frontend_energy_at_32_hz": float(np.interp(32.0, frontend.temporal_hz, frontend.normalized_combined_frontend_energy)),
            "normalized_combined_frontend_energy_at_45p25_hz": float(np.interp(TF_EDGE_HZ, frontend.temporal_hz, frontend.normalized_combined_frontend_energy)),
            "normalized_combined_frontend_energy_at_60_hz": float(np.interp(60.0, frontend.temporal_hz, frontend.normalized_combined_frontend_energy)),
            "median_observed_f0_45p25_to_32_ratio_in_informative_units": edge_ratio,
        },
        "recommendation": (
            "Extend the fixed-retina F0 probe densely above 32 Hz. Include points through 56 Hz; retain 60 Hz "
            "as a separately labeled Nyquist-edge control because a sampled drifting sinusoid is phase-degenerate there."
        ),
        "sources": {
            "source_csv": file_identity(source_csv),
            "old_trace_cache": file_identity(TRACE_CACHE),
            "model_checkpoint": file_identity(CHECKPOINT),
        },
    }
    write_json(OUT / "audit_summary.json", audit)
    (OUT / "README.md").write_text(
        "# RR100 eye-trace conditioning and 32–60 Hz audit\n\n"
        "This checkpoint identifies the exact visual trace contract used by the 120-Hz frozen model, "
        "recomputes one-sided temporal power with the 60-Hz Nyquist bin retained, and compares exact "
        "recorded model frames with RF-crop- and sign-validated crossed Figure-4 reconstructions. See `audit_summary.json` for the "
        "machine-readable conclusions and the two PDF figures for map-first inspection.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(audit), indent=2), flush=True)


if __name__ == "__main__":
    main()

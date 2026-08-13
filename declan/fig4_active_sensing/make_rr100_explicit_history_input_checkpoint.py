#!/usr/bin/env python3
"""Checkpoint 22a: input-level genuine versus synthetic prehistory audit."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
)
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import (
    SOURCE_RUN,
    corrected_patch,
    file_identity,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _load_twin_common
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_explicit_history_input_checkpoint_22a_v2"
IMAGE_INDEX = 3
TRACE_INDICES = (1, 13, 30, 31)
N_HISTORY = 32
N_SCORE = 40
FRAME_RATE = 120.0


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def explicit_segments(source: pd.Series, dset) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    start, stop = int(source["global_start"]), int(source["global_stop"])
    center = (start + stop) // 2
    score_start = center - N_SCORE
    if score_start % 2:
        score_start -= 1
    score_indices = score_start + 2 * np.arange(N_SCORE, dtype=int)
    history_indices = score_start - 2 * N_HISTORY + 2 * np.arange(N_HISTORY, dtype=int)
    session_origin = start - int(source["local_start"])
    epoch_start = session_origin + int(source["epoch_start_local"])
    epoch_stop = session_origin + int(source["epoch_stop_local"])
    if int(history_indices[0]) < epoch_start or int(score_indices[-1]) >= epoch_stop:
        raise ValueError(f"Insufficient same-epoch history for source_row={source['source_row']}")
    xy = corrected_crop_xy_deg(dset)
    score = xy[score_indices]
    history = xy[history_indices]
    center_xy = score.mean(axis=0, keepdims=True)
    return (
        (history - center_xy).astype(np.float32),
        (score - center_xy).astype(np.float32),
        {
            "epoch_start": epoch_start,
            "epoch_stop": epoch_stop,
            "history_start": int(history_indices[0]),
            "history_stop_exclusive": int(history_indices[-1] + 2),
            "score_start": int(score_indices[0]),
            "score_stop_exclusive": int(score_indices[-1] + 2),
        },
    )


def render_aligned(common, patch: np.ndarray, trace72: np.ndarray, ppd: float) -> torch.Tensor:
    image = _standardize_uint_like(patch)
    full_stack = np.broadcast_to(
        image[None], (trace72.shape[0] + int(common.N_LAGS) + 1, *image.shape)
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(-np.asarray(trace72, dtype=np.float32)))
    stim = common.make_counterfactual_stim(
        full_stack, eye, ppd=float(ppd), scale_factor=1.0,
        n_lags=int(common.N_LAGS), out_size=common.OUT_SIZE,
    )
    if int(stim.shape[0]) != int(trace72.shape[0]) + 1:
        raise ValueError(f"Expected T+1 native frames, got {stim.shape[0]}")
    return stim[1 : 1 + trace72.shape[0]].detach().cpu()


def path_arcmin(trace: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum() * 60.0)


def temporal_power(movie: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(movie, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    arr *= np.hanning(arr.shape[0])[:, None, None]
    power = np.mean(np.abs(np.fft.rfft(arr, axis=0)) ** 2, axis=(1, 2))
    freq = np.fft.rfftfreq(arr.shape[0], d=1.0 / FRAME_RATE)
    return freq, power / max(float(power.sum()), 1e-12)


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE_RUN / "summary.json").read_text())
    source_rows = load_source_rows(Path(summary["source_csv"]))
    images = pd.read_csv(SOURCE_RUN / "image_feature_table.csv")
    traces = pd.read_csv(SOURCE_RUN / "trace_feature_table.csv")
    dset_cache, canvas_cache = {}, {}

    image_row = images.loc[images.image_index.eq(IMAGE_INDEX)].iloc[0]
    image_source = source_row_by_id(source_rows, int(image_row.source_row))
    image_dset = load_dset(str(image_source.session), dset_cache)
    patch, patch_meta, _ = corrected_patch(image_source, image_dset, canvas_cache)
    common = _load_twin_common()

    records, metric_rows, tensor_summaries, power_rows = [], [], [], []
    for trace_index in TRACE_INDICES:
        row = traces.loc[traces.trace_index.eq(trace_index)].iloc[0]
        source = source_row_by_id(source_rows, int(row.trace_source_row))
        dset = load_dset(str(source.session), dset_cache)
        history, score, bounds = explicit_segments(source, dset)
        genuine72 = np.concatenate([history, score], axis=0)
        synthetic72 = np.concatenate([score[:N_HISTORY], score], axis=0)
        genuine = render_aligned(common, patch, genuine72, float(patch_meta["patch_ppd"]))
        synthetic = render_aligned(common, patch, synthetic72, float(patch_meta["patch_ppd"]))
        genuine_score = genuine[N_HISTORY:]
        synthetic_score = synthetic[N_HISTORY:]
        difference = torch.abs(genuine_score - synthetic_score)
        lag_zero_genuine = genuine_score[:, 0, 0].numpy()
        lag_zero_synthetic = synthetic_score[:, 0, 0].numpy()
        lag_zero_max = float(np.max(np.abs(lag_zero_genuine - lag_zero_synthetic)))
        if lag_zero_max > 1e-5:
            raise AssertionError(f"Scored lag-zero movies differ: {lag_zero_max}")
        mean_abs = difference.mean(dim=(1, 2, 3, 4)).numpy()
        max_abs = difference.amax(dim=(1, 2, 3, 4)).numpy()
        lag_changed = (difference.mean(dim=(1, 3, 4)) > 1e-5).sum(dim=1).numpy()
        f_g, p_g = temporal_power(genuine[:, 0, 0].numpy())
        f_s, p_s = temporal_power(synthetic[:, 0, 0].numpy())
        high = (f_g >= 32.0) & (f_g <= 60.0)
        genuine_high = float(p_g[high].sum())
        synthetic_high = float(p_s[high].sum())
        for frequency, genuine_power, synthetic_power in zip(f_g, p_g, p_s, strict=True):
            power_rows.append({
                "trace_index": int(trace_index), "temporal_frequency_hz": float(frequency),
                "genuine_normalized_power": float(genuine_power),
                "synthetic_normalized_power": float(synthetic_power),
            })
        records.append({
            "trace_index": int(trace_index), "source_row": int(row.trace_source_row),
            "session": str(source.session), "history": history, "score": score,
            "genuine_early_delta": difference[0, 0].mean(dim=0).numpy(),
            "genuine_late_delta": difference[-1, 0].mean(dim=0).numpy(),
            "mean_abs": mean_abs, "max_abs": max_abs, "lag_changed": lag_changed,
            "frequency": f_g, "power_genuine": p_g, "power_synthetic": p_s,
            "score_path_arcmin": path_arcmin(score), "bounds": bounds,
        })
        for frame in range(N_SCORE):
            metric_rows.append({
                "trace_index": int(trace_index), "scored_frame": frame,
                "mean_abs_model_input_difference": float(mean_abs[frame]),
                "max_abs_model_input_difference": float(max_abs[frame]),
                "n_changed_lag_channels": int(lag_changed[frame]),
                "lag_zero_max_abs_difference": lag_zero_max,
            })
        tensor_summaries.append({
            "trace_index": int(trace_index), "source_row": int(row.trace_source_row),
            "session": str(source.session), "score_path_arcmin": path_arcmin(score),
            "history_path_arcmin": path_arcmin(history), **bounds,
            "score_lag_zero_max_abs_difference": lag_zero_max,
            "first_score_mean_abs_tensor_difference": float(mean_abs[0]),
            "last_score_mean_abs_tensor_difference": float(mean_abs[-1]),
            "first_zero_difference_scored_frame": int(np.flatnonzero(mean_abs < 1e-7)[0]) if np.any(mean_abs < 1e-7) else -1,
            "genuine_32_60hz_power_fraction": genuine_high,
            "synthetic_32_60hz_power_fraction": synthetic_high,
            "genuine_minus_synthetic_32_60hz_power_fraction": genuine_high - synthetic_high,
            "genuine_trace_sha256": sha256_array(genuine72),
            "synthetic_trace_sha256": sha256_array(synthetic72),
        })

    fig, axes = plt.subplots(len(records), 5, figsize=(15.2, 2.75 * len(records)), constrained_layout=True)
    for r, rec in enumerate(records):
        history, score = rec["history"] * 60.0, rec["score"] * 60.0
        axes[r, 0].plot(history[:, 0], history[:, 1], color="0.55", lw=1.2, label="recorded history")
        axes[r, 0].plot(score[:, 0], score[:, 1], color="#0072B2", lw=1.5, label="scored segment")
        axes[r, 0].scatter(score[0, 0], score[0, 1], color="#009E73", s=22, zorder=3)
        axes[r, 0].set_aspect("equal", adjustable="datalim")
        axes[r, 0].set_title(f"trace {rec['trace_index']} · {rec['score_path_arcmin']:.1f}′\ngenuine recorded lead-in")
        if r == 0: axes[r, 0].legend(frameon=False, fontsize=7)
        axes[r, 1].plot(score[:, 0], score[:, 1], color="#0072B2", lw=1.5)
        axes[r, 1].plot(
            score[:N_HISTORY, 0], score[:N_HISTORY, 1], color="#D55E00", lw=1.3,
            ls="--", alpha=0.9, label="reused as lead-in",
        )
        axes[r, 1].set_aspect("equal", adjustable="datalim")
        axes[r, 1].set_title("synthetic repeated lead-in\nsame scored segment")
        early = rec["genuine_early_delta"]
        vmax = max(float(np.percentile(early, 99)), 1e-6)
        im = axes[r, 2].imshow(early, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        axes[r, 2].set_title("score frame 0\nmean |input Δ| over lags")
        fig.colorbar(im, ax=axes[r, 2], shrink=0.68, pad=0.01)
        axes[r, 3].plot(np.arange(N_SCORE), rec["mean_abs"], color="#6A3D9A", lw=1.8, label="mean |Δ|")
        axes[r, 3].plot(np.arange(N_SCORE), rec["lag_changed"] / N_HISTORY, color="#CC79A7", lw=1.3, label="changed lag fraction")
        axes[r, 3].axvline(N_HISTORY - 1, color="0.4", ls=":", lw=1)
        axes[r, 3].set(title="history effect across scored frames", xlabel="scored frame")
        if r == 0: axes[r, 3].legend(frameon=False, fontsize=7)
        axes[r, 4].plot(rec["frequency"], rec["power_genuine"], color="#009E73", lw=1.5, label="genuine 72-frame")
        axes[r, 4].plot(rec["frequency"], rec["power_synthetic"], color="#D55E00", lw=1.2, label="synthetic 72-frame")
        axes[r, 4].set(xlim=(0, 60), title="full-sequence temporal power", xlabel="temporal frequency (Hz)")
        if r == 0: axes[r, 4].legend(frameon=False, fontsize=7)
        for c in (2,): axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == len(records) - 1:
            axes[r, 0].set_xlabel("horizontal (arcmin)"); axes[r, 1].set_xlabel("horizontal (arcmin)")
        axes[r, 0].set_ylabel("vertical (arcmin)")
    fig.suptitle(
        "Checkpoint 22a — genuine recorded versus synthetic repeated model prehistory\n"
        "one fixed strong-contour image · 32 lead-in + identical 40-frame scored segment",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(OUT / "checkpoint_22a_explicit_history_inputs.png", dpi=220)
    fig.savefig(OUT / "checkpoint_22a_explicit_history_inputs.pdf")
    plt.close(fig)

    pd.DataFrame(metric_rows).to_csv(OUT / "scored_frame_input_difference_metrics.csv", index=False)
    pd.DataFrame(tensor_summaries).to_csv(OUT / "selected_trace_history_contract.csv", index=False)
    pd.DataFrame(power_rows).to_csv(OUT / "full_sequence_temporal_power_curves.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_checkpoint_complete_stop_before_response_maps",
        "scope": "one fixed strong-contour image x four representative traces",
        "comparison": "32 genuine recorded lead-in frames versus first 32 scored positions reused as synthetic lead-in; final 40 scored positions identical",
        "sampling": "dpi_pix; global-even raw indices at 120 Hz; same epoch; retinal sign applied in renderer",
        "model_input": "T+1 helper output drops first frame; final 40 of aligned 72-frame tensor retained",
        "image_index": IMAGE_INDEX, "trace_indices": list(TRACE_INDICES),
        "source_summary": file_identity(SOURCE_RUN / "summary.json"),
        "next_checkpoint": "score matched model tensors and inspect multiple RR100 response/difference maps before SSI summary",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(tensor_summaries).to_string(index=False))


if __name__ == "__main__":
    main()

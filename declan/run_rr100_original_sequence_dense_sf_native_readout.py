#!/usr/bin/env python3
"""Dense SF substitution of the recorded grating sequence through native RR100 readouts.

For each session contributing an RR100 movie-medoid unit, this script:

1. loads the exact model preprocessing and model-valid grating indices;
2. evaluates the unmodified sequence with that session's native fitted readout;
3. regenerates the retained 120 Hz retinal frames from the original 240 Hz
   ForageGrating sequence after replacing every positive carrier SF by one
   target SF (recorded SF=0 slots remain blank);
4. evaluates the same indices, behavior, 33-frame histories, orientations,
   gaze-dependent ROIs, and probe/face overlays for every target SF; and
5. saves unit curves plus auditable example curves and trial-resolved traces.

This is a map-first unit checkpoint.  It intentionally does not compute a
population-average conclusion.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.general import get_clock_functions
from DataYatesV1.utils.io import get_session

from models.data import prepare_data


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.utils import get_model_and_dataset_configs  # noqa: E402


SPEC_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_original_sequence_dense_sf_native_readout_v1"
)
TARGET_SFS = np.power(2.0, np.arange(0.0, 4.0 + 0.5, 0.5)).astype(np.float64)
DT = 1.0 / 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--population-version", type=str, default=RR100_VERSION)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-eval-frames", type=int, default=0)
    parser.add_argument("--audit-frames", type=int, default=64)
    parser.add_argument("--trace-frames", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_slug(value: str) -> str:
    return "_".join(
        part
        for part in "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).split("_")
        if part
    )


def load_rr100_rows(version: str) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    slug = safe_slug(version)
    json_path = SPEC_DIR / f"population_spec_{slug}.json"
    npz_path = SPEC_DIR / f"population_spec_{slug}.npz"
    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Missing RR100 population spec for {version!r}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    rows = sorted(meta["representatives"], key=lambda row: int(row["rep_idx"]))
    with np.load(npz_path) as payload:
        membership = np.asarray(payload["membership"], dtype=np.float32)
    if membership.shape != (len(rows), int(meta["n_input_channels"])):
        raise ValueError(f"Unexpected RR100 membership shape {membership.shape}")
    for row in rows:
        selected = np.flatnonzero(membership[int(row["rep_idx"])] != 0)
        if selected.size != 1 or int(selected[0]) != int(row["selected_channel"]):
            raise ValueError(f"RR100 row {row['rep_idx']} is not its declared one-hot medoid")
    return rows, meta, json_path, npz_path


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_eval_indices(train_data, val_data, *, max_eval_frames: int) -> torch.Tensor:
    train_local = train_data.get_dataset_inds("gratings")[:, 1]
    val_local = val_data.get_dataset_inds("gratings")[:, 1]
    local = torch.unique(torch.cat([train_local, val_local]), sorted=True)
    if max_eval_frames > 0 and len(local) > int(max_eval_frames):
        positions = torch.linspace(0, len(local) - 1, int(max_eval_frames)).round().long()
        local = local[positions]
    return local


def make_eval_dataset(val_data, local_indices: torch.Tensor):
    eval_data = val_data.shallow_copy()
    eval_data.inds = torch.stack([torch.zeros_like(local_indices), local_indices], dim=1)
    return eval_data


def predict_native(
    model,
    eval_data,
    dataset_idx: int,
    source_unit_indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.model.eval()
    with torch.inference_mode():
        for start in range(0, len(eval_data), int(batch_size)):
            batch = eval_data[start : start + int(batch_size)]
            stimulus = batch["stim"].to(device, non_blocking=True)
            behavior = batch.get("behavior")
            if behavior is not None:
                behavior = behavior.to(device, non_blocking=True)
            output = model.model(stimulus, int(dataset_idx), behavior)
            chunks.append(
                output[:, source_unit_indices].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            del batch, stimulus, behavior, output
    return np.concatenate(chunks, axis=0)


def map_bins_to_flips(trial: GratingsTrial, times: np.ndarray, ptb2ephys) -> np.ndarray:
    flip_times = np.asarray(ptb2ephys(trial.flip_times), dtype=np.float64)
    frame_inds = np.searchsorted(flip_times, times) - 1
    return np.maximum(frame_inds, 0).astype(np.int64)


def retained_raw_arrays(raw_dset, n_model_frames: int) -> dict[str, np.ndarray]:
    raw_indices = np.arange(int(n_model_frames), dtype=np.int64) * 2
    if int(raw_indices[-1]) >= len(raw_dset):
        raise ValueError("Prepared model sequence is longer than retained raw 240 Hz sequence")
    return {
        "raw_indices": raw_indices,
        "times": raw_dset["t_bins"][raw_indices].numpy(),
        "rois": raw_dset["roi"][raw_indices].numpy(),
        "trials": raw_dset["trial_inds"][raw_indices].numpy().astype(np.int64),
        "sf": raw_dset["sf"][raw_indices].numpy().astype(np.float64),
        "ori": raw_dset["ori"][raw_indices].numpy().astype(np.float64),
    }


def generate_target_movies(
    sess,
    raw_dset,
    n_model_frames: int,
    target_sfs: np.ndarray,
    *,
    audit_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    retained = retained_raw_arrays(raw_dset, n_model_frames)
    target_movies = np.empty(
        (len(target_sfs), n_model_frames, 51, 51),
        dtype=np.uint8,
    )
    ptb2ephys, _ = get_clock_functions(sess.exp)
    unique_trials = np.unique(retained["trials"])
    audit_candidates = np.linspace(0, n_model_frames - 1, int(audit_frames)).round().astype(int)
    audit_candidate_set = set(int(v) for v in audit_candidates)
    audit_rows: list[dict[str, Any]] = []

    for trial_number, trial_index in enumerate(unique_trials, start=1):
        local = np.flatnonzero(retained["trials"] == int(trial_index))
        trial = GratingsTrial(sess.exp["D"][int(trial_index)], sess.exp["S"])
        flip_inds = map_bins_to_flips(trial, retained["times"][local], ptb2ephys)
        rois = retained["rois"][local]
        if not np.allclose(trial.spatial_frequencies[flip_inds], retained["sf"][local], atol=1e-10, rtol=0):
            raise AssertionError(f"{sess.name} trial {trial_index}: SF label reconstruction failed")
        if not np.allclose(trial.orientations[flip_inds], retained["ori"][local], atol=1e-10, rtol=0):
            raise AssertionError(f"{sess.name} trial {trial_index}: orientation label reconstruction failed")

        audit_local_positions = [j for j, global_pos in enumerate(local) if int(global_pos) in audit_candidate_set]
        if audit_local_positions:
            audit_local_positions_arr = np.asarray(audit_local_positions, dtype=np.int64)
            audit_global = local[audit_local_positions_arr]
            regenerated = trial.get_frames(
                flip_inds[audit_local_positions_arr],
                roi=rois[audit_local_positions_arr],
            )
            stored = raw_dset["stim"][retained["raw_indices"][audit_global]].numpy()
            diff = np.abs(regenerated.astype(np.int16) - stored.astype(np.int16))
            for j, global_pos in enumerate(audit_global):
                audit_rows.append(
                    {
                        "session": str(sess.name),
                        "model_120hz_index": int(global_pos),
                        "raw_240hz_index": int(retained["raw_indices"][global_pos]),
                        "trial_index": int(trial_index),
                        "flip_index": int(flip_inds[audit_local_positions_arr[j]]),
                        "source_sf_cpd": float(retained["sf"][global_pos]),
                        "orientation_deg": float(retained["ori"][global_pos]),
                        "max_abs_pixel_difference": int(np.max(diff[j])),
                        "n_different_pixels": int(np.count_nonzero(diff[j])),
                        "exact_match": bool(not np.any(diff[j])),
                    }
                )

        original_sf = trial.spatial_frequencies.copy()
        try:
            for target_index, target_sf in enumerate(target_sfs):
                trial.spatial_frequencies = np.where(original_sf > 0, float(target_sf), 0.0)
                target_movies[target_index, local] = trial.get_frames(flip_inds, roi=rois)
        finally:
            trial.spatial_frequencies = original_sf
        print(
            f"  rendered trial {trial_number}/{len(unique_trials)} ({int(trial_index)})",
            flush=True,
        )
        del trial

    if not audit_rows or not all(bool(row["exact_match"]) for row in audit_rows):
        maximum = max((int(row["max_abs_pixel_difference"]) for row in audit_rows), default=-1)
        raise AssertionError(f"{sess.name}: original-renderer audit failed; max difference={maximum}")
    return target_movies, audit_rows


def response_rows_from_cache(cache_path: Path) -> list[dict[str, Any]]:
    with np.load(cache_path) as payload:
        target_sfs = np.asarray(payload["target_sf_cpd"], dtype=np.float64)
        rr100_indices = np.asarray(payload["rr100_indices"], dtype=np.int64)
        response = np.asarray(payload["target_response_counts"], dtype=np.float32)
        source_sf = np.asarray(payload["source_sf_cpd"], dtype=np.float64)
        source_ori = np.asarray(payload["source_orientation_deg"], dtype=np.float64)
        session = str(np.asarray(payload["session"]).reshape(-1)[0])
    rows: list[dict[str, Any]] = []
    state_masks = {
        "all_model_valid_bins": np.ones(source_sf.shape, dtype=bool),
        "carrier_current_bins": source_sf > 0,
        "blank_current_bins": source_sf == 0,
    }
    for unit_col, rr100_index in enumerate(rr100_indices):
        for target_index, target_sf in enumerate(target_sfs):
            values_hz = response[target_index, :, unit_col].astype(np.float64) / DT
            for state, mask in state_masks.items():
                selected = values_hz[mask & np.isfinite(values_hz)]
                rows.append(
                    {
                        "rr100_index": int(rr100_index),
                        "session": session,
                        "target_sf_cpd": float(target_sf),
                        "response_state": state,
                        "orientation_deg": float("nan"),
                        "n_samples": int(selected.size),
                        "mean_rate_hz": float(np.mean(selected)) if selected.size else float("nan"),
                        "median_rate_hz": float(np.median(selected)) if selected.size else float("nan"),
                        "std_rate_hz": float(np.std(selected)) if selected.size else float("nan"),
                    }
                )
            for orientation in np.unique(source_ori[source_sf > 0]):
                mask = (source_sf > 0) & np.isclose(source_ori, orientation)
                selected = values_hz[mask & np.isfinite(values_hz)]
                rows.append(
                    {
                        "rr100_index": int(rr100_index),
                        "session": session,
                        "target_sf_cpd": float(target_sf),
                        "response_state": "carrier_current_orientation_conditioned",
                        "orientation_deg": float(orientation),
                        "n_samples": int(selected.size),
                        "mean_rate_hz": float(np.mean(selected)) if selected.size else float("nan"),
                        "median_rate_hz": float(np.median(selected)) if selected.size else float("nan"),
                        "std_rate_hz": float(np.std(selected)) if selected.size else float("nan"),
                    }
                )
    return rows


def unit_metrics(curves: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    primary = curves[curves["response_state"] == "all_model_valid_bins"].copy()
    rows: list[dict[str, Any]] = []
    for rr100_index, sub in primary.groupby("rr100_index", sort=True):
        sub = sub.sort_values("target_sf_cpd")
        sf = sub["target_sf_cpd"].to_numpy(dtype=float)
        rate = sub["mean_rate_hz"].to_numpy(dtype=float)
        robust = sf <= 11.3137086 + 1e-6
        robust_sf = sf[robust]
        robust_rate = rate[robust]
        peak_all = int(np.nanargmax(rate))
        peak_robust = int(np.nanargmax(robust_rate))
        modulation = float(np.nanmax(robust_rate) - np.nanmin(robust_rate))
        mean_rate = float(np.nanmean(robust_rate))
        endpoint_mean = float(np.nanmean([robust_rate[0], robust_rate[-1]]))
        interior_peak = float(np.nanmax(robust_rate[1:-1])) if len(robust_rate) > 2 else float("nan")
        rows.append(
            {
                "rr100_index": int(rr100_index),
                "preferred_sf_cpd_all": float(sf[peak_all]),
                "preferred_sf_cpd_resolution_robust": float(robust_sf[peak_robust]),
                "robust_modulation_hz": modulation,
                "robust_modulation_fraction": modulation / max(abs(mean_rate), 1e-8),
                "robust_mean_rate_hz": mean_rate,
                "interior_over_endpoint_hz": interior_peak - endpoint_mean,
                "edge_16_minus_11p3_hz": float(rate[-1] - robust_rate[-1]),
                "absolute_edge_16_step_hz": float(abs(rate[-1] - robust_rate[-1])),
            }
        )
    return pd.DataFrame(rows).merge(mapping, on="rr100_index", how="left", validate="one_to_one")


def choose_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    """Predefined positive, preference, dissociation, and control roles."""
    role_specs: list[tuple[str, pd.DataFrame, str, bool, str]] = [
        (
            "largest_resolution_robust_modulation",
            metrics,
            "robust_modulation_fraction",
            False,
            "largest range/mean over 1-11.3 cpd",
        ),
        (
            "low_sf_preference",
            metrics[metrics["preferred_sf_cpd_resolution_robust"] <= 2.0],
            "robust_modulation_fraction",
            False,
            "strongest modulation among robust peaks <=2 cpd",
        ),
        (
            "high_sf_preference",
            metrics[metrics["preferred_sf_cpd_resolution_robust"] >= 8.0],
            "robust_modulation_fraction",
            False,
            "strongest modulation among robust peaks >=8 cpd",
        ),
        (
            "interior_bandpass_candidate",
            metrics[
                metrics["preferred_sf_cpd_resolution_robust"].isin([2.8284271247461903, 4.0, 5.656854249492381])
            ],
            "interior_over_endpoint_hz",
            False,
            "largest interior response above mean endpoint response",
        ),
        (
            "weak_modulation_control",
            metrics,
            "robust_modulation_fraction",
            True,
            "smallest range/mean over 1-11.3 cpd",
        ),
        (
            "sixteen_cpd_edge_sensitivity",
            metrics,
            "absolute_edge_16_step_hz",
            False,
            "largest absolute 16 vs 11.3 cpd step; sampling-edge diagnostic",
        ),
    ]
    chosen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for role, candidates, criterion, ascending, reason in role_specs:
        candidates = candidates[np.isfinite(candidates[criterion])].sort_values(criterion, ascending=ascending)
        available = candidates[~candidates["rr100_index"].astype(int).isin(chosen)]
        if available.empty:
            available = candidates
        if available.empty:
            continue
        row = available.iloc[0].to_dict()
        chosen.add(int(row["rr100_index"]))
        row.update(
            {
                "selection_role": role,
                "criterion_name": criterion,
                "criterion_value": float(row[criterion]),
                "selection_reason": reason,
                "selection_method": "predefined_algorithmic_role",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def longest_consecutive_runs(local_indices: np.ndarray) -> list[np.ndarray]:
    if local_indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(local_indices) != 1) + 1
    return [chunk for chunk in np.split(np.arange(len(local_indices)), breaks) if len(chunk)]


def choose_trace_window(
    local_indices: np.ndarray,
    trial_inds: np.ndarray,
    source_sf: np.ndarray,
    target_response: np.ndarray,
    *,
    trace_frames: int,
) -> np.ndarray:
    candidates: list[tuple[float, np.ndarray]] = []
    for run in longest_consecutive_runs(local_indices):
        trial_breaks = np.flatnonzero(np.diff(trial_inds[run]) != 0) + 1
        for trial_run in np.split(run, trial_breaks):
            if len(trial_run) < trace_frames:
                continue
            for start in range(0, len(trial_run) - trace_frames + 1, max(trace_frames // 4, 1)):
                window = trial_run[start : start + trace_frames]
                has_blank = bool(np.any(source_sf[window] == 0))
                has_carrier = bool(np.any(source_sf[window] > 0))
                if not (has_blank and has_carrier):
                    continue
                contrast = target_response[-1, window] - target_response[0, window]
                score = float(np.nanstd(contrast))
                candidates.append((score, window))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return np.arange(min(trace_frames, len(local_indices)), dtype=np.int64)


def plot_selected_examples(
    curves: pd.DataFrame,
    selected: pd.DataFrame,
    cache_by_session: dict[str, Path],
    output: Path,
    *,
    trace_frames: int,
) -> None:
    if selected.empty:
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.2 * len(selected)), squeeze=False)
    colors = {"all_model_valid_bins": "#111111", "carrier_current_bins": "#2166ac", "blank_current_bins": "#b2182b"}
    labels = {"all_model_valid_bins": "whole sequence", "carrier_current_bins": "carrier current", "blank_current_bins": "blank current"}
    cache_payloads: dict[str, dict[str, np.ndarray]] = {}
    for row_idx, selected_row in selected.reset_index(drop=True).iterrows():
        unit = int(selected_row["rr100_index"])
        session = str(selected_row["session"])
        unit_curves = curves[(curves["rr100_index"] == unit) & curves["response_state"].isin(colors)]
        for state, sub in unit_curves.groupby("response_state"):
            sub = sub.sort_values("target_sf_cpd")
            axes[row_idx, 0].plot(
                sub["target_sf_cpd"],
                sub["mean_rate_hz"],
                "o-",
                color=colors[state],
                lw=1.6,
                ms=4,
                label=labels[state],
            )
        axes[row_idx, 0].set_xscale("log", base=2)
        axes[row_idx, 0].set_xticks(TARGET_SFS)
        axes[row_idx, 0].set_xticklabels([f"{v:.3g}" for v in TARGET_SFS], rotation=45, ha="right")
        axes[row_idx, 0].axvspan(13.45, 19.0, color="#f4a261", alpha=0.18, label="<3 px/cycle")
        axes[row_idx, 0].set_ylabel("native fitted rate (Hz)")
        axes[row_idx, 0].grid(alpha=0.2)
        if row_idx == 0:
            axes[row_idx, 0].legend(fontsize=7, frameon=False, ncol=2)

        orientation = curves[
            (curves["rr100_index"] == unit)
            & (curves["response_state"] == "carrier_current_orientation_conditioned")
        ].copy()
        pivot = orientation.pivot(index="target_sf_cpd", columns="orientation_deg", values="mean_rate_hz")
        image = axes[row_idx, 1].imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        axes[row_idx, 1].set_yticks(np.arange(len(pivot.index)))
        axes[row_idx, 1].set_yticklabels([f"{v:.3g}" for v in pivot.index], fontsize=7)
        axes[row_idx, 1].set_xticks(np.arange(len(pivot.columns)))
        axes[row_idx, 1].set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right", fontsize=7)
        axes[row_idx, 1].set_ylabel("substituted SF (cpd)")
        axes[row_idx, 1].set_xlabel("recorded orientation (deg)")
        fig.colorbar(image, ax=axes[row_idx, 1], fraction=0.045, pad=0.02, label="Hz")

        if session not in cache_payloads:
            with np.load(cache_by_session[session]) as payload:
                cache_payloads[session] = {key: np.asarray(payload[key]) for key in payload.files}
        payload = cache_payloads[session]
        unit_col = int(np.flatnonzero(payload["rr100_indices"].astype(int) == unit)[0])
        local_indices = payload["local_indices"].astype(int)
        trial_inds = payload["trial_indices"].astype(int)
        source_sf = payload["source_sf_cpd"].astype(float)
        target_response = payload["target_response_counts"][:, :, unit_col].astype(float) / DT
        original_response = payload["original_response_counts"][:, unit_col].astype(float) / DT
        window = choose_trace_window(
            local_indices,
            trial_inds,
            source_sf,
            target_response,
            trace_frames=int(trace_frames),
        )
        time_s = np.arange(len(window)) / 120.0
        trace_ax = axes[row_idx, 2]
        trace_ax.plot(time_s, original_response[window], color="#777777", lw=1.1, label="original mixed SF")
        for target_sf, color in ((1.0, "#2166ac"), (4.0, "#4daf4a"), (16.0, "#b2182b")):
            target_idx = int(np.argmin(np.abs(payload["target_sf_cpd"].astype(float) - target_sf)))
            trace_ax.plot(time_s, target_response[target_idx, window], color=color, lw=1.1, label=f"{target_sf:g} cpd")
        source_ax = trace_ax.twinx()
        source_ax.step(time_s, source_sf[window], where="post", color="#111111", alpha=0.22, lw=0.9)
        source_ax.set_ylabel("original carrier SF", color="#555555", fontsize=7)
        source_ax.set_ylim(-0.5, 17.0)
        trace_ax.set_xlabel("matched recorded sequence time (s)")
        trace_ax.set_ylabel("native fitted rate (Hz)")
        trace_ax.grid(alpha=0.18)
        if row_idx == 0:
            trace_ax.legend(fontsize=7, frameon=False, ncol=2)

        axes[row_idx, 0].text(
            -0.42,
            0.5,
            f"{selected_row['selection_role']}\nRR100 {unit:03d}\n{session}\n"
            f"native unit {int(selected_row['source_unit_index'])}",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        "RR100 native-readout checkpoint: dense SF substitution of the recorded grating sequence",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.08, 0.02, 1.0, 0.98])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    cache_dir = out_dir / "session_response_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_sfs = TARGET_SFS[: int(args.max_targets)] if args.max_targets > 0 else TARGET_SFS.copy()

    reps, population_meta, spec_json, spec_npz = load_rr100_rows(str(args.population_version))
    sessions = list(dict.fromkeys(str(row["selected_session"]) for row in reps))
    if args.max_sessions > 0:
        sessions = sessions[: int(args.max_sessions)]
        allowed = set(sessions)
        reps = [row for row in reps if str(row["selected_session"]) in allowed]
    mapping = pd.DataFrame(
        [
            {
                "rr100_index": int(row["rep_idx"]),
                "canonical_channel": int(row["selected_channel"]),
                "session": str(row["selected_session"]),
                "source_unit_index": int(row["selected_source_unit_index"]),
                "ccnorm": float(row["selected_ccnorm"]),
                "group_kind": str(row["kind"]),
                "group_size": int(row["n_members"]),
                "selection_role": "fixed RR100 movie medoid before dense-SF substitution",
            }
            for row in reps
        ]
    ).sort_values("rr100_index")
    mapping.to_csv(out_dir / "rr100_unit_mapping.csv", index=False)

    print(f"Loading standard fitted model on {args.device}", flush=True)
    model, dataset_configs = get_model_and_dataset_configs(mode="standard")
    model = model.to(str(args.device))
    model.model.eval()
    configs_by_session = {str(config["session"]): config for config in dataset_configs}
    cache_by_session: dict[str, Path] = {}
    audit_csv_path = out_dir / "original_renderer_pixel_audit.csv"
    if audit_csv_path.exists() and not args.force:
        # Preserve completed audit rows when a response-cache run is resumed.
        # Session response caches are written only after their renderer audit
        # passes, but the aggregate CSV is written at the end of the run.
        audit_rows = pd.read_csv(audit_csv_path).to_dict(orient="records")
    else:
        audit_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for session_number, session in enumerate(sessions, start=1):
        cache_path = cache_dir / f"{session}.npz"
        cache_by_session[session] = cache_path
        if cache_path.exists() and not args.force:
            print(f"[{session_number}/{len(sessions)}] using cached {session}", flush=True)
            with np.load(cache_path) as payload:
                session_rows.append(
                    {
                        "session": session,
                        "n_eval_frames": int(len(payload["local_indices"])),
                        "n_rr100_units": int(len(payload["rr100_indices"])),
                        "n_target_sfs": int(len(payload["target_sf_cpd"])),
                        "cache_path": str(cache_path),
                    }
                )
            continue

        print(f"[{session_number}/{len(sessions)}] preparing {session}", flush=True)
        if session not in configs_by_session:
            raise KeyError(f"Missing dataset config for {session}")
        dataset_idx = list(model.names).index(session)
        config = copy.deepcopy(configs_by_session[session])
        config["types"] = ["gratings"]
        config["cids"] = None
        config["keys_lags"] = {
            "stim": list(range(33)),
            "behavior": 0,
        }
        train_data, val_data, _ = prepare_data(config, strict=True)
        dset = val_data.dsets[0]
        local_indices = make_eval_indices(
            train_data,
            val_data,
            max_eval_frames=int(args.max_eval_frames),
        )
        eval_data = make_eval_dataset(val_data, local_indices)
        session_reps = mapping[mapping["session"] == session].sort_values("rr100_index")
        rr100_indices = session_reps["rr100_index"].to_numpy(dtype=np.int64)
        source_unit_indices = session_reps["source_unit_index"].to_numpy(dtype=np.int64)
        n_native_units = int(model.model.readouts[dataset_idx].features.weight.shape[0])
        if int(np.max(source_unit_indices)) >= n_native_units:
            raise IndexError(
                f"{session}: RR100 source index {int(np.max(source_unit_indices))} exceeds native readout {n_native_units}"
            )

        print(
            f"[{session_number}/{len(sessions)}] original sequence: {len(local_indices)} frames, "
            f"{len(rr100_indices)} RR100 units",
            flush=True,
        )
        original_response = predict_native(
            model,
            eval_data,
            dataset_idx,
            source_unit_indices,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )

        sess = get_session(*session.split("_", maxsplit=1))
        raw_dset = sess.get_dataset("gratings", strict=True)
        n_model_frames = int(dset["stim"].shape[0])
        retained = retained_raw_arrays(raw_dset, n_model_frames)
        audit_positions = np.linspace(0, n_model_frames - 1, int(args.audit_frames)).round().astype(int)
        stored_retained = raw_dset["stim"][retained["raw_indices"][audit_positions]].numpy()
        prepared_retained = np.rint(
            dset["stim"][audit_positions, 0].numpy().astype(np.float64) * 255.0 + 127.0
        ).clip(0, 255).astype(np.uint8)
        prepared_diff = np.abs(stored_retained.astype(np.int16) - prepared_retained.astype(np.int16))
        if int(np.max(prepared_diff)) != 0:
            raise AssertionError(f"{session}: prepared 120 Hz pixels differ from raw decimation")

        print(f"[{session_number}/{len(sessions)}] rendering {len(target_sfs)} substituted movies", flush=True)
        target_movies, session_audit = generate_target_movies(
            sess,
            raw_dset,
            n_model_frames,
            target_sfs,
            audit_frames=int(args.audit_frames),
        )
        audit_rows.extend(session_audit)
        del raw_dset, retained, stored_retained, prepared_retained, prepared_diff, sess
        gc.collect()

        target_response = np.empty(
            (len(target_sfs), len(local_indices), len(rr100_indices)),
            dtype=np.float32,
        )
        for target_index, target_sf in enumerate(target_sfs):
            print(
                f"[{session_number}/{len(sessions)}] native readout {target_index + 1}/{len(target_sfs)}: "
                f"{float(target_sf):.5g} cpd",
                flush=True,
            )
            normalized = torch.from_numpy(target_movies[target_index]).to(torch.float32)
            normalized.sub_(127.0).div_(255.0)
            dset.covariates["stim"] = normalized.unsqueeze(1)
            target_response[target_index] = predict_native(
                model,
                eval_data,
                dataset_idx,
                source_unit_indices,
                device=str(args.device),
                batch_size=int(args.batch_size),
            )
            del normalized
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        local_np = local_indices.numpy().astype(np.int64)
        np.savez_compressed(
            cache_path,
            session=np.asarray([session]),
            dataset_idx=np.asarray([dataset_idx], dtype=np.int64),
            rr100_indices=rr100_indices,
            source_unit_indices=source_unit_indices,
            target_sf_cpd=target_sfs,
            local_indices=local_np,
            source_sf_cpd=dset["sf"][local_np].numpy().astype(np.float64),
            source_orientation_deg=dset["ori"][local_np].numpy().astype(np.float64),
            trial_indices=dset["trial_inds"][local_np].numpy().astype(np.float64),
            original_response_counts=original_response,
            target_response_counts=target_response,
            dt_seconds=np.asarray([DT], dtype=np.float64),
            model_input_lags=np.arange(33, dtype=np.int64),
        )
        session_rows.append(
            {
                "session": session,
                "n_eval_frames": int(len(local_indices)),
                "n_rr100_units": int(len(rr100_indices)),
                "n_target_sfs": int(len(target_sfs)),
                "cache_path": str(cache_path),
            }
        )
        print(f"[{session_number}/{len(sessions)}] wrote {cache_path}", flush=True)
        del train_data, val_data, eval_data, dset, target_movies, target_response, original_response
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    session_table = pd.DataFrame(session_rows)
    session_table.to_csv(out_dir / "session_evaluation_summary.csv", index=False)
    if audit_rows:
        audit_frame = pd.DataFrame(audit_rows).drop_duplicates(
            ["session", "model_120hz_index"], keep="last"
        )
        audit_frame.to_csv(audit_csv_path, index=False)
        audit_rows = audit_frame.to_dict(orient="records")

    curve_rows: list[dict[str, Any]] = []
    for session in sessions:
        curve_rows.extend(response_rows_from_cache(cache_by_session[session]))
    curves = pd.DataFrame(curve_rows).sort_values(
        ["rr100_index", "response_state", "orientation_deg", "target_sf_cpd"],
        na_position="first",
    )
    curves.to_csv(out_dir / "rr100_native_dense_sf_curves_long.csv", index=False)
    metrics = unit_metrics(curves, mapping)
    metrics.to_csv(out_dir / "rr100_native_dense_sf_unit_metrics.csv", index=False)
    selected = choose_examples(metrics)
    selected.to_csv(out_dir / "rr100_native_dense_sf_selected_examples.csv", index=False)
    plot_selected_examples(
        curves,
        selected,
        cache_by_session,
        out_dir / "rr100_native_dense_sf_selected_example_curves_and_traces.png",
        trace_frames=int(args.trace_frames),
    )

    status = (
        "smoke"
        if args.max_sessions > 0 or args.max_targets > 0 or args.max_eval_frames > 0
        else "all_rr100_all_model_valid_grating_bins"
    )
    audit_df = pd.DataFrame(audit_rows)
    manifest = {
        "analysis": "rr100_original_sequence_dense_sf_native_readout",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "population_inference_performed": False,
        "checkpoint": "unit curves, orientation maps, and trial-resolved example traces",
        "population_version": str(args.population_version),
        "population_spec_json": file_identity(spec_json),
        "population_spec_npz": file_identity(spec_npz),
        "n_rr100_units": int(len(mapping)),
        "n_sessions": int(len(sessions)),
        "session_unit_counts": dict(Counter(mapping["session"])),
        "target_sf_cpd": target_sfs.tolist(),
        "source_rate_hz": 240,
        "model_rate_hz": 120,
        "model_input_lags": list(range(33)),
        "evaluation_support": "union of training and validation model-valid grating bins",
        "substitution_rule": "target_sf when recorded sf>0; retain sf=0 blank slots",
        "native_readout_rule": (
            "evaluate the standard fitted multi-dataset model with the contributing session's own dataset_idx, "
            "then select selected_source_unit_index for each fixed RR100 movie medoid"
        ),
        "held_fixed": [
            "recorded grating trial and flip order",
            "recorded orientation sequence",
            "recorded sf=0 blank-carrier slots",
            "absolute screen-coordinate phase origin",
            "contrast and background",
            "gaze-dependent retinal ROI",
            "Gabor/face probe identity, position, and alpha blend",
            "behavior covariates",
            "model-valid evaluation indices",
        ],
        "response_definitions": {
            "primary": "mean native fitted rate over the whole matched model-valid sequence",
            "carrier_current": "mean rate where the recorded current carrier SF was positive",
            "blank_current": "mean rate where the recorded current carrier SF was zero",
            "orientation_conditioned": "carrier-current mean grouped by recorded orientation",
            "units": "spikes/s (model bin expectation divided by 1/120 s)",
        },
        "sampling_warning": "16 cpd is below pixel Nyquist but has only 2.34 pixels/cycle; retain as an edge diagnostic",
        "pixel_audit": {
            "n_frames": int(len(audit_df)),
            "all_exact": bool(audit_df["exact_match"].all()) if not audit_df.empty else None,
            "maximum_abs_pixel_difference": int(audit_df["max_abs_pixel_difference"].max())
            if not audit_df.empty
            else None,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "artifacts": {
            "unit_mapping": "rr100_unit_mapping.csv",
            "session_summary": "session_evaluation_summary.csv",
            "pixel_audit": "original_renderer_pixel_audit.csv",
            "session_response_cache": "session_response_cache/*.npz",
            "unit_curves": "rr100_native_dense_sf_curves_long.csv",
            "unit_metrics": "rr100_native_dense_sf_unit_metrics.csv",
            "selected_examples": "rr100_native_dense_sf_selected_examples.csv",
            "selected_figure": "rr100_native_dense_sf_selected_example_curves_and_traces.png",
        },
        "population_meta_summary": {
            "n_input_channels": int(population_meta["n_input_channels"]),
            "n_representatives": int(population_meta["n_representatives"]),
            "pooling_mode": str(population_meta["pooling_mode"]),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote map-first native-readout checkpoint to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconstruct corrected descriptors for the 1,000 legacy RR100 trace identities.

This is a no-model provenance checkpoint.  It keeps each legacy trace's source
session/trial and temporal center fixed, then compares:

* the cached 40-position trace interpreted at 120 Hz;
* the same cached source samples at their actual 240-Hz acquisition rate;
* a 40-frame, global-even 120-Hz raw-eyepos segment;
* the corresponding shifter-corrected dpi_pix visual-crop segment.

For the corrected visual trace it also checks that 32 preceding 120-Hz frames
exist, remain in the same trial, and are DPI-valid.  No neural responses are
computed and no legacy response magnitude is recalibrated by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
)


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_legacy1000_corrected_trace_descriptors_v1"
N_SCORED = 40
N_HISTORY = 32
SOURCE_HZ = 240.0
MODEL_HZ = 120.0
EPS = 1e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=LEGACY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def center(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    return arr - np.nanmean(arr, axis=0, keepdims=True)


def temporal_power(trace: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    arr = center(trace)
    arr = arr * np.hanning(arr.shape[0])[:, None]
    fft = np.fft.rfft(arr, axis=0)
    power = np.sum(np.abs(fft) ** 2, axis=1)
    weights = np.ones(power.shape[0], dtype=np.float64)
    if power.shape[0] > 2:
        weights[1:-1] = 2.0
    return np.fft.rfftfreq(arr.shape[0], d=1.0 / float(sample_rate_hz)), power * weights


def metrics(trace: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    arr = center(trace)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    freq, power = temporal_power(arr, sample_rate_hz)
    positive = freq > 0
    positive_power = float(power[positive].sum())
    high15 = positive & (freq >= 15.0)
    high32 = positive & (freq > 32.0)
    cov = np.cov(arr.T, ddof=0)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    major = vectors[:, 0]
    orientation = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return {
        "n_frames": int(arr.shape[0]),
        "sample_rate_hz": float(sample_rate_hz),
        "duration_s": float((arr.shape[0] - 1) / sample_rate_hz),
        "path_length_arcmin": float(steps.sum() * 60.0),
        "rms_radius_arcmin": float(np.sqrt(np.mean(np.sum(arr**2, axis=1))) * 60.0),
        "max_radius_arcmin": float(np.max(np.linalg.norm(arr, axis=1)) * 60.0),
        "median_step_arcmin": float(np.median(steps) * 60.0),
        "mean_speed_dps": float(np.mean(steps) * sample_rate_hz),
        "median_speed_dps": float(np.median(steps) * sample_rate_hz),
        "p95_speed_dps": float(np.percentile(steps, 95) * sample_rate_hz),
        "position_power_total_positive": positive_power,
        "position_power_fraction_15plus_hz": float(power[high15].sum() / max(positive_power, EPS)),
        "position_power_fraction_32plus_hz": float(power[high32].sum() / max(positive_power, EPS)),
        "position_power_centroid_hz": float(np.sum(freq[positive] * power[positive]) / max(positive_power, EPS)),
        "cov_major_sd_arcmin": float(np.sqrt(max(values[0], 0.0)) * 60.0),
        "cov_minor_sd_arcmin": float(np.sqrt(max(values[1], 0.0)) * 60.0),
        "cov_anisotropy": float((values[0] - values[1]) / max(values.sum(), EPS)),
        "cov_orientation_deg": orientation,
    }


def add_prefixed(out: dict[str, object], prefix: str, values: dict[str, float]) -> None:
    out.update({f"{prefix}{key}": value for key, value in values.items()})


def centered_model_indices(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    center_index = (int(row["snippet_global_start"]) + int(row["snippet_global_stop"])) // 2
    scored_start = center_index - N_SCORED
    if scored_start % 2:
        scored_start -= 1
    scored = np.arange(scored_start, scored_start + 2 * N_SCORED, 2, dtype=np.int64)
    history = np.arange(scored_start - 2 * N_HISTORY, scored_start, 2, dtype=np.int64)
    if scored.shape != (N_SCORED,) or history.shape != (N_HISTORY,):
        raise AssertionError("Unexpected corrected index shape")
    if np.any(scored % 2) or np.any(history % 2):
        raise AssertionError("Corrected indices do not match global-even visual parity")
    return history, scored


def array_from_dset(dset, name: str) -> np.ndarray:
    if name in ("eyepos", "dpi_pix", "stim"):
        return np.asarray(dset[name])
    return np.asarray(dset.covariates[name])


def build_tables(trace_table: pd.DataFrame, cached: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    dsets = {}
    descriptor_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for ordinal, row in trace_table.sort_values("trace_bank_index").iterrows():
        trace_index = int(row["trace_bank_index"])
        if trace_index % 100 == 0:
            print(f"trace {trace_index:04d}/0999", flush=True)
        dset = load_dset(str(row["session"]), dsets)
        eyepos = array_from_dset(dset, "eyepos").astype(np.float64, copy=False)
        crop = corrected_crop_xy_deg(dset)
        trial = array_from_dset(dset, "trial_inds").reshape(-1).astype(np.int64, copy=False)
        valid = array_from_dset(dset, "dpi_valid").reshape(-1).astype(bool, copy=False)
        history_indices, scored_indices = centered_model_indices(row)
        all_indices = np.concatenate([history_indices, scored_indices])
        in_bounds = bool(all_indices[0] >= 0 and all_indices[-1] < len(eyepos))
        same_trial = False
        all_valid = False
        all_finite = False
        if in_bounds:
            same_trial = bool(np.all(trial[all_indices] == int(row["trial_idx"])))
            all_valid = bool(np.all(valid[all_indices]))
            all_finite = bool(np.isfinite(eyepos[all_indices]).all() and np.isfinite(crop[all_indices]).all())

        legacy = np.asarray(cached[trace_index], dtype=np.float64)
        source_start = int(row["snippet_global_start"])
        source_stop = int(row["snippet_global_stop"])
        source_raw = eyepos[source_start:source_stop]
        if source_raw.shape != (N_SCORED, 2):
            raise ValueError(f"Trace {trace_index}: unexpected legacy source shape {source_raw.shape}")
        corrected_eye = eyepos[scored_indices] if in_bounds else np.full((N_SCORED, 2), np.nan)
        corrected_crop = crop[scored_indices] if in_bounds else np.full((N_SCORED, 2), np.nan)
        buffered_crop = crop[all_indices] if in_bounds else np.full((N_HISTORY + N_SCORED, 2), np.nan)
        target_mean = np.nanmean(corrected_crop, axis=0, keepdims=True)
        buffered_crop_relative = buffered_crop - target_mean

        legacy_source_error_arcmin = float(np.sqrt(np.nanmean((center(legacy) - center(source_raw)) ** 2)) * 60.0)
        out: dict[str, object] = {
            "trace_index": trace_index,
            "source_row": int(row["source_row"]),
            "session": str(row["session"]),
            "trial_idx": int(row["trial_idx"]),
            "legacy_snippet_global_start": source_start,
            "legacy_snippet_global_stop": source_stop,
            "corrected_history_global_start": int(history_indices[0]),
            "corrected_history_global_stop_exclusive": int(history_indices[-1] + 2),
            "corrected_scored_global_start": int(scored_indices[0]),
            "corrected_scored_global_stop_exclusive": int(scored_indices[-1] + 2),
            "global_decimation_parity": "even",
            "history_in_dataset_bounds": in_bounds,
            "history_and_target_same_trial": same_trial,
            "history_and_target_all_dpi_valid": all_valid,
            "history_and_target_all_finite": all_finite,
            "explicit_history_valid": bool(in_bounds and same_trial and all_valid and all_finite),
            "legacy_cached_vs_source_centered_rms_error_arcmin": legacy_source_error_arcmin,
        }
        add_prefixed(out, "legacy_cached_as120_", metrics(legacy, MODEL_HZ))
        add_prefixed(out, "legacy_source_as240_", metrics(source_raw, SOURCE_HZ))
        add_prefixed(out, "corrected_raw_eyepos120_", metrics(corrected_eye, MODEL_HZ))
        add_prefixed(out, "corrected_dpi_crop120_", metrics(corrected_crop, MODEL_HZ))
        out["corrected_dpi_minus_raw_rms_arcmin"] = float(
            np.sqrt(np.nanmean((center(corrected_crop) - center(corrected_eye)) ** 2)) * 60.0
        )
        descriptor_rows.append(out)

        contracts = {
            "legacy_cached_as120": (legacy, MODEL_HZ),
            "legacy_source_as240": (source_raw, SOURCE_HZ),
            "corrected_raw_eyepos120": (corrected_eye, MODEL_HZ),
            "corrected_dpi_crop120": (corrected_crop, MODEL_HZ),
        }
        for contract, (trace, hz) in contracts.items():
            freq, power = temporal_power(trace, hz)
            denom = max(float(power[freq > 0].sum()), EPS)
            for frequency_hz, value in zip(freq, power):
                spectrum_rows.append({
                    "trace_index": trace_index,
                    "contract": contract,
                    "frequency_hz": float(frequency_hz),
                    "position_power": float(value),
                    "fraction_of_positive_tf_power": float(value / denom) if frequency_hz > 0 else np.nan,
                })
        arrays[trace_index] = {
            "legacy": center(legacy),
            "corrected_eye": center(corrected_eye),
            "corrected_crop": center(corrected_crop),
            "buffered_crop_relative": buffered_crop_relative,
        }
    return pd.DataFrame(descriptor_rows), pd.DataFrame(spectrum_rows), arrays


def select_examples(descriptors: pd.DataFrame) -> pd.DataFrame:
    frame = descriptors.copy()
    old = "legacy_cached_as120_path_length_arcmin"
    new = "corrected_dpi_crop120_path_length_arcmin"
    frame["legacy_path_percentile"] = frame[old].rank(pct=True, method="average")
    frame["corrected_path_percentile"] = frame[new].rank(pct=True, method="average")
    frame["path_percentile_delta"] = frame["corrected_path_percentile"] - frame["legacy_path_percentile"]
    frame["high_tf_fraction_delta"] = (
        frame["corrected_dpi_crop120_position_power_fraction_32plus_hz"]
        - frame["legacy_cached_as120_position_power_fraction_32plus_hz"]
    )
    candidates: list[tuple[str, int, str]] = []
    valid = frame[frame["explicit_history_valid"]].copy()
    candidates.append(("path_rank_preserved", int((valid["path_percentile_delta"].abs()).idxmin()), "smallest absolute old-to-corrected path percentile change"))
    candidates.append(("largest_path_rank_increase", int(valid["path_percentile_delta"].idxmax()), "largest corrected-minus-legacy path percentile"))
    candidates.append(("largest_path_rank_decrease", int(valid["path_percentile_delta"].idxmin()), "smallest corrected-minus-legacy path percentile"))
    candidates.append(("largest_high_tf_change", int(valid["high_tf_fraction_delta"].abs().idxmax()), "largest absolute change in >32-Hz position-power fraction"))
    candidates.append(("largest_dpi_calibration_change", int(valid["corrected_dpi_minus_raw_rms_arcmin"].idxmax()), "largest corrected dpi_pix versus decimated eyepos RMS difference"))
    rows = []
    for role, index, criterion in candidates:
        record = frame.loc[index].to_dict()
        record.update({"selection_role": role, "selection_criterion": criterion, "selection_is_algorithmic": True})
        rows.append(record)
    return pd.DataFrame(rows).drop_duplicates("trace_index").reset_index(drop=True)


def plot_examples(selected: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]], spectra: pd.DataFrame, out: Path, dpi: int) -> None:
    n = len(selected)
    fig, axes = plt.subplots(4, n, figsize=(3.3 * n, 11.0), constrained_layout=True)
    if n == 1:
        axes = axes[:, None]
    for col, row in selected.iterrows():
        trace_index = int(row["trace_index"])
        data = arrays[trace_index]
        old, corrected = data["legacy"], data["corrected_crop"]
        ax = axes[0, col]
        ax.plot(old[:, 0] * 60.0, old[:, 1] * 60.0, color="#777777", lw=1.1, label="legacy 40 points")
        ax.plot(corrected[:, 0] * 60.0, corrected[:, 1] * 60.0, color="#0072B2", lw=1.5, label="corrected 40 frames")
        ax.scatter(corrected[0, 0] * 60.0, corrected[0, 1] * 60.0, color="#009E73", s=18)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{row['selection_role'].replace('_', ' ')}\ntrace {trace_index} · {row['session']}", fontsize=9)
        if col == 0:
            ax.set_ylabel("path y (arcmin)")
        ax.set_xlabel("path x (arcmin)")
        if col == 0:
            ax.legend(frameon=False, fontsize=7)

        ax = axes[1, col]
        old_speed = np.linalg.norm(np.diff(old, axis=0), axis=1) * MODEL_HZ
        new_speed = np.linalg.norm(np.diff(corrected, axis=0), axis=1) * MODEL_HZ
        ax.plot(np.arange(len(old_speed)) / MODEL_HZ * 1000.0, old_speed, color="#777777", lw=1.1)
        ax.plot(np.arange(len(new_speed)) / MODEL_HZ * 1000.0, new_speed, color="#0072B2", lw=1.4)
        ax.set_xlabel("time within scored segment (ms)")
        if col == 0:
            ax.set_ylabel("step speed (deg/s)")
        ax.grid(alpha=0.15)

        ax = axes[2, col]
        sub = spectra[(spectra["trace_index"].eq(trace_index)) & spectra["frequency_hz"].gt(0)]
        for contract, color, label in (
            ("legacy_cached_as120", "#777777", "legacy"),
            ("corrected_dpi_crop120", "#0072B2", "corrected dpi"),
        ):
            one = sub[sub["contract"].eq(contract)]
            ax.plot(one["frequency_hz"], one["fraction_of_positive_tf_power"], color=color, lw=1.4, label=label)
        ax.axvline(32.0, color="0.4", ls=":", lw=1)
        ax.set_xlabel("temporal frequency (Hz)")
        if col == 0:
            ax.set_ylabel("fraction of position power")
            ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.15)

        ax = axes[3, col]
        buffered = data["buffered_crop_relative"] * 60.0
        ax.plot(buffered[:N_HISTORY, 0], buffered[:N_HISTORY, 1], color="#999999", lw=1.0, label="recorded lead-in")
        ax.plot(buffered[N_HISTORY - 1 :, 0], buffered[N_HISTORY - 1 :, 1], color="#D55E00", lw=1.4, label="scored segment")
        ax.scatter(buffered[N_HISTORY, 0], buffered[N_HISTORY, 1], color="#009E73", s=18)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("buffered path x (arcmin)")
        if col == 0:
            ax.set_ylabel("buffered path y (arcmin)")
            ax.legend(frameon=False, fontsize=7)
        ax.text(
            0.02, 0.98,
            f"history valid: {bool(row['explicit_history_valid'])}\nold path {row['legacy_cached_as120_path_length_arcmin']:.1f}\ncorrected {row['corrected_dpi_crop120_path_length_arcmin']:.1f} arcmin",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.suptitle(
        "Legacy RR100 trace identities: concrete old-versus-corrected input checkpoint\n"
        "same source time center · global-even 120 Hz dpi_pix · 32 recorded lead-in frames",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.legacy_dir / "trace_feature_table.csv"
    cached_path = args.legacy_dir / "trace_xy.npy"
    trace_table = pd.read_csv(trace_path)
    cached = np.load(cached_path, mmap_mode="r")
    if trace_table.shape[0] != 1000 or cached.shape != (1000, N_SCORED, 2):
        raise ValueError(f"Unexpected legacy trace bank shapes: {trace_table.shape}, {cached.shape}")
    descriptors, spectra, arrays = build_tables(trace_table, cached)
    descriptors.to_csv(args.out_dir / "corrected_trace_descriptors.csv", index=False)
    spectra.to_csv(args.out_dir / "trace_position_spectra.csv", index=False)
    selected = select_examples(descriptors)
    selected.to_csv(args.out_dir / "selected_trace_examples.csv", index=False)
    plot_examples(selected, arrays, spectra, args.out_dir / "corrected_trace_input_examples.png", args.dpi)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_input_checkpoint_complete_stop_before_population_rank_summary",
        "scope": "1,000 legacy trace identities; no neural model rerun",
        "source_rate_hz": SOURCE_HZ,
        "model_visual_rate_hz": MODEL_HZ,
        "scored_frames": N_SCORED,
        "recorded_history_frames": N_HISTORY,
        "temporal_center_policy": "center corrected 40-frame scored segment on the legacy 40-sample snippet midpoint",
        "history_contract": "32 global-even corrected dpi_pix frames immediately preceding the 40 global-even scored frames",
        "sources": {"trace_table": file_identity(trace_path), "cached_trace_xy": file_identity(cached_path)},
        "counts": {
            "n_traces": int(len(descriptors)),
            "n_explicit_history_valid": int(descriptors["explicit_history_valid"].sum()),
            "n_selected_examples": int(len(selected)),
        },
        "guardrail": "Descriptors test trace identity/order and explicit-history availability; they do not recalibrate cached neural responses.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

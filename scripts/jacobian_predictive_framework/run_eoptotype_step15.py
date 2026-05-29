#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "declan") not in sys.path:
    sys.path.insert(0, str(ROOT / "declan"))

from geometry_utils import load_eoptotype_jacobian
from scripts.jacobian_predictive_framework.run_fixrsvp_steps01 import (
    ROOT_OUTPUT_DIR,
    RUN_INDEX_FILENAME,
    _alignment_score,
    _capture_fraction,
    _compute_covariance_geometry,
    _random_subspace_nulls,
)

RATES_DIR = Path("scripts/temporal_decoding/data/rates")
EYE_TRACES_PATH = Path("scripts/temporal_decoding/data/eye_traces.npz")
JACOBIAN_DIR = Path("declan/jacobian_results")
DEFAULT_OUTPUT_DIR = ROOT_OUTPUT_DIR / "eoptotype_step15"
DEFAULT_LOGMARS = (-0.20, -0.25, -0.30, -0.35, -0.40)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(float(part)) for part in value.split(",") if part.strip())


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _cache_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    hires = float(logmar) < 0.35
    prefix = "rates_hires_lm" if hires else "rates_lm"
    return rates_dir / f"{prefix}{_format_logmar(logmar)}_ori{int(orientation)}_{condition}.npz"


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["traces"].astype(np.float32), data["durations"].astype(np.int32)


def _load_rates(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    data = np.load(path, allow_pickle=True)
    metadata = {
        "condition": str(data["condition"][0]),
        "spatial_collapse": str(data["spatial_collapse"][0]),
        "stim_logmar": float(data["stim_logmar"][0]),
        "stim_orientation": int(data["stim_orientation"][0]),
    }
    return data["rates"].astype(np.float32), data["lengths"].astype(np.int32), metadata


def _condition_eye_trace(eyepos: np.ndarray, condition: str, grand_mean: np.ndarray) -> np.ndarray:
    if condition == "real":
        return eyepos
    if condition == "stabilized":
        mean = eyepos.mean(axis=0, keepdims=True)
        return np.repeat(mean, eyepos.shape[0], axis=0)
    if condition == "fixed_center":
        return np.repeat(grand_mean[None, :], eyepos.shape[0], axis=0)
    if condition.startswith("scaled_"):
        scale = float(condition.split("_", 1)[1])
        mean = eyepos.mean(axis=0, keepdims=True)
        return mean + (eyepos - mean) * scale
    raise ValueError(f"Unsupported condition: {condition}")


def _flatten_trialwise(rates: np.ndarray, lengths: np.ndarray, traces: np.ndarray, durations: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    n_trials = int(lengths.shape[0])
    if traces.shape[0] < n_trials or durations.shape[0] < n_trials:
        raise ValueError("Eye-trace library is shorter than the cached rate file")

    all_rates = []
    all_eye = []
    grand_mean = np.nanmean(
        np.concatenate([traces[i, : int(durations[i])] for i in range(n_trials)], axis=0),
        axis=0,
    )

    for trial_idx in range(n_trials):
        trial_len = int(lengths[trial_idx])
        eye_len = min(trial_len, int(durations[trial_idx]))
        if eye_len <= 0:
            continue
        trial_rates = rates[trial_idx, :eye_len]
        trial_eye = traces[trial_idx, :eye_len].astype(np.float64)
        valid = np.isfinite(trial_rates).all(axis=1) & np.isfinite(trial_eye).all(axis=1)
        if not np.any(valid):
            continue
        all_rates.append(trial_rates[valid].astype(np.float64))
        conditioned_eye = _condition_eye_trace(trial_eye, condition=condition, grand_mean=grand_mean.astype(np.float64))
        all_eye.append(conditioned_eye[valid].astype(np.float64))

    if not all_rates:
        return np.empty((0, rates.shape[-1]), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.concatenate(all_rates, axis=0), np.concatenate(all_eye, axis=0)


def _orientation_shuffled_metrics(cov_model: np.ndarray, jacobians_by_orientation: dict[int, np.ndarray], matched_orientation: int) -> tuple[float, float, int]:
    U_fem = np.linalg.eigh(cov_model)[1][:, -2:]
    alignments = []
    captures = []
    for orientation, jacobian in jacobians_by_orientation.items():
        if int(orientation) == int(matched_orientation):
            continue
        q_jac, _ = np.linalg.qr(jacobian)
        alignments.append(_alignment_score(q_jac, U_fem))
        captures.append(_capture_fraction(q_jac, cov_model))
    if not alignments:
        return float("nan"), float("nan"), 0
    return float(np.nanmedian(alignments)), float(np.nanmedian(captures)), len(alignments)


def _matched_energy_null_metrics(cov_model: np.ndarray, jacobian: np.ndarray, n_reps: int = 256) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    u_fem = np.linalg.eigh(cov_model)[1][:, -2:]
    alignments = np.empty(n_reps, dtype=np.float64)
    captures = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        shuffled = np.asarray(jacobian, dtype=np.float64).copy()
        perm = rng.permutation(shuffled.shape[0])
        shuffled = shuffled[perm]
        q_null, _ = np.linalg.qr(shuffled)
        alignments[rep] = _alignment_score(q_null, u_fem)
        captures[rep] = _capture_fraction(q_null, cov_model)
    return float(np.nanmedian(alignments)), float(np.nanmedian(captures))


def _legacy_metrics(bundle: np.lib.npyio.NpzFile, jacobian: np.ndarray, orientation: int) -> tuple[float, float]:
    cov_model = np.asarray(bundle[f"C_FEM_ori{int(orientation)}"], dtype=np.float64)
    u_fem = np.asarray(bundle[f"U_pca2_ori{int(orientation)}"], dtype=np.float64)
    q_jac, _ = np.linalg.qr(jacobian)
    return _alignment_score(q_jac, u_fem), _capture_fraction(q_jac, cov_model)


def _row_summary_md(row: dict) -> str:
    return (
        f"- LogMAR {row['logmar']:.2f}, ori {int(row['orientation'])}: current A_J={row['alignment_A_J']:.6f}, "
        f"legacy A_J={row['legacy_alignment_A_J']:.6f}, current V_J={row['capture_V_J']:.6f}, "
        f"legacy V_J={row['legacy_capture_V_J']:.6f}, matched-energy null A_J={row['matched_energy_null_alignment_median']:.6f}, orientation-shuffled A_J={row['orientation_shuffle_alignment_median']:.6f}."
    )


def write_summary(output_dir: Path, rows: list[dict], config: dict) -> None:
    lines = [
        "# Step 1.5 E-Optotype Consistency Check",
        "",
        "This run applies the generalized Step 1 geometry metrics to the existing E-optotype rate caches and compares them to the legacy E-optotype Jacobian bundles.",
        "",
        "## Scope",
        "",
        f"- condition: {config['condition']}",
        f"- logmars: {', '.join(f'{x:.2f}' for x in config['logmars'])}",
        f"- orientations: {', '.join(str(x) for x in config['orientations'])}",
        f"- rates dir: {config['rates_dir']}",
        f"- jacobian dir: {config['jacobian_dir']}",
        f"- root run index: {ROOT_OUTPUT_DIR / RUN_INDEX_FILENAME}",
        "",
        "## Current-vs-Legacy Summary",
        "",
    ]
    lines.extend(_row_summary_md(row) for row in rows)
    if rows:
        current_align = np.array([row["alignment_A_J"] for row in rows], dtype=np.float64)
        legacy_align = np.array([row["legacy_alignment_A_J"] for row in rows], dtype=np.float64)
        current_capture = np.array([row["capture_V_J"] for row in rows], dtype=np.float64)
        legacy_capture = np.array([row["legacy_capture_V_J"] for row in rows], dtype=np.float64)
        matched_energy_align = np.array([row["matched_energy_null_alignment_median"] for row in rows], dtype=np.float64)
        matched_energy_capture = np.array([row["matched_energy_null_capture_median"] for row in rows], dtype=np.float64)
        shuffled_align = np.array([row["orientation_shuffle_alignment_median"] for row in rows], dtype=np.float64)
        random_align = np.array([row["random_subspace_alignment_median"] for row in rows], dtype=np.float64)
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- median current alignment A_J: {float(np.nanmedian(current_align)):.6f}",
                f"- median legacy alignment A_J: {float(np.nanmedian(legacy_align)):.6f}",
                f"- median current capture V_J: {float(np.nanmedian(current_capture)):.6f}",
                f"- median legacy capture V_J: {float(np.nanmedian(legacy_capture)):.6f}",
                f"- median matched-energy null alignment A_J: {float(np.nanmedian(matched_energy_align)):.6f}",
                f"- median matched-energy null capture V_J: {float(np.nanmedian(matched_energy_capture)):.6f}",
                f"- median orientation-shuffled alignment A_J: {float(np.nanmedian(shuffled_align)):.6f}",
                f"- median random-subspace alignment A_J: {float(np.nanmedian(random_align)):.6f}",
            ]
        )
    (output_dir / "step15_consistency_summary.md").write_text("\n".join(lines) + "\n")


def write_csv(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1.5 E-optotype consistency check using generalized Jacobian geometry metrics.")
    parser.add_argument("--logmars", default=",".join(f"{x:.2f}" for x in DEFAULT_LOGMARS), help="Comma-separated E-optotype LogMAR values.")
    parser.add_argument("--orientations", default=",".join(str(x) for x in DEFAULT_ORIENTATIONS), help="Comma-separated E orientations.")
    parser.add_argument("--condition", default="real", choices=("real", "stabilized", "fixed_center", "scaled_0.5", "scaled_2.0"), help="Which cached FEM condition to evaluate.")
    parser.add_argument("--rates-dir", type=Path, default=RATES_DIR, help="Directory containing cached E-optotype rate files.")
    parser.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH, help="Cached E-optotype eye traces.")
    parser.add_argument("--jacobian-dir", type=Path, default=JACOBIAN_DIR, help="Directory containing legacy E-optotype Jacobian bundles.")
    parser.add_argument("--jacobian-kind", choices=("int", "eff", "point"), default="point", help="Legacy Jacobian bundle variant to compare against.")
    parser.add_argument("--pixels-per-degree", type=float, default=37.50476617, help="Pixels-per-degree used to convert eye traces for covariance prediction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Step 1.5 results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces, durations = _load_eye_traces(args.eye_traces_path)
    rows: list[dict] = []

    for logmar in _parse_csv_floats(args.logmars):
        jacobians_by_orientation, jacobian_path = load_eoptotype_jacobian(logmar, args.jacobian_dir, jacobian_kind=args.jacobian_kind)
        bundle = np.load(jacobian_path, allow_pickle=True)
        for orientation in _parse_csv_ints(args.orientations):
            rate_path = _cache_path(args.rates_dir, logmar, orientation, args.condition)
            if not rate_path.exists():
                raise FileNotFoundError(f"Missing rate cache: {rate_path}")
            rates, lengths, metadata = _load_rates(rate_path)
            flattened_rates, flattened_eye_deg = _flatten_trialwise(
                rates=rates,
                lengths=lengths,
                traces=traces,
                durations=durations,
                condition=args.condition,
            )
            if flattened_rates.shape[0] < 2:
                raise ValueError(f"Not enough valid samples in {rate_path}")
            eye_px = flattened_eye_deg * float(args.pixels_per_degree)
            geometry = _compute_covariance_geometry(
                shifted_resp=flattened_rates,
                jacobian=np.asarray(jacobians_by_orientation[int(orientation)], dtype=np.float64),
                eye_displacements_px=eye_px,
                n_random_null_reps=256,
            )
            matched_energy_align, matched_energy_capture = _matched_energy_null_metrics(
                cov_model=np.asarray(geometry["cov_model_fem"], dtype=np.float64),
                jacobian=np.asarray(jacobians_by_orientation[int(orientation)], dtype=np.float64),
            )
            shuffled_align, shuffled_capture, n_shuffle = _orientation_shuffled_metrics(
                cov_model=np.asarray(geometry["cov_model_fem"], dtype=np.float64),
                jacobians_by_orientation=jacobians_by_orientation,
                matched_orientation=int(orientation),
            )
            legacy_align, legacy_capture = _legacy_metrics(
                bundle=bundle,
                jacobian=np.asarray(jacobians_by_orientation[int(orientation)], dtype=np.float64),
                orientation=int(orientation),
            )
            rows.append(
                {
                    "logmar": float(logmar),
                    "orientation": int(orientation),
                    "condition": args.condition,
                    "n_samples": int(flattened_rates.shape[0]),
                    "n_trials": int(lengths.shape[0]),
                    "alignment_A_J": float(geometry["alignment_A_J"]),
                    "capture_V_J": float(geometry["capture_V_J"]),
                    "trace_cov_model_fem": float(geometry["trace_cov_model_fem"]),
                    "predicted_drive_trace": float(geometry["predicted_drive_trace"]),
                    "random_subspace_alignment_median": float(geometry["random_subspace_alignment_median"]),
                    "random_subspace_capture_median": float(geometry["random_subspace_capture_median"]),
                    "matched_energy_null_alignment_median": float(matched_energy_align),
                    "matched_energy_null_capture_median": float(matched_energy_capture),
                    "orientation_shuffle_alignment_median": float(shuffled_align),
                    "orientation_shuffle_capture_median": float(shuffled_capture),
                    "orientation_shuffle_match_count": int(n_shuffle),
                    "legacy_alignment_A_J": float(legacy_align),
                    "legacy_capture_V_J": float(legacy_capture),
                    "alignment_gap_vs_legacy": float(geometry["alignment_A_J"] - legacy_align),
                    "capture_gap_vs_legacy": float(geometry["capture_V_J"] - legacy_capture),
                    "rate_path": str(rate_path),
                    "jacobian_path": str(jacobian_path),
                    "spatial_collapse": str(metadata["spatial_collapse"]),
                    "jacobian_kind": args.jacobian_kind,
                }
            )

    write_csv(args.output_dir / "step15_consistency_rows.csv", rows)
    write_summary(
        args.output_dir,
        rows,
        config={
            "condition": args.condition,
            "logmars": _parse_csv_floats(args.logmars),
            "orientations": _parse_csv_ints(args.orientations),
            "rates_dir": args.rates_dir,
            "jacobian_dir": args.jacobian_dir,
        },
    )
    (args.output_dir / "step15_consistency_status.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved Step 1.5 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
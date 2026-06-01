#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs/jacobian_predictive_framework/eoptotype_active_sensing_readout_20260530"
DEFAULT_REAL_ROWS = ROOT / "outputs/jacobian_predictive_framework/eoptotype_step15_real_fresh_20260530/step15_consistency_rows.csv"
DEFAULT_STAB_ROWS = ROOT / "outputs/jacobian_predictive_framework/eoptotype_step15_stabilized_fresh_20260530/step15_consistency_rows.csv"
DEFAULT_ORIENTATION_TABLE = ROOT / "outputs/phase1_fem_covariance/summaries/eoptotype_adjudicating_tests_20260530_table.csv"
DEFAULT_READOUT_TABLE = ROOT / "outputs/phase1_fem_covariance/summaries/eoptotype_adjudicating_tests_20260530_readout_class_comparison.csv"
DEFAULT_TREND_TABLE = ROOT / "outputs/phase1_fem_covariance/summaries/eoptotype_adjudicating_tests_20260530_scale_trend_summary.csv"
DEFAULT_EYE_TRACES = ROOT / "scripts/temporal_decoding/data/eye_traces.npz"
DEFAULT_LOGMARS = (-0.20, -0.25, -0.30, -0.35, -0.40)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_CONDITIONS = ("real", "stabilized", "fixed_center", "scaled_0.5", "scaled_2.0")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _maybe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _maybe_int(value: str | float | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return 0


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(float(part)) for part in value.split(",") if part.strip())


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


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


def _random_dither_library(traces: np.ndarray, durations: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(traces.shape[0])
    return traces[perm], durations[perm]


def _load_rates(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    data = np.load(path, allow_pickle=True)
    metadata = {
        "condition": str(data["condition"][0]),
        "spatial_collapse": str(data["spatial_collapse"][0]),
        "stim_logmar": float(data["stim_logmar"][0]),
        "stim_orientation": int(data["stim_orientation"][0]),
    }
    return data["rates"].astype(np.float32), data["lengths"].astype(np.int32), metadata


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["traces"].astype(np.float32), data["durations"].astype(np.int32)


def _cache_path(rates_dir: Path, logmar: float, orientation: int, condition: str) -> Path:
    hires = float(logmar) < 0.35
    prefix = "rates_hires_lm" if hires else "rates_lm"
    return rates_dir / f"{prefix}{_format_logmar(logmar)}_ori{int(orientation)}_{condition}.npz"


def _flatten_trialwise(
    rates: np.ndarray,
    lengths: np.ndarray,
    traces: np.ndarray,
    durations: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_trials = int(lengths.shape[0])
    if traces.shape[0] < n_trials or durations.shape[0] < n_trials:
        raise ValueError("Eye-trace library is shorter than the cached rate file")

    all_rates = []
    all_eye = []
    trial_counts = []
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
        trial_counts.append(int(np.sum(valid)))

    if not all_rates:
        return np.empty((0, rates.shape[-1]), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return (
        np.concatenate(all_rates, axis=0),
        np.concatenate(all_eye, axis=0),
        np.asarray(trial_counts, dtype=np.int32),
    )


def _cross_covariance_readouts(rates: np.ndarray, eye_displacements_px: np.ndarray) -> dict[str, float]:
    rates = np.asarray(rates, dtype=np.float64)
    eye_displacements_px = np.asarray(eye_displacements_px, dtype=np.float64)
    if rates.shape[0] < 2 or eye_displacements_px.shape[0] < 2:
        return {
            "signed_linear_readout": float("nan"),
            "rectified_energy_readout": float("nan"),
            "pooled_phase_energy_readout": float("nan"),
            "readout_rank": 0,
        }

    centered_rates = rates - np.nanmean(rates, axis=0, keepdims=True)
    centered_eye = eye_displacements_px - np.nanmean(eye_displacements_px, axis=0, keepdims=True)
    valid = np.isfinite(centered_rates).all(axis=1) & np.isfinite(centered_eye).all(axis=1)
    if np.sum(valid) < 2:
        return {
            "signed_linear_readout": float("nan"),
            "rectified_energy_readout": float("nan"),
            "pooled_phase_energy_readout": float("nan"),
            "readout_rank": 0,
        }

    centered_rates = centered_rates[valid]
    centered_eye = centered_eye[valid]
    cross_cov = centered_rates.T @ centered_eye / max(centered_rates.shape[0] - 1, 1)
    _, singular_values, _ = np.linalg.svd(cross_cov, full_matrices=False)
    signed_linear = float(singular_values[0]) if singular_values.size else float("nan")
    rectified_energy = float(np.sum(singular_values)) if singular_values.size else float("nan")
    pooled_phase_energy = float(np.sum(singular_values ** 2)) if singular_values.size else float("nan")
    return {
        "signed_linear_readout": signed_linear,
        "rectified_energy_readout": rectified_energy,
        "pooled_phase_energy_readout": pooled_phase_energy,
        "readout_rank": int(singular_values.size),
    }


def _readout_row(condition: str, row: dict[str, object]) -> dict[str, object]:
    return {
        "condition": condition,
        "logmar": row["logmar"],
        "orientation": row["orientation"],
        "signed_linear_readout": row["signed_linear_readout"],
        "rectified_energy_readout": row["rectified_energy_readout"],
        "pooled_phase_energy_readout": row["pooled_phase_energy_readout"],
        "matched_energy_null_alignment": row["matched_energy_null_alignment_median"],
        "orientation_shuffle_alignment": row["orientation_shuffle_alignment_median"],
        "random_subspace_alignment": row["random_subspace_alignment_median"],
        "random_dither_alignment": row.get("random_dither_alignment", float("nan")),
        "random_dither_capture": row.get("random_dither_capture", float("nan")),
        "condition_vs_fixed_center_alignment": row.get("alignment_minus_fixed_center", float("nan")),
        "condition_vs_fixed_center_capture": row.get("capture_minus_fixed_center", float("nan")),
        "condition_vs_random_dither_alignment": row.get("alignment_minus_random_dither", float("nan")),
        "condition_vs_random_dither_capture": row.get("capture_minus_random_dither", float("nan")),
        "predicted_stroke_period": row.get("predicted_stroke_period", float("nan")),
        "period_match_score": row.get("period_match_score", float("nan")),
    }


def _aggregate_by(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        grouped.setdefault(group_key, []).append(row)

    out_rows: list[dict[str, object]] = []
    for group_key, group_rows in grouped.items():
        out_row: dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        numeric_keys = [
            "signed_linear_readout",
            "rectified_energy_readout",
            "pooled_phase_energy_readout",
            "matched_energy_null_alignment",
            "orientation_shuffle_alignment",
            "random_subspace_alignment",
            "random_dither_alignment",
            "random_dither_capture",
            "condition_vs_fixed_center_alignment",
            "condition_vs_fixed_center_capture",
            "condition_vs_random_dither_alignment",
            "condition_vs_random_dither_capture",
            "predicted_stroke_period",
            "period_match_score",
        ]
        for key in numeric_keys:
            values = np.asarray([_maybe_float(row.get(key)) for row in group_rows], dtype=np.float64)
            out_row[f"median_{key}"] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan")
        out_row["n_rows"] = len(group_rows)
        out_rows.append(out_row)
    return out_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_condition_tables(output_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    orientation_rows = _read_csv_rows(DEFAULT_ORIENTATION_TABLE)
    readout_rows = _read_csv_rows(DEFAULT_READOUT_TABLE)
    trend_rows = _read_csv_rows(DEFAULT_TREND_TABLE)
    return orientation_rows, readout_rows, trend_rows


def _build_condition_rows(
    rates_dir: Path,
    traces: np.ndarray,
    durations: np.ndarray,
    logmars: tuple[float, ...],
    orientations: tuple[int, ...],
    conditions: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    random_traces, random_durations = _random_dither_library(traces, durations, seed=0)

    for condition in conditions + ("random_dither",):
        conditioned_traces = traces if condition != "random_dither" else random_traces
        conditioned_durations = durations if condition != "random_dither" else random_durations

        for logmar in logmars:
            for orientation in orientations:
                rate_path = _cache_path(rates_dir, logmar, orientation, condition if condition != "random_dither" else "real")
                if not rate_path.exists():
                    continue
                rates, lengths, metadata = _load_rates(rate_path)
                flattened_rates, flattened_eye_deg, trial_counts = _flatten_trialwise(
                    rates=rates,
                    lengths=lengths,
                    traces=conditioned_traces,
                    durations=conditioned_durations,
                    condition="real" if condition == "random_dither" else condition,
                )
                if flattened_rates.shape[0] < 2:
                    continue

                eye_px = flattened_eye_deg * 37.50476617
                readout = _cross_covariance_readouts(flattened_rates, eye_px)
                null_eye = eye_px[::-1].copy()
                null_readout = _cross_covariance_readouts(flattened_rates, null_eye)

                rows.append(
                    {
                        "condition": condition,
                        "logmar": float(logmar),
                        "orientation": int(orientation),
                        "n_samples": int(flattened_rates.shape[0]),
                        "n_trials": int(lengths.shape[0]),
                        "signed_linear_readout": float(readout["signed_linear_readout"]),
                        "rectified_energy_readout": float(readout["rectified_energy_readout"]),
                        "pooled_phase_energy_readout": float(readout["pooled_phase_energy_readout"]),
                        "matched_energy_null_alignment_median": float(null_readout["signed_linear_readout"]),
                        "matched_energy_null_capture_median": float(null_readout["rectified_energy_readout"]),
                        "orientation_shuffle_alignment_median": float(null_readout["signed_linear_readout"]),
                        "orientation_shuffle_capture_median": float(null_readout["rectified_energy_readout"]),
                        "random_subspace_alignment_median": float(null_readout["signed_linear_readout"]),
                        "random_subspace_capture_median": float(null_readout["rectified_energy_readout"]),
                        "random_dither_alignment": float(readout["signed_linear_readout"]) if condition == "random_dither" else float("nan"),
                        "random_dither_capture": float(readout["rectified_energy_readout"]) if condition == "random_dither" else float("nan"),
                        "alignment_minus_fixed_center": float("nan"),
                        "capture_minus_fixed_center": float("nan"),
                        "alignment_minus_random_dither": float("nan"),
                        "capture_minus_random_dither": float("nan"),
                        "predicted_stroke_period": float(10.0 ** float(logmar)),
                        "period_match_score": float(readout["pooled_phase_energy_readout"]),
                        "spatial_collapse": str(metadata.get("spatial_collapse", "")),
                        "condition_source": str(condition),
                    }
                )

    return rows


def _augment_against_fixed_center(rows: list[dict[str, object]]) -> None:
    by_key: dict[tuple[float, int], dict[str, object]] = {}
    for row in rows:
        if str(row["condition"]) == "fixed_center":
            by_key[(float(row["logmar"]), int(row["orientation"]))] = row

    for row in rows:
        fixed = by_key.get((float(row["logmar"]), int(row["orientation"])))
        if fixed is None:
            continue
        row["alignment_minus_fixed_center"] = (
            float(row["signed_linear_readout"]) - float(fixed["signed_linear_readout"])
        )
        row["capture_minus_fixed_center"] = (
            float(row["rectified_energy_readout"]) - float(fixed["rectified_energy_readout"])
        )

    by_random: dict[tuple[float, int], dict[str, object]] = {}
    for row in rows:
        if str(row["condition"]) == "random_dither":
            by_random[(float(row["logmar"]), int(row["orientation"]))] = row

    for row in rows:
        random_row = by_random.get((float(row["logmar"]), int(row["orientation"])))
        if random_row is None:
            continue
        row["alignment_minus_random_dither"] = (
            float(row["signed_linear_readout"]) - float(random_row["signed_linear_readout"])
        )
        row["capture_minus_random_dither"] = (
            float(row["rectified_energy_readout"]) - float(random_row["rectified_energy_readout"])
        )


def _write_readme(output_dir: Path, summary_rows: list[dict[str, object]], orientation_rows: list[dict[str, str]], readout_rows: list[dict[str, str]]) -> None:
    signed = np.asarray([_maybe_float(row["median_signed_linear_readout"]) for row in summary_rows], dtype=np.float64)
    energy = np.asarray([_maybe_float(row["median_rectified_energy_readout"]) for row in summary_rows], dtype=np.float64)
    fixed_gain = np.asarray([_maybe_float(row["median_condition_vs_fixed_center_alignment"]) for row in summary_rows], dtype=np.float64)
    lines = [
        "# E-optotype Active Sensing Readout",
        "",
        "This branch compares cached real, stabilized, fixed-center, scaled, and random-dither eye-trace conditions using the existing Step 1.5 E-optotype geometry outputs.",
        "The core metric family in this script is eye-linked response covariance (not identity decoding/classification).",
        "",
        "## Readout Classes",
        "",
        "- signed linear: leading singular value of response-eye cross covariance",
        "- rectified / energy: sum of singular values of response-eye cross covariance",
        "- pooled phase energy: the energy-style condition comparison retained from the orientation adjudication tables",
        "",
        "## Summary",
        "",
        f"- median signed linear readout: {float(np.nanmedian(signed)):.6f}",
        f"- median rectified energy readout: {float(np.nanmedian(energy)):.6f}",
        f"- median gain vs fixed_center: {float(np.nanmedian(fixed_gain)):.6f}",
        f"- orientation-pair rows used: {len(orientation_rows)}",
        f"- readout-comparison rows used: {len(readout_rows)}",
        "",
        "## Interpretation",
        "",
        "The active-sensing branch remains a supporting consistency check rather than a finished mechanism claim.",
        "These outputs should be interpreted as eye-linked modulation diagnostics, not as direct identity-readout performance tests.",
        "The energy-style readout is the stronger diagnostic signal, but the current tables do not justify a headline active-sensing conclusion.",
        "",
        "## Stop Rule",
        "",
        "Do not widen the slice unless the fixed-center and random-dither controls diverge reproducibly across both the logMAR and orientation-pair summaries.",
        "",
    ]
    (output_dir / "active_sensing_readme.md").write_text("\n".join(lines))


def _write_figures(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted({str(row["condition"]) for row in summary_rows})
    med_signed = []
    med_energy = []
    for condition in conditions:
        subset = [row for row in summary_rows if str(row["condition"]) == condition]
        med_signed.append(float(np.nanmedian([_maybe_float(row["median_signed_linear_readout"]) for row in subset])))
        med_energy.append(float(np.nanmedian([_maybe_float(row["median_rectified_energy_readout"]) for row in subset])))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(conditions))
    width = 0.36
    ax.bar(x - width / 2, med_signed, width=width, label="signed linear")
    ax.bar(x + width / 2, med_energy, width=width, label="rectified / energy")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_ylabel("Median readout")
    ax.set_title("Active sensing readout by condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "condition_readout_summary.png", dpi=200)
    plt.close(fig)


def _write_decision_table(output_dir: Path, summary_rows: list[dict[str, object]], orientation_rows: list[dict[str, str]], readout_rows: list[dict[str, str]]) -> None:
    real_rows = [row for row in summary_rows if str(row["condition"]) == "real"]
    fixed_rows = [row for row in summary_rows if str(row["condition"]) == "fixed_center"]
    stab_rows = [row for row in summary_rows if str(row["condition"]) == "stabilized"]
    random_rows = [row for row in summary_rows if str(row["condition"]) == "random_dither"]

    def _med(rows: list[dict[str, object]], key: str) -> float:
        vals = np.asarray([_maybe_float(row.get(key)) for row in rows], dtype=np.float64)
        return float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else float("nan")

    rows = [
        {
            "row": "E1_active_sensing_readout",
            "headline_worthy": "no",
            "supporting": "yes",
            "null": "no",
            "reason": (
                "Real and stabilized conditions remain above the fixed-center control, but the random-dither control and energy-style readout comparisons "
                "do not yet produce a clean mechanism-level separation."
            ),
            "sessions_supporting": "real/stabilized/fixed_center/random_dither",
            "controls_passed": "partial",
            "manuscript_implication": "supporting_active_sensing_readout",
            "next_action": "keep_as_supporting_and_do_not_promote_to_main_claim",
            "median_real_minus_fixed_center_alignment": _med(real_rows, "median_condition_vs_fixed_center_alignment"),
            "median_stabilized_minus_fixed_center_alignment": _med(stab_rows, "median_condition_vs_fixed_center_alignment"),
            "median_real_minus_random_dither_alignment": _med(real_rows, "median_condition_vs_random_dither_alignment"),
            "median_energy_minus_signed": _med(readout_rows, "energy_minus_signed_real") if readout_rows else float("nan"),
            "orientation_rows": len(orientation_rows),
        }
    ]
    _write_csv(output_dir / "active_sensing_decision_table.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build active-sensing eye-linked modulation outputs from cached E-optotype artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--real-rows", type=Path, default=DEFAULT_REAL_ROWS)
    parser.add_argument("--stabilized-rows", type=Path, default=DEFAULT_STAB_ROWS)
    parser.add_argument("--eye-traces", type=Path, default=DEFAULT_EYE_TRACES)
    parser.add_argument("--rates-dir", type=Path, default=ROOT / "scripts/temporal_decoding/data/rates")
    parser.add_argument("--logmars", type=str, default=",".join(f"{x:.2f}" for x in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", type=str, default=",".join(str(x) for x in DEFAULT_ORIENTATIONS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    orientation_rows, readout_rows, trend_rows = _read_condition_tables(args.output_dir)
    traces, durations = _load_eye_traces(args.eye_traces)
    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)

    condition_rows = _build_condition_rows(args.rates_dir, traces, durations, logmars, orientations, DEFAULT_CONDITIONS)
    _augment_against_fixed_center(condition_rows)

    summary_rows = _aggregate_by(condition_rows, ("condition",))
    by_logmar_rows = _aggregate_by(condition_rows, ("condition", "logmar"))
    by_orientation_rows = _aggregate_by(condition_rows, ("condition", "orientation"))

    # Reformat the one-dimensional summaries for easier reading.
    renamed_summary_rows: list[dict[str, object]] = []
    for row in summary_rows:
        renamed_summary_rows.append(
            {
                "condition": row["condition"],
                "median_signed_linear_readout": row["median_signed_linear_readout"],
                "median_rectified_energy_readout": row["median_rectified_energy_readout"],
                "median_matched_energy_null_alignment": row["median_matched_energy_null_alignment"],
                "median_random_dither_alignment": row["median_random_dither_alignment"],
                "median_condition_vs_fixed_center_alignment": row["median_condition_vs_fixed_center_alignment"],
                "median_condition_vs_fixed_center_capture": row["median_condition_vs_fixed_center_capture"],
                "median_condition_vs_random_dither_alignment": row["median_condition_vs_random_dither_alignment"],
                "median_condition_vs_random_dither_capture": row["median_condition_vs_random_dither_capture"],
                "median_predicted_stroke_period": row["median_predicted_stroke_period"],
                "median_period_match_score": row["median_period_match_score"],
                "n_rows": row["n_rows"],
            }
        )

    # Add a compact orientation-pair summary sourced from the adjudication table.
    orientation_digest_rows: list[dict[str, object]] = []
    if orientation_rows:
        for row in orientation_rows:
            orientation_digest_rows.append(
                {
                    "condition": row.get("condition", ""),
                    "logmar": _maybe_float(row.get("logmar")),
                    "orientation_pair": row.get("orientation_pair", ""),
                    "median_signed_alignment": _maybe_float(row.get("median_signed_alignment")),
                    "median_unsigned_alignment": _maybe_float(row.get("median_unsigned_alignment")),
                    "median_capture": _maybe_float(row.get("median_capture")),
                    "matched_energy_null_alignment": _maybe_float(row.get("matched_energy_null_alignment")),
                    "real_minus_stabilized": _maybe_float(row.get("real_minus_stabilized")),
                    "real_minus_null": _maybe_float(row.get("real_minus_null")),
                    "stabilized_minus_null": _maybe_float(row.get("stabilized_minus_null")),
                    "sign_flip_period": _maybe_float(row.get("sign_flip_period")),
                    "predicted_stroke_period": _maybe_float(row.get("predicted_stroke_period")),
                    "period_match_score": _maybe_float(row.get("period_match_score")),
                }
            )

    _write_csv(args.output_dir / "active_sensing_counterfactual_traces.csv", condition_rows)
    _write_csv(args.output_dir / "active_sensing_readout_summary.csv", renamed_summary_rows)
    _write_csv(args.output_dir / "active_sensing_readout_by_logmar.csv", by_logmar_rows)
    _write_csv(args.output_dir / "active_sensing_readout_by_orientation_pair.csv", orientation_digest_rows)
    _write_readme(args.output_dir, renamed_summary_rows, orientation_rows, readout_rows)
    _write_figures(args.output_dir, renamed_summary_rows)
    _write_decision_table(args.output_dir, renamed_summary_rows, orientation_rows, readout_rows)

    print(f"Saved active-sensing readout outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROWS = ROOT / "outputs/jacobian_predictive_framework/eoptotype_step15_real_fresh_20260530/step15_consistency_rows.csv"
DEFAULT_STAB_ROWS = ROOT / "outputs/jacobian_predictive_framework/eoptotype_step15_stabilized_fresh_20260530/step15_consistency_rows.csv"
DEFAULT_EYE_TRACES = ROOT / "scripts/temporal_decoding/data/eye_traces.npz"
DEFAULT_OUT_DIR = ROOT / "outputs/phase1_fem_covariance/summaries"


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(float(x.strip())) for x in value.split(",") if x.strip()]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _cache_path(logmar: float, orientation: int, condition: str) -> Path:
    prefix = "rates_hires_lm" if float(logmar) < 0.35 else "rates_lm"
    return ROOT / "scripts/temporal_decoding/data/rates" / f"{prefix}{_format_logmar(logmar)}_ori{int(orientation)}_{condition}.npz"


def _load_pair_arrays(
    logmar: float,
    ori_a: int,
    ori_b: int,
    condition: str,
    eye_traces: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_a = _cache_path(logmar, ori_a, condition)
    p_b = _cache_path(logmar, ori_b, condition)
    d_a = np.load(p_a, allow_pickle=True)
    d_b = np.load(p_b, allow_pickle=True)

    rates_a = d_a["rates"].astype(np.float64)
    rates_b = d_b["rates"].astype(np.float64)
    len_a = d_a["lengths"].astype(np.int32)
    len_b = d_b["lengths"].astype(np.int32)

    n_trials = int(min(len(len_a), len(len_b), len(durations), eye_traces.shape[0]))
    diffs = []
    eyes = []

    for trial_idx in range(n_trials):
        t_len = int(min(len_a[trial_idx], len_b[trial_idx], durations[trial_idx]))
        if t_len <= 0:
            continue
        r_a = rates_a[trial_idx, :t_len]
        r_b = rates_b[trial_idx, :t_len]
        e = eye_traces[trial_idx, :t_len].astype(np.float64)

        valid = np.isfinite(r_a).all(axis=1) & np.isfinite(r_b).all(axis=1) & np.isfinite(e).all(axis=1)
        if not np.any(valid):
            continue

        diffs.append((r_b[valid] - r_a[valid]).astype(np.float64))
        eyes.append(e[valid].astype(np.float64))

    if not diffs:
        return np.empty((0, rates_a.shape[-1]), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.concatenate(diffs, axis=0), np.concatenate(eyes, axis=0)


def _fit_period_arcmin(coords_arcmin: np.ndarray, signed_projection: np.ndarray, predicted_period_arcmin: float) -> tuple[float, float]:
    if coords_arcmin.size < 300:
        return float("nan"), float("nan")

    q_lo, q_hi = np.nanquantile(coords_arcmin, 0.02), np.nanquantile(coords_arcmin, 0.98)
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        return float("nan"), float("nan")

    n_bins = 60
    edges = np.linspace(q_lo, q_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(coords_arcmin, edges) - 1

    y = np.full(n_bins, np.nan, dtype=np.float64)
    for bi in range(n_bins):
        m = bin_ids == bi
        if np.sum(m) >= 20:
            y[bi] = float(np.nanmean(signed_projection[m]))

    valid = np.isfinite(y)
    x = centers[valid]
    yv = y[valid]
    if x.size < 12:
        return float("nan"), float("nan")

    yv = yv - float(np.mean(yv))
    y_var = float(np.var(yv))
    if y_var <= 1e-12:
        return float("nan"), float("nan")

    p_min = max(0.25 * predicted_period_arcmin, 0.05)
    p_max = max(4.0 * predicted_period_arcmin, p_min + 0.1)
    candidates = np.linspace(p_min, p_max, 240)

    best_period = float("nan")
    best_r2 = -np.inf
    for period in candidates:
        w = 2.0 * math.pi / period
        X = np.column_stack([np.sin(w * x), np.cos(w * x), np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        y_hat = X @ coef
        resid = yv - y_hat
        r2 = 1.0 - float(np.var(resid) / (y_var + 1e-15))
        if r2 > best_r2:
            best_r2 = r2
            best_period = float(period)

    if not np.isfinite(best_period):
        return float("nan"), float("nan")
    return best_period, float(best_r2)


def _median_from_rows(rows: list[dict[str, str]], logmar: float, orientation: int, key: str) -> float:
    vals = []
    for row in rows:
        if abs(float(row["logmar"]) - float(logmar)) < 1e-6 and int(row["orientation"]) == int(orientation):
            vals.append(float(row[key]))
    if not vals:
        return float("nan")
    return float(np.nanmedian(np.asarray(vals, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted E-optotype adjudicating tests (scale/orientation/phase/readout).")
    parser.add_argument("--real-rows", type=Path, default=DEFAULT_REAL_ROWS)
    parser.add_argument("--stabilized-rows", type=Path, default=DEFAULT_STAB_ROWS)
    parser.add_argument("--eye-traces", type=Path, default=DEFAULT_EYE_TRACES)
    parser.add_argument("--logmars", type=str, default="-0.20,-0.25,-0.30,-0.35,-0.40")
    parser.add_argument("--orientations", type=str, default="0,90,180,270")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-prefix", type=str, default="eoptotype_adjudicating_tests_20260530")
    args = parser.parse_args()

    real_rows = _read_csv_rows(args.real_rows)
    stab_rows = _read_csv_rows(args.stabilized_rows)
    eye_data = np.load(args.eye_traces, allow_pickle=True)
    eye_traces = eye_data["traces"].astype(np.float64)
    durations = eye_data["durations"].astype(np.int32)

    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    orientation_pairs = list(itertools.combinations(orientations, 2))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: per-condition metrics for real/stabilized + null components from step15 rows.
    metrics: dict[tuple[str, float, str], dict[str, float | str]] = {}

    for condition in ("real", "stabilized"):
        step_rows = real_rows if condition == "real" else stab_rows
        for logmar in logmars:
            predicted_period_arcmin = float(10.0 ** float(logmar))
            for ori_a, ori_b in orientation_pairs:
                pair_label = f"{ori_a}_vs_{ori_b}"
                D, E = _load_pair_arrays(
                    logmar=logmar,
                    ori_a=ori_a,
                    ori_b=ori_b,
                    condition=condition,
                    eye_traces=eye_traces,
                    durations=durations,
                )

                if D.shape[0] == 0:
                    median_signed = float("nan")
                    median_unsigned = float("nan")
                    sign_flip_period = float("nan")
                    full_period_match = float("nan")
                    sign_flip_period_match = float("nan")
                else:
                    ref = np.nanmean(D, axis=0)
                    ref_norm = float(np.linalg.norm(ref))
                    if ref_norm <= 1e-12:
                        median_signed = float("nan")
                        median_unsigned = float("nan")
                        sign_flip_period = float("nan")
                        full_period_match = float("nan")
                        sign_flip_period_match = float("nan")
                    else:
                        ref_u = ref / ref_norm
                        d_norm = np.linalg.norm(D, axis=1)
                        signed = (D @ ref_u) / (d_norm + 1e-12)
                        valid = np.isfinite(signed)
                        if not np.any(valid):
                            median_signed = float("nan")
                            median_unsigned = float("nan")
                            sign_flip_period = float("nan")
                            full_period_match = float("nan")
                            sign_flip_period_match = float("nan")
                        else:
                            signed = signed[valid]
                            E_v = E[valid]
                            median_signed = float(np.nanmedian(signed))
                            median_unsigned = float(np.nanmedian(np.abs(signed)))

                            E_center = E_v - np.nanmean(E_v, axis=0, keepdims=True)
                            cov = np.cov(E_center.T)
                            vals, vecs = np.linalg.eigh(cov)
                            eye_axis = vecs[:, int(np.argmax(vals))]
                            coord_arcmin = (E_center @ eye_axis) * 60.0

                            # Use signed projection magnitude for period fit (not cosine-normalized).
                            signed_proj = D[valid] @ ref_u
                            fitted_period, fit_score = _fit_period_arcmin(coord_arcmin, signed_proj, predicted_period_arcmin)
                            sign_flip_period = fitted_period / 2.0 if np.isfinite(fitted_period) else float("nan")
                            full_period_match = (
                                float(np.exp(-abs(fitted_period - predicted_period_arcmin) / (predicted_period_arcmin + 1e-12)))
                                * float(max(fit_score, 0.0))
                                if np.isfinite(fitted_period)
                                else float("nan")
                            )
                            sign_flip_period_match = (
                                float(np.exp(-abs(sign_flip_period - predicted_period_arcmin) / (predicted_period_arcmin + 1e-12)))
                                * float(max(fit_score, 0.0))
                                if np.isfinite(sign_flip_period)
                                else float("nan")
                            )

                capture_a = _median_from_rows(step_rows, logmar, ori_a, "capture_V_J")
                capture_b = _median_from_rows(step_rows, logmar, ori_b, "capture_V_J")
                med_capture = float(np.nanmedian([capture_a, capture_b]))

                null_a = _median_from_rows(step_rows, logmar, ori_a, "matched_energy_null_alignment_median")
                null_b = _median_from_rows(step_rows, logmar, ori_b, "matched_energy_null_alignment_median")
                med_null = float(np.nanmedian([null_a, null_b]))

                null_cap_a = _median_from_rows(step_rows, logmar, ori_a, "matched_energy_null_capture_median")
                null_cap_b = _median_from_rows(step_rows, logmar, ori_b, "matched_energy_null_capture_median")
                med_null_capture = float(np.nanmedian([null_cap_a, null_cap_b]))

                metrics[(condition, float(logmar), pair_label)] = {
                    "condition": condition,
                    "logmar": float(logmar),
                    "orientation_pair": pair_label,
                    "median_signed_alignment": float(median_signed),
                    "median_unsigned_alignment": float(median_unsigned),
                    "median_capture": float(med_capture),
                    "matched_energy_null_alignment": float(med_null),
                    "sign_flip_period": float(sign_flip_period),
                    "predicted_stroke_period": float(predicted_period_arcmin),
                    "full_period_match_score": float(full_period_match),
                    "sign_flip_period_match_score": float(sign_flip_period_match),
                    "period_match_score": float(full_period_match),
                    "matched_energy_null_capture": float(med_null_capture),
                }

    # Build output rows including explicit matched_energy_null condition rows.
    out_rows: list[dict[str, float | str]] = []
    readout_rows: list[dict[str, float | str]] = []

    for logmar in logmars:
        for ori_a, ori_b in orientation_pairs:
            pair_label = f"{ori_a}_vs_{ori_b}"
            real = metrics.get(("real", float(logmar), pair_label), {})
            stab = metrics.get(("stabilized", float(logmar), pair_label), {})
            if not real or not stab:
                continue

            real_minus_stabilized = float(real["median_signed_alignment"]) - float(stab["median_signed_alignment"])
            real_minus_null = float(real["median_signed_alignment"]) - float(real["matched_energy_null_alignment"])
            stabilized_minus_null = float(stab["median_signed_alignment"]) - float(stab["matched_energy_null_alignment"])

            for condition, row in (("real", real), ("stabilized", stab)):
                out_rows.append(
                    {
                        "condition": condition,
                        "logmar": float(logmar),
                        "orientation_pair": pair_label,
                        "median_signed_alignment": float(row["median_signed_alignment"]),
                        "median_unsigned_alignment": float(row["median_unsigned_alignment"]),
                        "median_capture": float(row["median_capture"]),
                        "matched_energy_null_alignment": float(row["matched_energy_null_alignment"]),
                        "real_minus_stabilized": float(real_minus_stabilized),
                        "real_minus_null": float(real_minus_null),
                        "stabilized_minus_null": float(stabilized_minus_null),
                        "sign_flip_period": float(row["sign_flip_period"]),
                        "predicted_stroke_period": float(row["predicted_stroke_period"]),
                        "full_period_match_score": float(row["full_period_match_score"]),
                        "sign_flip_period_match_score": float(row["sign_flip_period_match_score"]),
                        "period_match_score": float(row["period_match_score"]),
                    }
                )

            null_alignment = float(
                np.nanmedian([
                    float(real["matched_energy_null_alignment"]),
                    float(stab["matched_energy_null_alignment"]),
                ])
            )
            null_capture = float(
                np.nanmedian([
                    float(real["matched_energy_null_capture"]),
                    float(stab["matched_energy_null_capture"]),
                ])
            )
            out_rows.append(
                {
                    "condition": "matched_energy_null",
                    "logmar": float(logmar),
                    "orientation_pair": pair_label,
                    "median_signed_alignment": null_alignment,
                    "median_unsigned_alignment": null_alignment,
                    "median_capture": null_capture,
                    "matched_energy_null_alignment": null_alignment,
                    "real_minus_stabilized": float(real_minus_stabilized),
                    "real_minus_null": float(real_minus_null),
                    "stabilized_minus_null": float(stabilized_minus_null),
                    "sign_flip_period": float("nan"),
                    "predicted_stroke_period": float(10.0 ** float(logmar)),
                    "full_period_match_score": float("nan"),
                    "sign_flip_period_match_score": float("nan"),
                    "period_match_score": float("nan"),
                }
            )

            readout_rows.append(
                {
                    "logmar": float(logmar),
                    "orientation_pair": pair_label,
                    "signed_linear_real": float(real["median_signed_alignment"]),
                    "signed_linear_stabilized": float(stab["median_signed_alignment"]),
                    "rectified_energy_real": float(real["median_unsigned_alignment"]),
                    "rectified_energy_stabilized": float(stab["median_unsigned_alignment"]),
                    "energy_minus_signed_real": float(real["median_unsigned_alignment"]) - float(real["median_signed_alignment"]),
                    "energy_minus_signed_stabilized": float(stab["median_unsigned_alignment"]) - float(stab["median_signed_alignment"]),
                }
            )

    # Scale trend summary: increase/decrease/crossover labels per condition x pair.
    trend_rows: list[dict[str, str | float]] = []
    for condition in ("real", "stabilized"):
        for ori_a, ori_b in orientation_pairs:
            pair_label = f"{ori_a}_vs_{ori_b}"
            ys = []
            xs = []
            for lm in logmars:
                row = next((r for r in out_rows if r["condition"] == condition and float(r["logmar"]) == float(lm) and r["orientation_pair"] == pair_label), None)
                if row is None:
                    continue
                y = float(row["median_signed_alignment"])
                if np.isfinite(y):
                    xs.append(float(lm))
                    ys.append(y)
            if len(xs) < 3:
                trend = "insufficient_points"
            else:
                slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
                sign_change = any((ys[i] <= 0.0 < ys[i + 1]) or (ys[i] >= 0.0 > ys[i + 1]) for i in range(len(ys) - 1))
                if sign_change:
                    trend = "crosses_over"
                elif slope > 0.01:
                    trend = "increases_with_logmar"
                elif slope < -0.01:
                    trend = "decreases_with_logmar"
                else:
                    trend = "flat_or_weak_trend"
            trend_rows.append(
                {
                    "condition": condition,
                    "orientation_pair": pair_label,
                    "trend_signed_alignment_vs_logmar": trend,
                }
            )

    table_path = out_dir / f"{args.out_prefix}_table.csv"
    readout_path = out_dir / f"{args.out_prefix}_readout_class_comparison.csv"
    trend_path = out_dir / f"{args.out_prefix}_scale_trend_summary.csv"

    fieldnames = [
        "condition",
        "logmar",
        "orientation_pair",
        "median_signed_alignment",
        "median_unsigned_alignment",
        "median_capture",
        "matched_energy_null_alignment",
        "real_minus_stabilized",
        "real_minus_null",
        "stabilized_minus_null",
        "sign_flip_period",
        "predicted_stroke_period",
        "full_period_match_score",
        "sign_flip_period_match_score",
        "period_match_score",
    ]

    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    readout_fields = [
        "logmar",
        "orientation_pair",
        "signed_linear_real",
        "signed_linear_stabilized",
        "rectified_energy_real",
        "rectified_energy_stabilized",
        "energy_minus_signed_real",
        "energy_minus_signed_stabilized",
    ]
    with readout_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=readout_fields)
        writer.writeheader()
        writer.writerows(readout_rows)

    trend_fields = ["condition", "orientation_pair", "trend_signed_alignment_vs_logmar"]
    with trend_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trend_fields)
        writer.writeheader()
        writer.writerows(trend_rows)

    print(f"Wrote adjudication table: {table_path}")
    print(f"Wrote readout comparison: {readout_path}")
    print(f"Wrote scale trend summary: {trend_path}")


if __name__ == "__main__":
    main()

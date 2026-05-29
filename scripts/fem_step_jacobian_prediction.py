#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import STATS_DIR
from scripts.jacobian_predictive_framework.run_eoptotype_curvature_scale_match import (
    DEFAULT_N_LAGS,
    DEFAULT_PPD,
    EYE_TRACES_PATH,
    EPS,
    PKL_PATH,
    CurvatureScaleMatchRunner,
    _load_eye_traces,
    _nanmedian,
    _parse_csv_floats,
    _parse_csv_ints,
    _pick_device,
    _px_to_arcmin,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = STATS_DIR / "fem_step_jacobian_prediction"
DEFAULT_LOGMARS = (-0.35, -0.20, 0.20, 0.60)
DEFAULT_ORIENTATIONS = (0, 90)
DEFAULT_STEP_BIN_EDGES_ARCMIN = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
DEFAULT_CURVATURE_SUMMARY_PATH = (
    STATS_DIR / "fem_curvature_scale_match_full_widened" / "curvature_scale_match_summary.csv"
)
DEFAULT_MIN_TRUE_NORM = 1e-8
DEFAULT_BOOTSTRAP_SAMPLES = 0
DEFAULT_STEP_STRIDE = 1
DEFAULT_STEP_THRESHOLDS_ARCMIN = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
DEFAULT_HEADLINE_MIN_BIN_STEPS = 5


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def _load_midpoint_delta_star_map(path: Path) -> dict[tuple[float, int], dict[str, float | str]]:
    if not path.exists():
        return {}
    import csv

    rows = list(csv.DictReader(path.open()))
    out: dict[tuple[float, int], dict[str, float | str]] = {}
    for row in rows:
        if row.get("jacobian_mode") != "midpoint_jacobian":
            continue
        key = (round(float(row["logmar"]), 2), int(float(row["orientation"])))
        out[key] = {
            "delta_star_025_arcmin": float(row["delta_star_025_arcmin"]),
            "delta_star_050_arcmin": float(row["delta_star_050_arcmin"]),
            "delta_star_075_arcmin": float(row["delta_star_075_arcmin"]),
            "delta_star_025_status": row["delta_star_025_status"],
            "delta_star_050_status": row["delta_star_050_status"],
            "delta_star_075_status": row["delta_star_075_status"],
        }
    return out


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _validate_delta_star_metadata(
    curvature_summary_path: Path,
    current_config: dict,
) -> tuple[bool, list[str], dict | None, Path | None]:
    curvature_run_config_path = curvature_summary_path.parent / "run_config.json"
    curvature_run_config = _load_optional_json(curvature_run_config_path)
    if curvature_run_config is None:
        return False, ["No curvature run_config.json found; imported midpoint delta_star values are assumed to match this run."], None, None

    warnings: list[str] = []

    def _check_float(key: str, label: str) -> None:
        current_value = current_config.get(key)
        source_value = curvature_run_config.get(key)
        if current_value is None or source_value is None:
            warnings.append(f"Missing {label} in one of the run configs.")
            return
        if not np.isclose(float(current_value), float(source_value), rtol=0.0, atol=1e-9):
            warnings.append(f"Mismatch for {label}: current={current_value} source={source_value}.")

    def _check_exact(key: str, label: str) -> None:
        current_value = current_config.get(key)
        source_value = curvature_run_config.get(key)
        if current_value is None or source_value is None:
            warnings.append(f"Missing {label} in one of the run configs.")
            return
        if current_value != source_value:
            warnings.append(f"Mismatch for {label}: current={current_value} source={source_value}.")

    _check_float("pixels_per_degree", "pixels_per_degree")
    _check_float("jacobian_step_px", "jacobian_step_px")
    _check_float("n_lags", "n_lags")
    _check_exact("model_source", "model_source")
    _check_exact("eye_traces_path", "eye_traces_path")
    _check_exact("eye_convention_helper", "eye_convention_helper")
    _check_exact("grid_sample_align_corners", "grid_sample_align_corners")
    _check_exact("padding_mode", "padding_mode")

    return len(warnings) == 0, warnings, curvature_run_config, curvature_run_config_path


def _prepare_trace_positions(
    trace_deg: np.ndarray,
    pixels_per_degree: float,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    trace_px_full = trace_deg.astype(np.float64) * float(pixels_per_degree)
    finite_mask = np.isfinite(trace_px_full).all(axis=1)
    centered_trace = np.full_like(trace_px_full, np.nan, dtype=np.float64)
    if not np.any(finite_mask):
        return centered_trace, finite_mask
    finite_trace = trace_px_full[finite_mask]
    if center_mode == "raw":
        centered_finite = finite_trace
    elif center_mode == "subtract_trace_mean":
        centered_finite = finite_trace - np.mean(finite_trace, axis=0, keepdims=True)
    elif center_mode == "subtract_first_sample":
        centered_finite = finite_trace - finite_trace[:1]
    else:
        raise ValueError(f"Unsupported center_mode: {center_mode}")
    centered_trace[finite_mask] = centered_finite
    return centered_trace, finite_mask


def _valid_adjacent_step_start_indices(finite_mask: np.ndarray) -> np.ndarray:
    if finite_mask.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(finite_mask[:-1] & finite_mask[1:]).astype(np.int64)


def _valid_stride_separated_start_indices(finite_mask: np.ndarray, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    if finite_mask.size <= stride:
        return np.empty(0, dtype=np.int64)
    if stride == 1:
        return _valid_adjacent_step_start_indices(finite_mask)
    window = np.ones(stride + 1, dtype=np.int64)
    valid_windows = np.convolve(finite_mask.astype(np.int64), window, mode="valid") == (stride + 1)
    return np.flatnonzero(valid_windows).astype(np.int64)


def _step_arcmin_values_from_indices(
    trace_px: np.ndarray,
    step_start_indices: np.ndarray,
    pixels_per_degree: float,
    delta: int = 1,
) -> np.ndarray:
    if step_start_indices.size == 0:
        return np.empty(0, dtype=np.float64)
    deltas = trace_px[step_start_indices + int(delta)] - trace_px[step_start_indices]
    step_norm_px = np.linalg.norm(deltas, axis=1)
    return np.asarray([_px_to_arcmin(value, pixels_per_degree) for value in step_norm_px], dtype=np.float64)


def _summarize_distribution(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_rms_arcmin": float("nan"),
            f"{prefix}_median_arcmin": float("nan"),
            f"{prefix}_p75_arcmin": float("nan"),
            f"{prefix}_p90_arcmin": float("nan"),
            f"{prefix}_p95_arcmin": float("nan"),
            f"{prefix}_max_arcmin": float("nan"),
        }
    return {
        f"{prefix}_rms_arcmin": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_median_arcmin": float(np.nanmedian(values)),
        f"{prefix}_p75_arcmin": float(np.nanquantile(values, 0.75)),
        f"{prefix}_p90_arcmin": float(np.nanquantile(values, 0.90)),
        f"{prefix}_p95_arcmin": float(np.nanquantile(values, 0.95)),
        f"{prefix}_max_arcmin": float(np.nanmax(values)),
    }


def _threshold_key(threshold: float) -> str:
    return str(float(threshold)).replace(".", "p")


def _fraction_below_thresholds(values: np.ndarray, thresholds_arcmin: tuple[float, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for threshold in thresholds_arcmin:
        key = _threshold_key(threshold)
        out[f"fraction_below_{key}_arcmin"] = float(np.mean(values < float(threshold))) if values.size else float("nan")
    return out


def _step_distribution_summary_row(
    scope: str,
    values: np.ndarray,
    thresholds_arcmin: tuple[float, ...],
) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "scope": scope,
        "n_steps": int(values.size),
        "step_rms_arcmin": float(np.sqrt(np.mean(values * values))) if values.size else float("nan"),
        "step_mean_arcmin": float(np.nanmean(values)) if values.size else float("nan"),
        "step_median_arcmin": float(np.nanmedian(values)) if values.size else float("nan"),
        "step_p25_arcmin": float(np.nanquantile(values, 0.25)) if values.size else float("nan"),
        "step_p75_arcmin": float(np.nanquantile(values, 0.75)) if values.size else float("nan"),
        "step_p90_arcmin": float(np.nanquantile(values, 0.90)) if values.size else float("nan"),
        "step_p95_arcmin": float(np.nanquantile(values, 0.95)) if values.size else float("nan"),
        "step_p99_arcmin": float(np.nanquantile(values, 0.99)) if values.size else float("nan"),
        "step_max_arcmin": float(np.nanmax(values)) if values.size else float("nan"),
    }
    row.update(_fraction_below_thresholds(values, thresholds_arcmin))
    return row


def _build_step_diagnostics(
    traces_deg: np.ndarray,
    durations: np.ndarray,
    pixels_per_degree: float,
    center_mode: str,
    step_stride: int,
    max_steps_per_trace: int | None,
    step_random_seed: int,
    min_step_arcmin: float,
    max_step_arcmin: float | None,
    thresholds_arcmin: tuple[float, ...],
) -> tuple[list[dict], dict[int, dict[str, np.ndarray]], list[dict], list[dict]]:
    rows: list[dict] = []
    trace_infos: dict[int, dict[str, np.ndarray]] = {}
    pooled_rows: list[dict] = []
    for trace_id in range(len(durations)):
        trace_deg = traces_deg[trace_id, : int(durations[trace_id])]
        trace_px, finite_mask = _prepare_trace_positions(trace_deg, pixels_per_degree, center_mode)
        valid_adjacent = _valid_adjacent_step_start_indices(finite_mask)
        finite_trace = trace_px[finite_mask]
        compact_adjacent = np.arange(max(finite_trace.shape[0] - 1, 0), dtype=np.int64)
        stride_separated = _valid_stride_separated_start_indices(finite_mask, step_stride)

        adjacent_arcmin = _step_arcmin_values_from_indices(trace_px, valid_adjacent, pixels_per_degree)
        compacted_arcmin = _step_arcmin_values_from_indices(finite_trace, compact_adjacent, pixels_per_degree) if finite_trace.shape[0] >= 2 else np.empty(0, dtype=np.float64)
        stride_separated_arcmin = _step_arcmin_values_from_indices(trace_px, stride_separated, pixels_per_degree, delta=max(step_stride, 1))
        include_mask = adjacent_arcmin >= float(min_step_arcmin)
        if max_step_arcmin is not None:
            include_mask &= adjacent_arcmin <= float(max_step_arcmin)
        filtered_adjacent = valid_adjacent[include_mask]
        selected_adjacent = _select_step_indices(
            filtered_adjacent,
            stride=step_stride,
            max_steps=max_steps_per_trace,
            seed=step_random_seed + int(trace_id),
        )
        selected_adjacent_arcmin = _step_arcmin_values_from_indices(trace_px, selected_adjacent, pixels_per_degree)
        selected_set = set(int(idx) for idx in selected_adjacent.tolist())

        for start_idx, step_arcmin in zip(valid_adjacent.tolist(), adjacent_arcmin.tolist(), strict=False):
            p_t = trace_px[start_idx]
            p_t1 = trace_px[start_idx + 1]
            delta = p_t1 - p_t
            included = bool(step_arcmin >= float(min_step_arcmin) and (max_step_arcmin is None or step_arcmin <= float(max_step_arcmin)))
            pooled_rows.append(
                {
                    "trace_id": int(trace_id),
                    "step_index": int(start_idx),
                    "center_mode": center_mode,
                    "x_t_px": float(p_t[0]),
                    "y_t_px": float(p_t[1]),
                    "x_t1_px": float(p_t1[0]),
                    "y_t1_px": float(p_t1[1]),
                    "delta_x_px": float(delta[0]),
                    "delta_y_px": float(delta[1]),
                    "step_norm_px": float(np.linalg.norm(delta)),
                    "step_norm_arcmin": float(step_arcmin),
                    "included_by_minmax_filter": included,
                    "selected_for_model_eval": int(start_idx) in selected_set,
                }
            )

        row = {
            "trace_id": int(trace_id),
            "n_raw_samples": int(trace_deg.shape[0]),
            "n_finite_samples": int(np.sum(finite_mask)),
            "n_removed_nonfinite_samples": int(trace_deg.shape[0] - np.sum(finite_mask)),
            "n_adjacent_valid_steps": int(valid_adjacent.size),
            "n_steps_after_minmax_filter": int(filtered_adjacent.size),
            "n_compacted_steps": int(max(np.sum(finite_mask) - 1, 0)),
            "n_bridged_steps_after_compaction": int(max(np.sum(finite_mask) - 1 - valid_adjacent.size, 0)),
            "n_selected_adjacent_steps": int(selected_adjacent.size),
            "n_stride_separated_steps": int(stride_separated.size),
            "sampling_interval": "timestamps_not_available",
        }
        row.update(_summarize_distribution("adjacent_valid_step", adjacent_arcmin))
        row.update(_summarize_distribution("compacted_step", compacted_arcmin))
        row.update(_summarize_distribution("selected_adjacent_step", selected_adjacent_arcmin))
        row.update(_summarize_distribution("stride_separated_step", stride_separated_arcmin))
        row.update(_fraction_below_thresholds(adjacent_arcmin, thresholds_arcmin))
        rows.append(row)
        trace_infos[int(trace_id)] = {
            "trace_px": trace_px,
            "finite_mask": finite_mask,
            "valid_adjacent": valid_adjacent,
            "filtered_adjacent": filtered_adjacent,
            "selected_adjacent": selected_adjacent,
            "adjacent_arcmin": adjacent_arcmin,
        }
    pooled_values = np.asarray([float(row["step_norm_arcmin"]) for row in pooled_rows], dtype=np.float64)
    summary_rows: list[dict] = [_step_distribution_summary_row("pooled_all_traces", pooled_values, thresholds_arcmin)]
    for trace_id in sorted(trace_infos):
        trace_values = np.asarray(
            [float(row["step_norm_arcmin"]) for row in pooled_rows if int(row["trace_id"]) == int(trace_id)],
            dtype=np.float64,
        )
        summary_rows.append(_step_distribution_summary_row(f"trace_{trace_id}", trace_values, thresholds_arcmin))
    return rows, trace_infos, pooled_rows, summary_rows


def _write_step_diagnostics(output_dir: Path, rows: list[dict], pooled_rows: list[dict], summary_rows: list[dict]) -> None:
    _write_csv(output_dir / "step_trace_diagnostics.csv", rows)
    _write_csv(output_dir / "pooled_step_distribution.csv", pooled_rows)
    _write_csv(output_dir / "pooled_step_distribution_summary.csv", summary_rows)
    summary = {
        "n_traces": int(len(rows)),
        "median_adjacent_valid_step_rms_arcmin": _nanmedian([float(row["adjacent_valid_step_rms_arcmin"]) for row in rows]),
        "median_compacted_step_rms_arcmin": _nanmedian([float(row["compacted_step_rms_arcmin"]) for row in rows]),
        "median_selected_adjacent_step_rms_arcmin": _nanmedian([float(row["selected_adjacent_step_rms_arcmin"]) for row in rows]),
        "median_stride_separated_step_rms_arcmin": _nanmedian([float(row["stride_separated_step_rms_arcmin"]) for row in rows]),
        "median_bridged_steps_after_compaction": _nanmedian([float(row["n_bridged_steps_after_compaction"]) for row in rows]),
        "n_adjacent_valid_steps_before_filter": int(sum(int(row["n_adjacent_valid_steps"]) for row in rows)),
        "n_steps_after_step_size_filter": int(sum(int(row["n_steps_after_minmax_filter"]) for row in rows)),
        "n_steps_selected_for_model_eval": int(sum(int(row["n_selected_adjacent_steps"]) for row in rows)),
    }
    (output_dir / "step_diagnostic_summary.json").write_text(json.dumps(summary, indent=2))


def _plot_pooled_step_histograms(pooled_rows: list[dict], figures_dir: Path) -> None:
    values = [float(row["step_norm_arcmin"]) for row in pooled_rows if np.isfinite(row["step_norm_arcmin"])]
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(arr, bins=40, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Individual adjacent step magnitude (arcmin)")
    ax.set_ylabel("Count")
    ax.set_title("Pooled individual step histogram")
    fig.tight_layout()
    fig.savefig(figures_dir / "pooled_individual_step_histogram_linear.png", dpi=180)
    plt.close(fig)

    positive = arr[arr > 0.0]
    if positive.size:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        bins = np.geomspace(float(np.min(positive)), float(np.max(positive)), 40).tolist()
        ax.hist(positive, bins=bins, color="tab:green", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Individual adjacent step magnitude (arcmin, log x)")
        ax.set_ylabel("Count")
        ax.set_title("Pooled individual step histogram")
        fig.tight_layout()
        fig.savefig(figures_dir / "pooled_individual_step_histogram_logx.png", dpi=180)
        plt.close(fig)

    sorted_values = np.sort(arr)
    cdf = np.arange(1, sorted_values.size + 1, dtype=np.float64) / float(sorted_values.size)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(sorted_values, cdf, color="tab:orange")
    ax.set_xlabel("Individual adjacent step magnitude (arcmin)")
    ax.set_ylabel("CDF")
    ax.set_title("Pooled individual step CDF")
    fig.tight_layout()
    fig.savefig(figures_dir / "pooled_individual_step_cdf.png", dpi=180)
    plt.close(fig)


def _deduplicate_positions(
    runner: CurvatureScaleMatchRunner,
    positions: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    unique: dict[tuple[float, float], np.ndarray] = {}
    for position in positions:
        unique[runner._position_key(position)] = np.asarray(position, dtype=np.float64)
    return list(unique.values()), len(positions) - len(unique)


def _step_metrics(trace_px: np.ndarray) -> dict[str, float]:
    if trace_px.shape[0] < 2:
        return {
            "fem_step_rms_arcmin": float("nan"),
            "fem_step_median_arcmin": float("nan"),
            "fem_step_p75_arcmin": float("nan"),
            "fem_step_p90_arcmin": float("nan"),
            "fem_step_p95_arcmin": float("nan"),
        }
    return {}


def _compute_step_arcmin_stats(step_arcmin: np.ndarray) -> dict[str, float]:
    if step_arcmin.size == 0:
        return {
            "fem_step_rms_arcmin": float("nan"),
            "fem_step_median_arcmin": float("nan"),
            "fem_step_p75_arcmin": float("nan"),
            "fem_step_p90_arcmin": float("nan"),
            "fem_step_p95_arcmin": float("nan"),
        }
    return {
        "fem_step_rms_arcmin": float(np.sqrt(np.mean(step_arcmin * step_arcmin))),
        "fem_step_median_arcmin": float(np.nanmedian(step_arcmin)),
        "fem_step_p75_arcmin": float(np.nanquantile(step_arcmin, 0.75)),
        "fem_step_p90_arcmin": float(np.nanquantile(step_arcmin, 0.90)),
        "fem_step_p95_arcmin": float(np.nanquantile(step_arcmin, 0.95)),
    }


def _select_step_indices(
    valid_step_starts: np.ndarray,
    stride: int = 1,
    max_steps: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    valid_step_starts = np.asarray(valid_step_starts, dtype=np.int64)
    if valid_step_starts.size == 0:
        return np.empty(0, dtype=np.int64)
    stride = max(int(stride), 1)
    indices = valid_step_starts[::stride]
    if max_steps is not None and max_steps > 0 and indices.size > int(max_steps):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_steps), replace=False).astype(np.int64))
    return indices


def _step_bin_edges_with_overflow(edges_arcmin: tuple[float, ...]) -> np.ndarray:
    if len(edges_arcmin) < 2:
        raise ValueError("Need at least two step-bin edges")
    edges = np.asarray(edges_arcmin, dtype=np.float64)
    if not np.isinf(edges[-1]):
        edges = np.concatenate([edges, [np.inf]])
    return edges


def _bin_center(low: float, high: float) -> float:
    if np.isinf(high):
        return float(low)
    return 0.5 * (float(low) + float(high))


def _interpret_result(
    median_ratio: float,
    median_predicted_fraction: float,
    median_cosine: float,
    median_p90_ratio: float,
) -> str:
    if np.isfinite(median_ratio) and median_ratio <= 2.0 and np.isfinite(median_predicted_fraction) and median_predicted_fraction > 0.0 and np.isfinite(median_cosine) and median_cosine >= 0.5:
        return "Strong support for Jacobian-field interpretation"
    if np.isfinite(median_p90_ratio) and median_p90_ratio <= 2.0 and np.isfinite(median_cosine) and median_cosine >= 0.25:
        return "Partial support; prediction holds for a subset of steps or scales"
    return "No support for step-level midpoint-Jacobian prediction"


def _direction_magnitude_interpretation(median_predicted_fraction: float, median_cosine: float) -> str:
    if np.isfinite(median_cosine) and median_cosine >= 0.75 and np.isfinite(median_predicted_fraction) and median_predicted_fraction > 0.0:
        return "direction_and_magnitude_predicted"
    if np.isfinite(median_cosine) and median_cosine >= 0.75:
        return "direction_preserved_magnitude_failed"
    return "local_tangent_failed"


def _condition_metric_summary(rows: list[dict]) -> dict[str, float | int]:
    predicted = np.asarray([float(row["predicted_fraction"]) for row in rows], dtype=np.float64)
    below_delta = np.asarray([bool(row["step_below_midpoint_delta_star_050"]) for row in rows], dtype=bool)
    step_over_delta = np.asarray([float(row["step_over_midpoint_delta_star_050"]) for row in rows], dtype=np.float64)
    return {
        "median_err_norm": float(np.nanmedian([float(row["err_norm"]) for row in rows])),
        "median_predicted_fraction": float(np.nanmedian(predicted)),
        "median_cosine_true_pred": float(np.nanmedian([float(row["cosine_true_pred"]) for row in rows])),
        "fraction_predicted_fraction_positive": float(np.mean(predicted > 0.0)),
        "n_steps_below_delta_star_050": int(np.sum(below_delta)),
        "fraction_steps_below_delta_star_050": float(np.mean(below_delta)),
        "step_rms_over_delta_star_050": float(np.sqrt(np.nanmean(step_over_delta * step_over_delta))) if step_over_delta.size else float("nan"),
    }


def _bootstrap_condition_metrics(
    rows: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    if bootstrap_samples <= 0:
        return {}
    rows_by_trace: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_trace[int(row["trace_id"])] .append(row)
    trace_ids = sorted(rows_by_trace)
    if not trace_ids:
        return {}

    rng = np.random.default_rng(bootstrap_seed)
    stat_names = (
        "median_err_norm",
        "median_predicted_fraction",
        "median_cosine_true_pred",
        "fraction_steps_below_delta_star_050",
        "fraction_predicted_fraction_positive",
        "step_rms_over_delta_star_050",
    )
    samples: dict[str, list[float]] = {name: [] for name in stat_names}
    for _ in range(int(bootstrap_samples)):
        sampled_ids = rng.choice(np.asarray(trace_ids, dtype=np.int64), size=len(trace_ids), replace=True)
        sampled_rows: list[dict] = []
        for trace_id in sampled_ids:
            sampled_rows.extend(rows_by_trace[int(trace_id)])
        if not sampled_rows:
            continue
        summary = _condition_metric_summary(sampled_rows)
        for name in stat_names:
            samples[name].append(float(summary[name]))

    out: dict[str, float] = {}
    for name, values in samples.items():
        if not values:
            out[f"{name}_ci_low"] = float("nan")
            out[f"{name}_ci_high"] = float("nan")
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[f"{name}_ci_low"] = float(np.nanquantile(arr, 0.025))
        out[f"{name}_ci_high"] = float(np.nanquantile(arr, 0.975))
    return out


def _build_positions_for_trace(
    trace_px: np.ndarray,
    jacobian_step_px: float,
    step_indices: np.ndarray,
) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for idx in step_indices:
        p_t = trace_px[idx]
        p_t1 = trace_px[idx + 1]
        p_mid = 0.5 * (p_t + p_t1)
        positions.extend([
            p_t,
            p_t1,
            p_mid + np.array([jacobian_step_px, 0.0], dtype=np.float64),
            p_mid + np.array([-jacobian_step_px, 0.0], dtype=np.float64),
            p_mid + np.array([0.0, jacobian_step_px], dtype=np.float64),
            p_mid + np.array([0.0, -jacobian_step_px], dtype=np.float64),
        ])
    return positions


def _relative_support_index_for_step(
    runner: CurvatureScaleMatchRunner,
    support_map: dict[tuple[float, float], float],
    p_t: np.ndarray,
    p_t1: np.ndarray,
    p_mid: np.ndarray,
    jacobian_step_px: float,
) -> float:
    probes = [
        p_t,
        p_t1,
        p_mid + np.array([jacobian_step_px, 0.0], dtype=np.float64),
        p_mid + np.array([-jacobian_step_px, 0.0], dtype=np.float64),
        p_mid + np.array([0.0, jacobian_step_px], dtype=np.float64),
        p_mid + np.array([0.0, -jacobian_step_px], dtype=np.float64),
    ]
    values = [float(support_map[runner._position_key(position)]) for position in probes]
    return float(min(values))


def _build_step_rows(
    runner: CurvatureScaleMatchRunner,
    logmar: float,
    orientation: int,
    trace_id: int,
    trace_px: np.ndarray,
    responses: dict[tuple[float, float], np.ndarray],
    support_map: dict[tuple[float, float], float],
    edge_valid_threshold: float,
    min_true_norm: float,
    midpoint_delta_summary: dict[str, float | str] | None,
    step_indices: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    midpoint_delta_025 = float(midpoint_delta_summary["delta_star_025_arcmin"]) if midpoint_delta_summary else float("nan")
    midpoint_delta_050 = float(midpoint_delta_summary["delta_star_050_arcmin"]) if midpoint_delta_summary else float("nan")
    midpoint_delta_075 = float(midpoint_delta_summary["delta_star_075_arcmin"]) if midpoint_delta_summary else float("nan")

    for step_index in step_indices:
        p_t = trace_px[step_index]
        p_t1 = trace_px[step_index + 1]
        delta_p = p_t1 - p_t
        p_mid = 0.5 * (p_t + p_t1)
        key_t = runner._position_key(p_t)
        key_t1 = runner._position_key(p_t1)
        r_t = responses[key_t]
        r_t1 = responses[key_t1]
        delta_r_true = r_t1 - r_t
        J_mid = runner.finite_difference_jacobian(responses, p_mid)
        delta_r_pred = J_mid @ delta_p
        residual = delta_r_true - delta_r_pred
        true_norm = float(np.linalg.norm(delta_r_true))
        pred_norm = float(np.linalg.norm(delta_r_pred))
        residual_norm = float(np.linalg.norm(residual))
        step_norm_px = float(np.linalg.norm(delta_p))
        step_norm_arcmin = _px_to_arcmin(step_norm_px, runner.pixels_per_degree)
        relative_support_index = _relative_support_index_for_step(
            runner,
            support_map,
            p_t,
            p_t1,
            p_mid,
            runner.jacobian_step_px,
        )
        frac_residual_energy = float((residual_norm ** 2) / (true_norm ** 2 + EPS))
        predicted_fraction = float(1.0 - frac_residual_energy)
        cosine_true_pred = float(np.dot(delta_r_true, delta_r_pred) / (true_norm * pred_norm + EPS))
        if not np.isfinite(true_norm) or not np.isfinite(pred_norm):
            failure_reason = "nonfinite_response_norm"
        elif true_norm <= float(min_true_norm):
            failure_reason = "tiny_true_response"
        elif relative_support_index < float(edge_valid_threshold):
            failure_reason = "low_relative_support"
        else:
            failure_reason = ""
        valid = failure_reason == ""
        rows.append(
            {
                "dataset": "eoptotype",
                "stimulus_family": "eoptotype",
                "trace_id": int(trace_id),
                "logmar": float(logmar),
                "orientation": int(orientation),
                "step_index": int(step_index),
                "p_t_x_px": float(p_t[0]),
                "p_t_y_px": float(p_t[1]),
                "p_t1_x_px": float(p_t1[0]),
                "p_t1_y_px": float(p_t1[1]),
                "delta_x_px": float(delta_p[0]),
                "delta_y_px": float(delta_p[1]),
                "step_norm_px": step_norm_px,
                "step_norm_arcmin": step_norm_arcmin,
                "jacobian_mode": "midpoint_jacobian",
                "history_mode": "constant_history",
                "true_norm": true_norm,
                "pred_norm": pred_norm,
                "residual_norm": residual_norm,
                "err_norm": residual_norm / (true_norm + EPS),
                "frac_residual_energy": frac_residual_energy,
                "predicted_fraction": predicted_fraction,
                "cosine_true_pred": cosine_true_pred,
                "midpoint_delta_star_025_arcmin": midpoint_delta_025,
                "midpoint_delta_star_050_arcmin": midpoint_delta_050,
                "midpoint_delta_star_075_arcmin": midpoint_delta_075,
                "step_over_midpoint_delta_star_050": step_norm_arcmin / midpoint_delta_050 if np.isfinite(midpoint_delta_050) and midpoint_delta_050 > 0 else float("nan"),
                "step_below_midpoint_delta_star_050": bool(np.isfinite(midpoint_delta_050) and step_norm_arcmin <= midpoint_delta_050),
                "relative_support_index": relative_support_index,
                "min_true_norm": float(min_true_norm),
                "valid": valid,
                "failure_reason": failure_reason,
            }
        )
    return rows


def _summarize_by_bin(step_rows: list[dict], step_bin_edges_arcmin: np.ndarray) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in step_rows:
        if not row["valid"]:
            continue
        groups[(row["dataset"], row["stimulus_family"], row["logmar"], row["orientation"])].append(row)

    out_rows: list[dict] = []
    for (dataset, stimulus_family, logmar, orientation), rows in groups.items():
        step_arcmin = np.asarray([float(row["step_norm_arcmin"]) for row in rows], dtype=np.float64)
        bin_indices = np.digitize(step_arcmin, step_bin_edges_arcmin[1:], right=False)
        for bin_idx in range(len(step_bin_edges_arcmin) - 1):
            bin_rows = [row for row, idx in zip(rows, bin_indices, strict=False) if idx == bin_idx]
            if not bin_rows:
                continue
            low = float(step_bin_edges_arcmin[bin_idx])
            high = float(step_bin_edges_arcmin[bin_idx + 1])
            predicted = [float(row["predicted_fraction"]) for row in bin_rows]
            out_rows.append(
                {
                    "dataset": dataset,
                    "stimulus_family": stimulus_family,
                    "logmar": logmar,
                    "orientation": orientation,
                    "step_bin_low_arcmin": low,
                    "step_bin_high_arcmin": high,
                    "step_bin_center_arcmin": _bin_center(low, high),
                    "n_steps": len(bin_rows),
                    "bin_included_in_headline": len(bin_rows) >= DEFAULT_HEADLINE_MIN_BIN_STEPS,
                    "median_err_norm": float(np.nanmedian([float(row["err_norm"]) for row in bin_rows])),
                    "mean_err_norm": float(np.nanmean([float(row["err_norm"]) for row in bin_rows])),
                    "median_frac_residual_energy": float(np.nanmedian([float(row["frac_residual_energy"]) for row in bin_rows])),
                    "median_predicted_fraction": float(np.nanmedian(predicted)),
                    "median_cosine_true_pred": float(np.nanmedian([float(row["cosine_true_pred"]) for row in bin_rows])),
                    "fraction_predicted_fraction_positive": float(np.mean(np.asarray(predicted) > 0.0)),
                }
            )
    return out_rows


def _summarize_by_condition(
    step_rows: list[dict],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in step_rows:
        if row["valid"]:
            groups[(row["dataset"], row["stimulus_family"], row["logmar"], row["orientation"])].append(row)

    out_rows: list[dict] = []
    for group_index, ((dataset, stimulus_family, logmar, orientation), rows) in enumerate(groups.items()):
        step_arcmin = np.asarray([float(row["step_norm_arcmin"]) for row in rows], dtype=np.float64)
        midpoint_delta_025 = _nanmedian([float(row["midpoint_delta_star_025_arcmin"]) for row in rows])
        midpoint_delta_050 = _nanmedian([float(row["midpoint_delta_star_050_arcmin"]) for row in rows])
        midpoint_delta_075 = _nanmedian([float(row["midpoint_delta_star_075_arcmin"]) for row in rows])
        step_stats = _compute_step_arcmin_stats(step_arcmin)
        metric_summary = _condition_metric_summary(rows)
        row = {
            "dataset": dataset,
            "stimulus_family": stimulus_family,
            "logmar": logmar,
            "orientation": orientation,
            "n_steps": len(rows),
            "n_traces": len({int(item["trace_id"]) for item in rows}),
            "fem_step_rms_arcmin": step_stats["fem_step_rms_arcmin"],
            "fem_step_median_arcmin": step_stats["fem_step_median_arcmin"],
            "fem_step_p75_arcmin": step_stats["fem_step_p75_arcmin"],
            "fem_step_p90_arcmin": step_stats["fem_step_p90_arcmin"],
            "fem_step_p95_arcmin": step_stats["fem_step_p95_arcmin"],
            "median_err_norm": float(metric_summary["median_err_norm"]),
            "median_predicted_fraction": float(metric_summary["median_predicted_fraction"]),
            "median_cosine_true_pred": float(metric_summary["median_cosine_true_pred"]),
            "fraction_predicted_fraction_positive": float(metric_summary["fraction_predicted_fraction_positive"]),
            "n_steps_below_delta_star_050": int(metric_summary["n_steps_below_delta_star_050"]),
            "fraction_steps_below_delta_star_050": float(metric_summary["fraction_steps_below_delta_star_050"]),
            "midpoint_delta_star_025_arcmin": midpoint_delta_025,
            "midpoint_delta_star_050_arcmin": midpoint_delta_050,
            "midpoint_delta_star_075_arcmin": midpoint_delta_075,
            "step_rms_over_delta_star_050": step_stats["fem_step_rms_arcmin"] / midpoint_delta_050 if np.isfinite(midpoint_delta_050) and midpoint_delta_050 > 0 else float("nan"),
            "step_p90_over_delta_star_050": step_stats["fem_step_p90_arcmin"] / midpoint_delta_050 if np.isfinite(midpoint_delta_050) and midpoint_delta_050 > 0 else float("nan"),
            "step_p95_over_delta_star_050": step_stats["fem_step_p95_arcmin"] / midpoint_delta_050 if np.isfinite(midpoint_delta_050) and midpoint_delta_050 > 0 else float("nan"),
        }
        row.update(_bootstrap_condition_metrics(rows, bootstrap_samples, bootstrap_seed + group_index))
        row["direction_preserved"] = bool(np.isfinite(row["median_cosine_true_pred"]) and float(row["median_cosine_true_pred"]) >= 0.75)
        row["magnitude_preserved"] = bool(np.isfinite(row["median_predicted_fraction"]) and float(row["median_predicted_fraction"]) > 0.0)
        row["regime_interpretation"] = _direction_magnitude_interpretation(
            float(row["median_predicted_fraction"]),
            float(row["median_cosine_true_pred"]),
        )
        out_rows.append(row)
    return out_rows


def _summarize_step_scales(condition_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in condition_rows:
        groups[(row["dataset"], row["stimulus_family"], row["logmar"])].append(row)

    out_rows: list[dict] = []
    for (dataset, stimulus_family, logmar), rows in groups.items():
        total_steps = int(sum(int(row["n_steps"]) for row in rows))
        total_below_delta = int(sum(int(row["n_steps_below_delta_star_050"]) for row in rows))
        out_rows.append(
            {
                "dataset": dataset,
                "stimulus_family": stimulus_family,
                "logmar": logmar,
                "n_steps": total_steps,
                "median_fem_step_rms_arcmin": _nanmedian([float(row["fem_step_rms_arcmin"]) for row in rows]),
                "median_fem_step_p90_arcmin": _nanmedian([float(row["fem_step_p90_arcmin"]) for row in rows]),
                "median_fem_step_p95_arcmin": _nanmedian([float(row["fem_step_p95_arcmin"]) for row in rows]),
                "median_midpoint_delta_star_050_arcmin": _nanmedian([float(row["midpoint_delta_star_050_arcmin"]) for row in rows]),
                "median_step_rms_over_delta_star_050": _nanmedian([float(row["step_rms_over_delta_star_050"]) for row in rows]),
                "median_step_p90_over_delta_star_050": _nanmedian([float(row["step_p90_over_delta_star_050"]) for row in rows]),
                "median_step_p95_over_delta_star_050": _nanmedian([float(row["step_p95_over_delta_star_050"]) for row in rows]),
                "median_err_norm": _nanmedian([float(row["median_err_norm"]) for row in rows]),
                "median_predicted_fraction": _nanmedian([float(row["median_predicted_fraction"]) for row in rows]),
                "median_cosine_true_pred": _nanmedian([float(row["median_cosine_true_pred"]) for row in rows]),
                "n_steps_below_delta_star_050": total_below_delta,
                "fraction_steps_below_delta_star_050": float(total_below_delta / total_steps) if total_steps > 0 else float("nan"),
            }
        )
    return out_rows


def _plot_curves_by_logmar(
    bin_rows: list[dict],
    output_path: Path,
    y_key: str,
    y_label: str,
    title: str,
    hline: float | None = None,
) -> None:
    groups: dict[float, dict[float, list[tuple[float, bool]]]] = defaultdict(lambda: defaultdict(list))
    for row in bin_rows:
        groups[float(row["logmar"])][float(row["step_bin_center_arcmin"])].append(
            (float(row[y_key]), bool(row.get("bin_included_in_headline", False)))
        )
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for logmar in sorted(groups):
        xs = sorted(groups[logmar])
        ys = [float(np.nanmedian([value for value, _ in groups[logmar][x]])) for x in xs]
        include_mask = [bool(np.any([include for _, include in groups[logmar][x]])) for x in xs]
        faint_xs = [x for x, include in zip(xs, include_mask, strict=False) if not include]
        faint_ys = [y for y, include in zip(ys, include_mask, strict=False) if not include]
        main_xs = [x for x, include in zip(xs, include_mask, strict=False) if include]
        main_ys = [y for y, include in zip(ys, include_mask, strict=False) if include]
        if main_xs:
            ax.plot(main_xs, main_ys, marker="o", label=f"LogMAR {logmar:.2f}")
        if faint_xs:
            ax.scatter(faint_xs, faint_ys, color="0.7", alpha=0.5, s=20)
    if hline is not None:
        ax.axhline(hline, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Step size (arcmin)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_step_scale_vs_delta_star(condition_rows: list[dict], output_path: Path) -> None:
    rows = [row for row in condition_rows if np.isfinite(row["midpoint_delta_star_050_arcmin"]) and np.isfinite(row["fem_step_rms_arcmin"])]
    if not rows:
        return
    xs = np.asarray([float(row["midpoint_delta_star_050_arcmin"]) for row in rows], dtype=np.float64)
    ys_rms = np.asarray([float(row["fem_step_rms_arcmin"]) for row in rows], dtype=np.float64)
    ys_p90 = np.asarray([float(row["fem_step_p90_arcmin"]) for row in rows], dtype=np.float64)
    lim = float(np.nanmax(np.concatenate([xs, ys_rms, ys_p90])) * 1.1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(xs, ys_rms, label="step RMS", color="tab:blue")
    ax.scatter(xs, ys_p90, label="step p90", color="tab:orange", marker="s")
    ax.plot([0.0, lim], [0.0, lim], color="0.25", linestyle="--", linewidth=1.0)
    ax.plot([0.0, lim], [0.0, 2.0 * lim], color="0.7", linestyle=":", linewidth=1.0)
    ax.plot([0.0, lim], [0.0, 0.5 * lim], color="0.7", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel("Midpoint delta_star_050 (arcmin)")
    ax.set_ylabel("FEM step scale (arcmin)")
    ax.set_title("FEM step scale versus midpoint delta_star")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_condition_summary(condition_rows: list[dict], output_path: Path) -> None:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in condition_rows:
        groups[float(row["logmar"])].append(row)
    if not groups:
        return
    xs = sorted(groups)
    predicted = [_nanmedian([float(row["median_predicted_fraction"]) for row in groups[x]]) for x in xs]
    ratio = [_nanmedian([float(row["step_rms_over_delta_star_050"]) for row in groups[x]]) for x in xs]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(xs, predicted, marker="o", color="tab:blue")
    ax1.set_xlabel("LogMAR")
    ax1.set_ylabel("Median predicted fraction", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    ax2 = ax1.twinx()
    ax2.plot(xs, ratio, marker="s", color="tab:orange")
    ax2.set_ylabel("step RMS / midpoint delta_star_050", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    config: dict,
    condition_rows: list[dict],
    scale_rows: list[dict],
) -> None:
    pooled_summary_row = config.get("pooled_step_summary", {})
    median_fem_step_rms = _nanmedian([float(row["fem_step_rms_arcmin"]) for row in condition_rows])
    median_midpoint_delta = _nanmedian([float(row["midpoint_delta_star_050_arcmin"]) for row in condition_rows])
    median_step_ratio = _nanmedian([float(row["step_rms_over_delta_star_050"]) for row in condition_rows])
    median_step_p90_ratio = _nanmedian([float(row["step_p90_over_delta_star_050"]) for row in condition_rows])
    median_step_p95_ratio = _nanmedian([float(row["step_p95_over_delta_star_050"]) for row in condition_rows])
    median_err = _nanmedian([float(row["median_err_norm"]) for row in condition_rows])
    median_pred = _nanmedian([float(row["median_predicted_fraction"]) for row in condition_rows])
    median_cos = _nanmedian([float(row["median_cosine_true_pred"]) for row in condition_rows])
    fraction_below_delta = _nanmedian([float(row["fraction_steps_below_delta_star_050"]) for row in condition_rows])
    interpretation = _interpret_result(median_step_ratio, median_pred, median_cos, median_step_p90_ratio)
    lines = [
        "# FEM step-level midpoint-Jacobian prediction",
        "",
        "## Scope",
        f"- Stimulus set: E-optotype",
        f"- Model checkpoint: {config['model_source']}",
        f"- LogMARs: {', '.join(f'{x:.2f}' for x in config['logmars'])}",
        f"- Orientations: {', '.join(str(x) for x in config['orientations'])}",
        f"- Number of FEM traces: {config['n_traces']}",
        f"- History mode: {config['history_mode']}",
        f"- Jacobian mode: {config['jacobian_mode']}",
        f"- Eye convention: {config['eye_convention_helper']}",
        f"- Trace center mode: {config['center_mode']}",
        f"- Step-size filter: min_step_arcmin = {config['min_step_arcmin']}, max_step_arcmin = {config['max_step_arcmin']}",
        "",
        "## Main result",
        f"- Pooled individual-step median: {pooled_summary_row.get('step_median_arcmin', float('nan')):.4f} arcmin" if np.isfinite(pooled_summary_row.get("step_median_arcmin", float("nan"))) else "- Pooled individual-step median: NaN",
        f"- Pooled individual-step RMS: {pooled_summary_row.get('step_rms_arcmin', float('nan')):.4f} arcmin" if np.isfinite(pooled_summary_row.get("step_rms_arcmin", float("nan"))) else "- Pooled individual-step RMS: NaN",
        f"- Fraction of individual adjacent steps below 1 arcmin: {pooled_summary_row.get('fraction_below_1p0_arcmin', float('nan')):.4f}" if np.isfinite(pooled_summary_row.get("fraction_below_1p0_arcmin", float("nan"))) else "- Fraction of individual adjacent steps below 1 arcmin: NaN",
        f"- Fraction of individual adjacent steps below max_step_arcmin: {config['fraction_below_max_step_arcmin']:.4f}" if np.isfinite(config.get("fraction_below_max_step_arcmin", float("nan"))) else "- Fraction of individual adjacent steps below max_step_arcmin: NaN",
        f"- Median FEM step RMS: {median_fem_step_rms:.4f} arcmin" if np.isfinite(median_fem_step_rms) else "- Median FEM step RMS: NaN",
        f"- Median midpoint delta_star_050: {median_midpoint_delta:.4f} arcmin" if np.isfinite(median_midpoint_delta) else "- Median midpoint delta_star_050: NaN",
        f"- Median step_rms / delta_star_050: {median_step_ratio:.4f}" if np.isfinite(median_step_ratio) else "- Median step_rms / delta_star_050: NaN",
        f"- Median step_p95 / delta_star_050: {median_step_p95_ratio:.4f}" if np.isfinite(median_step_p95_ratio) else "- Median step_p95 / delta_star_050: NaN",
        f"- Median prediction error: {median_err:.4f}" if np.isfinite(median_err) else "- Median prediction error: NaN",
        f"- Median predicted fraction: {median_pred:.4f}" if np.isfinite(median_pred) else "- Median predicted fraction: NaN",
        f"- Median cosine: {median_cos:.4f}" if np.isfinite(median_cos) else "- Median cosine: NaN",
        f"- Median fraction of valid steps below midpoint delta_star_050: {fraction_below_delta:.4f}" if np.isfinite(fraction_below_delta) else "- Median fraction of valid steps below midpoint delta_star_050: NaN",
        "",
        "## README heuristic interpretation",
        f"- Quick heuristic only: {interpretation}",
        "- Scientific interpretation should be taken from the binned step-size curves, not this heuristic line.",
        "- Is the result stimulus-scale dependent? inspect condition_summary_by_logmar.png and step_scale_summary.csv",
        f"- Does the result hold for p75/p90 steps or only for very small steps? median step_p90 / delta_star_050 = {median_step_p90_ratio:.4f}" if np.isfinite(median_step_p90_ratio) else "- Does the result hold for p75/p90 steps or only for very small steps? NaN",
        "",
        "## Caveats",
        "- Constant-history spatial geometry only: yes",
        "- Moving-history not yet tested: yes",
        "- Edge/support artifacts: validity is gated on relative_support_index, not the old support_fraction name",
        f"- Imported midpoint delta_star source: {config['delta_star_source']}",
        f"- Imported midpoint delta_star assumed same model and rendering: {config['delta_star_assumed_same_model_and_rendering']}",
        "- Imported midpoint delta_star must come from the same model/rendering/trace convention; otherwise step/delta_star ratios are only approximate.",
        f"- Trace centering changes midpoint evaluation locations even though delta_p is unchanged: {config['center_mode']}",
        f"- Minimum true response norm for validity gating: {config['min_true_norm']}",
        "- Any excluded conditions: invalid steps are marked in step_prediction_by_step.csv",
    ]
    for warning in config.get("delta_star_metadata_warnings", []):
        lines.append(f"- delta_star metadata warning: {warning}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FEM step-level midpoint-Jacobian prediction analysis.")
    parser.add_argument("--logmars", default=",".join(f"{x:.2f}" for x in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", default=",".join(str(x) for x in DEFAULT_ORIENTATIONS))
    parser.add_argument("--step-bin-edges-arcmin", default=",".join(f"{x:g}" for x in DEFAULT_STEP_BIN_EDGES_ARCMIN))
    parser.add_argument("--pixels-per-degree", type=float, default=DEFAULT_PPD)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--jacobian-step-px", type=float, default=0.125)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--device", default=_pick_device())
    parser.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH)
    parser.add_argument("--curvature-summary-path", type=Path, default=DEFAULT_CURVATURE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-valid-threshold", type=float, default=0.95)
    parser.add_argument("--min-true-norm", type=float, default=DEFAULT_MIN_TRUE_NORM)
    parser.add_argument(
        "--center-mode",
        choices=("raw", "subtract_trace_mean", "subtract_first_sample"),
        default="subtract_trace_mean",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--step-stride", type=int, default=DEFAULT_STEP_STRIDE)
    parser.add_argument("--max-steps-per-trace", type=int, default=None)
    parser.add_argument("--step-random-seed", type=int, default=0)
    parser.add_argument("--min-step-arcmin", type=float, default=0.0)
    parser.add_argument("--max-step-arcmin", type=float, default=None)
    parser.add_argument("--write-step-histograms", action="store_true")
    parser.add_argument(
        "--step-thresholds-arcmin",
        default=",".join(f"{x:g}" for x in DEFAULT_STEP_THRESHOLDS_ARCMIN),
    )
    parser.add_argument("--step-stats-only", action="store_true")
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    step_bin_edges_arcmin = _step_bin_edges_with_overflow(_parse_csv_floats(args.step_bin_edges_arcmin))
    step_thresholds_arcmin = _parse_csv_floats(args.step_thresholds_arcmin)
    midpoint_delta_map = _load_midpoint_delta_star_map(args.curvature_summary_path)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    traces_deg, durations = _load_eye_traces(args.eye_traces_path)
    if args.max_traces is not None:
        traces_deg = traces_deg[: args.max_traces]
        durations = durations[: args.max_traces]

    diagnostic_rows, trace_infos, pooled_step_rows, pooled_summary_rows = _build_step_diagnostics(
        traces_deg=traces_deg,
        durations=durations,
        pixels_per_degree=args.pixels_per_degree,
        center_mode=args.center_mode,
        step_stride=int(args.step_stride),
        max_steps_per_trace=args.max_steps_per_trace,
        step_random_seed=int(args.step_random_seed),
        min_step_arcmin=float(args.min_step_arcmin),
        max_step_arcmin=float(args.max_step_arcmin) if args.max_step_arcmin is not None else None,
        thresholds_arcmin=step_thresholds_arcmin,
    )
    _write_step_diagnostics(output_dir, diagnostic_rows, pooled_step_rows, pooled_summary_rows)
    if args.write_step_histograms:
        _plot_pooled_step_histograms(pooled_step_rows, figures_dir)
    if args.step_stats_only:
        print(f"Saved step diagnostics to {output_dir}")
        return

    runner = CurvatureScaleMatchRunner(
        device=args.device,
        pixels_per_degree=args.pixels_per_degree,
        n_lags=args.n_lags,
        jacobian_step_px=args.jacobian_step_px,
        model_batch_size=args.model_batch_size,
        load_model=True,
    )

    base_config = {
        "pixels_per_degree": float(args.pixels_per_degree),
        "jacobian_step_px": float(args.jacobian_step_px),
        "n_lags": int(args.n_lags),
        "model_source": str(PKL_PATH),
        "eye_traces_path": str(args.eye_traces_path),
        "eye_convention_helper": "static-position trace through HiResRetina",
        "grid_sample_align_corners": False,
        "padding_mode": "background fill 127.0 gray",
    }
    (
        delta_star_metadata_match,
        delta_star_metadata_warnings,
        curvature_run_config,
        curvature_run_config_path,
    ) = _validate_delta_star_metadata(args.curvature_summary_path, base_config)

    total_positions_requested = 0
    total_positions_deduplicated = 0
    total_conditions = len(logmars) * len(orientations)
    run_start_time = time.perf_counter()
    total_sampled_steps = 0
    total_steps_before_filter = int(sum(int(row["n_adjacent_valid_steps"]) for row in diagnostic_rows))
    total_steps_after_filter = int(sum(int(row["n_steps_after_minmax_filter"]) for row in diagnostic_rows))

    step_rows: list[dict] = []
    condition_index = 0
    for logmar in logmars:
        for orientation in orientations:
            condition_index += 1
            condition_start_time = time.perf_counter()
            positions_needed: list[np.ndarray] = []
            valid_traces: list[tuple[int, np.ndarray, np.ndarray]] = []
            condition_sampled_steps = 0
            for trace_id in range(len(durations)):
                trace_info = trace_infos[int(trace_id)]
                trace_px = trace_info["trace_px"]
                step_indices = trace_info["selected_adjacent"]
                if step_indices.size == 0:
                    continue
                valid_traces.append((trace_id, trace_px, step_indices))
                condition_sampled_steps += int(step_indices.size)
                positions_needed.extend(_build_positions_for_trace(trace_px, args.jacobian_step_px, step_indices))
            requested_positions = len(positions_needed)
            print(
                f"[{condition_index}/{total_conditions}] starting logmar={float(logmar):+.2f} orientation={int(orientation)} "
                f"valid_traces={len(valid_traces)} sampled_steps={condition_sampled_steps} requested_positions={requested_positions}",
                flush=True,
            )
            total_positions_requested += requested_positions
            total_sampled_steps += condition_sampled_steps
            positions_needed, duplicate_count = _deduplicate_positions(runner, positions_needed)
            total_positions_deduplicated += int(duplicate_count)
            responses, support_map = runner.evaluate_condition(float(logmar), int(orientation), positions_needed)
            condition_step_rows: list[dict] = []
            for trace_id, trace_px, step_indices in valid_traces:
                midpoint_delta_summary = midpoint_delta_map.get((round(float(logmar), 2), int(orientation)))
                condition_step_rows.extend(
                    _build_step_rows(
                        runner=runner,
                        logmar=float(logmar),
                        orientation=int(orientation),
                        trace_id=int(trace_id),
                        trace_px=trace_px,
                        responses=responses,
                        support_map=support_map,
                        edge_valid_threshold=float(args.edge_valid_threshold),
                        min_true_norm=float(args.min_true_norm),
                        midpoint_delta_summary=midpoint_delta_summary,
                        step_indices=step_indices,
                    )
                )
            step_rows.extend(condition_step_rows)
            elapsed = time.perf_counter() - condition_start_time
            total_elapsed = time.perf_counter() - run_start_time
            valid_steps = sum(1 for row in condition_step_rows if row["valid"])
            print(
                f"[{condition_index}/{total_conditions}] completed logmar={float(logmar):+.2f} orientation={int(orientation)} "
                f"unique_positions={len(positions_needed)} deduplicated={int(duplicate_count)} "
                f"sampled_steps={condition_sampled_steps} steps={len(condition_step_rows)} valid_steps={valid_steps} "
                f"elapsed={_format_duration(elapsed)} total_elapsed={_format_duration(total_elapsed)}",
                flush=True,
            )

    step_bin_rows = _summarize_by_bin(step_rows, step_bin_edges_arcmin)
    condition_rows = _summarize_by_condition(
        step_rows,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    scale_rows = _summarize_step_scales(condition_rows)

    _write_csv(output_dir / "step_prediction_by_step.csv", step_rows)
    _write_csv(output_dir / "step_prediction_by_bin.csv", step_bin_rows)
    _write_csv(output_dir / "step_prediction_by_condition.csv", condition_rows)
    _write_csv(output_dir / "step_scale_summary.csv", scale_rows)

    config = {
        "stimulus_set": "eoptotype",
        "model_source": str(PKL_PATH),
        "logmars": logmars,
        "orientations": orientations,
        "n_traces": int(len(traces_deg)),
        "history_mode": "constant_history",
        "jacobian_mode": "midpoint_jacobian",
        "eye_traces_path": str(args.eye_traces_path),
        "curvature_summary_path": str(args.curvature_summary_path),
        "delta_star_source": str(args.curvature_summary_path),
        "delta_star_run_config_path": str(curvature_run_config_path) if curvature_run_config_path is not None else "",
        "delta_star_assumed_same_model_and_rendering": bool(delta_star_metadata_match),
        "delta_star_metadata_warnings": delta_star_metadata_warnings,
        "eye_convention_helper": "static-position trace through HiResRetina",
        "pixels_per_degree": float(args.pixels_per_degree),
        "jacobian_step_px": float(args.jacobian_step_px),
        "n_lags": int(args.n_lags),
        "grid_sample_align_corners": False,
        "padding_mode": "background fill 127.0 gray",
        "retina_ppd": float(args.pixels_per_degree),
        "model_input_size": [int(runner.retina.retina_h), int(runner.retina.retina_w)],
        "edge_valid_threshold": float(args.edge_valid_threshold),
        "min_true_norm": float(args.min_true_norm),
        "center_mode": args.center_mode,
        "support_metric_name": "relative_support_index",
        "step_bin_edges_arcmin": step_bin_edges_arcmin.tolist(),
        "positions_requested": int(total_positions_requested),
        "positions_deduplicated_before_evaluate_condition": int(total_positions_deduplicated),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "step_stride": int(args.step_stride),
        "step_thresholds_arcmin": list(step_thresholds_arcmin),
        "max_steps_per_trace": int(args.max_steps_per_trace) if args.max_steps_per_trace is not None else None,
        "step_random_seed": int(args.step_random_seed),
        "min_step_arcmin": float(args.min_step_arcmin),
        "max_step_arcmin": float(args.max_step_arcmin) if args.max_step_arcmin is not None else None,
        "write_step_histograms": bool(args.write_step_histograms),
        "step_stats_only": bool(args.step_stats_only),
        "sampled_steps_total": int(total_sampled_steps),
        "n_adjacent_valid_steps_before_filter": int(total_steps_before_filter),
        "n_steps_after_step_size_filter": int(total_steps_after_filter),
        "n_steps_selected_for_model_eval": int(total_sampled_steps),
        "step_selection_semantics": "every_Nth_adjacent_valid_pair_not_N_frame_displacement",
        "pooled_step_summary": pooled_summary_rows[0] if pooled_summary_rows else {},
    }
    if pooled_summary_rows and args.max_step_arcmin is not None:
        pooled_values = np.asarray([float(row["step_norm_arcmin"]) for row in pooled_step_rows], dtype=np.float64)
        config["fraction_below_max_step_arcmin"] = float(np.mean(pooled_values <= float(args.max_step_arcmin))) if pooled_values.size else float("nan")
    else:
        config["fraction_below_max_step_arcmin"] = float("nan")
    if curvature_run_config is not None:
        config["delta_star_source_run_config"] = curvature_run_config
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    if not args.skip_figures:
        _plot_curves_by_logmar(
            step_bin_rows,
            figures_dir / "step_error_vs_step_size.png",
            y_key="median_err_norm",
            y_label="Median err_norm",
            title="Step error versus step size",
            hline=0.5,
        )
        _plot_curves_by_logmar(
            step_bin_rows,
            figures_dir / "step_cosine_vs_step_size.png",
            y_key="median_cosine_true_pred",
            y_label="Median cosine_true_pred",
            title="Step cosine versus step size",
        )
        _plot_curves_by_logmar(
            step_bin_rows,
            figures_dir / "step_predicted_fraction_vs_step_size.png",
            y_key="median_predicted_fraction",
            y_label="Median predicted_fraction",
            title="Predicted fraction versus step size",
            hline=0.0,
        )
        _plot_step_scale_vs_delta_star(condition_rows, figures_dir / "step_scale_vs_midpoint_delta_star.png")
        _plot_condition_summary(condition_rows, figures_dir / "condition_summary_by_logmar.png")

    _write_readme(output_dir, config, condition_rows, scale_rows)
    print(f"Saved FEM step-level Jacobian prediction outputs to {output_dir}")


if __name__ == "__main__":
    main()
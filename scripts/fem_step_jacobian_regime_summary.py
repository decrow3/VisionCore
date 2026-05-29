#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import STATS_DIR


DEFAULT_OUTPUT_DIR = STATS_DIR / "fem_step_jacobian_regime_summary"
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_HEADLINE_MIN_BIN_STEPS = 5
DEFAULT_RUN_SPECS = (
    "drift_lte_1p0=outputs/stats/fem_step_jacobian_prediction_drift_only_1arcmin",
    "intermediate_lte_1p5=outputs/stats/fem_step_jacobian_prediction_intermediate_1p5arcmin",
    "all_steps=outputs/stats/fem_step_jacobian_prediction_all_steps_10traces",
    "large_gte_2p0=outputs/stats/fem_step_jacobian_prediction_large_steps_gt2arcmin",
    "drift_lte_1p0_or90=outputs/stats/fem_step_jacobian_prediction_drift_only_1arcmin_or90",
)

REGIME_ORDER = {
    "drift_lte_1p0": 1,
    "intermediate_lte_1p5": 2,
    "all_steps": 3,
    "large_gte_2p0": 4,
}

STEP_FILTER_LABELS = {
    "drift_lte_1p0": "step <= 1.0 arcmin",
    "intermediate_lte_1p5": "step <= 1.5 arcmin",
    "all_steps": "all adjacent valid steps",
    "large_gte_2p0": "step >= 2.0 arcmin",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "NaN", "nan", "None"):
        return float("nan")
    return float(value)


def _to_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in ("", "NaN", "nan", "None"):
        return 0
    return int(float(value))


def _parse_run_specs(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Run spec must be label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        out.append((label.strip(), Path(raw_path.strip())))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanquantile(values: list[float], quantile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanquantile(arr, quantile))


def _safe_rms(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(values * values)))


def _direction_magnitude_interpretation(median_predicted_fraction: float, median_cosine: float) -> str:
    if np.isfinite(median_cosine) and median_cosine >= 0.75 and np.isfinite(median_predicted_fraction) and median_predicted_fraction > 0.0:
        return "direction_and_magnitude_predicted"
    if np.isfinite(median_cosine) and median_cosine >= 0.75:
        return "direction_preserved_magnitude_failed"
    return "local_tangent_failed"


def _condition_label(run_label: str, row: dict[str, str]) -> str:
    return f"{run_label} | logmar={float(row['logmar']):+.2f} | orientation={int(float(row['orientation']))}"


def _build_condition_rows(run_specs: list[tuple[str, Path]]) -> list[dict]:
    out: list[dict] = []
    for run_label, run_dir in run_specs:
        condition_rows = _read_csv_rows(run_dir / "step_prediction_by_condition.csv")
        for row in condition_rows:
            out.append(
                {
                    "run_label": run_label,
                    "condition": _condition_label(run_label, row),
                    "run_dir": str(run_dir),
                    "logmar": float(row["logmar"]),
                    "orientation": int(float(row["orientation"])),
                    "n_steps": _to_int(row, "n_steps"),
                    "step_rms_arcmin": _to_float(row, "fem_step_rms_arcmin"),
                    "step_median_arcmin": _to_float(row, "fem_step_median_arcmin"),
                    "step_p90_arcmin": _to_float(row, "fem_step_p90_arcmin"),
                    "step_rms_over_delta_star_050": _to_float(row, "step_rms_over_delta_star_050"),
                    "fraction_below_delta_star_050": _to_float(row, "fraction_steps_below_delta_star_050"),
                    "median_predicted_fraction": _to_float(row, "median_predicted_fraction"),
                    "median_predicted_fraction_ci_low": _to_float(row, "median_predicted_fraction_ci_low"),
                    "median_predicted_fraction_ci_high": _to_float(row, "median_predicted_fraction_ci_high"),
                    "median_cosine_true_pred": _to_float(row, "median_cosine_true_pred"),
                    "median_cosine_true_pred_ci_low": _to_float(row, "median_cosine_true_pred_ci_low"),
                    "median_cosine_true_pred_ci_high": _to_float(row, "median_cosine_true_pred_ci_high"),
                    "median_err_norm": _to_float(row, "median_err_norm"),
                    "median_err_norm_ci_low": _to_float(row, "median_err_norm_ci_low"),
                    "median_err_norm_ci_high": _to_float(row, "median_err_norm_ci_high"),
                    "fraction_predicted_fraction_positive": _to_float(row, "fraction_predicted_fraction_positive"),
                    "fraction_predicted_fraction_positive_ci_low": _to_float(row, "fraction_predicted_fraction_positive_ci_low"),
                    "fraction_predicted_fraction_positive_ci_high": _to_float(row, "fraction_predicted_fraction_positive_ci_high"),
                    "fraction_steps_below_delta_star_050_ci_low": _to_float(row, "fraction_steps_below_delta_star_050_ci_low"),
                    "fraction_steps_below_delta_star_050_ci_high": _to_float(row, "fraction_steps_below_delta_star_050_ci_high"),
                    "step_rms_over_delta_star_050_ci_low": _to_float(row, "step_rms_over_delta_star_050_ci_low"),
                    "step_rms_over_delta_star_050_ci_high": _to_float(row, "step_rms_over_delta_star_050_ci_high"),
                    "n_traces": _to_int(row, "n_traces"),
                    "direction_preserved": row.get("direction_preserved", "False") == "True",
                    "magnitude_preserved": row.get("magnitude_preserved", "False") == "True",
                    "regime_interpretation": row.get("regime_interpretation", _direction_magnitude_interpretation(_to_float(row, "median_predicted_fraction"), _to_float(row, "median_cosine_true_pred"))),
                }
            )
    return out


def _sum_ints(values: list[int]) -> int:
    return int(sum(int(v) for v in values))


def _read_run_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _metric_summary_from_step_rows(rows: list[dict]) -> dict[str, float | int]:
    # Group by condition so bootstrap CIs are computed on the same statistic
    # as the point estimate (median of condition-level medians, not pooled-step median).
    by_condition: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (float(row["logmar"]), int(float(row["orientation"])))
        by_condition[key].append(row)

    cond_predicted: list[float] = []
    cond_cosine: list[float] = []
    cond_err: list[float] = []
    cond_step_rms: list[float] = []
    cond_step_median: list[float] = []
    cond_step_p90: list[float] = []
    cond_frac_below: list[float] = []
    cond_frac_positive: list[float] = []
    cond_rms_over: list[float] = []
    total_steps = 0

    for cond_rows in by_condition.values():
        total_steps += len(cond_rows)
        predicted = np.asarray([float(r["predicted_fraction"]) for r in cond_rows], dtype=np.float64)
        cosine = np.asarray([float(r["cosine_true_pred"]) for r in cond_rows], dtype=np.float64)
        err = np.asarray([float(r["err_norm"]) for r in cond_rows], dtype=np.float64)
        step = np.asarray([float(r["step_norm_arcmin"]) for r in cond_rows], dtype=np.float64)
        below = np.asarray([str(r["step_below_midpoint_delta_star_050"]).lower() == "true" for r in cond_rows], dtype=bool)
        over = np.asarray([float(r["step_over_midpoint_delta_star_050"]) for r in cond_rows], dtype=np.float64)
        cond_predicted.append(float(np.nanmedian(predicted)))
        cond_cosine.append(float(np.nanmedian(cosine)))
        cond_err.append(float(np.nanmedian(err)))
        cond_step_rms.append(_safe_rms(step))
        cond_step_median.append(float(np.nanmedian(step)))
        cond_step_p90.append(float(np.nanquantile(step, 0.90)) if step.size else float("nan"))
        cond_frac_below.append(float(np.mean(below)) if below.size else float("nan"))
        cond_frac_positive.append(float(np.mean(predicted > 0.0)) if predicted.size else float("nan"))
        cond_rms_over.append(_safe_rms(over))

    n_traces = len({
        (int(float(r["logmar"]) * 100), int(float(r["orientation"])), int(float(r["trace_id"])))
        for r in rows
    })
    return {
        "n_steps": total_steps,
        "n_traces": n_traces,
        "median_step_rms_arcmin": _nanmedian(cond_step_rms),
        "median_fraction_below_delta_star_050": _nanmedian(cond_frac_below),
        "median_predicted_fraction": _nanmedian(cond_predicted),
        "median_cosine_true_pred": _nanmedian(cond_cosine),
        "median_err_norm": _nanmedian(cond_err),
        "fraction_predicted_fraction_positive": _nanmedian(cond_frac_positive),
        "step_rms_over_delta_star_050": _nanmedian(cond_rms_over),
        "step_median_arcmin": _nanmedian(cond_step_median),
        "step_p90_arcmin": _nanmedian(cond_step_p90),
    }


def _bootstrap_regime_metrics(step_rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> dict[str, float]:
    if bootstrap_samples <= 0 or not step_rows:
        return {}

    rows_by_condition_trace: dict[tuple[float, int, int], list[dict]] = defaultdict(list)
    trace_ids_by_condition: dict[tuple[float, int], list[int]] = defaultdict(list)
    for row in step_rows:
        condition_key = (float(row["logmar"]), int(float(row["orientation"])))
        trace_id = int(float(row["trace_id"]))
        rows_by_condition_trace[(condition_key[0], condition_key[1], trace_id)].append(row)
    for logmar, orientation, trace_id in rows_by_condition_trace:
        trace_ids_by_condition[(logmar, orientation)].append(trace_id)

    rng = np.random.default_rng(int(bootstrap_seed))
    metric_names = (
        "median_predicted_fraction",
        "median_cosine_true_pred",
        "median_err_norm",
        "fraction_predicted_fraction_positive",
        "median_fraction_below_delta_star_050",
        "step_rms_over_delta_star_050",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(int(bootstrap_samples)):
        sampled_rows: list[dict] = []
        for condition_key, trace_ids in trace_ids_by_condition.items():
            ids = np.asarray(trace_ids, dtype=np.int64)
            sampled = rng.choice(ids, size=ids.size, replace=True)
            for trace_id in sampled:
                sampled_rows.extend(rows_by_condition_trace[(condition_key[0], condition_key[1], int(trace_id))])
        summary = _metric_summary_from_step_rows(sampled_rows)
        for name in metric_names:
            samples[name].append(float(summary[name]))

    out: dict[str, float] = {}
    for name, values in samples.items():
        out[f"{name}_ci_low"] = _nanquantile(values, 0.025)
        out[f"{name}_ci_high"] = _nanquantile(values, 0.975)
    return out


def _build_regime_rows(condition_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in condition_rows:
        grouped[str(row["run_label"])].append(row)

    out: list[dict] = []
    for run_label in sorted(grouped):
        rows = grouped[run_label]
        out.append(
            {
                "condition": run_label,
                "n_conditions": len(rows),
                "n_traces": _sum_ints([int(row["n_traces"]) for row in rows]),
                "n_steps": _sum_ints([int(row["n_steps"]) for row in rows]),
                "step_rms_arcmin": _nanmedian([float(row["step_rms_arcmin"]) for row in rows]),
                "step_median_arcmin": _nanmedian([float(row["step_median_arcmin"]) for row in rows]),
                "step_p90_arcmin": _nanmedian([float(row["step_p90_arcmin"]) for row in rows]),
                "step_rms_over_delta_star_050": _nanmedian([float(row["step_rms_over_delta_star_050"]) for row in rows]),
                "fraction_below_delta_star_050": _nanmedian([float(row["fraction_below_delta_star_050"]) for row in rows]),
                "median_predicted_fraction": _nanmedian([float(row["median_predicted_fraction"]) for row in rows]),
                "median_cosine_true_pred": _nanmedian([float(row["median_cosine_true_pred"]) for row in rows]),
                "median_err_norm": _nanmedian([float(row["median_err_norm"]) for row in rows]),
            }
        )
    return out


def _build_bootstrap_rows(condition_rows: list[dict], run_specs: list[tuple[str, Path]], bootstrap_samples: int, bootstrap_seed: int) -> tuple[list[dict], list[dict]]:
    run_dir_by_label = {label: path for label, path in run_specs}
    condition_bootstrap_rows = [dict(row) for row in condition_rows]

    regime_bootstrap_rows: list[dict] = []
    for run_label, run_dir in run_specs:
        step_rows = [row for row in _read_csv_rows(run_dir / "step_prediction_by_step.csv") if row.get("valid", "False") == "True"]
        row = next((item for item in _build_regime_rows(condition_rows) if item["condition"] == run_label), None)
        if row is None:
            continue
        row = dict(row)
        row.update(_bootstrap_regime_metrics(step_rows, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed + REGIME_ORDER.get(run_label, 100)))
        row["direction_preserved"] = bool(np.isfinite(row["median_cosine_true_pred"]) and float(row["median_cosine_true_pred"]) >= 0.75)
        row["magnitude_preserved"] = bool(np.isfinite(row["median_predicted_fraction"]) and float(row["median_predicted_fraction"]) > 0.0)
        row["regime_interpretation"] = _direction_magnitude_interpretation(float(row["median_predicted_fraction"]), float(row["median_cosine_true_pred"]))
        regime_bootstrap_rows.append(row)

    for row in condition_bootstrap_rows:
        row["direction_preserved"] = bool(row["direction_preserved"])
        row["magnitude_preserved"] = bool(row["magnitude_preserved"])
        row["regime_order"] = REGIME_ORDER.get(str(row["run_label"]), 99)
        row["step_filter_label"] = STEP_FILTER_LABELS.get(str(row["run_label"]), str(row["run_label"]))
        row["run_config_path"] = str(run_dir_by_label[str(row["run_label"])] / "run_config.json")

    for row in regime_bootstrap_rows:
        row["regime_order"] = REGIME_ORDER.get(str(row["condition"]), 99)
        row["step_filter_label"] = STEP_FILTER_LABELS.get(str(row["condition"]), str(row["condition"]))

    regime_bootstrap_rows.sort(key=lambda row: (int(row["regime_order"]), str(row["condition"])))
    condition_bootstrap_rows.sort(key=lambda row: (int(row["regime_order"]), float(row["logmar"]), int(row["orientation"])))
    return condition_bootstrap_rows, regime_bootstrap_rows


def _build_bin_rows(run_specs: list[tuple[str, Path]]) -> list[dict]:
    out: list[dict] = []
    for run_label, run_dir in run_specs:
        for row in _read_csv_rows(run_dir / "step_prediction_by_bin.csv"):
            out.append(
                {
                    "run_label": run_label,
                    "run_dir": str(run_dir),
                    "logmar": float(row["logmar"]),
                    "orientation": int(float(row["orientation"])),
                    "step_bin_low_arcmin": _to_float(row, "step_bin_low_arcmin"),
                    "step_bin_high_arcmin": _to_float(row, "step_bin_high_arcmin"),
                    "step_bin_center_arcmin": _to_float(row, "step_bin_center_arcmin"),
                    "n_steps": _to_int(row, "n_steps"),
                    "bin_included_in_headline": row.get("bin_included_in_headline", "False") == "True",
                    "median_predicted_fraction": _to_float(row, "median_predicted_fraction"),
                    "median_cosine_true_pred": _to_float(row, "median_cosine_true_pred"),
                    "median_err_norm": _to_float(row, "median_err_norm"),
                }
            )
    return out


def _plot_regime_bin_metric(bin_rows: list[dict], output_path: Path, y_key: str, y_label: str, hline: float | None = None) -> None:
    groups: dict[str, dict[float, list[tuple[float, bool]]]] = defaultdict(lambda: defaultdict(list))
    for row in bin_rows:
        groups[str(row["run_label"])][float(row["step_bin_center_arcmin"])].append(
            (float(row[y_key]), bool(row.get("bin_included_in_headline", False)))
        )

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for run_label in sorted(groups):
        xs = sorted(groups[run_label])
        ys = [float(np.nanmedian([value for value, _ in groups[run_label][x]])) for x in xs]
        include_mask = [bool(np.any([include for _, include in groups[run_label][x]])) for x in xs]
        main_xs = [x for x, include in zip(xs, include_mask, strict=False) if include]
        main_ys = [y for y, include in zip(ys, include_mask, strict=False) if include]
        faint_xs = [x for x, include in zip(xs, include_mask, strict=False) if not include]
        faint_ys = [y for y, include in zip(ys, include_mask, strict=False) if not include]
        if main_xs:
            ax.plot(main_xs, main_ys, marker="o", label=run_label)
        if faint_xs:
            ax.scatter(faint_xs, faint_ys, color="0.7", alpha=0.5, s=20)
    if hline is not None:
        ax.axhline(hline, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Step size bin center (arcmin)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} versus step size bin")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_regime_summary_errorbar(
    regime_rows: list[dict],
    output_path: Path,
    y_key: str,
    low_key: str,
    high_key: str,
    y_label: str,
    hline: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if not regime_rows:
        return
    ordered = sorted(regime_rows, key=lambda row: int(row["regime_order"]))
    xs = np.arange(len(ordered), dtype=np.float64)
    ys = np.asarray([float(row[y_key]) for row in ordered], dtype=np.float64)
    lows = np.asarray([float(row.get(low_key, float("nan"))) for row in ordered], dtype=np.float64)
    highs = np.asarray([float(row.get(high_key, float("nan"))) for row in ordered], dtype=np.float64)
    yerr = np.vstack([np.maximum(0.0, ys - lows), np.maximum(0.0, highs - ys)])
    yerr[~np.isfinite(yerr)] = 0.0
    labels = [str(row["condition"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4, color="tab:blue")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Regime")
    if hline is not None:
        ax.axhline(hline, color="0.4", linestyle="--", linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
        # Annotate points outside the clipped range
        for x, y in zip(xs, ys, strict=False):
            if np.isfinite(y) and y < ylim[0]:
                ax.annotate(f"{y:.0f}", xy=(x, ylim[0]), xytext=(x, ylim[0] + 0.05 * (ylim[1] - ylim[0])),
                            ha="center", va="bottom", fontsize=8, color="0.4",
                            arrowprops={"arrowstyle": "-", "color": "0.4"})
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_regime_ratio(regime_rows: list[dict], output_path: Path) -> None:
    if not regime_rows:
        return
    ordered = sorted(regime_rows, key=lambda row: int(row["regime_order"]))
    xs = np.arange(len(ordered), dtype=np.float64)
    ys = np.asarray([float(row["step_rms_over_delta_star_050"]) for row in ordered], dtype=np.float64)
    labels = [str(row["condition"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(xs, ys, color="tab:orange", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("step RMS / midpoint delta_star_050")
    ax.set_xlabel("Regime")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_figure_panel_rows(regime_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in sorted(regime_rows, key=lambda item: int(item["regime_order"])):
        regime_label = str(row["condition"])
        out.append(
            {
                "regime_order": int(row["regime_order"]),
                "regime_label": regime_label,
                "step_filter_label": STEP_FILTER_LABELS.get(regime_label, regime_label),
                "n_conditions": int(row["n_conditions"]),
                "n_traces": int(row["n_traces"]),
                "n_steps": int(row["n_steps"]),
                "median_step_rms_arcmin": float(row["step_rms_arcmin"]),
                "median_fraction_below_delta_star_050": float(row["fraction_below_delta_star_050"]),
                "median_predicted_fraction": float(row["median_predicted_fraction"]),
                "predicted_fraction_ci_low": float(row.get("median_predicted_fraction_ci_low", float("nan"))),
                "predicted_fraction_ci_high": float(row.get("median_predicted_fraction_ci_high", float("nan"))),
                "median_cosine_true_pred": float(row["median_cosine_true_pred"]),
                "cosine_ci_low": float(row.get("median_cosine_true_pred_ci_low", float("nan"))),
                "cosine_ci_high": float(row.get("median_cosine_true_pred_ci_high", float("nan"))),
                "median_err_norm": float(row["median_err_norm"]),
                "err_norm_ci_low": float(row.get("median_err_norm_ci_low", float("nan"))),
                "err_norm_ci_high": float(row.get("median_err_norm_ci_high", float("nan"))),
                "regime_interpretation": str(row["regime_interpretation"]),
            }
        )
    return out


def _write_readme(output_dir: Path, condition_rows: list[dict], regime_rows: list[dict], run_specs: list[tuple[str, Path]]) -> None:
    lines = [
        "# FEM step-Jacobian regime summary",
        "",
        "## Included runs",
    ]
    for label, path in run_specs:
        lines.append(f"- {label}: {path}")
    lines.extend(
        [
            "",
            "## Outputs",
            "- comparison_summary_by_condition.csv: requested per-condition table",
            "- comparison_summary_by_regime.csv: median-across-condition regime table",
            "- comparison_summary_by_condition_bootstrap.csv: manuscript-facing condition table with trace-bootstrap CIs",
            "- comparison_summary_by_regime_bootstrap.csv: manuscript-facing regime table with trace-bootstrap CIs",
            "- figure_panel_regime_summary.csv: compact figure-panel export",
            "- comparison_step_prediction_by_bin.csv: merged step-bin summaries",
            "- figures/regime_predicted_fraction_vs_step_size_bin.png: regime comparison curve",
            "- figures/regime_cosine_vs_step_size_bin.png: regime comparison curve",
            "- figures/regime_predicted_fraction_summary.png: regime-level predicted fraction with CIs",
            "- figures/regime_cosine_summary.png: regime-level cosine with CIs",
            "- figures/regime_step_rms_over_delta_star.png: regime-level step RMS / delta_star",
            "",
            f"## Counts",
            f"- Number of condition rows: {len(condition_rows)}",
            f"- Number of regime rows: {len(regime_rows)}",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FEM step-Jacobian regime runs into comparison tables and figures.")
    parser.add_argument("--run", action="append", default=None, help="Run spec as label=path. Repeat for multiple runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = _parse_run_specs(args.run if args.run else list(DEFAULT_RUN_SPECS))
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = _build_condition_rows(run_specs)
    condition_bootstrap_rows, regime_bootstrap_rows = _build_bootstrap_rows(
        condition_rows,
        run_specs,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    regime_rows = _build_regime_rows(condition_rows)
    bin_rows = _build_bin_rows(run_specs)
    figure_panel_rows = _build_figure_panel_rows(regime_bootstrap_rows)

    _write_csv(output_dir / "comparison_summary_by_condition.csv", condition_rows)
    _write_csv(output_dir / "comparison_summary_by_regime.csv", regime_rows)
    _write_csv(output_dir / "comparison_summary_by_condition_bootstrap.csv", condition_bootstrap_rows)
    _write_csv(output_dir / "comparison_summary_by_regime_bootstrap.csv", regime_bootstrap_rows)
    _write_csv(output_dir / "comparison_step_prediction_by_bin.csv", bin_rows)
    _write_csv(output_dir / "figure_panel_regime_summary.csv", figure_panel_rows)

    _plot_regime_bin_metric(
        bin_rows,
        figures_dir / "regime_predicted_fraction_vs_step_size_bin.png",
        y_key="median_predicted_fraction",
        y_label="Median predicted fraction",
        hline=0.0,
    )
    _plot_regime_bin_metric(
        bin_rows,
        figures_dir / "regime_cosine_vs_step_size_bin.png",
        y_key="median_cosine_true_pred",
        y_label="Median cosine_true_pred",
    )
    _plot_regime_summary_errorbar(
        regime_bootstrap_rows,
        figures_dir / "regime_predicted_fraction_summary.png",
        y_key="median_predicted_fraction",
        low_key="median_predicted_fraction_ci_low",
        high_key="median_predicted_fraction_ci_high",
        y_label="Median predicted_fraction",
        hline=0.0,
    )
    _plot_regime_summary_errorbar(
        regime_bootstrap_rows,
        figures_dir / "regime_cosine_summary.png",
        y_key="median_cosine_true_pred",
        low_key="median_cosine_true_pred_ci_low",
        high_key="median_cosine_true_pred_ci_high",
        y_label="Median cosine_true_pred",
    )
    _plot_regime_summary_errorbar(
        regime_bootstrap_rows,
        figures_dir / "regime_predicted_fraction_summary_clipped.png",
        y_key="median_predicted_fraction",
        low_key="median_predicted_fraction_ci_low",
        high_key="median_predicted_fraction_ci_high",
        y_label="Median predicted_fraction",
        hline=0.0,
        ylim=(-2.0, 1.0),
    )
    _plot_regime_ratio(regime_bootstrap_rows, figures_dir / "regime_step_rms_over_delta_star.png")
    _write_readme(output_dir, condition_bootstrap_rows, regime_bootstrap_rows, run_specs)
    print(f"Saved FEM step-Jacobian regime summary to {output_dir}")


if __name__ == "__main__":
    main()
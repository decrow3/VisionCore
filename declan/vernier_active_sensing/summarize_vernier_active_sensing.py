#!/usr/bin/env python3
"""Summarize and plot Vernier active-sensing first-pass outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import expected_counts, poisson_fisher_counts, pose_blind_diagonal_fisher


DEFAULT_RUN_DIR = Path("outputs") / "vernier_active_sensing_first_pass"
PLOT_CONDITIONS = (
    "static_center",
    "static_phase_cloud_matched_positions",
    "real_fem",
    "scaled_real_0.5",
)
SCALE_CONDITIONS = (
    ("static_center", 0.0, "static center"),
    ("scaled_real_0.5", 0.5, "0.5x"),
    ("real_fem", 1.0, "1.0x real"),
    ("scaled_real_1.5", 1.5, "1.5x"),
)
SCALE_FAMILY_ORDER = (
    "scaled real",
    "matched phase cloud",
    "order shuffled",
    "static center",
)
THRESHOLD_CONTRASTS = (
    ("real_fem", "static_center"),
    ("real_fem", "static_phase_cloud_matched_positions"),
    ("scaled_real_0.5", "static_phase_cloud_matched_positions"),
    ("scaled_real_1.5", "static_phase_cloud_matched_positions"),
)
COLORS = {
    "static_center": "#555555",
    "static_repeated_phase": "#8c6d31",
    "static_phase_cloud_matched_positions": "#1f77b4",
    "real_fem": "#d62728",
    "order_shuffled_positions": "#9467bd",
    "axis_horizontal": "#2ca02c",
    "axis_vertical": "#ff7f0e",
    "scaled_real_0.5": "#17becf",
    "scaled_real_1.5": "#bcbd22",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_rate_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        condition = str(npz["condition"][0])
        fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
        inference_mode = str(npz["inference_mode"][0])
        plus = np.asarray(npz["plus"], dtype=np.float32)
        minus = np.asarray(npz["minus"], dtype=np.float32)
        lengths = np.asarray(npz["lengths"], dtype=np.int32)
    plus_trials = [plus[i, : int(lengths[i])] for i in range(plus.shape[0])]
    minus_trials = [minus[i, : int(lengths[i])] for i in range(minus.shape[0])]
    return {
        "path": path,
        "condition": condition,
        "fd_step_arcmin": fd_step,
        "inference_mode": inference_mode,
        "plus_trials": plus_trials,
        "minus_trials": minus_trials,
    }


def compute_curve_tables(run_dir: Path, *, bin_seconds: float, phi: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pose_aware_rows: list[dict[str, Any]] = []
    pose_blind_rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "cache").glob("rates_*_fd*arcmin.npz")):
        item = load_rate_cache(path)
        condition = item["condition"]
        fd_step = float(item["fd_step_arcmin"])
        inference_mode = item["inference_mode"]
        curves = []
        for trace_index, (plus, minus) in enumerate(zip(item["plus_trials"], item["minus_trials"], strict=True)):
            t = min(plus.shape[0], minus.shape[0])
            info = poisson_fisher_counts(
                expected_counts(plus[:t], bin_seconds),
                expected_counts(minus[:t], bin_seconds),
                step_arcmin=fd_step,
                phi=phi,
            )
            curves.append(info.cumulative_fisher)
            for timebin, value in enumerate(info.cumulative_fisher):
                pose_aware_rows.append(
                    {
                        "readout": "pose_aware_diagonal_poisson",
                        "condition": condition,
                        "fd_step_arcmin": fd_step,
                        "inference_mode": inference_mode,
                        "trace_index": trace_index,
                        "timebin": timebin,
                        "cumulative_fisher": float(value),
                    }
                )
        if curves:
            t_min = min(curve.shape[0] for curve in curves)
            arr = np.stack([curve[:t_min] for curve in curves], axis=0)
            for timebin in range(t_min):
                vals = arr[:, timebin]
                pose_aware_rows.append(
                    {
                        "readout": "pose_aware_diagonal_poisson_mean",
                        "condition": condition,
                        "fd_step_arcmin": fd_step,
                        "inference_mode": inference_mode,
                        "trace_index": "mean",
                        "timebin": timebin,
                        "cumulative_fisher": float(np.mean(vals)),
                        "sem": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                        "p25": float(np.percentile(vals, 25)),
                        "p75": float(np.percentile(vals, 75)),
                    }
                )
        if len(item["plus_trials"]) >= 2:
            blind = pose_blind_diagonal_fisher(
                item["plus_trials"],
                item["minus_trials"],
                step_arcmin=fd_step,
                bin_seconds=bin_seconds,
                phi=phi,
            )
            for timebin, value in enumerate(blind["cumulative_fisher"]):
                pose_blind_rows.append(
                    {
                        "readout": "pose_blind_diagonal_count_plus_marginal",
                        "condition": condition,
                        "fd_step_arcmin": fd_step,
                        "inference_mode": inference_mode,
                        "timebin": timebin,
                        "cumulative_fisher": float(value),
                    }
                )
    return pose_aware_rows, pose_blind_rows


def row_lookup(rows: list[dict[str, str]], *keys: str) -> dict[tuple[Any, ...], dict[str, str]]:
    out: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(k, "") if k != "fd_step_arcmin" else fnum(row.get(k)) for k in keys)
        out[key] = row
    return out


def plot_cumulative_curves(out_dir: Path, curve_rows: list[dict[str, Any]], fd_steps: list[float]) -> list[Path]:
    paths: list[Path] = []
    for fd_step in fd_steps:
        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
        for condition in PLOT_CONDITIONS:
            rows = [
                row
                for row in curve_rows
                if row["readout"] == "pose_aware_diagonal_poisson_mean"
                and row["condition"] == condition
                and np.isclose(float(row["fd_step_arcmin"]), fd_step)
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: int(r["timebin"]))
            x = np.asarray([int(row["timebin"]) for row in rows], dtype=float)
            y = np.asarray([float(row["cumulative_fisher"]) for row in rows], dtype=float)
            sem = np.asarray([float(row.get("sem", 0.0)) for row in rows], dtype=float)
            ax.plot(x, y, label=label_condition(condition), color=COLORS.get(condition), linewidth=2.0)
            ax.fill_between(x, y - sem, y + sem, color=COLORS.get(condition), alpha=0.15, linewidth=0)
        ax.set_title(f"Cumulative Vernier Fisher ({fd_step:g} arcmin FD)")
        ax.set_xlabel("time bin")
        ax.set_ylabel("cumulative Fisher")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"cumulative_fisher_fd{fd_step:g}arcmin.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_threshold_ratios(out_dir: Path, contrast_rows: list[dict[str, str]]) -> Path:
    selected = [
        row
        for row in contrast_rows
        if row.get("readout") == "pose_aware_diagonal_poisson"
        and (row.get("condition"), row.get("baseline_condition")) in THRESHOLD_CONTRASTS
    ]
    selected = sorted(selected, key=lambda r: (fnum(r.get("fd_step_arcmin")), row_contrast_label(r)))
    labels = [f"{row_contrast_label(row)}\n{fnum(row.get('fd_step_arcmin')):g} arcmin" for row in selected]
    vals = np.asarray([fnum(row.get("mean_threshold_ratio")) for row in selected], dtype=float)
    probs = np.asarray([fnum(row.get("p_condition_beats_baseline")) for row in selected], dtype=float)
    colors = [COLORS.get(row.get("condition", ""), "#777777") for row in selected]

    fig, ax = plt.subplots(figsize=(max(6.5, 0.72 * len(labels)), 4.0), dpi=160)
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    for xi, val, prob in zip(x, vals, probs, strict=False):
        ax.text(xi, val + 0.025, f"p={prob:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("threshold ratio")
    ax.set_title("Paired Threshold Ratios (below 1 improves)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "threshold_ratio_bars.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_axis_specificity(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path:
    lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in reliability_rows if row.get("readout") == "pose_aware_diagonal_poisson"})
    x = np.arange(len(fd_steps))
    width = 0.36
    horiz = [fnum(lookup.get(("pose_aware_diagonal_poisson", "axis_horizontal", fd), {}).get("mean_final_fisher")) for fd in fd_steps]
    vert = [fnum(lookup.get(("pose_aware_diagonal_poisson", "axis_vertical", fd), {}).get("mean_final_fisher")) for fd in fd_steps]
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=160)
    ax.bar(x - width / 2, horiz, width, label="horizontal", color=COLORS["axis_horizontal"])
    ax.bar(x + width / 2, vert, width, label="vertical", color=COLORS["axis_vertical"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fd:g}" for fd in fd_steps])
    ax.set_xlabel("FD step (arcmin)")
    ax.set_ylabel("mean final Fisher")
    ax.set_title("Motion Axis Specificity")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "axis_specificity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_scaled_condition(condition: str) -> tuple[str, float] | None:
    if condition == "static_center":
        return "static center", 0.0
    if condition == "real_fem":
        return "scaled real", 1.0
    if condition == "static_phase_cloud_matched_positions":
        return "matched phase cloud", 1.0
    if condition == "order_shuffled_positions":
        return "order shuffled", 1.0
    prefixes = (
        ("scaled_real_", "scaled real"),
        ("scaled_phase_cloud_matched_positions_", "matched phase cloud"),
        ("static_phase_cloud_matched_scaled_", "matched phase cloud"),
        ("scaled_order_shuffled_positions_", "order shuffled"),
        ("order_shuffled_scaled_", "order shuffled"),
    )
    for prefix, family in prefixes:
        if condition.startswith(prefix):
            try:
                return family, float(condition[len(prefix) :])
            except ValueError:
                return None
    return None


def plot_scale_curve(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path:
    lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in reliability_rows if row.get("readout") == "pose_aware_diagonal_poisson"})
    parsed_rows = []
    for row in reliability_rows:
        if row.get("readout") != "pose_aware_diagonal_poisson":
            continue
        parsed = parse_scaled_condition(row.get("condition", ""))
        if parsed is None:
            continue
        family, scale = parsed
        parsed_rows.append((family, scale, row))

    if parsed_rows:
        fig, axes = plt.subplots(1, len(fd_steps), figsize=(5.8 * len(fd_steps), 3.8), dpi=160, sharey=True)
        if len(fd_steps) == 1:
            axes = [axes]
        for ax, fd_step in zip(axes, fd_steps, strict=True):
            for family in SCALE_FAMILY_ORDER:
                rows = [
                    (scale, row)
                    for fam, scale, row in parsed_rows
                    if fam == family and np.isclose(fnum(row.get("fd_step_arcmin")), fd_step)
                ]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda item: item[0])
                x = np.asarray([scale for scale, _row in rows], dtype=float)
                y = np.asarray([fnum(row.get("mean_final_fisher")) for _scale, row in rows], dtype=float)
                ax.plot(x, y, marker="o", linewidth=2.0, label=family)
            ax.set_title(f"{fd_step:g} arcmin FD")
            ax.set_xlabel("motion scale D")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("mean final Fisher")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle("Motion Scale Curve with Scale-Matched Controls", y=1.02)
        fig.tight_layout()
        path = out_dir / "scale_curve.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
    for fd_step in fd_steps:
        xs = []
        ys = []
        labels = []
        for condition, scale, label in SCALE_CONDITIONS:
            row = lookup.get(("pose_aware_diagonal_poisson", condition, fd_step), {})
            xs.append(scale)
            ys.append(fnum(row.get("mean_final_fisher")))
            labels.append(label)
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{fd_step:g} arcmin FD")
    ax.set_xticks([scale for _condition, scale, _label in SCALE_CONDITIONS])
    ax.set_xticklabels([label for _condition, _scale, label in SCALE_CONDITIONS])
    ax.set_xlabel("motion scale condition")
    ax.set_ylabel("mean final Fisher")
    ax.set_title("Motion Scale Curve")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "scale_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pose_readout_comparison(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path:
    conditions = ["static_center", "static_phase_cloud_matched_positions", "real_fem", "scaled_real_0.5"]
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in reliability_rows if row.get("readout") == "pose_aware_diagonal_poisson"})
    lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fig, axes = plt.subplots(1, len(fd_steps), figsize=(5.8 * len(fd_steps), 3.8), dpi=160, sharey=True)
    if len(fd_steps) == 1:
        axes = [axes]
    for ax, fd_step in zip(axes, fd_steps, strict=True):
        x = np.arange(len(conditions))
        aware = [fnum(lookup.get(("pose_aware_diagonal_poisson", cond, fd_step), {}).get("mean_final_fisher")) for cond in conditions]
        blind = [
            fnum(lookup.get(("pose_blind_diagonal_count_plus_marginal", cond, fd_step), {}).get("mean_final_fisher"))
            for cond in conditions
        ]
        width = 0.36
        ax.bar(x - width / 2, aware, width, label="pose-aware", color="#4c78a8")
        ax.bar(x + width / 2, blind, width, label="pose-blind", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels([label_condition(cond) for cond in conditions], rotation=25, ha="right")
        ax.set_title(f"{fd_step:g} arcmin FD")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean final Fisher")
    axes[-1].legend(frameon=False)
    fig.suptitle("Pose-Aware vs Pose-Blind Readouts", y=1.02)
    fig.tight_layout()
    path = out_dir / "pose_readout_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pose_uncertainty(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path | None:
    rows = [row for row in reliability_rows if str(row.get("readout", "")).startswith("pose_uncertain_diagonal_sigma")]
    if not rows:
        return None
    conditions = [cond for cond in ("real_fem", "scaled_real_0.5", "static_phase_cloud_matched_positions") if any(row.get("condition") == cond for row in rows)]
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in rows})
    lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fig, axes = plt.subplots(1, len(fd_steps), figsize=(5.8 * len(fd_steps), 3.8), dpi=160, sharey=True)
    if len(fd_steps) == 1:
        axes = [axes]
    for ax, fd_step in zip(axes, fd_steps, strict=True):
        for condition in conditions:
            cond_rows = [
                row for row in rows
                if row.get("condition") == condition and np.isclose(fnum(row.get("fd_step_arcmin")), fd_step)
            ]
            cond_rows = sorted(cond_rows, key=lambda row: fnum(row.get("pose_sigma_arcmin")))
            if not cond_rows:
                continue
            x = np.asarray([fnum(row.get("pose_sigma_arcmin")) for row in cond_rows], dtype=float)
            y = np.asarray([fnum(row.get("mean_final_fisher")) for row in cond_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=label_condition(condition))
            aware = fnum(lookup.get(("pose_aware_diagonal_poisson", condition, fd_step), {}).get("mean_final_fisher"))
            blind = fnum(lookup.get(("pose_blind_diagonal_count_plus_marginal", condition, fd_step), {}).get("mean_final_fisher"))
            if np.isfinite(aware):
                ax.axhline(aware, color=COLORS.get(condition, "#777777"), linestyle=":", linewidth=1.0, alpha=0.55)
            if np.isfinite(blind):
                ax.axhline(blind, color=COLORS.get(condition, "#777777"), linestyle="--", linewidth=1.0, alpha=0.45)
        ax.set_title(f"{fd_step:g} arcmin FD")
        ax.set_xlabel("pose uncertainty sigma (arcmin)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean final Fisher")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Pose-Uncertainty Sweep (dotted=aware, dashed=blind)", y=1.02)
    fig.tight_layout()
    path = out_dir / "pose_uncertainty_sweep.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


COMPACT_STATIC_MATCH_FIELDS = (
    "readout",
    "fd_step_arcmin",
    "compact_mode",
    "subspace_source",
    "compact_k",
    "compact_alpha",
    "cov_shrinkage",
    "unit_subset",
    "n_units_used",
)


def _same_field(a: Any, b: Any) -> bool:
    fa = fnum(a)
    fb = fnum(b)
    if np.isfinite(fa) or np.isfinite(fb):
        return bool(np.isclose(fa, fb, equal_nan=True))
    return str(a) == str(b)


def _static_ratio(row: dict[str, str], reliability_rows: list[dict[str, str]]) -> float:
    static = float("nan")
    for candidate in reliability_rows:
        if candidate.get("condition") != "static_center":
            continue
        if all(_same_field(row.get(field, ""), candidate.get(field, "")) for field in COMPACT_STATIC_MATCH_FIELDS):
            static = fnum(candidate.get("mean_final_fisher"))
            break
    val = fnum(row.get("mean_final_fisher"))
    return val / static if np.isfinite(val) and static > 0 else float("nan")


def plot_compact_aware_k_sweep(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path | None:
    rows = [row for row in reliability_rows if row.get("compact_mode") == "hard_project"]
    if not rows:
        return None
    reference_lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in rows})
    conditions = [cond for cond in ("real_fem", "scaled_real_0.5", "static_phase_cloud_matched_positions") if any(row.get("condition") == cond for row in rows)]
    fig, axes = plt.subplots(1, len(fd_steps), figsize=(6.0 * len(fd_steps), 3.8), dpi=160, sharey=True)
    if len(fd_steps) == 1:
        axes = [axes]
    for ax, fd_step in zip(axes, fd_steps, strict=True):
        for condition in conditions:
            cond_rows = [
                row for row in rows
                if row.get("condition") == condition
                and row.get("subspace_source") != "random_orthonormal"
                and np.isclose(fnum(row.get("fd_step_arcmin")), fd_step)
            ]
            cond_rows = sorted(cond_rows, key=lambda row: (str(row.get("subspace_source")), fnum(row.get("compact_k"))))
            for source in sorted({str(row.get("subspace_source")) for row in cond_rows}):
                src_rows = [row for row in cond_rows if str(row.get("subspace_source")) == source]
                if not src_rows:
                    continue
                x = np.asarray([fnum(row.get("compact_k")) for row in src_rows], dtype=float)
                y = np.asarray([_static_ratio(row, reliability_rows) for row in src_rows], dtype=float)
                ax.plot(x, y, marker="o", linewidth=1.8, label=f"{label_condition(condition)} | {source}")
            aware = _reference_ratio(reference_lookup, "pose_aware_diagonal_poisson", condition, fd_step)
            blind = _reference_ratio(reference_lookup, "pose_blind_diagonal_count_plus_marginal", condition, fd_step)
            if np.isfinite(aware):
                ax.axhline(aware, color=COLORS.get(condition, "#777777"), linestyle=":", linewidth=1.0, alpha=0.5)
            if np.isfinite(blind):
                ax.axhline(blind, color=COLORS.get(condition, "#777777"), linestyle="--", linewidth=1.0, alpha=0.4)
        ax.set_title(f"{fd_step:g} arcmin FD")
        ax.set_xlabel("removed nuisance rank k")
        ax.set_ylabel("Fisher/static-center ratio")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Compact-Aware Pose-Blind Hard Projection", y=1.02)
    fig.tight_layout()
    path = out_dir / "compact_aware_pose_blind_k_sweep.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_compact_aware_alpha_sweep(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path | None:
    rows = [row for row in reliability_rows if row.get("compact_mode") == "soft_discount"]
    if not rows:
        return None
    reference_lookup = row_lookup(reliability_rows, "readout", "condition", "fd_step_arcmin")
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in rows})
    primary_k = sorted({int(fnum(row.get("compact_k"))) for row in rows if np.isfinite(fnum(row.get("compact_k")))})
    primary_k = primary_k[min(1, len(primary_k) - 1)] if primary_k else 1
    conditions = [cond for cond in ("real_fem", "scaled_real_0.5", "static_phase_cloud_matched_positions") if any(row.get("condition") == cond for row in rows)]
    fig, axes = plt.subplots(1, len(fd_steps), figsize=(6.0 * len(fd_steps), 3.8), dpi=160, sharey=True)
    if len(fd_steps) == 1:
        axes = [axes]
    for ax, fd_step in zip(axes, fd_steps, strict=True):
        for condition in conditions:
            cond_rows = [
                row for row in rows
                if row.get("condition") == condition
                and int(fnum(row.get("compact_k"))) == int(primary_k)
                and row.get("subspace_source") != "random_orthonormal"
                and np.isclose(fnum(row.get("fd_step_arcmin")), fd_step)
            ]
            for source in sorted({str(row.get("subspace_source")) for row in cond_rows}):
                src_rows = sorted([row for row in cond_rows if str(row.get("subspace_source")) == source], key=lambda row: fnum(row.get("compact_alpha")))
                if not src_rows:
                    continue
                x = np.asarray([fnum(row.get("compact_alpha")) for row in src_rows], dtype=float)
                y = np.asarray([_static_ratio(row, reliability_rows) for row in src_rows], dtype=float)
                ax.plot(x, y, marker="o", linewidth=1.8, label=f"{label_condition(condition)} | {source}")
            aware = _reference_ratio(reference_lookup, "pose_aware_diagonal_poisson", condition, fd_step)
            blind = _reference_ratio(reference_lookup, "pose_blind_diagonal_count_plus_marginal", condition, fd_step)
            if np.isfinite(aware):
                ax.axhline(aware, color=COLORS.get(condition, "#777777"), linestyle=":", linewidth=1.0, alpha=0.5)
            if np.isfinite(blind):
                ax.axhline(blind, color=COLORS.get(condition, "#777777"), linestyle="--", linewidth=1.0, alpha=0.4)
        ax.set_title(f"{fd_step:g} arcmin FD, k={primary_k}")
        ax.set_xlabel("nuisance discount alpha")
        ax.set_ylabel("Fisher/static-center ratio")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Compact-Aware Pose-Blind Soft Discounting", y=1.02)
    fig.tight_layout()
    path = out_dir / "compact_aware_pose_blind_alpha_sweep.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _reference_ratio(lookup: dict[tuple[Any, ...], dict[str, str]], readout: str, condition: str, fd_step: float) -> float:
    val = fnum(lookup.get((readout, condition, fd_step), {}).get("mean_final_fisher"))
    static = fnum(lookup.get((readout, "static_center", fd_step), {}).get("mean_final_fisher"))
    return val / static if np.isfinite(val) and static > 0 else float("nan")


def parse_component_condition(condition: str) -> tuple[str, float] | None:
    prefixes = (
        ("drift_only_scaled_", "drift only"),
        ("microsaccade_only_scaled_", "microsaccade only"),
        ("drift_scaled_", "scale drift, real microsaccades"),
        ("microsaccade_scaled_", "scale microsaccades, real drift"),
    )
    for prefix, family in prefixes:
        if condition.startswith(prefix):
            try:
                return family, float(condition[len(prefix) :])
            except ValueError:
                return None
    if condition == "drift_only":
        return "drift only", 1.0
    if condition == "microsaccade_only":
        return "microsaccade only", 1.0
    return None


def plot_component_scale(out_dir: Path, reliability_rows: list[dict[str, str]]) -> Path | None:
    component_rows = []
    for row in reliability_rows:
        if row.get("readout") != "pose_aware_diagonal_poisson":
            continue
        parsed = parse_component_condition(row.get("condition", ""))
        if parsed is None:
            continue
        family, scale = parsed
        component_rows.append((family, scale, row))
    if not component_rows:
        return None

    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for _family, _scale, row in component_rows})
    families = [
        "drift only",
        "microsaccade only",
        "scale drift, real microsaccades",
        "scale microsaccades, real drift",
    ]
    fig, axes = plt.subplots(1, len(fd_steps), figsize=(6.0 * len(fd_steps), 4.0), dpi=160, sharey=True)
    if len(fd_steps) == 1:
        axes = [axes]
    for ax, fd_step in zip(axes, fd_steps, strict=True):
        for family in families:
            rows = [
                (scale, row)
                for fam, scale, row in component_rows
                if fam == family and np.isclose(fnum(row.get("fd_step_arcmin")), fd_step)
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: item[0])
            x = np.asarray([scale for scale, _row in rows], dtype=float)
            y = np.asarray([fnum(row.get("mean_final_fisher")) for _scale, row in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=family)
        ax.set_title(f"{fd_step:g} arcmin FD")
        ax.set_xlabel("component scale")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean final Fisher")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Drift vs Microsaccade Component Scaling", y=1.02)
    fig.tight_layout()
    path = out_dir / "component_scale_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def label_condition(condition: str) -> str:
    labels = {
        "static_center": "static center",
        "static_repeated_phase": "repeated phase",
        "static_phase_cloud_matched_positions": "matched phase cloud",
        "real_fem": "real FEM",
        "order_shuffled_positions": "order shuffled",
        "axis_horizontal": "horizontal motion",
        "axis_vertical": "vertical motion",
        "scaled_real_0.5": "0.5x real",
        "scaled_real_1.5": "1.5x real",
        "drift_only": "drift only",
        "microsaccade_only": "microsaccade only",
    }
    parsed = parse_component_condition(condition)
    if parsed is not None:
        family, scale = parsed
        return f"{family} {scale:g}x"
    scaled = parse_scaled_condition(condition)
    if scaled is not None:
        family, scale = scaled
        if condition in {"static_center", "real_fem", "static_phase_cloud_matched_positions", "order_shuffled_positions"}:
            return labels.get(condition, f"{family} {scale:g}x")
        return f"{family} {scale:g}x"
    return labels.get(condition, condition)


def row_contrast_label(row: dict[str, str]) -> str:
    return f"{label_condition(row.get('condition', ''))} / {label_condition(row.get('baseline_condition', ''))}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--phi", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "figures"
    source_dir = out_dir / "source_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    pose_aware_curves, pose_blind_curves = compute_curve_tables(run_dir, bin_seconds=args.bin_seconds, phi=args.phi)
    write_csv_rows(source_dir / "pose_aware_cumulative_fisher_curves.csv", pose_aware_curves)
    write_csv_rows(source_dir / "pose_blind_cumulative_fisher_curves.csv", pose_blind_curves)

    reliability_rows = read_csv_rows(run_dir / "condition_reliability_summary.csv")
    contrast_rows = read_csv_rows(run_dir / "paired_baseline_contrast_summary.csv")
    fd_steps = sorted({fnum(row.get("fd_step_arcmin")) for row in reliability_rows if row.get("readout") == "pose_aware_diagonal_poisson"})

    figure_paths: list[Path] = []
    figure_paths.extend(plot_cumulative_curves(out_dir, pose_aware_curves, fd_steps))
    figure_paths.append(plot_threshold_ratios(out_dir, contrast_rows))
    figure_paths.append(plot_axis_specificity(out_dir, reliability_rows))
    figure_paths.append(plot_scale_curve(out_dir, reliability_rows))
    figure_paths.append(plot_pose_readout_comparison(out_dir, reliability_rows))
    pose_uncertainty_path = plot_pose_uncertainty(out_dir, reliability_rows)
    if pose_uncertainty_path is not None:
        figure_paths.append(pose_uncertainty_path)
    compact_k_path = plot_compact_aware_k_sweep(out_dir, reliability_rows)
    if compact_k_path is not None:
        figure_paths.append(compact_k_path)
    compact_alpha_path = plot_compact_aware_alpha_sweep(out_dir, reliability_rows)
    if compact_alpha_path is not None:
        figure_paths.append(compact_alpha_path)
    component_path = plot_component_scale(out_dir, reliability_rows)
    if component_path is not None:
        figure_paths.append(component_path)

    write_json(
        out_dir / "vernier_summary_figure_manifest.json",
        {
            "run_dir": run_dir,
            "out_dir": out_dir,
            "source_tables": [source_dir / "pose_aware_cumulative_fisher_curves.csv", source_dir / "pose_blind_cumulative_fisher_curves.csv"],
            "figures": figure_paths,
            "bin_seconds": args.bin_seconds,
            "phi": args.phi,
        },
    )
    print(f"Wrote Vernier summary figures to {out_dir}")


if __name__ == "__main__":
    main()

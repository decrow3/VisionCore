#!/usr/bin/env python3
"""Assemble the active-sensing movie-information figure from twininfo outputs.

The completed ``jake.twininfo`` run contains paired trajectory conditions for
intact natural images plus FEM-rendered visual controls. It does not yet contain
stabilized versions of each visual control. This generator is intentionally
explicit about that distinction so visual phase/spectral effects are not
mistaken for FEM-vs-stabilized gains within those image controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "outputs" / "twininfo" / "active-sensing-all-images-1crop-2fix2ms-16units-gpu"
DEFAULT_OUT = ROOT / "outputs" / "active_sensing_movie_information" / "active_sensing_movie_information_figure"
PRIMARY_METRIC = "final_cumulative_spatial_ssi_bits_per_spike"


COND_LABELS = {
    "real": "real FEM",
    "stabilized": "stabilized",
    "random_amp": "amp-matched random dirs",
    "random_amp_cloud_matched": "amp+cloud matched dirs",
    "random_cov": "step-cov Gaussian",
    "trajectory_order_shuffle": "position-order shuffle",
    "phase_order_shuffle": "trajectory order shuffle",
    "pyramid_phase_scrambled": "visual phase scramble",
    "sf_low": "lowpass",
    "sf_mid_low": "mid-low SF",
    "sf_mid_high": "mid-high SF",
    "sf_high": "highpass",
    "stabilized_pyramid_phase_scrambled": "visual phase scramble stabilized",
    "stabilized_sf_low": "lowpass stabilized",
    "stabilized_sf_mid_low": "mid-low SF stabilized",
    "stabilized_sf_mid_high": "mid-high SF stabilized",
    "stabilized_sf_high": "highpass stabilized",
}

COND_COLORS = {
    "real": "#1f77b4",
    "stabilized": "#2ca02c",
    "random_amp": "#7f7f7f",
    "random_amp_cloud_matched": "#525252",
    "random_cov": "#bcbd22",
    "trajectory_order_shuffle": "#17becf",
    "phase_order_shuffle": "#17becf",
    "pyramid_phase_scrambled": "#d62728",
    "sf_low": "#9467bd",
    "sf_mid_low": "#8c564b",
    "sf_mid_high": "#ff7f0e",
    "sf_high": "#4c78a8",
    "stabilized_pyramid_phase_scrambled": "#f2a4a4",
    "stabilized_sf_low": "#c7a9dd",
    "stabilized_sf_mid_low": "#d7b5d8",
    "stabilized_sf_mid_high": "#ffbb78",
    "stabilized_sf_high": "#9ecae1",
}


def canonical_condition(condition: str) -> str:
    if condition == "phase_order_shuffle":
        return "trajectory_order_shuffle"
    return condition


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        str(row["example_id"]),
        str(row["kind"]),
        int(row["image_index"]),
        int(row["crop_rank"]),
    )


def paired_table(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str, int, int], dict[str, float]]:
    table: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for row in rows:
        table.setdefault(pair_key(row), {})[canonical_condition(str(row["condition"]))] = float(row[metric])
    return table


def available_conditions(table: dict[tuple[str, str, int, int], dict[str, float]]) -> set[str]:
    return {condition for conds in table.values() for condition in conds}


def write_csv_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cloud_matched_mode(description: str) -> str:
    if description.startswith("inverted_time_reversed_exact_scale_cloud_fallback"):
        return "inverted_time_reversed_fallback"
    if "best_effort" in description:
        return "random_candidate_best_effort"
    if description.startswith("step_amplitude_and_cloud_matched_random_directions"):
        return "random_candidate_accepted"
    if description == "empty_trace_stabilized_fallback":
        return "empty_trace_stabilized_fallback"
    return "unknown"


def write_cloud_matched_mode_audit(run_dir: Path, out_dir: Path) -> dict[str, object]:
    """Report whether random_amp_cloud_matched rows used random candidates or fallback."""
    qc_path = run_dir / "metadata" / "03_trajectory_control_qc.csv"
    out_path = out_dir / "random_amp_cloud_matched_mode_summary.csv"
    if not qc_path.exists():
        write_csv_rows(
            [{"condition": "random_amp_cloud_matched", "scope": "overall", "mode": "missing_qc", "n": 0}],
            out_path,
        )
        return {"summary_csv": str(out_path), "n": 0, "fraction_random_candidate": float("nan"), "fraction_fallback": float("nan")}

    rows = [
        row for row in read_csv_rows(qc_path)
        if canonical_condition(str(row.get("condition", ""))) == "random_amp_cloud_matched"
    ]
    mode_counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        mode = cloud_matched_mode(str(row.get("control_description", "")))
        for scope, scope_value in [("overall", "all"), ("kind", str(row.get("kind", "unknown")))]:
            key = (scope, scope_value, mode)
            mode_counts[key] = mode_counts.get(key, 0) + 1

    summary_rows: list[dict[str, object]] = []
    scopes = [("overall", "all")] + sorted({("kind", str(row.get("kind", "unknown"))) for row in rows})
    for scope, scope_value in scopes:
        scope_total = sum(
            count for (row_scope, row_value, _mode), count in mode_counts.items()
            if row_scope == scope and row_value == scope_value
        )
        modes = sorted({
            mode for (row_scope, row_value, mode) in mode_counts
            if row_scope == scope and row_value == scope_value
        })
        for mode in modes:
            n = mode_counts.get((scope, scope_value, mode), 0)
            summary_rows.append({
                "condition": "random_amp_cloud_matched",
                "scope": scope,
                "scope_value": scope_value,
                "mode": mode,
                "n": n,
                "fraction": float(n / scope_total) if scope_total else float("nan"),
            })
    write_csv_rows(summary_rows, out_path)

    total = len(rows)
    n_random = sum(1 for row in rows if cloud_matched_mode(str(row.get("control_description", ""))).startswith("random_candidate"))
    n_fallback = sum(1 for row in rows if "fallback" in cloud_matched_mode(str(row.get("control_description", ""))))
    return {
        "summary_csv": str(out_path),
        "n": total,
        "n_random_candidate": n_random,
        "n_fallback": n_fallback,
        "fraction_random_candidate": float(n_random / total) if total else float("nan"),
        "fraction_fallback": float(n_fallback / total) if total else float("nan"),
    }


def mean_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, int]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if arr.size == 1:
        val = float(arr[0])
        return val, val, val, 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), int(arr.size)


def hierarchical_mean_ci(
    entries: list[dict[str, object]],
    *,
    n_boot: int = 800,
    seed: int = 0,
) -> tuple[float, float, float, int, int]:
    """Image -> trace bootstrap for paired movie-level quantities."""
    finite = [
        entry for entry in entries
        if np.isfinite(float(entry["value"]))
    ]
    if not finite:
        return float("nan"), float("nan"), float("nan"), 0, 0
    images = sorted({int(entry["image_index"]) for entry in finite})
    by_image: dict[int, list[dict[str, object]]] = {
        image: [entry for entry in finite if int(entry["image_index"]) == image]
        for image in images
    }
    arr = np.asarray([float(entry["value"]) for entry in finite], dtype=np.float64)
    if len(images) == 1 or len(finite) == 1:
        val = float(np.mean(arr))
        return val, val, val, len(finite), len(images)
    nested: list[list[np.ndarray]] = []
    for image in images:
        image_entries = by_image[int(image)]
        trace_ids = sorted({(str(e["example_id"]), str(e["kind"]), int(e["crop_rank"])) for e in image_entries})
        trace_arrays = []
        for trace_id in trace_ids:
            vals = [
                float(e["value"]) for e in image_entries
                if (str(e["example_id"]), str(e["kind"]), int(e["crop_rank"])) == trace_id
            ]
            trace_arrays.append(np.asarray(vals, dtype=np.float64))
        nested.append(trace_arrays)
    rng = np.random.default_rng(seed)
    boot = np.zeros((int(n_boot),), dtype=np.float64)
    for b in range(int(n_boot)):
        vals: list[float] = []
        sampled_image_ix = rng.integers(0, len(nested), size=len(nested))
        for image_ix in sampled_image_ix:
            trace_arrays = nested[int(image_ix)]
            sampled_trace_ix = rng.integers(0, len(trace_arrays), size=len(trace_arrays))
            for trace_ix in sampled_trace_ix:
                trace_vals = trace_arrays[int(trace_ix)]
                vals.append(float(trace_vals[int(rng.integers(0, trace_vals.size))]))
        boot[b] = float(np.mean(vals)) if vals else float("nan")
    boot = boot[np.isfinite(boot)]
    return (
        float(np.mean(arr)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        len(finite),
        len(images),
    )


def condition_values(table: dict[tuple[str, str, int, int], dict[str, float]], condition: str, kind: str | None = None) -> list[float]:
    condition = canonical_condition(condition)
    vals = []
    for key, conds in table.items():
        if kind is not None and key[1] != kind:
            continue
        if condition in conds:
            vals.append(conds[condition])
    return vals


def condition_entries(
    table: dict[tuple[str, str, int, int], dict[str, float]],
    condition: str,
    kind: str | None = None,
) -> list[dict[str, object]]:
    condition = canonical_condition(condition)
    entries = []
    for key, conds in table.items():
        example_id, key_kind, image_index, crop_rank = key
        if kind is not None and key_kind != kind:
            continue
        if condition in conds:
            entries.append({
                "example_id": example_id,
                "kind": key_kind,
                "image_index": int(image_index),
                "crop_rank": int(crop_rank),
                "value": float(conds[condition]),
            })
    return entries


def paired_deltas(
    table: dict[tuple[str, str, int, int], dict[str, float]],
    a: str,
    b: str,
    *,
    kind: str | None = None,
) -> list[float]:
    a = canonical_condition(a)
    b = canonical_condition(b)
    vals = []
    for key, conds in table.items():
        if kind is not None and key[1] != kind:
            continue
        if a in conds and b in conds:
            vals.append(conds[a] - conds[b])
    return vals


def paired_delta_entries(
    table: dict[tuple[str, str, int, int], dict[str, float]],
    a: str,
    b: str,
    *,
    kind: str | None = None,
) -> list[dict[str, object]]:
    a = canonical_condition(a)
    b = canonical_condition(b)
    entries = []
    for key, conds in table.items():
        example_id, key_kind, image_index, crop_rank = key
        if kind is not None and key_kind != kind:
            continue
        if a in conds and b in conds:
            entries.append({
                "example_id": example_id,
                "kind": key_kind,
                "image_index": int(image_index),
                "crop_rank": int(crop_rank),
                "value": float(conds[a]) - float(conds[b]),
            })
    return entries


def load_series(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    npz = np.load(run_dir / "cache" / "cumulative_information_series.npz")
    time_s = np.asarray(npz["time_s"], dtype=np.float64)
    y = np.asarray(npz["cumulative_spatial_ssi_bits_per_spike"], dtype=np.float64)
    records = []
    for i in range(y.shape[0]):
        records.append(
            {
                "example_id": str(npz["record_example_id"][i]),
                "kind": str(npz["record_kind"][i]),
                "condition": canonical_condition(str(npz["record_condition"][i])),
                "image_index": str(int(npz["record_image_index"][i])),
                "crop_rank": str(int(npz["record_crop_rank"][i])),
            }
        )
    return time_s, y, records


def plot_mean_trace(ax: plt.Axes, time_s: np.ndarray, y: np.ndarray, records: list[dict[str, str]], condition: str) -> None:
    condition = canonical_condition(condition)
    ix = [i for i, row in enumerate(records) if canonical_condition(row["condition"]) == condition]
    if not ix:
        return
    arr = y[ix]
    mean = np.nanmean(arr, axis=0)
    lo = np.nanpercentile(arr, 2.5, axis=0)
    hi = np.nanpercentile(arr, 97.5, axis=0)
    color = COND_COLORS.get(condition, "0.4")
    ax.plot(time_s, mean, color=color, lw=2.0, label=COND_LABELS.get(condition, condition))
    ax.fill_between(time_s, lo, hi, color=color, alpha=0.045, linewidth=0)


def paired_series_curves(
    y: np.ndarray,
    records: list[dict[str, str]],
    a: str,
    b: str,
) -> np.ndarray:
    by_key: dict[tuple[str, str, int, int, str], np.ndarray] = {}
    for i, row in enumerate(records):
        key = (
            str(row["example_id"]),
            str(row["kind"]),
            int(row["image_index"]),
            int(row["crop_rank"]),
            canonical_condition(str(row["condition"])),
        )
        by_key[key] = np.asarray(y[i], dtype=np.float64)
    curves = []
    for base in sorted({key[:4] for key in by_key}):
        ka = (*base, canonical_condition(a))
        kb = (*base, canonical_condition(b))
        if ka in by_key and kb in by_key:
            curves.append(by_key[ka] - by_key[kb])
    if not curves:
        return np.empty((0, y.shape[1]), dtype=np.float64)
    return np.stack(curves, axis=0)


def write_time_resolved_real_stabilized_audit(
    *,
    out_dir: Path,
    time_s: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, str]],
) -> dict[str, object]:
    curves = paired_series_curves(y, records, "real", "stabilized")
    time_csv = out_dir / "time_resolved_real_minus_stabilized_delta.csv"
    epoch_csv = out_dir / "time_resolved_real_minus_stabilized_epoch_summary.csv"
    if curves.size == 0:
        write_csv_rows([], time_csv)
        write_csv_rows([], epoch_csv)
        return {"time_csv": str(time_csv), "epoch_csv": str(epoch_csv), "n": 0}

    mean = np.nanmean(curves, axis=0)
    lo = np.nanpercentile(curves, 2.5, axis=0)
    hi = np.nanpercentile(curves, 97.5, axis=0)
    time_rows = []
    for i, t in enumerate(time_s):
        vals = curves[:, i]
        vals = vals[np.isfinite(vals)]
        time_rows.append({
            "time_s": float(t),
            "mean_delta": float(mean[i]),
            "ci95_low_movie_percentile": float(lo[i]),
            "ci95_high_movie_percentile": float(hi[i]),
            "fraction_movies_positive": float(np.mean(vals > 0.0)) if vals.size else float("nan"),
            "n": int(vals.size),
        })
    write_csv_rows(time_rows, time_csv)

    epoch_specs = [
        ("early", 0.0, 1.0 / 3.0),
        ("middle", 1.0 / 3.0, 2.0 / 3.0),
        ("late", 2.0 / 3.0, 1.0),
        ("full", 0.0, 1.0),
    ]
    t_min = float(np.nanmin(time_s))
    t_max = float(np.nanmax(time_s))
    duration = max(t_max - t_min, 1e-12)
    epoch_rows = []
    for label, start_frac, end_frac in epoch_specs:
        start_t = t_min + duration * start_frac
        end_t = t_min + duration * end_frac
        mask = (time_s >= start_t) & (time_s <= end_t if label == "full" else time_s < end_t)
        if not np.any(mask):
            continue
        vals = np.nanmean(curves[:, mask], axis=1)
        mean_epoch, lo_epoch, hi_epoch, n_epoch = mean_ci(vals, seed=1200 + len(epoch_rows))
        epoch_rows.append({
            "epoch": label,
            "start_s": float(start_t),
            "end_s": float(end_t),
            "mean_delta": mean_epoch,
            "ci95_low": lo_epoch,
            "ci95_high": hi_epoch,
            "fraction_movies_positive": float(np.mean(vals[np.isfinite(vals)] > 0.0)) if np.any(np.isfinite(vals)) else float("nan"),
            "n": n_epoch,
        })
    write_csv_rows(epoch_rows, epoch_csv)

    positive_ix = np.flatnonzero(mean > 0.0)
    ci_positive_ix = np.flatnonzero(lo > 0.0)
    return {
        "time_csv": str(time_csv),
        "epoch_csv": str(epoch_csv),
        "n": int(curves.shape[0]),
        "final_mean_delta": float(mean[-1]),
        "min_mean_delta": float(np.nanmin(mean)),
        "fraction_timepoints_mean_positive": float(np.mean(mean > 0.0)),
        "fraction_timepoints_ci_low_positive": float(np.mean(lo > 0.0)),
        "first_mean_positive_time_s": float(time_s[int(positive_ix[0])]) if positive_ix.size else float("nan"),
        "first_ci_low_positive_time_s": float(time_s[int(ci_positive_ix[0])]) if ci_positive_ix.size else float("nan"),
        "peak_mean_delta": float(np.nanmax(mean)),
        "peak_mean_time_s": float(time_s[int(np.nanargmax(mean))]),
    }


def plot_direct_delta_curves(
    *,
    out_dir: Path,
    time_s: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, str]],
) -> dict[str, object]:
    by_key: dict[tuple[str, str, int, int, str], np.ndarray] = {}
    for i, row in enumerate(records):
        key = (
            str(row["example_id"]),
            str(row["kind"]),
            int(row["image_index"]),
            int(row["crop_rank"]),
            canonical_condition(str(row["condition"])),
        )
        by_key[key] = np.asarray(y[i], dtype=np.float64)

    pair_specs = [
        ("real - stabilized", "real", "stabilized", "#1f77b4"),
        ("real - amp-matched", "real", "random_amp", "#7f7f7f"),
        ("real - amp+cloud", "real", "random_amp_cloud_matched", "#525252"),
        ("lowpass FEM - stabilized", "sf_low", "stabilized_sf_low", "#9467bd"),
        ("mid-low FEM - stabilized", "sf_mid_low", "stabilized_sf_mid_low", "#8c564b"),
        ("mid-high FEM - stabilized", "sf_mid_high", "stabilized_sf_mid_high", "#ff7f0e"),
        ("highpass FEM - stabilized", "sf_high", "stabilized_sf_high", "#4c78a8"),
        ("phase FEM - stabilized", "pyramid_phase_scrambled", "stabilized_pyramid_phase_scrambled", "#d62728"),
    ]
    base_keys = sorted({key[:4] for key in by_key})
    stats: list[dict[str, object]] = []
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.4), sharex=True)
    for label, a, b, color in pair_specs:
        curves = []
        for base in base_keys:
            ka = (*base, a)
            kb = (*base, b)
            if ka in by_key and kb in by_key:
                curves.append(by_key[ka] - by_key[kb])
        if not curves:
            continue
        arr = np.stack(curves, axis=0)
        mean = np.nanmean(arr, axis=0)
        lo = np.nanpercentile(arr, 2.5, axis=0)
        hi = np.nanpercentile(arr, 97.5, axis=0)
        ax = axs[0] if label.startswith("real") else axs[1]
        ax.plot(time_s, mean, color=color, lw=2.0, label=label)
        ax.fill_between(time_s, lo, hi, color=color, alpha=0.14, linewidth=0)
        stats.append({
            "comparison": label,
            "condition_a": a,
            "condition_b": b,
            "n": int(arr.shape[0]),
            "final_mean": float(mean[-1]),
            "final_ci95_low_movie_percentile": float(lo[-1]),
            "final_ci95_high_movie_percentile": float(hi[-1]),
        })
    for ax, title in zip(axs, ("Trajectory-control deltas", "Image-control FEM-stabilized deltas"), strict=True):
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("time in movie (s)")
        ax.set_ylabel("additive delta bits / expected spike")
        ax.grid(color="0.9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Direct time-resolved information-efficiency gain curves", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf = out_dir / "active_sensing_direct_delta_curves.pdf"
    png = out_dir / "active_sensing_direct_delta_curves.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv_rows(stats, out_dir / "active_sensing_direct_delta_curves_summary.csv")
    return {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "summary_csv": str(out_dir / "active_sensing_direct_delta_curves_summary.csv"),
        "comparisons": stats,
    }


def bar_with_ci(
    ax: plt.Axes,
    labels: list[str],
    vals_by_label: list[list[float]],
    *,
    colors: list[str],
    ylabel: str,
    title: str,
) -> list[dict[str, object]]:
    stats = []
    x = np.arange(len(labels), dtype=np.float64)
    means, los, his = [], [], []
    for i, vals in enumerate(vals_by_label):
        mean, lo, hi, n = mean_ci(vals, seed=17 + i)
        means.append(mean)
        los.append(lo)
        his.append(hi)
        stats.append({"label": labels[i], "mean": mean, "ci95_low": lo, "ci95_high": hi, "n": n})
    yerr = np.vstack([np.asarray(means) - np.asarray(los), np.asarray(his) - np.asarray(means)])
    ax.bar(x, means, yerr=yerr, capsize=3, color=colors, alpha=0.88)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="0.9", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return stats


def bar_with_hierarchical_ci(
    ax: plt.Axes,
    labels: list[str],
    entries_by_label: list[list[dict[str, object]]],
    *,
    colors: list[str],
    ylabel: str,
    title: str,
    seed0: int = 17,
) -> list[dict[str, object]]:
    stats = []
    x = np.arange(len(labels), dtype=np.float64)
    means, los, his = [], [], []
    for i, entries in enumerate(entries_by_label):
        mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=seed0 + i)
        means.append(mean)
        los.append(lo)
        his.append(hi)
        stats.append({
            "label": labels[i],
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": n,
            "n_images": n_images,
            "bootstrap": "image_then_trace",
        })
    yerr = np.vstack([np.asarray(means) - np.asarray(los), np.asarray(his) - np.asarray(means)])
    ax.bar(x, means, yerr=yerr, capsize=3, color=colors, alpha=0.88)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="0.9", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return stats


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="0.9", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_stabilization_schematic(ax: plt.Axes) -> dict[str, object]:
    """Draw a compact same-image/same-trace schematic for the counterfactual."""
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 96)
    y = np.linspace(-1.0, 1.0, 96)
    xx, yy = np.meshgrid(x, y)
    image = (
        0.55
        + 0.16 * np.sin(8.0 * xx + 2.0 * yy)
        + 0.12 * np.cos(5.0 * yy - 3.0 * xx)
        + 0.06 * rng.normal(size=xx.shape)
    )
    image = np.clip(image, 0.0, 1.0)

    trace_t = np.linspace(0.0, 1.0, 80)
    trace_x = 0.15 * np.sin(2.0 * np.pi * trace_t) + 0.45 * (trace_t - 0.5)
    trace_y = 0.18 * np.cos(3.0 * np.pi * trace_t) + 0.10 * np.sin(7.0 * np.pi * trace_t)
    centers = [(-0.58, 0.0), (0.58, 0.0)]
    labels = ["real FEM\nretinal motion", "stabilized\ncounterfactual"]
    for i, (cx, cy) in enumerate(centers):
        extent = (cx - 0.34, cx + 0.34, cy - 0.34, cy + 0.34)
        ax.imshow(image, cmap="gray", extent=extent, vmin=0, vmax=1, zorder=0)
        rect = plt.Rectangle((extent[0], extent[2]), 0.76, 0.76, fill=False, ec="0.15", lw=1.2)
        ax.add_patch(rect)
        if i == 0:
            ax.plot(cx + 0.38 * trace_x, cy + 0.38 * trace_y, color=COND_COLORS["real"], lw=2.0)
            ax.scatter(cx + 0.38 * trace_x[-1], cy + 0.38 * trace_y[-1], s=24, color=COND_COLORS["real"], zorder=3)
        else:
            ax.plot([cx - 0.18, cx + 0.18], [cy, cy], color=COND_COLORS["stabilized"], lw=2.0)
            ax.plot([cx, cx], [cy - 0.18, cy + 0.18], color=COND_COLORS["stabilized"], lw=2.0)
        ax.text(cx, -0.58, labels[i], ha="center", va="top", fontsize=9)
    ax.annotate(
        "",
        xy=(0.16, 0.0),
        xytext=(-0.16, 0.0),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "0.25"},
    )
    ax.set_title("A  Stabilization counterfactual", loc="left", fontweight="bold")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.78, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {"description": "schematic_same_image_trace_real_motion_vs_stabilized"}


def plot_real_stabilized_timecourse(
    ax: plt.Axes,
    *,
    time_s: np.ndarray,
    series: np.ndarray,
    records: list[dict[str, str]],
) -> dict[str, object]:
    for condition in ("real", "stabilized"):
        plot_mean_trace(ax, time_s, series, records, condition)
    ax.set_title("B  Time-resolved efficiency", loc="left", fontweight="bold")
    ax.set_xlabel("time in movie (s)")
    ax.set_ylabel("cumulative bits / expected spike")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax, grid_axis="both")

    curves = paired_series_curves(series, records, "real", "stabilized")
    if curves.size == 0:
        return {"n": 0}
    mean = np.nanmean(curves, axis=0)
    lo = np.nanpercentile(curves, 2.5, axis=0)
    hi = np.nanpercentile(curves, 97.5, axis=0)
    inset = ax.inset_axes([0.08, 0.58, 0.38, 0.34])
    inset.axhline(0.0, color="0.2", lw=0.7)
    inset.plot(time_s, mean, color=COND_COLORS["real"], lw=1.6)
    inset.fill_between(time_s, lo, hi, color=COND_COLORS["real"], alpha=0.065, linewidth=0)
    inset.set_title("real - stabilized", fontsize=8, loc="left")
    inset.set_xlabel("s", fontsize=7)
    inset.set_ylabel("delta bits/spike", fontsize=7)
    inset.tick_params(labelsize=7)
    style_axis(inset, grid_axis="both")
    return {
        "n": int(curves.shape[0]),
        "final_mean_delta": float(mean[-1]),
        "min_mean_delta": float(np.nanmin(mean)),
        "fraction_timepoints_mean_positive": float(np.mean(mean > 0.0)),
    }


def paired_relative_delta_entries(
    table: dict[tuple[str, str, int, int], dict[str, float]],
    a: str,
    b: str,
    *,
    kind: str | None = None,
    eps: float = 1e-12,
) -> list[dict[str, object]]:
    a = canonical_condition(a)
    b = canonical_condition(b)
    entries = []
    for key, conds in table.items():
        example_id, key_kind, image_index, crop_rank = key
        if kind is not None and key_kind != kind:
            continue
        if a in conds and b in conds:
            denom = max(abs(float(conds[b])), eps)
            entries.append({
                "example_id": example_id,
                "kind": key_kind,
                "image_index": int(image_index),
                "crop_rank": int(crop_rank),
                "value": 100.0 * (float(conds[a]) - float(conds[b])) / denom,
            })
    return entries


def plot_primary_endpoint_with_spike_inset(
    ax: plt.Axes,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    efficiency_table = paired_table(rows, PRIMARY_METRIC)
    endpoint_specs = [
        ("stabilized", condition_entries(efficiency_table, "stabilized"), COND_COLORS["stabilized"]),
        ("real FEM", condition_entries(efficiency_table, "real"), COND_COLORS["real"]),
    ]
    endpoint_stats = []
    means, lows, highs = [], [], []
    for i, (label, entries, _color) in enumerate(endpoint_specs):
        mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=301 + i)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)
        endpoint_stats.append({
            "label": label,
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": n,
            "n_images": n_images,
            "bootstrap": "image_then_trace",
        })
    delta_entries = paired_delta_entries(efficiency_table, "real", "stabilized")
    delta_mean, delta_lo, delta_hi, delta_n, delta_n_images = hierarchical_mean_ci(delta_entries, seed=305)
    endpoint_stats.append({
        "label": "real - stabilized",
        "mean": delta_mean,
        "ci95_low": delta_lo,
        "ci95_high": delta_hi,
        "n": delta_n,
        "n_images": delta_n_images,
        "bootstrap": "image_then_trace",
    })
    x_main = np.arange(len(endpoint_specs), dtype=np.float64)
    yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
    ax.bar(x_main, means, yerr=yerr, capsize=3, color=[color for _label, _entries, color in endpoint_specs], alpha=0.88)
    y_top = max(highs) * 1.08
    ax.text(
        0.55,
        0.88,
        f"paired gain\n+{delta_mean:.3f} bits/spike",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.2",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "0.8"},
    )
    ax.set_ylim(0.0, y_top * 1.18)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xticks(x_main)
    ax.set_xticklabels([label for label, _entries, _color in endpoint_specs], rotation=0, ha="center")
    ax.set_ylabel("bits / expected spike")
    ax.set_title("C  Primary endpoint and spike audit", loc="left", fontweight="bold")
    style_axis(ax)

    audit_rows = _rows_with_derived_rates(rows, elapsed_s=1.0)
    raw_table = paired_table(audit_rows, "final_cumulative_spatial_ssi_bits")
    spikes_table = paired_table(audit_rows, "final_cumulative_expected_spikes")
    inset_specs = [
        ("raw bits", paired_relative_delta_entries(raw_table, "real", "stabilized")),
        ("expected\nspikes", paired_relative_delta_entries(spikes_table, "real", "stabilized")),
    ]
    inset_stats = []
    means, lows, highs = [], [], []
    for i, (_label, entries) in enumerate(inset_specs):
        mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=401 + i)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)
        inset_stats.append({
            "label": _label.replace("\n", " "),
            "mean_percent_delta": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": n,
            "n_images": n_images,
            "bootstrap": "image_then_trace",
        })
    inset = ax.inset_axes([0.08, 0.54, 0.38, 0.36])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.96))
    inset.patch.set_edgecolor("0.82")
    inset.patch.set_linewidth(0.8)
    x = np.arange(len(inset_specs), dtype=np.float64)
    yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
    inset.bar(x, means, yerr=yerr, capsize=2, color=["#9ecae1", "#bdbdbd"], alpha=0.9)
    inset.axhline(0.0, color="0.2", lw=0.7)
    inset.set_xticks(x)
    inset.set_xticklabels([label for label, _entries in inset_specs], fontsize=7)
    inset.set_ylabel("")
    inset.set_title("real - stabilized\npaired change (%)", fontsize=8, loc="left")
    inset.tick_params(axis="y", labelsize=7)
    style_axis(inset)
    return {"endpoint": endpoint_stats, "spike_inset": inset_stats}


def plot_sf_mechanism_panel(
    ax: plt.Axes,
    table: dict[tuple[str, str, int, int], dict[str, float]],
    present: set[str],
) -> tuple[list[dict[str, object]], bool]:
    direct_visual_pairs = [
        ("lowpass", "sf_low", "stabilized_sf_low", COND_COLORS["sf_low"]),
        ("mid-low", "sf_mid_low", "stabilized_sf_mid_low", COND_COLORS["sf_mid_low"]),
        ("mid-high", "sf_mid_high", "stabilized_sf_mid_high", COND_COLORS["sf_mid_high"]),
        ("highpass", "sf_high", "stabilized_sf_high", COND_COLORS["sf_high"]),
        ("intact", "real", "stabilized", COND_COLORS["real"]),
        ("phase\nscramble", "pyramid_phase_scrambled", "stabilized_pyramid_phase_scrambled", "#8c8c8c"),
    ]
    available = [(label, fem, stable, color) for label, fem, stable, color in direct_visual_pairs if fem in present and stable in present]
    stats: list[dict[str, object]] = []
    means, lows, highs = [], [], []
    for i, (label, fem, stable, _color) in enumerate(available):
        entries = paired_delta_entries(table, fem, stable)
        mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=501 + i)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)
        stats.append({
            "label": label.replace("\n", " "),
            "fem_condition": fem,
            "stabilized_condition": stable,
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": n,
            "n_images": n_images,
            "bootstrap": "image_then_trace",
        })
    x = np.arange(len(available), dtype=np.float64)
    colors = [color for _label, _fem, _stable, color in available]
    yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
    ax.bar(x, means, yerr=yerr, capsize=3, color=colors, alpha=0.88)
    sf_ix = [i for i, (_label, fem, _stable, _color) in enumerate(available) if fem.startswith("sf_")]
    if sf_ix:
        ax.plot(x[sf_ix], np.asarray(means)[sf_ix], color="0.15", marker="o", lw=1.6, ms=4, zorder=4)
    if any(label == "highpass" for label, _fem, _stable, _color in available):
        highpass_ix = [i for i, (label, _fem, _stable, _color) in enumerate(available) if label == "highpass"][0]
        ax.axvline(highpass_ix + 0.5, color="0.55", lw=0.8, ls="--")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _fem, _stable, _color in available], rotation=24, ha="right")
    ax.set_ylabel("FEM - stabilized\nbits / expected spike")
    ax.set_title("D  Spatial-frequency dependence", loc="left", fontweight="bold")
    style_axis(ax)
    has_augmented = all(fem in present and stable in present for _label, fem, stable, _color in direct_visual_pairs[:4])
    return stats, bool(has_augmented)


def plot_matched_motion_boundary_panel(
    ax: plt.Axes,
    table: dict[tuple[str, str, int, int], dict[str, float]],
    present: set[str],
) -> list[dict[str, object]]:
    conditions = ["stabilized", "real", "random_amp"]
    if "random_amp_cloud_matched" in present:
        conditions.append("random_amp_cloud_matched")
    stats = bar_with_hierarchical_ci(
        ax,
        ["stabilized", "real FEM", "amp-matched\nrandom", "amp+cloud\nrandom"][: len(conditions)],
        [condition_entries(table, c) for c in conditions],
        colors=[COND_COLORS[c] for c in conditions],
        ylabel="final bits / expected spike",
        title="E  Matched motion bounds trajectory-specific claims",
        seed0=601,
    )
    ax.set_xticklabels(["stabilized", "real FEM", "amp-matched\nrandom", "amp+cloud\nrandom"][: len(conditions)], rotation=18, ha="right")
    return stats


def write_supplement_event_content_panels(
    *,
    out_dir: Path,
    table: dict[tuple[str, str, int, int], dict[str, float]],
    present: set[str],
) -> dict[str, object]:
    fig, axs = plt.subplots(1, 3, figsize=(15.4, 4.4))

    controls = ["stabilized", "random_amp"]
    if "random_amp_cloud_matched" in present:
        controls.append("random_amp_cloud_matched")
    controls.extend(["random_cov", "trajectory_order_shuffle"])
    labels = [f"real -\n{COND_LABELS[c]}" for c in controls]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    stats_c = []
    for offset, kind, color in [(-width / 2, "fixation", "#4c78a8"), (width / 2, "microsaccade", "#f58518")]:
        means, lows, highs = [], [], []
        for i, control in enumerate(controls):
            entries = paired_delta_entries(table, "real", control, kind=kind)
            mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=701 + i + (0 if kind == "fixation" else 20))
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
            stats_c.append({
                "kind": kind,
                "comparison": f"real_minus_{control}",
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": n,
                "n_images": n_images,
                "bootstrap": "image_then_trace",
            })
        yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        axs[0].bar(x + offset, means, width=width, yerr=yerr, capsize=3, label=kind, color=color, alpha=0.86)
    axs[0].axhline(0, color="0.2", lw=0.8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=25, ha="right")
    axs[0].set_ylabel("paired delta bits / expected spike")
    axs[0].set_title("S1  Event-class split", loc="left", fontweight="bold")
    axs[0].legend(frameon=False, fontsize=8)
    style_axis(axs[0])

    content_conditions = ["real", "sf_low", "sf_mid_low", "sf_mid_high", "sf_high", "pyramid_phase_scrambled"]
    stats_d = bar_with_hierarchical_ci(
        axs[1],
        [COND_LABELS[c] for c in content_conditions],
        [condition_entries(table, c) for c in content_conditions],
        colors=[COND_COLORS[c] for c in content_conditions],
        ylabel="final bits / expected spike",
        title="S2  Image content under measured FEMs",
        seed0=801,
    )

    content_controls = ["sf_low", "sf_mid_low", "sf_mid_high", "sf_high", "pyramid_phase_scrambled"]
    stats_e = bar_with_hierarchical_ci(
        axs[2],
        [f"real -\n{COND_LABELS[c]}" for c in content_controls],
        [paired_delta_entries(table, "real", c) for c in content_controls],
        colors=[COND_COLORS[c] for c in content_controls],
        ylabel="paired delta bits / expected spike",
        title="S3  FEM-rendered image-control losses",
        seed0=901,
    )
    axs[2].text(
        0.03,
        0.96,
        "Same FEM trajectory in both terms",
        transform=axs[2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8"},
    )
    fig.suptitle("Supplemental active-sensing controls moved out of main Figure 5", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf = out_dir / "active_sensing_supplement_event_content_panels.pdf"
    png = out_dir / "active_sensing_supplement_event_content_panels.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "event_class_split": stats_c,
        "fem_image_content_means": stats_d,
        "real_minus_fem_image_controls": stats_e,
    }


def _stat_by_label(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    for row in rows:
        if str(row.get("label", "")).replace("\n", " ") == label:
            return row
    return {}


def _fmt(value: object, digits: int = 3, prefix_plus: bool = False) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if prefix_plus and val >= 0 else ""
    return f"{sign}{val:.{digits}f}"


def _fmt_ci(row: dict[str, object], digits: int = 3) -> str:
    return f"{_fmt(row.get('ci95_low'), digits)} to {_fmt(row.get('ci95_high'), digits)}"


def write_manuscript_caption(stats: dict[str, object], out_dir: Path) -> Path:
    """Write the manuscript-style Figure 5 caption beside the rendered figure."""
    caption_path = out_dir / "active_sensing_movie_information_figure_caption.md"
    panel_c = stats.get("panel_c_primary_endpoint_and_spike_audit", {})
    endpoint = panel_c.get("endpoint", []) if isinstance(panel_c, dict) else []
    spike_inset = panel_c.get("spike_inset", []) if isinstance(panel_c, dict) else []
    stabilized = _stat_by_label(endpoint, "stabilized")
    real = _stat_by_label(endpoint, "real FEM")
    delta = _stat_by_label(endpoint, "real - stabilized")
    raw_bits = _stat_by_label(spike_inset, "raw bits")
    expected_spikes = _stat_by_label(spike_inset, "expected spikes")

    sf_rows = stats.get("panel_d_spatial_frequency_dependence", [])
    sf = {str(row.get("label", "")).replace("\n", " "): row for row in sf_rows if isinstance(row, dict)}
    boundary_rows = stats.get("panel_e_matched_motion_boundary", [])
    boundary = {str(row.get("label", "")).replace("\n", " "): row for row in boundary_rows if isinstance(row, dict)}
    time_stats = stats.get("panel_b_time_resolved_real_vs_stabilized", {})
    cloud = stats.get("random_amp_cloud_matched_mode_audit", {})

    lines = [
        "# Figure 5 Caption",
        "",
        "**Figure 5. Real FEM retinal motion improves spatial information efficiency in the V1 model.** "
        "(A) Paired stabilization counterfactual. For each image history and measured fixation trace, "
        "the real retinal movie was compared with a stabilized movie in which retinal translation was removed. "
        "(B) Cumulative spatial information per expected spike over time for real and stabilized movies; "
        "inset shows the paired real-minus-stabilized gain. The final paired gain was "
        f"{_fmt(time_stats.get('final_mean_delta'), 3, prefix_plus=True)} bits/expected spike and the mean "
        f"real-minus-stabilized curve remained positive at {_fmt(100.0 * float(time_stats.get('fraction_timepoints_mean_positive', 0.0)), 1)}% "
        "of sampled time points. "
        "(C) Final endpoint and spike-count audit. Real FEMs increased spatial information per expected spike "
        f"from {_fmt(stabilized.get('mean'), 3)} to {_fmt(real.get('mean'), 3)} bits/expected spike "
        f"(paired gain {_fmt(delta.get('mean'), 3, prefix_plus=True)}, 95% CI {_fmt_ci(delta)}). "
        "Raw information and expected spike count also increased "
        f"({_fmt(raw_bits.get('mean_percent_delta'), 1, prefix_plus=True)}% and "
        f"{_fmt(expected_spikes.get('mean_percent_delta'), 1, prefix_plus=True)}%, respectively), "
        "but the efficiency gain survived normalization. "
        "(D) Direct FEM-minus-stabilized gain across spatial-frequency controls. The gain rose from lowpass "
        f"({_fmt(sf.get('lowpass', {}).get('mean'), 3, prefix_plus=True)}) to mid-low "
        f"({_fmt(sf.get('mid-low', {}).get('mean'), 3, prefix_plus=True)}) and mid-high "
        f"({_fmt(sf.get('mid-high', {}).get('mean'), 3, prefix_plus=True)}) spatial-frequency content, "
        f"with highpass also elevated ({_fmt(sf.get('highpass', {}).get('mean'), 3, prefix_plus=True)}); "
        f"intact natural images are shown as a reference ({_fmt(sf.get('intact', {}).get('mean'), 3, prefix_plus=True)}) "
        f"and phase scramble as a diagnostic ({_fmt(sf.get('phase scramble', {}).get('mean'), 3, prefix_plus=True)}). "
        "(E) Matched-motion controls. Amp-matched and amp+cloud-matched random trajectories equaled or exceeded "
        "real FEM endpoint efficiency "
        f"(stabilized {_fmt(boundary.get('stabilized', {}).get('mean'), 3)}, real FEM {_fmt(boundary.get('real FEM', {}).get('mean'), 3)}, "
        f"amp-matched random {_fmt(boundary.get('amp-matched random', {}).get('mean'), 3)}, "
        f"amp+cloud random {_fmt(boundary.get('amp+cloud random', {}).get('mean'), 3)} bits/expected spike), "
        "bounding the claim to a retinal-motion benefit rather than unique optimality of measured trajectory statistics.",
        "",
        "Validation notes: all intervals are image-then-trace bootstrap 95% CIs unless otherwise noted. "
        f"The `random_amp_cloud_matched` validation used {cloud.get('n_random_candidate', 0)}/{cloud.get('n', 0)} "
        f"random candidates and {cloud.get('n_fallback', 0)}/{cloud.get('n', 0)} fallback trajectories. "
        "Event-class and measured-FEM-only image-content controls are saved in "
        "`active_sensing_supplement_event_content_panels`.",
        "",
    ]
    caption_path.write_text("\n".join(lines), encoding="utf-8")
    return caption_path


def annotate_missing(ax: plt.Axes, available: bool) -> None:
    if available:
        return
    ax.text(
        0.5,
        0.5,
        "Stabilized visual-control movies\nnot present in this run.\n\nLowpass/highpass/visual-phase panels\nshow FEM-rendered image controls,\nnot FEM-minus-stabilized gains.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.75"},
    )


def _rows_with_derived_rates(rows: list[dict[str, str]], *, elapsed_s: float) -> list[dict[str, str]]:
    out = []
    for row in rows:
        enriched = dict(row)
        expected_spikes = float(row["final_cumulative_expected_spikes"])
        enriched["final_expected_spikes_per_second"] = str(expected_spikes / max(float(elapsed_s), 1e-12))
        out.append(enriched)
    return out


def write_spike_count_audit(
    rows: list[dict[str, str]],
    *,
    out_dir: Path,
    elapsed_s: float,
) -> dict[str, object]:
    """Write H7 diagnostics: raw bits, bits/sec, spikes, spikes/sec, and bits/spike."""
    audit_rows = _rows_with_derived_rates(rows, elapsed_s=elapsed_s)
    metrics = [
        ("final_cumulative_spatial_ssi_bits_per_spike", "bits_per_expected_spike"),
        ("final_cumulative_spatial_ssi_bits", "cumulative_bits"),
        ("final_cumulative_spatial_ssi_bits_per_second", "bits_per_second"),
        ("final_cumulative_expected_spikes", "expected_spikes"),
        ("final_expected_spikes_per_second", "expected_spikes_per_second"),
    ]
    all_conditions = [
        "real",
        "stabilized",
        "random_amp",
        "random_amp_cloud_matched",
        "random_cov",
        "trajectory_order_shuffle",
        "sf_low",
        "sf_mid_low",
        "sf_mid_high",
        "sf_high",
        "pyramid_phase_scrambled",
        "stabilized_sf_low",
        "stabilized_sf_mid_low",
        "stabilized_sf_mid_high",
        "stabilized_sf_high",
        "stabilized_pyramid_phase_scrambled",
    ]

    condition_summary: list[dict[str, object]] = []
    for metric_key, metric_label in metrics:
        table = paired_table(audit_rows, metric_key)
        for condition in all_conditions:
            entries = condition_entries(table, condition)
            if not entries:
                continue
            mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=700 + len(condition_summary))
            condition_summary.append({
                "condition": condition,
                "metric": metric_label,
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": n,
                "n_images": n_images,
                "bootstrap": "image_then_trace",
            })

    paired_specs = [
        ("intact", "real", "stabilized"),
        ("lowpass", "sf_low", "stabilized_sf_low"),
        ("mid-low", "sf_mid_low", "stabilized_sf_mid_low"),
        ("mid-high", "sf_mid_high", "stabilized_sf_mid_high"),
        ("highpass", "sf_high", "stabilized_sf_high"),
        ("phase_scramble", "pyramid_phase_scrambled", "stabilized_pyramid_phase_scrambled"),
    ]
    paired_summary: list[dict[str, object]] = []
    for metric_key, metric_label in metrics:
        table = paired_table(audit_rows, metric_key)
        for label, fem, stable in paired_specs:
            entries = paired_delta_entries(table, fem, stable)
            if not entries:
                continue
            mean, lo, hi, n, n_images = hierarchical_mean_ci(entries, seed=900 + len(paired_summary))
            paired_summary.append({
                "comparison": label,
                "fem_condition": fem,
                "stabilized_condition": stable,
                "metric": metric_label,
                "mean_delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": n,
                "n_images": n_images,
                "bootstrap": "image_then_trace",
            })

    write_csv_rows(condition_summary, out_dir / "spike_count_audit_condition_summary.csv")
    write_csv_rows(paired_summary, out_dir / "spike_count_audit_paired_deltas.csv")

    fig_metrics = [
        ("bits_per_expected_spike", "bits / expected spike"),
        ("cumulative_bits", "cumulative bits"),
        ("bits_per_second", "bits / s"),
        ("expected_spikes", "expected spikes"),
        ("expected_spikes_per_second", "expected spikes / s"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(14.6, 7.4))
    axs = axs.ravel()
    colors = ["#1f77b4", "#9467bd", "#8c564b", "#ff7f0e", "#4c78a8", "#d62728"]
    for ax, (metric_label, ylabel) in zip(axs, fig_metrics, strict=False):
        rows_metric = [row for row in paired_summary if row["metric"] == metric_label]
        labels = [str(row["comparison"]) for row in rows_metric]
        means = np.asarray([float(row["mean_delta"]) for row in rows_metric], dtype=np.float64)
        lows = np.asarray([float(row["ci95_low"]) for row in rows_metric], dtype=np.float64)
        highs = np.asarray([float(row["ci95_high"]) for row in rows_metric], dtype=np.float64)
        x = np.arange(len(labels), dtype=np.float64)
        yerr = np.vstack([means - lows, highs - means])
        ax.bar(x, means, yerr=yerr, capsize=3, color=colors[: len(labels)], alpha=0.88)
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(f"FEM - stabilized\n{ylabel}")
        ax.set_title(metric_label.replace("_", " "), loc="left", fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[-1].axis("off")
    fig.suptitle("Spike-count audit: additive FEM-minus-stabilized diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf = out_dir / "active_sensing_spike_count_audit.pdf"
    png = out_dir / "active_sensing_spike_count_audit.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "elapsed_s": float(elapsed_s),
        "condition_summary_csv": str(out_dir / "spike_count_audit_condition_summary.csv"),
        "paired_deltas_csv": str(out_dir / "spike_count_audit_paired_deltas.csv"),
        "figure_pdf": str(pdf),
        "figure_png": str(png),
    }


def make_figure(run_dir: Path, out_dir: Path) -> dict[str, object]:
    rows = read_csv_rows(run_dir / "metadata" / "05_lagcube_information_summary.csv")
    table = paired_table(rows, PRIMARY_METRIC)
    present = available_conditions(table)
    time_s, series, records = load_series(run_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_mode_audit = write_cloud_matched_mode_audit(run_dir, out_dir)
    time_delta_audit = write_time_resolved_real_stabilized_audit(
        out_dir=out_dir,
        time_s=time_s,
        y=series,
        records=records,
    )
    spike_audit = write_spike_count_audit(rows, out_dir=out_dir, elapsed_s=float(time_s[-1]))
    delta_curves = plot_direct_delta_curves(out_dir=out_dir, time_s=time_s, y=series, records=records)
    supplement_controls = write_supplement_event_content_panels(out_dir=out_dir, table=table, present=present)

    fig = plt.figure(figsize=(15.0, 8.0))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[0.92, 1.0], wspace=0.95, hspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    stats_a = plot_stabilization_schematic(ax_a)
    stats_b = plot_real_stabilized_timecourse(ax_b, time_s=time_s, series=series, records=records)
    stats_c = plot_primary_endpoint_with_spike_inset(ax_c, rows)
    stats_d, has_augmented = plot_sf_mechanism_panel(ax_d, table, present)
    stats_e = plot_matched_motion_boundary_panel(ax_e, table, present)
    annotate_missing(ax_d, has_augmented)

    fig.suptitle(
        "Real FEM retinal motion improves spatial information efficiency in the V1 model",
        y=0.965,
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.11)

    pdf = out_dir / "active_sensing_movie_information_figure.pdf"
    png = out_dir / "active_sensing_movie_information_figure.png"
    svg = out_dir / "active_sensing_movie_information_figure.svg"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {
        "source_run": str(run_dir),
        "primary_metric": PRIMARY_METRIC,
        "n_rows": len(rows),
        "n_pairs": len(table),
        "has_stabilized_visual_controls": bool(has_augmented),
        "panel_a_schematic": stats_a,
        "panel_b_time_resolved_real_vs_stabilized": stats_b,
        "panel_c_primary_endpoint_and_spike_audit": stats_c,
        "panel_d_spatial_frequency_dependence": stats_d,
        "panel_e_matched_motion_boundary": stats_e,
        "supplement_event_and_content_controls": supplement_controls,
        "random_amp_cloud_matched_mode_audit": cloud_mode_audit,
        "time_resolved_real_minus_stabilized_delta": time_delta_audit,
        "spike_count_audit": spike_audit,
        "direct_delta_curves": delta_curves,
        "outputs": {"pdf": str(pdf), "png": str(png), "svg": str(svg)},
    }
    caption_path = write_manuscript_caption(stats, out_dir)
    stats["outputs"]["caption_md"] = str(caption_path)
    with (out_dir / "active_sensing_movie_information_figure_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (out_dir / "active_sensing_movie_information_figure_legend.md").open("w", encoding="utf-8") as handle:
        stabilized_note = (
            "The source run contains the augmented stabilized lowpass, mid-low, mid-high, "
            "highpass, and visual-phase controls, so panel F is the direct FEM-minus-stabilized "
            "interaction within each image-control family."
            if has_augmented
            else
            "The current source run contains direct FEM-vs-stabilized comparisons for intact images, "
            "and FEM-rendered spectral/visual phase controls. It does not yet contain stabilized "
            "versions of the lowpass, mid-low, mid-high, highpass, or visual phase-scrambled controls."
        )
        cloud_note = (
            f"`random_amp_cloud_matched` provenance: {cloud_mode_audit['n_random_candidate']}/"
            f"{cloud_mode_audit['n']} rows used random candidates and "
            f"{cloud_mode_audit['n_fallback']}/{cloud_mode_audit['n']} used fallback. "
            if cloud_mode_audit.get("n", 0)
            else "`random_amp_cloud_matched` provenance was unavailable. "
        )
        handle.write(
            "# Active-Sensing Movie Information Figure Legend\n\n"
            "Real FEM retinal motion improves spatial information efficiency relative to the "
            "matched stabilized counterfactual. The primary endpoint is cumulative spatial "
            "single-spike information from the deterministic V1-model rate movie, normalized "
            "by expected spike count. Panel D shows direct FEM-minus-stabilized gains across "
            "spatial-frequency image controls; intact natural images are shown as a reference "
            "and phase-scrambled images as a diagnostic. Matched random controls in Panel E "
            "bound trajectory-specific optimality claims rather than negating the retinal-motion "
            "benefit. "
            f"{cloud_note}"
            f"{stabilized_note} Event-class and measured-FEM-only image-content panels were moved "
            "to `active_sensing_supplement_event_content_panels`.\n"
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = make_figure(args.run_dir, args.out_dir)
    print(f"Wrote {stats['outputs']['pdf']}")
    print(f"Wrote {stats['outputs']['png']}")
    print(f"has_stabilized_visual_controls={stats['has_stabilized_visual_controls']}")


if __name__ == "__main__":
    main()

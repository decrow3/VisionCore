#!/usr/bin/env python3
"""Build a contour-axis balanced BackImage window manifest for long RR100 runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _prepare_windows


DEFAULT_INPUT = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_balanced_window_manifest_axis30_n576_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument("--axis-bin-width-deg", type=float, default=30.0)
    parser.add_argument("--target-per-bin", type=int, default=96)
    parser.add_argument(
        "--energy-balance-column",
        type=str,
        default="",
        help=(
            "Optional image-structure column to balance within each contour-axis bin. "
            "Derived columns include image_oriented_gradient_energy, image_multi_orientation_energy, "
            "image_abs_8plus_power_proxy, and image_oriented_8plus_power_proxy."
        ),
    )
    parser.add_argument(
        "--energy-quantile-bins",
        type=int,
        default=2,
        help="Number of global quantile bins for --energy-balance-column.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Optional total cap after balanced bin sampling. 0 keeps target-per-bin from each bin.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--allow-underfilled-bins", action="store_true")
    parser.add_argument("--allow-underfilled-energy-bins", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


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
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def axis_values(frame: pd.DataFrame, axis_column: str) -> np.ndarray:
    if axis_column not in frame.columns:
        raise ValueError(f"Missing axis column {axis_column!r}")
    axis = pd.to_numeric(frame[axis_column], errors="coerce").to_numpy(dtype=np.float64) % 180.0
    if not np.isfinite(axis).all():
        n_bad = int(np.count_nonzero(~np.isfinite(axis)))
        raise ValueError(f"Found {n_bad} non-finite values in {axis_column!r}")
    return axis


def circular_axis_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def add_derived_energy_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    gradient = pd.to_numeric(out.get("image_gradient_energy", np.nan), errors="coerce").astype(float)
    coherence = pd.to_numeric(out.get("image_orientation_coherence", np.nan), errors="coerce").astype(float)
    out["image_oriented_gradient_energy"] = gradient * np.maximum(coherence, 0.0)
    out["image_multi_orientation_energy"] = gradient * np.maximum(1.0 - coherence, 0.0)
    if {
        "image_power_8plus_cpd_fraction",
        "image_patch_std",
        "image_spectrum_anisotropy",
        "image_spectrum_orientation_deg",
        "image_edge_axis_deg",
    }.issubset(out.columns):
        abs8 = (
            pd.to_numeric(out["image_power_8plus_cpd_fraction"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
        )
        spectrum_contour_axis = pd.to_numeric(out["image_spectrum_orientation_deg"], errors="coerce").to_numpy(float) + 90.0
        edge_axis = pd.to_numeric(out["image_edge_axis_deg"], errors="coerce").to_numpy(float)
        agreement = np.cos(2.0 * np.radians(circular_axis_delta_deg(edge_axis, spectrum_contour_axis)))
        out["image_abs_8plus_power_proxy"] = abs8
        out["image_edge_spectrum_contour_axis_agreement"] = agreement
        out["image_oriented_8plus_power_proxy"] = (
            abs8
            * np.maximum(pd.to_numeric(out["image_spectrum_anisotropy"], errors="coerce").astype(float), 0.0)
            * np.maximum(agreement, 0.0)
        )
    return out


def bin_edges(width_deg: float) -> np.ndarray:
    width = float(width_deg)
    if width <= 0.0 or 180.0 % width > 1e-9:
        raise ValueError("--axis-bin-width-deg must be a positive divisor of 180")
    return np.arange(0.0, 180.0 + 0.5 * width, width, dtype=np.float64)


def target_counts(total: int, n_bins: int) -> list[int]:
    base = int(total) // int(n_bins)
    remainder = int(total) % int(n_bins)
    return [base + (1 if idx < remainder else 0) for idx in range(int(n_bins))]


def choose_balanced_windows(work: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(args.seed))
    edges = bin_edges(float(args.axis_bin_width_deg))
    annotated = add_derived_energy_features(work)
    axis = axis_values(annotated, str(args.axis_column))
    bin_index = np.clip(np.digitize(axis, edges, right=False) - 1, 0, len(edges) - 2)
    annotated["axis_balance_deg"] = axis
    annotated["axis_balance_bin"] = bin_index.astype(int)
    annotated["axis_balance_bin_start_deg"] = edges[bin_index]
    annotated["axis_balance_bin_stop_deg"] = edges[bin_index + 1]

    energy_column = str(args.energy_balance_column).strip()
    energy_bin_labels: np.ndarray | None = None
    if energy_column:
        if energy_column not in annotated.columns:
            raise ValueError(f"--energy-balance-column {energy_column!r} is not present or derivable.")
        values = pd.to_numeric(annotated[energy_column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            n_bad = int(np.count_nonzero(~np.isfinite(values.to_numpy(dtype=float))))
            raise ValueError(f"--energy-balance-column {energy_column!r} has {n_bad} non-finite values.")
        qbins = int(args.energy_quantile_bins)
        if qbins <= 1:
            raise ValueError("--energy-quantile-bins must be > 1 when --energy-balance-column is set.")
        energy_bin_labels = pd.qcut(values, q=qbins, labels=False, duplicates="raise").to_numpy(dtype=int)
        annotated["energy_balance_column"] = energy_column
        annotated["energy_balance_value"] = values.to_numpy(dtype=float)
        annotated["energy_balance_bin"] = energy_bin_labels
        annotated["energy_balance_quantile_bins"] = qbins

    target = int(args.target_per_bin)
    if target <= 0:
        raise ValueError("--target-per-bin must be positive")
    selected_parts: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    energy_count_rows: list[dict[str, Any]] = []
    for idx in range(len(edges) - 1):
        sub = annotated[annotated["axis_balance_bin"].astype(int) == idx].copy()
        available = int(sub.shape[0])
        chosen_all: list[int] = []
        if energy_column and energy_bin_labels is not None:
            qbins = int(args.energy_quantile_bins)
            for eb, take_target in enumerate(target_counts(target, qbins)):
                cell = sub[sub["energy_balance_bin"].astype(int) == eb]
                available_cell = int(cell.shape[0])
                take_cell = min(int(take_target), available_cell)
                underfilled_cell = available_cell < int(take_target)
                if underfilled_cell and not bool(args.allow_underfilled_energy_bins):
                    raise ValueError(
                        f"Axis bin {edges[idx]:g}-{edges[idx + 1]:g} deg, energy bin {eb} "
                        f"has {available_cell} windows, below target={take_target}. Lower target-per-bin, "
                        "lower energy-quantile-bins, or pass --allow-underfilled-energy-bins."
                    )
                if take_cell:
                    chosen_all.extend(rng.choice(cell.index.to_numpy(dtype=int), size=take_cell, replace=False).tolist())
                energy_count_rows.append(
                    {
                        "axis_bin": int(idx),
                        "axis_bin_start_deg": float(edges[idx]),
                        "axis_bin_stop_deg": float(edges[idx + 1]),
                        "energy_balance_column": energy_column,
                        "energy_balance_bin": int(eb),
                        "available_windows": available_cell,
                        "target_windows": int(take_target),
                        "selected_windows": int(take_cell),
                        "underfilled": bool(underfilled_cell),
                    }
                )
            if len(chosen_all) < target and bool(args.allow_underfilled_energy_bins):
                remaining = np.asarray(sorted(set(sub.index.to_numpy(dtype=int)).difference(chosen_all)), dtype=int)
                top_up = min(target - len(chosen_all), int(remaining.size))
                if top_up:
                    chosen_all.extend(rng.choice(remaining, size=top_up, replace=False).tolist())
        else:
            take = min(target, available)
            underfilled = available < target
            if underfilled and not bool(args.allow_underfilled_bins):
                raise ValueError(
                    f"Axis bin {edges[idx]:g}-{edges[idx + 1]:g} deg has {available} windows, "
                    f"below target-per-bin={target}. Lower the target or pass --allow-underfilled-bins."
                )
            chosen_all = rng.choice(sub.index.to_numpy(dtype=int), size=take, replace=False).tolist()
        selected_parts.append(sub.loc[np.sort(np.asarray(chosen_all, dtype=int))].copy())
        count_rows.append(
            {
                "axis_bin": int(idx),
                "axis_bin_start_deg": float(edges[idx]),
                "axis_bin_stop_deg": float(edges[idx + 1]),
                "available_windows": available,
                "selected_windows": int(len(chosen_all)),
                "underfilled": bool(available < target),
            }
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    if int(args.max_windows) > 0 and selected.shape[0] > int(args.max_windows):
        selected = selected.sample(n=int(args.max_windows), replace=False, random_state=int(args.seed)).copy()
    selected = selected.sort_values(["session", "trial_idx", "source_row"]).reset_index(drop=True)
    selected["balanced_manifest_index"] = np.arange(selected.shape[0], dtype=int)
    return selected, pd.DataFrame(count_rows), pd.DataFrame(energy_count_rows)


def plot_histograms(out_dir: Path, full: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> Path:
    edges = bin_edges(float(args.axis_bin_width_deg))
    full_axis = axis_values(full, str(args.axis_column))
    selected_axis = axis_values(selected, str(args.axis_column))
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    ax.hist(full_axis, bins=edges, color="#cccccc", edgecolor="white", label=f"eligible (n={len(full_axis)})")
    ax.hist(
        selected_axis,
        bins=edges,
        color="#168a96",
        alpha=0.75,
        edgecolor="white",
        label=f"selected (n={len(selected_axis)})",
    )
    ax.set_xlabel(f"{args.axis_column} (deg, axial)")
    ax.set_ylabel("fixations")
    ax.set_title("BackImage contour-axis balanced long-run manifest")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="0.9", linewidth=0.7)
    out = out_dir / "backimage_contour_axis_balanced_manifest_histogram.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    return out


FEATURE_BALANCE_COLUMNS = [
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_spectrum_anisotropy",
    "image_abs_8plus_power_proxy",
    "image_oriented_8plus_power_proxy",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]


def annotate_axis_bins(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    edges = bin_edges(float(args.axis_bin_width_deg))
    out = add_derived_energy_features(frame)
    axis = axis_values(out, str(args.axis_column))
    bin_index = np.clip(np.digitize(axis, edges, right=False) - 1, 0, len(edges) - 2)
    out["axis_balance_deg"] = axis
    out["axis_balance_bin"] = bin_index.astype(int)
    out["axis_balance_bin_start_deg"] = edges[bin_index]
    out["axis_balance_bin_stop_deg"] = edges[bin_index + 1]
    return out


def feature_balance_summary(full: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_annotated = annotate_axis_bins(full, args)
    selected_annotated = annotate_axis_bins(selected, args)
    for sample_label, frame in [("eligible", full_annotated), ("selected", selected_annotated)]:
        for axis_bin, sub in frame.groupby("axis_balance_bin", sort=True):
            for col in FEATURE_BALANCE_COLUMNS:
                if col not in sub.columns:
                    continue
                values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "sample": sample_label,
                        "axis_bin": int(axis_bin),
                        "axis_bin_start_deg": float(sub["axis_balance_bin_start_deg"].iloc[0]),
                        "axis_bin_stop_deg": float(sub["axis_balance_bin_stop_deg"].iloc[0]),
                        "feature": col,
                        "n": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_balance(out_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> Path:
    features = [
        "image_oriented_gradient_energy",
        "image_gradient_energy",
        "image_orientation_coherence",
        "image_oriented_8plus_power_proxy",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), features, strict=True):
        sub = summary[(summary["sample"].astype(str) == "selected") & (summary["feature"].astype(str) == feature)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("axis_bin")
        x = np.arange(sub.shape[0])
        y = sub["median"].to_numpy(dtype=float)
        lo = sub["q25"].to_numpy(dtype=float)
        hi = sub["q75"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color="#168a96", linewidth=1.8)
        ax.fill_between(x, lo, hi, color="#168a96", alpha=0.18, linewidth=0.0)
        labels = [f"{float(a):g}-{float(b):g}" for a, b in zip(sub["axis_bin_start_deg"], sub["axis_bin_stop_deg"], strict=True)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(feature.replace("image_", "").replace("_", " "), fontsize=9)
        ax.grid(True, color="0.9", linewidth=0.7)
    fig.suptitle("Selected-window image-structure balance by contour-axis bin", fontsize=12)
    out = out_dir / "backimage_contour_axis_feature_balance.png"
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_args = argparse.Namespace(
        input=Path(args.input),
        window_manifest=None,
        max_images=0,
        seed=int(args.seed),
        patch_size_px=int(args.patch_size_px),
        min_patch_image_margin_px=args.min_patch_image_margin_px,
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
    )
    work = _prepare_windows(prepare_args)
    selected, counts, energy_counts = choose_balanced_windows(work, args)
    selected_path = out_dir / "selected_windows.csv"
    counts_path = out_dir / "axis_bin_counts.csv"
    energy_counts_path = out_dir / "axis_energy_bin_counts.csv"
    feature_balance_path = out_dir / "axis_feature_balance_summary.csv"
    selected.to_csv(selected_path, index=False)
    counts.to_csv(counts_path, index=False)
    energy_counts.to_csv(energy_counts_path, index=False)
    feature_summary = feature_balance_summary(work, selected, args)
    feature_summary.to_csv(feature_balance_path, index=False)
    histogram = plot_histograms(out_dir, work, selected, args)
    feature_balance_png = plot_feature_balance(out_dir, feature_summary, args)
    write_json(
        out_dir / "run_metadata.json",
        {
            "analysis": "backimage_contour_axis_balanced_window_manifest",
            "config": vars(args),
            "input": Path(args.input),
            "selected_windows_csv": selected_path,
            "axis_bin_counts_csv": counts_path,
            "axis_energy_bin_counts_csv": energy_counts_path,
            "axis_feature_balance_summary_csv": feature_balance_path,
            "histogram_png": histogram,
            "feature_balance_png": feature_balance_png,
            "n_eligible_windows": int(work.shape[0]),
            "n_selected_windows": int(selected.shape[0]),
            "n_sessions_selected": int(selected["session"].nunique()),
        },
    )
    print(f"Wrote {selected_path}")
    print(f"Wrote {counts_path}")
    print(f"Wrote {energy_counts_path}")
    print(f"Wrote {feature_balance_path}")
    print(f"Wrote {histogram}")
    print(f"Wrote {feature_balance_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot SF-group population SSI schematics from Vernier-style unit SSI caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (  # noqa: E402
    DEFAULT_SCALES,
    condition_name,
    parse_scales,
)


DEFAULT_RUN_DIR = ROOT / "outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid_n128"
DEFAULT_UNIT_METADATA = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/unit_feature_table.csv"
)
DEFAULT_CACHE_PREFIX = "rr100_real_trace_along0_unit_ssi"
SF_ORDER = ("low_sf", "middle_sf", "high_sf")
SF_LABELS = {
    "low_sf": "Low SF",
    "middle_sf": "Middle SF",
    "high_sf": "High SF",
}
SF_COLORS = {
    "low_sf": "#3b6ea8",
    "middle_sf": "#5f8f3f",
    "high_sf": "#b65a41",
}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--unit-ssi-dir",
        type=Path,
        default=None,
        help="Directory containing cache/*.npz unit SSI caches. Defaults to RUN_DIR/unit_ssi_along0_diagnostics.",
    )
    parser.add_argument(
        "--unit-metadata",
        type=Path,
        default=DEFAULT_UNIT_METADATA,
        help="RR100 unit feature table with unit_index and sf_group columns.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--cache-prefix", type=str, default=DEFAULT_CACHE_PREFIX)
    parser.add_argument("--baseline-condition", type=str, default="static_center")
    parser.add_argument("--x-axis", choices=("path_length", "across_scale"), default="path_length")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", type=str, default="RR100 Vernier movement SSI by SF group")
    parser.add_argument("--stem", type=str, default="vernier_along0")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def cache_root(unit_ssi_dir: Path) -> Path:
    candidate = Path(unit_ssi_dir) / "cache"
    return candidate if candidate.exists() else Path(unit_ssi_dir)


def unit_cache_path(
    *,
    unit_ssi_dir: Path,
    cache_prefix: str,
    condition: str,
    fd_step_arcmin: float,
    max_frames: int,
) -> Path:
    root = cache_root(unit_ssi_dir)
    exact = root / f"{cache_prefix}_{condition}_frames{int(max_frames)}_fd{float(fd_step_arcmin):.4f}arcmin.npz"
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"*_{condition}_frames{int(max_frames)}_fd{float(fd_step_arcmin):.4f}arcmin.npz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous unit SSI caches for {condition}: {matches}")
    raise FileNotFoundError(f"Missing unit SSI cache for {condition}: expected {exact}")


def rate_cache_path(run_dir: Path, condition: str, fd_step_arcmin: float) -> Path:
    return Path(run_dir) / "cache" / f"rr100_rates_{condition}_fd{float(fd_step_arcmin):.4f}arcmin.npz"


def condition_rows(across_scales: list[float], along_scale: float, baseline_condition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": str(baseline_condition),
            "condition_label": "static" if str(baseline_condition) == "static_center" else str(baseline_condition),
            "across_scale": float("nan"),
            "along_scale": float("nan"),
            "is_static_baseline": True,
            "path_bin_order": -1,
        }
    ]
    for order, across in enumerate(across_scales):
        condition = condition_name(float(across), float(along_scale))
        rows.append(
            {
                "condition": condition,
                "condition_label": f"{float(across):g}x",
                "across_scale": float(across),
                "along_scale": float(along_scale),
                "is_static_baseline": False,
                "path_bin_order": order,
            }
        )
    return rows


def load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out = {
            "unit_bits_per_trace": np.asarray(data["unit_bits_per_trace"], dtype=np.float64),
            "unit_mean_rate_per_trace": np.asarray(data["unit_mean_rate_per_trace"], dtype=np.float64),
            "population_bits_per_trace": np.asarray(data["population_bits_per_trace"], dtype=np.float64),
        }
        if "pose_traces" in data:
            out["pose_traces"] = np.asarray(data["pose_traces"], dtype=np.float64)
        return out


def load_sf_groups(unit_metadata: Path, n_units: int) -> pd.DataFrame:
    table = pd.read_csv(unit_metadata)
    needed = {"unit_index", "sf_group"}
    missing = needed.difference(table.columns)
    if missing:
        raise ValueError(f"Unit metadata is missing columns: {sorted(missing)}")
    out = table[["unit_index", "sf_group"]].copy()
    out["unit_index"] = out["unit_index"].astype(int)
    out = out[out["unit_index"].between(0, int(n_units) - 1)].copy()
    if out["unit_index"].nunique() != int(n_units):
        raise ValueError(
            f"Unit metadata covers {out['unit_index'].nunique()} units, but caches contain {int(n_units)} units."
        )
    return out


def per_trace_weighted_parts(bits: np.ndarray, rates: np.ndarray, unit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(bits, dtype=np.float64)[:, unit_indices]
    r = np.asarray(rates, dtype=np.float64)[:, unit_indices]
    valid = np.isfinite(b) & np.isfinite(r) & (r >= 0.0)
    numer = np.sum(np.where(valid, r * b, 0.0), axis=1)
    denom = np.sum(np.where(valid, r, 0.0), axis=1)
    return numer.astype(np.float64), denom.astype(np.float64)


def weighted_value(numer: np.ndarray, denom: np.ndarray) -> float:
    total_denom = float(np.sum(denom))
    if not np.isfinite(total_denom) or total_denom <= EPS:
        return float("nan")
    return float(np.sum(numer) / total_denom)


def bootstrap_weighted(numer: np.ndarray, denom: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    n = int(np.asarray(numer).size)
    if n <= 0 or int(n_bootstrap) <= 0:
        return np.asarray([], dtype=np.float64)
    weights = rng.multinomial(n, np.full(n, 1.0 / float(n)), size=int(n_bootstrap)).astype(np.float64)
    boot_denom = weights @ np.asarray(denom, dtype=np.float64)
    boot_numer = weights @ np.asarray(numer, dtype=np.float64)
    return np.divide(boot_numer, boot_denom, out=np.full_like(boot_numer, np.nan), where=boot_denom > EPS)


def ci95(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p_two_sided(delta_values: np.ndarray) -> float:
    vals = np.asarray(delta_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(vals <= 0.0)), float(np.mean(vals >= 0.0)))
    return float(min(max(p, 0.0), 1.0))


def path_length_median_arcmin(run_dir: Path, condition: str, fd_step_arcmin: float) -> float:
    if condition == "static_center":
        return 0.0
    path = rate_cache_path(run_dir, condition, fd_step_arcmin)
    if not path.exists():
        return float("nan")
    with np.load(path) as data:
        if "pose_traces" not in data:
            return float("nan")
        poses = np.asarray(data["pose_traces"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1] < 2:
        return float("nan")
    trace_lengths = np.sum(np.linalg.norm(np.diff(poses, axis=1), axis=2), axis=1) * 60.0
    return float(np.nanmedian(trace_lengths))


def path_length_median_arcmin_from_stats(stats: dict[str, np.ndarray]) -> float:
    if "pose_traces" not in stats:
        return float("nan")
    poses = np.asarray(stats["pose_traces"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1] < 2:
        return float("nan")
    trace_lengths = np.sum(np.linalg.norm(np.diff(poses, axis=1), axis=2), axis=1) * 60.0
    return float(np.nanmedian(trace_lengths))


def summarize_population(
    *,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, np.ndarray]],
    sf_table: pd.DataFrame,
    run_dir: Path,
    fd_step_arcmin: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sf_pos, sf_group in enumerate(SF_ORDER):
        unit_indices = (
            sf_table.loc[sf_table["sf_group"].eq(sf_group), "unit_index"].astype(int).sort_values().to_numpy()
        )
        if unit_indices.size == 0:
            continue
        baseline = stats_by_condition[str(rows[0]["condition"])]
        base_numer, base_denom = per_trace_weighted_parts(
            baseline["unit_bits_per_trace"], baseline["unit_mean_rate_per_trace"], unit_indices
        )
        base_point = weighted_value(base_numer, base_denom)
        for row_pos, row in enumerate(rows):
            condition = str(row["condition"])
            stats = stats_by_condition[condition]
            cached_path_median = path_length_median_arcmin_from_stats(stats)
            if not np.isfinite(cached_path_median):
                cached_path_median = path_length_median_arcmin(run_dir, condition, fd_step_arcmin)
            numer, denom = per_trace_weighted_parts(
                stats["unit_bits_per_trace"], stats["unit_mean_rate_per_trace"], unit_indices
            )
            point = weighted_value(numer, denom)
            rng = np.random.default_rng(int(seed) + 100003 * sf_pos + 9176 * row_pos)
            boot = bootstrap_weighted(numer, denom, n_bootstrap=int(n_bootstrap), rng=rng)
            ci_low, ci_high = ci95(boot)
            if bool(row["is_static_baseline"]):
                delta = 0.0
                delta_low = 0.0
                delta_high = 0.0
                delta_p = float("nan")
            else:
                n = min(int(numer.size), int(base_numer.size))
                rng_delta = np.random.default_rng(int(seed) + 390001 * sf_pos + 7919 * row_pos)
                counts = rng_delta.multinomial(n, np.full(n, 1.0 / float(n)), size=max(int(n_bootstrap), 0)).astype(
                    np.float64
                )
                cond_den = counts @ denom[:n]
                cond_num = counts @ numer[:n]
                base_den = counts @ base_denom[:n]
                base_num = counts @ base_numer[:n]
                cond_boot = np.divide(cond_num, cond_den, out=np.full_like(cond_num, np.nan), where=cond_den > EPS)
                base_boot = np.divide(base_num, base_den, out=np.full_like(base_num, np.nan), where=base_den > EPS)
                delta_boot = cond_boot - base_boot
                delta = float(point - base_point)
                delta_low, delta_high = ci95(delta_boot)
                delta_p = bootstrap_p_two_sided(delta_boot)
            records.append(
                {
                    "sf_group": sf_group,
                    "sf_group_label": SF_LABELS.get(sf_group, sf_group),
                    "condition": condition,
                    "condition_label": row["condition_label"],
                    "across_scale": row["across_scale"],
                    "along_scale": row["along_scale"],
                    "is_static_baseline": bool(row["is_static_baseline"]),
                    "path_bin_order": int(row["path_bin_order"]),
                    "path_median_arcmin": cached_path_median,
                    "n_traces": int(numer.size),
                    "n_units": int(unit_indices.size),
                    "weighted_ssi_numerator": float(np.sum(numer)),
                    "expected_spikes": float(np.sum(denom)),
                    "population_ssi_bits_per_spike": float(point),
                    "population_ssi_delta_vs_baseline": float(delta),
                    "population_ci95_low_trace_boot": float(ci_low),
                    "population_ci95_high_trace_boot": float(ci_high),
                    "population_delta_ci95_low_trace_boot": float(delta_low),
                    "population_delta_ci95_high_trace_boot": float(delta_high),
                    "population_delta_bootstrap_p_two_sided": float(delta_p),
                    "baseline_condition": str(rows[0]["condition"]),
                    "baseline_population_ssi_bits_per_spike": float(base_point),
                }
            )
    return pd.DataFrame(records)


def display_x(summary: pd.DataFrame, *, x_axis: str) -> tuple[pd.DataFrame, str]:
    out = summary.copy()
    if x_axis == "across_scale":
        finite = out.loc[~out["is_static_baseline"], "across_scale"].to_numpy(dtype=np.float64)
        max_x = float(np.nanmax(finite)) if finite.size else 1.0
        out["plot_x"] = out["across_scale"].astype(float)
        out.loc[out["is_static_baseline"], "plot_x"] = -0.18 * max(max_x, 1.0)
        return out, "across-contour motion scale, along=0"

    finite = out.loc[~out["is_static_baseline"], "path_median_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        out["plot_x"] = out["across_scale"].astype(float)
        out.loc[out["is_static_baseline"], "plot_x"] = -0.08
        return out, "across-contour motion scale, along=0"
    max_x = float(np.nanmax(finite))
    out["plot_x"] = out["path_median_arcmin"].astype(float)
    out.loc[out["is_static_baseline"], "plot_x"] = -0.18 * max(max_x, 1.0)
    return out, "median retinal path length (arcmin)"


def finite_limits(values: np.ndarray, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi <= lo:
        return lo - 0.5, hi + 0.5
    pad = (hi - lo) * float(pad_fraction)
    return lo - pad, hi + pad


def asymmetric_yerr(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    return np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])


def plot_six_panel(summary: pd.DataFrame, *, path: Path, title: str, x_axis: str, dpi: int) -> None:
    plot_df, xlabel = display_x(summary, x_axis=x_axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(SF_ORDER),
        2,
        figsize=(8.8, 7.1),
        dpi=int(dpi),
        sharex="col",
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes)

    for row_idx, sf_group in enumerate(SF_ORDER):
        sf = plot_df[plot_df["sf_group"].eq(sf_group)].sort_values(["is_static_baseline", "path_bin_order"])
        color = SF_COLORS.get(sf_group, "#333333")
        for col_idx, metric in enumerate(("population_ssi_bits_per_spike", "population_ssi_delta_vs_baseline")):
            ax = axes_arr[row_idx, col_idx]
            for is_static, sub in sf.groupby("is_static_baseline", sort=False):
                sub = sub.sort_values("path_bin_order")
                x = sub["plot_x"].to_numpy(dtype=np.float64)
                y = sub[metric].to_numpy(dtype=np.float64)
                if metric == "population_ssi_bits_per_spike":
                    lo = sub["population_ci95_low_trace_boot"].to_numpy(dtype=np.float64)
                    hi = sub["population_ci95_high_trace_boot"].to_numpy(dtype=np.float64)
                else:
                    lo = sub["population_delta_ci95_low_trace_boot"].to_numpy(dtype=np.float64)
                    hi = sub["population_delta_ci95_high_trace_boot"].to_numpy(dtype=np.float64)
                marker = "s" if bool(is_static) else "o"
                linestyle = "none" if bool(is_static) else "-"
                face = "white" if bool(is_static) else color
                ax.errorbar(
                    x,
                    y,
                    yerr=asymmetric_yerr(y, lo, hi),
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.35,
                    markersize=4.8,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    capsize=2.0,
                    zorder=3 if not bool(is_static) else 4,
                )
            ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if col_idx == 1:
                ax.axhline(0.0, color="#707070", linestyle="--", linewidth=0.8)
            if x_axis == "path_length":
                ax.axvline(0.0, color="#c4c4c4", linewidth=0.65)
            if row_idx == 0:
                ax.set_title("Absolute SSI" if col_idx == 0 else "Delta vs static", fontsize=10.2)
            if col_idx == 0:
                ax.set_ylabel(f"{SF_LABELS.get(sf_group, sf_group)}\nbits/spike", fontsize=9.4)
            else:
                ax.set_ylabel("bits/spike", fontsize=9.4)

    for col_idx in range(2):
        axes_arr[-1, col_idx].set_xlabel(xlabel, fontsize=9.4)

    all_x = plot_df["plot_x"].to_numpy(dtype=np.float64)
    xlo, xhi = finite_limits(all_x, pad_fraction=0.04)
    for ax in axes_arr.ravel():
        ax.set_xlim(xlo, xhi)
        if x_axis == "path_length":
            static_x = float(plot_df.loc[plot_df["is_static_baseline"], "plot_x"].iloc[0])
            max_path = float(np.nanmax(plot_df.loc[~plot_df["is_static_baseline"], "path_median_arcmin"]))
            ticks = [static_x, 0.0, 60.0, 120.0, 240.0]
            if max_path > 300.0:
                ticks.append(360.0)
            ticks = [tick for tick in ticks if xlo <= tick <= xhi]
            labels = ["static" if np.isclose(tick, static_x) else f"{tick:g}" for tick in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

    fig.suptitle(title, fontsize=12.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    unit_ssi_dir = Path(args.unit_ssi_dir) if args.unit_ssi_dir is not None else run_dir / "unit_ssi_along0_diagnostics"
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "sf_group_population_schematic"
    rows = condition_rows(parse_scales(args.across_scales), float(args.along_scale), str(args.baseline_condition))
    stats_by_condition: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        condition = str(row["condition"])
        path = unit_cache_path(
            unit_ssi_dir=unit_ssi_dir,
            cache_prefix=str(args.cache_prefix),
            condition=condition,
            fd_step_arcmin=float(args.fd_step_arcmin),
            max_frames=int(args.max_frames),
        )
        stats_by_condition[condition] = load_stats(path)

    n_units = int(next(iter(stats_by_condition.values()))["unit_bits_per_trace"].shape[1])
    sf_table = load_sf_groups(Path(args.unit_metadata), n_units)
    summary = summarize_population(
        rows=rows,
        stats_by_condition=stats_by_condition,
        sf_table=sf_table,
        run_dir=run_dir,
        fd_step_arcmin=float(args.fd_step_arcmin),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{args.stem}_low_middle_high_sf_spike_weighted_population_summary.csv"
    summary.to_csv(summary_csv, index=False)
    figure_path = out_dir / "figures" / f"{args.stem}_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png"
    plot_six_panel(summary, path=figure_path, title=str(args.title), x_axis=str(args.x_axis), dpi=int(args.dpi))
    manifest_path = out_dir / f"{args.stem}_sf_group_population_schematic_manifest.json"
    manifest = {
        "analysis": "rr100_movement_sf_group_population_schematic",
        "run_dir": run_dir,
        "unit_ssi_dir": unit_ssi_dir,
        "unit_metadata": Path(args.unit_metadata),
        "out_dir": out_dir,
        "summary_csv": summary_csv,
        "figure_path": figure_path,
        "conditions": [row["condition"] for row in rows],
        "baseline_condition": str(args.baseline_condition),
        "across_scales": parse_scales(args.across_scales),
        "along_scale": float(args.along_scale),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "max_frames": int(args.max_frames),
        "cache_prefix": str(args.cache_prefix),
        "x_axis": str(args.x_axis),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "population_contract": "trace-bootstrap spike-weighted unit SSI, sum(rate * unit_bits_per_spike) / sum(rate)",
    }
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(f"Wrote SF-group population summary: {summary_csv}", flush=True)
    print(f"Wrote SF-group six-panel: {figure_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

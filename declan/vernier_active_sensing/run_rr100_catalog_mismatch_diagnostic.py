#!/usr/bin/env python3
"""Diagnose whether a held-out trajectory catalog is dense enough for Vernier.

For each observed trajectory, this compares two diagonal-Poisson quadratic
distances:

``D_sign``
    The Vernier signal distance between ``+delta`` and ``-delta`` for the true
    trajectory.

``D_traj``
    The same-sign response mismatch between the observed trajectory and its
    nearest held-out catalog trajectory.

When ``D_traj >> D_sign``, a leave-one-trajectory-out catalog marginal is
expected to struggle: every retained trajectory is a worse response match than
the Vernier sign effect we want to decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS
from .run_rr100_heldout_trajectory_observer import split_trace_indices
from .run_rr100_noisy_trajectory_observer import (
    _cache_tables,
    _condition_metadata,
    _load_rr100_caches,
    json_ready,
)


DEFAULT_SOURCE_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc")
DEFAULT_OUT_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_catalog_mismatch_diagnostic_along1")


def _parse_conditions(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _poisson_quadratic(a: np.ndarray, b: np.ndarray, *, epsilon: float = 1e-8) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Distance inputs must have same shape, got {x.shape} and {y.shape}")
    diag = np.maximum(0.5 * (x + y), 0.0) + float(epsilon)
    return float(np.sum((x - y) * (x - y) / diag))


def _nearest_prior(
    observed_poses_arcmin: np.ndarray,
    prior_poses_arcmin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed_poses_arcmin, dtype=np.float64)
    prior = np.asarray(prior_poses_arcmin, dtype=np.float64)
    diff = obs[:, None, :, :] - prior[None, :, :, :]
    dist2 = np.mean(np.sum(diff * diff, axis=3), axis=2)
    nearest_local = np.argmin(dist2, axis=1).astype(int)
    nearest_dist2 = dist2[np.arange(obs.shape[0]), nearest_local]
    return nearest_local, nearest_dist2


def score_condition(
    *,
    condition: str,
    cache: dict[str, Any],
    metadata: dict[str, Any],
    obs_indices: np.ndarray,
    prior_indices: np.ndarray,
    bin_seconds: float,
    max_timebins: int,
) -> list[dict[str, Any]]:
    table, poses_arcmin, t = _cache_tables(cache, bin_seconds=bin_seconds, max_timebins=max_timebins)
    obs_indices = np.asarray(obs_indices, dtype=int)
    prior_indices = np.asarray(prior_indices, dtype=int)
    if np.intersect1d(obs_indices, prior_indices).size:
        raise ValueError("Observation and prior indices must be disjoint")

    obs_poses = poses_arcmin[obs_indices]
    prior_poses = poses_arcmin[prior_indices]
    nearest_local, nearest_dist2 = _nearest_prior(obs_poses, prior_poses)
    nearest_global = prior_indices[nearest_local]

    rows: list[dict[str, Any]] = []
    for local_obs, trace_idx in enumerate(obs_indices):
        nearest_idx = int(nearest_global[local_obs])
        for true_label in THETA_LABELS:
            other_label = THETA_MINUS if true_label == THETA_PLUS else THETA_PLUS
            true_counts = table[true_label][trace_idx]
            nearest_same_sign = table[true_label][nearest_idx]
            other_same_trace = table[other_label][trace_idx]
            d_traj = _poisson_quadratic(true_counts, nearest_same_sign)
            d_sign = _poisson_quadratic(true_counts, other_same_trace)
            ratio = d_traj / d_sign if np.isfinite(d_sign) and d_sign > 1e-12 else float("nan")
            rows.append(
                {
                    "condition": condition,
                    "source_cache": str(cache["path"]),
                    "fd_step_arcmin": float(cache["fd_step_arcmin"]),
                    "n_timebins": int(t),
                    "n_units": int(table[true_label].shape[2]),
                    "true_label": true_label,
                    "trace_index": int(trace_idx),
                    "nearest_prior_trace_index": nearest_idx,
                    "nearest_prior_local_index": int(nearest_local[local_obs]),
                    "nearest_rms_dist_arcmin": float(np.sqrt(max(float(nearest_dist2[local_obs]), 0.0))),
                    "d_traj_same_sign_to_nearest": float(d_traj),
                    "d_sign_same_trace": float(d_sign),
                    "d_traj_over_d_sign": float(ratio),
                    **metadata,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = [
        "condition",
        "label",
        "across_scale",
        "along_scale",
        "is_static_baseline",
        "fd_step_arcmin",
        "n_timebins",
        "n_units",
    ]
    out = []
    for key, grp in df.groupby(group_cols, dropna=False, sort=True):
        row = {col: val for col, val in zip(group_cols, key, strict=True)}
        ratio = pd.to_numeric(grp["d_traj_over_d_sign"], errors="coerce").to_numpy(dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        d_traj = pd.to_numeric(grp["d_traj_same_sign_to_nearest"], errors="coerce").to_numpy(dtype=float)
        d_sign = pd.to_numeric(grp["d_sign_same_trace"], errors="coerce").to_numpy(dtype=float)
        nearest = pd.to_numeric(grp["nearest_rms_dist_arcmin"], errors="coerce").to_numpy(dtype=float)
        row.update(
            {
                "n": int(len(grp)),
                "n_observation_trajectories": int(grp["trace_index"].nunique()),
                "n_prior_trajectories": int(grp["nearest_prior_trace_index"].nunique()),
                "mean_nearest_rms_dist_arcmin": float(np.nanmean(nearest)),
                "median_nearest_rms_dist_arcmin": float(np.nanmedian(nearest)),
                "mean_d_traj": float(np.nanmean(d_traj)),
                "median_d_traj": float(np.nanmedian(d_traj)),
                "mean_d_sign": float(np.nanmean(d_sign)),
                "median_d_sign": float(np.nanmedian(d_sign)),
                "mean_d_traj_over_d_sign": float(np.nanmean(ratio)) if ratio.size else float("nan"),
                "median_d_traj_over_d_sign": float(np.nanmedian(ratio)) if ratio.size else float("nan"),
                "fraction_d_traj_gt_d_sign": float(np.mean(ratio > 1.0)) if ratio.size else float("nan"),
                "fraction_d_traj_gt_10x_d_sign": float(np.mean(ratio > 10.0)) if ratio.size else float("nan"),
                "fraction_d_traj_gt_100x_d_sign": float(np.mean(ratio > 100.0)) if ratio.size else float("nan"),
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def write_plot(out_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    grid = summary[~summary["is_static_baseline"].astype(str).str.lower().isin({"true", "1"})].copy()
    if grid.empty:
        return
    for col in [
        "across_scale",
        "median_d_traj_over_d_sign",
        "mean_d_traj_over_d_sign",
        "mean_nearest_rms_dist_arcmin",
        "median_nearest_rms_dist_arcmin",
    ]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid = grid.sort_values("across_scale")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=180, constrained_layout=True)
    axes[0].plot(grid["across_scale"], grid["median_d_traj_over_d_sign"], marker="o", label="median")
    axes[0].plot(grid["across_scale"], grid["mean_d_traj_over_d_sign"], marker="s", label="mean", alpha=0.75)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    axes[0].axhline(10.0, color="#777777", linestyle=":", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("D_traj / D_sign")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(grid["across_scale"], grid["median_nearest_rms_dist_arcmin"], marker="o", label="median")
    axes[1].plot(grid["across_scale"], grid["mean_nearest_rms_dist_arcmin"], marker="s", label="mean", alpha=0.75)
    axes[1].set_ylabel("nearest held-out RMS distance (arcmin)")

    axes[2].plot(grid["across_scale"], grid["fraction_d_traj_gt_d_sign"], marker="o", label="> 1x")
    axes[2].plot(grid["across_scale"], grid["fraction_d_traj_gt_10x_d_sign"], marker="o", label="> 10x")
    axes[2].plot(grid["across_scale"], grid["fraction_d_traj_gt_100x_d_sign"], marker="o", label="> 100x")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("fraction of observations")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("across scale, along fixed at 1x")
        ax.set_xscale("symlog", linthresh=0.125, linscale=0.7)
        ax.set_xticks([0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0", "0.125", "0.25", "0.5", "0.75", "1", "1.5", "2", "3"], rotation=35, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RR100 held-out catalog mismatch diagnostic", y=1.04)
    fig.savefig(out_dir / "rr100_catalog_mismatch_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = _load_rr100_caches(source_dir)
    meta = _condition_metadata(source_dir)
    any_cache = next(iter(caches.values()))
    bin_seconds = float(args.bin_seconds) if float(args.bin_seconds) > 0 else float(any_cache["bin_seconds"])
    n_traces = int(np.asarray(any_cache["plus_rates"]).shape[0])
    obs_indices, prior_indices = split_trace_indices(
        n_traces,
        n_observation_traces=int(args.n_observation_traces),
        n_prior_traces=int(args.n_prior_traces),
        seed=int(args.split_seed),
    )
    conditions = args.conditions or sorted(caches)
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        if condition not in caches:
            raise FileNotFoundError(f"Missing condition cache: {condition}")
        rows.extend(
            score_condition(
                condition=condition,
                cache=caches[condition],
                metadata=meta.get(condition, {}),
                obs_indices=obs_indices,
                prior_indices=prior_indices,
                bin_seconds=bin_seconds,
                max_timebins=int(args.max_timebins),
            )
        )
    trials = pd.DataFrame(rows)
    summary = summarize(rows)
    trials.to_csv(out_dir / "rr100_catalog_mismatch_trials.csv", index=False)
    summary.to_csv(out_dir / "rr100_catalog_mismatch_summary.csv", index=False)
    manifest = {
        "source_dir": source_dir,
        "out_dir": out_dir,
        "conditions": conditions,
        "bin_seconds": float(bin_seconds),
        "n_observation_traces": int(args.n_observation_traces),
        "n_prior_traces": int(args.n_prior_traces),
        "split_seed": int(args.split_seed),
        "observation_indices": obs_indices,
        "prior_indices": prior_indices,
        "distance_definition": (
            "D_traj is same-sign diagonal-Poisson quadratic distance from the observed "
            "trajectory response to its nearest held-out catalog trajectory. D_sign is "
            "the same-trajectory +delta versus -delta distance."
        ),
    }
    (out_dir / "rr100_catalog_mismatch_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plot(out_dir, summary)
    return trials, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--n-observation-traces", type=int, default=32)
    parser.add_argument("--n-prior-traces", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=0.0)
    parser.add_argument("--max-timebins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = _parse_conditions(args.conditions)
    trials, summary = run(args)
    print(f"Wrote {len(trials)} mismatch trial rows and {len(summary)} summary rows", flush=True)


if __name__ == "__main__":
    main()

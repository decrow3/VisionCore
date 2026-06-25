"""Constrained temporal-filter diagnostic for the 4C local observation model.

The unrestricted lagged-regressor diagnostic showed apparent extra dimensions
but worse trajectory-held-out prediction. This script tests a stricter family:
causal temporal filters that keep the eye regressor two-dimensional. These are
closer to a latency or retinal temporal-filter hypothesis and do not increase
the latent state dimension.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    _load_npz,
    _observation_matrix_condition_metrics,
    _trajectory_xy_by_candidate,
    project_response_delta,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_lagged_observation_diagnostic import (
    MANIFEST,
    OUT_DIR,
    SIDECAR_ROOT,
    SOURCE_ROOT,
    _clean_axis,
    _configure_matplotlib,
    _energy_r2,
    _load_basis,
    _ridge_fit,
)


@dataclass(frozen=True)
class FilterSpec:
    label: str
    family: str
    parameter: float


FILTER_SPECS = [
    FilterSpec("instant", "delay", 0.0),
    FilterSpec("delay1", "delay", 1.0),
    FilterSpec("delay2", "delay", 2.0),
    FilterSpec("delay3", "delay", 3.0),
    FilterSpec("delay4", "delay", 4.0),
    FilterSpec("delay6", "delay", 6.0),
    FilterSpec("delay8", "delay", 8.0),
    FilterSpec("box2", "boxcar", 2.0),
    FilterSpec("box4", "boxcar", 4.0),
    FilterSpec("box8", "boxcar", 8.0),
    FilterSpec("ema0p25", "ema", 0.25),
    FilterSpec("ema0p50", "ema", 0.50),
    FilterSpec("ema0p75", "ema", 0.75),
    FilterSpec("ema0p90", "ema", 0.90),
]
PLOT_ORDER = [spec.label for spec in FILTER_SPECS]
PLOT_LABELS = {
    "instant": "0",
    "delay1": "d1",
    "delay2": "d2",
    "delay3": "d3",
    "delay4": "d4",
    "delay6": "d6",
    "delay8": "d8",
    "box2": "box2",
    "box4": "box4",
    "box8": "box8",
    "ema0p25": "e.25",
    "ema0p50": "e.50",
    "ema0p75": "e.75",
    "ema0p90": "e.90",
}


def _causal_delay(xy: np.ndarray, delay: int) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    n_time = arr.shape[1]
    source_time = np.maximum(0, np.arange(n_time) - int(delay))
    return arr[:, source_time, :]


def _causal_boxcar(xy: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    win = max(1, int(window))
    out = np.empty_like(arr)
    for time_index in range(arr.shape[1]):
        start = max(0, time_index - win + 1)
        out[:, time_index, :] = np.mean(arr[:, start : time_index + 1, :], axis=1)
    return out


def _causal_ema(xy: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    a = float(alpha)
    if a < 0.0 or a >= 1.0 or not np.isfinite(a):
        raise ValueError("EMA alpha must be finite and in [0, 1)")
    out = np.empty_like(arr)
    out[:, 0, :] = arr[:, 0, :]
    for time_index in range(1, arr.shape[1]):
        out[:, time_index, :] = a * out[:, time_index - 1, :] + (1.0 - a) * arr[:, time_index, :]
    return out


def _filtered_design(xy: np.ndarray, spec: FilterSpec) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"xy must be (trajectory,time,2), got {arr.shape}")
    if spec.family == "delay":
        return _causal_delay(arr, int(spec.parameter))
    if spec.family == "boxcar":
        return _causal_boxcar(arr, int(spec.parameter))
    if spec.family == "ema":
        return _causal_ema(arr, float(spec.parameter))
    raise ValueError(f"unknown filter family: {spec.family}")


def _candidate_metrics(
    *,
    z: np.ndarray,
    xy: np.ndarray,
    spec: FilterSpec,
    n_folds: int,
    ridge: float,
) -> dict[str, float]:
    design = _filtered_design(xy, spec)
    n_traj, n_time, feature_dim = design.shape
    if feature_dim != 2:
        raise ValueError(f"filtered design must stay 2D, got {feature_dim}")
    k_dim = int(z.shape[2])
    x_flat = design.reshape(n_traj * n_time, feature_dim)
    y_flat = np.asarray(z, dtype=np.float64).reshape(n_traj * n_time, k_dim)
    coef = _ridge_fit(x_flat, y_flat, ridge)
    pred = x_flat @ coef
    train_r2 = _energy_r2(y_flat, pred)

    fold_indices = np.array_split(np.arange(n_traj), max(1, min(int(n_folds), n_traj)))
    sse = 0.0
    denom = 0.0
    fold_r2 = []
    for test_idx in fold_indices:
        train_mask = np.ones(n_traj, dtype=bool)
        train_mask[test_idx] = False
        if not bool(np.any(train_mask)):
            continue
        x_train = design[train_mask].reshape(-1, feature_dim)
        y_train = z[train_mask].reshape(-1, k_dim)
        x_test = design[test_idx].reshape(-1, feature_dim)
        y_test = z[test_idx].reshape(-1, k_dim)
        fold_coef = _ridge_fit(x_train, y_train, ridge)
        test_pred = x_test @ fold_coef
        resid = y_test - test_pred
        sse += float(np.sum(resid * resid))
        denom += float(np.sum(y_test * y_test))
        fold_r2.append(_energy_r2(y_test, test_pred))
    cv_r2 = 1.0 - sse / denom if denom > 1e-12 else float("nan")

    coef_matrix = coef.T
    singular = np.linalg.svd(coef_matrix, compute_uv=False)
    s1 = float(singular[0]) if singular.size > 0 else float("nan")
    s2 = float(singular[1]) if singular.size > 1 else float("nan")
    metrics = {
        "train_r2_energy": float(train_r2),
        "cv_r2_energy": float(cv_r2),
        "mean_fold_r2_energy": float(np.nanmean(fold_r2)) if fold_r2 else float("nan"),
        "coef_singular1": s1,
        "coef_singular2": s2,
        "coef_s2_over_s1": float(s2 / max(s1, 1e-12)) if np.isfinite(s1) and np.isfinite(s2) else float("nan"),
    }
    metrics.update(_observation_matrix_condition_metrics(coef_matrix))
    return metrics


def _table_rows(
    *,
    table: dict[str, np.ndarray],
    manifest_row: pd.Series,
    basis: np.ndarray,
    n_folds: int,
    ridge: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
    zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
    n_candidates, n_traj, n_time, _n_units = prior.shape
    xy = _trajectory_xy_by_candidate(
        np.asarray(table["prior_trajectory_xy"], dtype=np.float64),
        n_candidates=n_candidates,
        n_trajectories=n_traj,
        n_time=n_time,
    )
    z = project_response_delta(prior - zero[:, None, :, :], basis)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_index in range(n_candidates):
        for spec in FILTER_SPECS:
            row = {
                "trial_id": int(manifest_row["trial_id"]),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_family": str(manifest_row["prior_family"]),
                "prior_scale": float(manifest_row["scale"]),
                "candidate_index": int(candidate_index),
                "filter_model": spec.label,
                "filter_family": spec.family,
                "filter_parameter": float(spec.parameter),
                "n_trajectories": int(n_traj),
                "n_timebins": int(n_time),
                "basis_dim": int(basis.shape[1]),
                "ridge": float(ridge),
            }
            row.update(
                _candidate_metrics(
                    z=z[candidate_index],
                    xy=xy[candidate_index],
                    spec=spec,
                    n_folds=n_folds,
                    ridge=ridge,
                )
            )
            candidate_rows.append(row)

    frame = pd.DataFrame(candidate_rows)
    table_rows = []
    for model, group in frame.groupby("filter_model", sort=False):
        spec = next(item for item in FILTER_SPECS if item.label == model)
        table_rows.append(
            {
                "trial_id": int(manifest_row["trial_id"]),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_family": str(manifest_row["prior_family"]),
                "prior_scale": float(manifest_row["scale"]),
                "filter_model": str(model),
                "filter_family": spec.family,
                "filter_parameter": float(spec.parameter),
                "n_candidates": int(n_candidates),
                "n_trajectories": int(n_traj),
                "n_timebins": int(n_time),
                "basis_dim": int(basis.shape[1]),
                "ridge": float(ridge),
                "mean_cv_r2_energy": float(group["cv_r2_energy"].mean()),
                "median_cv_r2_energy": float(group["cv_r2_energy"].median()),
                "mean_train_r2_energy": float(group["train_r2_energy"].mean()),
                "median_coef_s2_over_s1": float(group["coef_s2_over_s1"].median()),
            }
        )
    return candidate_rows, table_rows


def _summarize(table_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        table_rows.groupby(
            ["filter_model", "filter_family", "filter_parameter", "prior_scale"],
            as_index=False,
            sort=False,
        )
        .agg(
            n_tables=("mean_cv_r2_energy", "size"),
            mean_cv_r2_energy=("mean_cv_r2_energy", "mean"),
            median_cv_r2_energy=("mean_cv_r2_energy", "median"),
            mean_train_r2_energy=("mean_train_r2_energy", "mean"),
            median_coef_s2_over_s1=("median_coef_s2_over_s1", "median"),
        )
    )


def _overall_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        summary.assign(
            weighted_cv=lambda df: df["mean_cv_r2_energy"] * df["n_tables"].astype(float),
            weighted_train=lambda df: df["mean_train_r2_energy"] * df["n_tables"].astype(float),
        )
        .groupby(["filter_model", "filter_family", "filter_parameter"], as_index=False, sort=False)
        .agg(
            n_tables=("n_tables", "sum"),
            weighted_cv=("weighted_cv", "sum"),
            weighted_train=("weighted_train", "sum"),
            median_coef_s2_over_s1=("median_coef_s2_over_s1", "median"),
        )
    )
    out["mean_cv_r2_energy"] = out["weighted_cv"] / out["n_tables"].astype(float)
    out["mean_train_r2_energy"] = out["weighted_train"] / out["n_tables"].astype(float)
    return out.drop(columns=["weighted_cv", "weighted_train"])


def _plot(summary: pd.DataFrame, overall: pd.DataFrame, suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), constrained_layout=True)
    x = np.arange(len(PLOT_ORDER), dtype=float)
    block = overall.set_index("filter_model").reindex(PLOT_ORDER)
    axes[0].plot(x, block["mean_cv_r2_energy"], marker="o", color="#235789", lw=1.8)
    axes[0].axhline(float(block.loc["instant", "mean_cv_r2_energy"]), color="#6b7280", lw=1.0, ls="--")
    axes[0].set_title("Trajectory-held-out prediction")
    axes[0].set_ylabel("compact residual CV R2")

    axes[1].plot(x, block["median_coef_s2_over_s1"], marker="o", color="#2f8f6a", lw=1.8)
    axes[1].set_title("2D coefficient anisotropy")
    axes[1].set_ylabel("s2 / s1")

    colors = {0.5: "#235789", 1.0: "#b35c2e", 2.0: "#2f8f6a"}
    for scale, group in summary.groupby("prior_scale", sort=True):
        scale_block = group.set_index("filter_model").reindex(PLOT_ORDER)
        axes[2].plot(
            x,
            scale_block["mean_cv_r2_energy"],
            marker="o",
            lw=1.5,
            color=colors.get(float(scale), "#4b5563"),
            label=f"{float(scale):g}x",
        )
    axes[2].set_title("CV R2 by motion scale")
    axes[2].set_ylabel("compact residual CV R2")
    axes[2].legend(frameon=False)

    labels = [PLOT_LABELS[name] for name in PLOT_ORDER]
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel("causal temporal filter")
        _clean_axis(ax)
    fig.suptitle("Constrained temporal-filter observation diagnostic")
    fig.savefig(OUT_DIR / f"continuous_joint_temporal_filter_observation_diagnostic{suffix}.png", dpi=220)
    fig.savefig(OUT_DIR / f"continuous_joint_temporal_filter_observation_diagnostic{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tables", type=int, default=0, help="0 means all manifest rows.")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--basis-max-dim", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    manifest = pd.read_csv(MANIFEST)
    if int(args.max_tables) > 0:
        manifest = manifest.head(int(args.max_tables)).copy()

    candidate_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    basis_meta: dict[str, Any] | None = None
    for row_index, row in manifest.iterrows():
        table_path = SOURCE_ROOT / str(row["response_cache_path"])
        sidecar_path = SIDECAR_ROOT / str(row["response_cache_path"])
        table = {**_load_npz(table_path), **_load_npz(sidecar_path)}
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        _n_candidates, _n_traj, _n_time, n_units = prior.shape
        basis, basis_meta = _load_basis(n_units, int(args.basis_max_dim))
        cand, tab = _table_rows(
            table=table,
            manifest_row=row,
            basis=basis,
            n_folds=int(args.n_folds),
            ridge=float(args.ridge),
        )
        candidate_rows.extend(cand)
        table_rows.extend(tab)
        if (len(table_rows) // len(FILTER_SPECS)) % 50 == 0:
            print(f"processed {row_index + 1}/{len(manifest)} tables", flush=True)

    suffix = str(args.suffix)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    candidate_df = pd.DataFrame(candidate_rows)
    table_df = pd.DataFrame(table_rows)
    summary = _summarize(table_df)
    overall = _overall_summary(summary)
    candidate_df.to_csv(OUT_DIR / f"continuous_joint_temporal_filter_candidate_rows{suffix}.csv", index=False)
    table_df.to_csv(OUT_DIR / f"continuous_joint_temporal_filter_table_rows{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"continuous_joint_temporal_filter_summary{suffix}.csv", index=False)
    overall.to_csv(OUT_DIR / f"continuous_joint_temporal_filter_overall{suffix}.csv", index=False)
    _plot(summary, overall, suffix)

    best = overall.sort_values("mean_cv_r2_energy", ascending=False).iloc[0]
    readme = [
        "# Constrained Temporal-Filter Observation Diagnostic",
        "",
        "This diagnostic compares instantaneous eye position against causal delay, boxcar, and EMA filters that keep the eye regressor two-dimensional.",
        "Fits are evaluated with trajectory-held-out folds within each candidate image.",
        "",
        f"Basis: `{(basis_meta or {}).get('basis_source', '')}`",
        f"Basis dim: {(basis_meta or {}).get('basis_dim', '')}",
        f"Manifest rows: {len(manifest)}",
        f"Best filter by mean CV R2: {best['filter_model']} ({best['mean_cv_r2_energy']:.6g})",
        "",
        "Overall:",
        "",
        overall.to_string(index=False),
        "",
    ]
    (OUT_DIR / f"continuous_joint_temporal_filter_README{suffix}.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

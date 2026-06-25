"""Diagnose whether lagged eye-position features improve the 4C encoder.

The active no-anchor continuous-joint model uses an instantaneous local
observation model, ``z_t ~= A_I(t) tau_t``. This script asks a narrower
principled question before changing the estimator: do causal lagged trajectory
features predict compact response residuals better, and do they make the local
map less rank-1?

The fit is evaluated by trajectory-held-out folds inside each response table,
so gains should reflect model structure rather than simply adding parameters.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    _as_basis,
    _load_npz,
    _observation_matrix_condition_metrics,
    _trajectory_xy_by_candidate,
    project_response_delta,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
MANIFEST = SOURCE_ROOT / "response_cache_manifest.csv"
SIDECAR_ROOT = SOURCE_ROOT / "continuous_joint_trajectory_sidecars"
BASIS_PATH = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_disjoint_compact_basis_delta025_v1"
    / "image_disjoint_compact_basis_delta0p25_fold0of2.npz"
)
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
)


@dataclass(frozen=True)
class LagSpec:
    label: str
    lags: tuple[int, ...]


LAG_SPECS = [
    LagSpec("instant", (0,)),
    LagSpec("lag0_1", (0, 1)),
    LagSpec("lag0_1_2", (0, 1, 2)),
    LagSpec("lag0_1_2_4", (0, 1, 2, 4)),
    LagSpec("lag0_1_2_4_8", (0, 1, 2, 4, 8)),
]
LAG_LABELS = {
    "instant": "0",
    "lag0_1": "0,1",
    "lag0_1_2": "0,1,2",
    "lag0_1_2_4": "0,1,2,4",
    "lag0_1_2_4_8": "0,1,2,4,8",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_basis(n_units: int, basis_max_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(BASIS_PATH) as data:
        key = "U" if "U" in data.files else "basis"
        if key not in data.files:
            key = next((name for name in data.files if str(name).startswith("basis")), "")
        if not key:
            raise ValueError(f"No basis-like array in {BASIS_PATH}; keys={data.files}")
        basis = np.asarray(data[key], dtype=np.float64)
    basis = _as_basis(basis, n_units)
    if int(basis_max_dim) > 0:
        basis = basis[:, : min(int(basis_max_dim), basis.shape[1])]
    return basis, {"basis_source": str(BASIS_PATH), "basis_key": key, "basis_dim": int(basis.shape[1])}


def _lagged_design(xy: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"xy must be (trajectory,time,2), got {arr.shape}")
    n_traj, n_time, _ = arr.shape
    design = np.empty((n_traj, n_time, 2 * len(lags)), dtype=np.float64)
    time = np.arange(n_time)
    for lag_index, lag in enumerate(lags):
        source_time = np.maximum(0, time - int(lag))
        design[:, :, 2 * lag_index : 2 * lag_index + 2] = arr[:, source_time, :]
    return design


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    normal = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(normal, x.T @ y)


def _energy_r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(y * y))
    if denom <= 1e-12:
        return float("nan")
    resid = y - pred
    return 1.0 - float(np.sum(resid * resid)) / denom


def _candidate_metrics(
    *,
    z: np.ndarray,
    xy: np.ndarray,
    lag_spec: LagSpec,
    n_folds: int,
    ridge: float,
) -> dict[str, float]:
    design = _lagged_design(xy, lag_spec.lags)
    n_traj, n_time, feature_dim = design.shape
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
    s3 = float(singular[2]) if singular.size > 2 else float("nan")
    return {
        "train_r2_energy": float(train_r2),
        "cv_r2_energy": float(cv_r2),
        "mean_fold_r2_energy": float(np.nanmean(fold_r2)) if fold_r2 else float("nan"),
        "feature_dim": int(feature_dim),
        "coef_singular1": s1,
        "coef_singular2": s2,
        "coef_singular3": s3,
        "coef_s2_over_s1": float(s2 / max(s1, 1e-12)) if np.isfinite(s1) and np.isfinite(s2) else float("nan"),
        "coef_s3_over_s1": float(s3 / max(s1, 1e-12)) if np.isfinite(s1) and np.isfinite(s3) else float("nan"),
    }


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
    n_candidates, n_traj, n_time, n_units = prior.shape
    xy = _trajectory_xy_by_candidate(
        np.asarray(table["prior_trajectory_xy"], dtype=np.float64),
        n_candidates=n_candidates,
        n_trajectories=n_traj,
        n_time=n_time,
    )
    z = project_response_delta(prior - zero[:, None, :, :], basis)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_index in range(n_candidates):
        for spec in LAG_SPECS:
            row = {
                "trial_id": int(manifest_row["trial_id"]),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_family": str(manifest_row["prior_family"]),
                "prior_scale": float(manifest_row["scale"]),
                "candidate_index": int(candidate_index),
                "lag_model": spec.label,
                "lags": ",".join(str(v) for v in spec.lags),
                "n_lags": int(len(spec.lags)),
                "n_trajectories": int(n_traj),
                "n_timebins": int(n_time),
                "basis_dim": int(basis.shape[1]),
                "ridge": float(ridge),
            }
            row.update(
                _candidate_metrics(
                    z=z[candidate_index],
                    xy=xy[candidate_index],
                    lag_spec=spec,
                    n_folds=n_folds,
                    ridge=ridge,
                )
            )
            if spec.lags == (0,):
                # Keep the existing 2D condition metrics for direct comparison
                # with the current instantaneous estimator diagnostics.
                design = _lagged_design(xy[candidate_index], spec.lags).reshape(-1, 2)
                target = z[candidate_index].reshape(-1, basis.shape[1])
                coef = _ridge_fit(design, target, ridge).T
                row.update(_observation_matrix_condition_metrics(coef))
            candidate_rows.append(row)

    table_rows = []
    frame = pd.DataFrame(candidate_rows)
    for lag_model, group in frame.groupby("lag_model", sort=False):
        spec = next(item for item in LAG_SPECS if item.label == lag_model)
        base = {
            "trial_id": int(manifest_row["trial_id"]),
            "response_cache_path": str(manifest_row["response_cache_path"]),
            "prior_family": str(manifest_row["prior_family"]),
            "prior_scale": float(manifest_row["scale"]),
            "lag_model": str(lag_model),
            "lags": ",".join(str(v) for v in spec.lags),
            "n_lags": int(len(spec.lags)),
            "n_candidates": int(n_candidates),
            "n_trajectories": int(n_traj),
            "n_timebins": int(n_time),
            "basis_dim": int(basis.shape[1]),
            "ridge": float(ridge),
        }
        base.update(
            {
                "mean_cv_r2_energy": float(group["cv_r2_energy"].mean()),
                "median_cv_r2_energy": float(group["cv_r2_energy"].median()),
                "mean_train_r2_energy": float(group["train_r2_energy"].mean()),
                "median_coef_s2_over_s1": float(group["coef_s2_over_s1"].median()),
                "median_coef_s3_over_s1": float(group["coef_s3_over_s1"].median()),
            }
        )
        table_rows.append(base)
    return candidate_rows, table_rows


def _summarize(table_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        table_rows.groupby(["lag_model", "lags", "n_lags", "prior_scale"], as_index=False, sort=False)
        .agg(
            n_tables=("mean_cv_r2_energy", "size"),
            mean_cv_r2_energy=("mean_cv_r2_energy", "mean"),
            median_cv_r2_energy=("mean_cv_r2_energy", "median"),
            mean_train_r2_energy=("mean_train_r2_energy", "mean"),
            median_coef_s2_over_s1=("median_coef_s2_over_s1", "median"),
            median_coef_s3_over_s1=("median_coef_s3_over_s1", "median"),
        )
    )


def _overall_summary(summary: pd.DataFrame) -> pd.DataFrame:
    weights = summary["n_tables"].astype(float)
    out = (
        summary.assign(
            weighted_cv=lambda df: df["mean_cv_r2_energy"] * df["n_tables"].astype(float),
            weighted_train=lambda df: df["mean_train_r2_energy"] * df["n_tables"].astype(float),
        )
        .groupby(["lag_model", "lags", "n_lags"], as_index=False, sort=False)
        .agg(
            n_tables=("n_tables", "sum"),
            weighted_cv=("weighted_cv", "sum"),
            weighted_train=("weighted_train", "sum"),
            median_coef_s2_over_s1=("median_coef_s2_over_s1", "median"),
            median_coef_s3_over_s1=("median_coef_s3_over_s1", "median"),
        )
    )
    out["mean_cv_r2_energy"] = out["weighted_cv"] / out["n_tables"].astype(float)
    out["mean_train_r2_energy"] = out["weighted_train"] / out["n_tables"].astype(float)
    return out.drop(columns=["weighted_cv", "weighted_train"])


def _plot(summary: pd.DataFrame, overall: pd.DataFrame, suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    order = [spec.label for spec in LAG_SPECS]
    x = np.arange(len(order), dtype=float)
    block = overall.set_index("lag_model").reindex(order)
    axes[0].plot(x, block["mean_cv_r2_energy"], marker="o", color="#235789", lw=1.8)
    axes[0].set_title("Trajectory-held-out prediction")
    axes[0].set_ylabel("compact residual CV R2")
    axes[0].set_xticks(x, [LAG_LABELS[name] for name in order], rotation=25, ha="right")

    axes[1].plot(x, block["median_coef_s2_over_s1"], marker="o", color="#2f8f6a", lw=1.8, label="s2/s1")
    axes[1].plot(x, block["median_coef_s3_over_s1"], marker="s", color="#8a5ca8", lw=1.5, label="s3/s1")
    axes[1].set_title("Coefficient spectrum")
    axes[1].set_ylabel("singular ratio")
    axes[1].set_xticks(x, [LAG_LABELS[name] for name in order], rotation=25, ha="right")
    axes[1].legend(frameon=False)

    colors = {0.5: "#235789", 1.0: "#b35c2e", 2.0: "#2f8f6a"}
    for scale, group in summary.groupby("prior_scale", sort=True):
        scale_block = group.set_index("lag_model").reindex(order)
        axes[2].plot(
            x,
            scale_block["mean_cv_r2_energy"],
            marker="o",
            lw=1.6,
            color=colors.get(float(scale), "#4b5563"),
            label=f"{float(scale):g}x",
        )
    axes[2].set_title("CV R2 by motion scale")
    axes[2].set_ylabel("compact residual CV R2")
    axes[2].set_xticks(x, [LAG_LABELS[name] for name in order], rotation=25, ha="right")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("causal lags")
        _clean_axis(ax)
    fig.suptitle("Lagged local observation model diagnostic")
    fig.savefig(OUT_DIR / f"continuous_joint_lagged_observation_diagnostic{suffix}.png", dpi=220)
    fig.savefig(OUT_DIR / f"continuous_joint_lagged_observation_diagnostic{suffix}.pdf")
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
        table = _load_npz(table_path)
        sidecar = _load_npz(sidecar_path)
        table = {**table, **sidecar}
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
        if (len(table_rows) // len(LAG_SPECS)) % 50 == 0:
            print(f"processed {row_index + 1}/{len(manifest)} tables", flush=True)

    suffix = str(args.suffix)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    candidate_df = pd.DataFrame(candidate_rows)
    table_df = pd.DataFrame(table_rows)
    summary = _summarize(table_df)
    overall = _overall_summary(summary)
    candidate_df.to_csv(OUT_DIR / f"continuous_joint_lagged_observation_candidate_rows{suffix}.csv", index=False)
    table_df.to_csv(OUT_DIR / f"continuous_joint_lagged_observation_table_rows{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"continuous_joint_lagged_observation_summary{suffix}.csv", index=False)
    overall.to_csv(OUT_DIR / f"continuous_joint_lagged_observation_overall{suffix}.csv", index=False)
    _plot(summary, overall, suffix)

    readme = [
        "# Lagged Observation Diagnostic",
        "",
        "This diagnostic compares the current instantaneous compact response model against causal lagged eye-position designs.",
        "Fits are evaluated with trajectory-held-out folds within each candidate image.",
        "",
        f"Basis: `{(basis_meta or {}).get('basis_source', '')}`",
        f"Basis dim: {(basis_meta or {}).get('basis_dim', '')}",
        f"Manifest rows: {len(manifest)}",
        "",
        "Overall:",
        "",
        overall.to_string(index=False),
        "",
    ]
    (OUT_DIR / f"continuous_joint_lagged_observation_README{suffix}.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Polynomial local-observation diagnostic for Figure 4C.

The current no-anchor estimator uses an origin-constrained linear map from a
2D eye position to compact response residuals. This script asks whether the
response manifold has useful curvature by adding polynomial terms while keeping
the underlying eye state two-dimensional.

All scores are trajectory-held-out within each candidate image. The affine
models are included as diagnostics for a possible response/coordinate offset;
the origin-constrained models are the principled candidates for a nonlinear
joint estimator.
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
class PolySpec:
    label: str
    degree: int
    include_intercept: bool = False


POLY_SPECS = [
    PolySpec("linear", 1, False),
    PolySpec("quadratic", 2, False),
    PolySpec("cubic", 3, False),
    PolySpec("affine_linear", 1, True),
    PolySpec("affine_quadratic", 2, True),
]
PLOT_ORDER = [spec.label for spec in POLY_SPECS]
PLOT_LABELS = {
    "linear": "linear",
    "quadratic": "quad",
    "cubic": "cubic",
    "affine_linear": "affine\nlinear",
    "affine_quadratic": "affine\nquad",
}


def _poly_design(xy: np.ndarray, spec: PolySpec) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"xy must be (trajectory,time,2), got {arr.shape}")
    x = arr[:, :, 0]
    y = arr[:, :, 1]
    terms = []
    if spec.include_intercept:
        terms.append(np.ones_like(x))
    terms.extend([x, y])
    if spec.degree >= 2:
        terms.extend([x * x, x * y, y * y])
    if spec.degree >= 3:
        terms.extend([x * x * x, x * x * y, x * y * y, y * y * y])
    if spec.degree > 3:
        raise ValueError("Only polynomial degrees up to 3 are implemented")
    return np.stack(terms, axis=2)


def _candidate_metrics(
    *,
    z: np.ndarray,
    xy: np.ndarray,
    spec: PolySpec,
    n_folds: int,
    ridge: float,
) -> dict[str, float]:
    design = _poly_design(xy, spec)
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
    row = {
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
    if spec.label == "linear":
        row.update(_observation_matrix_condition_metrics(coef_matrix))
    return row


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
        for spec in POLY_SPECS:
            row = {
                "trial_id": int(manifest_row["trial_id"]),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_family": str(manifest_row["prior_family"]),
                "prior_scale": float(manifest_row["scale"]),
                "candidate_index": int(candidate_index),
                "poly_model": spec.label,
                "degree": int(spec.degree),
                "include_intercept": bool(spec.include_intercept),
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
    for model, group in frame.groupby("poly_model", sort=False):
        spec = next(item for item in POLY_SPECS if item.label == model)
        table_rows.append(
            {
                "trial_id": int(manifest_row["trial_id"]),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_family": str(manifest_row["prior_family"]),
                "prior_scale": float(manifest_row["scale"]),
                "poly_model": str(model),
                "degree": int(spec.degree),
                "include_intercept": bool(spec.include_intercept),
                "n_candidates": int(n_candidates),
                "n_trajectories": int(n_traj),
                "n_timebins": int(n_time),
                "basis_dim": int(basis.shape[1]),
                "ridge": float(ridge),
                "mean_cv_r2_energy": float(group["cv_r2_energy"].mean()),
                "median_cv_r2_energy": float(group["cv_r2_energy"].median()),
                "mean_train_r2_energy": float(group["train_r2_energy"].mean()),
                "median_coef_s2_over_s1": float(group["coef_s2_over_s1"].median()),
                "median_coef_s3_over_s1": float(group["coef_s3_over_s1"].median()),
            }
        )
    return candidate_rows, table_rows


def _summarize(table_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        table_rows.groupby(
            ["poly_model", "degree", "include_intercept", "prior_scale"],
            as_index=False,
            sort=False,
        )
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
    out = (
        summary.assign(
            weighted_cv=lambda df: df["mean_cv_r2_energy"] * df["n_tables"].astype(float),
            weighted_train=lambda df: df["mean_train_r2_energy"] * df["n_tables"].astype(float),
        )
        .groupby(["poly_model", "degree", "include_intercept"], as_index=False, sort=False)
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
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), constrained_layout=True)
    x = np.arange(len(PLOT_ORDER), dtype=float)
    block = overall.set_index("poly_model").reindex(PLOT_ORDER)
    axes[0].bar(x, block["mean_cv_r2_energy"], color=["#235789", "#2f8f6a", "#8a5ca8", "#b35c2e", "#6b7280"])
    axes[0].set_title("Trajectory-held-out prediction")
    axes[0].set_ylabel("compact residual CV R2")

    axes[1].plot(x, block["mean_train_r2_energy"], marker="o", color="#b35c2e", lw=1.8)
    axes[1].set_title("Training fit")
    axes[1].set_ylabel("compact residual train R2")

    colors = {0.5: "#235789", 1.0: "#b35c2e", 2.0: "#2f8f6a"}
    for scale, group in summary.groupby("prior_scale", sort=True):
        scale_block = group.set_index("poly_model").reindex(PLOT_ORDER)
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
        ax.set_xticks(x, labels, rotation=25, ha="right")
        _clean_axis(ax)
    fig.suptitle("Polynomial local observation model diagnostic")
    fig.savefig(OUT_DIR / f"continuous_joint_polynomial_observation_diagnostic{suffix}.png", dpi=220)
    fig.savefig(OUT_DIR / f"continuous_joint_polynomial_observation_diagnostic{suffix}.pdf")
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
        if (len(table_rows) // len(POLY_SPECS)) % 50 == 0:
            print(f"processed {row_index + 1}/{len(manifest)} tables", flush=True)

    suffix = str(args.suffix)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    candidate_df = pd.DataFrame(candidate_rows)
    table_df = pd.DataFrame(table_rows)
    summary = _summarize(table_df)
    overall = _overall_summary(summary)
    candidate_df.to_csv(OUT_DIR / f"continuous_joint_polynomial_candidate_rows{suffix}.csv", index=False)
    table_df.to_csv(OUT_DIR / f"continuous_joint_polynomial_table_rows{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"continuous_joint_polynomial_summary{suffix}.csv", index=False)
    overall.to_csv(OUT_DIR / f"continuous_joint_polynomial_overall{suffix}.csv", index=False)
    _plot(summary, overall, suffix)

    best = overall.sort_values("mean_cv_r2_energy", ascending=False).iloc[0]
    readme = [
        "# Polynomial Observation Diagnostic",
        "",
        "This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.",
        "Fits are evaluated with trajectory-held-out folds within each candidate image.",
        "",
        f"Basis: `{(basis_meta or {}).get('basis_source', '')}`",
        f"Basis dim: {(basis_meta or {}).get('basis_dim', '')}",
        f"Manifest rows: {len(manifest)}",
        f"Best model by mean CV R2: {best['poly_model']} ({best['mean_cv_r2_energy']:.6g})",
        "",
        "Overall:",
        "",
        overall.to_string(index=False),
        "",
    ]
    (OUT_DIR / f"continuous_joint_polynomial_README{suffix}.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

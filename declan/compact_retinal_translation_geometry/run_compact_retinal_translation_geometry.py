#!/usr/bin/env python3
"""Build compact retinal-translation geometry panel data and figures.

This module intentionally lives outside ``declan/fig4_cov_TFTS``.  It reuses
the production tangent-structure and finite-difference closure artifacts, but
writes the new compact-geometry output contract described in
``declan/compact_retinal_translation_geometry_implementation_spec.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TFTS_ROOT = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
DEFAULT_CLOSURE_ROOT = (
    VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_rf_null_step025_rfbacked_v2"
)
DEFAULT_OUT_ROOT = VISIONCORE_ROOT / "outputs" / "compact_retinal_translation_geometry"

TEXT = "#202124"
MODEL = "#2f5f9f"
MODEL_LIGHT = "#d8e6f5"
BRIDGE = "#7b5ea7"
BRIDGE_LIGHT = "#e8ddf3"
NULL = "#8d8d8d"
ACCENT = "#c44e52"
GREEN = "#4d8f62"


@dataclass(frozen=True)
class SourcePaths:
    tfts_root: Path
    closure_root: Path
    out_root: Path

    @property
    def figures_dir(self) -> Path:
        return self.out_root / "figures"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _clean_axes(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.18, linewidth=0.7, zorder=-2)


def _finite_float(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _nearest_delta(available: list[float], requested: float) -> float:
    if not available:
        raise ValueError("No finite-difference delta values are available.")
    return float(min(available, key=lambda d: abs(float(d) - float(requested))))


def _load_tangent_maps(tfts_root: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]]]:
    path = tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing tangent map cache: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    deltas = [float(v) for v in payload["delta_arcmins"]]
    object_payload = {
        float(delta): {str(oid): meta for oid, meta in objects.items()}
        for delta, objects in payload["object_payload"].items()
    }
    return deltas, object_payload


def _pca_basis(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(x, axis=0)
    xc = x - mean[None, :]
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:n_components].T
    return mean, basis


def _project(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean[None, :]) @ basis


def _panel_a(paths: SourcePaths, primary_delta: float, n_contexts: int, seed: int) -> dict[str, Any]:
    deltas, payload_by_delta = _load_tangent_maps(paths.tfts_root)
    delta = _nearest_delta(deltas, primary_delta)
    payload = payload_by_delta[delta]
    object_ids_all = []
    for oid, meta in sorted(payload.items()):
        arrays = [np.asarray(meta[name], dtype=np.float64) for name in ("r0", "bx", "by")]
        if all(arr.ndim == 1 and np.all(np.isfinite(arr)) for arr in arrays):
            object_ids_all.append(str(oid))
    if not object_ids_all:
        raise ValueError(f"No finite tangent-map objects found for delta={delta}.")

    rng = np.random.default_rng(int(seed))
    n_pick = min(int(n_contexts), len(object_ids_all))
    object_ids = sorted(rng.choice(object_ids_all, size=n_pick, replace=False).tolist())

    r0 = np.stack([np.asarray(payload[oid]["r0"], dtype=np.float64) for oid in object_ids], axis=0)
    bx = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=0)
    by = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=0)
    pca_input = np.concatenate([r0, r0 + bx, r0 - bx, r0 + by, r0 - by], axis=0)
    mean, basis = _pca_basis(pca_input, n_components=2)
    np.save(paths.out_root / "panelA_projection_basis.npy", basis)
    np.save(paths.out_root / "panelA_projection_mean.npy", mean)

    z0 = _project(r0, mean, basis)
    dbx = bx @ basis
    dby = by @ basis
    base_span = float(np.median(np.linalg.norm(z0 - np.median(z0, axis=0, keepdims=True), axis=1)))
    tangent_norm = float(np.median(np.r_[np.linalg.norm(dbx, axis=1), np.linalg.norm(dby, axis=1)]))
    display_scale = 0.28 * base_span / max(tangent_norm, 1e-12)
    if not np.isfinite(display_scale) or display_scale <= 0:
        display_scale = 1.0

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for i, oid in enumerate(object_ids):
        meta = payload[oid]
        row = {
            "object_id": oid,
            "finite_difference_step_arcmin": delta,
            "image_id": int(meta.get("image_id", -1)),
            "trial_index": int(meta.get("trial_index", -1)),
            "time_index": int(meta.get("time_index", -1)),
            "r0_pc1": float(z0[i, 0]),
            "r0_pc2": float(z0[i, 1]),
            "bx_pc1_display": float(dbx[i, 0] * display_scale),
            "bx_pc2_display": float(dbx[i, 1] * display_scale),
            "by_pc1_display": float(dby[i, 0] * display_scale),
            "by_pc2_display": float(dby[i, 1] * display_scale),
            "bx_norm_population": float(np.linalg.norm(bx[i])),
            "by_norm_population": float(np.linalg.norm(by[i])),
            "display_arrow_scale": float(display_scale),
        }
        rows.append(row)
        selected_rows.append(
            {
                "object_id": oid,
                "finite_difference_step_arcmin": delta,
                "image_id": row["image_id"],
                "trial_index": row["trial_index"],
                "time_index": row["time_index"],
            }
        )
    _write_csv(paths.out_root / "panelA_local_charts.csv", rows)
    _write_csv(paths.out_root / "panelA_selected_contexts.csv", selected_rows)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(z0[:, 0], z0[:, 1], s=16, color=TEXT, alpha=0.75, linewidths=0)
    for i in range(len(rows)):
        ax.arrow(
            z0[i, 0],
            z0[i, 1],
            dbx[i, 0] * display_scale,
            dbx[i, 1] * display_scale,
            color=MODEL,
            alpha=0.72,
            width=0.0,
            head_width=0.035 * max(base_span, 1e-6),
            length_includes_head=True,
            linewidth=0.75,
        )
        ax.arrow(
            z0[i, 0],
            z0[i, 1],
            dby[i, 0] * display_scale,
            dby[i, 1] * display_scale,
            color=ACCENT,
            alpha=0.72,
            width=0.0,
            head_width=0.035 * max(base_span, 1e-6),
            length_includes_head=True,
            linewidth=0.75,
        )
    ax.set_title("A. Image-dependent local translation charts", loc="left", fontweight="bold")
    ax.set_xlabel("population PC 1")
    ax.set_ylabel("population PC 2")
    _clean_axes(ax, grid=True)
    ax.legend(
        [
            plt.Line2D([0], [0], color=MODEL, lw=1.6),
            plt.Line2D([0], [0], color=ACCENT, lw=1.6),
        ],
        ["horizontal tangent", "vertical tangent"],
        frameon=False,
        loc="best",
    )
    fig.tight_layout()
    _save_fig(fig, paths.figures_dir / "panelA_local_translation_charts")
    return {
        "delta_arcmin": delta,
        "n_contexts": len(object_ids),
        "n_units": int(r0.shape[1]),
        "source": str((paths.tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").resolve()),
    }


def _panel_b(paths: SourcePaths, primary_delta: float, n_units: int) -> dict[str, Any]:
    spectrum_path = paths.tfts_root / "union_spectrum" / "twin_tangent_union_spectrum.csv"
    summary_path = paths.tfts_root / "union_spectrum" / "twin_tangent_union_summary.csv"
    null_summary_path = paths.tfts_root / "union_spectrum" / "twin_tangent_union_null_spectrum_summary.csv"
    for path in (spectrum_path, summary_path, null_summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing tangent spectrum artifact: {path}")

    spec = pd.read_csv(spectrum_path)
    summary = pd.read_csv(summary_path)
    null = pd.read_csv(null_summary_path)
    delta = _nearest_delta(sorted(float(v) for v in spec["delta"].dropna().unique()), primary_delta)
    obs = spec[
        (np.isclose(spec["delta"].astype(float), delta))
        & (spec["space"].astype(str) == "raw")
        & (spec["tangent_set"].astype(str) == "combined")
    ].copy()
    summ = summary[
        (np.isclose(summary["delta"].astype(float), delta))
        & (summary["space"].astype(str) == "raw")
        & (summary["component_set"].astype(str) == "combined")
    ].copy()
    n_contexts = int(summ["n_objects"].iloc[0]) if len(summ) and "n_objects" in summ else -1

    rows = [
        {
            "session_id": "pooled_twin_contexts",
            "projection_control": "none",
            "finite_difference_step_arcmin": float(row.delta),
            "null_type": "observed",
            "null_draw": 0,
            "rank": int(row.component_index),
            "eigenvalue": float(row.eigenvalue),
            "cumulative_variance": float(row.cumulative_fraction_variance),
            "participation_ratio": float(row.participation_ratio),
            "n_units": int(n_units),
            "n_contexts": n_contexts,
        }
        for row in obs.itertuples(index=False)
    ]
    _write_csv(paths.out_root / "panelB_tangent_spectrum.csv", rows)

    summary_rows = [
        {
            "session_id": "pooled_twin_contexts",
            "projection_control": "none",
            "finite_difference_step_arcmin": float(row.delta),
            "null_type": "observed",
            "participation_ratio": float(row.participation_ratio),
            "n_dims_50pct": int(row.n_dims_50pct),
            "n_dims_80pct": int(row.n_dims_80pct),
            "n_dims_90pct": int(row.n_dims_90pct),
            "n_dims_95pct": int(row.n_dims_95pct),
            "null_pr_mean": _finite_float(row.null_pr_mean),
            "null_pr_ci_low": _finite_float(row.null_pr_ci_low),
            "null_pr_ci_high": _finite_float(row.null_pr_ci_high),
            "n_units": int(n_units),
            "n_contexts": int(row.n_objects),
        }
        for row in summary.itertuples(index=False)
    ]
    _write_csv(paths.out_root / "panelB_participation_ratio_summary.csv", summary_rows)

    null_rows = []
    for row in null.itertuples(index=False):
        null_rows.append(
            {
                "session_id": "pooled_twin_contexts",
                "projection_control": "none",
                "finite_difference_step_arcmin": float(row.delta),
                "null_type": "samplewise_unit_shuffle",
                "rank": int(row.component_index),
                "n_null_draws": int(row.n_null_repeats),
                "cumulative_variance_median": float(row.cumvar_median),
                "cumulative_variance_ci_low": float(row.cumvar_ci_low),
                "cumulative_variance_ci_high": float(row.cumvar_ci_high),
                "fraction_variance_median": float(row.fracvar_median),
                "fraction_variance_ci_low": float(row.fracvar_ci_low),
                "fraction_variance_ci_high": float(row.fracvar_ci_high),
                "n_units": int(n_units),
                "n_contexts": n_contexts,
            }
        )
    _write_csv(paths.out_root / "panelB_null_spectra.csv", null_rows)

    obs_plot = obs.head(64)
    null_plot = null[
        (np.isclose(null["delta"].astype(float), delta))
        & (null["space"].astype(str) == "raw")
        & (null["tangent_set"].astype(str) == "combined")
    ].head(64)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.fill_between(
        null_plot["component_index"],
        null_plot["cumvar_ci_low"],
        null_plot["cumvar_ci_high"],
        color=NULL,
        alpha=0.22,
        linewidth=0,
        label="unit-shuffle 95% band",
    )
    ax.plot(null_plot["component_index"], null_plot["cumvar_median"], color=NULL, lw=1.4)
    ax.plot(
        obs_plot["component_index"],
        obs_plot["cumulative_fraction_variance"],
        color=MODEL,
        lw=2.0,
        label="observed",
    )
    ax.set_title("B. Compact tangent spectrum", loc="left", fontweight="bold")
    ax.set_xlabel("tangent component rank")
    ax.set_ylabel("cumulative tangent variance")
    ax.set_ylim(0, 1.02)
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, paths.figures_dir / "panelB_compact_tangent_spectrum")
    pr = float(summ["participation_ratio"].iloc[0]) if len(summ) else float("nan")
    null_pr = float(summ["null_pr_mean"].iloc[0]) if len(summ) else float("nan")
    return {
        "delta_arcmin": delta,
        "participation_ratio": pr,
        "unit_shuffle_null_pr_mean": null_pr,
        "n_contexts": n_contexts,
    }


def _panel_c(paths: SourcePaths, primary_delta: float) -> dict[str, Any]:
    basis_path = (
        paths.tfts_root
        / "split_modes"
        / "image_disjoint"
        / "twin_tangent_train_test_basis_image_disjoint.csv"
    )
    split_mode = "image_disjoint"
    if not basis_path.exists():
        basis_path = paths.tfts_root / "train_test_basis" / "twin_tangent_train_test_basis.csv"
        split_mode = "object_random_or_legacy"
    if not basis_path.exists():
        raise FileNotFoundError(f"Missing train/test basis artifact: {basis_path}")

    df = pd.read_csv(basis_path)
    delta = _nearest_delta(sorted(float(v) for v in df["delta"].dropna().unique()), primary_delta)
    block = df[
        (np.isclose(df["delta"].astype(float), delta))
        & (df["tangent_set"].astype(str) == "combined")
        & (df["fold_status"].astype(str) == "ok")
    ].copy()

    rows: list[dict[str, Any]] = []
    for row in block.itertuples(index=False):
        rows.append(
            {
                "session_id": "pooled_twin_contexts",
                "fold_id": int(row.fold),
                "projection_control": "none",
                "finite_difference_step_arcmin": float(row.delta),
                "k": int(row.basis_rank_k),
                "capture_observed": float(row.test_variance_captured),
                "null_type": "samplewise_unit_shuffle_train_basis",
                "null_draw": "summary",
                "capture_null": float(row.null_mean),
                "capture_null_ci_low": float(row.null_ci_low),
                "capture_null_ci_high": float(row.null_ci_high),
                "n_train_contexts": int(row.train_n_objects),
                "n_test_contexts": int(row.test_n_objects),
                "split_mode": getattr(row, "split_mode", split_mode),
            }
        )
    _write_csv(paths.out_root / "panelC_cross_image_generalization.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    for (k,), g in block.groupby(["basis_rank_k"]):
        obs = g["test_variance_captured"].to_numpy(dtype=np.float64)
        nul = g["null_mean"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "session_id": "pooled_twin_contexts",
                "projection_control": "none",
                "finite_difference_step_arcmin": delta,
                "k": int(k),
                "capture_observed_mean": float(np.nanmean(obs)),
                "capture_observed_ci_low": float(np.nanpercentile(obs, 2.5)),
                "capture_observed_ci_high": float(np.nanpercentile(obs, 97.5)),
                "capture_null_mean": float(np.nanmean(nul)),
                "capture_null_ci_low": float(np.nanpercentile(g["null_ci_low"].to_numpy(dtype=np.float64), 2.5)),
                "capture_null_ci_high": float(np.nanpercentile(g["null_ci_high"].to_numpy(dtype=np.float64), 97.5)),
                "n_folds": int(len(g)),
                "split_mode": str(g["split_mode"].iloc[0]) if "split_mode" in g else split_mode,
            }
        )
    summary_rows = sorted(summary_rows, key=lambda r: int(r["k"]))
    _write_csv(paths.out_root / "panelC_cross_image_generalization_summary.csv", summary_rows)

    ks = [int(r["k"]) for r in summary_rows]
    observed = [float(r["capture_observed_mean"]) for r in summary_rows]
    null_mean = [float(r["capture_null_mean"]) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(ks, observed, marker="o", color=MODEL, lw=2.0, label="observed")
    ax.plot(ks, null_mean, marker="o", color=NULL, lw=1.5, label="unit-shuffle")
    ax.set_title("C. Cross-image tangent generalization", loc="left", fontweight="bold")
    ax.set_xlabel("basis dimension k")
    ax.set_ylabel("held-out tangent variance captured")
    ax.set_ylim(0, max(0.8, float(np.nanmax(observed + null_mean)) * 1.12))
    ax.set_xticks(ks)
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save_fig(fig, paths.figures_dir / "panelC_cross_image_generalization")
    k10 = next((r for r in summary_rows if int(r["k"]) == 10), None)
    return {
        "delta_arcmin": delta,
        "split_mode": split_mode,
        "k10_capture_mean": None if k10 is None else float(k10["capture_observed_mean"]),
        "k10_null_mean": None if k10 is None else float(k10["capture_null_mean"]),
    }


def _read_closure_metrics(root: Path) -> pd.DataFrame:
    path = root / "finite_difference_capture_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing closure metrics: {path}")
    return pd.read_csv(path)


def _source_label(source: str) -> str:
    if source == "fd_sample_eye_trace_cov":
        return "full_finite_difference_source"
    if source == "fd_sample_eye_trace_xfit_compact_k10_cov":
        return "compact_k10_crossfit_source"
    return str(source)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if vals.size == 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    boot = np.mean(vals[idx], axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _stable_seed_offset(label: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(str(label))) % 100000)


def _closure_subset(
    metrics: pd.DataFrame,
    *,
    target_variant: str,
    projection_control: str,
    source_variants: set[str],
) -> pd.DataFrame:
    out = metrics[
        (metrics["row_status"].astype(str) == "ok")
        & (metrics["target_variant"].astype(str) == target_variant)
        & (metrics["projection_control"].astype(str) == projection_control)
        & (metrics["basis_source"].astype(str).isin(source_variants))
    ].copy()
    if out.empty:
        available = metrics[["target_variant", "projection_control", "basis_source", "k"]].drop_duplicates()
        raise ValueError(
            "No closure rows matched requested target/projection/source filters. "
            f"Available combinations include:\n{available.head(20).to_string(index=False)}"
        )
    return out


def _panel_e(
    paths: SourcePaths,
    *,
    target_variant: str,
    projection_control: str,
    seed: int,
) -> dict[str, Any]:
    metrics = _read_closure_metrics(paths.closure_root)
    source_variants = {"fd_sample_eye_trace_cov", "fd_sample_eye_trace_xfit_compact_k10_cov"}
    block = _closure_subset(
        metrics,
        target_variant=target_variant,
        projection_control=projection_control,
        source_variants=source_variants,
    )
    rows: list[dict[str, Any]] = []
    for row in block.itertuples(index=False):
        for null_type, null_col, effect_col in [
            ("unit_shuffle", "unit_shuffle_null_median", "effect_minus_unit_shuffle_median"),
            ("random_subspace", "random_subspace_null_median", "effect_minus_random_subspace_median"),
            (
                "rf_readout_fixed_permutation",
                "rf_fixed_permutation_null_median",
                "effect_minus_rf_fixed_permutation_median",
            ),
        ]:
            rows.append(
                {
                    "session_id": row.session,
                    "target_variant": row.target_variant,
                    "projection_control": row.projection_control,
                    "source_variant": _source_label(row.basis_source),
                    "basis_source": row.basis_source,
                    "k": int(row.k),
                    "capture_observed": float(row.capture),
                    "null_type": null_type,
                    "null_draw": "summary_median",
                    "capture_null": _finite_float(getattr(row, null_col, float("nan"))),
                    "excess_over_null": _finite_float(getattr(row, effect_col, float("nan"))),
                    "trace_target": float(row.target_trace),
                    "trace_source": float("nan"),
                    "n_units": int(row.n_common_units),
                    "n_samples": int(row.n_samples_used),
                    "rf_null_status": getattr(row, "rf_null_status", ""),
                    "rf_null_n_bins": getattr(row, "rf_null_n_bins", ""),
                    "rf_null_largest_bin_fraction": getattr(row, "rf_null_largest_bin_fraction", ""),
                }
            )
    _write_csv(paths.out_root / "panelE_covariance_closure_metrics.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    for (basis_source, k), g in block.groupby(["basis_source", "k"]):
        capture_mean, capture_lo, capture_hi = _bootstrap_mean_ci(
            g["capture"].to_numpy(dtype=np.float64),
            seed=int(seed) + int(k) * 17 + _stable_seed_offset(str(basis_source)),
        )
        for null_label, effect_col in [
            ("unit_shuffle", "effect_minus_unit_shuffle_median"),
            ("random_subspace", "effect_minus_random_subspace_median"),
            ("rf_readout_fixed_permutation", "effect_minus_rf_fixed_permutation_median"),
        ]:
            vals = g[effect_col].to_numpy(dtype=np.float64) if effect_col in g else np.array([], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            excess_mean, excess_lo, excess_hi = _bootstrap_mean_ci(
                vals,
                seed=int(seed) + int(k) * 31 + _stable_seed_offset(str(basis_source) + null_label),
            )
            summary_rows.append(
                {
                    "target_variant": target_variant,
                    "projection_control": projection_control,
                    "source_variant": _source_label(str(basis_source)),
                    "basis_source": str(basis_source),
                    "k": int(k),
                    "null_type": null_label,
                    "n_sessions": int(g["session"].nunique()),
                    "capture_mean": capture_mean,
                    "capture_boot_ci_low": capture_lo,
                    "capture_boot_ci_high": capture_hi,
                    "excess_mean": excess_mean,
                    "excess_boot_ci_low": excess_lo,
                    "excess_boot_ci_high": excess_hi,
                    "n_effect_positive": int(np.sum(vals > 0.0)) if vals.size else 0,
                    "n_effect_nonzero": int(vals.size),
                }
            )
    _write_csv(paths.out_root / "panelE_covariance_closure_bootstrap_summary.csv", summary_rows)
    _write_csv(paths.out_root / "panelE_covariance_closure_k_sweep.csv", rows)

    raw_psd = metrics[
        (metrics["row_status"].astype(str) == "ok")
        & (metrics["projection_control"].astype(str) == projection_control)
        & (metrics["basis_source"].astype(str).isin(source_variants))
        & (metrics["k"].astype(int).isin([2, 10]))
    ].copy()
    raw_psd_rows = [
        {
            "session_id": row.session,
            "target_variant": row.target_variant,
            "projection_control": row.projection_control,
            "source_variant": _source_label(row.basis_source),
            "k": int(row.k),
            "capture_observed": float(row.capture),
            "trace_target": float(row.target_trace),
            "target_trace_raw": float(row.target_trace_raw),
            "target_trace_psd": float(row.target_trace_psd),
        }
        for row in raw_psd.itertuples(index=False)
    ]
    _write_csv(paths.out_root / "panelE_covariance_closure_raw_vs_psd.csv", raw_psd_rows)

    plot_block = block[block["k"].astype(int) == 2].copy()
    plot_summary = (
        plot_block.groupby("basis_source")
        .agg(capture=("capture", "mean"), rf=("effect_minus_rf_fixed_permutation_median", "mean"))
        .reset_index()
    )
    labels = [_source_label(str(v)).replace("_", "\n") for v in plot_summary["basis_source"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(x - 0.16, plot_summary["capture"], width=0.3, color=BRIDGE, label="capture")
    ax.bar(x + 0.16, plot_summary["rf"], width=0.3, color=GREEN, label="excess vs RF null")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("target covariance fraction")
    ax.set_title("E. Recorded covariance closure", loc="left", fontweight="bold")
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, paths.figures_dir / "panelE_covariance_closure_full_vs_compact")
    compact = plot_summary[plot_summary["basis_source"].astype(str) == "fd_sample_eye_trace_xfit_compact_k10_cov"]
    full = plot_summary[plot_summary["basis_source"].astype(str) == "fd_sample_eye_trace_cov"]
    ratio = float(compact["capture"].iloc[0] / full["capture"].iloc[0]) if len(compact) and len(full) else float("nan")
    return {
        "target_variant": target_variant,
        "projection_control": projection_control,
        "n_sessions": int(block["session"].nunique()),
        "compact_to_full_capture_ratio_k2": ratio,
        "source": str((paths.closure_root / "finite_difference_capture_metrics.csv").resolve()),
    }


def _panel_d(
    paths: SourcePaths,
    *,
    target_variant: str,
    projection_control: str,
) -> dict[str, Any]:
    metrics = _read_closure_metrics(paths.closure_root)
    source_variants = {"fd_sample_eye_trace_cov", "fd_sample_eye_trace_xfit_compact_k10_cov"}
    block = _closure_subset(
        metrics,
        target_variant=target_variant,
        projection_control=projection_control,
        source_variants=source_variants,
    )
    rows: list[dict[str, Any]] = []
    for row in block.itertuples(index=False):
        if int(row.k) != 2:
            continue
        absolute_capture = float(row.capture) * float(row.target_trace)
        rf_effect = _finite_float(getattr(row, "effect_minus_rf_fixed_permutation_median", float("nan")))
        unit_effect = _finite_float(getattr(row, "effect_minus_unit_shuffle_median", float("nan")))
        denominators = [
            ("non_global_projected_FEM_target", float(row.target_trace), "available_from_projected_target_trace"),
            ("full_FEM_linked_covariance_raw_trace", float(row.target_trace_raw), "available_raw_target_trace"),
            ("full_FEM_linked_covariance_psd_trace", float(row.target_trace_psd), "available_psd_target_trace"),
        ]
        for denom_name, denom_trace, status in denominators:
            rows.append(
                {
                    "session_id": row.session,
                    "projection_control": row.projection_control,
                    "budget_level": "finite_difference_closure",
                    "denominator_name": denom_name,
                    "denominator_status": status,
                    "denominator_trace": denom_trace,
                    "source_variant": _source_label(row.basis_source),
                    "k": int(row.k),
                    "absolute_capture_trace": absolute_capture,
                    "fraction_of_denominator": absolute_capture / denom_trace if denom_trace > 0 else float("nan"),
                    "null_type": "rf_readout_fixed_permutation",
                    "null_adjusted_fraction": (
                        rf_effect * float(row.target_trace) / denom_trace if np.isfinite(rf_effect) and denom_trace > 0 else float("nan")
                    ),
                    "unit_shuffle_adjusted_fraction": (
                        unit_effect * float(row.target_trace) / denom_trace
                        if np.isfinite(unit_effect) and denom_trace > 0
                        else float("nan")
                    ),
                    "reliability_ceiling": float("nan"),
                    "ceiling_normalized_capture": float("nan"),
                }
            )
    missing_denominators = [
        "positive_shared_covariance_trace",
        "total_reliable_shared_covariance_trace",
        "total_trial_to_trial_covariance_trace",
        "split_half_reliability_ceiling_for_FEM_covariance",
    ]
    for name in missing_denominators:
        rows.append(
            {
                "session_id": "not_available",
                "projection_control": projection_control,
                "budget_level": "denominator_inventory",
                "denominator_name": name,
                "denominator_status": "not_available_in_closure_summary",
                "denominator_trace": float("nan"),
                "source_variant": "",
                "k": "",
                "absolute_capture_trace": float("nan"),
                "fraction_of_denominator": float("nan"),
                "null_type": "",
                "null_adjusted_fraction": float("nan"),
                "reliability_ceiling": float("nan"),
                "ceiling_normalized_capture": float("nan"),
            }
        )
    _write_csv(paths.out_root / "panelD_variability_budget.csv", rows)

    df = pd.DataFrame([r for r in rows if r["session_id"] != "not_available"])
    summary_rows: list[dict[str, Any]] = []
    if not df.empty:
        for (denom, source), g in df.groupby(["denominator_name", "source_variant"]):
            summary_rows.append(
                {
                    "projection_control": projection_control,
                    "target_variant": target_variant,
                    "denominator_name": denom,
                    "source_variant": source,
                    "n_sessions": int(g["session_id"].nunique()),
                    "fraction_of_denominator_mean": float(np.nanmean(g["fraction_of_denominator"])),
                    "fraction_of_denominator_ci_low": float(np.nanpercentile(g["fraction_of_denominator"], 2.5)),
                    "fraction_of_denominator_ci_high": float(np.nanpercentile(g["fraction_of_denominator"], 97.5)),
                    "null_adjusted_fraction_mean": float(np.nanmean(g["null_adjusted_fraction"])),
                    "null_adjusted_fraction_ci_low": float(np.nanpercentile(g["null_adjusted_fraction"], 2.5)),
                    "null_adjusted_fraction_ci_high": float(np.nanpercentile(g["null_adjusted_fraction"], 97.5)),
                }
            )
    _write_csv(paths.out_root / "panelD_variability_budget_summary.csv", summary_rows)
    _write_csv(
        paths.out_root / "panelD_reliability_ceiling.csv",
        [
            {
                "projection_control": projection_control,
                "status": "not_available_in_current_closure_artifacts",
                "required_raw_object": "split_half_reliability_ceiling_for_FEM_covariance",
            }
        ],
    )

    plot_df = pd.DataFrame(summary_rows)
    plot_df = plot_df[plot_df["denominator_name"] == "non_global_projected_FEM_target"] if not plot_df.empty else plot_df
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    if not plot_df.empty:
        labels = [str(v).replace("_", "\n") for v in plot_df["source_variant"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.16, plot_df["fraction_of_denominator_mean"], width=0.3, color=BRIDGE, label="capture")
        ax.bar(x + 0.16, plot_df["null_adjusted_fraction_mean"], width=0.3, color=GREEN, label="RF-null adjusted")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of projected FEM target")
    ax.set_title("D. Variability budget", loc="left", fontweight="bold")
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, paths.figures_dir / "panelD_variability_budget")
    return {
        "available_denominators": sorted({r["denominator_name"] for r in rows if r["session_id"] != "not_available"}),
        "missing_denominators": missing_denominators,
    }


def _write_manifest(paths: SourcePaths, args: argparse.Namespace, panel_status: dict[str, Any]) -> None:
    closure_manifest_path = paths.closure_root / "run_manifest.json"
    closure_manifest = json.loads(closure_manifest_path.read_text(encoding="utf-8")) if closure_manifest_path.exists() else {}
    tfts_summary_path = paths.tfts_root / "twin_feature_tangent_summary.json"
    tfts_summary = json.loads(tfts_summary_path.read_text(encoding="utf-8")) if tfts_summary_path.exists() else {}
    manifest = {
        "status": "ok",
        "analysis_package": "declan.compact_retinal_translation_geometry",
        "purpose": "Upgrade/replacement harness for declan/fig4_cov_TFTS",
        "finite_difference_step_arcmin": panel_status.get("panelA", {}).get("delta_arcmin", args.primary_delta),
        "response_window_ms": "inherited_from_closure_artifacts",
        "eye_history_window_ms": "inherited_from_closure_artifacts",
        "eye_neural_latency_ms": "inherited_from_closure_artifacts",
        "eye_coordinate_units": "closure eye positions converted to pixels for finite-difference source; tangent maps report arcmin step",
        "eye_coordinate_sign_convention": "not used for unsigned subspace/covariance panels; must be set before signed decoding",
        "sessions_included": closure_manifest.get("sessions_requested", []),
        "matched_units_per_session": "see panelE_covariance_closure_metrics.csv and closure session summary",
        "context_definition": "twin tangent image/trial/time objects",
        "projection_controls": args.projection_control,
        "random_seed": int(args.seed),
        "number_of_null_draws": {
            "tangent_spectrum": "inherited from T_FTS null_spectrum_summary",
            "closure": closure_manifest.get("n_nulls"),
            "closure_rf_readout": closure_manifest.get("rf_null_n_nulls"),
        },
        "code_version_or_git_commit": "not recorded by this adapter",
        "input_cache_paths": {
            "tfts_root": str(paths.tfts_root.resolve()),
            "closure_root": str(paths.closure_root.resolve()),
        },
        "source_manifests": {
            "closure": closure_manifest,
            "tfts_summary_keys": sorted(tfts_summary.keys()),
        },
        "panel_status": panel_status,
        "guardrails": [
            "Panel D reports unavailable denominators explicitly.",
            "Panel B RF/readout samplewise nulls are not fabricated; only existing unit-shuffle spectra are labeled.",
            "Signed coordinate/decoding claims are not made by this adapter.",
        ],
    }
    _write_json(paths.out_root / "compact_retinal_translation_geometry_manifest.json", manifest)


def run(args: argparse.Namespace) -> None:
    paths = SourcePaths(
        tfts_root=Path(args.tfts_root),
        closure_root=Path(args.closure_root),
        out_root=Path(args.out_root),
    )
    paths.out_root.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    panel_status: dict[str, Any] = {}
    panel_status["panelA"] = _panel_a(
        paths,
        primary_delta=float(args.primary_delta),
        n_contexts=int(args.panel_a_contexts),
        seed=int(args.seed),
    )
    panel_status["panelB"] = _panel_b(
        paths,
        primary_delta=float(args.primary_delta),
        n_units=int(panel_status["panelA"]["n_units"]),
    )
    panel_status["panelC"] = _panel_c(paths, primary_delta=float(args.primary_delta))
    panel_status["panelE"] = _panel_e(
        paths,
        target_variant=str(args.target_variant),
        projection_control=str(args.projection_control),
        seed=int(args.seed),
    )
    panel_status["panelD"] = _panel_d(
        paths,
        target_variant=str(args.target_variant),
        projection_control=str(args.projection_control),
    )
    _write_manifest(paths, args, panel_status)
    print(f"Wrote compact retinal-translation geometry outputs to {paths.out_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_CLOSURE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--primary-delta", type=float, default=0.25)
    parser.add_argument("--target-variant", choices=["raw", "psd"], default="psd")
    parser.add_argument(
        "--projection-control",
        choices=["none", "global_rate", "target_pc1", "global_rate+target_pc1"],
        default="global_rate+target_pc1",
    )
    parser.add_argument("--panel-a-contexts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V11 curvature/amplitude-law analysis for Figure 4.

This analysis asks whether local finite-difference retinal-translation
tangents behave like a local first-order approximation: controlled finite
translations should be predicted best at small drift-scale amplitudes and
degrade at larger offsets. It reuses the matched finite-difference closure
machinery so sessions/units/sampling match the Figure 4 covariance closure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _basis_from_cov_or_matrix,
    _capture,
    _cov_rows,
    _load_pickle,
    _null_captures,
    _orth,
    _write_csv,
    _write_json,
    build_inventory,
)
from declan.matched_twin_covariance_closure.run_finite_difference_closure import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    _behavior_batch,
    _collect_samples,
    _compute_jacobians,
    _fit_rescale_gains,
    _load_twin_model,
    _predict,
    _shift_stimulus_batch,
    _stim_batch,
    _target_for_session,
    _tangent_matrix,
)


DEFAULT_OUT = ROOT / "outputs" / "covTFTS_v11_remaining_analysis" / "curvature_amplitude_law_smoke"


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _parse_bins(spec: str) -> tuple[np.ndarray, list[str]]:
    raw = [x.strip() for x in str(spec).split(",") if x.strip()]
    vals: list[float] = []
    for x in raw:
        vals.append(float("inf") if x.lower() in {"inf", "infinity"} else float(x))
    if len(vals) < 2:
        raise ValueError("--amplitude-bins-arcmin must contain at least two edges")
    edges = np.asarray(vals, dtype=np.float64)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("--amplitude-bins-arcmin edges must be strictly increasing")
    labels: list[str] = []
    defaults = ["drift_scale", "intermediate", "microsaccade_scale", "larger_offsets"]
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i < len(defaults):
            labels.append(defaults[i])
        elif np.isfinite(hi):
            labels.append(f"amp_{lo:g}_to_{hi:g}_arcmin")
        else:
            labels.append(f"amp_ge_{lo:g}_arcmin")
    return edges, labels


def _rows_to_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 3:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=np.float64)
    return _cov_rows(x)


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ok = np.all(np.isfinite(actual), axis=1) & np.all(np.isfinite(pred), axis=1)
    a = np.asarray(actual[ok], dtype=np.float64)
    p = np.asarray(pred[ok], dtype=np.float64)
    if a.shape[0] == 0:
        return {
            "pointwise_r2": float("nan"),
            "mean_residual_energy_fraction": float("nan"),
            "median_residual_norm_fraction": float("nan"),
            "median_cosine_actual_pred": float("nan"),
            "actual_energy_mean": float("nan"),
            "predicted_energy_mean": float("nan"),
        }
    resid = a - p
    a2 = np.sum(a * a, axis=1)
    p2 = np.sum(p * p, axis=1)
    r2 = np.sum(resid * resid, axis=1)
    denom = float(np.sum(a2))
    pointwise_r2 = 1.0 - float(np.sum(r2)) / denom if denom > 1e-12 else float("nan")
    residual_norm_fraction = np.sqrt(r2) / np.maximum(np.sqrt(a2), 1e-12)
    cos = np.sum(a * p, axis=1) / np.maximum(np.sqrt(a2 * p2), 1e-12)
    return {
        "pointwise_r2": pointwise_r2,
        "mean_residual_energy_fraction": float(np.mean(r2 / np.maximum(a2, 1e-12))),
        "median_residual_norm_fraction": float(np.median(residual_norm_fraction)),
        "median_cosine_actual_pred": float(np.median(cos)),
        "actual_energy_mean": float(np.mean(a2)),
        "predicted_energy_mean": float(np.mean(p2)),
    }


def _compact_project_linear(
    *,
    j: np.ndarray,
    linear: np.ndarray,
    group_ids: np.ndarray,
    compact_k: int,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    projected = np.full_like(linear, np.nan, dtype=np.float64)
    stats: dict[str, Any] = {
        "compact_basis_k": int(compact_k),
        "compact_n_groups": int(unique_groups.size),
        "compact_requested_folds": int(n_folds),
        "compact_status": "not_run",
    }
    if unique_groups.size < 2:
        stats["compact_status"] = "too_few_groups"
        return projected, stats

    rng = np.random.default_rng(int(seed))
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    folds = [x for x in np.array_split(shuffled, min(int(n_folds), unique_groups.size)) if x.size]
    rank_train: list[int] = []
    rank_used: list[int] = []
    for test_groups in folds:
        test_mask = np.isin(group_ids, test_groups)
        train_mask = ~test_mask
        m_train = _tangent_matrix(j, train_mask)
        if m_train.shape[1] == 0 or not np.isfinite(m_train).all():
            continue
        vals, vecs = _basis_from_cov_or_matrix("compact_tangent_matrix", None, m_train)
        vmax = float(np.max(np.abs(vals))) if vals.size else 0.0
        rank = int(np.sum(vals > max(vmax, 1.0) * 1e-10))
        use_k = min(int(compact_k), rank, int(vecs.shape[1]))
        rank_train.append(rank)
        rank_used.append(use_k)
        if use_k <= 0:
            continue
        basis = _orth(vecs[:, :use_k])
        projected[test_mask] = linear[test_mask] @ basis @ basis.T
    ok = np.all(np.isfinite(projected), axis=1)
    stats.update(
        {
            "compact_status": "ok" if ok.all() else "partial_or_invalid",
            "compact_n_folds": int(len(folds)),
            "compact_min_train_rank": int(min(rank_train)) if rank_train else 0,
            "compact_min_rank_used": int(min(rank_used)) if rank_used else 0,
            "compact_projected_samples": int(np.sum(ok)),
        }
    )
    return projected, stats


def _expanded_displacements(
    *,
    eye_px: np.ndarray,
    samples_source_indices: np.ndarray,
    samples_trial_ids: np.ndarray,
    j: np.ndarray,
    pixels_per_degree: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = str(args.displacement_mode)
    if mode == "eye_cloud":
        delta_px = eye_px - np.mean(eye_px, axis=0, keepdims=True)
        return (
            np.asarray(samples_source_indices, dtype=np.int64),
            np.asarray(samples_trial_ids, dtype=np.int64),
            np.asarray(j, dtype=np.float64),
            delta_px,
        )
    if mode != "controlled_radii":
        raise ValueError(f"Unknown displacement mode: {mode}")

    radii_arcmin = np.asarray([float(x) for x in str(args.controlled_radii_arcmin).split(",") if x.strip()], dtype=np.float64)
    if radii_arcmin.size == 0 or not np.all(np.isfinite(radii_arcmin)) or np.any(radii_arcmin <= 0.0):
        raise ValueError("--controlled-radii-arcmin must contain positive finite radii")
    centered = eye_px - np.mean(eye_px, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    dirs = np.zeros_like(centered, dtype=np.float64)
    ok = norms > 1e-9
    dirs[ok] = centered[ok] / norms[ok, None]
    if np.any(~ok):
        fallback_angles = np.linspace(0.0, 2.0 * np.pi, int(np.sum(~ok)), endpoint=False)
        dirs[~ok, 0] = np.cos(fallback_angles)
        dirs[~ok, 1] = np.sin(fallback_angles)

    repeated_indices: list[np.ndarray] = []
    repeated_groups: list[np.ndarray] = []
    repeated_j: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    for radius_arcmin in radii_arcmin:
        radius_px = float(radius_arcmin) / 60.0 * float(pixels_per_degree)
        repeated_indices.append(np.asarray(samples_source_indices, dtype=np.int64))
        repeated_groups.append(np.asarray(samples_trial_ids, dtype=np.int64))
        repeated_j.append(np.asarray(j, dtype=np.float64))
        displacements.append(dirs * radius_px)
    return (
        np.concatenate(repeated_indices, axis=0),
        np.concatenate(repeated_groups, axis=0),
        np.concatenate(repeated_j, axis=0),
        np.concatenate(displacements, axis=0),
    )


def _empirical_eye_anchor_rows(eye_deg: np.ndarray, trial_ids: np.ndarray, time_indices: np.ndarray) -> dict[str, float]:
    eye = np.asarray(eye_deg, dtype=np.float64)
    centered = eye - np.nanmean(eye, axis=0, keepdims=True)
    radius_arcmin = np.linalg.norm(centered, axis=1) * 60.0
    step_norms: list[float] = []
    trial_ids = np.asarray(trial_ids)
    time_indices = np.asarray(time_indices)
    for trial in np.unique(trial_ids):
        idx = np.flatnonzero(trial_ids == trial)
        if idx.size < 2:
            continue
        order = idx[np.argsort(time_indices[idx], kind="mergesort")]
        d = np.diff(eye[order], axis=0)
        step_norms.extend((np.linalg.norm(d, axis=1) * 60.0).tolist())
    steps = np.asarray(step_norms, dtype=np.float64)
    steps = steps[np.isfinite(steps)]
    return {
        "empirical_eye_cloud_radius_arcmin_p50": float(np.nanpercentile(radius_arcmin, 50)),
        "empirical_eye_cloud_radius_arcmin_p90": float(np.nanpercentile(radius_arcmin, 90)),
        "empirical_eye_step_arcmin_p50": float(np.nanpercentile(steps, 50)) if steps.size else float("nan"),
        "empirical_eye_step_arcmin_p90": float(np.nanpercentile(steps, 90)) if steps.size else float("nan"),
        "empirical_eye_step_arcmin_rms": float(np.sqrt(np.nanmean(steps * steps))) if steps.size else float("nan"),
        "empirical_eye_step_n": int(steps.size),
    }


def _finite_response_changes(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    source_indices: np.ndarray,
    displacements_px: np.ndarray,
    common_units: np.ndarray,
    gains: np.ndarray,
    dataset_idx: int,
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, int(source_indices.size), int(batch_size)):
        idx = source_indices[start : start + int(batch_size)]
        disp = displacements_px[start : start + int(batch_size)]
        stim = _stim_batch(dset, idx, stim_lags).to(model.device)
        behavior = _behavior_batch(dset, idx)
        base = _predict(model, stim, behavior, dataset_idx).detach().cpu().numpy()
        shifted = _predict(model, _shift_stimulus_batch(stim, disp), behavior, dataset_idx).detach().cpu().numpy()
        delta = (shifted[:, common_units] - base[:, common_units]).astype(np.float64)
        delta *= np.asarray(gains, dtype=np.float64)[None, :]
        out.append(delta)
    return np.concatenate(out, axis=0)


def _cov_capture_rows(
    *,
    rng: np.random.Generator,
    session: str,
    bin_label: str,
    bin_low: float,
    bin_high: float,
    actual: np.ndarray,
    sources: dict[str, np.ndarray],
    k_list: list[int],
    n_nulls: int,
) -> list[dict[str, Any]]:
    target = _rows_to_cov(actual)
    rows: list[dict[str, Any]] = []
    target_trace = float(np.trace(target)) if np.isfinite(target).all() else float("nan")
    for source_name, pred in sources.items():
        cov = _rows_to_cov(pred)
        if not np.isfinite(cov).all():
            for k in k_list:
                rows.append(
                    {
                        "session": session,
                        "amplitude_bin": bin_label,
                        "bin_low_arcmin": bin_low,
                        "bin_high_arcmin": bin_high,
                        "source_variant": source_name,
                        "k": int(k),
                        "n_samples": int(actual.shape[0]),
                        "target_actual_cov_trace": target_trace,
                        "source_cov_trace": float("nan"),
                        "capture_fraction": float("nan"),
                        "random_subspace_null_median": float("nan"),
                        "unit_shuffle_null_median": float("nan"),
                        "effect_minus_random_subspace_median": float("nan"),
                        "effect_minus_unit_shuffle_median": float("nan"),
                        "row_status": "not_evaluable",
                    }
                )
            continue
        vals, vecs = _basis_from_cov_or_matrix(f"{source_name}_cov", cov, None)
        vmax = float(np.max(np.abs(vals))) if vals.size else 0.0
        rank = int(np.sum(vals > max(vmax, 1.0) * 1e-10))
        for k in k_list:
            if int(k) > max(rank, 0):
                rows.append(
                    {
                        "session": session,
                        "amplitude_bin": bin_label,
                        "bin_low_arcmin": bin_low,
                        "bin_high_arcmin": bin_high,
                        "source_variant": source_name,
                        "k": int(k),
                        "n_samples": int(actual.shape[0]),
                        "target_actual_cov_trace": target_trace,
                        "source_cov_trace": float(np.trace(cov)),
                        "source_rank": rank,
                        "capture_fraction": float("nan"),
                        "random_subspace_null_median": float("nan"),
                        "unit_shuffle_null_median": float("nan"),
                        "effect_minus_random_subspace_median": float("nan"),
                        "effect_minus_unit_shuffle_median": float("nan"),
                        "row_status": "k_exceeds_rank",
                    }
                )
                continue
            cap = _capture(target, vecs[:, : int(k)])
            nulls = _null_captures(
                rng=rng,
                target=target,
                basis_vecs=vecs,
                source_matrix=None,
                k=int(k),
                n_nulls=int(n_nulls),
            )
            rows.append(
                {
                    "session": session,
                    "amplitude_bin": bin_label,
                    "bin_low_arcmin": bin_low,
                    "bin_high_arcmin": bin_high,
                    "source_variant": source_name,
                    "k": int(k),
                    "n_samples": int(actual.shape[0]),
                    "target_actual_cov_trace": target_trace,
                    "source_cov_trace": float(np.trace(cov)),
                    "source_rank": rank,
                    "capture_fraction": cap,
                    **nulls,
                    "effect_minus_random_subspace_median": cap - nulls["random_subspace_null_median"],
                    "effect_minus_unit_shuffle_median": cap - nulls["unit_shuffle_null_median"],
                    "row_status": "ok",
                }
            )
    return rows


def _plot(out: Path, metric_rows: list[dict[str, Any]], cov_rows: list[dict[str, Any]]) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bins = []
    for row in metric_rows:
        label = str(row["amplitude_bin"])
        if label not in bins:
            bins.append(label)
    x = np.arange(len(bins))
    display = {
        "drift_scale": "drift",
        "intermediate": "intermed.",
        "microsaccade_scale": "micro",
        "larger_offsets": "large",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0), sharex=True, constrained_layout=True)
    colors = {"full_linear_tangent": "#2f5f9f", "compact_k10_tangent": "#7b5ea7"}
    for predictor in ["full_linear_tangent", "compact_k10_tangent"]:
        vals = []
        resid = []
        for b in bins:
            rows = [
                r
                for r in metric_rows
                if r["amplitude_bin"] == b
                and r["prediction_variant"] == predictor
                and r.get("row_status") == "ok"
            ]
            vals.append(np.nanmean([float(r["pointwise_r2"]) for r in rows]) if rows else np.nan)
            resid.append(np.nanmean([float(r["median_residual_norm_fraction"]) for r in rows]) if rows else np.nan)
        axes[0].plot(x, vals, marker="o", color=colors[predictor], label=predictor.replace("_", " "))
        axes[1].plot(x, resid, marker="o", color=colors[predictor], label=predictor.replace("_", " "))
    for predictor in ["full_linear_tangent", "compact_k10_tangent"]:
        vals = []
        for b in bins:
            rows = [
                r
                for r in cov_rows
                if r["amplitude_bin"] == b
                and r["source_variant"] == predictor
                and int(r["k"]) == 10
                and r["row_status"] == "ok"
            ]
            vals.append(np.nanmean([float(r["capture_fraction"]) for r in rows]) if rows else np.nan)
        axes[2].plot(x, vals, marker="o", color=colors[predictor], label=predictor.replace("_", " "))
    axes[0].set_ylabel("pointwise R2", fontsize=8)
    axes[1].set_ylabel("residual norm / actual", fontsize=8)
    axes[2].set_ylabel("covariance captured, k=10", fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([display.get(b, b) for b in bins], rotation=0, ha="center", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.suptitle("Curvature/amplitude law: local tangent prediction degrades with displacement scale", fontsize=9)
    fig.savefig(fig_dir / "curvature_amplitude_law.png", dpi=220)
    fig.savefig(fig_dir / "curvature_amplitude_law.pdf")
    plt.close(fig)


def _write_readme(out: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Curvature / Amplitude Law",
        "",
        "This output implements the v11 curvature/amplitude-law analysis for Figure 4.",
        "",
        "## Outputs",
        "",
        "- `curvature_amplitude_law_metrics.csv`: pointwise finite-response prediction metrics by amplitude bin.",
        "- `curvature_amplitude_law_covariance_capture.csv`: covariance capture of actual finite responses by full and compact tangent sources.",
        "- `curvature_residual_enrichment.csv`: residual covariance/energy diagnostics by amplitude bin.",
        "- `figures/curvature_amplitude_law.png`: summary figure.",
        "- `run_manifest.json`: provenance and run settings.",
        "",
        "## Interpretation Guardrail",
        "",
        "This analysis tests local first-order prediction quality and finite-displacement curvature. It should not be described as evidence for an explicit eye-position coordinate readout.",
        "",
        "## Run Summary",
        "",
        f"- Sessions requested: {manifest.get('sessions_requested')}",
        f"- Sessions completed: {manifest.get('sessions_completed')}",
        f"- Max samples: {manifest.get('max_samples')}",
        f"- Step px for local finite differences: {manifest.get('step_px')}",
        f"- Displacement mode: {manifest.get('displacement_mode')}",
        f"- Controlled radii arcmin: {manifest.get('controlled_radii_arcmin')}",
        f"- Amplitude bins arcmin: {manifest.get('amplitude_bins_arcmin')}",
        "",
        "Empirical eye-amplitude anchors are written per session in `curvature_session_summary.csv`.",
        "",
        "Production command template:",
        "",
        "```bash",
        ".venv/bin/python declan/fig4_cov_TFTS/run_curvature_amplitude_law.py --sessions all --max-samples 0 --output-root outputs/covTFTS_v11_remaining_analysis/curvature_amplitude_law",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figure_companion(
    *,
    out: Path,
    manifest: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    cov_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
) -> None:
    ok_metrics = [r for r in metric_rows if r.get("row_status") == "ok"]
    bins: list[str] = []
    for row in ok_metrics:
        b = str(row["amplitude_bin"])
        if b not in bins:
            bins.append(b)
    completed = list(manifest.get("sessions_completed", []))
    max_samples = int(manifest.get("max_samples", 0))
    run_scope = (
        "all-session uncapped production run"
        if len(completed) >= 24 and max_samples == 0
        else "all-session capped production validation"
        if len(completed) >= 24
        else "smoke/validation run"
    )

    def val(rows: list[dict[str, Any]], key: str) -> str:
        arr = np.asarray([_finite(r.get(key)) for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return "NA" if arr.size == 0 else f"{float(np.mean(arr)):.3f}"

    lines = [
        "# Curvature / Amplitude Law Figure Companion",
        "",
        "This document sits next to `curvature_amplitude_law.png` and summarizes the generated v11 curvature/amplitude-law artifact.",
        "",
        "## Status",
        "",
        f"- Output root: `{out}`",
        f"- Sessions completed: {manifest.get('sessions_completed')}",
        f"- Max samples per session: {manifest.get('max_samples')}",
        f"- Displacement mode: `{manifest.get('displacement_mode')}`",
        f"- Controlled radii arcmin: `{manifest.get('controlled_radii_arcmin')}`",
        "",
        f"Run scope: **{run_scope}**. Regenerate with `--sessions all --max-samples 0` for an uncapped production pass.",
        "",
        "## Empirical Eye-Amplitude Anchors",
        "",
        "| quantity | session median | session p90 |",
        "|---|---:|---:|",
        (
            "| eye-cloud radius (arcmin) | "
            f"{val(session_rows, 'empirical_eye_cloud_radius_arcmin_p50')} | "
            f"{val(session_rows, 'empirical_eye_cloud_radius_arcmin_p90')} |"
        ),
        (
            "| within-trial eye step (arcmin) | "
            f"{val(session_rows, 'empirical_eye_step_arcmin_p50')} | "
            f"{val(session_rows, 'empirical_eye_step_arcmin_p90')} |"
        ),
        "",
        "## Pointwise Prediction Summary",
        "",
        "| amplitude bin | full tangent R2 | compact k=10 R2 | full residual norm | compact residual norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in bins:
        full = [r for r in ok_metrics if r["amplitude_bin"] == b and r["prediction_variant"] == "full_linear_tangent"]
        compact = [r for r in ok_metrics if r["amplitude_bin"] == b and r["prediction_variant"] == "compact_k10_tangent"]
        lines.append(
            f"| {b} | {val(full, 'pointwise_r2')} | {val(compact, 'pointwise_r2')} | "
            f"{val(full, 'median_residual_norm_fraction')} | {val(compact, 'median_residual_norm_fraction')} |"
        )
    lines.extend(
        [
            "",
            "## Covariance Capture Summary",
            "",
            "| amplitude bin | full tangent capture k=10 | compact k=10 capture k=10 |",
            "|---|---:|---:|",
        ]
    )
    ok_cov = [r for r in cov_rows if r.get("row_status") == "ok" and int(r.get("k", -1)) == 10]
    for b in bins:
        full = [r for r in ok_cov if r["amplitude_bin"] == b and r["source_variant"] == "full_linear_tangent"]
        compact = [r for r in ok_cov if r["amplitude_bin"] == b and r["source_variant"] == "compact_k10_tangent"]
        lines.append(f"| {b} | {val(full, 'capture_fraction')} | {val(compact, 'capture_fraction')} |")
    lines.extend(
        [
            "",
            "![Curvature amplitude law](curvature_amplitude_law.png)",
            "",
            "## Generated Outputs",
            "",
            "- [`../curvature_amplitude_law_metrics.csv`](../curvature_amplitude_law_metrics.csv): pointwise finite-response prediction metrics.",
            "- [`../curvature_amplitude_law_covariance_capture.csv`](../curvature_amplitude_law_covariance_capture.csv): actual finite-response covariance capture by tangent sources.",
            "- [`../curvature_residual_enrichment.csv`](../curvature_residual_enrichment.csv): residual diagnostics by amplitude bin.",
            "- [`../curvature_session_summary.csv`](../curvature_session_summary.csv): session/unit/sample provenance.",
            "- [`../run_manifest.json`](../run_manifest.json): run manifest.",
            "- [`curvature_amplitude_law.png`](curvature_amplitude_law.png): bitmap figure.",
            "- [`curvature_amplitude_law.pdf`](curvature_amplitude_law.pdf): PDF figure.",
            "",
            "## Safe Wording",
            "",
            "> Controlled finite translations showed the expected local-amplitude dependence: pointwise first-order tangent predictions were strongest at drift-scale offsets and degraded for larger displacements, while covariance-level tangent capture was more stable. This supports interpreting the compact geometry as a local first-order reafferent approximation rather than a global coordinate system.",
            "",
        ]
    )
    (out / "figures" / "curvature_amplitude_law_companion.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    edges, labels = _parse_bins(args.amplitude_bins_arcmin)
    rng = np.random.default_rng(int(args.seed))

    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    inventory = build_inventory(fig3_rows, fig2_rows, int(args.window_idx))
    _write_csv(out / "session_inventory.csv", inventory)
    fig2_by_session = {str(row["session"]): row for row in fig2_rows}
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    requested = [x.strip() for x in str(args.sessions).split(",") if x.strip()]
    if len(requested) == 1 and requested[0].lower() == "all":
        requested = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]

    model, model_info = _load_twin_model(args)
    k_list = [int(x) for x in str(args.k_list).split(",") if x.strip()]
    metrics_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for session in requested:
        if session not in fig2_by_session or session not in fig3_by_session:
            session_rows.append({"session": session, "status": "missing_cache_rows"})
            continue
        if session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_model_session"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, _target_raw, _target_psd, target_meta = _target_for_session(fig2_by_session[session], sr, args)
        if common_units.size < max(k_list):
            session_rows.append({"session": session, "status": "too_few_common_units", **target_meta})
            continue
        dset, stim_lags, samples = _collect_samples(
            model=model,
            dataset_idx=dataset_idx,
            common_units=common_units,
            args=args,
        )
        gains, rescale_status = _fit_rescale_gains(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            dataset_idx=dataset_idx,
            args=args,
        )
        j = _compute_jacobians(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            gains=gains,
            dataset_idx=dataset_idx,
            args=args,
        )
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
        empirical_eye_anchors = _empirical_eye_anchor_rows(samples.eyepos_deg, samples.trial_ids, samples.time_indices)
        source_indices_exp, trial_ids_exp, j_exp, delta_px = _expanded_displacements(
            eye_px=eye_px,
            samples_source_indices=samples.source_indices,
            samples_trial_ids=samples.trial_ids,
            j=j,
            pixels_per_degree=float(samples.pixels_per_degree),
            args=args,
        )
        amp_arcmin = np.linalg.norm(delta_px, axis=1) / float(samples.pixels_per_degree) * 60.0
        actual = _finite_response_changes(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            source_indices=source_indices_exp,
            displacements_px=delta_px,
            common_units=common_units,
            gains=gains,
            dataset_idx=dataset_idx,
            batch_size=int(args.batch_size),
        )
        linear = np.einsum("nua,na->nu", j_exp, delta_px)
        compact, compact_stats = _compact_project_linear(
            j=j_exp,
            linear=linear,
            group_ids=trial_ids_exp,
            compact_k=int(args.compact_k),
            n_folds=int(args.compact_n_folds),
            seed=int(args.seed) + dataset_idx * 7919,
        )
        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": dataset_idx,
                "n_common_units": int(common_units.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_samples_used": int(samples.source_indices.size),
                "n_displacement_samples": int(source_indices_exp.size),
                "pixels_per_degree": float(samples.pixels_per_degree),
                "step_px": float(args.step_px),
                "displacement_mode": str(args.displacement_mode),
                "controlled_radii_arcmin": str(args.controlled_radii_arcmin),
                **empirical_eye_anchors,
                "rescale_status": rescale_status,
                "gain_median": float(np.median(gains)),
                "controlled_displacement_arcmin_p50": float(np.percentile(amp_arcmin, 50)),
                "controlled_displacement_arcmin_p90": float(np.percentile(amp_arcmin, 90)),
                **target_meta,
                **compact_stats,
            }
        )
        for i, label in enumerate(labels):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            mask = (amp_arcmin >= lo) & (amp_arcmin < hi)
            if int(np.sum(mask)) < int(args.min_bin_samples):
                for pred_name in ["full_linear_tangent", "compact_k10_tangent"]:
                    metrics_rows.append(
                        {
                            "session": session,
                            "amplitude_bin": label,
                            "bin_low_arcmin": lo,
                            "bin_high_arcmin": hi,
                            "prediction_variant": pred_name,
                            "n_samples": int(np.sum(mask)),
                            "row_status": "too_few_samples",
                        }
                    )
                continue
            a = actual[mask]
            l = linear[mask]
            c = compact[mask]
            source_map = {
                "full_linear_tangent": l,
                "compact_k10_tangent": c,
            }
            for pred_name, pred in source_map.items():
                row = {
                    "session": session,
                    "amplitude_bin": label,
                    "bin_low_arcmin": lo,
                    "bin_high_arcmin": hi,
                    "prediction_variant": pred_name,
                    "n_samples": int(np.sum(mask)),
                    "amplitude_arcmin_median": float(np.median(amp_arcmin[mask])),
                    "amplitude_arcmin_mean": float(np.mean(amp_arcmin[mask])),
                    "row_status": "ok",
                }
                row.update(_metrics(a, pred))
                metrics_rows.append(row)
                resid = a - pred
                residual_rows.append(
                    {
                        "session": session,
                        "amplitude_bin": label,
                        "bin_low_arcmin": lo,
                        "bin_high_arcmin": hi,
                        "residual_from": pred_name,
                        "n_samples": int(np.sum(mask)),
                        "actual_cov_trace": float(np.trace(_rows_to_cov(a))),
                        "prediction_cov_trace": float(np.trace(_rows_to_cov(pred))),
                        "residual_cov_trace": float(np.trace(_rows_to_cov(resid))),
                        "residual_cov_trace_fraction_of_actual": (
                            float(np.trace(_rows_to_cov(resid))) / float(np.trace(_rows_to_cov(a)))
                            if float(np.trace(_rows_to_cov(a))) > 1e-12
                            else float("nan")
                        ),
                        "mean_residual_energy_fraction": row["mean_residual_energy_fraction"],
                        "median_residual_norm_fraction": row["median_residual_norm_fraction"],
                        "curvature_or_finite_displacement_status": (
                            "residual_increases_with_amplitude_candidate"
                            if row["median_residual_norm_fraction"] > 1.0
                            else "locally_well_predicted_candidate"
                        ),
                    }
                )
            cov_rows.extend(
                _cov_capture_rows(
                    rng=rng,
                    session=session,
                    bin_label=label,
                    bin_low=lo,
                    bin_high=hi,
                    actual=a,
                    sources=source_map,
                    k_list=k_list,
                    n_nulls=int(args.n_nulls),
                )
            )

    _write_csv(out / "curvature_amplitude_law_metrics.csv", metrics_rows)
    _write_csv(out / "curvature_amplitude_law_covariance_capture.csv", cov_rows)
    _write_csv(out / "curvature_residual_enrichment.csv", residual_rows)
    _write_csv(out / "curvature_session_summary.csv", session_rows)
    _plot(out, metrics_rows, cov_rows)
    manifest = {
        "status": "ok",
        "script": str(Path(__file__).resolve()),
        "fig2_cache": str(Path(args.fig2_cache).resolve()),
        "fig3_cache": str(Path(args.fig3_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": str(Path(args.model_config).resolve()),
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "model_info": {k: str(v) for k, v in dict(model_info).items()} if isinstance(model_info, dict) else str(model_info),
        "sessions_requested": requested,
        "sessions_completed": [r["session"] for r in session_rows if r.get("status") == "ok"],
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "step_px": float(args.step_px),
        "displacement_mode": str(args.displacement_mode),
        "controlled_radii_arcmin": str(args.controlled_radii_arcmin),
        "compact_k": int(args.compact_k),
        "compact_n_folds": int(args.compact_n_folds),
        "amplitude_bins_arcmin": [float(x) if np.isfinite(x) else "inf" for x in edges],
        "k_list": k_list,
        "n_nulls": int(args.n_nulls),
        "output_files": [
            "curvature_amplitude_law_metrics.csv",
            "curvature_amplitude_law_covariance_capture.csv",
            "curvature_residual_enrichment.csv",
            "curvature_session_summary.csv",
            "figures/curvature_amplitude_law.png",
        ],
    }
    _write_json(out / "run_manifest.json", manifest)
    _write_readme(out, manifest)
    _write_figure_companion(
        out=out,
        manifest=manifest,
        metric_rows=metrics_rows,
        cov_rows=cov_rows,
        session_rows=session_rows,
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run v11 curvature/amplitude-law finite-translation analysis.")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--sessions", type=str, default="Allen_2022-02-16")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--step-px", type=float, default=0.25)
    p.add_argument("--compact-k", type=int, default=10)
    p.add_argument("--compact-n-folds", type=int, default=5)
    p.add_argument("--k-list", type=str, default="2,10")
    p.add_argument("--n-nulls", type=int, default=20)
    p.add_argument("--amplitude-bins-arcmin", type=str, default="0,1,2,5,inf")
    p.add_argument("--displacement-mode", choices=("controlled_radii", "eye_cloud"), default="controlled_radii")
    p.add_argument("--controlled-radii-arcmin", type=str, default="0.5,1.5,3.5,8.0")
    p.add_argument("--min-bin-samples", type=int, default=5)
    p.add_argument("--window-idx", type=int, default=1)
    p.add_argument("--fixation-radius-deg", type=float, default=2.0)
    p.add_argument("--sample-dfs-mode", choices=("all", "any"), default="any")
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.504766)
    p.add_argument("--rescale-mode", type=str, default="affine")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--verbose-model-load", action="store_true")
    return p


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()

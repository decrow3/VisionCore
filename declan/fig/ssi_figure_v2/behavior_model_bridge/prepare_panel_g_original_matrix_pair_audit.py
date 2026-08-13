#!/usr/bin/env python3
"""Freeze and visualize original-matrix pairs for a direct rotation audit.

Selection uses only quantities available before any fresh rotated model
evaluation: the historical one-dimensional RMS surrogate, original Figure 4
image metadata, trace metadata, and within-image surrogate calibration.  The
selected set is therefore auditable and cannot be chosen from the new direct
outcomes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge


BANK_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
AUDIT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_original_matrix_pair_rotation_audit_v1"
)
CALIBRATION_CSV = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1/implementation_audit_checkpoint1/"
    "old_matrix_within_image_surrogate_calibration.csv"
)
N_SURROGATE_ROTATIONS = 256
N_DISPLAY_ANGLES = 181
COHERENCE_MIN = 0.50


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _predict_from_covariance(
    var_parallel: np.ndarray,
    var_normal: np.ndarray,
    cov_parallel_normal: np.ndarray,
    angles_rad: np.ndarray,
    curves: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component-mean prediction and the two marginal predictions."""
    angle = np.asarray(angles_rad, dtype=np.float64).reshape(1, -1)
    c = np.cos(angle)
    s = np.sin(angle)
    vp = np.asarray(var_parallel, dtype=np.float64).reshape(-1, 1)
    vn = np.asarray(var_normal, dtype=np.float64).reshape(-1, 1)
    cv = np.asarray(cov_parallel_normal, dtype=np.float64).reshape(-1, 1)
    rotated_parallel_var = c * c * vp + s * s * vn - 2.0 * c * s * cv
    rotated_normal_var = s * s * vp + c * c * vn + 2.0 * c * s * cv
    parallel_rms = 60.0 * np.sqrt(np.maximum(rotated_parallel_var, 0.0))
    normal_rms = 60.0 * np.sqrt(np.maximum(rotated_normal_var, 0.0))
    pred_parallel, _ = bridge._interpolate_curve(parallel_rms, curves["along"])
    pred_normal, _ = bridge._interpolate_curve(normal_rms, curves["across"])
    stacked = np.stack([pred_parallel, pred_normal])
    finite = np.isfinite(stacked)
    pred_mean = np.divide(
        np.nansum(stacked, axis=0),
        np.sum(finite, axis=0),
        out=np.full(stacked.shape[1:], np.nan, dtype=np.float64),
        where=np.sum(finite, axis=0) > 0,
    )
    return pred_mean, pred_parallel, pred_normal


def build_candidate_table() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    images = pd.read_csv(BANK_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    traces = np.load(BANK_DIR / "trace_xy.npy")
    trace_meta = pd.read_csv(BANK_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    calibration = pd.read_csv(CALIBRATION_CSV)[["image_index", "pearson", "spearman"]].rename(
        columns={"pearson": "within_image_calibration_pearson", "spearman": "within_image_calibration_spearman"}
    )
    model_values = pd.read_csv(bridge.MODEL_VALUES_CSV)
    curves = {
        component: bridge._curve_for(
            model_values,
            population_key="high_sf_aligned",
            metric_family="component_rms",
            component=component,
        )
        for component in ("along", "across")
    }

    centered = np.asarray(traces, dtype=np.float64)
    centered -= np.mean(centered, axis=1, keepdims=True)
    rotation_angles = (np.arange(N_SURROGATE_ROTATIONS, dtype=np.float64) + 0.5) * (
        np.pi / N_SURROGATE_ROTATIONS
    )
    drift_only = trace_meta["rendered_n_microsaccade_events"].astype(int).eq(0).to_numpy()
    rows: list[pd.DataFrame] = []
    for image in images.itertuples(index=False):
        axis = math.radians(float(image.image_edge_axis_deg))
        parallel = np.asarray([math.cos(axis), math.sin(axis)])
        normal = np.asarray([-math.sin(axis), math.cos(axis)])
        pos_parallel = centered @ parallel
        pos_normal = centered @ normal
        var_parallel = np.mean(pos_parallel * pos_parallel, axis=1)
        var_normal = np.mean(pos_normal * pos_normal, axis=1)
        covariance = np.mean(pos_parallel * pos_normal, axis=1)
        observed, observed_parallel, observed_normal = _predict_from_covariance(
            var_parallel, var_normal, covariance, np.zeros(1), curves
        )
        rotated, rot_parallel, rot_normal = _predict_from_covariance(
            var_parallel, var_normal, covariance, rotation_angles, curves
        )
        rotation_count = np.sum(np.isfinite(rotated), axis=1)
        rotation_mean = np.divide(
            np.nansum(rotated, axis=1), rotation_count,
            out=np.full(len(rotation_count), np.nan), where=rotation_count > 0,
        )
        finite_fraction = np.mean(np.isfinite(rotated), axis=1)
        frame = pd.DataFrame(
            {
                "image_index": int(image.image_index),
                "trace_index": np.arange(len(trace_meta), dtype=int),
                "movie_index": int(image.image_index) * len(trace_meta) + np.arange(len(trace_meta), dtype=int),
                "image_source_row": int(image.source_row),
                "trace_source_row": trace_meta["source_row"].astype(int).to_numpy(),
                "image_session": str(image.session),
                "image_trial_idx": int(image.trial_idx),
                "image_orientation_coherence": float(image.image_orientation_coherence),
                "image_edge_axis_gaze_deg": float(image.image_edge_axis_deg),
                "image_edge_axis_array_deg": float(image.image_edge_axis_array_deg),
                "trace_is_drift_only": drift_only,
                "rendered_path_length_arcmin": trace_meta["rendered_path_length_arcmin"].to_numpy(dtype=float),
                "rendered_rms_radius_arcmin": trace_meta["rendered_rms_radius_arcmin"].to_numpy(dtype=float),
                "parallel_rms_arcmin": 60.0 * np.sqrt(np.maximum(var_parallel, 0.0)),
                "normal_rms_arcmin": 60.0 * np.sqrt(np.maximum(var_normal, 0.0)),
                "surrogate_observed_percent": observed[:, 0],
                "surrogate_parallel_observed_percent": observed_parallel[:, 0],
                "surrogate_normal_observed_percent": observed_normal[:, 0],
                "surrogate_rotation_mean_percent": rotation_mean,
                "surrogate_match_advantage_percent_points": observed[:, 0] - rotation_mean,
                "surrogate_rotation_finite_fraction": finite_fraction,
                "surrogate_parallel_rotation_finite_fraction": np.mean(np.isfinite(rot_parallel), axis=1),
                "surrogate_normal_rotation_finite_fraction": np.mean(np.isfinite(rot_normal), axis=1),
                "surrogate_both_components_rotation_finite_fraction": np.mean(
                    np.isfinite(rot_parallel) & np.isfinite(rot_normal), axis=1
                ),
            }
        )
        rows.append(frame)
    candidates = pd.concat(rows, ignore_index=True).merge(calibration, on="image_index", validate="many_to_one")
    return candidates, curves


def _take_distinct(
    pool: pd.DataFrame,
    *,
    n: int,
    sort_column: str,
    ascending: bool,
    used_images: set[int],
    used_traces: set[int],
) -> pd.DataFrame:
    selected = []
    for row in pool.sort_values(sort_column, ascending=ascending, kind="stable").itertuples(index=False):
        if int(row.image_index) in used_images or int(row.trace_index) in used_traces:
            continue
        selected.append(row._asdict())
        used_images.add(int(row.image_index))
        used_traces.add(int(row.trace_index))
        if len(selected) == int(n):
            break
    if len(selected) != int(n):
        raise RuntimeError(f"Could select only {len(selected)}/{n} distinct rows for {sort_column}")
    return pd.DataFrame(selected)


def freeze_selection(candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = candidates[
        candidates["trace_is_drift_only"].astype(bool)
        & (candidates["image_orientation_coherence"] >= COHERENCE_MIN)
        & np.isfinite(candidates["surrogate_match_advantage_percent_points"])
        & (candidates["surrogate_rotation_finite_fraction"] >= 1.0)
    ].copy()
    image_cal = eligible.drop_duplicates("image_index")[
        ["image_index", "within_image_calibration_spearman"]
    ]
    low_q, high_q = image_cal["within_image_calibration_spearman"].quantile([0.25, 0.75])
    used_images: set[int] = set()
    used_traces: set[int] = set()
    parts: list[pd.DataFrame] = []

    role_specs = [
        (
            "positive_surrogate_calibrated_image",
            "largest positive surrogate match advantage among top-quartile within-image calibration images",
            eligible[eligible["within_image_calibration_spearman"] >= high_q],
            3,
            "surrogate_match_advantage_percent_points",
            False,
        ),
        (
            "positive_surrogate_miscalibrated_image",
            "largest positive surrogate match advantage among bottom-quartile within-image calibration images",
            eligible[eligible["within_image_calibration_spearman"] <= low_q],
            3,
            "surrogate_match_advantage_percent_points",
            False,
        ),
        (
            "negative_surrogate_calibrated_image",
            "most negative surrogate match advantage among top-quartile within-image calibration images",
            eligible[eligible["within_image_calibration_spearman"] >= high_q],
            2,
            "surrogate_match_advantage_percent_points",
            True,
        ),
    ]
    for role, rule, pool, n, column, ascending in role_specs:
        selected = _take_distinct(
            pool, n=n, sort_column=column, ascending=ascending,
            used_images=used_images, used_traces=used_traces,
        )
        selected["selection_role"] = role
        selected["selection_rule"] = rule
        parts.append(selected)

    middle = eligible[eligible["within_image_calibration_spearman"].abs() <= 0.20].copy()
    middle["absolute_surrogate_match_advantage"] = middle["surrogate_match_advantage_percent_points"].abs()
    selected = _take_distinct(
        middle, n=2, sort_column="absolute_surrogate_match_advantage", ascending=True,
        used_images=used_images, used_traces=used_traces,
    )
    selected["selection_role"] = "near_zero_surrogate_control"
    selected["selection_rule"] = "smallest absolute surrogate match advantage among weakly calibrated images"
    parts.append(selected)

    high_coherence = eligible[eligible["image_orientation_coherence"] >= 0.70]
    selected = _take_distinct(
        high_coherence, n=2, sort_column="surrogate_match_advantage_percent_points", ascending=False,
        used_images=used_images, used_traces=used_traces,
    )
    selected["selection_role"] = "high_coherence_positive_surrogate"
    selected["selection_rule"] = "largest positive surrogate match advantage among coherence >=0.70 images"
    parts.append(selected)

    frozen = pd.concat(parts, ignore_index=True)
    frozen.insert(0, "selection_index", np.arange(len(frozen), dtype=int))
    frozen["selection_uses_fresh_rotation_outcome"] = False
    frozen["calibration_bottom_quartile_threshold"] = float(low_q)
    frozen["calibration_top_quartile_threshold"] = float(high_q)
    return frozen


def _angle_curve(row: pd.Series, traces: np.ndarray, curves: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(traces[int(row["trace_index"])], dtype=np.float64)
    trace -= np.mean(trace, axis=0, keepdims=True)
    theta = math.radians(float(row["image_edge_axis_gaze_deg"]))
    parallel = np.asarray([math.cos(theta), math.sin(theta)])
    normal = np.asarray([-math.sin(theta), math.cos(theta)])
    p = trace @ parallel
    n = trace @ normal
    angles_deg = np.linspace(0.0, 180.0, N_DISPLAY_ANGLES)
    prediction, _pp, _pn = _predict_from_covariance(
        np.asarray([np.mean(p * p)]),
        np.asarray([np.mean(n * n)]),
        np.asarray([np.mean(p * n)]),
        np.radians(angles_deg),
        curves,
    )
    return angles_deg, prediction[0]


def plot_inputs(selection: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> None:
    traces = np.load(BANK_DIR / "trace_xy.npy")
    fig, axes = plt.subplots(len(selection), 3, figsize=(11.8, 2.25 * len(selection)), constrained_layout=True)
    if len(selection) == 1:
        axes = axes[None, :]
    for row_index, selected in selection.iterrows():
        print(f"[pair-audit-prep] render selected input {row_index + 1}/{len(selection)}", flush=True)
        aperture_path = AUDIT_ROOT / "input_aperture_cache" / f"selection_{int(selected['selection_index']):02d}.npz"
        if not aperture_path.exists():
            raise FileNotFoundError(
                f"Missing {aperture_path}; run extract_panel_g_audit_aperture.py once per frozen selection."
            )
        with np.load(aperture_path) as payload:
            aperture = np.asarray(payload["aperture"]).copy()
            ppd = float(np.asarray(payload["ppd"]).ravel()[0])
        trace = np.asarray(traces[int(selected["trace_index"])], dtype=np.float64)
        trace -= np.mean(trace, axis=0, keepdims=True)

        ax = axes[row_index, 0]
        ax.imshow(aperture, cmap="gray", origin="upper")
        center = np.asarray([aperture.shape[1] / 2.0, aperture.shape[0] / 2.0])
        screen_trace = np.column_stack(
            [center[0] + trace[:, 0] * ppd, center[1] - trace[:, 1] * ppd]
        )
        ax.plot(screen_trace[:, 0], screen_trace[:, 1], color="#f28e2b", lw=1.0)
        ax.scatter(screen_trace[0, 0], screen_trace[0, 1], s=13, color="#2ca02c", zorder=4)
        angle = math.radians(float(selected["image_edge_axis_array_deg"]))
        length = 0.38 * ppd
        delta = np.asarray([math.cos(angle), math.sin(angle)]) * length
        ax.plot([center[0] - delta[0], center[0] + delta[0]], [center[1] - delta[1], center[1] + delta[1]], color="#00bcd4", lw=1.8)
        ax.set_xlim(0, aperture.shape[1] - 1)
        ax.set_ylim(aperture.shape[0] - 1, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(
            f"{int(selected['selection_index']):02d}  {selected['selection_role']}\n"
            f"image {int(selected['image_index'])}, trace {int(selected['trace_index'])}",
            rotation=0, ha="right", va="center", fontsize=7.0,
        )

        ax = axes[row_index, 1]
        theta = math.radians(float(selected["image_edge_axis_gaze_deg"]))
        parallel = np.asarray([math.cos(theta), math.sin(theta)])
        normal = np.asarray([-math.sin(theta), math.cos(theta)])
        centered = trace - np.mean(trace, axis=0, keepdims=True)
        ax.plot(centered @ parallel * 60.0, centered @ normal * 60.0, color="#6f4aa8", lw=1.1)
        ax.scatter((centered @ parallel)[0] * 60.0, (centered @ normal)[0] * 60.0, s=14, color="#2ca02c")
        ax.axhline(0, color="0.75", lw=0.6); ax.axvline(0, color="0.75", lw=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("contour-parallel (arcmin)")
        ax.set_ylabel("contour-normal (arcmin)")

        ax = axes[row_index, 2]
        angle_deg, prediction = _angle_curve(selected, traces, curves)
        ax.plot(angle_deg, prediction, color="#b4492d", lw=1.4)
        ax.axhline(np.nanmean(prediction), color="0.35", ls="--", lw=0.8, label="angle mean")
        ax.scatter([0], [prediction[0]], s=22, color="#2ca02c", zorder=4, label="recorded")
        ax.set_xlim(0, 180)
        ax.set_xlabel("trajectory rotation relative to recorded (deg)")
        ax.set_ylabel("historical surrogate (% vs stabilized)")
        ax.set_title(
            f"coh={float(selected['image_orientation_coherence']):.3f}; "
            f"surrogate Δ={float(selected['surrogate_match_advantage_percent_points']):+.3f} pp\n"
            f"within-image ρ={float(selected['within_image_calibration_spearman']):+.3f}",
            fontsize=7.5,
        )
        if row_index == 0:
            ax.legend(frameon=False, fontsize=6.5)

    axes[0, 0].set_title("Exact 1° image aperture + recorded trace\ncyan: stored contour axis; green: first sample", fontsize=8.5)
    axes[0, 1].set_title("Recorded trajectory in contour coordinates", fontsize=8.5)
    fig.suptitle(
        "Frozen original-matrix pairs for direct rotation audit\nSelection uses historical quantities only; no fresh rotation outcome inspected",
        fontsize=11, weight="bold",
    )
    fig.savefig(AUDIT_ROOT / "checkpoint_input_selected_original_matrix_pairs.png", dpi=220)
    fig.savefig(AUDIT_ROOT / "checkpoint_input_selected_original_matrix_pairs.pdf")
    plt.close(fig)


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[pair-audit-prep] build pre-rotation candidate table", flush=True)
    candidates, curves = build_candidate_table()
    print("[pair-audit-prep] freeze selection", flush=True)
    selection = freeze_selection(candidates)
    candidates.to_csv(AUDIT_ROOT / "pre_rotation_candidate_table.csv", index=False)
    selection.to_csv(AUDIT_ROOT / "frozen_pair_selection.csv", index=False)
    print("[pair-audit-prep] render input checkpoint", flush=True)
    plot_inputs(selection, curves)
    metadata = {
        "analysis": "panel_g_original_matrix_pair_rotation_audit_pre_rotation_checkpoint",
        "selection_contract": "role-stratified from historical surrogate and within-image calibration before fresh rotation evaluation",
        "n_selected_pairs": int(len(selection)),
        "n_candidates": int(len(candidates)),
        "drift_only_required": True,
        "minimum_image_coherence": COHERENCE_MIN,
        "surrogate_rotations": N_SURROGATE_ROTATIONS,
        "fresh_model_evaluation_performed": False,
        "inputs": {
            "bank_dir": BANK_DIR,
            "model_curves": bridge.MODEL_VALUES_CSV,
            "within_image_calibration": CALIBRATION_CSV,
        },
    }
    (AUDIT_ROOT / "pre_rotation_metadata.json").write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(selection[[
        "selection_index", "selection_role", "image_index", "trace_index",
        "image_orientation_coherence", "surrogate_match_advantage_percent_points",
        "within_image_calibration_spearman",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

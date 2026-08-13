#!/usr/bin/env python3
"""Freeze and visualize an outcome-blind native-pair cohort for Panel G.

This script deliberately reads no direct SSI results and no historical
surrogate predictions.  Eligibility uses only native trace provenance, image
quality, and contour strength.  It writes a strict primary cohort, a broader
development envelope, image-identity/repeat summaries, and a blinded contact
sheet for human contour-axis QC before any new RR100 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    _extract_patch,
)
from declan.fixation_statistics_by_stimulus.image_features import (
    _cached_session,
    backimage_trial_geometry,
)


OLD_NATIVE_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)
BANK_ROOT = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
SOURCE_WINDOWS = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/"
    "backimage_image_fem_windows.csv"
)
OUT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_direct_replacement_strong_contour_v1/input_checkpoint"
)

DEVELOPMENT_COHERENCE_MIN = 0.50
PRIMARY_COHERENCE_MIN = 0.60
MIN_INSIDE_FRACTION = 0.99
N_TIMEPOINTS = 40
LOCAL_CLUSTER_RADIUS_DEG = 1.0
CONTACT_ROWS = 4
CONTACT_COLUMNS = 5


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stimulus_image_hash(session_name: str, trial_idx: int) -> str:
    from DataYatesV1.exp.backimage import BackImageTrial

    session = _cached_session(str(session_name))
    trial = BackImageTrial(session.exp["D"][int(trial_idx)], session.exp["S"])
    image = np.asarray(trial.get_image())
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    return _array_hash(image)


def _axis_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 90.0) % 180.0 - 90.0)


def _add_source_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_WINDOWS).copy()
    source.insert(0, "source_row", np.arange(len(source), dtype=int))
    keep = [
        "source_row", "mean_x_deg", "mean_y_deg", "image_patch_radius_px",
        "image_patch_distance_to_image_border_px", "image_patch_mean", "image_patch_std",
        "image_gradient_axis_array_deg", "image_edge_axis_array_deg",
        "image_spectrum_orientation_deg", "image_spectrum_orientation_array_deg",
        "image_power_0_2_cpd_fraction", "image_power_2_4_cpd_fraction",
        "image_power_4_8_cpd_fraction", "image_power_8plus_cpd_fraction",
    ]
    keep = [column for column in keep if column in source.columns]
    out = cohort.merge(source[keep], on="source_row", how="left", validate="one_to_one")
    expected_array = (-pd.to_numeric(out["image_edge_axis_deg"], errors="coerce")) % 180.0
    stored_array = pd.to_numeric(out["image_edge_axis_array_deg"], errors="coerce") % 180.0
    out["corrected_edge_axis_array_deg"] = stored_array
    out["gaze_to_array_axis_abs_error_deg"] = _axis_delta_deg(expected_array, stored_array)
    spectrum_contour_axis = (
        pd.to_numeric(out["image_spectrum_orientation_deg"], errors="coerce") + 90.0
    ) % 180.0
    out["image_edge_spectrum_contour_axis_agreement"] = np.cos(
        2.0 * np.radians(_axis_delta_deg(out["image_edge_axis_deg"], spectrum_contour_axis))
    )
    return out


def _basic_gate_columns(cohort: pd.DataFrame, trace_xy: np.ndarray) -> pd.DataFrame:
    out = cohort.copy()
    trace_valid = np.asarray(
        [trace.shape == (N_TIMEPOINTS, 2) and np.isfinite(trace).all() for trace in trace_xy],
        dtype=bool,
    )
    if len(trace_valid) != len(out):
        raise RuntimeError("Trace bank length differs from native cohort manifest")
    out["gate_native_trace_valid"] = trace_valid & (
        pd.to_numeric(out["native_trace_max_abs_error"], errors="coerce").fillna(np.inf).to_numpy() <= 1e-6
    )
    out["gate_drift_only"] = pd.to_numeric(
        out["rendered_n_microsaccade_events"], errors="coerce"
    ).fillna(1).astype(int).eq(0)
    out["gate_image_feature_ok"] = out["image_feature_ok"].fillna(False).astype(bool)
    out["gate_inside_image"] = pd.to_numeric(
        out["image_patch_fraction_inside_image"], errors="coerce"
    ).ge(MIN_INSIDE_FRACTION)
    finite_columns = [
        "image_patch_rms_contrast", "image_gradient_energy", "image_edge_density",
        "image_orientation_coherence", "image_edge_axis_deg", "corrected_edge_axis_array_deg",
    ]
    finite = np.ones(len(out), dtype=bool)
    for column in finite_columns:
        finite &= np.isfinite(pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float))
    out["gate_finite_image_measurements"] = finite
    out["gate_axis_coordinate_consistency"] = pd.to_numeric(
        out["gaze_to_array_axis_abs_error_deg"], errors="coerce"
    ).le(1e-6)
    base_columns = [
        "gate_native_trace_valid", "gate_drift_only", "gate_image_feature_ok",
        "gate_inside_image", "gate_finite_image_measurements", "gate_axis_coordinate_consistency",
    ]
    out["base_input_qc_pass"] = out[base_columns].all(axis=1)
    coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce")
    out["development_eligible"] = out["base_input_qc_pass"] & coherence.ge(DEVELOPMENT_COHERENCE_MIN)
    out["primary_eligible"] = out["base_input_qc_pass"] & coherence.ge(PRIMARY_COHERENCE_MIN)
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        failed = []
        for column in base_columns:
            if not bool(getattr(row, column)):
                failed.append(column.removeprefix("gate_"))
        if bool(getattr(row, "base_input_qc_pass")) and float(row.image_orientation_coherence) < DEVELOPMENT_COHERENCE_MIN:
            failed.append("coherence_below_development_min")
        reasons.append(";".join(failed))
    out["development_exclusion_reason"] = reasons
    out["selection_uses_historical_surrogate"] = False
    out["selection_uses_fresh_model_outcome"] = False
    return out


def _add_image_identity(cohort: pd.DataFrame) -> pd.DataFrame:
    out = cohort.copy()
    trial_keys = out.loc[out["development_eligible"], ["session", "trial_idx"]].drop_duplicates()
    hashes: dict[tuple[str, int], str] = {}
    geometry: dict[tuple[str, int], dict[str, Any]] = {}
    for ordinal, row in enumerate(trial_keys.itertuples(index=False), start=1):
        key = (str(row.session), int(row.trial_idx))
        print(f"[panel-g-input] image identity {ordinal}/{len(trial_keys)} {key[0]} trial {key[1]}", flush=True)
        hashes[key] = _stimulus_image_hash(*key)
        geometry[key] = backimage_trial_geometry(*key)
    stimulus_hash: list[str | None] = []
    local_x: list[float] = []
    local_y: list[float] = []
    display_key: list[str] = []
    for row in out.itertuples(index=False):
        key = (str(row.session), int(row.trial_idx))
        display_key.append(f"{key[0]}__trial_{key[1]}")
        if key not in hashes:
            stimulus_hash.append(None)
            local_x.append(np.nan); local_y.append(np.nan)
            continue
        info = geometry[key]
        x0, y0, x1, y1 = info["dest_rect"]
        ppd = float(info["ppd"])
        stimulus_hash.append(hashes[key])
        local_x.append((float(row.image_patch_center_x_px) - (x0 + x1) / 2.0) / ppd)
        local_y.append(-((float(row.image_patch_center_y_px) - (y0 + y1) / 2.0) / ppd))
    out["display_trial_key"] = display_key
    out["stimulus_image_sha256"] = stimulus_hash
    out["stimulus_local_x_deg"] = local_x
    out["stimulus_local_y_deg"] = local_y
    out["local_window_cluster_id"] = ""

    eligible = out[out["development_eligible"]].copy()
    for image_hash, group in eligible.groupby("stimulus_image_sha256", sort=True):
        indices = group.index.to_numpy(dtype=int)
        xy = group[["stimulus_local_x_deg", "stimulus_local_y_deg"]].to_numpy(dtype=float)
        parent = np.arange(len(indices), dtype=int)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if float(np.linalg.norm(xy[i] - xy[j])) <= LOCAL_CLUSTER_RADIUS_DEG:
                    union(i, j)
        roots = [find(i) for i in range(len(indices))]
        ordered_roots = {root: number for number, root in enumerate(sorted(set(roots)))}
        prefix = str(image_hash)[:12]
        for row_index, root in zip(indices, roots):
            out.at[row_index, "local_window_cluster_id"] = f"{prefix}_w{ordered_roots[root]:03d}"
    return out


def _flow_table(cohort: pd.DataFrame) -> pd.DataFrame:
    gates = [
        ("all_native_pairs", np.ones(len(cohort), dtype=bool)),
        ("native_trace_valid", cohort["gate_native_trace_valid"]),
        ("plus_drift_only", cohort["gate_native_trace_valid"] & cohort["gate_drift_only"]),
        (
            "plus_image_qc",
            cohort["gate_native_trace_valid"] & cohort["gate_drift_only"]
            & cohort["gate_image_feature_ok"] & cohort["gate_inside_image"]
            & cohort["gate_finite_image_measurements"] & cohort["gate_axis_coordinate_consistency"],
        ),
        ("development_coherence_ge_0p50", cohort["development_eligible"]),
        ("primary_coherence_ge_0p60", cohort["primary_eligible"]),
    ]
    rows = []
    for stage, mask_like in gates:
        mask = np.asarray(mask_like, dtype=bool)
        subset = cohort.loc[mask]
        rows.append(
            {
                "stage": stage,
                "n_pairs": int(len(subset)),
                "n_subjects": int(subset["subject"].nunique()),
                "n_sessions": int(subset["session"].nunique()),
                "n_display_trials": int(subset["display_trial_key"].nunique()),
                "n_stimulus_images": int(subset["stimulus_image_sha256"].nunique()),
                "n_local_window_clusters": int(subset.loc[subset["local_window_cluster_id"].ne(""), "local_window_cluster_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _repeat_summary(cohort: pd.DataFrame, eligibility_column: str) -> pd.DataFrame:
    subset = cohort[cohort[eligibility_column]].copy()
    rows: list[dict[str, Any]] = []
    for level, column in (
        ("subject", "subject"),
        ("session", "session"),
        ("stimulus_image", "stimulus_image_sha256"),
        ("display_trial", "display_trial_key"),
        ("local_window_cluster", "local_window_cluster_id"),
    ):
        counts = subset.groupby(column, dropna=False).size()
        rows.append(
            {
                "cohort": eligibility_column,
                "level": level,
                "n_groups": int(len(counts)),
                "n_pairs": int(counts.sum()),
                "min_pairs_per_group": int(counts.min()),
                "median_pairs_per_group": float(counts.median()),
                "mean_pairs_per_group": float(counts.mean()),
                "max_pairs_per_group": int(counts.max()),
                "n_groups_with_repeats": int((counts >= 2).sum()),
            }
        )
    return pd.DataFrame(rows)


def _draw_contact_tile(
    ax: plt.Axes,
    row: pd.Series,
    trace: np.ndarray,
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]],
) -> None:
    patch, meta = _extract_patch(row, canvas_cache=canvas_cache, patch_size_px=540)
    ppd = float(meta["patch_ppd"])
    half = max(2, int(round(ppd)))
    cy, cx = np.asarray(patch.shape) // 2
    aperture = np.asarray(patch[cy - half : cy + half, cx - half : cx + half])
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    center = np.asarray([aperture.shape[1] / 2.0, aperture.shape[0] / 2.0])
    screen_trace = np.column_stack(
        [center[0] + centered[:, 0] * ppd, center[1] - centered[:, 1] * ppd]
    )
    ax.imshow(aperture, cmap="gray", origin="upper")
    ax.plot(screen_trace[:, 0], screen_trace[:, 1], color="#f28e2b", lw=0.9)
    ax.scatter(screen_trace[0, 0], screen_trace[0, 1], color="#2ca02c", s=9, zorder=4)
    angle = math.radians(float(row.corrected_edge_axis_array_deg))
    delta = np.asarray([math.cos(angle), math.sin(angle)]) * 0.72 * ppd
    ax.plot(
        [center[0] - delta[0], center[0] + delta[0]],
        [center[1] - delta[1], center[1] + delta[1]],
        color="#00bcd4", lw=1.5,
    )
    one_degree_half = 0.5 * ppd
    ax.add_patch(
        Rectangle(
            (center[0] - one_degree_half, center[1] - one_degree_half),
            2 * one_degree_half, 2 * one_degree_half,
            fill=False, edgecolor="white", linestyle="--", linewidth=0.6, alpha=0.75,
        )
    )
    ax.set_xlim(0, aperture.shape[1] - 1)
    ax.set_ylim(aperture.shape[0] - 1, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"pair {int(row.pair_index):03d}  coh={float(row.image_orientation_coherence):.3f}\n"
        f"{row.session} tr{int(row.trial_idx)}  cluster {str(row.local_window_cluster_id)[-4:]}",
        fontsize=6.6,
    )


def _contact_figure(
    rows: pd.DataFrame,
    trace_xy: np.ndarray,
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(CONTACT_ROWS, CONTACT_COLUMNS, figsize=(12.2, 10.2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row_tuple in zip(axes.ravel(), rows.itertuples(index=False)):
        row = pd.Series(row_tuple._asdict())
        _draw_contact_tile(ax, row, trace_xy[int(row.trace_bank_index)], canvas_cache)
        ax.axis("on")
    fig.suptitle(
        title + "\n"
        "2° contour-estimation window; dashed white: central 1°; cyan: corrected contour axis; orange: native trace",
        fontsize=10.5, weight="bold",
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.02, hspace=0.31, wspace=0.08)
    return fig


def _render_contact_sheets(primary: pd.DataFrame, trace_xy: np.ndarray) -> None:
    ordered = primary.sort_values(
        ["stimulus_image_sha256", "local_window_cluster_id", "session", "trial_idx", "pair_index"],
        kind="stable",
    ).reset_index(drop=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    per_page = CONTACT_ROWS * CONTACT_COLUMNS
    pdf_path = OUT_ROOT / "primary_strong_contour_blinded_contact_sheet_all.pdf"
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(ordered), per_page):
            stop = min(start + per_page, len(ordered))
            print(f"[panel-g-input] render contact tiles {start + 1}-{stop}/{len(ordered)}", flush=True)
            fig = _contact_figure(
                ordered.iloc[start:stop], trace_xy, canvas_cache,
                f"Outcome-blind primary cohort — page {start // per_page + 1}",
            )
            pdf.savefig(fig)
            plt.close(fig)
    compact_indices = np.unique(
        np.linspace(0, max(len(ordered) - 1, 0), min(20, len(ordered))).round().astype(int)
    )
    compact = _contact_figure(
        ordered.iloc[compact_indices], trace_xy, canvas_cache,
        "Outcome-blind primary cohort — deterministic overview",
    )
    compact.savefig(OUT_ROOT / "primary_strong_contour_blinded_contact_sheet_overview.png", dpi=220)
    plt.close(compact)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cohort_path = OLD_NATIVE_ROOT / "exact_pair_cohort_manifest.csv"
    trace_path = BANK_ROOT / "trace_xy.npy"
    cohort = pd.read_csv(cohort_path)
    trace_xy = np.load(trace_path)
    cohort = _add_source_columns(cohort)
    cohort = _basic_gate_columns(cohort, trace_xy)
    cohort = _add_image_identity(cohort)

    primary = cohort[cohort["primary_eligible"]].copy().reset_index(drop=True)
    development = cohort[cohort["development_eligible"]].copy().reset_index(drop=True)
    primary.insert(0, "replacement_cohort_index", np.arange(len(primary), dtype=int))
    development.insert(0, "development_cohort_index", np.arange(len(development), dtype=int))

    cohort.to_csv(OUT_ROOT / "all_native_pairs_input_qc_manifest.csv", index=False)
    primary.to_csv(OUT_ROOT / "primary_strong_contour_native_pair_manifest.csv", index=False)
    development.to_csv(OUT_ROOT / "development_strong_contour_native_pair_manifest.csv", index=False)
    flow = _flow_table(cohort)
    flow.to_csv(OUT_ROOT / "cohort_flow_counts.csv", index=False)
    repeats = pd.concat(
        [
            _repeat_summary(cohort, "primary_eligible"),
            _repeat_summary(cohort, "development_eligible"),
        ],
        ignore_index=True,
    )
    repeats.to_csv(OUT_ROOT / "image_repeat_structure_summary.csv", index=False)
    review = primary[
        [
            "replacement_cohort_index", "pair_index", "trace_bank_index", "source_row",
            "session", "subject", "trial_idx", "image_orientation_coherence",
            "corrected_edge_axis_array_deg", "stimulus_image_sha256",
            "local_window_cluster_id",
        ]
    ].copy()
    review["human_axis_qc"] = ""
    review["human_contour_strength_qc"] = ""
    review["human_exclusion_reason"] = ""
    review.to_csv(OUT_ROOT / "primary_blinded_human_review_template.csv", index=False)
    _render_contact_sheets(primary, trace_xy)

    metadata = {
        "analysis": "panel_g_direct_replacement_input_checkpoint",
        "artifact_type": "outcome_blind_map_first_input_checkpoint",
        "selection_contract": {
            "native_exact_image_fixation_pairs": True,
            "n_timepoints": N_TIMEPOINTS,
            "native_trace_max_abs_error_max": 1e-6,
            "drift_only_primary": True,
            "image_feature_ok": True,
            "minimum_patch_fraction_inside_image": MIN_INSIDE_FRACTION,
            "finite_image_measurements_required": True,
            "corrected_array_axis_consistency_required": True,
            "development_minimum_orientation_coherence": DEVELOPMENT_COHERENCE_MIN,
            "primary_minimum_orientation_coherence": PRIMARY_COHERENCE_MIN,
            "historical_surrogate_used": False,
            "fresh_model_outcomes_used": False,
        },
        "repeat_identity_contract": {
            "stimulus_image_identity": "SHA256 of native BackImageTrial.get_image float32 pixels and shape",
            "display_trial_identity": "session plus trial index",
            "local_window_cluster": f"connected components within {LOCAL_CLUSTER_RADIUS_DEG:.1f} deg in stimulus-local coordinates, grouped by stimulus image hash",
        },
        "counts": {
            "all_native_pairs": int(len(cohort)),
            "development_pairs": int(len(development)),
            "primary_pairs": int(len(primary)),
            "primary_sessions": int(primary["session"].nunique()),
            "primary_display_trials": int(primary["display_trial_key"].nunique()),
            "primary_stimulus_images": int(primary["stimulus_image_sha256"].nunique()),
            "primary_local_window_clusters": int(primary["local_window_cluster_id"].nunique()),
        },
        "inputs": {
            "native_cohort_manifest": cohort_path,
            "native_cohort_manifest_sha256": _file_sha256(cohort_path),
            "trace_xy": trace_path,
            "trace_xy_sha256": _file_sha256(trace_path),
            "source_windows": SOURCE_WINDOWS,
            "source_windows_sha256": _file_sha256(SOURCE_WINDOWS),
        },
        "fresh_model_evaluation_performed": False,
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _write_json(OUT_ROOT / "input_checkpoint_metadata.json", metadata)
    print("\n[panel-g-input] cohort flow", flush=True)
    print(flow.to_string(index=False), flush=True)
    print("\n[panel-g-input] repeat structure", flush=True)
    print(repeats.to_string(index=False), flush=True)
    print(f"\n[panel-g-input] wrote input checkpoint to {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()

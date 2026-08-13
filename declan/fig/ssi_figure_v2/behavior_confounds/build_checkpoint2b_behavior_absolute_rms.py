#!/usr/bin/env python3
"""Checkpoint 2B: local versus same-image 5-degree patches, behavior only.

This reuses the image extraction, patch QC, and display helpers from the first
Checkpoint-2B attempt, but deliberately excludes its Panel-G interpolation.
For every local or offset contour axis, the primary behavior statistic is the
absolute difference between contour-parallel and contour-normal RMS spread.
The measured trace is compared with the same shared set of axial rotations at
every valid location in a window.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint1_reference_frame_examples as cp1,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2_pairing_null_examples as cp2,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2b_local_offset_examples as first_attempt,
)
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = first_attempt.OUT_DIR / "checkpoint2b_behavior_absolute_rms_v1"
SOURCE_WINDOWS = first_attempt.SOURCE_WINDOWS
SELECTED_WINDOWS = first_attempt.SELECTED_WINDOWS

N_ROTATIONS = first_attempt.N_ROTATIONS
SEED = first_attempt.SEED

LOCAL_COLOR = first_attempt.LOCAL_COLOR
PRESERVE_COLOR = first_attempt.PRESERVE_COLOR
CHANGE_COLOR = first_attempt.CHANGE_COLOR
OFFSET_COLOR = first_attempt.OFFSET_COLOR
NULL_COLOR = "#7a3b9a"


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
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rms_components(trace: np.ndarray, edge_axis_deg: float) -> dict[str, float]:
    """Return full-window positional RMS in contour coordinates, in arcmin."""
    centered = np.asarray(trace, dtype=np.float64) - np.mean(
        trace, axis=0, keepdims=True
    )
    parallel_axis = cp1.axis_vector(float(edge_axis_deg))
    normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
    parallel = centered @ parallel_axis
    normal = centered @ normal_axis
    parallel_rms = float(np.sqrt(np.mean(parallel * parallel)) * 60.0)
    normal_rms = float(np.sqrt(np.mean(normal * normal)) * 60.0)
    return {
        "parallel_rms_arcmin": parallel_rms,
        "normal_rms_arcmin": normal_rms,
        "parallel_minus_normal_rms_arcmin": parallel_rms - normal_rms,
    }


def _score_locations(
    target: pd.Series,
    locations: pd.DataFrame,
    trace: np.ndarray,
    rotation_angles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rotated_traces = [
        cp2.rotate_trace(trace, float(angle)) for angle in rotation_angles
    ]
    value_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []

    for location in locations.itertuples(index=False):
        edge_axis = float(location.image_edge_axis_deg)
        valid = bool(location.passes_primary_patch_qc) and math.isfinite(edge_axis)
        if valid:
            real = _rms_components(trace, edge_axis)
            rotations = [_rms_components(item, edge_axis) for item in rotated_traces]
            null_delta = np.asarray(
                [item["parallel_minus_normal_rms_arcmin"] for item in rotations],
                dtype=np.float64,
            )
            null_mean = float(np.mean(null_delta))
            null_sd = float(np.std(null_delta, ddof=1))
            real_delta = float(real["parallel_minus_normal_rms_arcmin"])
            advantage = real_delta - null_mean
            for rotation_index, (angle, item) in enumerate(
                zip(rotation_angles, rotations, strict=True)
            ):
                rotation_rows.append(
                    {
                        "example_role": str(target["example_role"]),
                        "location_id": str(location.location_id),
                        "rotation_index": int(rotation_index),
                        "rotation_angle_deg": float(np.degrees(angle)),
                        **item,
                    }
                )
        else:
            real = {
                "parallel_rms_arcmin": float("nan"),
                "normal_rms_arcmin": float("nan"),
                "parallel_minus_normal_rms_arcmin": float("nan"),
            }
            null_delta = np.asarray([], dtype=np.float64)
            null_mean = null_sd = advantage = float("nan")

        value_rows.append(
            {
                "example_role": str(target["example_role"]),
                "session": str(target["session"]),
                "trial_idx": int(target["trial_idx"]),
                "global_start": int(target["global_start"]),
                "global_stop": int(target["global_stop"]),
                "location_id": str(location.location_id),
                "location_kind": str(location.location_kind),
                "offset_angle_gaze_deg": float(location.offset_angle_gaze_deg),
                "axis_relationship": str(location.axis_relationship),
                "passes_primary_patch_qc": bool(location.passes_primary_patch_qc),
                "image_edge_axis_deg": edge_axis,
                "image_orientation_coherence": float(
                    location.image_orientation_coherence
                ),
                "local_offset_axis_delta_deg": float(
                    location.local_offset_axis_delta_deg
                ),
                **real,
                "rotation_mean_parallel_minus_normal_rms_arcmin": null_mean,
                "rotation_sd_parallel_minus_normal_rms_arcmin": null_sd,
                "rotation_se_parallel_minus_normal_rms_arcmin": (
                    null_sd / math.sqrt(len(null_delta)) if len(null_delta) else float("nan")
                ),
                "rotation_ci95_low_parallel_minus_normal_rms_arcmin": (
                    float(np.percentile(null_delta, 2.5))
                    if len(null_delta)
                    else float("nan")
                ),
                "rotation_ci95_high_parallel_minus_normal_rms_arcmin": (
                    float(np.percentile(null_delta, 97.5))
                    if len(null_delta)
                    else float("nan")
                ),
                "real_minus_rotation_mean_rms_advantage_arcmin": advantage,
                "n_rotation_draws": int(len(rotation_angles)),
            }
        )

    values = pd.DataFrame(value_rows)
    local = values[values["location_kind"].astype(str).eq("local")]
    offsets = values[
        values["location_kind"].astype(str).eq("offset")
        & values["passes_primary_patch_qc"].astype(bool)
    ]
    if len(local) != 1 or not bool(local["passes_primary_patch_qc"].iloc[0]):
        raise RuntimeError(f"Expected one valid local patch for {target['example_role']}")
    local_advantage = float(
        local["real_minus_rotation_mean_rms_advantage_arcmin"].iloc[0]
    )
    offset_advantage = pd.to_numeric(
        offsets["real_minus_rotation_mean_rms_advantage_arcmin"], errors="coerce"
    ).to_numpy(dtype=float)
    offset_mean = float(np.mean(offset_advantage))
    values["D_local_arcmin"] = local_advantage
    values["D_offset_mean_arcmin"] = offset_mean
    values["D_locality_arcmin"] = local_advantage - offset_mean
    values["n_valid_offsets"] = int(len(offsets))
    return values, pd.DataFrame(rotation_rows)


def _display_selection(locations: pd.DataFrame) -> pd.DataFrame:
    local = locations[locations["location_kind"].astype(str).eq("local")].iloc[0]
    preserving, changing, preserve_label, change_label = (
        first_attempt._representative_offsets(locations)
    )
    rows = []
    preserve_is_threshold_case = (
        float(preserving["local_offset_axis_delta_deg"]) <= 10.0
    )
    change_is_threshold_case = float(changing["local_offset_axis_delta_deg"]) >= 30.0
    for row, display_slot, display_role, criterion in (
        (local, "local", "local", "true gaze-centered patch"),
        (
            preserving,
            "preserve_slot",
            (
                "orientation_preserving"
                if preserve_is_threshold_case
                else "closest_axis_no_preserving_case"
            ),
            f"{preserve_label}; smallest valid axial difference",
        ),
        (
            changing,
            "change_slot",
            (
                "orientation_changing"
                if change_is_threshold_case
                else "farthest_axis_no_changing_case"
            ),
            f"{change_label}; largest valid axial difference",
        ),
    ):
        rows.append(
            {
                "example_role": str(row["example_role"]),
                "display_slot": display_slot,
                "display_role": display_role,
                "location_id": str(row["location_id"]),
                "selection_criterion": criterion,
                "offset_angle_gaze_deg": float(row["offset_angle_gaze_deg"]),
                "local_offset_axis_delta_deg": float(
                    row["local_offset_axis_delta_deg"]
                ),
                "image_orientation_coherence": float(
                    row["image_orientation_coherence"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_advantage(ax: plt.Axes, values: pd.DataFrame) -> None:
    offsets = values[
        values["location_kind"].astype(str).eq("offset")
        & values["passes_primary_patch_qc"].astype(bool)
    ]
    local = values[values["location_kind"].astype(str).eq("local")].iloc[0]
    column = "real_minus_rotation_mean_rms_advantage_arcmin"
    ax.axhline(0.0, color="#8b9299", lw=0.7, ls=":")
    ax.scatter(
        offsets["offset_angle_gaze_deg"], offsets[column], color=OFFSET_COLOR, s=22
    )
    ax.scatter([-35], [local[column]], color=LOCAL_COLOR, marker="*", s=60)
    ax.set_xlim(-50, 365)
    ax.set_xticks(
        [-35, 0, 90, 180, 270, 360], ["local", "0", "90", "180", "270", "360"]
    )
    ax.set_xlabel("offset direction (deg)", fontsize=6.8)
    ax.set_ylabel("real − rotation-mean ΔRMS (arcmin)", fontsize=6.8)
    ax.tick_params(labelsize=6.1)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_title(
        f"locality = {float(local['D_locality_arcmin']):+.2f} arcmin",
        fontsize=8.2,
        weight="bold",
    )


def _plot_displayed_nulls(
    ax: plt.Axes, locations: pd.DataFrame, values: pd.DataFrame
) -> None:
    selection = _display_selection(locations)
    order = ["local", "preserve_slot", "change_slot"]
    colors = [LOCAL_COLOR, PRESERVE_COLOR, CHANGE_COLOR]
    labels = ["local"]
    preserve_role = str(
        selection.loc[selection["display_slot"].eq("preserve_slot"), "display_role"].iloc[0]
    )
    change_role = str(
        selection.loc[selection["display_slot"].eq("change_slot"), "display_role"].iloc[0]
    )
    labels.append("axis\npreserved" if preserve_role == "orientation_preserving" else "closest\naxis")
    labels.append("axis\nchanged" if change_role == "orientation_changing" else "farthest\naxis")
    for x, display_slot, color in zip(range(3), order, colors, strict=True):
        location_id = str(
            selection.loc[selection["display_slot"].eq(display_slot), "location_id"].iloc[0]
        )
        row = values[values["location_id"].astype(str).eq(location_id)].iloc[0]
        null_mean = float(
            row["rotation_mean_parallel_minus_normal_rms_arcmin"]
        )
        low = float(row["rotation_ci95_low_parallel_minus_normal_rms_arcmin"])
        high = float(row["rotation_ci95_high_parallel_minus_normal_rms_arcmin"])
        ax.errorbar(
            [x - 0.10],
            [null_mean],
            yerr=[[null_mean - low], [high - null_mean]],
            color=NULL_COLOR,
            marker="o",
            ms=3.8,
            capsize=2.2,
            lw=0.9,
        )
        real_delta = float(row["parallel_minus_normal_rms_arcmin"])
        ax.scatter([x + 0.10], [real_delta], color=color, marker="D", s=25, zorder=4)
        ax.plot([x - 0.10, x + 0.10], [null_mean, real_delta], color="#aab0b6", lw=0.7)
    ax.axhline(0.0, color="#8b9299", lw=0.7, ls=":")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("ΔRMS: parallel − normal (arcmin)", fontsize=6.8)
    ax.tick_params(labelsize=6.1)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_title("measured ◆ vs rotation null ● (95%)", fontsize=8.2, weight="bold")


def _render(
    targets: list[pd.Series],
    location_tables: list[pd.DataFrame],
    value_tables: list[pd.DataFrame],
    traces: list[np.ndarray],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(targets),
        6,
        figsize=(17.2, 15.7),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0, 1.12, 1.22]},
    )
    for row_index, (target, locations, values, trace) in enumerate(
        zip(targets, location_tables, value_tables, traces, strict=True)
    ):
        local = locations[locations["location_kind"].astype(str).eq("local")].iloc[0]
        preserving, changing, preserve_label, change_label = (
            first_attempt._representative_offsets(locations)
        )
        first_attempt._plot_overview(axes[row_index, 0], locations)
        axes[row_index, 0].set_ylabel(
            f"{row_index + 1}. {cp1.ROLE_LABEL[str(target['example_role'])]}\n"
            f"{target['subject']} | tr {int(target['trial_idx'])}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=8.0,
            weight="bold",
        )
        first_attempt._plot_patch_with_trace(
            axes[row_index, 1],
            local,
            trace,
            color=LOCAL_COLOR,
            title="actual local patch",
        )
        first_attempt._plot_patch_with_trace(
            axes[row_index, 2],
            preserving,
            trace,
            color=PRESERVE_COLOR,
            title=(
                f"{preserve_label}\n"
                f"Δaxis={float(preserving['local_offset_axis_delta_deg']):.1f}°"
            ),
        )
        first_attempt._plot_patch_with_trace(
            axes[row_index, 3],
            changing,
            trace,
            color=CHANGE_COLOR,
            title=(
                f"{change_label}\n"
                f"Δaxis={float(changing['local_offset_axis_delta_deg']):.1f}°"
            ),
        )
        _plot_advantage(axes[row_index, 4], values)
        _plot_displayed_nulls(axes[row_index, 5], locations, values)

    fig.suptitle(
        "Figure 4 behavior audit — Checkpoint 2B: does trajectory–contour alignment localize to the true patch?\n"
        "One measured 128-sample trace is held fixed; every valid same-image 5° patch uses the same 256 trajectory rotations",
        y=0.997,
        fontsize=12.6,
        weight="bold",
    )
    fig.text(
        0.35,
        0.952,
        "OBSERVED IMAGE CONTENT + SAME MEASURED TRACE",
        ha="center",
        fontsize=10.2,
        weight="bold",
    )
    fig.text(
        0.83,
        0.952,
        "BEHAVIOR-ONLY ROTATION TEST",
        ha="center",
        fontsize=10.2,
        weight="bold",
    )
    fig.subplots_adjust(
        left=0.13, right=0.992, top=0.925, bottom=0.05, hspace=0.52, wspace=0.40
    )
    boundary = 0.5 * (
        axes[0, 3].get_position().x1 + axes[0, 4].get_position().x0
    )
    fig.add_artist(
        plt.Line2D(
            [boundary, boundary],
            [0.04, 0.95],
            transform=fig.transFigure,
            color="#a6adb5",
            lw=0.9,
        )
    )
    fig.text(
        0.99,
        0.014,
        "Primary statistic: ΔRMS = full-window contour-parallel RMS − contour-normal RMS (arcmin). "
        "D = measured ΔRMS − mean rotated ΔRMS. No SSI interpolation or neural-model output is used.",
        ha="right",
        fontsize=6.9,
        color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint2b_behavior_absolute_rms_examples.png", dpi=220)
    fig.savefig(out_dir / "checkpoint2b_behavior_absolute_rms_examples.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-rotations", type=int, default=N_ROTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_WINDOWS)
    selected = pd.read_csv(SELECTED_WINDOWS)
    rng = np.random.default_rng(int(args.seed))

    targets: list[pd.Series] = []
    locations_all: list[pd.DataFrame] = []
    values_all: list[pd.DataFrame] = []
    rotations_all: list[pd.DataFrame] = []
    traces: list[np.ndarray] = []
    display_all: list[pd.DataFrame] = []

    for selected_row in selected.itertuples(index=False):
        target = first_attempt._find_target(
            source, pd.Series(selected_row._asdict())
        )
        print(
            f"[checkpoint2b-absolute] {target['example_role']} "
            f"({target['session']} trial {int(target['trial_idx'])})",
            flush=True,
        )
        trace = np.asarray(_window_trace(target), dtype=np.float64)
        if trace.shape != (128, 2) or not np.isfinite(trace).all():
            raise RuntimeError(
                f"Expected one native 128x2 trace for {target['example_role']}, got {trace.shape}"
            )
        locations = pd.DataFrame(first_attempt._location_rows(target))
        angles = rng.uniform(0.0, np.pi, size=int(args.n_rotations))
        values, rotation_values = _score_locations(
            target, locations, trace, angles
        )
        locations["rotation_seed"] = int(args.seed)
        locations["n_rotation_draws"] = int(args.n_rotations)
        locations["rotation_angles_hash"] = hashlib.sha256(
            np.asarray(angles, dtype=np.float64).tobytes()
        ).hexdigest()[:20]

        targets.append(target)
        locations_all.append(locations)
        values_all.append(values)
        rotations_all.append(rotation_values)
        traces.append(trace)
        display_all.append(_display_selection(locations))

    manifest = pd.concat(locations_all, ignore_index=True)
    values = pd.concat(values_all, ignore_index=True)
    rotations = pd.concat(rotations_all, ignore_index=True)
    display = pd.concat(display_all, ignore_index=True)
    summary = values[values["location_kind"].astype(str).eq("local")][
        [
            "example_role",
            "n_valid_offsets",
            "D_local_arcmin",
            "D_offset_mean_arcmin",
            "D_locality_arcmin",
            "parallel_rms_arcmin",
            "normal_rms_arcmin",
            "parallel_minus_normal_rms_arcmin",
            "rotation_mean_parallel_minus_normal_rms_arcmin",
        ]
    ].copy()
    axis_relationship_summary = (
        values[
            values["location_kind"].astype(str).eq("offset")
            & values["passes_primary_patch_qc"].astype(bool)
        ]
        .groupby(["example_role", "axis_relationship"], as_index=False)
        .agg(
            n_offsets=("location_id", "size"),
            mean_D_arcmin=(
                "real_minus_rotation_mean_rms_advantage_arcmin",
                "mean",
            ),
            mean_axis_delta_deg=("local_offset_axis_delta_deg", "mean"),
        )
    )

    manifest.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_offset_manifest.csv", index=False
    )
    values.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_values.csv", index=False
    )
    rotations.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_rotation_values.csv", index=False
    )
    display.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_display_selection.csv", index=False
    )
    summary.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_locality_summary.csv", index=False
    )
    axis_relationship_summary.to_csv(
        out_dir / "checkpoint2b_behavior_absolute_rms_axis_relationship_summary.csv",
        index=False,
    )
    _render(targets, locations_all, values_all, traces, out_dir)

    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": "2B behavior-only absolute-RMS correction",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "neural_model_evaluated": False,
        "ssi_interpolation_used": False,
        "input_windows": SOURCE_WINDOWS,
        "checkpoint1_selection": SELECTED_WINDOWS,
        "reused_implementation": (
            "build_checkpoint2b_local_offset_examples.py image extraction, patch QC, "
            "representative-offset selection, and patch rendering"
        ),
        "patch_contract": {
            "offset_distance_deg": first_attempt.OFFSET_DISTANCE_DEG,
            "offset_angles_gaze_deg": first_attempt.OFFSET_ANGLES_DEG,
            "feature_patch_radius_deg": first_attempt.PATCH_RADIUS_DEG,
            "min_patch_fraction_inside_image": first_attempt.MIN_PATCH_FRACTION_INSIDE,
            "max_patch_fraction_background": first_attempt.MAX_PATCH_FRACTION_BACKGROUND,
            "primary_offset_summary": (
                "unweighted mean over every valid pre-specified offset; displayed "
                "orientation-preserving/changing patches are diagnostics only"
            ),
        },
        "trace_contract": "native reviewed 128-sample measured trace; no temporal compression",
        "primary_statistic": (
            "parallel-minus-normal positional RMS in arcmin; D_location is measured "
            "minus mean rotation; D_locality is local D minus unweighted mean valid-offset D"
        ),
        "rotation_contract": (
            "per-window uniform axial rotations in [0,180 deg); identical draws reused "
            "for the local patch and every offset patch"
        ),
        "n_rotations": int(args.n_rotations),
        "seed": int(args.seed),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint2b_behavior_absolute_rms_examples.png",
            "checkpoint2b_behavior_absolute_rms_examples.pdf",
            "checkpoint2b_behavior_absolute_rms_offset_manifest.csv",
            "checkpoint2b_behavior_absolute_rms_values.csv",
            "checkpoint2b_behavior_absolute_rms_rotation_values.csv",
            "checkpoint2b_behavior_absolute_rms_display_selection.csv",
            "checkpoint2b_behavior_absolute_rms_locality_summary.csv",
            "checkpoint2b_behavior_absolute_rms_axis_relationship_summary.csv",
            "checkpoint2b_behavior_absolute_rms_run_metadata.json",
        ],
    }
    _write_json(
        out_dir / "checkpoint2b_behavior_absolute_rms_run_metadata.json", metadata
    )
    print(f"[checkpoint2b-absolute] wrote artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

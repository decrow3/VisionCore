#!/usr/bin/env python3
"""Map-first checkpoint 2: normal-vs-stabilized activation maps.

This is a targeted visualization render. It materializes only selected RR100
units for one image/trace pair and stops before population summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
    DEFAULT_DENSE_SFTF_CSV,
    DEFAULT_RUN_DIR,
    choose_representative_units,
    image_table_row_by_position,
    one_trace_from_source,
    source_row_by_id,
    speed_by_frame,
)
from declan.active_sensing_movie_information.analyze_temporal_remapping_sftf_power_explanation import (
    DEFAULT_PARAMETRIC_MODEL_CSV,
    load_parametric_fit_table,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    DEFAULT_SOURCE_CSV,
    load_source_rows,
)
from declan.active_sensing_movie_information.temporal_remapping import MODEL_RATE_HZ
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_02_activation_maps_v1"
EPS = 1e-8

# Okabe-Ito palette for categorical lines and markers.
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_RED = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#000000"
UNIT_COLORS = [OI_BLUE, OI_GREEN, OI_RED]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--dense-sftf-csv", type=Path, default=DEFAULT_DENSE_SFTF_CSV)
    parser.add_argument("--parametric-model-csv", type=Path, default=DEFAULT_PARAMETRIC_MODEL_CSV)
    parser.add_argument(
        "--preference-source",
        choices=("legacy", "parametric"),
        default="legacy",
        help="Use legacy unit metadata or canonical RR100 parametric SF/TF preferences for labels and tables.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--image-position", type=int, default=10)
    parser.add_argument("--trace-index", type=int, default=30)
    parser.add_argument("--example-label", type=str, default="")
    parser.add_argument("--units", type=str, default="", help="Optional comma-separated RR100 unit indices.")
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--frame-rate-hz", type=float, default=MODEL_RATE_HZ)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument(
        "--reuse-map-cache",
        type=Path,
        default=None,
        help="Optional checkpoint_02_selected_activation_maps.npz to re-render without recomputing maps.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_units(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def selected_units_from_args(
    args: argparse.Namespace,
    unit_table: pd.DataFrame,
    dense_sftf: pd.DataFrame,
) -> list[dict[str, Any]]:
    requested = parse_units(str(args.units))
    if not requested:
        return choose_representative_units(unit_table, dense_sftf)

    auto_rows = {int(row["unit_index"]): row for row in choose_representative_units(unit_table, dense_sftf)}
    out: list[dict[str, Any]] = []
    dense_lookup = dense_sftf.set_index("unit_index", drop=False) if not dense_sftf.empty else pd.DataFrame()
    unit_lookup = unit_table.set_index("unit_index", drop=False)
    parametric_lookup: pd.DataFrame | None = None
    if str(args.preference_source) == "parametric":
        parametric = load_parametric_fit_table(Path(args.parametric_model_csv), include_tf_edge_fits=True)
        parametric_lookup = parametric.set_index("unit_index", drop=False)
    for unit in requested:
        if parametric_lookup is not None:
            if int(unit) not in parametric_lookup.index:
                raise ValueError(f"Requested unit u{int(unit):03d} does not have a valid parametric SF/TF model.")
            meta = unit_lookup.loc[int(unit)]
            param = parametric_lookup.loc[int(unit)]
            row = {
                "unit_index": int(unit),
                "unit_label": str(meta.get("unit_label", f"u{int(unit):03d}")),
                "selection_role": "requested_unit",
                "sf_group": str(param.get("parametric_sf_group", "")),
                "preferred_sf_cpd": finite_float(param.get("fit_pref_sf_cpd")),
                "dense_fit_pref_tf_hz": finite_float(param.get("fit_pref_tf_hz")),
                "fit_pref_tf_hz": finite_float(param.get("fit_pref_tf_hz")),
                "dense_fit_status": "rr100_parametric_model",
                "preference_source": "parametric",
            }
        elif unit in auto_rows:
            row = dict(auto_rows[unit])
        else:
            meta = unit_lookup.loc[int(unit)]
            dense = dense_lookup.loc[int(unit)] if not dense_lookup.empty and int(unit) in dense_lookup.index else {}
            row = {
                "unit_index": int(unit),
                "unit_label": str(meta.get("unit_label", f"u{int(unit):03d}")),
                "selection_role": "requested_unit",
                "sf_group": str(meta.get("sf_group", "")),
                "preferred_sf_cpd": finite_float(meta.get("preferred_sf_cpd")),
                "dense_fit_pref_tf_hz": finite_float(dense.get("fit_pref_tf_hz") if isinstance(dense, pd.Series) else None),
                "fit_pref_tf_hz": finite_float(dense.get("fit_pref_tf_hz") if isinstance(dense, pd.Series) else None),
                "dense_fit_status": str(dense.get("fit_status", "") if isinstance(dense, pd.Series) else ""),
                "preference_source": "legacy",
            }
        out.append(row)
    return out


def choose_display_frames(speed: np.ndarray, n_frames: int) -> list[int]:
    speed = np.asarray(speed, dtype=np.float64)
    n = int(speed.shape[0])
    if n == 0:
        return []
    keep = {0, n - 1}
    order = np.argsort(speed)[::-1]
    min_gap = max(2, n // 12)
    for idx in order:
        frame = int(idx)
        if any(abs(frame - other) < min_gap for other in keep):
            continue
        keep.add(frame)
        if len(keep) >= int(n_frames):
            break
    if len(keep) < int(n_frames):
        for frame in np.linspace(0, n - 1, int(n_frames), dtype=int):
            keep.add(int(frame))
            if len(keep) >= int(n_frames):
                break
    return sorted(keep)[: int(n_frames)]


def instantaneous_bits_for_maps(maps: np.ndarray) -> np.ndarray:
    """Return framewise spatial SSI for maps with shape T x U x H x W."""
    y = np.maximum(np.asarray(maps, dtype=np.float64), 0.0)
    t, u, h, w = y.shape
    flat = y.reshape(t, u, h * w)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    return np.mean(gain * np.log2(gain + EPS), axis=2)


def unit_movie_ssi(maps: np.ndarray, *, bin_seconds: float) -> dict[str, np.ndarray]:
    y = np.maximum(np.asarray(maps, dtype=np.float64), 0.0)
    flat = y.reshape(y.shape[0], y.shape[1], -1)
    rbar = np.mean(flat, axis=2)
    bits_t = instantaneous_bits_for_maps(y)
    weights = rbar * float(bin_seconds)
    expected = np.sum(weights, axis=0)
    bits = np.sum(bits_t * weights, axis=0) / np.maximum(expected, EPS)
    return {
        "unit_bits_per_spike": bits.astype(np.float32),
        "unit_expected_spikes": expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
        "frame_mean_rate": rbar.astype(np.float32),
        "frame_bits_per_spike": bits_t.astype(np.float32),
    }


def selected_rr100_maps_for_trace(
    scorer: CanonicalTwinScorer,
    population_view: Any,
    patch: np.ndarray,
    trace: np.ndarray,
    selected_units: list[int],
    *,
    n_timepoints: int,
) -> np.ndarray:
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None, :, :],
        (trace.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
    ).copy()
    eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    stim = scorer.common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=scorer.common.PPD,
        scale_factor=1.0,
        n_lags=scorer.common.N_LAGS,
        out_size=scorer.common.OUT_SIZE,
    )
    stim = (stim - 127.0) / 255.0
    device = next(scorer.ctx.model.model.parameters()).device
    scorer.ctx.model.model.eval()
    scorer.ctx.readout.eval()
    chunks: list[np.ndarray] = []
    unit_indices = np.asarray(selected_units, dtype=np.int64)
    with scorer.torch.no_grad():
        for t_start in range(0, int(stim.shape[0]), int(scorer.batch_size)):
            t_end = min(t_start + int(scorer.batch_size), int(stim.shape[0]))
            x = stim[t_start:t_end].to(device)
            full_map = scorer.compute_rate_map(scorer.ctx.model, scorer.ctx.readout, x)
            rr100 = apply_population_view(full_map, population_view).clamp_min(0.0)
            selected = rr100[:, unit_indices]
            chunks.append(selected.detach().cpu().numpy().astype(np.float32, copy=False))
            del x, full_map, rr100, selected
            if str(device).startswith("cuda"):
                scorer.torch.cuda.empty_cache()
    maps = np.concatenate(chunks, axis=0)
    return _align_response_to_trace(maps, int(n_timepoints)).astype(np.float32, copy=False)


def render_maps(
    *,
    out_dir: Path,
    example_label: str,
    patch: np.ndarray,
    patch_meta: dict[str, Any],
    image_source_row: int,
    trace: np.ndarray,
    maps_by_condition: dict[str, np.ndarray],
    selected_rows: list[dict[str, Any]],
    frame_indices: list[int],
    speed: np.ndarray,
    population_static: float,
    population_original: float,
    frame_rate_hz: float,
    dpi: int,
) -> tuple[Path, Path]:
    units = [int(row["unit_index"]) for row in selected_rows]
    unit_labels = [str(row["unit_label"]) for row in selected_rows]
    static = maps_by_condition["stabilized"]
    normal = maps_by_condition["normal"]
    diff = normal - static
    n_units = len(units)
    n_frames = len(frame_indices)
    rows_per_unit = 3
    fig_height = 2.45 + 1.2 * rows_per_unit * n_units
    fig_width = max(12.0, 1.75 * n_frames + 4.8)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(
        rows_per_unit * n_units + 1,
        n_frames + 2,
        left=0.055,
        right=0.965,
        top=0.905,
        bottom=0.045,
        hspace=0.12,
        wspace=0.08,
        height_ratios=[1.45] + [1.0] * rows_per_unit * n_units,
        width_ratios=[1.45] + [1.0] * n_frames + [0.16],
    )
    ax_patch = fig.add_subplot(gs[0, 0])
    patch_arr = np.asarray(patch, dtype=np.float32)
    pvmin, pvmax = np.nanpercentile(patch_arr, [1.0, 99.0])
    ax_patch.imshow(patch_arr, cmap="gray", vmin=pvmin, vmax=pvmax)
    ppd = float(patch_meta["patch_ppd"])
    centered = np.asarray(trace, dtype=np.float32) - np.mean(trace, axis=0, keepdims=True)
    xy_px = np.column_stack(
        [
            patch_arr.shape[1] / 2.0 + centered[:, 0] * ppd,
            patch_arr.shape[0] / 2.0 - centered[:, 1] * ppd,
        ]
    )
    selected_xy = xy_px[np.asarray(frame_indices, dtype=np.int64)]
    ax_patch.plot(xy_px[:, 0], xy_px[:, 1], color=OI_SKY, lw=1.4, label="retinal path")
    ax_patch.scatter(
        selected_xy[:, 0],
        selected_xy[:, 1],
        s=34,
        marker="o",
        facecolor=OI_ORANGE,
        edgecolor=OI_BLACK,
        linewidth=0.45,
        zorder=3,
        label="map frames",
    )
    ax_patch.scatter(
        [xy_px[0, 0]],
        [xy_px[0, 1]],
        s=42,
        marker="s",
        color=OI_BLUE,
        zorder=4,
        label="start f0",
    )
    ax_patch.scatter(
        [xy_px[-1, 0]],
        [xy_px[-1, 1]],
        s=50,
        marker="^",
        color=OI_RED,
        zorder=4,
        label=f"end f{trace.shape[0] - 1}",
    )
    for frame in frame_indices:
        x, y = xy_px[int(frame)]
        ax_patch.text(
            x + 4,
            y + 4,
            f"f{int(frame)}",
            fontsize=6.6,
            color=OI_BLACK,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax_patch.set_title(f"patch + path | source row {int(image_source_row)}", fontsize=9)
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    ax_patch.legend(frameon=True, framealpha=0.82, fontsize=6.8, loc="lower right")

    ax_speed = fig.add_subplot(gs[0, 1 : n_frames + 1])
    time_ms = np.arange(speed.shape[0]) * 1000.0 / float(frame_rate_hz)
    ax_speed.plot(time_ms, speed, color=OI_BLUE, lw=1.7, label="normal speed")
    ax_speed.plot(time_ms, np.zeros_like(speed), color="0.55", lw=1.1, label="stabilized")
    ax_speed.scatter(
        time_ms[np.asarray(frame_indices, dtype=np.int64)],
        speed[np.asarray(frame_indices, dtype=np.int64)],
        s=36,
        marker="o",
        facecolor=OI_ORANGE,
        edgecolor=OI_BLACK,
        linewidth=0.45,
        zorder=4,
        label="map frames",
    )
    for frame in frame_indices:
        ax_speed.axvline(time_ms[int(frame)], color=OI_ORANGE, lw=1.0, alpha=0.8)
        ax_speed.text(
            time_ms[int(frame)],
            0.96,
            f"f{int(frame)}",
            transform=ax_speed.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=OI_ORANGE,
        )
    ax_speed.set_title(
        f"{example_label or 'example'}: selected frames for map comparison; "
        f"population SSI normal-static {population_original - population_static:+.4f}",
        fontsize=10,
    )
    ax_speed.set_xlabel("time (ms)")
    ax_speed.set_ylabel("speed (deg/s)")
    ax_speed.grid(True, color="0.9", lw=0.7)
    ax_speed.legend(frameon=False, fontsize=8, loc="upper left")

    for unit_pos, row in enumerate(selected_rows):
        unit_label = str(row["unit_label"])
        sf = finite_float(row.get("preferred_sf_cpd"))
        tf_pref = finite_float(row.get("dense_fit_pref_tf_hz"))
        meta_label = f"{unit_label} | SF {sf:.3g} cpd"
        if math.isfinite(tf_pref):
            meta_label += f" | TF pref {tf_pref:.2g} Hz"

        unit_static = static[:, unit_pos]
        unit_normal = normal[:, unit_pos]
        unit_diff = diff[:, unit_pos]
        act_vmin, act_vmax = np.nanpercentile(np.concatenate([unit_static.ravel(), unit_normal.ravel()]), [1.0, 99.5])
        if not math.isfinite(float(act_vmax)) or float(act_vmax) <= float(act_vmin):
            act_vmin, act_vmax = 0.0, max(float(np.nanmax([unit_static, unit_normal])), EPS)
        diff_abs = float(np.nanpercentile(np.abs(unit_diff), 99.0))
        diff_abs = max(diff_abs, EPS)
        row_base = 1 + unit_pos * rows_per_unit
        for local_row, (label, stack, cmap, vmin, vmax) in enumerate(
            [
                ("static", unit_static, "cividis", act_vmin, act_vmax),
                ("normal", unit_normal, "cividis", act_vmin, act_vmax),
                ("normal - static", unit_diff, "PuOr_r", -diff_abs, diff_abs),
            ]
        ):
            first_image = None
            label_ax = fig.add_subplot(gs[row_base + local_row, 0])
            label_ax.axis("off")
            if local_row == 0:
                label_ax.text(1.0, 0.72, meta_label, ha="right", va="center", fontsize=8.5, fontweight="bold")
            label_ax.text(1.0, 0.28, label, ha="right", va="center", fontsize=8.5)
            for col, frame in enumerate(frame_indices):
                ax = fig.add_subplot(gs[row_base + local_row, col + 1])
                image = ax.imshow(stack[int(frame)], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                if first_image is None:
                    first_image = image
                ax.set_xticks([])
                ax.set_yticks([])
                if row_base + local_row == 1:
                    ax.set_title(f"frame {int(frame)}\n{speed[int(frame)]:.1f} deg/s", fontsize=8)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_edgecolor("0.45")
            if first_image is not None and local_row == 0:
                cax = fig.add_subplot(gs[row_base : row_base + 2, n_frames + 1])
                cbar = fig.colorbar(first_image, cax=cax)
                cbar.ax.tick_params(labelsize=6.5, length=2)
                cbar.set_label("model rate\nstatic & normal", fontsize=7)
                cbar.set_ticks([float(act_vmin), float(act_vmax)])
                cbar.set_ticklabels([f"{float(act_vmin):.2g}", f"{float(act_vmax):.2g}"])
            elif first_image is not None and local_row == 2:
                cax = fig.add_subplot(gs[row_base + local_row, n_frames + 1])
                cbar = fig.colorbar(first_image, cax=cax)
                cbar.ax.tick_params(labelsize=6.5, length=2)
                cbar.set_label("normal-static\nmodel rate", fontsize=7)
                cbar.set_ticks([-diff_abs, 0.0, diff_abs])
                cbar.set_ticklabels([f"{-diff_abs:.2g}", "0", f"{diff_abs:.2g}"])

    fig.suptitle(
        "Checkpoint 2: raw RR100 activation maps; per-unit colorbars, symmetric difference scale",
        fontsize=13,
    )
    stem = "checkpoint_02_activation_map_tiles"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_timecourses(
    *,
    out_dir: Path,
    example_label: str,
    selected_rows: list[dict[str, Any]],
    metrics_by_condition: dict[str, dict[str, np.ndarray]],
    frame_indices: list[int],
    speed: np.ndarray,
    frame_rate_hz: float,
    dpi: int,
) -> tuple[Path, Path]:
    n_units = len(selected_rows)
    fig, axes = plt.subplots(n_units, 2, figsize=(11.5, max(3.2, 2.2 * n_units)), sharex=True, constrained_layout=True)
    if n_units == 1:
        axes = np.asarray([axes])
    time_ms = np.arange(speed.shape[0]) * 1000.0 / float(frame_rate_hz)
    for row_idx, row in enumerate(selected_rows):
        unit_label = str(row["unit_label"])
        for ax in axes[row_idx]:
            for frame in frame_indices:
                ax.axvline(time_ms[int(frame)], color=OI_ORANGE, lw=0.8, alpha=0.45)
            ax.grid(True, color="0.9", lw=0.7)
        for condition, color, linestyle in [("stabilized", "0.55", "--"), ("normal", OI_BLUE, "-")]:
            metrics = metrics_by_condition[condition]
            axes[row_idx, 0].plot(
                time_ms,
                metrics["frame_mean_rate"][:, row_idx],
                color=color,
                linestyle=linestyle,
                lw=1.5,
                label=condition,
            )
            axes[row_idx, 1].plot(
                time_ms,
                metrics["frame_bits_per_spike"][:, row_idx],
                color=color,
                linestyle=linestyle,
                lw=1.5,
                label=condition,
            )
        axes[row_idx, 0].set_ylabel(f"{unit_label}\nmean rate", rotation=0, ha="right", va="center", labelpad=34)
        axes[row_idx, 1].set_ylabel("instantaneous SSI\nbits/spike")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Spatial mean activation")
            axes[row_idx, 1].set_title("Instantaneous map SSI")
            axes[row_idx, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.suptitle(f"{example_label or 'example'}: map-derived timecourses for selected units", fontsize=12)
    png = out_dir / "checkpoint_02_map_metric_timecourses.png"
    pdf = out_dir / "checkpoint_02_map_metric_timecourses.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def metric_rows(
    *,
    selected_rows: list[dict[str, Any]],
    metrics_by_condition: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit_pos, row in enumerate(selected_rows):
        for condition, metrics in metrics_by_condition.items():
            rows.append(
                {
                    "condition": condition,
                    "unit_index": int(row["unit_index"]),
                    "unit_label": str(row["unit_label"]),
                    "selection_role": str(row.get("selection_role", "")),
                    "preferred_sf_cpd": finite_float(row.get("preferred_sf_cpd")),
                    "dense_fit_pref_tf_hz": finite_float(row.get("dense_fit_pref_tf_hz")),
                    "fit_pref_tf_hz": finite_float(row.get("fit_pref_tf_hz", row.get("dense_fit_pref_tf_hz"))),
                    "sf_group": str(row.get("sf_group", "")),
                    "preference_source": str(row.get("preference_source", "legacy")),
                    "movie_ssi_bits_per_spike": float(metrics["unit_bits_per_spike"][unit_pos]),
                    "movie_expected_spikes": float(metrics["unit_expected_spikes"][unit_pos]),
                    "movie_mean_rate": float(metrics["unit_mean_rate"][unit_pos]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_table = pd.read_csv(run_dir / "image_feature_table.csv")
    trace_table = pd.read_csv(run_dir / "trace_feature_table.csv")
    unit_table = pd.read_csv(run_dir / "unit_feature_table.csv")
    dense_sftf = pd.read_csv(args.dense_sftf_csv) if Path(args.dense_sftf_csv).exists() else pd.DataFrame()
    observations = pd.read_csv(run_dir / "retiming_population_observations.csv")
    source_rows = load_source_rows(Path(args.source_csv))

    image_row = image_table_row_by_position(image_table, int(args.image_position))
    trace_row = trace_table[trace_table["trace_index"].astype(int) == int(args.trace_index)].iloc[0]
    patch_row = source_row_by_id(source_rows, int(image_row["source_row"]))
    patch, patch_meta = _extract_patch(patch_row, canvas_cache={}, patch_size_px=int(args.patch_size_px))
    normal_trace = one_trace_from_source(
        source_rows,
        int(trace_row["trace_source_row"]),
        n_timepoints=int(args.n_timepoints),
        bin_seconds=1.0 / float(args.frame_rate_hz),
    )
    static_trace = np.zeros_like(normal_trace, dtype=np.float32)
    selected_rows = selected_units_from_args(args, unit_table, dense_sftf)
    selected_units = [int(row["unit_index"]) for row in selected_rows]
    speed = speed_by_frame(normal_trace, float(args.frame_rate_hz))
    frame_indices = choose_display_frames(speed, int(args.n_frames))

    obs_pair = observations[
        (observations["image_position"].astype(int) == int(args.image_position))
        & (observations["trace_index"].astype(int) == int(args.trace_index))
        & observations["condition_id"].astype(str).isin(["stabilized_static", "original_natural_timing"])
    ]
    by_condition = {
        str(row.condition_id): float(row.population_ssi_bits_per_spike)
        for row in obs_pair.itertuples(index=False)
    }
    population_static = float(by_condition.get("stabilized_static", float("nan")))
    population_original = float(by_condition.get("original_natural_timing", float("nan")))

    if args.reuse_map_cache is not None:
        cache_path_in = Path(args.reuse_map_cache)
        with np.load(cache_path_in, allow_pickle=False) as cache:
            cached_units = [int(unit) for unit in np.asarray(cache["selected_units"]).tolist()]
            if cached_units != selected_units:
                raise ValueError(
                    f"Cached selected units {cached_units} do not match requested units {selected_units}."
                )
            maps_by_condition = {
                "stabilized": np.asarray(cache["static_maps"], dtype=np.float32),
                "normal": np.asarray(cache["normal_maps"], dtype=np.float32),
            }
            if "frame_indices" in cache.files:
                frame_indices = [int(frame) for frame in np.asarray(cache["frame_indices"]).tolist()]
    else:
        population_view = load_population_view(version_name=str(args.rr100_version))
        scorer = CanonicalTwinScorer(
            device=str(args.device),
            batch_size=int(args.batch_size),
            empty_cache_every_batch=True,
        )
        maps_by_condition = {
            "stabilized": selected_rr100_maps_for_trace(
                scorer,
                population_view,
                np.asarray(patch),
                static_trace,
                selected_units,
                n_timepoints=int(args.n_timepoints),
            ),
            "normal": selected_rr100_maps_for_trace(
                scorer,
                population_view,
                np.asarray(patch),
                normal_trace,
                selected_units,
                n_timepoints=int(args.n_timepoints),
            ),
        }
    metrics_by_condition = {
        condition: unit_movie_ssi(maps, bin_seconds=1.0 / float(args.frame_rate_hz))
        for condition, maps in maps_by_condition.items()
    }
    label = str(args.example_label).strip() or f"image{int(args.image_position)}_trace{int(args.trace_index)}"
    map_png, map_pdf = render_maps(
        out_dir=out_dir,
        example_label=label,
        patch=np.asarray(patch),
        patch_meta=patch_meta,
        image_source_row=int(image_row["source_row"]),
        trace=normal_trace,
        maps_by_condition=maps_by_condition,
        selected_rows=selected_rows,
        frame_indices=frame_indices,
        speed=speed,
        population_static=population_static,
        population_original=population_original,
        frame_rate_hz=float(args.frame_rate_hz),
        dpi=int(args.dpi),
    )
    time_png, time_pdf = render_timecourses(
        out_dir=out_dir,
        example_label=label,
        selected_rows=selected_rows,
        metrics_by_condition=metrics_by_condition,
        frame_indices=frame_indices,
        speed=speed,
        frame_rate_hz=float(args.frame_rate_hz),
        dpi=int(args.dpi),
    )
    selected_csv = out_dir / "checkpoint_02_selected_units.csv"
    metrics_csv = out_dir / "checkpoint_02_unit_metric_summary.csv"
    frame_csv = out_dir / "checkpoint_02_selected_frames.csv"
    write_csv(selected_csv, selected_rows)
    write_csv(metrics_csv, metric_rows(selected_rows=selected_rows, metrics_by_condition=metrics_by_condition))
    write_csv(
        frame_csv,
        [
            {
                "frame_index": int(frame),
                "time_ms": float(frame * 1000.0 / float(args.frame_rate_hz)),
                "normal_speed_deg_s": float(speed[int(frame)]),
            }
            for frame in frame_indices
        ],
    )
    cache_path = out_dir / "checkpoint_02_selected_activation_maps.npz"
    np.savez_compressed(
        cache_path,
        condition_names=np.asarray(["stabilized", "normal"]),
        selected_units=np.asarray(selected_units, dtype=np.int32),
        selected_unit_labels=np.asarray([str(row["unit_label"]) for row in selected_rows]),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        speed_deg_s=speed.astype(np.float32),
        static_maps=maps_by_condition["stabilized"].astype(np.float32),
        normal_maps=maps_by_condition["normal"].astype(np.float32),
    )
    write_json(
        out_dir / "checkpoint_02_metadata.json",
        {
            "analysis": "temporal_power_shift_map_first_checkpoint_02_activation_maps",
            "render_type": "targeted_visualization_render",
            "run_dir": run_dir,
            "source_csv": Path(args.source_csv),
            "dense_sftf_csv": Path(args.dense_sftf_csv),
            "parametric_model_csv": Path(args.parametric_model_csv),
            "preference_source": str(args.preference_source),
            "rr100_version": str(args.rr100_version),
            "image_position": int(args.image_position),
            "image_source_row": int(image_row["source_row"]),
            "trace_index": int(args.trace_index),
            "trace_source_row": int(trace_row["trace_source_row"]),
            "example_label": label,
            "selected_units": selected_rows,
            "selected_frames": frame_indices,
            "population_static_ssi_bits_per_spike": population_static,
            "population_original_ssi_bits_per_spike": population_original,
            "reuse_map_cache": Path(args.reuse_map_cache) if args.reuse_map_cache is not None else None,
            "color_contract": (
                "Okabe-Ito categorical colors; cividis activation maps; PuOr_r signed difference maps; "
                "per-unit activation colorbars shared across static/normal rows; symmetric difference colorbars."
            ),
            "outputs": {
                "activation_map_tiles_png": map_png,
                "activation_map_tiles_pdf": map_pdf,
                "map_metric_timecourses_png": time_png,
                "map_metric_timecourses_pdf": time_pdf,
                "selected_units_csv": selected_csv,
                "unit_metric_summary_csv": metrics_csv,
                "selected_frames_csv": frame_csv,
                "selected_activation_maps_npz": cache_path,
            },
            "checkpoint_policy": "Stop after raw map comparison before selecting deeper example units.",
        },
    )
    print(f"Wrote {map_png}")
    print(f"Wrote {time_png}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {cache_path}")


if __name__ == "__main__":
    main()

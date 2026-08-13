#!/usr/bin/env python3
"""Audit stored contour axes against tensors recomputed from displayed pixels."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import _extract_patch


ROOT = Path(__file__).resolve().parents[4]
RUN_DIR = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)
OUT_DIR = RUN_DIR / "checkpoint1_production_readout"
SELECTED = OUT_DIR / "checkpoint1_selected_pairs.csv"
SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)


def _axial_delta(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 90.0) % 180.0 - 90.0))


def _tensor(patch: np.ndarray) -> dict[str, float | np.ndarray]:
    values = np.asarray(patch, dtype=np.float64)
    gx = ndimage.sobel(values, axis=1, mode="nearest")
    gy = ndimage.sobel(values, axis=0, mode="nearest")
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    denominator = jxx + jyy
    coherence = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / denominator if denominator > 0 else math.nan
    gradient_axis = math.degrees(0.5 * math.atan2(2.0 * jxy, jxx - jyy))
    edge_axis = gradient_axis + 90.0
    return {
        "gradient_axis_array_deg": gradient_axis,
        "edge_axis_array_deg": edge_axis,
        "coherence": coherence,
        "gradient_magnitude": np.hypot(gx, gy),
        "jxx": jxx,
        "jyy": jyy,
        "jxy": jxy,
    }


def _axis_line(ax: plt.Axes, angle_deg: float, *, radius: float, color: str, label: str, linewidth: float = 2.0) -> None:
    theta = math.radians(float(angle_deg))
    center = radius
    dx = 0.88 * radius * math.cos(theta)
    dy = 0.88 * radius * math.sin(theta)
    ax.plot([center - dx, center + dx], [center - dy, center + dy], color=color, linewidth=linewidth, label=label)


def main() -> None:
    selected = pd.read_csv(SELECTED)
    source = pd.read_csv(SOURCE)
    if "source_row" not in source.columns:
        source = source.copy()
        source["source_row"] = np.arange(len(source), dtype=int)
    source_by_id = source.set_index(source["source_row"].astype(int), drop=False)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(selected), 2, figsize=(7.8, 2.45 * len(selected)), squeeze=False)
    for row_index, selection in enumerate(selected.itertuples(index=False)):
        source_row = source_by_id.loc[int(selection.source_row)]
        patch, _meta = _extract_patch(source_row, canvas_cache=canvas_cache, patch_size_px=540)
        radius = int(source_row["image_patch_radius_px"])
        center = patch.shape[0] // 2
        local = np.asarray(patch[center - radius : center + radius + 1, center - radius : center + radius + 1])
        tensor = _tensor(local)
        stored_edge_array = float(source_row["image_edge_axis_array_deg"])
        stored_gradient_array = float(source_row["image_gradient_axis_array_deg"])
        rows.append(
            {
                "selection_role": str(selection.selection_role),
                "pair_index": int(selection.pair_index),
                "source_row": int(selection.source_row),
                "patch_radius_px": radius,
                "stored_coherence": float(source_row["image_orientation_coherence"]),
                "recomputed_coherence": float(tensor["coherence"]),
                "stored_edge_axis_array_deg": stored_edge_array,
                "recomputed_edge_axis_array_deg": float(tensor["edge_axis_array_deg"]),
                "stored_vs_recomputed_edge_delta_deg": _axial_delta(stored_edge_array, float(tensor["edge_axis_array_deg"])),
                "stored_gradient_axis_array_deg": stored_gradient_array,
                "recomputed_gradient_axis_array_deg": float(tensor["gradient_axis_array_deg"]),
                "stored_vs_recomputed_gradient_delta_deg": _axial_delta(stored_gradient_array, float(tensor["gradient_axis_array_deg"])),
                "edge_vs_gradient_delta_deg": _axial_delta(stored_edge_array, stored_gradient_array),
            }
        )
        for column_index, (image, title) in enumerate(
            ((local, "exact 1° tensor aperture"), (tensor["gradient_magnitude"], "Sobel gradient magnitude"))
        ):
            ax = axes[row_index, column_index]
            ax.imshow(image, cmap="gray", origin="upper")
            _axis_line(ax, stored_edge_array, radius=radius, color="#ff3f3f", label="stored edge axis")
            _axis_line(ax, stored_gradient_array, radius=radius, color="#00d6de", label="stored gradient axis", linewidth=1.5)
            ax.scatter([radius], [radius], color="#ffe75b", s=16, zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title, fontsize=9, weight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{selection.selection_role}\npair {int(selection.pair_index)}", fontsize=8, weight="bold")
            if row_index == 0 and column_index == 1:
                ax.legend(frameon=True, fontsize=7, loc="lower right")
            ax.text(
                0.02, 0.02,
                f"coh={float(tensor['coherence']):.3f}\nedge={stored_edge_array:.1f}°, grad={stored_gradient_array:.1f}°",
                transform=ax.transAxes, color="white", fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1.5},
            )
    fig.suptitle(
        "Contour-axis provenance audit\nred = stored edge/tangent axis; cyan = orthogonal gradient axis; angles in image-array coordinates",
        fontsize=11.5, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "checkpoint1_contour_axis_overlay_audit.png", dpi=240)
    fig.savefig(OUT_DIR / "checkpoint1_contour_axis_overlay_audit.pdf")
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "checkpoint1_contour_axis_overlay_audit.csv", index=False)
    print(table.to_string(index=False))
    print(f"[axis-audit] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

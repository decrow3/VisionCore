#!/usr/bin/env python3
"""Render an overlap-masked review of Stage 3 directional translation maps."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_stage3_empirical_directional_translation import (
    shift_with_zero,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_translation_v1"
)
DESIGN = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1"
)
OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_translation_review_v1"
)


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed review exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    probes = pd.read_csv(DESIGN / "selected_empirical_directional_probes.csv")
    translation_design = pd.read_csv(SOURCE / "probe_relative_translation_design.csv")
    metrics = pd.read_csv(SOURCE / "probe_relative_translation_response_metrics.csv")
    with np.load(SOURCE / "all_probe_relative_translation_activation_maps.npz", allow_pickle=False) as archive:
        units = np.asarray(archive["rr100_index"], dtype=int)
        maps = np.asarray(archive["selected_unit_rate_maps"], dtype=np.float32)
        blank = np.asarray(archive["selected_unit_blank_rate_maps"], dtype=np.float32)
    unit_to_column = {int(unit): index for index, unit in enumerate(units)}
    key_lookup = {
        (int(row.probe_index), str(row.translation_role)): index
        for index, row in translation_design.iterrows()
    }
    peak_roles = ("empirical peak tensor cell", "least-suppressed empirical control cell")
    peak_indices = probes.index[probes.probe_role.isin(peak_roles)].to_list()
    figure, axes = plt.subplots(len(peak_indices), 7, figsize=(23, 3.2 * len(peak_indices)), constrained_layout=True)
    review_rows: list[dict[str, object]] = []
    difference_cmap = plt.get_cmap("RdBu_r").copy()
    difference_cmap.set_bad("0.86")
    for row_index, probe_index in enumerate(peak_indices):
        probe = probes.loc[probe_index]
        owner = unit_to_column[int(probe.rr100_index)]
        center_map = maps[key_lookup[(probe_index, "center")], owner] - blank[owner]
        axis_data = []
        for translation_role in ("far positive motion axis", "far positive bar axis"):
            design_row = translation_design.loc[
                translation_design.probe_index.eq(probe_index)
                & translation_design.translation_role.eq(translation_role)
            ].iloc[0]
            observed = maps[key_lookup[(probe_index, translation_role)], owner] - blank[owner]
            shift_y = int(design_row.expected_translation_y_map_bins)
            shift_x = int(design_row.expected_translation_x_map_bins)
            prediction = shift_with_zero(center_map, shift_y, shift_x)
            valid = shift_with_zero(np.ones_like(center_map), shift_y, shift_x).astype(bool)
            difference = np.where(valid, observed - prediction, np.nan)
            valid_difference = difference[valid]
            metric_row = metrics.loc[
                metrics.probe_index.eq(probe_index) & metrics.translation_role.eq(translation_role)
            ].iloc[0]
            axis_data.append((design_row, observed, prediction, difference, valid_difference, metric_row))
            review_rows.append(
                {
                    "rr100_index": int(probe.rr100_index),
                    "selection_role": probe.selection_role,
                    "probe_role": probe.probe_role,
                    "translation_role": translation_role,
                    "translation_y_input_px": int(design_row.translation_y_input_px),
                    "translation_x_input_px": int(design_row.translation_x_input_px),
                    "valid_overlap_map_bins": int(valid.sum()),
                    "translation_map_pearson_r": float(metric_row.translation_map_pearson_r),
                    "translation_map_normalized_root_mean_square_error": float(
                        metric_row.translation_map_normalized_root_mean_square_error
                    ),
                    "valid_overlap_maximum_absolute_difference_hz": float(
                        np.max(np.abs(valid_difference))
                    ),
                }
            )
        response_limit = max(
            float(np.quantile(np.abs(np.stack([center_map, axis_data[0][1], axis_data[1][1]])), 0.995)),
            1e-8,
        )
        difference_limit = max(
            float(np.quantile(np.abs(np.concatenate([axis_data[0][4], axis_data[1][4]])), 0.995)),
            1e-10,
        )
        panels = [(center_map, "centered activation map", response_limit, plt.get_cmap("RdBu_r"))]
        for axis_name, data in zip(("motion", "bar"), axis_data, strict=True):
            design_row, observed, prediction, difference, valid_difference, metric_row = data
            panels.extend(
                [
                    (
                        observed,
                        f"observed after far translation along {axis_name} axis\n"
                        f"input offset (vertical, horizontal)=({int(design_row.translation_y_input_px)}, "
                        f"{int(design_row.translation_x_input_px)}) pixels",
                        response_limit,
                        plt.get_cmap("RdBu_r"),
                    ),
                    (
                        prediction,
                        f"centered map shifted to predicted {axis_name}-axis location",
                        response_limit,
                        plt.get_cmap("RdBu_r"),
                    ),
                    (
                        difference,
                        f"valid-overlap observed minus shifted prediction\n"
                        f"correlation {float(metric_row.translation_map_pearson_r):.6f}; "
                        f"maximum difference {np.max(np.abs(valid_difference)):.2e} Hz",
                        difference_limit,
                        difference_cmap,
                    ),
                ]
            )
        for column, (values, title, limit, cmap) in enumerate(panels):
            axes[row_index, column].imshow(
                values,
                cmap=cmap,
                origin="lower",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            )
            axes[row_index, column].set_title(title, fontsize=7.4)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"RR100 unit {int(probe.rr100_index)}\n{probe.selection_role.replace('_', ' ')}\n"
            f"motion {probe.motion_direction_image_deg:g}°; bar {probe.bar_orientation_image_deg:g}°",
            fontsize=8,
        )
    figure.suptitle(
        "Stage 3 translation review: far motion-axis and bar-axis maps agree with shifted centered predictions\n"
        "Gray borders in difference panels are nonoverlapping map regions excluded from all numerical comparisons",
        fontsize=14,
        weight="bold",
    )
    figure_path = OUT / "01_far_translation_maps_valid_overlap_only"
    figure.savefig(figure_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    review = pd.DataFrame(review_rows)
    review_path = OUT / "far_translation_valid_overlap_metrics.csv"
    review.to_csv(review_path, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_directional_translation_overlap_masked_review",
        "status": "visualization_correction_complete_awaiting_human_review",
        "correction": (
            "mask nonoverlapping borders in observed-minus-shifted prediction panels; the source checkpoint's "
            "numerical metrics already used overlap-only comparisons and are unchanged"
        ),
        "validation": {
            "minimum_far_peak_probe_translation_map_pearson_r": float(review.translation_map_pearson_r.min()),
            "maximum_far_peak_probe_normalized_root_mean_square_error": float(
                review.translation_map_normalized_root_mean_square_error.max()
            ),
            "maximum_far_peak_probe_valid_overlap_absolute_difference_hz": float(
                review.valid_overlap_maximum_absolute_difference_hz.max()
            ),
        },
        "source": identity(SOURCE / "manifest.json"),
        "outputs": {
            "figure": identity(figure_path.with_suffix(".png")),
            "metrics": identity(review_path),
        },
        "runner": identity(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Stage 3 directional translation overlap-masked review\n\n"
        "This review corrects only the raw difference-map visualization. Nonoverlapping borders are gray rather "
        "than being interpreted as response differences. The source checkpoint's saved overlap metrics were "
        "already correct and are not recomputed or replaced.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

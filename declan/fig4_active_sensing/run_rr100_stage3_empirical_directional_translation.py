#!/usr/bin/env python3
"""Test directional Stage 3 probes across motion-relative large-canvas positions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig4_active_sensing.run_rr100_spatial_coordinate_contract_stage3 import (
    MAP_SIZE,
    embed_native,
    forward_selected_maps,
    shifted_overlap_metrics,
)
from declan.redundancy_resolved_v1_population import (
    load_canonical_twin_bundle,
    load_population_view,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1"
)
DEFAULT_TRANSFER = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_transfer_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_translation_v1"
)
RADII_PX = (8, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


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


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def quantized_offset(angle_deg: float, radius_px: int, sign: int) -> tuple[int, int]:
    angle = np.deg2rad(float(angle_deg))
    dx = int(2 * round(float(sign) * float(radius_px) * np.cos(angle) / 2.0))
    dy = int(2 * round(float(sign) * float(radius_px) * np.sin(angle) / 2.0))
    if dx == 0 and dy == 0:
        raise ValueError(f"Translation collapsed to zero for angle={angle_deg}, radius={radius_px}")
    return dy, dx


def build_translation_design(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe_index, probe in probes.iterrows():
        rows.append(
            {
                "probe_index": int(probe_index),
                "rr100_index": int(probe.rr100_index),
                "probe_role": str(probe.probe_role),
                "translation_role": "center",
                "translation_axis": "center",
                "translation_sign": 0,
                "requested_translation_radius_input_px": 0,
                "translation_y_input_px": 0,
                "translation_x_input_px": 0,
                "actual_translation_radius_input_px": 0.0,
                "expected_translation_y_map_bins": 0,
                "expected_translation_x_map_bins": 0,
            }
        )
        axes = (
            ("motion axis", float(probe.motion_direction_image_deg)),
            ("bar axis", float(probe.bar_orientation_image_deg)),
        )
        for radius in RADII_PX:
            distance = "near" if radius == min(RADII_PX) else "far"
            for axis_name, angle in axes:
                for sign, sign_name in ((1, "positive"), (-1, "negative")):
                    dy, dx = quantized_offset(angle, radius, sign)
                    rows.append(
                        {
                            "probe_index": int(probe_index),
                            "rr100_index": int(probe.rr100_index),
                            "probe_role": str(probe.probe_role),
                            "translation_role": f"{distance} {sign_name} {axis_name}",
                            "translation_axis": axis_name,
                            "translation_sign": int(sign),
                            "requested_translation_radius_input_px": int(radius),
                            "translation_y_input_px": int(dy),
                            "translation_x_input_px": int(dx),
                            "actual_translation_radius_input_px": float(np.hypot(dy, dx)),
                            "expected_translation_y_map_bins": int(dy // 2),
                            "expected_translation_x_map_bins": int(dx // 2),
                        }
                    )
    return pd.DataFrame(rows)


def shift_with_zero(values: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    output = np.zeros_like(values)
    height, width = values.shape
    source_y = slice(0, height - shift_y) if shift_y >= 0 else slice(-shift_y, height)
    target_y = slice(shift_y, height) if shift_y >= 0 else slice(0, height + shift_y)
    source_x = slice(0, width - shift_x) if shift_x >= 0 else slice(-shift_x, width)
    target_x = slice(shift_x, width) if shift_x >= 0 else slice(0, width + shift_x)
    output[target_y, target_x] = values[source_y, source_x]
    return output


def short_probe_role(role: str) -> str:
    return {
        "empirical peak tensor cell": "empirical peak",
        "least-suppressed empirical control cell": "least-suppressed control",
        "opposite drift at peak SF and TF": "opposite drift",
        "strong frequency slice with changed preferred direction": "changed direction",
    }.get(str(role), str(role))


def ordered_translation_labels() -> list[str]:
    return [
        "center",
        "near positive motion axis",
        "near negative motion axis",
        "near positive bar axis",
        "near negative bar axis",
        "far positive motion axis",
        "far negative motion axis",
        "far positive bar axis",
        "far negative bar axis",
    ]


def plot_position_effects(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    units = metrics.rr100_index.drop_duplicates().to_list()
    labels = ordered_translation_labels()
    abbreviated = [
        "center", "+ motion\nnear", "− motion\nnear", "+ bar\nnear", "− bar\nnear",
        "+ motion\nfar", "− motion\nfar", "+ bar\nfar", "− bar\nfar",
    ]
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(2, len(units), figsize=(4.25 * len(units), 9), constrained_layout=True)
    for column, unit in enumerate(units):
        subset = metrics.loc[metrics.rr100_index.eq(unit)]
        roles = subset.probe_role.drop_duplicates().to_list()
        for color, role in zip(colors, roles, strict=False):
            probe = subset.loc[subset.probe_role.eq(role)].set_index("translation_role").loc[labels]
            x = np.arange(len(labels))
            axes[0, column].plot(
                x,
                probe.native_transfer_error_at_position_hz,
                marker="o",
                linewidth=1.8,
                color=color,
                label=short_probe_role(role),
            )
            axes[1, column].plot(
                x,
                probe.position_effect_relative_to_center_hz,
                marker="o",
                linewidth=1.8,
                color=color,
                label=short_probe_role(role),
            )
        for row in range(2):
            axes[row, column].axhline(0, color="0.35", linewidth=0.8)
            axes[row, column].set_xticks(np.arange(len(labels)), abbreviated, rotation=30, ha="right")
            axes[row, column].grid(axis="y", alpha=0.2)
        role_label = subset.selection_role.iloc[0].replace("_", " ")
        axes[0, column].set_title(f"RR100 unit {unit}\n{role_label}", fontsize=10)
        axes[0, column].set_ylabel("large-canvas local modulation minus native modulation (Hz)")
        axes[1, column].set_ylabel("large-canvas local modulation minus its centered value (Hz)")
        axes[1, column].set_title("Position-dependent change within the large canvas", fontsize=9)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Stage 3 position test: is the native-to-large response mismatch constant across motion-relative locations?\n"
        "Near translations request 8 input pixels and far translations request 40; diagonal offsets are rounded to even pixels so map locations remain exact",
        fontsize=14,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_far_translation_maps(
    probes: pd.DataFrame,
    design: pd.DataFrame,
    maps: np.ndarray,
    blank: np.ndarray,
    owner_columns: np.ndarray,
    path: Path,
    dpi: int,
) -> None:
    peak_roles = ("empirical peak tensor cell", "least-suppressed empirical control cell")
    peak_indices = probes.index[probes.probe_role.isin(peak_roles)].to_list()
    figure, axes = plt.subplots(len(peak_indices), 7, figsize=(23, 3.2 * len(peak_indices)), constrained_layout=True)
    key_lookup = {
        (int(row.probe_index), str(row.translation_role)): index
        for index, row in design.iterrows()
    }
    for row_index, probe_index in enumerate(peak_indices):
        probe = probes.loc[probe_index]
        owner = int(owner_columns[probe_index])
        center_map = maps[key_lookup[(probe_index, "center")], owner] - blank[owner]
        motion_record = design.loc[
            design.probe_index.eq(probe_index) & design.translation_role.eq("far positive motion axis")
        ].iloc[0]
        bar_record = design.loc[
            design.probe_index.eq(probe_index) & design.translation_role.eq("far positive bar axis")
        ].iloc[0]
        motion_observed = maps[key_lookup[(probe_index, "far positive motion axis")], owner] - blank[owner]
        bar_observed = maps[key_lookup[(probe_index, "far positive bar axis")], owner] - blank[owner]
        motion_prediction = shift_with_zero(
            center_map,
            int(motion_record.expected_translation_y_map_bins),
            int(motion_record.expected_translation_x_map_bins),
        )
        bar_prediction = shift_with_zero(
            center_map,
            int(bar_record.expected_translation_y_map_bins),
            int(bar_record.expected_translation_x_map_bins),
        )
        motion_difference = motion_observed - motion_prediction
        bar_difference = bar_observed - bar_prediction
        shared_limit = max(
            float(np.quantile(np.abs(np.stack([center_map, motion_observed, bar_observed])), 0.995)),
            1e-8,
        )
        difference_limit = max(
            float(np.quantile(np.abs(np.stack([motion_difference, bar_difference])), 0.995)),
            1e-10,
        )
        panels = (
            (center_map, "centered activation map", shared_limit),
            (
                motion_observed,
                "observed after far translation along motion axis\n"
                f"input offset (vertical, horizontal)=({int(motion_record.translation_y_input_px)}, "
                f"{int(motion_record.translation_x_input_px)}) pixels",
                shared_limit,
            ),
            (motion_prediction, "centered map shifted to predicted motion-axis location", shared_limit),
            (motion_difference, "motion-axis observed minus shifted prediction", difference_limit),
            (
                bar_observed,
                "observed after far translation along bar axis\n"
                f"input offset (vertical, horizontal)=({int(bar_record.translation_y_input_px)}, "
                f"{int(bar_record.translation_x_input_px)}) pixels",
                shared_limit,
            ),
            (bar_prediction, "centered map shifted to predicted bar-axis location", shared_limit),
            (bar_difference, "bar-axis observed minus shifted prediction", difference_limit),
        )
        for column, (values, title, limit) in enumerate(panels):
            axes[row_index, column].imshow(
                values,
                cmap="RdBu_r",
                origin="lower",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            )
            axes[row_index, column].set_title(title, fontsize=7.5)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"RR100 unit {int(probe.rr100_index)}\n{probe.selection_role.replace('_', ' ')}\n"
            f"motion {probe.motion_direction_image_deg:g}°; bar {probe.bar_orientation_image_deg:g}°",
            fontsize=8,
        )
    figure.suptitle(
        "Stage 3 raw translation maps: far displacements along each probe's motion and bar axes\n"
        "Response maps share a color scale within each row; observed-minus-predicted differences use a separate scale",
        fontsize=14,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    design_dir = args.design_dir.resolve()
    transfer_dir = args.transfer_dir.resolve()
    out = args.out_dir.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed directional translation checkpoint exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(design_dir / "selected_empirical_directional_probes.csv")
    with np.load(design_dir / "selected_localized_probe_histories.npz", allow_pickle=False) as archive:
        native = np.asarray(archive["localized_probe_histories"], dtype=np.float32)
    transfer_metrics = pd.read_csv(transfer_dir / "owner_probe_native_to_large_canvas_transfer.csv")
    if not np.array_equal(transfer_metrics.rr100_index.to_numpy(int), probes.rr100_index.to_numpy(int)):
        raise ValueError("Transfer rows do not match the approved probe ordering")

    units = probes[["rr100_index", "selection_role"]].drop_duplicates().reset_index(drop=True)
    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapped_channels = np.argmax(view.membership, axis=1).astype(int)
    canonical_channels = mapped_channels[units.rr100_index.to_numpy(int)]
    unit_to_column = {int(unit): column for column, unit in enumerate(units.rr100_index)}
    owner_columns = np.asarray([unit_to_column[int(unit)] for unit in probes.rr100_index], dtype=int)

    translation_design = build_translation_design(probes)
    if translation_design.duplicated(["probe_index", "translation_role"]).any():
        raise ValueError("Translation roles are not unique within probes")
    translation_design.to_csv(out / "probe_relative_translation_design.csv", index=False)
    cubes = np.stack(
        [
            embed_native(
                native[int(row.probe_index)],
                int(row.translation_y_input_px),
                int(row.translation_x_input_px),
            )
            for row in translation_design.itertuples(index=False)
        ]
    )

    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    maps = forward_selected_maps(bundle, cubes, canonical_channels, batch_size=int(args.batch_size))
    blank = forward_selected_maps(
        bundle,
        np.zeros((1, native.shape[1], cubes.shape[-2], cubes.shape[-1]), dtype=np.float32),
        canonical_channels,
        batch_size=1,
    )[0]
    if maps.shape != (len(translation_design), len(units), MAP_SIZE, MAP_SIZE):
        raise ValueError(f"Unexpected translated map shape {maps.shape}")

    center_map_index = MAP_SIZE // 2
    key_lookup = {
        (int(row.probe_index), str(row.translation_role)): index
        for index, row in translation_design.iterrows()
    }
    rows: list[dict[str, object]] = []
    for design_index, design_row in translation_design.iterrows():
        probe_index = int(design_row.probe_index)
        probe = probes.loc[probe_index]
        owner = int(owner_columns[probe_index])
        y = center_map_index + int(design_row.expected_translation_y_map_bins)
        x = center_map_index + int(design_row.expected_translation_x_map_bins)
        if not (0 <= y < MAP_SIZE and 0 <= x < MAP_SIZE):
            raise ValueError(f"Predicted map sample {(y, x)} is out of bounds")
        observed_map = maps[design_index, owner] - blank[owner]
        centered_design_index = key_lookup[(probe_index, "center")]
        centered_map = maps[centered_design_index, owner] - blank[owner]
        local_modulation = float(observed_map[y, x])
        centered_large_modulation = float(centered_map[center_map_index, center_map_index])
        translation_metrics = shifted_overlap_metrics(
            centered_map,
            observed_map,
            int(design_row.expected_translation_y_map_bins),
            int(design_row.expected_translation_x_map_bins),
        )
        rows.append(
            {
                **probe.to_dict(),
                **design_row.to_dict(),
                "canonical_channel": int(canonical_channels[owner]),
                "sample_y_map_index": int(y),
                "sample_x_map_index": int(x),
                "native_modulation_hz": float(transfer_metrics.loc[probe_index, "native_modulation_hz"]),
                "centered_large_canvas_modulation_hz": centered_large_modulation,
                "translated_local_large_canvas_modulation_hz": local_modulation,
                "native_transfer_error_at_position_hz": (
                    local_modulation - float(transfer_metrics.loc[probe_index, "native_modulation_hz"])
                ),
                "position_effect_relative_to_center_hz": local_modulation - centered_large_modulation,
                **translation_metrics,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "probe_relative_translation_response_metrics.csv", index=False)
    np.savez_compressed(
        out / "all_probe_relative_translation_activation_maps.npz",
        rr100_index=units.rr100_index.to_numpy(int),
        canonical_channel=canonical_channels,
        translation_probe_index=translation_design.probe_index.to_numpy(int),
        translation_role=translation_design.translation_role.to_numpy(dtype="U40"),
        translation_y_input_px=translation_design.translation_y_input_px.to_numpy(int),
        translation_x_input_px=translation_design.translation_x_input_px.to_numpy(int),
        selected_unit_rate_maps=maps,
        selected_unit_blank_rate_maps=blank,
    )

    summaries: list[dict[str, object]] = []
    for (unit, probe_role), subset in metrics.groupby(["rr100_index", "probe_role"], sort=False):
        noncenter = subset.loc[~subset.translation_role.eq("center")]
        near = noncenter.loc[noncenter.requested_translation_radius_input_px.eq(min(RADII_PX))]
        far = noncenter.loc[noncenter.requested_translation_radius_input_px.eq(max(RADII_PX))]
        summaries.append(
            {
                "rr100_index": int(unit),
                "selection_role": subset.selection_role.iloc[0],
                "probe_role": probe_role,
                "centered_native_transfer_error_hz": float(
                    subset.loc[subset.translation_role.eq("center"), "native_transfer_error_at_position_hz"].iloc[0]
                ),
                "maximum_near_position_effect_absolute_hz": float(
                    near.position_effect_relative_to_center_hz.abs().max()
                ),
                "maximum_far_position_effect_absolute_hz": float(
                    far.position_effect_relative_to_center_hz.abs().max()
                ),
                "minimum_near_translation_map_pearson_r": float(near.translation_map_pearson_r.min()),
                "minimum_far_translation_map_pearson_r": float(far.translation_map_pearson_r.min()),
                "maximum_far_translation_normalized_root_mean_square_error": float(
                    far.translation_map_normalized_root_mean_square_error.max()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "descriptive_probe_translation_summary.csv", index=False)

    plot_position_effects(metrics, out / "01_motion_relative_position_response_effects", int(args.dpi))
    plot_far_translation_maps(
        probes,
        translation_design,
        maps,
        blank,
        owner_columns,
        out / "02_far_motion_and_bar_axis_raw_translation_maps",
        int(args.dpi),
    )

    noncenter = metrics.loc[~metrics.translation_role.eq("center")]
    near = noncenter.loc[noncenter.requested_translation_radius_input_px.eq(min(RADII_PX))]
    far = noncenter.loc[noncenter.requested_translation_radius_input_px.eq(max(RADII_PX))]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_empirical_directional_motion_relative_translation",
        "status": "targeted_translation_multi_map_checkpoint_complete_awaiting_human_review",
        "scope": {
            "n_units": int(len(units)),
            "n_probes": int(len(probes)),
            "n_translation_positions_per_probe": 9,
            "n_translated_histories": int(len(translation_design)),
            "n_saved_translated_history_by_selected_unit_maps": int(len(translation_design) * len(units)),
            "reserved_final_test_identities_opened": False,
        },
        "contracts": {
            "translations": "center plus positive and negative near/far offsets along each probe's empirical motion and bar axes",
            "requested_translation_radii_input_px": list(RADII_PX),
            "diagonal_quantization": "round vertical and horizontal components to even input pixels so the two-pixel map stride remains exact",
            "local_response": "blank-subtracted large-canvas activation-map value at the predicted translated probe location",
            "native_reference": "the separately blank-subtracted native modulation saved by the preceding transfer checkpoint",
            "aperture_frozen": False,
            "panels_d_e_modified": False,
        },
        "validation": {
            "maximum_near_position_effect_absolute_hz": float(near.position_effect_relative_to_center_hz.abs().max()),
            "maximum_far_position_effect_absolute_hz": float(far.position_effect_relative_to_center_hz.abs().max()),
            "minimum_near_translation_map_pearson_r": float(near.translation_map_pearson_r.min()),
            "minimum_far_translation_map_pearson_r": float(far.translation_map_pearson_r.min()),
            "maximum_far_translation_normalized_root_mean_square_error": float(
                far.translation_map_normalized_root_mean_square_error.max()
            ),
        },
        "decision_gate": (
            "inspect position-effect traces and far observed-minus-shifted maps; only then decide whether the "
            "translation evidence supports revisiting a spatial aperture"
        ),
        "sources": {
            "probe_table": identity(design_dir / "selected_empirical_directional_probes.csv"),
            "probe_histories": identity(design_dir / "selected_localized_probe_histories.npz"),
            "transfer_metrics": identity(transfer_dir / "owner_probe_native_to_large_canvas_transfer.csv"),
            "runner": identity(Path(__file__)),
        },
        "outputs": {
            name: identity(out / name)
            for name in (
                "probe_relative_translation_design.csv",
                "probe_relative_translation_response_metrics.csv",
                "descriptive_probe_translation_summary.csv",
                "all_probe_relative_translation_activation_maps.npz",
                "01_motion_relative_position_response_effects.png",
                "02_far_motion_and_bar_axis_raw_translation_maps.png",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Cartographer Stage 3 empirical directional translation checkpoint\n\n"
        "This targeted checkpoint translates every approved localized grating along its empirical motion and "
        "bar axes. It separates the pre-existing native-to-large canvas mismatch from response changes caused "
        "by position within the large canvas. It does not select an aperture or modify Figure 4 Panels D/E.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()

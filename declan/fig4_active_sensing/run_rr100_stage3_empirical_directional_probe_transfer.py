#!/usr/bin/env python3
"""Score approved empirical directional probes through native and large canvases."""
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
    LARGE_SIZE,
    MAP_SIZE,
    N_HISTORY,
    NATIVE_SIZE,
    embed_native,
    forward_selected_maps,
    make_probe_cube,
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
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_transfer_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def short_probe_role(role: str) -> str:
    replacements = {
        "empirical peak tensor cell": "empirical peak",
        "least-suppressed empirical control cell": "least-suppressed control",
        "opposite drift at peak SF and TF": "opposite drift",
        "strong frequency slice with changed preferred direction": "changed direction",
    }
    return replacements.get(str(role), str(role))


def plot_response_transfer(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    units = metrics.rr100_index.drop_duplicates().to_list()
    figure, axes = plt.subplots(1, len(units), figsize=(4.1 * len(units), 5.3), constrained_layout=True)
    if len(units) == 1:
        axes = np.asarray([axes])
    colors = {
        "native 51-by-51 pathway": "#0072B2",
        "large canvas with identical embedded probe": "#D55E00",
        "large canvas with analytically extended probe": "#009E73",
    }
    for axis, unit in zip(axes, units, strict=True):
        subset = metrics.loc[metrics.rr100_index.eq(unit)].reset_index(drop=True)
        x = np.arange(len(subset))
        series = {
            "native 51-by-51 pathway": subset.native_modulation_hz,
            "large canvas with identical embedded probe": subset.large_embedded_center_modulation_hz,
            "large canvas with analytically extended probe": subset.large_extended_center_modulation_hz,
        }
        for label, values in series.items():
            axis.plot(x, values, marker="o", linewidth=2, label=label, color=colors[label])
        axis.axhline(0, color="0.4", linewidth=0.8)
        axis.set_xticks(x, [short_probe_role(value) for value in subset.probe_role], rotation=25, ha="right")
        axis.set_title(
            f"RR100 unit {unit}\n{subset.selection_role.iloc[0].replace('_', ' ')}",
            fontsize=10,
        )
        axis.set_ylabel("blank-subtracted firing-rate modulation (Hz)")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Stage 3 response transfer: does each native grating modulation reappear at the center of the large-canvas activation map?\n"
        "Each line compares the same 32-frame localized grating history; blank responses are subtracted separately for each canvas",
        fontsize=14,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_owner_maps(
    probes: pd.DataFrame,
    native_histories: np.ndarray,
    embedded_maps: np.ndarray,
    extended_maps: np.ndarray,
    blank_large: np.ndarray,
    owner_columns: np.ndarray,
    metrics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(len(probes), 4, figsize=(16, 2.8 * len(probes)), constrained_layout=True)
    for row, probe in enumerate(probes.itertuples(index=False)):
        owner_column = int(owner_columns[row])
        embedded = embedded_maps[row, owner_column] - blank_large[owner_column]
        extended = extended_maps[row, owner_column] - blank_large[owner_column]
        difference = extended - embedded
        shared_limit = max(float(np.quantile(np.abs(np.stack([embedded, extended])), 0.995)), 1e-8)
        difference_limit = max(float(np.quantile(np.abs(difference), 0.995)), 1e-10)
        record = metrics.iloc[row]

        axes[row, 0].imshow(native_histories[row, 0], cmap="gray", origin="lower", vmin=-0.35, vmax=0.35)
        axes[row, 0].set_title(
            "localized grating at current lag\n"
            f"spatial frequency {probe.spatial_frequency_cpd:g} cycles/degree; "
            f"temporal frequency {probe.temporal_frequency_magnitude_hz:g} Hz\n"
            f"motion {probe.motion_direction_image_deg:g}°; bar {probe.bar_orientation_image_deg:g}°; "
            f"{probe.drift_sign} drift",
            fontsize=7.5,
        )
        for column, values, title in (
            (
                1,
                embedded,
                "large-canvas activation map: identical embedded probe\n"
                f"central modulation {record.large_embedded_center_modulation_hz:.3f} Hz",
            ),
            (
                2,
                extended,
                "large-canvas activation map: analytically extended probe\n"
                f"central modulation {record.large_extended_center_modulation_hz:.3f} Hz",
            ),
        ):
            axes[row, column].imshow(
                values,
                cmap="RdBu_r",
                origin="lower",
                norm=TwoSlopeNorm(vmin=-shared_limit, vcenter=0, vmax=shared_limit),
            )
            axes[row, column].set_title(title, fontsize=8)
        axes[row, 3].imshow(
            difference,
            cmap="RdBu_r",
            origin="lower",
            norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0, vmax=difference_limit),
        )
        axes[row, 3].set_title(
            "analytically extended minus embedded activation map\n"
            f"maximum absolute difference {np.max(np.abs(difference)):.2e} Hz",
            fontsize=8,
        )
        axes[row, 0].set_ylabel(
            f"RR100 unit {probe.rr100_index}\n{probe.selection_role.replace('_', ' ')}\n"
            f"{short_probe_role(probe.probe_role)}\n"
            f"native modulation {record.native_modulation_hz:.3f} Hz",
            fontsize=7.5,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Cartographer Stage 3 raw activation maps for empirical direction-tuning probes\n"
        "Embedded and extended maps share a color scale within each row; difference maps use a separate labeled scale",
        fontsize=14,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    design = args.design_dir.resolve()
    out = args.out_dir.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed transfer checkpoint exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(design / "selected_empirical_directional_probes.csv")
    with np.load(design / "selected_localized_probe_histories.npz", allow_pickle=False) as archive:
        native = np.asarray(archive["localized_probe_histories"], dtype=np.float32)
    if native.shape != (len(probes), N_HISTORY, NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError(f"Unexpected localized-probe shape {native.shape}")

    units = probes[["rr100_index", "selection_role"]].drop_duplicates().reset_index(drop=True)
    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    if view.membership is None or view.membership.shape != (100, 756):
        raise ValueError("Stage 3 requires the one-hot RR100 movie-medoid population view")
    mapped_channels = np.argmax(view.membership, axis=1).astype(int)
    canonical_channels = mapped_channels[units.rr100_index.to_numpy(int)]
    if not np.allclose(view.membership[units.rr100_index.to_numpy(int), canonical_channels], 1.0):
        raise ValueError("Selected RR100 membership entries are not one-hot")
    unit_to_column = {int(unit): column for column, unit in enumerate(units.rr100_index)}
    owner_columns = np.asarray([unit_to_column[int(unit)] for unit in probes.rr100_index], dtype=int)

    embedded = np.stack([embed_native(cube, 0, 0) for cube in native])
    extended = np.stack(
        [
            make_probe_cube(
                LARGE_SIZE,
                sf=float(row.spatial_frequency_cpd),
                tf=float(row.signed_temporal_frequency_hz_for_cartographer_renderer),
                orientation_deg=float(row.bar_orientation_image_deg),
            )
            for row in probes.itertuples(index=False)
        ]
    )
    crop_start = (LARGE_SIZE - NATIVE_SIZE) // 2
    central_crop_error = float(
        np.max(
            np.abs(
                extended[
                    :,
                    :,
                    crop_start:crop_start + NATIVE_SIZE,
                    crop_start:crop_start + NATIVE_SIZE,
                ]
                - native
            )
        )
    )
    if central_crop_error > 1e-6:
        raise ValueError(f"Analytic large-canvas probes do not reproduce native centers: {central_crop_error}")

    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    native_maps = forward_selected_maps(
        bundle, native, canonical_channels, batch_size=int(args.batch_size)
    )
    embedded_maps = forward_selected_maps(
        bundle, embedded, canonical_channels, batch_size=int(args.batch_size)
    )
    extended_maps = forward_selected_maps(
        bundle, extended, canonical_channels, batch_size=int(args.batch_size)
    )
    blank_native = forward_selected_maps(
        bundle,
        np.zeros((1, N_HISTORY, NATIVE_SIZE, NATIVE_SIZE), dtype=np.float32),
        canonical_channels,
        batch_size=1,
    )[0, :, 0, 0]
    blank_large = forward_selected_maps(
        bundle,
        np.zeros((1, N_HISTORY, LARGE_SIZE, LARGE_SIZE), dtype=np.float32),
        canonical_channels,
        batch_size=1,
    )[0]
    if native_maps.shape[-2:] != (1, 1):
        raise ValueError(f"Native histories did not produce scalar outputs: {native_maps.shape}")
    if embedded_maps.shape[-2:] != (MAP_SIZE, MAP_SIZE) or extended_maps.shape != embedded_maps.shape:
        raise ValueError(f"Unexpected large-canvas map shapes {embedded_maps.shape}/{extended_maps.shape}")

    center = MAP_SIZE // 2
    rows: list[dict[str, object]] = []
    for probe_index, probe in probes.iterrows():
        owner_column = int(owner_columns[probe_index])
        native_rate = float(native_maps[probe_index, owner_column, 0, 0])
        native_modulation = native_rate - float(blank_native[owner_column])
        embedded_rate = float(embedded_maps[probe_index, owner_column, center, center])
        embedded_modulation = embedded_rate - float(blank_large[owner_column, center, center])
        extended_rate = float(extended_maps[probe_index, owner_column, center, center])
        extended_modulation = extended_rate - float(blank_large[owner_column, center, center])
        embedded_owner_map = embedded_maps[probe_index, owner_column] - blank_large[owner_column]
        extended_owner_map = extended_maps[probe_index, owner_column] - blank_large[owner_column]
        rows.append(
            {
                **probe.to_dict(),
                "canonical_channel": int(canonical_channels[owner_column]),
                "native_blank_rate_hz": float(blank_native[owner_column]),
                "large_canvas_blank_center_rate_hz": float(blank_large[owner_column, center, center]),
                "native_rate_hz": native_rate,
                "native_modulation_hz": native_modulation,
                "large_embedded_center_rate_hz": embedded_rate,
                "large_embedded_center_modulation_hz": embedded_modulation,
                "large_extended_center_rate_hz": extended_rate,
                "large_extended_center_modulation_hz": extended_modulation,
                "embedded_native_modulation_difference_hz": embedded_modulation - native_modulation,
                "extended_native_modulation_difference_hz": extended_modulation - native_modulation,
                "embedded_extended_center_modulation_difference_hz": extended_modulation - embedded_modulation,
                "embedded_extended_owner_map_pearson_r": correlation(embedded_owner_map, extended_owner_map),
                "embedded_extended_owner_map_maximum_absolute_difference_hz": float(
                    np.max(np.abs(extended_owner_map - embedded_owner_map))
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "owner_probe_native_to_large_canvas_transfer.csv", index=False)
    units.assign(canonical_channel=canonical_channels).to_csv(out / "selected_unit_channels.csv", index=False)
    np.savez_compressed(
        out / "all_selected_unit_directional_probe_activation_maps.npz",
        rr100_index=units.rr100_index.to_numpy(int),
        canonical_channel=canonical_channels,
        probe_owner_rr100_index=probes.rr100_index.to_numpy(int),
        probe_role=probes.probe_role.to_numpy(dtype="U80"),
        native_selected_unit_rates=native_maps[:, :, 0, 0],
        embedded_selected_unit_large_canvas_rate_maps=embedded_maps,
        extended_selected_unit_large_canvas_rate_maps=extended_maps,
        native_selected_unit_blank_rates=blank_native,
        large_canvas_selected_unit_blank_rate_maps=blank_large,
    )

    plot_response_transfer(metrics, out / "01_native_to_large_canvas_directional_response_transfer", int(args.dpi))
    plot_owner_maps(
        probes,
        native,
        embedded_maps,
        extended_maps,
        blank_large,
        owner_columns,
        metrics,
        out / "02_empirical_directional_probe_raw_activation_maps",
        int(args.dpi),
    )

    per_unit = []
    for unit, subset in metrics.groupby("rr100_index", sort=False):
        per_unit.append(
            {
                "rr100_index": int(unit),
                "selection_role": subset.selection_role.iloc[0],
                "probe_count": int(len(subset)),
                "native_vs_embedded_modulation_pearson_r": correlation(
                    subset.native_modulation_hz, subset.large_embedded_center_modulation_hz
                ),
                "maximum_embedded_native_modulation_absolute_difference_hz": float(
                    np.max(np.abs(subset.embedded_native_modulation_difference_hz))
                ),
                "mean_embedded_native_modulation_absolute_difference_hz": float(
                    np.mean(np.abs(subset.embedded_native_modulation_difference_hz))
                ),
            }
        )
    per_unit_frame = pd.DataFrame(per_unit)
    per_unit_frame.to_csv(out / "descriptive_per_unit_transfer_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_empirical_directional_probe_native_large_transfer",
        "status": "targeted_multi_map_checkpoint_complete_awaiting_human_review",
        "scope": {
            "n_units": int(len(units)),
            "n_probes": int(len(probes)),
            "n_saved_probe_by_selected_unit_maps": int(len(probes) * len(units)),
            "reserved_final_test_identities_opened": False,
        },
        "contracts": {
            "probe_source": "approved complete empirical SF-by-absolute-TF-by-direction tensor cells",
            "native_pathway": "localized 32-frame 51-by-51 grating history producing one scalar unit response",
            "large_embedded_pathway": "identical localized history zero-embedded at the center of a 151-by-151 canvas",
            "large_extended_pathway": "same analytic Gaussian-windowed grating rendered directly on a 151-by-151 canvas",
            "response": "post-activation firing rate; blank modulation subtracts a separately scored blank for each canvas",
            "parametric_fit_used": False,
            "old_four_bin_orientation_used": False,
            "aperture_frozen": False,
            "panels_d_e_modified": False,
        },
        "validation": {
            "maximum_analytic_large_center_crop_input_error": central_crop_error,
            "maximum_embedded_extended_center_modulation_difference_hz": float(
                np.max(np.abs(metrics.embedded_extended_center_modulation_difference_hz))
            ),
            "minimum_embedded_extended_owner_map_pearson_r": float(
                np.nanmin(metrics.embedded_extended_owner_map_pearson_r)
            ),
            "maximum_embedded_extended_owner_map_absolute_difference_hz": float(
                metrics.embedded_extended_owner_map_maximum_absolute_difference_hz.max()
            ),
            "all_probe_native_vs_embedded_modulation_pearson_r": correlation(
                metrics.native_modulation_hz, metrics.large_embedded_center_modulation_hz
            ),
            "maximum_embedded_native_modulation_absolute_difference_hz": float(
                np.max(np.abs(metrics.embedded_native_modulation_difference_hz))
            ),
            "mean_embedded_native_modulation_absolute_difference_hz": float(
                np.mean(np.abs(metrics.embedded_native_modulation_difference_hz))
            ),
        },
        "decision_gate": (
            "inspect the per-probe response traces and raw owner activation maps; do not choose an aperture or "
            "advance to Panels D/E until the canvas-transfer pattern is understood"
        ),
        "sources": {
            "probe_table": identity(design / "selected_empirical_directional_probes.csv"),
            "probe_histories": identity(design / "selected_localized_probe_histories.npz"),
            "design_manifest": identity(design / "manifest.json"),
            "runner": identity(Path(__file__)),
        },
        "outputs": {
            name: identity(out / name)
            for name in (
                "owner_probe_native_to_large_canvas_transfer.csv",
                "selected_unit_channels.csv",
                "all_selected_unit_directional_probe_activation_maps.npz",
                "descriptive_per_unit_transfer_summary.csv",
                "01_native_to_large_canvas_directional_response_transfer.png",
                "02_empirical_directional_probe_raw_activation_maps.png",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Cartographer Stage 3 empirical directional-probe transfer checkpoint\n\n"
        "This targeted checkpoint scores the approved empirical directional probes through the native scalar "
        "pathway and two direct large-canvas constructions. It saves raw activation maps for all five selected "
        "units and displays the map belonging to each probe owner. Blank responses are scored and subtracted "
        "separately for the native and large canvases. This checkpoint does not select an aperture or modify "
        "Figure 4 Panels D/E.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()

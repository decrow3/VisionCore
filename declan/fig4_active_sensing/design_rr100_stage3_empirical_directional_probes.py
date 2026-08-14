#!/usr/bin/env python3
"""Select Cartographer Stage 3 probes from the empirical SF×TF×direction tensor."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_spatial_coordinate_contract_stage3 import (
    NATIVE_SIZE,
    make_probe_cube,
)


ROOT = Path(__file__).resolve().parents[2]
TUNING_DIR = ROOT / "outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v1"
TENSOR = TUNING_DIR / "rr100_empirical_joint_sftf_direction_tuning.npz"
SELECTION = TUNING_DIR / "selected_unit_roles.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1"


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def circular_distance(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


def bar_and_sign(motion_direction_deg: float, temporal_hz: float) -> tuple[float, str, float]:
    direction = float(motion_direction_deg) % 360.0
    bar = (direction - 90.0) % 180.0
    positive_motion = (bar + 90.0) % 360.0
    sign = 1.0 if circular_distance(direction, positive_motion) < 1e-8 else -1.0
    return bar, "positive" if sign > 0 else "negative", sign * float(temporal_hz)


def append_probe(
    rows: list[dict[str, object]],
    *,
    unit: int,
    selection_role: str,
    probe_role: str,
    sf_index: int,
    tf_index: int,
    direction_index: int,
    sf: np.ndarray,
    tf: np.ndarray,
    directions: np.ndarray,
    signed: np.ndarray,
    positive: np.ndarray,
    criterion: str,
) -> None:
    direction = float(directions[direction_index])
    bar, drift_sign, signed_tf = bar_and_sign(direction, float(tf[tf_index]))
    rows.append(
        {
            "rr100_index": int(unit),
            "selection_role": selection_role,
            "probe_role": probe_role,
            "probe_selection_criterion": criterion,
            "spatial_frequency_cpd": float(sf[sf_index]),
            "temporal_frequency_magnitude_hz": float(tf[tf_index]),
            "motion_direction_image_deg": direction,
            "bar_orientation_image_deg": bar,
            "drift_sign": drift_sign,
            "signed_temporal_frequency_hz_for_cartographer_renderer": signed_tf,
            "empirical_signed_f0_hz": float(signed[sf_index, tf_index, direction_index]),
            "empirical_positive_f0_hz": float(positive[sf_index, tf_index, direction_index]),
            "sf_index": int(sf_index),
            "tf_index": int(tf_index),
            "direction_index": int(direction_index),
            "source_is_full_empirical_tensor": True,
            "parametric_fit_used_for_selection": False,
            "old_four_bin_orientation_field_used": False,
        }
    )


def select_probes() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    selected = pd.read_csv(SELECTION)
    with np.load(TENSOR, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    units = arrays["rr100_index"].astype(int)
    sf = arrays["spatial_cpd"].astype(float)
    tf = arrays["temporal_hz"].astype(float)
    directions = arrays["motion_direction_image_deg"].astype(float)
    signed_all = arrays["signed_f0_hz"].astype(float)
    positive_all = arrays["positive_f0_hz"].astype(float)
    preferred_region = arrays["preferred_region_mask"].astype(bool)
    rows: list[dict[str, object]] = []

    for selection in selected.itertuples(index=False):
        unit = int(selection.rr100_index)
        unit_position = int(np.flatnonzero(units == unit)[0])
        signed = signed_all[unit_position]
        positive = positive_all[unit_position]
        responsive = float(positive.max()) > 1e-12
        if responsive:
            peak_sf, peak_tf, peak_direction = np.unravel_index(int(np.argmax(positive)), positive.shape)
        else:
            peak_sf, peak_tf, peak_direction = np.unravel_index(int(np.argmax(signed)), signed.shape)
        append_probe(
            rows,
            unit=unit,
            selection_role=str(selection.selection_role),
            probe_role="empirical peak tensor cell" if responsive else "least-suppressed empirical control cell",
            sf_index=peak_sf,
            tf_index=peak_tf,
            direction_index=peak_direction,
            sf=sf,
            tf=tf,
            directions=directions,
            signed=signed,
            positive=positive,
            criterion=(
                "maximum positive F0 over the complete empirical SF-by-absolute-TF-by-direction tensor"
                if responsive
                else "maximum signed F0 because the unit has no positive above-blank tensor cell"
            ),
        )

        opposite_direction = int((peak_direction + len(directions) // 2) % len(directions))
        append_probe(
            rows,
            unit=unit,
            selection_role=str(selection.selection_role),
            probe_role="opposite drift at peak SF and TF",
            sf_index=peak_sf,
            tf_index=peak_tf,
            direction_index=opposite_direction,
            sf=sf,
            tf=tf,
            directions=directions,
            signed=signed,
            positive=positive,
            criterion="180-degree motion-direction control at the same empirical peak SF and TF",
        )

        sensitivity = positive.sum(axis=-1)
        preferred_direction_index = np.argmax(positive, axis=-1)
        peak_direction_deg = float(directions[peak_direction])
        candidates: list[tuple[float, float, int, int, int]] = []
        for sf_index, tf_index in np.argwhere(preferred_region[unit_position]):
            direction_index = int(preferred_direction_index[sf_index, tf_index])
            if int(sf_index) == int(peak_sf) and int(tf_index) == int(peak_tf):
                continue
            delta = circular_distance(float(directions[direction_index]), peak_direction_deg)
            candidates.append(
                (
                    delta,
                    float(sensitivity[sf_index, tf_index]),
                    int(sf_index),
                    int(tf_index),
                    direction_index,
                )
            )
        if candidates:
            _, _, contrast_sf, contrast_tf, contrast_direction = max(candidates, key=lambda item: (item[0], item[1]))
            append_probe(
                rows,
                unit=unit,
                selection_role=str(selection.selection_role),
                probe_role="strong frequency slice with changed preferred direction",
                sf_index=contrast_sf,
                tf_index=contrast_tf,
                direction_index=contrast_direction,
                sf=sf,
                tf=tf,
                directions=directions,
                signed=signed,
                positive=positive,
                criterion=(
                    "largest preferred-direction change from the peak among SF-by-TF cells at or above "
                    "half the unit's peak direction-summed positive F0; ties favor higher sensitivity"
                ),
            )
    frame = pd.DataFrame(rows)
    peak_direction = frame.loc[
        frame.probe_role.isin(["empirical peak tensor cell", "least-suppressed empirical control cell"]),
        ["rr100_index", "motion_direction_image_deg"],
    ].rename(columns={"motion_direction_image_deg": "peak_motion_direction_image_deg"})
    frame = frame.merge(peak_direction, on="rr100_index", validate="many_to_one")
    frame["angular_difference_from_peak_direction_deg"] = [
        circular_distance(left, right)
        for left, right in zip(frame.motion_direction_image_deg, frame.peak_motion_direction_image_deg, strict=True)
    ]
    return frame, arrays


def plot_design(frame: pd.DataFrame, arrays: dict[str, np.ndarray], path: Path) -> None:
    units = frame.rr100_index.drop_duplicates().to_list()
    tensor_units = arrays["rr100_index"].astype(int)
    sf = arrays["spatial_cpd"].astype(float)
    tf = arrays["temporal_hz"].astype(float)
    directions = arrays["motion_direction_image_deg"].astype(float)
    positive = arrays["positive_f0_hz"].astype(float)
    figure = plt.figure(figsize=(17, 3.4 * len(units)), constrained_layout=True)
    grid = figure.add_gridspec(len(units), 5)
    for row, unit in enumerate(units):
        position = int(np.flatnonzero(tensor_units == int(unit))[0])
        unit_frame = frame.loc[frame.rr100_index.eq(unit)].reset_index(drop=True)
        sensitivity = positive[position].sum(axis=-1)
        axis = figure.add_subplot(grid[row, 0])
        image = axis.imshow(sensitivity.T, origin="lower", aspect="auto", cmap="magma")
        axis.set_xticks(range(len(sf)), [f"{value:g}" for value in sf], rotation=45)
        axis.set_yticks(range(len(tf)), [f"{value:g}" for value in tf])
        axis.set(xlabel="spatial frequency (cycles/degree)", ylabel="temporal frequency (Hz)", title="direction-summed positive F0")
        for probe in unit_frame.itertuples(index=False):
            if "opposite drift" not in probe.probe_role:
                axis.scatter(probe.sf_index, probe.tf_index, s=55, facecolors="none", edgecolors="cyan", linewidths=1.5)
        figure.colorbar(image, ax=axis, label="summed positive F0 (Hz)", shrink=0.7)

        for column, probe in enumerate(unit_frame.itertuples(index=False), start=1):
            if column >= 4:
                break
            profile = positive[position, int(probe.sf_index), int(probe.tf_index)]
            profile_axis = figure.add_subplot(grid[row, column], projection="polar")
            angles = np.deg2rad(np.r_[directions, directions[0]])
            values = np.r_[profile, profile[0]]
            profile_axis.plot(angles, values, marker="o")
            profile_axis.set_theta_zero_location("E")
            profile_axis.set_theta_direction(1)
            profile_axis.set_title(
                f"{probe.probe_role}\nSF {probe.spatial_frequency_cpd:g}; TF {probe.temporal_frequency_magnitude_hz:g} Hz\n"
                f"motion {probe.motion_direction_image_deg:g}°; bar {probe.bar_orientation_image_deg:g}°; {probe.drift_sign} drift",
                fontsize=8,
            )

        image_axis = figure.add_subplot(grid[row, 4])
        peak = unit_frame.iloc[0]
        cube = make_probe_cube(
            NATIVE_SIZE,
            sf=float(peak.spatial_frequency_cpd),
            tf=float(peak.signed_temporal_frequency_hz_for_cartographer_renderer),
            orientation_deg=float(peak.bar_orientation_image_deg),
        )
        image_axis.imshow(cube[0], cmap="gray", origin="lower")
        image_axis.set_title(
            "localized peak probe, current lag\n"
            f"empirical signed/positive F0={peak.empirical_signed_f0_hz:.2f}/{peak.empirical_positive_f0_hz:.2f} Hz",
            fontsize=8,
        )
        image_axis.set_xticks([]); image_axis.set_yticks([])
        image_axis.set_ylabel(
            f"RR100 unit {unit}\n{unit_frame.selection_role.iloc[0].replace('_', ' ')}",
            fontsize=9,
        )
    figure.suptitle(
        "Cartographer Stage 3 directional-probe design: empirical tensor cells preserve bar orientation, drift sign, and frequency-dependent angle",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed design checkpoint exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    frame, arrays = select_probes()
    frame_path = OUT / "selected_empirical_directional_probes.csv"
    frame.to_csv(frame_path, index=False)
    histories = np.stack(
        [
            make_probe_cube(
                NATIVE_SIZE,
                sf=float(row.spatial_frequency_cpd),
                tf=float(row.signed_temporal_frequency_hz_for_cartographer_renderer),
                orientation_deg=float(row.bar_orientation_image_deg),
            )
            for row in frame.itertuples(index=False)
        ]
    )
    history_path = OUT / "selected_localized_probe_histories.npz"
    np.savez_compressed(
        history_path,
        localized_probe_histories=histories,
        rr100_index=frame.rr100_index.to_numpy(int),
        probe_role=frame.probe_role.to_numpy(dtype="U80"),
        spatial_frequency_cpd=frame.spatial_frequency_cpd.to_numpy(float),
        temporal_frequency_magnitude_hz=frame.temporal_frequency_magnitude_hz.to_numpy(float),
        motion_direction_image_deg=frame.motion_direction_image_deg.to_numpy(float),
        bar_orientation_image_deg=frame.bar_orientation_image_deg.to_numpy(float),
        signed_temporal_frequency_hz=frame.signed_temporal_frequency_hz_for_cartographer_renderer.to_numpy(float),
    )
    figure_path = OUT / "01_empirical_directional_probe_design"
    plot_design(frame, arrays, figure_path)
    frequency_role = frame.loc[frame.selection_role.eq("frequency_dependent_direction")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_empirical_directional_probe_design",
        "status": "input_design_checkpoint_complete_no_neural_scoring",
        "scope": {
            "n_units": int(frame.rr100_index.nunique()),
            "n_probes": int(len(frame)),
            "frequency_dependent_direction_unit": int(frequency_role.rr100_index.iloc[0]),
            "frequency_dependent_direction_probe_count": int(len(frequency_role)),
            "reserved_final_test_identities_opened": False,
        },
        "contracts": {
            "primary_source": "complete empirical SF x absolute-TF x eight-motion-direction tensor",
            "coordinates_saved": "SF, TF magnitude, motion direction, bar orientation, drift sign, signed TF",
            "frequency_dependent_angular_tuning": "preserved as a predeclared dissociation",
            "parametric_fits": "not used",
            "old_four_bin_orientation_field": "not used",
            "aperture_or_calibration_frozen": False,
            "panels_d_e_modified": False,
        },
        "decision_gate": "inspect selected tensor cells and rendered localized inputs before direct large-canvas neural scoring",
        "sources": {"tensor": identity(TENSOR), "selection": identity(SELECTION)},
        "outputs": {
            "probe_table": identity(frame_path),
            "probe_histories": identity(history_path),
            "probe_design_figure": identity(figure_path.with_suffix(".png")),
        },
        "runner": identity(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Cartographer Stage 3 empirical directional-probe design\n\n"
        "This input-only checkpoint selects localized grating probes directly from the complete empirical "
        "SF×|TF|×motion-direction tensor. Exact bar orientation and drift sign are saved separately. "
        "Frequency-dependent angular tuning is retained, parametric fits and the old four-bin preferred-"
        "orientation field are not used, and no aperture or map calibration is frozen. Stop for inspection "
        "before neural scoring.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

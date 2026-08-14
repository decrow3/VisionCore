#!/usr/bin/env python3
"""Design an adapter-aware, phase-matched native-to-map tuning checkpoint."""
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

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v1"
TENSOR = TUNING / "rr100_empirical_joint_sftf_direction_tuning.npz"
SELECTION = TUNING / "selected_unit_roles.csv"
PRIOR_PROBES = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1/"
    "selected_empirical_directional_probes.csv"
)
NATIVE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_production_v1"
)
CONDITIONS = NATIVE / "condition_table.csv"
MAPPING = NATIVE / "rr100_unit_mapping.csv"
PATHWAY_AUDIT = ROOT / "outputs/fig4_active_sensing/rr100_stage3_pathway_contract_audit_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_stage3_matched_tuning_transfer_design_v2"


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


def circular_distance(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


def native_rendering_for_image_motion(
    motion_direction_deg: float, temporal_hz: float
) -> tuple[float, float, str, float]:
    """Map image motion to image bar axis and the native renderer convention.

    The cached native tensor defines image coordinates by
    bar_image=(-orientation_native) mod 180 and, for positive native TF,
    motion_image=(90-orientation_native) mod 360.  We invert those equations
    explicitly; image bar orientation must never be passed directly as the
    native renderer orientation for oblique probes.
    """
    direction = float(motion_direction_deg) % 360.0
    native_orientation = (90.0 - direction) % 180.0
    image_bar = (-native_orientation) % 180.0
    positive_direction = (90.0 - native_orientation) % 360.0
    sign = 1.0 if circular_distance(direction, positive_direction) < 1e-8 else -1.0
    return (
        image_bar,
        native_orientation,
        "positive" if sign > 0 else "negative",
        sign * float(temporal_hz),
    )


def choose_slices(
    selected: pd.DataFrame,
    prior: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    units = arrays["rr100_index"].astype(int)
    sf = arrays["spatial_cpd"].astype(float)
    tf = arrays["temporal_hz"].astype(float)
    signed_all = arrays["signed_f0_hz"].astype(float)
    positive_all = arrays["positive_f0_hz"].astype(float)
    rows: list[dict[str, object]] = []
    for role in selected.itertuples(index=False):
        unit = int(role.rr100_index)
        unit_position = int(np.flatnonzero(units == unit)[0])
        signed = signed_all[unit_position]
        positive = positive_all[unit_position]
        unit_prior = prior.loc[prior.rr100_index.eq(unit)]
        primary_role = (
            "empirical peak tensor cell"
            if unit_prior.probe_role.eq("empirical peak tensor cell").any()
            else "least-suppressed empirical control cell"
        )
        primary = unit_prior.loc[unit_prior.probe_role.eq(primary_role)].iloc[0]
        selected_pairs = {(int(primary.sf_index), int(primary.tf_index))}
        rows.append(
            {
                "rr100_index": unit,
                "selection_role": str(role.selection_role),
                "slice_role": "primary empirical tensor slice",
                "slice_selection_criterion": str(primary.probe_selection_criterion),
                "sf_index": int(primary.sf_index),
                "tf_index": int(primary.tf_index),
                "spatial_frequency_cpd": float(primary.spatial_frequency_cpd),
                "temporal_frequency_magnitude_hz": float(primary.temporal_frequency_magnitude_hz),
            }
        )
        changed = unit_prior.loc[
            unit_prior.probe_role.eq("strong frequency slice with changed preferred direction")
        ]
        if not changed.empty:
            alternate = changed.iloc[0]
            alternate_sf = int(alternate.sf_index)
            alternate_tf = int(alternate.tf_index)
            criterion = str(alternate.probe_selection_criterion)
            alternate_role = "frequency-dependent angular dissociation slice"
        else:
            peak_sf, peak_tf = next(iter(selected_pairs))
            candidates: list[tuple[float, float, int, int]] = []
            for sf_index in range(len(sf)):
                for tf_index in range(len(tf)):
                    if (sf_index, tf_index) in selected_pairs:
                        continue
                    distance = float(
                        np.hypot(
                            np.log2(sf[sf_index] / sf[peak_sf]),
                            np.log2(tf[tf_index] / tf[peak_tf]),
                        )
                    )
                    least_suppressed = float(np.max(signed[sf_index, tf_index]))
                    candidates.append((least_suppressed, distance, sf_index, tf_index))
            top_threshold = float(np.quantile([value[0] for value in candidates], 0.75))
            eligible = [value for value in candidates if value[0] >= top_threshold]
            _, _, alternate_sf, alternate_tf = max(eligible, key=lambda value: (value[1], value[0]))
            criterion = (
                "most distant SF-by-TF slice from the least-suppressed primary among slices in the top quartile "
                "of maximum signed F0; predeclared weak-response control"
            )
            alternate_role = "frequency-separated weak-response control slice"
        rows.append(
            {
                "rr100_index": unit,
                "selection_role": str(role.selection_role),
                "slice_role": alternate_role,
                "slice_selection_criterion": criterion,
                "sf_index": int(alternate_sf),
                "tf_index": int(alternate_tf),
                "spatial_frequency_cpd": float(sf[alternate_sf]),
                "temporal_frequency_magnitude_hz": float(tf[alternate_tf]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["rr100_index", "sf_index", "tf_index"]).any():
        raise ValueError("Each unit must have two distinct SF-by-TF slices")
    return frame


def expand_cells(
    slices: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    tensor_units = arrays["rr100_index"].astype(int)
    directions = arrays["motion_direction_image_deg"].astype(float)
    signed_all = arrays["signed_f0_hz"].astype(float)
    positive_all = arrays["positive_f0_hz"].astype(float)
    mapping = pd.read_csv(MAPPING)
    rows: list[dict[str, object]] = []
    for slice_row in slices.itertuples(index=False):
        unit = int(slice_row.rr100_index)
        unit_position = int(np.flatnonzero(tensor_units == unit)[0])
        profile = positive_all[unit_position, int(slice_row.sf_index), int(slice_row.tf_index)]
        preferred_direction = float(directions[int(np.argmax(profile))])
        for direction_index, direction in enumerate(directions):
            bar, native_orientation, drift_sign, signed_tf = native_rendering_for_image_motion(
                float(direction), float(slice_row.temporal_frequency_magnitude_hz)
            )
            rows.append(
                {
                    **slice_row._asdict(),
                    "direction_index": int(direction_index),
                    "motion_direction_image_deg": float(direction),
                    "bar_orientation_image_deg": float(bar),
                    "native_renderer_orientation_deg": float(native_orientation),
                    "drift_sign": drift_sign,
                    "signed_temporal_frequency_hz": float(signed_tf),
                    "native_tensor_signed_f0_hz": float(
                        signed_all[unit_position, int(slice_row.sf_index), int(slice_row.tf_index), direction_index]
                    ),
                    "native_tensor_positive_f0_hz": float(
                        positive_all[unit_position, int(slice_row.sf_index), int(slice_row.tf_index), direction_index]
                    ),
                    "slice_preferred_motion_direction_image_deg": preferred_direction,
                    "angular_difference_from_slice_preference_deg": circular_distance(
                        float(direction), preferred_direction
                    ),
                }
            )
    frame = pd.DataFrame(rows).merge(
        mapping[["rr100_index", "session", "source_unit_index", "canonical_channel"]],
        on="rr100_index",
        validate="many_to_one",
    )
    frame.insert(0, "routing_cell_id", np.arange(len(frame), dtype=int))
    return frame


def attach_native_phase_schedule(cells: pd.DataFrame) -> pd.DataFrame:
    conditions = pd.read_csv(CONDITIONS)
    dynamic = conditions.loc[conditions.condition_kind.eq("drifting_grating")].copy()
    rows: list[dict[str, object]] = []
    for cell in cells.itertuples(index=False):
        matches = dynamic.loc[
            np.isclose(dynamic.spatial_cpd, float(cell.spatial_frequency_cpd))
            & np.isclose(dynamic.signed_temporal_hz, float(cell.signed_temporal_frequency_hz))
            & np.isclose(dynamic.orientation_deg, float(cell.native_renderer_orientation_deg))
        ].sort_values("phase_index")
        if matches.empty:
            raise ValueError(
                f"No native production condition for unit {cell.rr100_index}, "
                f"SF={cell.spatial_frequency_cpd}, signed TF={cell.signed_temporal_frequency_hz}, "
                f"image bar={cell.bar_orientation_image_deg}, native renderer orientation={cell.native_renderer_orientation_deg}"
            )
        for condition in matches.itertuples(index=False):
            rows.append(
                {
                    **cell._asdict(),
                    "native_condition_id": int(condition.condition_id),
                    "phase_index": int(condition.phase_index),
                    "n_phases": int(condition.n_phases),
                    "phase_rad": float(condition.phase_rad),
                    "n_valid_response_frames": int(condition.n_valid_response_frames),
                    "valid_response_duration_s": float(condition.valid_response_duration_s),
                    "dynamic_cycles_observed": float(condition.dynamic_cycles_observed),
                    "history_frames": 33,
                    "history_lag_order": "current,t-1,...,t-32",
                    "session_adapter_application": "apply selected session adapter to every native 33-frame history before canvas comparison",
                    "native_scalar_path": "adapter output -> shared core -> exact selected session readout",
                    "large_canvas_path": "same adapter output embedded in 151-by-151 zero-normalized context -> shared core -> assembled matching readout center",
                    "response_estimand": "phase-specific temporal mean in Hz after 32 history-warmup frames; subtract matched pathway blank; then average phases",
                }
            )
    return pd.DataFrame(rows)


def plot_design(
    cells: pd.DataFrame,
    schedule: pd.DataFrame,
    path: Path,
) -> None:
    units = cells.rr100_index.drop_duplicates().to_list()
    figure, axes = plt.subplots(len(units), 3, figsize=(16, 3.2 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units):
        unit_cells = cells.loc[cells.rr100_index.eq(unit)]
        slice_roles = unit_cells.slice_role.drop_duplicates().to_list()
        for column, slice_role in enumerate(slice_roles):
            profile = unit_cells.loc[unit_cells.slice_role.eq(slice_role)].sort_values("motion_direction_image_deg")
            axes[row, column].plot(
                profile.motion_direction_image_deg,
                profile.native_tensor_signed_f0_hz,
                marker="o",
                linewidth=2,
            )
            axes[row, column].axhline(0, color="0.4", linewidth=0.8)
            axes[row, column].set_xticks(np.arange(0, 360, 45))
            axes[row, column].set(
                xlabel="motion direction in image coordinates (degrees)",
                ylabel="native phase-averaged firing-rate modulation (Hz)",
                title=(
                    f"{slice_role}\nspatial frequency {profile.spatial_frequency_cpd.iloc[0]:g} cycles/degree; "
                    f"temporal frequency {profile.temporal_frequency_magnitude_hz.iloc[0]:g} Hz"
                ),
            )
            axes[row, column].grid(alpha=0.2)
        text_axis = axes[row, 2]
        text_axis.axis("off")
        unit_schedule = schedule.loc[schedule.rr100_index.eq(unit)]
        text_axis.text(
            0.0,
            0.95,
            f"RR100 unit {unit}\n{unit_cells.selection_role.iloc[0].replace('_', ' ')}\n\n"
            f"16 routing cells: two SF-by-TF slices × eight directions\n"
            f"{len(unit_schedule)} phase-specific render conditions\n"
            f"{int(unit_schedule.n_valid_response_frames.sum()):,} valid response histories\n\n"
            "Matched comparison:\n"
            "1. render the original session-native 51×51 grating movie;\n"
            "2. construct 33-frame current-through-t−32 histories;\n"
            "3. apply the learned session adapter once;\n"
            "4. send the identical adapted history to the 51-input scalar\n"
            "   and the center of the 151-input activation-map pathway;\n"
            "5. multiply expected counts/frame by 120, subtract matched\n"
            "   blanks, temporally average, then average carrier phases.",
            va="top",
            fontsize=10,
            linespacing=1.35,
        )
    figure.suptitle(
        "Stage 3 matched tuning-transfer input design: compare phase-averaged directional profiles under identical adapted histories",
        fontsize=15,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed matched-transfer design exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTION)
    prior = pd.read_csv(PRIOR_PROBES)
    with np.load(TENSOR, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    slices = choose_slices(selected, prior, arrays)
    cells = expand_cells(slices, arrays, selected)
    schedule = attach_native_phase_schedule(cells)
    slices.to_csv(OUT / "selected_sf_tf_slices.csv", index=False)
    cells.to_csv(OUT / "matched_directional_routing_cells.csv", index=False)
    schedule.to_csv(OUT / "phase_specific_render_schedule.csv", index=False)
    plot_design(cells, schedule, OUT / "01_matched_tuning_transfer_input_design")

    unique_render = schedule.drop_duplicates(
        ["session", "spatial_frequency_cpd", "signed_temporal_frequency_hz", "bar_orientation_image_deg", "phase_index"]
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_stage3_matched_tuning_transfer_input_design",
        "status": "input_design_complete_no_neural_scoring",
        "scope": {
            "n_units": int(cells.rr100_index.nunique()),
            "n_sf_tf_slices": int(len(slices)),
            "n_directional_routing_cells": int(len(cells)),
            "n_phase_specific_unit_conditions": int(len(schedule)),
            "n_unique_session_render_conditions": int(len(unique_render)),
            "n_valid_response_histories_before_batching": int(unique_render.n_valid_response_frames.sum()),
            "reserved_final_test_identities_opened": False,
        },
        "contracts": {
            "slice_selection": "two predeclared SF-by-TF slices per unit; primary plus frequency-dependent or frequency-separated control",
            "angular_sampling": "all eight empirical motion directions in every slice",
            "phase_and_duration": "exact native production phase count and valid-response duration for every condition",
            "history": "33 frames with lag order current through t-32",
            "adapter": "apply the selected session's learned adapter before branching to scalar and map pathways",
            "rate_units": "multiply model expected counts per frame by 120 before storing Hz",
            "blank": "score and subtract a matched blank separately in both pathways before phase averaging",
            "parametric_fits_used": False,
            "old_four_bin_orientation_used": False,
            "aperture_fitted": False,
        },
        "decision_gate": (
            "inspect the two complete direction profiles and exact render schedule per unit before GPU scoring; "
            "this design tests tuning-shape transfer, not a fitted receptive-field aperture"
        ),
        "sources": {
            "empirical_tensor": identity(TENSOR),
            "selected_roles": identity(SELECTION),
            "prior_probe_design": identity(PRIOR_PROBES),
            "native_condition_table": identity(CONDITIONS),
            "native_unit_mapping": identity(MAPPING),
            "pathway_audit": identity(PATHWAY_AUDIT / "manifest.json"),
            "runner": identity(Path(__file__)),
        },
        "outputs": {
            name: identity(OUT / name)
            for name in (
                "selected_sf_tf_slices.csv",
                "matched_directional_routing_cells.csv",
                "phase_specific_render_schedule.csv",
                "01_matched_tuning_transfer_input_design.png",
            )
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Stage 3 matched tuning-transfer input design\n\n"
        "This input-only checkpoint defines a like-for-like adapter-aware comparison between the native scalar "
        "and central activation-map pathways. It uses the exact native phase/duration schedule, 33-frame histories, "
        "matched blank subtraction, and physical Hz conversion. No neural responses were scored.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

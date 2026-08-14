#!/usr/bin/env python3
"""Audit new native-grating versus legacy BackImage orientation tuning.

This is a targeted provenance and unit-selection checkpoint.  It does not
regenerate Figure 4 population curves.  Native grating angles use the opposite
x handedness from the legacy image-array grating helper, so the comparison uses
theta_image = (-theta_native) mod 180.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
LEGACY_PROBE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1/cache/"
    "backimage_rr100_orientation_probe_tuning.npz"
)
NEW_DIR = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
HALVES = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_half_checks_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_new_vs_legacy_orientation_audit_v1"


def identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def axial_delta(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(left, dtype=float) - np.asarray(right, dtype=float) + 90.0) % 180.0 - 90.0)


def continuous_preference(orientations_deg: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(np.asarray(orientations_deg, dtype=float))
    values = np.maximum(np.asarray(responses, dtype=float), 0.0)
    vector = np.sum(values * np.exp(2j * theta)[None, :], axis=1)
    total = np.sum(values, axis=1)
    preferred = (0.5 * np.rad2deg(np.angle(vector))) % 180.0
    selectivity = np.divide(np.abs(vector), total, out=np.zeros_like(total), where=total > 1e-12)
    return preferred, selectivity


def relation_for_delta(delta: np.ndarray) -> np.ndarray:
    return np.where(delta <= 22.5, "aligned", np.where(delta >= 67.5, "orthogonal", "intermediate"))


def selection_sets(
    units: pd.DataFrame,
    images: pd.DataFrame,
    *,
    preference_col: str,
    selectivity_col: str,
) -> dict[tuple[int, str], set[int]]:
    axes = pd.to_numeric(images["image_edge_axis_deg"], errors="coerce").to_numpy(float)
    indices = images["image_index"].astype(int).to_numpy()
    strong = images["image_contour_strong"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    output: dict[tuple[int, str], set[int]] = {}
    for row in units.itertuples(index=False):
        unit = int(row.rr100_index)
        pref = float(getattr(row, preference_col))
        osi = float(getattr(row, selectivity_col))
        if not (math.isfinite(pref) and math.isfinite(osi) and osi >= 0.05):
            for relation in ("aligned", "intermediate", "orthogonal"):
                output[(unit, relation)] = set()
            continue
        relation = relation_for_delta(axial_delta(axes, pref))
        for name in ("aligned", "intermediate", "orthogonal"):
            output[(unit, name)] = set(indices[strong & np.isfinite(axes) & (relation == name)].tolist())
    return output


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def harmonic_curve(orientations: np.ndarray, responses: np.ndarray, query: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(orientations)
    design = np.column_stack([np.ones(len(theta)), np.cos(2 * theta), np.sin(2 * theta)])
    coef = np.linalg.pinv(design) @ responses
    q = np.deg2rad(query)
    return np.maximum(coef[0] + coef[1] * np.cos(2 * q) + coef[2] * np.sin(2 * q), 0.0)


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_csv(MATRIX / "unit_feature_table.csv").rename(columns={"unit_index": "rr100_index"})
    new_metrics = pd.read_csv(NEW_DIR / "orientation_tuning_fit_quality_and_movie_overlap.csv")
    halves = pd.read_csv(HALVES / "sf_half_unit_assignments.csv")[["rr100_index", "sf_half", "model_valid"]]
    images = pd.read_csv(MATRIX / "image_feature_table.csv")

    with np.load(NEW_DIR / "orientation_aware_f0_tuning_and_routing.npz", allow_pickle=False) as data:
        new_units = np.asarray(data["rr100_index"], dtype=int)
        new_orientations = np.asarray(data["measured_grating_orientation_deg"], dtype=float)
        new_marginal = np.asarray(data["measured_positive_f0_hz"], dtype=float).mean(axis=(1, 2))
    new_pref_native, new_osi_recomputed = continuous_preference(new_orientations, new_marginal)
    derived = pd.DataFrame(
        {
            "rr100_index": new_units,
            "new_preferred_orientation_native_deg": new_pref_native,
            "new_preferred_orientation_image_deg": (-new_pref_native) % 180.0,
            "new_orientation_selectivity_recomputed": new_osi_recomputed,
        }
    )

    units = (
        legacy.merge(new_metrics, on="rr100_index", how="inner", validate="one_to_one", suffixes=("_legacy", "_new"))
        .merge(derived, on="rr100_index", validate="one_to_one")
        .merge(halves, on="rr100_index", how="left", validate="one_to_one")
    )
    units = units.rename(
        columns={
            "prior_preferred_orientation_deg": "legacy_preferred_orientation_image_deg",
            "prior_orientation_selectivity_index": "legacy_orientation_selectivity",
            "preferred_orientation_deg": "new_preferred_orientation_grid_native_deg",
            "orientation_vector_strength": "new_orientation_selectivity",
        }
    )
    units["new_preferred_orientation_grid_image_deg"] = (-units["new_preferred_orientation_grid_native_deg"]) % 180.0
    units["raw_native_vs_legacy_delta_deg"] = axial_delta(
        units["legacy_preferred_orientation_image_deg"], units["new_preferred_orientation_native_deg"]
    )
    units["converted_new_vs_legacy_delta_deg"] = axial_delta(
        units["legacy_preferred_orientation_image_deg"], units["new_preferred_orientation_image_deg"]
    )
    units["agreement_within_22p5deg"] = units["converted_new_vs_legacy_delta_deg"] <= 22.5
    units["large_change_ge_45deg"] = units["converted_new_vs_legacy_delta_deg"] >= 45.0

    legacy_sets = selection_sets(
        units, images, preference_col="legacy_preferred_orientation_image_deg",
        selectivity_col="legacy_orientation_selectivity",
    )
    new_pref_sets = selection_sets(
        units, images, preference_col="new_preferred_orientation_image_deg",
        selectivity_col="legacy_orientation_selectivity",
    )
    new_full_sets = selection_sets(
        units, images, preference_col="new_preferred_orientation_image_deg",
        selectivity_col="new_orientation_selectivity",
    )
    selection_rows: list[dict[str, object]] = []
    for row in units.itertuples(index=False):
        for relation in ("aligned", "intermediate", "orthogonal"):
            key = (int(row.rr100_index), relation)
            old = legacy_sets[key]
            pref = new_pref_sets[key]
            full = new_full_sets[key]
            selection_rows.append(
                {
                    "rr100_index": int(row.rr100_index),
                    "sf_half": str(row.sf_half),
                    "relation": relation,
                    "legacy_n_images": len(old),
                    "new_preference_legacy_osi_n_images": len(pref),
                    "new_preference_new_osi_n_images": len(full),
                    "legacy_vs_new_preference_jaccard": jaccard(old, pref),
                    "legacy_vs_full_new_jaccard": jaccard(old, full),
                    "legacy_image_indices": " ".join(map(str, sorted(old))),
                    "new_preference_image_indices": " ".join(map(str, sorted(pref))),
                    "full_new_image_indices": " ".join(map(str, sorted(full))),
                }
            )
    selection = pd.DataFrame(selection_rows)
    aligned = selection[selection["relation"].eq("aligned")]
    units = units.merge(
        aligned[
            [
                "rr100_index", "legacy_n_images", "new_preference_legacy_osi_n_images",
                "new_preference_new_osi_n_images", "legacy_vs_new_preference_jaccard",
                "legacy_vs_full_new_jaccard",
            ]
        ],
        on="rr100_index", how="left", validate="one_to_one",
    )

    eligible = units[units["model_valid"].fillna(False).astype(bool)].copy()
    roles = []
    role_specs = [
        ("largest_preference_change", eligible["converted_new_vs_legacy_delta_deg"].idxmax()),
        ("closest_preference_agreement", eligible["converted_new_vs_legacy_delta_deg"].idxmin()),
        ("largest_aligned_selection_change", (eligible["legacy_n_images"] - eligible["new_preference_new_osi_n_images"]).abs().idxmax()),
        ("high_sf_largest_change", eligible[eligible["sf_half"].eq("sf_high_half")]["converted_new_vs_legacy_delta_deg"].idxmax()),
    ]
    for role, index in role_specs:
        record = eligible.loc[index].copy()
        record["selection_role"] = role
        roles.append(record)
    selected = pd.DataFrame(roles).drop_duplicates("rr100_index", keep="first")

    with np.load(LEGACY_PROBE, allow_pickle=False) as data:
        legacy_orientations = np.asarray(data["orientations_deg"], dtype=float)
        legacy_units = np.asarray(data["selected_units"], dtype=int)
        # The saved legacy preferred orientation and OSI were computed from the
        # spatially averaged activation map, not the center-pixel diagnostic.
        legacy_rates = np.asarray(data["unit_mean_rate"], dtype=float)
    legacy_position = {int(unit): i for i, unit in enumerate(legacy_units)}

    fig = plt.figure(figsize=(13.5, 9.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(
        eligible["legacy_preferred_orientation_image_deg"],
        eligible["new_preferred_orientation_image_deg"],
        c=eligible["converted_new_vs_legacy_delta_deg"], cmap="magma", vmin=0, vmax=90,
        s=32, edgecolor="white", linewidth=0.4,
    )
    ax.plot([0, 180], [0, 180], color="0.35", ls="--", lw=1)
    ax.set(xlim=(0, 180), ylim=(0, 180), xlabel="legacy preferred orientation (deg)", ylabel="new preferred orientation (deg)")
    ax.set_title("After native-to-image coordinate conversion", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    ax.hist(eligible["converted_new_vs_legacy_delta_deg"], bins=np.arange(0, 95, 7.5), color="#3B6FB6", edgecolor="white")
    ax.axvline(22.5, color="0.2", ls="--", lw=1.2)
    ax.axvline(45.0, color="#B94A48", ls=":", lw=1.2)
    ax.set(xlabel="axial preferred-orientation difference (deg)", ylabel="units")
    ax.set_title("Difference among 85 valid-SF units", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 2])
    ax.scatter(
        eligible["legacy_orientation_selectivity"], eligible["new_orientation_selectivity"],
        c=eligible["converted_new_vs_legacy_delta_deg"], cmap="magma", vmin=0, vmax=90,
        s=32, edgecolor="white", linewidth=0.4,
    )
    ax.axvline(0.05, color="0.4", ls="--", lw=1)
    ax.axhline(0.05, color="0.4", ls="--", lw=1)
    ax.set(xlabel="legacy orientation selectivity", ylabel="new orientation vector strength")
    ax.set_title("The OSI gate also changes", loc="left", fontweight="bold")

    query_image = np.linspace(0.0, 180.0, 361, endpoint=False)
    for panel, row in zip(range(3), selected.itertuples(index=False), strict=False):
        ax = fig.add_subplot(grid[1, panel])
        position = legacy_position[int(row.rr100_index)]
        legacy_curve = normalized(legacy_rates[:, position])
        new_position = int(np.flatnonzero(new_units == int(row.rr100_index))[0])
        native_query = (-query_image) % 180.0
        new_curve = harmonic_curve(new_orientations, new_marginal[new_position], native_query)
        new_curve = normalized(new_curve)
        new_points_image = (-new_orientations) % 180.0
        order = np.argsort(new_points_image)
        ax.plot(legacy_orientations, legacy_curve, color="#3B6FB6", lw=2, label="legacy static probe")
        ax.plot(query_image, new_curve, color="#D55E00", lw=2, label="new SFxTF-marginal F0")
        ax.scatter(
            new_points_image[order], normalized(new_marginal[new_position])[order],
            color="#D55E00", s=28, zorder=3,
        )
        ax.axvline(float(row.legacy_preferred_orientation_image_deg), color="#3B6FB6", ls="--", lw=1)
        ax.axvline(float(row.new_preferred_orientation_image_deg), color="#D55E00", ls="--", lw=1)
        ax.set(xlim=(0, 180), ylim=(-0.05, 1.05), xticks=[0, 45, 90, 135, 180], xlabel="bar orientation in image-array frame (deg)")
        if panel == 0:
            ax.set_ylabel("normalized response")
        ax.set_title(
            f"RR100 {int(row.rr100_index)} · {row.selection_role.replace('_', ' ')}\n"
            f"legacy {row.legacy_preferred_orientation_image_deg:.1f}°, new {row.new_preferred_orientation_image_deg:.1f}°, "
            f"Δ={row.converted_new_vs_legacy_delta_deg:.1f}°",
            fontsize=9, loc="left",
        )
        if panel == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "RR100 preferred orientation changed, and the refreshed Figure 4 alignment panels still use the legacy field\n"
        "New native angles are reflected into the legacy image-array coordinate frame before comparison",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.savefig(OUT / "new_vs_legacy_orientation_checkpoint.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "new_vs_legacy_orientation_checkpoint.pdf", bbox_inches="tight")
    plt.close(fig)

    units.to_csv(OUT / "unit_orientation_comparison.csv", index=False)
    selection.to_csv(OUT / "panel_alignment_selection_comparison.csv", index=False)
    selected.to_csv(OUT / "selected_unit_roles.csv", index=False)

    def cohort_summary(frame: pd.DataFrame) -> dict[str, object]:
        delta = frame["converted_new_vs_legacy_delta_deg"].to_numpy(float)
        aligned_frame = frame.merge(
            aligned[["rr100_index", "legacy_vs_new_preference_jaccard", "legacy_vs_full_new_jaccard"]],
            on="rr100_index", suffixes=("", "_selection"), how="left",
        )
        return {
            "n_units": int(len(frame)),
            "median_axial_difference_deg": float(np.median(delta)),
            "mean_axial_difference_deg": float(np.mean(delta)),
            "fraction_within_22p5_deg": float(np.mean(delta <= 22.5)),
            "fraction_ge_45_deg": float(np.mean(delta >= 45.0)),
            "median_aligned_image_jaccard_new_preference_legacy_osi": float(np.nanmedian(aligned_frame["legacy_vs_new_preference_jaccard"])),
            "median_aligned_image_jaccard_full_new": float(np.nanmedian(aligned_frame["legacy_vs_full_new_jaccard"])),
        }

    summary = {
        "status": "targeted_orientation_provenance_and_selection_checkpoint_complete",
        "coordinate_contract": {
            "legacy": "image_array_x_right_y_down; bar/edge axis",
            "new_native": "native grating renderer; bar axis with opposite x handedness",
            "conversion": "theta_image = (-theta_native) mod 180",
        },
        "figure4_finding": (
            "The SF-halves Panel D/E source tables use prior_preferred_orientation_deg and "
            "prior_orientation_selectivity_index from the legacy matrix unit table."
        ),
        "all_100": cohort_summary(units),
        "valid_sf_85": cohort_summary(eligible),
        "recorded_sf_validated_61": cohort_summary(units[units["recorded_validation_pass"].fillna(False).astype(bool)]),
        "sources": {
            "legacy_unit_table": identity(MATRIX / "unit_feature_table.csv"),
            "legacy_probe_cache": identity(LEGACY_PROBE),
            "new_tuning_metrics": identity(NEW_DIR / "orientation_tuning_fit_quality_and_movie_overlap.csv"),
            "new_tuning_arrays": identity(NEW_DIR / "orientation_aware_f0_tuning_and_routing.npz"),
            "sf_half_assignments": identity(HALVES / "sf_half_unit_assignments.csv"),
            "image_table": identity(MATRIX / "image_feature_table.csv"),
        },
        "next_step_not_run": "Regenerate Panel D/E selections and population summaries with the converted new orientation preference and new vector-strength gate.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

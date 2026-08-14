#!/usr/bin/env python3
"""Map-first checkpoint for RR100 orientation and direction tuning.

This analysis performs no twin inference.  It uses the completed native
signed-TF grating sweep to expose eight motion directions and compares its
axial orientation estimate with the independent 18-angle static BackImage
probe.  Population inference is deliberately deferred until the conditioned
unit profiles have been inspected.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_production_v1/"
    "native_condition_unit_summary.csv"
)
LEGACY = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1/cache/"
    "backimage_rr100_orientation_probe_tuning.npz"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_cached_orientation_direction_tuning_checkpoint_v1"


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def axial_delta(left: float | np.ndarray, right: float | np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(left, float) - np.asarray(right, float) + 90.0) % 180.0 - 90.0)


def vector_metrics(angles_deg: np.ndarray, weights: np.ndarray, harmonic: int) -> tuple[float, float]:
    angles = np.deg2rad(np.asarray(angles_deg, float))
    weights = np.maximum(np.asarray(weights, float), 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return float("nan"), 0.0
    vector = np.sum(weights * np.exp(1j * harmonic * angles))
    preference = (np.rad2deg(np.angle(vector)) / harmonic) % (360.0 / harmonic)
    return float(preference), float(abs(vector) / total)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    maximum = float(np.nanmax(np.maximum(values, 0.0)))
    return np.maximum(values, 0.0) / maximum if maximum > 1e-12 else np.zeros_like(values)


def prepare_native() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(NATIVE)
    dynamic = raw.loc[
        raw["condition_kind"].eq("drifting_grating") & raw["primary_fit_support"].astype(bool)
    ].copy()
    # Carrier phase is a nuisance replicate here.  Keep the signed above-blank
    # response so suppression remains visible in the saved conditioned table;
    # positive weights are used only for circular preference vectors.
    points = dynamic.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd", "signed_temporal_hz"],
        as_index=False,
    ).agg(
        signed_f0_hz=("mean_rate_above_blank_hz", "mean"),
        absolute_rate_hz=("mean_rate_hz", "mean"),
        f1_amplitude_hz=("f1_amplitude_hz", "mean"),
        n_carrier_phases=("phase_index", "nunique"),
    )
    points["temporal_hz"] = points["signed_temporal_hz"].abs()
    points["positive_f0_hz"] = points["signed_f0_hz"].clip(lower=0.0)
    # Native grating orientation is a bar axis with the opposite x handedness
    # to the legacy image-array helper.  Positive TF travels along +normal.
    points["bar_orientation_image_deg"] = (-points["orientation_deg"]) % 180.0
    points["motion_direction_image_deg"] = np.where(
        points["signed_temporal_hz"] > 0,
        (90.0 - points["orientation_deg"]) % 360.0,
        (270.0 - points["orientation_deg"]) % 360.0,
    )

    slice_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for unit, frame in points.groupby("rr100_index", sort=True):
        session = str(frame["session"].iloc[0])
        local: list[dict[str, object]] = []
        for (sf, tf), sub in frame.groupby(["spatial_cpd", "temporal_hz"], sort=True):
            sub = sub.sort_values("motion_direction_image_deg")
            direction, dsi = vector_metrics(
                sub["motion_direction_image_deg"].to_numpy(float), sub["positive_f0_hz"].to_numpy(float), 1
            )
            normal_axis, osi = vector_metrics(
                sub["motion_direction_image_deg"].to_numpy(float), sub["positive_f0_hz"].to_numpy(float), 2
            )
            row = {
                "rr100_index": int(unit),
                "session": session,
                "spatial_cpd": float(sf),
                "temporal_hz": float(tf),
                "total_positive_f0_hz": float(sub["positive_f0_hz"].sum()),
                "maximum_positive_f0_hz": float(sub["positive_f0_hz"].max()),
                "preferred_motion_direction_image_deg": direction,
                "direction_vector_strength": dsi,
                "preferred_bar_orientation_image_deg": (normal_axis - 90.0) % 180.0,
                "orientation_vector_strength": osi,
            }
            local.append(row)
            slice_rows.append(row)
        local_frame = pd.DataFrame(local)
        preferred_slice = local_frame.loc[local_frame["total_positive_f0_hz"].idxmax()]

        marginal = frame.groupby("motion_direction_image_deg", as_index=False)["positive_f0_hz"].sum()
        direction, dsi = vector_metrics(
            marginal["motion_direction_image_deg"].to_numpy(float), marginal["positive_f0_hz"].to_numpy(float), 1
        )
        normal_axis, osi = vector_metrics(
            marginal["motion_direction_image_deg"].to_numpy(float), marginal["positive_f0_hz"].to_numpy(float), 2
        )
        strong = local_frame[
            local_frame["total_positive_f0_hz"] >= 0.25 * local_frame["total_positive_f0_hz"].max()
        ]
        valid = strong[strong["direction_vector_strength"] > 1e-12]
        if len(valid):
            angles = np.deg2rad(valid["preferred_motion_direction_image_deg"].to_numpy(float))
            weights = (
                valid["direction_vector_strength"].to_numpy(float)
                * valid["total_positive_f0_hz"].to_numpy(float)
            )
            stability = float(abs(np.sum(weights * np.exp(1j * angles))) / max(weights.sum(), 1e-12))
        else:
            stability = 0.0
        metric_rows.append(
            {
                "rr100_index": int(unit),
                "session": session,
                "preferred_sf_cpd_by_direction_sum": float(preferred_slice["spatial_cpd"]),
                "preferred_tf_hz_by_direction_sum": float(preferred_slice["temporal_hz"]),
                "preferred_motion_direction_image_deg": direction,
                "marginal_direction_vector_strength": dsi,
                "preferred_bar_orientation_image_deg": (normal_axis - 90.0) % 180.0,
                "marginal_orientation_vector_strength": osi,
                "preferred_slice_direction_vector_strength": float(preferred_slice["direction_vector_strength"]),
                "preferred_slice_orientation_vector_strength": float(preferred_slice["orientation_vector_strength"]),
                "direction_consistency_across_strong_sf_tf_slices": stability,
                "n_strong_sf_tf_slices": int(len(strong)),
                "maximum_positive_f0_hz": float(frame["positive_f0_hz"].max()),
                "total_positive_f0_hz": float(frame["positive_f0_hz"].sum()),
                "minimum_signed_f0_hz": float(frame["signed_f0_hz"].min()),
            }
        )
    return points, pd.DataFrame(slice_rows), pd.DataFrame(metric_rows)


def add_legacy(metrics: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(LEGACY, allow_pickle=False) as cache:
        orientations = np.asarray(cache["orientations_deg"], float)
        units = np.asarray(cache["selected_units"], int)
        rates = np.asarray(cache["unit_mean_rate"], float)
    rows = []
    for position, unit in enumerate(units):
        preference, osi = vector_metrics(orientations, rates[:, position], 2)
        rows.append(
            {
                "rr100_index": int(unit),
                "legacy_static_preferred_bar_orientation_image_deg": preference,
                "legacy_static_orientation_vector_strength": osi,
            }
        )
    result = metrics.merge(pd.DataFrame(rows), on="rr100_index", validate="one_to_one")
    result["dynamic_vs_legacy_orientation_delta_deg"] = axial_delta(
        result["preferred_bar_orientation_image_deg"],
        result["legacy_static_preferred_bar_orientation_image_deg"],
    )
    return result, orientations, units, rates


def select_units(metrics: pd.DataFrame) -> pd.DataFrame:
    responsive = metrics[metrics["maximum_positive_f0_hz"] >= 0.5].copy()
    candidates: list[tuple[str, str, int]] = [
        (
            "strong_consistent_direction",
            "maximum marginal DSI x cross-SF/TF direction consistency",
            int((responsive["marginal_direction_vector_strength"] * responsive["direction_consistency_across_strong_sf_tf_slices"]).idxmax()),
        ),
        (
            "orientation_without_direction",
            "maximum marginal OSI minus marginal DSI",
            int((responsive["marginal_orientation_vector_strength"] - responsive["marginal_direction_vector_strength"]).idxmax()),
        ),
        (
            "frequency_dependent_direction",
            "maximum preferred-slice DSI minus marginal DSI",
            int((responsive["preferred_slice_direction_vector_strength"] - responsive["marginal_direction_vector_strength"]).idxmax()),
        ),
        (
            "static_dynamic_orientation_dissociation",
            "largest axial difference with OSI >= 0.05 in both probes",
            int(
                responsive[
                    (responsive["marginal_orientation_vector_strength"] >= 0.05)
                    & (responsive["legacy_static_orientation_vector_strength"] >= 0.05)
                ]["dynamic_vs_legacy_orientation_delta_deg"].idxmax()
            ),
        ),
        (
            "weak_response_control",
            "minimum maximum positive F0 over the full cached sweep",
            int(metrics["maximum_positive_f0_hz"].idxmin()),
        ),
    ]
    selected_rows = []
    used: set[int] = set()
    for role, criterion, index in candidates:
        row = metrics.loc[index].copy()
        unit = int(row["rr100_index"])
        if unit in used:
            pool = responsive.loc[~responsive["rr100_index"].isin(used)].copy()
            if role == "orientation_without_direction":
                index = (pool["marginal_orientation_vector_strength"] - pool["marginal_direction_vector_strength"]).idxmax()
            elif role == "frequency_dependent_direction":
                index = (pool["preferred_slice_direction_vector_strength"] - pool["marginal_direction_vector_strength"]).idxmax()
            else:
                index = pool["dynamic_vs_legacy_orientation_delta_deg"].idxmax()
            row = metrics.loc[index].copy()
            unit = int(row["rr100_index"])
        used.add(unit)
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def plot_support(points: pd.DataFrame) -> None:
    pairs = (
        points[["orientation_deg", "signed_temporal_hz", "bar_orientation_image_deg", "motion_direction_image_deg"]]
        .assign(sign=lambda x: np.where(x["signed_temporal_hz"] > 0, "+TF", "-TF"))
        .drop_duplicates(["orientation_deg", "sign"])
        .sort_values("motion_direction_image_deg")
    )
    fig = plt.figure(figsize=(11.5, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5])
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    theta = np.deg2rad(pairs["motion_direction_image_deg"].to_numpy(float))
    ax.scatter(theta, np.ones(len(theta)), s=70, color="#D55E00")
    for row, angle in zip(pairs.itertuples(index=False), theta):
        ax.text(angle, 1.18, f"{row.motion_direction_image_deg:.0f}°", ha="center", va="center", fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("Four bar axes × two TF signs\nproduce eight motion directions", pad=20, fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    table = pairs[["orientation_deg", "sign", "bar_orientation_image_deg", "motion_direction_image_deg"]].copy()
    table.columns = ["native bar axis", "TF sign", "image bar axis", "image motion direction"]
    table = table.map(lambda value: f"{value:g}°" if isinstance(value, (float, np.floating)) else value)
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1.0, 1.45)
    ax.set_title("Exact cached direction support and coordinate conversion", loc="left", fontweight="bold")
    fig.suptitle("RR100 cached signed-TF stimulus support", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "cached_signed_tf_direction_support.png", dpi=210, bbox_inches="tight")
    fig.savefig(OUT / "cached_signed_tf_direction_support.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selected(
    points: pd.DataFrame,
    selected: pd.DataFrame,
    legacy_orientations: np.ndarray,
    legacy_units: np.ndarray,
    legacy_rates: np.ndarray,
) -> None:
    directions = np.arange(0.0, 360.0, 45.0)
    unit_position = {int(unit): index for index, unit in enumerate(legacy_units)}
    fig = plt.figure(figsize=(15.5, 3.05 * len(selected)), constrained_layout=True)
    grid = fig.add_gridspec(len(selected), 4, width_ratios=[1.15, 1.0, 1.25, 1.25])
    for row_number, row in enumerate(selected.itertuples(index=False)):
        unit = int(row.rr100_index)
        frame = points[points["rr100_index"].eq(unit)].copy()
        sf0 = float(row.preferred_sf_cpd_by_direction_sum)
        tf0 = float(row.preferred_tf_hz_by_direction_sum)
        local = frame[np.isclose(frame["spatial_cpd"], sf0) & np.isclose(frame["temporal_hz"], tf0)]
        local = local.groupby("motion_direction_image_deg", as_index=False)["positive_f0_hz"].sum().set_index("motion_direction_image_deg").reindex(directions)
        marginal = frame.groupby("motion_direction_image_deg")["positive_f0_hz"].sum().reindex(directions)

        ax = fig.add_subplot(grid[row_number, 0])
        legacy_curve = normalize(legacy_rates[:, unit_position[unit]])
        dynamic_axis = frame.groupby("bar_orientation_image_deg")["positive_f0_hz"].sum().sort_index()
        ax.plot(legacy_orientations, legacy_curve, color="#3B6FB6", lw=2, label="18-angle static")
        ax.plot(dynamic_axis.index, normalize(dynamic_axis.to_numpy()), "o-", color="#D55E00", lw=1.8, label="dynamic marginal")
        ax.axvline(float(row.legacy_static_preferred_bar_orientation_image_deg), color="#3B6FB6", ls="--", lw=1)
        ax.axvline(float(row.preferred_bar_orientation_image_deg), color="#D55E00", ls="--", lw=1)
        ax.set(xlim=(0, 180), ylim=(-0.04, 1.05), xticks=[0, 45, 90, 135, 180], ylabel="normalized positive response")
        ax.set_title(
            f"RR100 {unit}: {str(row.selection_role).replace('_', ' ')}\n"
            f"orientation Δ={row.dynamic_vs_legacy_orientation_delta_deg:.1f}°",
            loc="left", fontsize=9, fontweight="bold",
        )
        if row_number == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = fig.add_subplot(grid[row_number, 1], projection="polar")
        closed = np.r_[directions, 360.0]
        ax.plot(np.deg2rad(closed), np.r_[normalize(local["positive_f0_hz"].to_numpy()), normalize(local["positive_f0_hz"].to_numpy())[0]], color="#D55E00", lw=2, label="preferred SF×TF")
        ax.plot(np.deg2rad(closed), np.r_[normalize(marginal.to_numpy()), normalize(marginal.to_numpy())[0]], color="0.25", lw=1.4, ls="--", label="all SF×TF")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_title(f"motion direction\nSF={sf0:g} cpd, TF={tf0:g} Hz", fontsize=9)
        if row_number == 0:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=7)

        for column, varying, fixed, title in (
            (2, "spatial_cpd", ("temporal_hz", tf0), f"direction × SF at TF={tf0:g} Hz"),
            (3, "temporal_hz", ("spatial_cpd", sf0), f"direction × TF at SF={sf0:g} cpd"),
        ):
            ax = fig.add_subplot(grid[row_number, column])
            subset = frame[np.isclose(frame[fixed[0]], fixed[1])]
            matrix = subset.pivot_table(index="motion_direction_image_deg", columns=varying, values="positive_f0_hz", aggfunc="sum").reindex(index=directions)
            values = normalize(matrix.to_numpy())
            image = ax.imshow(values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma")
            ax.set_yticks(range(len(directions)), [f"{value:g}°" for value in directions])
            ax.set_xticks(range(len(matrix.columns)), [f"{value:g}" for value in matrix.columns], rotation=45, ha="right")
            ax.set(xlabel="SF (cpd)" if varying == "spatial_cpd" else "TF (Hz)", ylabel="motion direction")
            ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
            if column == 3:
                fig.colorbar(image, ax=ax, fraction=0.045, label="unit-normalized positive F0")
    fig.suptitle(
        "Cached RR100 orientation/direction checkpoint: angular tuning can change across SF and TF\n"
        "Signed above-blank responses are preserved in CSV; plots use positive F0 only for circular weighting",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT / "selected_unit_cached_orientation_direction_profiles.png", dpi=210, bbox_inches="tight")
    fig.savefig(OUT / "selected_unit_cached_orientation_direction_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, slices, metrics = prepare_native()
    metrics, legacy_orientations, legacy_units, legacy_rates = add_legacy(metrics)
    selected = select_units(metrics)
    selected_units = selected["rr100_index"].astype(int).tolist()

    plot_support(points)
    plot_selected(points, selected, legacy_orientations, legacy_units, legacy_rates)

    points.to_csv(OUT / "cached_signed_tf_direction_points.csv", index=False)
    slices.to_csv(OUT / "unit_sf_tf_direction_metrics.csv", index=False)
    metrics.to_csv(OUT / "unit_orientation_direction_metrics.csv", index=False)
    selected.to_csv(OUT / "selected_unit_roles.csv", index=False)
    points[points["rr100_index"].isin(selected_units)].to_csv(
        OUT / "selected_unit_conditioned_direction_points.csv", index=False
    )

    summary = {
        "analysis": "rr100_cached_orientation_direction_tuning_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_conditioned_unit_checkpoint_complete_population_inference_not_run",
        "n_units": int(metrics["rr100_index"].nunique()),
        "native_support": {
            "spatial_cpds": sorted(points["spatial_cpd"].unique().tolist()),
            "temporal_hz_magnitudes": sorted(points["temporal_hz"].unique().tolist()),
            "native_bar_orientations_deg": sorted(points["orientation_deg"].unique().tolist()),
            "image_motion_directions_deg": sorted(points["motion_direction_image_deg"].unique().tolist()),
            "n_phase_collapsed_condition_points": int(len(points)),
        },
        "response_contract": {
            "saved_raw_quantity": "carrier-phase mean of mean_rate_above_blank_hz for each signed-TF condition",
            "circular_weight": "max(signed_f0_hz, 0)",
            "marginal": "sum positive F0 over all primary-support SF x |TF| conditions",
            "preferred_sf_tf": "SF x |TF| slice with largest sum of positive F0 over eight directions",
            "coordinate_conversion": "bar_image=(-bar_native) mod 180; +TF direction_image=(90-bar_native) mod 360; -TF adds 180",
        },
        "selected_units": selected[["rr100_index", "selection_role", "selection_criterion"]].to_dict("records"),
        "sources": {"native_signed_tf_summary": file_identity(NATIVE), "legacy_18_angle_static_probe": file_identity(LEGACY)},
        "next_step_not_run": (
            "After inspecting selected conditioned profiles, choose whether the production estimate should be "
            "preferred-region weighted, fully marginalized, or explicitly SF/TF-conditional; then bootstrap and summarize the population."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

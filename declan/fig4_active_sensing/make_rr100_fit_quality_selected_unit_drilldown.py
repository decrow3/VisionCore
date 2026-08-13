#!/usr/bin/env python3
"""Drill into selected units from checkpoint 15A using raw grating maps and LOO profiles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CP15A = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_predictor_checkpoint_15a_v1"
CP11 = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
SURFACE = (
    ROOT
    / "outputs/redundancy_resolved_v1_twin"
    / "rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
    / "f0_surface_fit_and_residual_points.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_selected_unit_checkpoint_15b_v1"


def log_gaussian(frequency: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    values = np.asarray(frequency, dtype=float)
    return baseline + amplitude * np.exp(-0.5 * ((np.log2(values) - center) / sigma) ** 2)


def build_surface_rows(selected: pd.DataFrame, models: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for selection in selected.itertuples(index=False):
        unit = int(selection.rr100_index)
        model = models.loc[unit]
        frame = points.loc[points["rr100_index"].eq(unit)].copy()
        sf_grid = np.sort(frame["spatial_cpd"].unique())
        tf_grid = np.sort(frame["temporal_hz"].unique())
        sf_factor = log_gaussian(
            sf_grid,
            model.sf_baseline,
            model.sf_amplitude,
            model.sf_center_log2_cpd,
            model.sf_sigma_octaves,
        )
        tf_factor = log_gaussian(
            tf_grid,
            model.tf_baseline,
            model.tf_amplitude,
            model.tf_center_log2_hz,
            model.tf_sigma_octaves,
        )
        # This sample-grid normalization exactly reproduces the saved joint-fit R2.
        fitted = model.joint_rank1_gain_f0_hz * np.outer(
            sf_factor / max(float(sf_factor.max()), 1e-15),
            tf_factor / max(float(tf_factor.max()), 1e-15),
        )
        fitted_long = pd.DataFrame(
            {
                "spatial_cpd": np.repeat(sf_grid, len(tf_grid)),
                "temporal_hz": np.tile(tf_grid, len(sf_grid)),
                "parametric_fitted_positive_f0_hz": fitted.reshape(-1),
            }
        )
        frame = frame.merge(fitted_long, on=["spatial_cpd", "temporal_hz"], validate="one_to_one")
        frame["parametric_residual_f0_hz"] = (
            frame["observed_positive_f0_hz"] - frame["parametric_fitted_positive_f0_hz"]
        )
        centered = np.square(
            frame["observed_positive_f0_hz"] - frame["observed_positive_f0_hz"].mean()
        ).sum()
        reproduced_r2 = 1.0 - np.square(frame["parametric_residual_f0_hz"]).sum() / centered
        if not np.isclose(reproduced_r2, model.joint_parametric_surface_r2, atol=1e-10):
            raise AssertionError(f"Unit {unit}: joint R2 reproduction failed")
        frame["selection_role"] = selection.selection_role
        frame["reported_joint_parametric_surface_r2"] = model.joint_parametric_surface_r2
        frame["reproduced_joint_parametric_surface_r2"] = reproduced_r2
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_profile_rows(selected: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(CP11 / "predictor_variant_leave_one_image_out_predictions.csv")
    variants = {
        "primary_unit_specific_amplitude": "unit_sftf",
        "total_power_amplitude_no_unit_tuning": "total_power",
    }
    predictions = predictions.loc[
        predictions["predictor_variant"].isin(variants)
        & predictions["rr100_index"].isin(selected["rr100_index"])
    ].copy()
    predictions["short_variant"] = predictions["predictor_variant"].map(variants)
    keys = ["rr100_index", "held_out_image_index", "observed_modulation_sd_hz"]
    profile = predictions.pivot_table(
        index=keys,
        columns="short_variant",
        values="held_out_predicted_modulation_sd_hz",
        aggfunc="first",
    ).reset_index()
    baseline = predictions.loc[
        predictions["short_variant"].eq("unit_sftf"),
        keys + ["held_out_intercept_only_prediction_hz"],
    ]
    profile = profile.merge(baseline, on=keys, validate="one_to_one")
    profile = profile.merge(
        selected[
            [
                "rr100_index",
                "selection_role",
                "joint_parametric_surface_r2",
                "primary_unit_specific_amplitude",
                "total_power_amplitude_no_unit_tuning",
                "delta_cv_r2_unit_minus_total",
                "response_modulation_sd_across_images_hz",
            ]
        ],
        on="rr100_index",
        validate="many_to_one",
    )
    return profile.sort_values(["rr100_index", "held_out_image_index"])


def matrix(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot(index="temporal_hz", columns="spatial_cpd", values=value).sort_index().sort_index(axis=1)
    return pivot.to_numpy(), pivot.columns.to_numpy(float), pivot.index.to_numpy(float)


def make_figure(selected: pd.DataFrame, surfaces: pd.DataFrame, profiles: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4, "axes.titlesize": 9.5})
    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16.5, 17.0),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 2.1]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.045, top=0.85, hspace=1.12, wspace=0.28)

    role_titles = {
        "best_joint_fit_control": "Best grating fit; neither predictor succeeds",
        "high_fit_positive_increment": "High fit; clearest credible tuning increment",
        "high_fit_both_predictors_succeed": "High fit; both predictors succeed similarly",
        "high_fit_negative_increment": "High fit; tuning-weighting destroys prediction",
        "largest_nominal_increment_control": "Largest nominal increment; both predictions weak",
    }

    for row_index, selection in enumerate(selected.itertuples(index=False)):
        unit = int(selection.rr100_index)
        surf = surfaces.loc[surfaces["rr100_index"].eq(unit)]
        profile = profiles.loc[profiles["rr100_index"].eq(unit)].sort_values("held_out_image_index")
        observed, sf_grid, tf_grid = matrix(surf, "observed_positive_f0_hz")
        fitted, _, _ = matrix(surf, "parametric_fitted_positive_f0_hz")
        residual, _, _ = matrix(surf, "parametric_residual_f0_hz")
        extent = [np.log2(sf_grid.min()), np.log2(sf_grid.max()), np.log2(tf_grid.min()), np.log2(tf_grid.max())]
        vmax = max(float(observed.max()), float(fitted.max()), 1e-9)
        rlim = max(float(np.abs(residual).max()), 1e-9)

        for column, values, title in [
            (0, observed, "Measured grating F0"),
            (1, fitted, "Parametric fit"),
        ]:
            ax = axes[row_index, column]
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            ax.set_title(f"{title}\n0–{vmax:.2g} Hz")
            ax.set_xticks(np.log2([1, 2, 4, 8]), ["1", "2", "4", "8"])
            ax.set_yticks(np.log2([0.5, 1, 2, 4, 8, 16, 32]), [".5", "1", "2", "4", "8", "16", "32"])
            if column == 0:
                ax.set_ylabel("TF (Hz)")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("SF (cpd)")

        ax = axes[row_index, 2]
        ax.imshow(
            residual,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-rlim,
            vmax=rlim,
        )
        ax.set_title(f"Measured − fit\n±{rlim:.2g} Hz")
        ax.set_xticks(np.log2([1, 2, 4, 8]), ["1", "2", "4", "8"])
        ax.set_yticks(np.log2([0.5, 1, 2, 4, 8, 16, 32]))
        ax.set_yticklabels([])
        ax.set_xlabel("SF (cpd)")

        ax = axes[row_index, 3]
        x = profile["held_out_image_index"].to_numpy(int)
        ax.plot(x, profile["observed_modulation_sd_hz"], "o-", color="#172029", lw=1.7, ms=4.2, label="observed")
        ax.plot(x, profile["total_power"], "o-", color="#d65238", lw=1.25, ms=3.3, label="total power")
        ax.plot(x, profile["unit_sftf"], "o-", color="#356fa3", lw=1.25, ms=3.3, label="unit SF×TF")
        ax.plot(
            x,
            profile["held_out_intercept_only_prediction_hz"],
            color="#aeb5ba",
            lw=1,
            linestyle=":",
            label="training mean",
        )
        ax.axhline(0, color="#c7ccd0", lw=0.8)
        ax.axvline(6, color="#d7b000", lw=1, linestyle="--", alpha=0.8)
        ax.set_xticks(range(16))
        ax.set_xlabel("Held-out image–trajectory pair (pair 6 highlighted)")
        ax.set_ylabel("FEM-effect temporal SD (Hz)")
        ax.grid(color="#e9ecef", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(
            f"Held-out natural-image predictions | unit R²={selection.primary_unit_specific_amplitude:.2f}, "
            f"total R²={selection.total_power_amplitude_no_unit_tuning:.2f}"
        )
        if row_index == 0:
            ax.legend(ncol=4, frameon=False, fontsize=7.8, loc="upper left")

        axes[row_index, 0].text(
            -0.42,
            1.34,
            f"RR100 {unit}  ·  {role_titles[selection.selection_role]}\n"
            f"joint fit R²={selection.joint_parametric_surface_r2:.2f}; "
            f"Δ prediction R²={selection.delta_cv_r2_unit_minus_total:+.2f}; "
            f"response SD={selection.response_modulation_sd_across_images_hz:.4f} Hz",
            transform=axes[row_index, 0].transAxes,
            fontsize=10.2,
            weight="bold",
            va="bottom",
        )

    fig.suptitle(
        "Checkpoint 15B: good grating fits do not guarantee the right natural-image condition ordering",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "Each row uses the unit's preferred-orientation fixed-eye grating surface; natural-image prediction remains orientation-collapsed. Map scales are shared within a row, not across units.",
        ha="center",
        fontsize=9.5,
        color="#505960",
    )
    fig.savefig(OUT / "checkpoint_15b_selected_unit_grating_maps_and_heldout_profiles.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "checkpoint_15b_selected_unit_grating_maps_and_heldout_profiles.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(CP15A / "auditable_fit_quality_example_selection.csv")
    models = pd.read_csv(MODELS).set_index("rr100_index")
    points = pd.read_csv(SURFACE)
    surfaces = build_surface_rows(selected, models, points)
    profiles = build_profile_rows(selected)
    surfaces.to_csv(OUT / "selected_unit_measured_parametric_grating_surfaces.csv", index=False)
    profiles.to_csv(OUT / "selected_unit_heldout_natural_image_profiles.csv", index=False)
    selected.to_csv(OUT / "selected_units_from_checkpoint_15a.csv", index=False)
    make_figure(selected, surfaces, profiles)

    manifest = {
        "analysis": "rr100_fit_quality_selected_unit_drilldown_checkpoint_15b",
        "status": "selected_unit_map_first_checkpoint_stop_before_population_threshold_models",
        "n_selected_units": int(len(selected)),
        "grating_surface": "positive F0 at each unit's preferred orientation; parametric map sample-normalized to reproduce reported joint fit R2",
        "natural_image_outcome": "temporal SD of frozen RR100 FEM-minus-zero response",
        "prediction_contract": "leave one complete image-trajectory pair out; nonnegative per-unit calibration on remaining 15",
        "orientation_contract": "grating map is preferred-orientation; natural-image SFxTF predictor is orientation-collapsed",
        "source_files": [
            str((CP15A / "auditable_fit_quality_example_selection.csv").relative_to(ROOT)),
            str((CP11 / "predictor_variant_leave_one_image_out_predictions.csv").relative_to(ROOT)),
            str(MODELS.relative_to(ROOT)),
            str(SURFACE.relative_to(ROOT)),
        ],
        "next_checkpoint": "quality thresholds and continuous/covariate models with stable-error sensitivity",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Checkpoint 15B: selected-unit drill-down\n\n"
        "Five units selected algorithmically in checkpoint 15A are shown with measured and "
        "parametric fixed-eye grating F0 surfaces, direct residuals, and complete leave-one-pair-out "
        "natural-image prediction profiles. This checkpoint stops before population threshold or "
        "covariate analysis.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

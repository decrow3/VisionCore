#!/usr/bin/env python3
"""Checkpoint 09: RR100 population validation of the spectral-overlap proxy.

Uses the preregistered six-image/four-trajectory checkpoint-08 cache. The
primary endpoint is within-image trajectory-direction agreement, evaluated in
a predeclared high-quality SF/TF fit cohort. Absolute response-scale prediction
is tested separately with a gain-scaled square-root power-overlap proxy.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from declan.fig4_active_sensing.run_rr100_four_orientation_proxy_response_checkpoint import file_identity


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_multiimage_trajectory_generalization_checkpoint_08_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_population_proxy_validation_checkpoint_09_v1"
SF_R2_MIN = 0.70
TF_R2_MIN = 0.70
JOINT_R2_MIN = 0.50
N_BOOTSTRAP = 5000
RNG_SEED = 20260812


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def orientation_selectivity(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = scores.groupby("rr100_index")["mean_positive_f0_hz"]
    table = grouped.agg(orientation_response_min_hz="min", orientation_response_max_hz="max").reset_index()
    table["orientation_modulation_index"] = (
        (table["orientation_response_max_hz"] - table["orientation_response_min_hz"])
        / np.maximum(table["orientation_response_max_hz"] + table["orientation_response_min_hz"], 1e-30)
    )
    return table


def tertile(values: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    valid = values.notna()
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    result.loc[valid] = pd.qcut(values.loc[valid].rank(method="first"), 3, labels=labels).astype(str)
    return result


def build_unit_summary(values: pd.DataFrame, agreement: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    unit_rows = []
    for unit, frame in values.groupby("rr100_index"):
        frame = frame.copy()
        direction_rho = spearmanr(
            frame["predicted_fraction_of_image_unit_max"], frame["observed_fraction_of_image_unit_max"]
        ).statistic
        amplitude_rho = spearmanr(
            frame["predicted_response_sd_proxy_arbitrary"], frame["fem_delta_temporal_sd_hz"]
        ).statistic
        image_agreement = agreement.loc[agreement["rr100_index"].eq(int(unit))]
        unit_rows.append({
            "rr100_index": int(unit),
            "all_24_direction_shape_spearman_rho": float(direction_rho),
            "all_24_amplitude_proxy_spearman_rho": float(amplitude_rho),
            "median_within_image_spearman_rho": float(image_agreement["four_point_spearman_rho"].median()),
            "mean_within_image_spearman_rho": float(image_agreement["four_point_spearman_rho"].mean()),
            "fraction_images_positive_rho": float(np.mean(image_agreement["four_point_spearman_rho"] > 0)),
            "fraction_images_exact_peak": float(np.mean(image_agreement["peak_trajectory_rotation_axial_error_deg"] == 0)),
            "fraction_images_peak_within_45deg": float(np.mean(image_agreement["peak_trajectory_rotation_axial_error_deg"] <= 45)),
            "median_peak_error_deg": float(image_agreement["peak_trajectory_rotation_axial_error_deg"].median()),
            "median_modulation_sd_hz": float(frame["fem_delta_temporal_sd_hz"].median()),
            "maximum_modulation_sd_hz": float(frame["fem_delta_temporal_sd_hz"].max()),
            "median_within_image_peak_to_trough_ratio": float(image_agreement["observed_peak_to_trough_ratio"].median()),
        })
    summary = pd.DataFrame(unit_rows).merge(models, on="rr100_index", how="left", validate="one_to_one")
    summary["minimum_fit_r2"] = summary[["sf_fit_r2", "tf_fit_r2", "joint_parametric_surface_r2"]].min(axis=1)
    summary["passes_primary_fit_quality"] = (
        summary["model_valid"].fillna(False).astype(bool)
        & summary["sf_fit_r2"].ge(SF_R2_MIN)
        & summary["tf_fit_r2"].ge(TF_R2_MIN)
        & summary["joint_parametric_surface_r2"].ge(JOINT_R2_MIN)
    )
    cohort = summary["passes_primary_fit_quality"]
    summary.loc[cohort, "sf_tertile"] = tertile(
        summary.loc[cohort, "preferred_sf_cpd"], ("low SF", "middle SF", "high SF")
    )
    summary.loc[cohort, "tf_tertile"] = tertile(
        summary.loc[cohort, "preferred_tf_hz"], ("low TF", "middle TF", "high TF")
    )
    summary.loc[cohort, "orientation_selectivity_tertile"] = tertile(
        summary.loc[cohort, "orientation_modulation_index"], ("low OSI", "middle OSI", "high OSI")
    )
    summary["preferred_orientation_group"] = summary["preferred_orientation_deg"].map(lambda x: f"{x:.0f}°" if pd.notna(x) else pd.NA)
    return summary


def bootstrap_summary(unit_summary: pd.DataFrame) -> pd.DataFrame:
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]].copy()
    endpoints = {
        "unit_median_within_image_spearman_rho": cohort["median_within_image_spearman_rho"].to_numpy(float),
        "unit_all24_direction_spearman_rho": cohort["all_24_direction_shape_spearman_rho"].to_numpy(float),
        "unit_fraction_images_exact_peak": cohort["fraction_images_exact_peak"].to_numpy(float),
        "unit_fraction_images_peak_within_45deg": cohort["fraction_images_peak_within_45deg"].to_numpy(float),
    }
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for name, data in endpoints.items():
        boot = np.empty(N_BOOTSTRAP)
        for index in range(N_BOOTSTRAP):
            sample = data[rng.integers(0, len(data), size=len(data))]
            boot[index] = np.median(sample) if "spearman" in name else np.mean(sample)
        point = np.median(data) if "spearman" in name else np.mean(data)
        rows.append({
            "endpoint": name, "n_units": len(data), "point_estimate": float(point),
            "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
            "bootstrap_resampling_unit": "RR100 unit", "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
        })
    return pd.DataFrame(rows)


def group_summary(unit_summary: pd.DataFrame) -> pd.DataFrame:
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]].copy()
    rows = []
    for variable in ("sf_tertile", "tf_tertile", "orientation_selectivity_tertile", "preferred_orientation_group"):
        for group, frame in cohort.groupby(variable, dropna=False):
            rows.append({
                "grouping_variable": variable, "group": str(group), "n_units": len(frame),
                "median_within_image_spearman_rho": float(frame["median_within_image_spearman_rho"].median()),
                "mean_within_image_spearman_rho": float(frame["median_within_image_spearman_rho"].mean()),
                "median_all24_direction_spearman_rho": float(frame["all_24_direction_shape_spearman_rho"].median()),
                "mean_fraction_images_exact_peak": float(frame["fraction_images_exact_peak"].mean()),
                "median_modulation_sd_hz": float(frame["median_modulation_sd_hz"].median()),
            })
    return pd.DataFrame(rows)


def select_examples(unit_summary: pd.DataFrame) -> pd.DataFrame:
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]].copy()
    chosen = []

    def take(role: str, frame: pd.DataFrame, column: str, ascending: bool) -> None:
        available = frame.loc[~frame["rr100_index"].isin([row["rr100_index"] for row in chosen])]
        row = available.sort_values([column, "rr100_index"], ascending=[ascending, True]).iloc[0]
        chosen.append({
            "selection_role": role, "rr100_index": int(row["rr100_index"]),
            "criterion_name": column, "criterion_value": float(row[column]),
            "median_within_image_spearman_rho": float(row["median_within_image_spearman_rho"]),
            "all_24_direction_shape_spearman_rho": float(row["all_24_direction_shape_spearman_rho"]),
            "median_modulation_sd_hz": float(row["median_modulation_sd_hz"]),
            "preferred_sf_cpd": float(row["preferred_sf_cpd"]), "preferred_tf_hz": float(row["preferred_tf_hz"]),
            "joint_parametric_surface_r2": float(row["joint_parametric_surface_r2"]),
        })

    take("best proxy generalizer", cohort, "median_within_image_spearman_rho", False)
    take("worst proxy generalizer", cohort, "median_within_image_spearman_rho", True)
    failure = cohort.loc[cohort["median_within_image_spearman_rho"] <= 0]
    take("strong-response proxy failure", failure, "maximum_modulation_sd_hz", False)
    take("weak-response quality control", cohort, "median_modulation_sd_hz", True)
    return pd.DataFrame(chosen)


def plot_population(unit_summary: pd.DataFrame, agreement: pd.DataFrame, values: pd.DataFrame,
                    bootstrap: pd.DataFrame, out_base: Path, dpi: int) -> None:
    cohort_units = unit_summary.loc[unit_summary["passes_primary_fit_quality"], "rr100_index"]
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]]
    pair = agreement.loc[agreement["rr100_index"].isin(cohort_units)]
    cohort_values = values.loc[values["rr100_index"].isin(cohort_units)]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2), constrained_layout=True)
    axes[0, 0].hist(pair["four_point_spearman_rho"], bins=np.linspace(-1.05, 1.05, 12), color="#4C78A8", edgecolor="white")
    axes[0, 0].axvline(pair["four_point_spearman_rho"].median(), color="#B2182B", lw=1.5)
    axes[0, 0].set(title=f"A  Within-image direction agreement\n{len(pair)} unit×image tests; median={pair['four_point_spearman_rho'].median():+.2f}",
                   xlabel="four-angle Spearman ρ", ylabel="unit×image count")
    axes[0, 1].hist(cohort["median_within_image_spearman_rho"], bins=np.linspace(-1.05, 1.05, 16), color="#72B7B2", edgecolor="white")
    axes[0, 1].axvline(cohort["median_within_image_spearman_rho"].median(), color="#B2182B", lw=1.5)
    ci = bootstrap.set_index("endpoint").loc["unit_median_within_image_spearman_rho"]
    axes[0, 1].set(title=f"B  Unit-level generalization across six images\nmedian={ci.point_estimate:+.2f} [{ci.bootstrap_ci95_low:+.2f}, {ci.bootstrap_ci95_high:+.2f}]",
                   xlabel="unit median within-image ρ", ylabel="RR100 units")
    errors = [0, 45, 90]
    counts = [(pair["peak_trajectory_rotation_axial_error_deg"] == value).mean() for value in errors]
    axes[0, 2].bar(range(3), counts, color=["#009E73", "#E69F00", "#D55E00"])
    axes[0, 2].set_xticks(range(3), ["exact", "45° off", "90° off"])
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set(title="C  Predicted versus observed peak eye direction", ylabel="fraction of unit×image tests")

    sample = cohort_values.copy()
    axes[1, 0].scatter(
        sample["predicted_fraction_of_image_unit_max"], sample["observed_fraction_of_image_unit_max"],
        s=6, alpha=0.12, color="#7B3294", rasterized=True,
    )
    rho_global = spearmanr(sample["predicted_fraction_of_image_unit_max"], sample["observed_fraction_of_image_unit_max"]).statistic
    axes[1, 0].plot([0, 1], [0, 1], color="0.7", ls="--", lw=0.8)
    axes[1, 0].set(title=f"D  All normalized direction conditions\nSpearman ρ={rho_global:+.2f}; n={len(sample)}",
                   xlabel="predicted fraction of image×unit maximum", ylabel="observed fraction of image×unit maximum")
    positive = (sample["predicted_response_sd_proxy_arbitrary"] > 0) & (sample["fem_delta_temporal_sd_hz"] > 0)
    axes[1, 1].scatter(
        sample.loc[positive, "predicted_response_sd_proxy_arbitrary"], sample.loc[positive, "fem_delta_temporal_sd_hz"],
        s=6, alpha=0.12, color="#008837", rasterized=True,
    )
    axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
    rho_amp = spearmanr(sample.loc[positive, "predicted_response_sd_proxy_arbitrary"], sample.loc[positive, "fem_delta_temporal_sd_hz"]).statistic
    axes[1, 1].set(title=f"E  Absolute modulation-scale proxy\nSpearman ρ={rho_amp:+.2f}",
                   xlabel="gain × √(raw overlap), arbitrary", ylabel="measured modulation SD (Hz)")
    axes[1, 2].scatter(cohort["minimum_fit_r2"], cohort["median_within_image_spearman_rho"], s=25, alpha=0.7, color="#4C78A8")
    rho_fit = spearmanr(cohort["minimum_fit_r2"], cohort["median_within_image_spearman_rho"]).statistic
    axes[1, 2].axhline(0, color="0.6", lw=0.8)
    axes[1, 2].set(title=f"F  Better fits do not guarantee prediction\nSpearman ρ={rho_fit:+.2f}",
                   xlabel="minimum of SF, TF, and joint fit R²", ylabel="unit median within-image ρ")
    for axis in axes.ravel(): axis.grid(color="0.93", lw=0.6)
    fig.suptitle(
        f"Checkpoint 09: population validation in {len(cohort)} high-quality RR100 tuning fits\n"
        "Six preregistered images × four eye directions; frozen-model responses",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_groups(unit_summary: pd.DataFrame, out_base: Path, dpi: int) -> None:
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]].copy()
    specifications = [
        ("sf_tertile", ["low SF", "middle SF", "high SF"], "preferred SF"),
        ("tf_tertile", ["low TF", "middle TF", "high TF"], "preferred TF"),
        ("orientation_selectivity_tertile", ["low OSI", "middle OSI", "high OSI"], "orientation selectivity"),
        ("preferred_orientation_group", ["0°", "45°", "90°", "135°"], "preferred orientation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    for axis, (column, order, label) in zip(axes.ravel(), specifications, strict=True):
        data = [cohort.loc[cohort[column].eq(group), "median_within_image_spearman_rho"].dropna() for group in order]
        axis.boxplot(data, tick_labels=order, showfliers=False, medianprops={"color": "#B2182B", "lw": 1.5})
        for position, values in enumerate(data, start=1):
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else np.asarray([])
            axis.scatter(position + jitter, values, s=12, alpha=0.5, color="#4C78A8")
        axis.axhline(0, color="0.6", lw=0.8)
        axis.set_ylim(-1.05, 1.05)
        axis.set(title=f"Trajectory-proxy agreement by {label}", ylabel="unit median within-image Spearman ρ")
        axis.tick_params(axis="x", rotation=15); axis.grid(axis="y", color="0.93")
    fig.suptitle("Checkpoint 09 tuning-stratified population diagnostics", fontsize=14)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_population_examples(selection: pd.DataFrame, values: pd.DataFrame, out_base: Path, dpi: int) -> None:
    images = sorted(values["image_index"].unique())
    fig, axes = plt.subplots(len(selection), 3, figsize=(13.5, 12.0), constrained_layout=True)
    for row, selected in selection.reset_index(drop=True).iterrows():
        unit = int(selected["rr100_index"])
        frame = values.loc[values["rr100_index"].eq(unit)]
        predicted = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="predicted_fraction_of_image_unit_max").reindex(images)
        observed = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="observed_fraction_of_image_unit_max").reindex(images)
        axes[row, 0].imshow(predicted, vmin=0, vmax=1, aspect="auto", cmap="magma")
        axes[row, 1].imshow(observed, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axes[row, 0].set_title(f"{selected['selection_role']} · RR100 {unit}\npredicted direction shape", fontsize=9.5)
        axes[row, 1].set_title("measured direction shape", fontsize=9.5)
        for column in (0, 1):
            axes[row, column].set_xticks(range(4), ["0°", "45°", "90°", "135°"])
            axes[row, column].set_yticks(range(len(images)), images)
            axes[row, column].set(xlabel="eye direction; image fixed", ylabel="image index")
        axes[row, 2].scatter(frame["predicted_fraction_of_image_unit_max"], frame["observed_fraction_of_image_unit_max"], s=28, alpha=0.7)
        axes[row, 2].plot([0, 1], [0, 1], color="0.7", ls="--", lw=0.8)
        axes[row, 2].set_xlim(0, 1.05); axes[row, 2].set_ylim(0, 1.05); axes[row, 2].grid(color="0.93")
        axes[row, 2].set(title=f"all 24 points\nunit median within-image ρ={selected['median_within_image_spearman_rho']:+.2f}",
                         xlabel="predicted fraction", ylabel="observed fraction")
    fig.suptitle("Auditable population examples: success, failure, dissociation, and weak control", fontsize=14)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    values_path = args.source_dir / "multiimage_trajectory_proxy_response_values_all_rr100.csv"
    agreement_path = args.source_dir / "multiimage_unit_image_four_angle_agreement_all_rr100.csv"
    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    orientation_path = args.f0_dir / "f0_orientation_scores.csv"
    values = pd.read_csv(values_path)
    agreement = pd.read_csv(agreement_path)
    models = pd.read_csv(models_path)
    osi = orientation_selectivity(pd.read_csv(orientation_path))
    models = models.merge(osi, on="rr100_index", how="left", validate="one_to_one")
    values = values.merge(
        models[["rr100_index", "joint_rank1_gain_f0_hz"]], on="rr100_index", validate="many_to_one"
    )
    values["predicted_response_sd_proxy_arbitrary"] = (
        values["joint_rank1_gain_f0_hz"] * np.sqrt(np.maximum(values["predicted_overlap_raw_arbitrary"], 0.0))
    )
    unit_summary = build_unit_summary(values, agreement, models)
    bootstrap = bootstrap_summary(unit_summary)
    groups = group_summary(unit_summary)
    selection = select_examples(unit_summary)
    unit_summary.to_csv(args.out_dir / "rr100_population_proxy_validation_by_unit.csv", index=False)
    bootstrap.to_csv(args.out_dir / "rr100_population_bootstrap_summary.csv", index=False)
    groups.to_csv(args.out_dir / "rr100_population_tuning_group_summary.csv", index=False)
    selection.to_csv(args.out_dir / "population_example_selection.csv", index=False)
    values.to_csv(args.out_dir / "rr100_population_proxy_response_values_with_amplitude_proxy.csv", index=False)
    plot_population(unit_summary, agreement, values, bootstrap, args.out_dir / "checkpoint_09_population_validation", args.dpi)
    plot_groups(unit_summary, args.out_dir / "checkpoint_09_tuning_stratified_diagnostics", args.dpi)
    plot_population_examples(selection, values, args.out_dir / "checkpoint_09_auditable_population_examples", args.dpi)

    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]]
    pair = agreement.loc[agreement["rr100_index"].isin(cohort["rr100_index"])]
    global_direction_rho = float(spearmanr(
        values.loc[values["rr100_index"].isin(cohort["rr100_index"]), "predicted_fraction_of_image_unit_max"],
        values.loc[values["rr100_index"].isin(cohort["rr100_index"]), "observed_fraction_of_image_unit_max"],
    ).statistic)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "RR100 population validation of orientation-aware SFxTF power-overlap proxy",
        "status": "checkpoint_09_population_first_pass_complete",
        "primary_cohort": {
            "definition": f"model_valid and SF R2 >= {SF_R2_MIN}, TF R2 >= {TF_R2_MIN}, joint surface R2 >= {JOINT_R2_MIN}",
            "n_units": int(len(cohort)), "n_unit_image_tests": int(len(pair)),
        },
        "primary_results": {
            "median_unit_median_within_image_spearman_rho": float(cohort["median_within_image_spearman_rho"].median()),
            "mean_unit_median_within_image_spearman_rho": float(cohort["median_within_image_spearman_rho"].mean()),
            "unit_image_median_spearman_rho": float(pair["four_point_spearman_rho"].median()),
            "unit_image_fraction_exact_peak": float(np.mean(pair["peak_trajectory_rotation_axial_error_deg"] == 0)),
            "unit_image_fraction_peak_within_45deg": float(np.mean(pair["peak_trajectory_rotation_axial_error_deg"] <= 45)),
            "global_normalized_direction_spearman_rho": global_direction_rho,
        },
        "interpretation_guardrail": "four-angle correlations are descriptive and discrete; population evidence is summarized across preregistered images and quality-filtered units, not treated as independent per-point inference",
        "inputs": {
            "values": file_identity(values_path), "agreement": file_identity(agreement_path),
            "models": file_identity(models_path), "orientation_scores": file_identity(orientation_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 09: RR100 population proxy validation\n\n"
        "The primary cohort requires valid SF/TF parametric fits with SF and TF R² at least 0.70 and joint surface R² "
        "at least 0.50. Direction prediction is evaluated within each fixed image across four rotated eye trajectories. "
        "Absolute response scale is evaluated separately with gain times square-root raw overlap. Results are stratified "
        "by SF, TF, preferred orientation, and orientation selectivity. Four auditable examples were selected by explicit roles.\n"
    )
    print(json.dumps(manifest["primary_cohort"], indent=2)); print(json.dumps(manifest["primary_results"], indent=2))
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()

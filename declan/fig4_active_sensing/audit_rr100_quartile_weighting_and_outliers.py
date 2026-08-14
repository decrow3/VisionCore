#!/usr/bin/env python3
"""Audit the corrected-cache quartile estimand, unit gates, and outliers."""

from __future__ import annotations

import json
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
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_011_n012_quartile_snapshot_v1"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_rounds000_011_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_no_bottom_row_rounds000_011_v2_unit_assignments.csv"
)
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
FIT_POINTS = ROOT / (
    "outputs/redundancy_resolved_v1_twin/rr100_joint_f0_parametric_recorded_validation_v2/"
    "rr100_joint_f0_parametric_factor_points.csv"
)
RECORDED_POINTS = ROOT / (
    "outputs/redundancy_resolved_v1_twin/rr100_joint_f0_parametric_recorded_validation_v2/"
    "rr100_parametric_recorded_validation_curve_points.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_quartile_weighting_outlier_audit_v2"
GROUPS = ["sf_q1", "sf_q2", "sf_q3", "sf_q4"]
LABELS = {f"sf_q{i}": f"Q{i}" for i in range(1, 5)}
COLORS = {"sf_q1": "#087EBC", "sf_q2": "#009E73", "sf_q3": "#E69F00", "sf_q4": "#CC79A7"}


def residualize(v: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) - pd.Series(v).groupby(image_ids).transform("mean").to_numpy()


def slope(x: np.ndarray, y: np.ndarray, image_ids: np.ndarray) -> float:
    xx = residualize(x, image_ids)
    yy = residualize(y, image_ids)
    return float(np.dot(xx, yy) / np.dot(xx, xx))


def load_data() -> dict[str, object]:
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv").merge(
        pd.read_csv(COHORT / "corrected1000_traces.csv")[[
            "trace_index", "corrected_dpi_crop120_path_length_arcmin"
        ]], on="trace_index", validate="many_to_one"
    )
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    assignments = pd.read_csv(ASSIGNMENTS)
    moving_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")
    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_ssi = np.asarray(archive["movie_ssi_bits_per_spike"], float)
        baseline_spikes = np.asarray(archive["expected_spikes"], float)
    return locals()


def compute_estimands(data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition = data["condition"]
    images = data["images"]
    assignments = data["assignments"]
    moving_ssi = data["moving_ssi"]
    moving_info = data["moving_info"]
    moving_spikes = data["moving_spikes"]
    baseline_ssi = data["baseline_ssi"]
    baseline_spikes = data["baseline_spikes"]
    image_ids = condition.image_index.to_numpy(int)
    paths = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    strong = images.corrected_reconstruction_orientation_coherence.to_numpy(float)[image_ids] >= 0.20
    scopes = {"all_images": np.ones(len(condition), bool), "strong_contours": strong}
    summaries, units_out, decompositions = [], [], []
    for scope, use in scopes.items():
        ids = image_ids[use]
        x = paths[use]
        for group in GROUPS:
            unit_ids = assignments.loc[assignments.sf_quartile.eq(group), "rr100_index"].to_numpy(int)
            delta = np.asarray(moving_ssi[:, unit_ids], float) - baseline_ssi[image_ids][:, unit_ids]
            population = np.asarray(moving_info[:, unit_ids], float).sum(1) / np.maximum(
                np.asarray(moving_spikes[:, unit_ids], float).sum(1), 1e-12
            ) - (baseline_ssi * baseline_spikes)[:, unit_ids].sum(1)[image_ids] / np.maximum(
                baseline_spikes[:, unit_ids].sum(1)[image_ids], 1e-12
            )
            weights = np.asarray(moving_spikes[use][:, unit_ids], float).mean(0)
            weights += baseline_spikes[np.unique(ids)][:, unit_ids].mean(0)
            fixed = (delta * weights).sum(1) / weights.sum()
            unit_slopes = np.asarray([slope(x, delta[use, j], ids) for j in range(len(unit_ids))])
            estimands = {
                "pooled_spike": slope(x, population[use], ids),
                "fixed_spike_weight": slope(x, fixed[use], ids),
                "equal_unit_mean": slope(x, delta[use].mean(1), ids),
                "conditionwise_unit_median": slope(x, np.median(delta[use], axis=1), ids),
                "median_unit_slope": float(np.median(unit_slopes)),
            }
            for name, value in estimands.items():
                summaries.append({"scope": scope, "sf_quartile": group, "estimand": name, "path_slope": value})
            total = np.asarray(moving_spikes[:, unit_ids], float).sum(1)
            spike_share = np.asarray(moving_spikes[:, unit_ids], float) / np.maximum(total[:, None], 1e-12)
            mean_share = spike_share[use].mean(axis=0)
            for j, unit in enumerate(unit_ids):
                per_image = []
                for image in np.unique(ids):
                    take = use & (image_ids == image)
                    per_image.append(slope(paths[take], delta[take, j], image_ids[take]))
                share = spike_share[:, j]
                fixed_contribution = mean_share[j] * unit_slopes[j]
                composition_contribution = slope(
                    x, ((share[use] - mean_share[j]) * np.asarray(moving_ssi[use, unit], float)), ids
                )
                units_out.append({
                    "scope": scope, "sf_quartile": group, "rr100_index": int(unit),
                    "unit_path_slope": unit_slopes[j],
                    "fraction_image_slopes_negative": float(np.mean(np.asarray(per_image) < 0)),
                    "median_image_slope": float(np.median(per_image)),
                    "mean_expected_spikes": float(np.asarray(moving_spikes[use, unit], float).mean()),
                    "mean_population_spike_share": float(share[use].mean()),
                    "spike_share_path_slope": slope(x, share[use], ids),
                    "moving_spike_path_slope": slope(x, np.asarray(moving_spikes[use, unit], float), ids),
                })
                decompositions.append({
                    "scope": scope, "sf_quartile": group, "rr100_index": int(unit),
                    "mean_spike_share": float(mean_share[j]),
                    "mean_moving_ssi_bits_per_spike": float(np.asarray(moving_ssi[use, unit], float).mean()),
                    "unit_ssi_slope": unit_slopes[j],
                    "fixed_share_within_unit_contribution": fixed_contribution,
                    "spike_composition_contribution": composition_contribution,
                    "total_population_term_slope": fixed_contribution + composition_contribution,
                })
    return pd.DataFrame(summaries), pd.DataFrame(units_out), pd.DataFrame(decompositions)


def gates(assignments: pd.DataFrame, models: pd.DataFrame, estimands: pd.DataFrame, data: dict[str, object]) -> pd.DataFrame:
    base = assignments[assignments.sf_quartile.isin(GROUPS)][["rr100_index", "sf_quartile"]].merge(
        models, on="rr100_index", validate="one_to_one"
    )
    definitions = {
        "current_recorded_sf_gate": np.ones(len(base), bool),
        "plus_ccnorm_ge_0p5": base.ccnorm.ge(0.5).to_numpy(),
        "plus_joint_r2_ge_0p5": base.joint_parametric_surface_r2.ge(0.5).to_numpy(),
        "plus_tf_r2_ge_0p7": base.tf_fit_r2.ge(0.7).to_numpy(),
        "strict_joint0p5_tf0p7": (base.joint_parametric_surface_r2.ge(0.5) & base.tf_fit_r2.ge(0.7)).to_numpy(),
    }
    condition = data["condition"]
    images = data["images"]
    moving_ssi = data["moving_ssi"]
    moving_info = data["moving_info"]
    moving_spikes = data["moving_spikes"]
    baseline_ssi = data["baseline_ssi"]
    baseline_spikes = data["baseline_spikes"]
    iid = condition.image_index.to_numpy(int)
    path = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    use = images.corrected_reconstruction_orientation_coherence.to_numpy(float)[iid] >= 0.20
    rows = []
    for gate_name, gate in definitions.items():
        eligible = base.loc[gate]
        for group in GROUPS:
            unit_ids = eligible.loc[eligible.sf_quartile.eq(group), "rr100_index"].to_numpy(int)
            if len(unit_ids) < 2:
                continue
            delta = np.asarray(moving_ssi[:, unit_ids], float) - baseline_ssi[iid][:, unit_ids]
            pop = np.asarray(moving_info[:, unit_ids], float).sum(1) / np.maximum(
                np.asarray(moving_spikes[:, unit_ids], float).sum(1), 1e-12
            ) - (baseline_ssi * baseline_spikes)[:, unit_ids].sum(1)[iid] / np.maximum(
                baseline_spikes[:, unit_ids].sum(1)[iid], 1e-12
            )
            rows.append({"gate": gate_name, "sf_quartile": group, "n_units": len(unit_ids),
                         "pooled_spike_path_slope": slope(path[use], pop[use], iid[use]),
                         "equal_unit_path_slope": slope(path[use], delta[use].mean(1), iid[use]),
                         "excluded_units": ";".join(map(str, sorted(set(base.loc[base.sf_quartile.eq(group), "rr100_index"]) - set(unit_ids))))})
    return pd.DataFrame(rows)


def make_figure(estimands: pd.DataFrame, unit_diag: pd.DataFrame, models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), constrained_layout=True)
    methods = ["pooled_spike", "fixed_spike_weight", "equal_unit_mean", "conditionwise_unit_median", "median_unit_slope"]
    names = ["pooled\nspike", "fixed\nweights", "equal-unit\nmean", "conditionwise\nmedian", "median\nunit slope"]
    for ax, scope, title in zip(axes[0], ["all_images", "strong_contours"], ["A  All images", "B  Strong contours"]):
        for group in GROUPS:
            sub = estimands[(estimands.scope == scope) & (estimands.sf_quartile == group)].set_index("estimand")
            ax.plot(range(len(methods)), sub.loc[methods, "path_slope"] * 1e4, marker="o", color=COLORS[group], label=LABELS[group])
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_xticks(range(len(methods)), names)
        ax.set_ylabel("path slope (×10⁻⁴ bits/spike/arcmin)")
        ax.set_title(title, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=4)
    strong = unit_diag[unit_diag.scope.eq("strong_contours")]
    for ax, unit, title in zip(axes[1], [54, 18], ["C  u054 · Q3", "D  u018 · Q4"]):
        q = strong.loc[strong.rr100_index.eq(unit), "sf_quartile"].iloc[0]
        sub = strong[strong.sf_quartile.eq(q)]
        ax.scatter(sub.mean_population_spike_share * 100, sub.unit_path_slope * 1e4, color=COLORS[q], alpha=0.75)
        row = sub[sub.rr100_index.eq(unit)].iloc[0]
        ax.scatter(row.mean_population_spike_share * 100, row.unit_path_slope * 1e4, s=95, facecolor="white", edgecolor="black", lw=1.8)
        ax.annotate(f"u{unit:03d}\nnegative in {row.fraction_image_slopes_negative:.0%} of images", (row.mean_population_spike_share * 100, row.unit_path_slope * 1e4), xytext=(8, 7), textcoords="offset points")
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_xlabel("mean share of quartile expected spikes (%)")
        ax.set_ylabel("unit path slope (×10⁻⁴ bits/spike/arcmin)")
        ax.set_title(title, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Quartile SSI audit: pooled-spike and typical-unit estimands diverge", fontsize=15, weight="bold")
    fig.savefig(OUT / "quartile_weighting_and_outlier_audit.png", dpi=210, facecolor="white")
    fig.savefig(OUT / "quartile_weighting_and_outlier_audit.pdf", facecolor="white")
    plt.close(fig)


def make_tuning_figure(models: pd.DataFrame) -> None:
    factor = pd.read_csv(FIT_POINTS)
    recorded = pd.read_csv(RECORDED_POINTS)
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.7), constrained_layout=True)
    for col, unit in enumerate([54, 18]):
        row = models[models.rr100_index.eq(unit)].iloc[0]
        for r, axis_name, xlabel in [(0, "spatial_frequency", "spatial frequency (cpd)"), (1, "temporal_frequency", "temporal frequency (Hz)")]:
            ax = axes[r, col]
            sub = factor[(factor.rr100_index == unit) & (factor.axis == axis_name)]
            ax.plot(sub.frequency, sub.parametric_prediction, color="black", lw=1.8, label="parametric fit")
            ax.scatter(sub.frequency, sub.observed_normalized_factor, color="#356fa3", s=28, label="twin factor samples")
            if r == 0:
                rec = recorded[recorded.rr100_index.eq(unit)]
                ax.scatter(rec.sf_cpd, rec.recorded_range_normalized, marker="s", facecolor="#D55E00", edgecolor="white", s=42, label="recorded SF curve")
            ax.set_xscale("log", base=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("normalized response")
            ax.spines[["top", "right"]].set_visible(False)
            quality = f"SF R²={row.sf_fit_r2:.2f}; recorded r={row.recorded_sf_curve_r_full_support:.2f}" if r == 0 else f"TF R²={row.tf_fit_r2:.2f}; joint R²={row.joint_parametric_surface_r2:.2f}"
            ax.set_title(f"{'AC'[r]}{col + 1}  u{unit:03d} · {quality}", loc="left", weight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("The SSI outliers do not have failed SF/TF parametric fits", fontsize=15, weight="bold")
    fig.savefig(OUT / "outlier_sf_tf_tuning_recheck.png", dpi=210, facecolor="white")
    fig.savefig(OUT / "outlier_sf_tf_tuning_recheck.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    data = load_data()
    estimands, unit_diag, decomposition = compute_estimands(data)
    models = pd.read_csv(MODELS)
    gate_table = gates(data["assignments"], models, estimands, data)
    estimands.to_csv(OUT / "quartile_estimand_comparison.csv", index=False)
    unit_diag.to_csv(OUT / "unit_outlier_diagnostics.csv", index=False)
    decomposition.to_csv(OUT / "pooled_spike_slope_unit_decomposition.csv", index=False)
    gate_table.to_csv(OUT / "candidate_gate_sensitivity.csv", index=False)
    models[models.rr100_index.isin([18, 54])].to_csv(OUT / "outlier_tuning_model_rows.csv", index=False)
    pd.read_csv(FIT_POINTS).query("rr100_index in [18, 54]").to_csv(OUT / "outlier_tuning_factor_points.csv", index=False)
    pd.read_csv(RECORDED_POINTS).query("rr100_index in [18, 54]").to_csv(OUT / "outlier_recorded_sf_points.csv", index=False)
    make_figure(estimands, unit_diag, models)
    make_tuning_figure(models)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "weighting_outlier_audit_complete",
        "scope": "12 complete corrected-cache rounds; interim",
        "interpretive_contract": {
            "pooled_spike": "information per expected spike after pooling units; weights vary with condition",
            "equal_unit": "mean change for an equally weighted validated unit",
            "exclusion_policy": "only predeclared quality/identity gates are admissible; outcome-based removal is diagnostic only",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

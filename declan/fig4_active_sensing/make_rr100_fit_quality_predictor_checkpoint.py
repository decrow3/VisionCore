#!/usr/bin/env python3
"""Map-first checkpoint for grating fit quality vs SFxTF predictor improvement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fit_quality_predictor_checkpoint_15a_v1"


def load_unit_table() -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "predictor_variant_per_unit_explainability.csv")
    source = source.loc[source["quality_cohort"].astype(bool)].copy()
    wide = source.pivot(
        index="rr100_index",
        columns="predictor_variant",
        values="cv_r2_vs_train_mean_baseline",
    )
    primary = source.loc[
        source["predictor_variant"].eq("primary_unit_specific_amplitude")
    ].set_index("rr100_index")
    metadata = [
        "model_valid",
        "quality_cohort",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "sf_fit_r2",
        "tf_fit_r2",
        "joint_parametric_surface_r2",
        "response_modulation_sd_across_images_hz",
        "oof_rmse_hz",
        "oof_pearson_r",
    ]
    for column in metadata:
        wide[column] = primary[column]
    wide = wide.reset_index()
    wide["delta_cv_r2_unit_minus_total"] = (
        wide["primary_unit_specific_amplitude"]
        - wide["total_power_amplitude_no_unit_tuning"]
    )
    wide["both_predictors_positive"] = (
        (wide["primary_unit_specific_amplitude"] > 0)
        & (wide["total_power_amplitude_no_unit_tuning"] > 0)
    )
    return wide


def select_examples(table: pd.DataFrame) -> pd.DataFrame:
    q75 = table["joint_parametric_surface_r2"].quantile(0.75)
    high = table.loc[table["joint_parametric_surface_r2"] >= q75]

    roles: list[tuple[str, str, int]] = []
    roles.append(
        (
            "best_joint_fit_control",
            "maximum joint separable-surface R2",
            int(table.loc[table["joint_parametric_surface_r2"].idxmax(), "rr100_index"]),
        )
    )
    credible_positive = high.loc[high["primary_unit_specific_amplitude"] > 0]
    roles.append(
        (
            "high_fit_positive_increment",
            "largest delta CV R2 among upper-quartile joint fits with positive unit-specific CV R2",
            int(
                credible_positive.loc[
                    credible_positive["delta_cv_r2_unit_minus_total"].idxmax(), "rr100_index"
                ]
            ),
        )
    )
    broad_success = high.loc[
        (high["primary_unit_specific_amplitude"] > 0.5)
        & (high["total_power_amplitude_no_unit_tuning"] > 0.5)
    ]
    roles.append(
        (
            "high_fit_both_predictors_succeed",
            "highest joint fit among units with both predictor CV R2 values above 0.5",
            int(
                broad_success.loc[
                    broad_success["joint_parametric_surface_r2"].idxmax(), "rr100_index"
                ]
            ),
        )
    )
    high_total_positive = high.loc[high["total_power_amplitude_no_unit_tuning"] > 0]
    roles.append(
        (
            "high_fit_negative_increment",
            "smallest delta CV R2 among upper-quartile joint fits with positive total-power CV R2",
            int(
                high_total_positive.loc[
                    high_total_positive["delta_cv_r2_unit_minus_total"].idxmin(), "rr100_index"
                ]
            ),
        )
    )
    roles.append(
        (
            "largest_nominal_increment_control",
            "maximum delta CV R2, retained to expose unstable improvement when the control is very poor",
            int(table.loc[table["delta_cv_r2_unit_minus_total"].idxmax(), "rr100_index"]),
        )
    )

    rows = []
    for role, criterion, unit in roles:
        row = table.loc[table["rr100_index"].eq(unit)].iloc[0].to_dict()
        row.update(
            {
                "selection_role": role,
                "selection_criterion": criterion,
                "selection_scope": "algorithmic selection from 66-unit quality cohort",
            }
        )
        rows.append(row)
    columns = [
        "selection_role",
        "rr100_index",
        "selection_criterion",
        "selection_scope",
        "joint_parametric_surface_r2",
        "sf_fit_r2",
        "tf_fit_r2",
        "primary_unit_specific_amplitude",
        "total_power_amplitude_no_unit_tuning",
        "delta_cv_r2_unit_minus_total",
        "response_modulation_sd_across_images_hz",
        "preferred_sf_cpd",
        "preferred_tf_hz",
    ]
    return pd.DataFrame(rows)[columns]


def annotate_selected(
    ax: plt.Axes,
    selected: pd.DataFrame,
    xcol: str,
    ycol: str,
    offsets: dict[int, tuple[int, int]],
) -> None:
    for _, row in selected.iterrows():
        unit = int(row.rr100_index)
        offset = offsets.get(unit, (6, 8))
        ax.annotate(
            f"u{unit}",
            (row[xcol], row[ycol]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            arrowprops=dict(arrowstyle="-", color="#454c52", linewidth=0.7),
        )


def make_figure(table: pd.DataFrame, selected: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.15), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.97, bottom=0.13, top=0.79, wspace=0.34)
    response_sd = table["response_modulation_sd_across_images_hz"].to_numpy()
    log_sd = np.log10(response_sd)
    cmap = "viridis"

    ax = axes[0]
    sc = ax.scatter(
        table["joint_parametric_surface_r2"],
        table["delta_cv_r2_unit_minus_total"],
        c=log_sd,
        cmap=cmap,
        s=46,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="#6a7075", linestyle="--", linewidth=1)
    ax.axvline(table["joint_parametric_surface_r2"].quantile(0.75), color="#b7bdc2", linestyle=":")
    annotate_selected(
        ax,
        selected,
        "joint_parametric_surface_r2",
        "delta_cv_r2_unit_minus_total",
        {62: (-34, -18), 49: (7, -17), 36: (-18, 12), 97: (-34, -12), 82: (7, 8)},
    )
    rho = stats.spearmanr(
        table["joint_parametric_surface_r2"], table["delta_cv_r2_unit_minus_total"]
    )
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ={rho.statistic:.2f}, p={rho.pvalue:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3),
    )
    ax.set(
        xlabel="Joint separable grating-fit $R^2$",
        ylabel=r"$\Delta R^2$ = unit SF×TF overlap − total power",
        title="A  Better grating fits do not visibly yield larger increments",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.02)
    cbar.set_label("log₁₀ response SD across conditions (Hz)")

    ax = axes[1]
    ax.scatter(
        response_sd,
        table["delta_cv_r2_unit_minus_total"],
        c=table["joint_parametric_surface_r2"],
        cmap="plasma",
        vmin=0.5,
        vmax=1.0,
        s=46,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.axhline(0, color="#6a7075", linestyle="--", linewidth=1)
    annotate_selected(
        ax,
        selected,
        "response_modulation_sd_across_images_hz",
        "delta_cv_r2_unit_minus_total",
        {62: (-26, -18), 49: (-22, 12), 36: (7, 10), 97: (-40, -18), 82: (7, 8)},
    )
    ax.set(
        xlabel="Response-effect SD across 16 conditions (Hz)",
        ylabel=r"$\Delta R^2$",
        title="B  Extreme increments occur mainly for low-variance responses",
    )

    ax = axes[2]
    sc2 = ax.scatter(
        table["total_power_amplitude_no_unit_tuning"],
        table["primary_unit_specific_amplitude"],
        c=table["joint_parametric_surface_r2"],
        cmap="plasma",
        vmin=0.5,
        vmax=1.0,
        s=46,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    lo = min(
        table["total_power_amplitude_no_unit_tuning"].min(),
        table["primary_unit_specific_amplitude"].min(),
    )
    hi = max(
        table["total_power_amplitude_no_unit_tuning"].max(),
        table["primary_unit_specific_amplitude"].max(),
    )
    ax.plot([lo, hi], [lo, hi], color="#6a7075", linestyle="--", linewidth=1)
    ax.axhline(0, color="#c4c9cd", linewidth=0.8)
    ax.axvline(0, color="#c4c9cd", linewidth=0.8)
    annotate_selected(
        ax,
        selected,
        "total_power_amplitude_no_unit_tuning",
        "primary_unit_specific_amplitude",
        {62: (7, -18), 49: (-32, 12), 36: (7, 8), 97: (7, -18), 82: (7, 8)},
    )
    ax.set(
        xlim=(lo - 0.08, hi + 0.08),
        ylim=(lo - 0.08, hi + 0.08),
        xlabel="Total-power held-out $R^2$",
        ylabel="Unit SF×TF held-out $R^2$",
        title="C  High-quality fits occur on both sides of equality",
    )
    cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.048, pad=0.02)
    cbar2.set_label("joint grating-fit $R^2$")

    for ax in axes:
        ax.grid(color="#e8ebed", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Checkpoint 15A: first look at whether grating-fit error suppresses unit-specific prediction",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    subtitle = (
        "Raw per-unit values; 66 RR100 units passing the pre-existing fit-quality gate. "
        "Labels are algorithmically selected roles saved in the companion table."
    )
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=10, color="#50585f")
    fig.savefig(OUT / "checkpoint_15a_fit_quality_vs_predictor_increment.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "checkpoint_15a_fit_quality_vs_predictor_increment.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_unit_table()
    selected = select_examples(table)
    table.to_csv(OUT / "unit_fit_quality_vs_predictor_improvement.csv", index=False)
    selected.to_csv(OUT / "auditable_fit_quality_example_selection.csv", index=False)
    make_figure(table, selected)

    rho = stats.spearmanr(
        table["joint_parametric_surface_r2"], table["delta_cv_r2_unit_minus_total"]
    )
    summary = {
        "analysis": "rr100_fit_quality_vs_unit_specific_predictor_increment_checkpoint_15a",
        "status": "map_first_checkpoint_stop_before_threshold_and_covariate_models",
        "n_quality_units": int(len(table)),
        "delta_definition": "unit-specific SFxTF overlap CV R2 minus total supported power CV R2",
        "spearman_joint_fit_vs_delta": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "pearson_joint_fit_vs_delta": float(
            stats.pearsonr(
                table["joint_parametric_surface_r2"], table["delta_cv_r2_unit_minus_total"]
            ).statistic
        ),
        "median_delta": float(table["delta_cv_r2_unit_minus_total"].median()),
        "fraction_delta_positive": float((table["delta_cv_r2_unit_minus_total"] > 0).mean()),
        "source_table": str(
            (SOURCE / "predictor_variant_per_unit_explainability.csv").relative_to(ROOT)
        ),
        "selection_table": "auditable_fit_quality_example_selection.csv",
        "next_checkpoint": "raw held-out condition profiles for selected units, then threshold/covariate summaries",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Checkpoint 15A: fit quality versus predictor increment\n\n"
        "This is the first map-first diagnostic for the hypothesis that imperfect grating fits "
        "suppress the unit-specific SF×TF predictor. It shows raw values for the 66 units in the "
        "pre-existing quality cohort and stops before threshold, covariate, or empirical-surface "
        "population conclusions.\n\n"
        "The vertical outcome is the paired difference between held-out R² for unit-specific "
        "SF×TF overlap and total supported dynamic power. Negative values favor total power.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

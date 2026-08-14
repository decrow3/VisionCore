#!/usr/bin/env python3
"""Test RF-local orientation-aware power on 61 units and 100 clean-history conditions."""
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

from declan.fig4_active_sensing.spectral_cache_contract import (
    validate_artifact_not_superseded,
    validate_spectral_cache,
)

from declan.fig4_active_sensing.analyze_rr100_natural_image_rf_local_oriented_power_expanded_clean_history import (
    file_identity,
    select_clean_conditions,
    summarize_units,
)
from declan.fig4_active_sensing.analyze_rr100_natural_image_rf_local_oriented_power_response_checkpoint import (
    audit_clean_history,
    load_response_rows,
)
from declan.fig4_active_sensing.make_rr100_natural_image_rf_local_oriented_power_checkpoint import (
    build_metrics,
    load_selected_movies,
    verify_reconstruction,
)
from declan.fig4_active_sensing.run_rr100_recorded_grating_power_formula_population import (
    load_session_apertures,
    load_tuning,
)


ROOT = Path(__file__).resolve().parents[2]
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
TRACE_FLAGS = RESPONSES / "quality_control/pre_fixation_history_trace_flags.csv"
ASSEMBLED = RESPONSES / "assembled/rounds_000_002_n003"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
RF_POPULATION = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_rf_local_population_v1"
TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_population_n100x61_clean_history_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectral-dir", type=Path, required=True,
        help="Explicit frozen corrected spectral cache; superseded caches are rejected.",
    )
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSES)
    parser.add_argument("--trace-flags", type=Path, default=TRACE_FLAGS)
    parser.add_argument("--assembled-dir", type=Path, default=ASSEMBLED)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--rf-population-dir", type=Path, default=RF_POPULATION)
    parser.add_argument("--tuning-dir", type=Path, default=TUNING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--condition-bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--population-bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--n-condition-bootstrap", type=int, default=2000)
    parser.add_argument("--n-population-bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def load_population_contract(
    response_dir: Path, tuning_dir: Path
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    units = pd.read_csv(response_dir / "all_unit_rf_metadata.csv").sort_values("rr100_index").reset_index(drop=True)
    if len(units) != 61 or units.session.nunique() != 15:
        raise ValueError(f"Expected 61 units from 15 sessions, got {len(units)} from {units.session.nunique()}")
    radial_weights, oriented_weights, sf, tf = load_tuning(tuning_dir)
    apertures: dict[int, np.ndarray] = {}
    for session, group in units.groupby("session", sort=True):
        session_apertures = load_session_apertures(response_dir, str(session))
        for unit in group.rr100_index.astype(int):
            if int(unit) not in session_apertures:
                raise ValueError(f"Missing receptive-field aperture for unit {unit} in {session}")
            apertures[int(unit)] = session_apertures[int(unit)]
    missing = [
        int(unit)
        for unit in units.rr100_index.astype(int)
        if int(unit) not in apertures or int(unit) not in radial_weights or int(unit) not in oriented_weights
    ]
    if missing:
        raise ValueError(f"Population contract is missing aperture or tuning for units {missing}")
    return units, apertures, radial_weights, oriented_weights, sf, tf


def population_bootstrap(
    unit_summary: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = {
        "correlation_difference": "orientation_aware_minus_collapsed_r",
        "held_out_r2_difference": "orientation_aware_minus_collapsed_held_out_r2",
        "standardized_error_difference": "orientation_aware_minus_collapsed_z_mae",
    }
    sessions = unit_summary.session.drop_duplicates().to_numpy(str)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for bootstrap_index in range(n_bootstrap):
        sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True)
        record: dict[str, object] = {"bootstrap_index": int(bootstrap_index)}
        for label, column in metrics.items():
            session_values = []
            for session in sampled_sessions:
                group = unit_summary.loc[unit_summary.session.eq(session), column].to_numpy(float)
                session_values.append(float(rng.choice(group, size=len(group), replace=True).mean()))
            record[label] = float(np.mean(session_values))
        rows.append(record)
    bootstrap = pd.DataFrame(rows)

    summaries = []
    for label, column in metrics.items():
        session_means = unit_summary.groupby("session")[column].mean()
        values = bootstrap[label].to_numpy(float)
        summaries.append(
            {
                "metric": label,
                "session_balanced_point_estimate": float(session_means.mean()),
                "session_unit_bootstrap_low": float(np.quantile(values, 0.025)),
                "session_unit_bootstrap_high": float(np.quantile(values, 0.975)),
                "bootstrap_fraction_positive": float(np.mean(values > 0)),
                "fraction_units_positive": float(np.mean(unit_summary[column].to_numpy(float) > 0)),
                "fraction_sessions_positive": float(np.mean(session_means.to_numpy(float) > 0)),
            }
        )
    return pd.DataFrame(summaries), bootstrap


def plot_population(unit_summary: pd.DataFrame, population: pd.DataFrame, out: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    color = np.where(unit_summary.sf_outer_third.eq("sf_high_half"), "#D55E00", "#0072B2")
    identity = np.linspace(-0.4, 0.6, 100)

    axes[0, 0].scatter(
        unit_summary.orientation_collapsed_vs_modulation_r,
        unit_summary.orientation_aware_vs_modulation_r,
        c=color,
        s=34,
        alpha=0.8,
    )
    axes[0, 0].plot(identity, identity, color="0.4", ls="--")
    axes[0, 0].set(
        xlabel="orientation-collapsed correlation",
        ylabel="orientation-aware correlation",
        title="Correlation with response-modulation magnitude",
    )

    axes[0, 1].hist(unit_summary.orientation_aware_minus_collapsed_r, bins=16, color="#D55E00", alpha=0.85)
    axes[0, 1].axvline(0, color="0.3", ls="--")
    correlation = population.loc[population.metric.eq("correlation_difference")].iloc[0]
    axes[0, 1].set(
        xlabel="orientation-aware minus orientation-collapsed correlation",
        ylabel="units",
        title=(
            f"Session-balanced mean difference={correlation.session_balanced_point_estimate:+.03f}\n"
            f"95% interval [{correlation.session_unit_bootstrap_low:+.03f}, "
            f"{correlation.session_unit_bootstrap_high:+.03f}]"
        ),
    )

    axes[0, 2].scatter(
        unit_summary.orientation_collapsed_held_out_r2,
        unit_summary.orientation_aware_held_out_r2,
        c=color,
        s=34,
        alpha=0.8,
    )
    limits = [
        float(min(unit_summary.orientation_collapsed_held_out_r2.min(), unit_summary.orientation_aware_held_out_r2.min())),
        float(max(unit_summary.orientation_collapsed_held_out_r2.max(), unit_summary.orientation_aware_held_out_r2.max())),
    ]
    axes[0, 2].plot(limits, limits, color="0.4", ls="--")
    axes[0, 2].set(
        xlabel="orientation-collapsed held-out R²",
        ylabel="orientation-aware held-out R²",
        title="Five-fold prediction of response-modulation magnitude",
    )

    r2_difference = unit_summary.orientation_aware_minus_collapsed_held_out_r2
    axes[1, 0].hist(r2_difference, bins=16, color="#009E73", alpha=0.85)
    axes[1, 0].axvline(0, color="0.3", ls="--")
    r2 = population.loc[population.metric.eq("held_out_r2_difference")].iloc[0]
    axes[1, 0].set(
        xlabel="orientation-aware minus orientation-collapsed held-out R²",
        ylabel="units",
        title=(
            f"Session-balanced mean difference={r2.session_balanced_point_estimate:+.03f}\n"
            f"95% interval [{r2.session_unit_bootstrap_low:+.03f}, {r2.session_unit_bootstrap_high:+.03f}]"
        ),
    )

    session = unit_summary.groupby("session", as_index=False).agg(
        correlation_difference=("orientation_aware_minus_collapsed_r", "mean"),
        held_out_r2_difference=("orientation_aware_minus_collapsed_held_out_r2", "mean"),
    )
    y = np.arange(len(session))
    axes[1, 1].scatter(session.correlation_difference, y, color="#D55E00", label="correlation difference")
    axes[1, 1].scatter(session.held_out_r2_difference, y, color="#009E73", label="held-out R² difference")
    axes[1, 1].axvline(0, color="0.3", ls="--")
    axes[1, 1].set_yticks(y, session.session, fontsize=7)
    axes[1, 1].set(xlabel="session mean orientation-aware minus collapsed", title="Results by recording session")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].bar(
        ["correlation", "held-out R²", "standardized\nabsolute error"],
        [
            correlation.fraction_units_positive,
            r2.fraction_units_positive,
            1.0
            - population.loc[population.metric.eq("standardized_error_difference"), "fraction_units_positive"].iloc[0],
        ],
        color=["#D55E00", "#009E73", "#0072B2"],
    )
    axes[1, 2].axhline(0.5, color="0.4", ls="--")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set(
        ylabel="fraction of 61 units favoring orientation-aware power",
        title="How consistently does orientation information help?",
    )

    figure.suptitle(
        "Population clean-history checkpoint: does orientation-aware retinal-image power better predict digital-twin response-modulation magnitude?\n"
        "61 recorded-spatial-frequency-validated units · 15 sessions · 100 input-only image/trace conditions per unit",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_spectral_cache(args.spectral_dir)
    validate_artifact_not_superseded(args.tuning_dir, label="orientation tuning")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected, eligibility, spectral = select_clean_conditions(
        args.spectral_dir, args.trace_flags, 100, int(args.selection_seed)
    )
    history_audit = audit_clean_history(selected, args.trace_flags)
    payload = load_selected_movies(selected, args.cohort_dir, args.response_cache_dir, args.device)
    reconstruction = verify_reconstruction(selected, payload, spectral)
    units, apertures, radial_weights, oriented_weights, sf, tf = load_population_contract(
        args.rf_population_dir, args.tuning_dir
    )
    power, _, _ = build_metrics(
        selected, payload, units, apertures, radial_weights, oriented_weights, sf, tf, spectral
    )
    response, _, response_audit = load_response_rows(
        power, selected, args.response_cache_dir, args.assembled_dir
    )
    unit_summary, condition_bootstrap = summarize_units(
        response, int(args.n_condition_bootstrap), int(args.condition_bootstrap_seed)
    )
    unit_summary["orientation_aware_minus_collapsed_held_out_r2"] = (
        unit_summary.orientation_aware_held_out_r2 - unit_summary.orientation_collapsed_held_out_r2
    )
    metadata_columns = ["rr100_index", "session", "sf_outer_third", "preferred_sf_cpd", "preferred_orientation_deg"]
    unit_summary = unit_summary.merge(
        units[metadata_columns], on="rr100_index", how="left", validate="one_to_one"
    )
    population, population_bootstraps = population_bootstrap(
        unit_summary, int(args.n_population_bootstrap), int(args.population_bootstrap_seed)
    )

    selected.to_csv(args.out_dir / "selected_conditions.csv", index=False)
    eligibility.to_csv(args.out_dir / "condition_history_eligibility.csv", index=False)
    history_audit.to_csv(args.out_dir / "selected_condition_history_audit.csv", index=False)
    reconstruction.to_csv(args.out_dir / "movie_and_spectrum_reconstruction_audit.csv", index=False)
    response.to_csv(args.out_dir / "condition_unit_power_response_metrics.csv", index=False)
    response_audit.to_csv(args.out_dir / "response_join_audit.csv", index=False)
    unit_summary.to_csv(args.out_dir / "unit_population_summary.csv", index=False)
    condition_bootstrap.to_csv(args.out_dir / "condition_bootstrap_unit_correlation_differences.csv", index=False)
    population.to_csv(args.out_dir / "session_balanced_population_summary.csv", index=False)
    population_bootstraps.to_csv(args.out_dir / "session_unit_bootstrap_population_differences.csv", index=False)
    plot_population(
        unit_summary,
        population,
        args.out_dir / "population_clean_history_power_response_summary",
        int(args.dpi),
    )

    audit_columns = [
        column
        for column in response_audit
        if column.endswith("_error_vs_assembled_hz") or column.endswith("_error_vs_shard_hz")
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "population clean-history natural-image RF-local orientation-aware power checkpoint",
        "status": "provisional_61_unit_population_checkpoint_complete",
        "scope": {
            "conditions": int(selected.matrix_row_index.nunique()),
            "unique_images": int(selected.image_index.nunique()),
            "unique_traces": int(selected.trace_index.nunique()),
            "units": int(unit_summary.rr100_index.nunique()),
            "sessions": int(unit_summary.session.nunique()),
            "condition_unit_pairs": int(len(response)),
        },
        "contracts": {
            "conditions": "same input-only seeded one-image/one-trace clean-history matching as the five-unit expanded checkpoint",
            "units": "all 61 recorded-spatial-frequency-validated units with frozen all-session receptive-field apertures",
            "primary_outcome": "40-frame digital-twin RMS of moving minus matched stabilized firing rate",
            "primary_comparison": "orientation-aware versus orientation-collapsed receptive-field-local power",
            "population_uncertainty": "session-balanced hierarchical bootstrap of sessions and units within sessions",
            "limitation": "provisional 577-trace quarantine subset and reconstructed old three-round spectral crosswalk; not the final replacement cohort",
        },
        "verification": {
            "all_selected_histories_within_fixation": bool(history_audit.clean_history_gate_pass.all()),
            "maximum_cached_radial_reconstruction_relative_error": float(reconstruction.maximum_radial_relative_error.max()),
            "maximum_cached_oriented_reconstruction_relative_error": float(reconstruction.maximum_oriented_relative_error.max()),
            "maximum_orientation_sum_relative_error": float(reconstruction.orientation_sum_relative_error.max()),
            "maximum_response_join_or_formula_error_hz": float(response_audit[audit_columns].to_numpy(float).max()),
            "all_response_values_finite": bool(np.isfinite(response.select_dtypes("number")).all().all()),
        },
        "inputs": {
            "spectral_cache": file_identity(args.spectral_dir / "condition_spectra.npz"),
            "trace_history_flags": file_identity(args.trace_flags),
            "population_rf_metadata": file_identity(args.rf_population_dir / "all_unit_rf_metadata.csv"),
            "orientation_tuning": file_identity(args.tuning_dir / "orientation_aware_f0_tuning_and_routing.npz"),
        },
        "artifacts": {
            "figure": "population_clean_history_power_response_summary.png",
            "selected_conditions": "selected_conditions.csv",
            "pair_metrics": "condition_unit_power_response_metrics.csv",
            "unit_summary": "unit_population_summary.csv",
            "population_summary": "session_balanced_population_summary.csv",
            "condition_bootstrap": "condition_bootstrap_unit_correlation_differences.csv",
            "population_bootstrap": "session_unit_bootstrap_population_differences.csv",
            "history_audit": "selected_condition_history_audit.csv",
            "spectrum_audit": "movie_and_spectrum_reconstruction_audit.csv",
            "response_audit": "response_join_audit.csv",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(population.to_string(index=False))


if __name__ == "__main__":
    main()

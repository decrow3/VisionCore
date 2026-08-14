#!/usr/bin/env python3
"""Run the RF-local three-way recorded-grating checkpoint across all eligible RR100 units."""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.utils import get_model_and_dataset_configs
from declan.fig4_active_sensing import make_rr100_recorded_grating_three_way_response_checkpoint as three
from declan.fig4_active_sensing.make_rr100_recorded_grating_retinal_power_input_checkpoint import (
    candidate_windows,
    load_heldout_grating_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_rf_local_population_v1"
BOOTSTRAPS = 5000
SEED = 1731


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def response_cache_by_session(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        outputs = dill.load(handle)
    result = {}
    for row in outputs:
        grating = row.get("bps_results", {}).get("gratings")
        if grating is not None:
            result[str(row["sess"])] = grating
    return result


def session_balanced_bootstrap(
    frame: pd.DataFrame, column: str, *, n_bootstraps: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    grouped = {
        str(session): values[column].dropna().to_numpy(float)
        for session, values in frame.groupby("session", sort=True)
    }
    grouped = {key: value for key, value in grouped.items() if value.size}
    sessions = np.asarray(sorted(grouped), dtype=object)
    session_means = np.asarray([np.mean(grouped[str(session)]) for session in sessions], dtype=float)
    point = float(np.mean(session_means)) if session_means.size else float("nan")
    if session_means.size == 0 or n_bootstraps <= 0:
        return point, float("nan"), float("nan")
    boot = np.empty(n_bootstraps, dtype=float)
    for index in range(n_bootstraps):
        sampled = rng.integers(0, len(sessions), size=len(sessions))
        draw_means = []
        for session_index in sampled:
            values = grouped[str(sessions[session_index])]
            draw = values[rng.integers(0, len(values), size=len(values))]
            draw_means.append(float(np.mean(draw)))
        boot[index] = float(np.mean(draw_means))
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize_population(metrics: pd.DataFrame, n_bootstraps: int) -> pd.DataFrame:
    requested = {
        "full_twin_vs_recorded_within_trial_r": "full twin → recorded",
        "power_vs_recorded_within_trial_r": "RF-local power → recorded",
        "power_vs_recorded_cv_r2": "RF-local power held-out ΔR²",
        "whole_crop_power_vs_recorded_within_trial_r": "whole-crop control → recorded",
        "whole_crop_power_vs_recorded_cv_r2": "whole-crop control held-out ΔR²",
        "raw_local_routed_power_vs_full_twin_r": "RF-local power → full twin",
        "rf_local_minus_whole_crop_within_trial_r": "RF-local minus whole-crop recorded r",
        "rf_local_minus_whole_crop_cv_r2": "RF-local minus whole-crop held-out ΔR²",
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for column, label in requested.items():
        values = metrics[column].to_numpy(float)
        finite = np.isfinite(values)
        point, low, high = session_balanced_bootstrap(
            metrics, column, n_bootstraps=n_bootstraps, rng=rng
        )
        rows.append(
            {
                "metric": column,
                "label": label,
                "n_units_finite": int(np.count_nonzero(finite)),
                "n_sessions_finite": int(metrics.loc[finite, "session"].nunique()),
                "unit_mean": float(np.mean(values[finite])) if np.any(finite) else float("nan"),
                "unit_median": float(np.median(values[finite])) if np.any(finite) else float("nan"),
                "fraction_units_positive": float(np.mean(values[finite] > 0)) if np.any(finite) else float("nan"),
                "session_balanced_mean": point,
                "session_cluster_boot_ci_low": low,
                "session_cluster_boot_ci_high": high,
                "n_bootstraps": int(n_bootstraps),
            }
        )
    return pd.DataFrame(rows)


def eligibility_audit(cohort: pd.DataFrame) -> pd.DataFrame:
    audit = cohort.copy()
    audit["included_in_population"] = audit.routing_quality_pass.astype(bool)
    audit["exclusion_reason"] = "included"
    invalid_tuning = ~audit.responsive_positive_f0_flag.astype(bool)
    audit.loc[invalid_tuning, "exclusion_reason"] = (
        "no_positive_responsive_F0_and_no_valid_extended_SF_TF_fit"
    )
    validation_failure = (
        audit.responsive_positive_f0_flag.astype(bool)
        & ~audit.recorded_validation_pass.astype(bool)
    )
    audit.loc[validation_failure, "exclusion_reason"] = "failed_recorded_grating_validation"
    inconsistent = audit.included_in_population.ne(audit.exclusion_reason.eq("included"))
    if bool(inconsistent.any()):
        raise ValueError("Routing-quality flag and explicit exclusion reason disagree")
    return audit


def plot_population(metrics: pd.DataFrame, summary: pd.DataFrame, out: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    sessions = sorted(metrics.session.unique())
    colors = dict(zip(sessions, plt.cm.tab20(np.linspace(0, 1, len(sessions))), strict=True))
    for session, frame in metrics.groupby("session", sort=True):
        axes[0, 0].scatter(
            frame.full_twin_vs_recorded_within_trial_r,
            frame.power_vs_recorded_within_trial_r,
            s=30,
            alpha=0.75,
            color=colors[session],
        )
    limits = (-0.55, 0.75)
    axes[0, 0].plot(limits, limits, "--", color="0.55", lw=1)
    axes[0, 0].axhline(0, color="0.75", lw=0.8)
    axes[0, 0].axvline(0, color="0.75", lw=0.8)
    axes[0, 0].set(
        xlim=limits,
        ylim=limits,
        xlabel="full twin → recorded within-trial r",
        ylabel="RF-local power → recorded within-trial r",
        title="A  Full model versus RF-local power",
    )

    finite = metrics[["whole_crop_power_vs_recorded_within_trial_r", "power_vs_recorded_within_trial_r"]].dropna()
    axes[0, 1].scatter(
        finite.whole_crop_power_vs_recorded_within_trial_r,
        finite.power_vs_recorded_within_trial_r,
        s=30,
        color="#E69F00",
        alpha=0.7,
    )
    lo = float(np.nanmin(finite.to_numpy())) - 0.05
    hi = float(np.nanmax(finite.to_numpy())) + 0.05
    axes[0, 1].plot([lo, hi], [lo, hi], "--", color="0.45", lw=1)
    axes[0, 1].set(
        xlabel="whole-crop power → recorded r",
        ylabel="RF-local power → recorded r",
        title="B  Spatial localization control",
    )

    axes[0, 2].hist(
        metrics.power_vs_recorded_cv_r2.dropna(), bins=18, color="#E69F00", alpha=0.8, label="RF-local"
    )
    axes[0, 2].hist(
        metrics.whole_crop_power_vs_recorded_cv_r2.dropna(),
        bins=18,
        histtype="step",
        linewidth=1.8,
        color="0.25",
        label="whole crop",
    )
    axes[0, 2].axvline(0, color="0.35", lw=1)
    axes[0, 2].set(xlabel="trial-held-out ΔR²", ylabel="units", title="C  Predictive gain over fold baseline")
    axes[0, 2].legend(frameon=False)

    session_summary = metrics.groupby("session", as_index=False).agg(
        n_units=("rr100_index", "size"),
        full_twin_r=("full_twin_vs_recorded_within_trial_r", "mean"),
        rf_local_power_r=("power_vs_recorded_within_trial_r", "mean"),
    )
    y = np.arange(len(session_summary))
    axes[1, 0].scatter(session_summary.full_twin_r, y, color="#0072B2", label="full twin", s=35)
    axes[1, 0].scatter(session_summary.rf_local_power_r, y, color="#E69F00", label="RF-local power", s=35)
    for position, row in enumerate(session_summary.itertuples(index=False)):
        axes[1, 0].plot([row.full_twin_r, row.rf_local_power_r], [position, position], color="0.75", lw=1)
    axes[1, 0].axvline(0, color="0.55", lw=0.8)
    axes[1, 0].set_yticks(y, [f"{row.session} (n={row.n_units})" for row in session_summary.itertuples(index=False)], fontsize=7)
    axes[1, 0].set(xlabel="session mean within-trial r", title="D  Session-level replication")
    axes[1, 0].legend(frameon=False, fontsize=8)

    labels = ["full twin\n→ recorded", "RF-local power\n→ recorded", "RF-local power\n→ full twin"]
    columns = [
        "full_twin_vs_recorded_within_trial_r",
        "power_vs_recorded_within_trial_r",
        "raw_local_routed_power_vs_full_twin_r",
    ]
    data = [metrics[column].dropna().to_numpy(float) for column in columns]
    parts = axes[1, 1].violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#0072B2", "#E69F00", "#009E73"], strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    axes[1, 1].axhline(0, color="0.55", lw=0.8)
    axes[1, 1].set_xticks([1, 2, 3], labels)
    axes[1, 1].set(ylabel="correlation", title="E  Three-way population distributions")

    lookup = summary.set_index("metric")
    shown = [
        "full_twin_vs_recorded_within_trial_r",
        "power_vs_recorded_within_trial_r",
        "rf_local_minus_whole_crop_within_trial_r",
        "power_vs_recorded_cv_r2",
    ]
    positions = np.arange(len(shown))
    means = np.asarray([lookup.loc[key, "session_balanced_mean"] for key in shown], dtype=float)
    lows = np.asarray([lookup.loc[key, "session_cluster_boot_ci_low"] for key in shown], dtype=float)
    highs = np.asarray([lookup.loc[key, "session_cluster_boot_ci_high"] for key in shown], dtype=float)
    axes[1, 2].errorbar(
        means,
        positions,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color="black",
        capsize=3,
    )
    axes[1, 2].axvline(0, color="0.55", lw=0.8)
    axes[1, 2].set_yticks(
        positions,
        ["full twin r", "RF-local power r", "local − whole-crop r", "RF-local ΔR²"],
    )
    axes[1, 2].invert_yaxis()
    axes[1, 2].set(xlabel="session-balanced mean [95% cluster bootstrap CI]", title="F  Population estimates")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"Recorded gratings across the routing-qualified R100 cohort\n"
        f"{len(metrics)} units · {metrics.session.nunique()} sessions · RF-local power · whole trials held out",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = args.out_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(three.ROUTING_DATA / "routing_unit_cohort.csv")
    unit_eligibility = eligibility_audit(cohort)
    eligible = cohort.loc[cohort.routing_quality_pass.astype(bool)].copy()
    sessions = sorted(eligible.session.unique())
    if args.max_sessions > 0:
        sessions = sessions[: int(args.max_sessions)]
        eligible = eligible.loc[eligible.session.isin(sessions)].copy()
    alignment = pd.read_csv(three.ALIGNMENT).set_index("session")

    print("Loading the fitted model once for all RF apertures", flush=True)
    model, _ = get_model_and_dataset_configs(mode="standard")
    model.model.eval()
    print("Loading the monolithic response cache once", flush=True)
    response_cache = response_cache_by_session(three.CACHE)

    all_predictions = []
    all_metrics = []
    all_folds = []
    all_trials = []
    all_rf = []
    audit_rows = []
    for number, session in enumerate(sessions, start=1):
        expected_units = int(np.count_nonzero(eligible.session.eq(session)))
        print(f"[{number}/{len(sessions)}] {session}: {expected_units} eligible units", flush=True)
        if session not in response_cache:
            raise KeyError(f"No cached grating responses for eligible session {session}")
        if session not in alignment.index or not bool(alignment.loc[session, "validation_length_alignment"]):
            raise ValueError(f"No passing cache alignment for eligible session {session}")

        session_args = copy.copy(args)
        session_args.dataset_config = three.DATASET_CONFIG
        session_args.response_cache = three.CACHE
        session_args.mapping_csv = three.MAPPING
        session_args.grating_metrics_csv = three.GRATING_METRICS
        session_args.alignment_csv = three.ALIGNMENT
        session_args.routing_data_dir = three.ROUTING_DATA
        session_args.session = session
        session_args.stride = three.N_SCORE
        dset, local, _ = load_heldout_grating_dataset(
            three.DATASET_CONFIG,
            session,
            preserve_config_cids=not session.startswith("Logan_"),
        )
        window_metrics, payload, candidate_sf, candidate_tf, ppd = candidate_windows(
            dset, local, three.N_SCORE, 0, session=session
        )
        units, sensitivity, power_sf, power_tf, footprints, apertures = three.unit_table(
            session_args, fitted_model=model
        )
        rows = three.build_window_unit_table(
            window_metrics,
            payload,
            units,
            sensitivity,
            power_sf,
            power_tf,
            candidate_sf,
            candidate_tf,
            ppd,
            apertures,
            dset,
            local,
            response_cache[session],
            str(alignment.loc[session, "alignment_basis"]),
        )
        predictions, metrics, fold_fits, trial_tracking = three.fit_and_score(rows, int(args.n_folds))
        metadata_columns = [
            "rr100_index", "session", "source_unit_index", "peak_lag_bins", "peak_lag_ms",
            "extended_rank1_centered_r2", "extended_sf_fit_r2", "extended_tf_fit_r2",
            "preferred_sf_cpd", "extended_tf_center_frequency", "readout_mean_y_feature_pixel",
            "readout_mean_x_feature_pixel", "readout_std_y_feature_pixel", "readout_std_x_feature_pixel",
            "readout_theta_radian", "rf_center_x_pixel", "rf_center_y_pixel",
            "rf_rms_radius_pixel", "rf_radius95_pixel",
        ]
        metrics = metrics.merge(units[metadata_columns], on="rr100_index", validate="one_to_one")
        predictions.insert(0, "session", session)
        fold_fits.insert(0, "session", session)
        trial_tracking.insert(0, "session", session)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_folds.append(fold_fits)
        all_trials.append(trial_tracking)
        all_rf.append(units)
        np.savez_compressed(
            session_dir / f"{session}_rf_apertures.npz",
            rr100_index=units.rr100_index.to_numpy(np.int64),
            rf_footprint=np.stack([footprints[int(unit)] for unit in units.rr100_index]),
            spectral_aperture=np.stack([apertures[int(unit)] for unit in units.rr100_index]),
        )
        audit_rows.append(
            {
                "session": session,
                "status": "complete",
                "n_expected_units": expected_units,
                "n_completed_units": int(len(metrics)),
                "n_candidate_windows": int(len(window_metrics)),
                "n_scored_window_unit_rows": int(len(predictions)),
                "alignment_basis": str(alignment.loc[session, "alignment_basis"]),
            }
        )
        del dset, payload, rows

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True).sort_values("rr100_index").reset_index(drop=True)
    folds = pd.concat(all_folds, ignore_index=True)
    trials = pd.concat(all_trials, ignore_index=True)
    rf_metadata = pd.concat(all_rf, ignore_index=True).sort_values("rr100_index")
    audit = pd.DataFrame(audit_rows)
    metrics["rf_local_minus_whole_crop_within_trial_r"] = (
        metrics.power_vs_recorded_within_trial_r - metrics.whole_crop_power_vs_recorded_within_trial_r
    )
    metrics["rf_local_minus_whole_crop_cv_r2"] = (
        metrics.power_vs_recorded_cv_r2 - metrics.whole_crop_power_vs_recorded_cv_r2
    )
    summary = summarize_population(metrics, int(args.n_bootstraps))
    session_summary = metrics.groupby("session", as_index=False).agg(
        n_units=("rr100_index", "size"),
        full_twin_recorded_r_mean=("full_twin_vs_recorded_within_trial_r", "mean"),
        rf_local_power_recorded_r_mean=("power_vs_recorded_within_trial_r", "mean"),
        rf_local_power_cv_r2_mean=("power_vs_recorded_cv_r2", "mean"),
        whole_crop_power_recorded_r_mean=("whole_crop_power_vs_recorded_within_trial_r", "mean"),
        local_minus_whole_crop_r_mean=("rf_local_minus_whole_crop_within_trial_r", "mean"),
    )

    predictions.to_csv(args.out_dir / "all_window_unit_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "all_unit_metrics.csv", index=False)
    folds.to_csv(args.out_dir / "all_power_rate_cv_fold_fits.csv", index=False)
    trials.to_csv(args.out_dir / "all_unit_trial_tracking.csv", index=False)
    rf_metadata.to_csv(args.out_dir / "all_unit_rf_metadata.csv", index=False)
    unit_eligibility.to_csv(args.out_dir / "unit_eligibility_audit.csv", index=False)
    audit.to_csv(args.out_dir / "session_coverage_audit.csv", index=False)
    summary.to_csv(args.out_dir / "population_summary.csv", index=False)
    session_summary.to_csv(args.out_dir / "session_summary.csv", index=False)
    figure_base = args.out_dir / "rf_local_three_way_population_summary"
    plot_population(metrics, summary, figure_base, int(args.dpi))

    finite_pair = metrics[["full_twin_vs_recorded_within_trial_r", "power_vs_recorded_within_trial_r"]].dropna()
    rank_r = float(spearmanr(finite_pair.iloc[:, 0], finite_pair.iloc[:, 1]).statistic)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_three_way_rf_local_population",
        "status": "all_eligible_units_complete",
        "coverage": {
            "n_rr100_total": int(len(cohort)),
            "n_routing_qualified": int(len(eligible)),
            "n_completed_units": int(len(metrics)),
            "n_completed_sessions": int(metrics.session.nunique()),
            "all_eligible_units_completed": bool(len(metrics) == len(eligible)),
            "alignment_basis_counts": audit.alignment_basis.value_counts().to_dict(),
        },
        "contracts": {
            "response_objects": "full cached twin rhat; trial-held-out RF-local routed-power rate; recorded robs rate",
            "rf_localization": "fitted Gaussian readout back-projected through feedforward spatial support to a normalized 51x51 Fourier aperture",
            "primary_metric": "within-trial Pearson r; complete experimental trials held out in five folds",
            "population_uncertainty": "session-balanced hierarchical bootstrap resampling sessions and units within sessions",
            "whole_crop_role": "diagnostic control only",
        },
        "diagnostics": {"unit_level_spearman_full_twin_r_vs_rf_local_power_r": rank_r},
        "artifacts": {
            "unit_metrics": "all_unit_metrics.csv",
            "population_summary": "population_summary.csv",
            "session_summary": "session_summary.csv",
            "coverage_audit": "session_coverage_audit.csv",
            "unit_eligibility_audit": "unit_eligibility_audit.csv",
            "figure_png": "rf_local_three_way_population_summary.png",
            "figure_pdf": "rf_local_three_way_population_summary.pdf",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote population checkpoint to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

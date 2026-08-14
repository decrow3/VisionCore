#!/usr/bin/env python3
"""Run the recorded-grating power-formula comparison for all eligible RR100 units."""
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

from declan.fig4_active_sensing.analyze_rr100_recorded_grating_power_formula_response_checkpoint import (
    NESTED_FORMULA,
    formula_differences,
    score_formulas,
    summarize_metrics,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_oriented_power_checkpoint import (
    localized_oriented_spectrum,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_retinal_power_input_checkpoint import (
    candidate_windows,
    load_heldout_grating_dataset,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_three_way_response_checkpoint import (
    indices_for_support,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    N_SCORE,
    ORIENTATION_EDGES_DEG,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSE_DIR = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_rf_local_population_v1"
)
DEFAULT_TUNING_DIR = ROOT / (
    "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
)
DEFAULT_DATASET_CONFIG = ROOT / "experiments/dataset_configs/multi_basic_120_long_legacy.yaml"
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_power_formula_population_v2"
)
BOOTSTRAPS = 5000
SEED = 2381


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--n-bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def load_tuning(tuning_dir: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    path = tuning_dir / "orientation_aware_f0_tuning_and_routing.npz"
    with np.load(path, allow_pickle=False) as archive:
        units = archive["rr100_index"].astype(int)
        sf = archive["movie_sf_cpd"].astype(float)
        tf = archive["movie_tf_hz"].astype(float)
        orientations = archive["movie_fourier_orientation_deg"].astype(float)
        radial = archive["smoothed_radial_f0_weight"].astype(float)
        oriented = archive["orientation_aware_f0_weight"].astype(float)
    expected = 0.5 * (ORIENTATION_EDGES_DEG[:-1] + ORIENTATION_EDGES_DEG[1:])
    if not np.allclose(orientations, expected):
        raise ValueError("Tuning and Fourier orientation axes do not match")
    radial_lookup = {int(unit): radial[index] for index, unit in enumerate(units)}
    oriented_lookup = {int(unit): oriented[index] for index, unit in enumerate(units)}
    for unit in units:
        if not np.allclose(
            oriented_lookup[int(unit)].mean(axis=-1),
            radial_lookup[int(unit)],
            rtol=2e-6,
            atol=1e-8,
        ):
            raise ValueError(f"Orientation weights fail radial nesting for RR100 {unit}")
    return radial_lookup, oriented_lookup, sf, tf


def load_session_apertures(response_dir: Path, session: str) -> dict[int, np.ndarray]:
    path = response_dir / "sessions" / f"{session}_rf_apertures.npz"
    with np.load(path, allow_pickle=False) as archive:
        units = archive["rr100_index"].astype(int)
        apertures = archive["spectral_aperture"].astype(float)
    if len(units) != len(np.unique(units)) or apertures.shape != (len(units), 51, 51):
        raise ValueError(f"Invalid aperture archive for {session}")
    return {int(unit): apertures[index] for index, unit in enumerate(units)}


def build_session_power_rows(
    window_metrics: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    unit_ids: np.ndarray,
    apertures: dict[int, np.ndarray],
    radial_weights: dict[int, np.ndarray],
    oriented_weights: dict[int, np.ndarray],
    tuning_sf: np.ndarray,
    tuning_tf: np.ndarray,
    candidate_sf: np.ndarray,
    candidate_tf: np.ndarray,
    ppd: float,
) -> pd.DataFrame:
    sf_index = indices_for_support(candidate_sf, tuning_sf, "SF")
    tf_index = indices_for_support(candidate_tf, tuning_tf, "TF")
    rows: list[dict[str, object]] = []
    for number, window in enumerate(window_metrics.itertuples(index=False), start=1):
        item = payload[int(window.window_index)]
        movie = (item["movie_uint8"].astype(np.float32) - 127.0) / 255.0
        for unit in unit_ids:
            unit_id = int(unit)
            radial_full, oriented_full = localized_oriented_spectrum(
                movie, ppd=float(ppd), spatial_aperture=apertures[unit_id]
            )
            radial = radial_full[np.ix_(tf_index, sf_index)]
            oriented = oriented_full[tf_index][:, sf_index, :]
            radial_drive = float(np.sum(radial * radial_weights[unit_id]))
            oriented_drive = float(
                np.sum(oriented * oriented_weights[unit_id])
            )
            rows.append(
                {
                    "window_index": int(window.window_index),
                    "trial_index": int(window.trial_index),
                    "start_index_120hz": int(window.start_index_120hz),
                    "rr100_index": unit_id,
                    "rf_local_total_power_amplitude": float(
                        np.sqrt(max(float(radial.sum()), 0.0))
                    ),
                    "radial_direct_f0_drive": radial_drive,
                    "oriented_direct_f0_drive": oriented_drive,
                    "orientation_delta_drive": oriented_drive - radial_drive,
                }
            )
        if number % 50 == 0 or number == len(window_metrics):
            print(f"  spectralized {number}/{len(window_metrics)} windows", flush=True)
    return pd.DataFrame(rows)


def session_balanced_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    grouped = {
        str(session): values[value_column].dropna().to_numpy(float)
        for session, values in frame.groupby("session", sort=True)
    }
    grouped = {key: values for key, values in grouped.items() if len(values)}
    sessions = np.asarray(sorted(grouped), dtype=object)
    point = float(
        np.mean([np.mean(grouped[str(session)]) for session in sessions])
    )
    if n_bootstraps <= 0:
        return point, float("nan"), float("nan")
    samples = np.empty(n_bootstraps, dtype=float)
    for index in range(n_bootstraps):
        session_draw = rng.integers(0, len(sessions), size=len(sessions))
        means = []
        for session_index in session_draw:
            values = grouped[str(sessions[session_index])]
            unit_draw = values[rng.integers(0, len(values), size=len(values))]
            means.append(float(np.mean(unit_draw)))
        samples[index] = float(np.mean(means))
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def population_summary(
    metrics: pd.DataFrame, n_bootstraps: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for (target, formula), frame in metrics.groupby(["target", "formula"], sort=True):
        point, low, high = session_balanced_bootstrap(
            frame, "heldout_cv_r2", n_bootstraps=n_bootstraps, rng=rng
        )
        rows.append(
            {
                "target": target,
                "formula": formula,
                "n_units": int(frame.rr100_index.nunique()),
                "n_sessions": int(frame.session.nunique()),
                "unit_mean_cv_r2": float(frame.heldout_cv_r2.mean()),
                "unit_median_cv_r2": float(frame.heldout_cv_r2.median()),
                "fraction_units_positive_cv_r2": float(np.mean(frame.heldout_cv_r2 > 0)),
                "session_balanced_mean_cv_r2": point,
                "session_cluster_ci_low": low,
                "session_cluster_ci_high": high,
                "unit_mean_within_trial_r": float(frame.heldout_within_trial_r.mean()),
            }
        )
    return pd.DataFrame(rows)


def increment_summary(
    differences: pd.DataFrame, unit_metadata: pd.DataFrame, n_bootstraps: int
) -> pd.DataFrame:
    merged = differences.merge(
        unit_metadata[["rr100_index", "session"]], on="rr100_index", validate="many_to_one"
    )
    columns = {
        "oriented_minus_radial_direct_f0_cv_r2": "oriented single predictor minus radial direct-F0",
        "nested_minus_radial_direct_f0_cv_r2": "nested orientation increment over radial direct-F0",
        "h2_minus_radial_direct_f0_cv_r2": "H2 minus radial direct-F0",
    }
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for target in sorted(merged.target.unique()):
        target_rows = merged.loc[merged.target.eq(target)]
        for column, label in columns.items():
            point, low, high = session_balanced_bootstrap(
                target_rows, column, n_bootstraps=n_bootstraps, rng=rng
            )
            rows.append(
                {
                    "target": target,
                    "comparison": label,
                    "value_column": column,
                    "n_units": int(len(target_rows)),
                    "n_sessions": int(target_rows.session.nunique()),
                    "unit_mean_delta_cv_r2": float(target_rows[column].mean()),
                    "unit_median_delta_cv_r2": float(target_rows[column].median()),
                    "fraction_units_positive": float(np.mean(target_rows[column] > 0)),
                    "session_balanced_mean_delta_cv_r2": point,
                    "session_cluster_ci_low": low,
                    "session_cluster_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def select_population_units(differences: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    recorded = differences.loc[differences.target.eq("recorded")].merge(
        metadata[["rr100_index", "session"]], on="rr100_index", validate="one_to_one"
    )
    selected = []
    used: set[int] = set()

    def add(role: str, column: str, direction: str) -> None:
        available = recorded.loc[~recorded.rr100_index.isin(used)]
        index = available[column].idxmax() if direction == "max" else available[column].idxmin()
        row = recorded.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = column
        row["selection_value"] = float(row[column])
        selected.append(row)
        used.add(int(row.rr100_index))

    add("largest orientation increment", "nested_minus_radial_direct_f0_cv_r2", "max")
    add("largest orientation failure", "nested_minus_radial_direct_f0_cv_r2", "min")
    add("largest H2 advantage", "h2_minus_radial_direct_f0_cv_r2", "max")
    add("largest oriented-single advantage", "oriented_minus_radial_direct_f0_cv_r2", "max")
    return pd.DataFrame(selected)


def plot_population(
    metrics: pd.DataFrame,
    increment: pd.DataFrame,
    out: Path,
    dpi: int,
) -> None:
    formulas = [
        "whole_crop_total_power",
        "rf_local_total_power",
        "rf_local_sf_tf_h2",
        "rf_local_radial_direct_f0",
        "rf_local_oriented_direct_f0",
        NESTED_FORMULA,
    ]
    labels = [
        "Whole-image total\ndynamic power",
        "Receptive-field-local\ntotal dynamic power",
        "Local power weighted by squared\nspatial- and temporal-frequency tuning",
        "Local power weighted by phase-averaged\nspatial- and temporal-frequency response",
        "Local power weighted by phase-averaged spatial-\nfrequency, orientation, and temporal-frequency response",
        "Orientation-collapsed model plus separately\nfitted orientation contribution",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)
    for axis, target, title in zip(
        axes[0],
        ["recorded", "full_twin"],
        ["Prediction of recorded neural firing rates", "Prediction of digital-twin firing rates"],
    ):
        subset = metrics.loc[metrics.target.eq(target)]
        x = np.arange(len(formulas))
        for unit, frame in subset.groupby("rr100_index", sort=True):
            lookup = frame.set_index("formula").heldout_cv_r2
            axis.plot(x, [lookup[value] for value in formulas], color="0.82", linewidth=0.6)
        means = [subset.loc[subset.formula.eq(value), "heldout_cv_r2"].mean() for value in formulas]
        axis.scatter(x, means, color="#0072B2", edgecolor="black", s=70, zorder=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.set(
            ylabel="Held-out fraction of firing-rate variance explained (R²)",
            title=title,
        )

    for axis, target, title in zip(
        axes[1],
        ["recorded", "full_twin"],
        [
            "Recorded firing rates: additional prediction from orientation tuning",
            "Digital-twin firing rates: additional prediction from orientation tuning",
        ],
    ):
        row = increment.loc[
            increment.target.eq(target)
            & increment.value_column.eq("nested_minus_radial_direct_f0_cv_r2")
        ].iloc[0]
        differences = metrics.loc[metrics.target.eq(target)].pivot(
            index="rr100_index", columns="formula", values="heldout_cv_r2"
        )
        values = differences[NESTED_FORMULA] - differences["rf_local_radial_direct_f0"]
        axis.hist(values, bins=18, color="#CC79A7", alpha=0.82)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axvline(row.session_balanced_mean_delta_cv_r2, color="#D55E00", linewidth=2)
        axis.set(
            xlabel=(
                "Change in held-out R² after adding orientation tuning to the\n"
                "phase-averaged spatial- and temporal-frequency response model"
            ),
            ylabel="Number of units",
            title=(
                f"{title}\nSession-balanced mean change in R² = "
                f"{row.session_balanced_mean_delta_cv_r2:.4f}; 95% interval "
                f"[{row.session_cluster_ci_low:.4f}, {row.session_cluster_ci_high:.4f}]"
            ),
        )
    figure.suptitle(
        "Can retinal image power, filtered by measured visual tuning, predict firing rates during recorded grating trials?\n"
        "61 visually responsive units from 15 recording sessions; entire experimental trials held out for evaluation\n"
        "Each light line is one unit; blue points are means across units; predictors are calibrated only on training trials",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite population checkpoint: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    response_rows = pd.read_csv(args.response_dir / "all_window_unit_predictions.csv")
    metadata = pd.read_csv(args.response_dir / "all_unit_rf_metadata.csv")
    coverage = pd.read_csv(args.response_dir / "session_coverage_audit.csv")
    sessions = sorted(metadata.session.unique())
    if args.max_sessions > 0:
        sessions = sessions[: int(args.max_sessions)]
        response_rows = response_rows.loc[response_rows.session.isin(sessions)].copy()
        metadata = metadata.loc[metadata.session.isin(sessions)].copy()
    radial_weights, oriented_weights, tuning_sf, tuning_tf = load_tuning(args.tuning_dir)

    all_joined = []
    all_predictions = []
    all_metrics = []
    all_fits = []
    session_audit = []
    for number, session in enumerate(sessions, start=1):
        unit_ids = metadata.loc[metadata.session.eq(session), "rr100_index"].to_numpy(int)
        missing_tuning = [
            int(unit) for unit in unit_ids
            if int(unit) not in radial_weights or int(unit) not in oriented_weights
        ]
        if missing_tuning:
            raise ValueError(f"{session}: missing orientation tuning for {missing_tuning}")
        apertures = load_session_apertures(args.response_dir, session)
        if set(unit_ids) != set(apertures):
            raise ValueError(f"{session}: metadata and aperture unit sets differ")
        print(f"[{number}/{len(sessions)}] {session}: {len(unit_ids)} units", flush=True)
        dset, local, _ = load_heldout_grating_dataset(args.dataset_config, session)
        window_metrics, payload, candidate_sf, candidate_tf, ppd = candidate_windows(
            dset, local, N_SCORE, 0, session=session
        )
        power_rows = build_session_power_rows(
            window_metrics,
            payload,
            unit_ids,
            apertures,
            radial_weights,
            oriented_weights,
            tuning_sf,
            tuning_tf,
            candidate_sf,
            candidate_tf,
            float(ppd),
        )
        response = response_rows.loc[response_rows.session.eq(session)].copy()
        keys = ["window_index", "trial_index", "start_index_120hz", "rr100_index"]
        joined = response.merge(power_rows, on=keys, how="left", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError(f"{session}: at least one response row lacks a power row")
        joined = joined.drop(columns="_merge")
        predictions, metrics, fits = score_formulas(joined)
        predictions.insert(0, "session", session)
        metrics.insert(0, "session", session)
        fits.insert(0, "session", session)
        all_joined.append(joined)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_fits.append(fits)
        session_audit.append(
            {
                "session": session,
                "n_units": int(len(unit_ids)),
                "n_candidate_windows": int(len(window_metrics)),
                "n_power_rows": int(len(power_rows)),
                "n_lag_valid_response_rows": int(len(response)),
                "n_joined_rows": int(len(joined)),
                "all_response_rows_matched": True,
            }
        )
        del dset, payload, power_rows, joined

    joined_rows = pd.concat(all_joined, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    fits = pd.concat(all_fits, ignore_index=True)
    differences = formula_differences(metrics)
    summary = summarize_metrics(metrics)
    population = population_summary(metrics, int(args.n_bootstraps))
    increment = increment_summary(differences, metadata, int(args.n_bootstraps))
    selected = select_population_units(differences, metadata)
    audit = pd.DataFrame(session_audit)

    joined_rows.to_csv(args.out_dir / "all_joined_window_unit_rows.csv", index=False)
    predictions.to_csv(args.out_dir / "all_heldout_formula_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "all_unit_formula_metrics.csv", index=False)
    fits.to_csv(args.out_dir / "all_formula_fold_fits.csv", index=False)
    differences.to_csv(args.out_dir / "all_unit_formula_differences.csv", index=False)
    summary.to_csv(args.out_dir / "unweighted_formula_summary.csv", index=False)
    population.to_csv(args.out_dir / "population_formula_summary.csv", index=False)
    increment.to_csv(args.out_dir / "population_increment_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_population_units.csv", index=False)
    audit.to_csv(args.out_dir / "session_coverage_audit.csv", index=False)
    figure_base = args.out_dir / "recorded_grating_power_formula_population"
    plot_population(metrics, increment, figure_base, int(args.dpi))

    recorded_increment = increment.loc[
        increment.target.eq("recorded")
        & increment.value_column.eq("nested_minus_radial_direct_f0_cv_r2")
    ].iloc[0]
    twin_increment = increment.loc[
        increment.target.eq("full_twin")
        & increment.value_column.eq("nested_minus_radial_direct_f0_cv_r2")
    ].iloc[0]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_power_formula_population",
        "status": "population_checkpoint_complete",
        "scope": {
            "n_units": int(metadata.rr100_index.nunique()),
            "n_sessions": int(metadata.session.nunique()),
            "n_lag_valid_window_unit_rows": int(len(joined_rows)),
            "all_declared_sessions_complete": bool(
                len(sessions) == len(coverage) and audit.all_response_rows_matched.all()
            ),
        },
        "contracts": {
            "spectra": "exact held-out 40-frame gaze-cropped grating movies with frozen unit RF apertures",
            "response_rows": "frozen lag-valid recorded/full-twin rows from the all-session RF-local checkpoint",
            "folds": "same complete-trial folds frozen in the response rows",
            "single_predictors": "training-fold intercept plus nonnegative slope",
            "nested_orientation": "nonnegative radial direct-F0 slope plus signed orientation correction",
            "uncertainty": "session-balanced hierarchical bootstrap of sessions and units within session",
        },
        "primary_orientation_increment": {
            "recorded_session_balanced_mean_delta_cv_r2": float(recorded_increment.session_balanced_mean_delta_cv_r2),
            "recorded_cluster_ci": [
                float(recorded_increment.session_cluster_ci_low),
                float(recorded_increment.session_cluster_ci_high),
            ],
            "recorded_fraction_units_positive": float(recorded_increment.fraction_units_positive),
            "full_twin_session_balanced_mean_delta_cv_r2": float(twin_increment.session_balanced_mean_delta_cv_r2),
            "full_twin_cluster_ci": [
                float(twin_increment.session_cluster_ci_low),
                float(twin_increment.session_cluster_ci_high),
            ],
            "full_twin_fraction_units_positive": float(twin_increment.fraction_units_positive),
        },
        "artifacts": {
            "figure_png": figure_base.with_suffix(".png").name,
            "figure_pdf": figure_base.with_suffix(".pdf").name,
            "population_formula_summary": "population_formula_summary.csv",
            "population_increment_summary": "population_increment_summary.csv",
            "unit_metrics": "all_unit_formula_metrics.csv",
            "unit_differences": "all_unit_formula_differences.csv",
            "selected_units": "selected_population_units.csv",
            "session_coverage": "session_coverage_audit.csv",
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(population.to_string(index=False), flush=True)
    print(increment.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

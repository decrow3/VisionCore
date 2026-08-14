#!/usr/bin/env python3
"""Compare grating power-routing formulas against recorded and twin responses.

This one-session checkpoint joins frozen lag-aligned response windows with
frozen RF-local radial and orientation-resolved spectra. Every formula uses the
same window-unit rows and exact preassigned trial folds. This is a diagnostic,
not a population conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
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

from declan.fig4_active_sensing.make_rr100_recorded_grating_three_way_response_checkpoint import (
    cv_r2,
    grouped_cv_line,
    safe_pearson,
    trial_demean,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSE_DIR = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
)
DEFAULT_ORIENTATION_DIR = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_oriented_power_input_checkpoint_v2"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_recorded_grating_power_formula_response_checkpoint_v3"
)
KEYS = ["window_index", "trial_index", "start_index_120hz", "rr100_index"]
TARGETS = {
    "recorded": "recorded_mean_rate_hz",
    "full_twin": "full_twin_mean_rate_hz",
}
SINGLE_FORMULAS = {
    "whole_crop_total_power": "whole_crop_power_amplitude",
    "whole_crop_sf_tf_h2": "whole_crop_routed_amplitude",
    "rf_local_total_power": "rf_local_total_power_amplitude",
    "rf_local_sf_tf_h2": "gain_weighted_local_routed_amplitude",
    "rf_local_radial_direct_f0": "radial_direct_f0_drive",
    "rf_local_oriented_direct_f0": "oriented_direct_f0_drive",
}
NESTED_FORMULA = "rf_local_radial_plus_orientation_delta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--orientation-dir", type=Path, default=DEFAULT_ORIENTATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def grouped_cv_radial_plus_signed_delta(
    predictors: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit radial plus signed orientation delta with radial slope >= 0.

    Predictors are standardized on each training fold. If unconstrained OLS
    gives a negative radial coefficient, the convex constrained optimum lies on
    the radial=0 boundary, where the orientation-delta coefficient is refit.
    """
    x = np.asarray(predictors, dtype=float)
    y = np.asarray(target, dtype=float)
    fold_values = np.asarray(folds, dtype=int)
    if x.ndim != 2 or x.shape[0] != len(y) or len(y) != len(fold_values):
        raise ValueError("Predictor, target, and fold dimensions do not align")
    prediction = np.full(len(y), np.nan, dtype=float)
    baseline = np.full(len(y), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold in np.unique(fold_values):
        train = fold_values != fold
        test = fold_values == fold
        center = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        if np.any(scale <= 1e-15):
            raise ValueError(f"At least one predictor is constant in training fold {fold}")
        train_z = (x[train] - center) / scale
        test_z = (x[test] - center) / scale
        design = np.column_stack([np.ones(np.count_nonzero(train)), train_z])
        coefficients, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        radial_constraint_active = bool(coefficients[1] < 0)
        if radial_constraint_active:
            boundary_design = design[:, [0, 2]]
            boundary_coefficients, *_ = np.linalg.lstsq(
                boundary_design, y[train], rcond=None
            )
            coefficients = np.asarray(
                [boundary_coefficients[0], 0.0, boundary_coefficients[1]],
                dtype=float,
            )
        prediction[test] = np.column_stack(
            [np.ones(np.count_nonzero(test)), test_z]
        ) @ coefficients
        baseline[test] = float(y[train].mean())
        row: dict[str, object] = {
            "fold": int(fold),
            "n_train": int(np.count_nonzero(train)),
            "n_test": int(np.count_nonzero(test)),
            "intercept_hz": float(coefficients[0]),
            "radial_nonnegative_constraint_active": radial_constraint_active,
        }
        for index, value in enumerate(coefficients[1:]):
            row[f"standardized_coefficient_{index}"] = float(value)
        rows.append(row)
    return prediction, baseline, pd.DataFrame(rows)


def load_joined_rows(
    response_dir: Path, orientation_dir: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    response_path = response_dir / "window_unit_three_way_predictions.csv"
    orientation_path = orientation_dir / "window_unit_oriented_power_metrics.csv"
    arrays_path = orientation_dir / "rf_local_oriented_power_arrays.npz"
    response = pd.read_csv(response_path)
    orientation = pd.read_csv(orientation_path)
    if response.duplicated(KEYS).any() or orientation.duplicated(KEYS).any():
        raise ValueError("Window-unit keys are not unique")
    with np.load(arrays_path, allow_pickle=False) as archive:
        radial_power = np.asarray(archive["radial_power"], dtype=np.float64)
        array_row = np.asarray(archive["array_row"], dtype=int)
    if radial_power.shape[0] != len(orientation):
        raise ValueError("Orientation spectrum rows do not match the metrics table")
    if not np.array_equal(array_row, orientation.array_row.to_numpy(int)):
        raise ValueError("Orientation array rows do not match the metrics table")
    orientation = orientation.copy()
    orientation["rf_local_total_power_amplitude"] = np.sqrt(
        np.maximum(radial_power.sum(axis=(1, 2)), 0.0)
    )
    joined = response.merge(
        orientation[
            KEYS
            + [
                "array_row",
                "radial_direct_f0_drive",
                "oriented_direct_f0_drive",
                "orientation_delta_drive",
                "rf_local_total_power_amplitude",
            ]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), KEYS]
        raise ValueError(f"Response rows are missing orientation power:\n{missing.head()}")
    joined = joined.drop(columns="_merge")
    audit = {
        "response_rows": int(len(response)),
        "orientation_rows": int(len(orientation)),
        "joined_rows": int(len(joined)),
        "orientation_rows_excluded_by_response_lag_or_dfs": int(len(orientation) - len(joined)),
        "all_response_rows_matched_exactly_once": True,
    }
    return joined, audit


def score_formulas(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    fold_fits: list[pd.DataFrame] = []
    for unit, frame in joined.groupby("rr100_index", sort=True):
        frame = frame.sort_values("start_index_120hz").reset_index(drop=True)
        folds = frame.fold.to_numpy(int)
        trials = frame.trial_index.to_numpy(int)
        for target_name, target_column in TARGETS.items():
            observed = frame[target_column].to_numpy(float)
            baseline_by_formula: dict[str, np.ndarray] = {}
            formula_predictions: dict[str, np.ndarray] = {}
            for formula, predictor_column in SINGLE_FORMULAS.items():
                predictor = frame[predictor_column].to_numpy(float)
                predicted, baseline, fits = grouped_cv_line(
                    predictor, observed, folds, nonnegative=True
                )
                formula_predictions[formula] = predicted
                baseline_by_formula[formula] = baseline
                fits.insert(0, "rr100_index", int(unit))
                fits.insert(1, "target", target_name)
                fits.insert(2, "formula", formula)
                fits.insert(3, "fit_constraint", "nonnegative_single_slope")
                fold_fits.append(fits)

            nested_x = frame[
                ["radial_direct_f0_drive", "orientation_delta_drive"]
            ].to_numpy(float)
            nested_prediction, nested_baseline, nested_fits = grouped_cv_radial_plus_signed_delta(
                nested_x, observed, folds
            )
            formula_predictions[NESTED_FORMULA] = nested_prediction
            baseline_by_formula[NESTED_FORMULA] = nested_baseline
            nested_fits.insert(0, "rr100_index", int(unit))
            nested_fits.insert(1, "target", target_name)
            nested_fits.insert(2, "formula", NESTED_FORMULA)
            nested_fits.insert(3, "fit_constraint", "nonnegative_radial_plus_signed_orientation_delta")
            fold_fits.append(nested_fits)

            observed_within = trial_demean(observed, trials)
            for formula, predicted in formula_predictions.items():
                baseline = baseline_by_formula[formula]
                predicted_within = trial_demean(predicted, trials)
                metrics.append(
                    {
                        "rr100_index": int(unit),
                        "target": target_name,
                        "formula": formula,
                        "n_windows": int(len(frame)),
                        "n_trials": int(np.unique(trials).size),
                        "heldout_cv_r2": cv_r2(observed, predicted, baseline),
                        "heldout_prediction_r": safe_pearson(predicted, observed),
                        "heldout_within_trial_r": safe_pearson(
                            predicted_within, observed_within
                        ),
                    }
                )
                prediction_frame = frame[KEYS + ["fold"]].copy()
                prediction_frame["target"] = target_name
                prediction_frame["formula"] = formula
                prediction_frame["observed_rate_hz"] = observed
                prediction_frame["predicted_rate_hz"] = predicted
                prediction_frame["training_fold_baseline_rate_hz"] = baseline
                predictions.append(prediction_frame)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(fold_fits, ignore_index=True),
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["target", "formula"], as_index=False)
        .agg(
            n_units=("rr100_index", "size"),
            unit_mean_cv_r2=("heldout_cv_r2", "mean"),
            unit_median_cv_r2=("heldout_cv_r2", "median"),
            unit_mean_within_trial_r=("heldout_within_trial_r", "mean"),
            unit_median_within_trial_r=("heldout_within_trial_r", "median"),
        )
        .sort_values(["target", "unit_mean_cv_r2"], ascending=[True, False])
    )


def formula_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index=["rr100_index", "target"], columns="formula")
    result = wide["heldout_cv_r2"].reset_index()
    result["oriented_minus_radial_direct_f0_cv_r2"] = (
        result["rf_local_oriented_direct_f0"]
        - result["rf_local_radial_direct_f0"]
    )
    result["nested_minus_radial_direct_f0_cv_r2"] = (
        result[NESTED_FORMULA] - result["rf_local_radial_direct_f0"]
    )
    result["h2_minus_radial_direct_f0_cv_r2"] = (
        result["rf_local_sf_tf_h2"] - result["rf_local_radial_direct_f0"]
    )
    return result


def select_units(differences: pd.DataFrame) -> pd.DataFrame:
    recorded = differences.loc[differences.target.eq("recorded")].copy()
    selections: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, column: str, direction: str) -> None:
        available = recorded.loc[~recorded.rr100_index.isin(used)]
        index = available[column].idxmax() if direction == "max" else available[column].idxmin()
        row = recorded.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = column
        row["selection_value"] = float(row[column])
        selections.append(row)
        used.add(int(row.rr100_index))

    add("largest orientation increment", "nested_minus_radial_direct_f0_cv_r2", "max")
    add("largest orientation failure", "nested_minus_radial_direct_f0_cv_r2", "min")
    add("largest H2 advantage", "h2_minus_radial_direct_f0_cv_r2", "max")
    return pd.DataFrame(selections)


def plot_checkpoint(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    out: Path,
    dpi: int,
) -> None:
    reader_facing_roles = {
        "largest orientation increment": (
            "largest held-out improvement after adding measured orientation tuning"
        ),
        "largest orientation failure": (
            "largest held-out decrease after adding measured orientation tuning"
        ),
        "largest H2 advantage": (
            "largest advantage for squared spatial- and temporal-frequency tuning"
        ),
    }
    formulas = [
        "whole_crop_total_power",
        "rf_local_total_power",
        "rf_local_sf_tf_h2",
        "rf_local_radial_direct_f0",
        "rf_local_oriented_direct_f0",
        NESTED_FORMULA,
    ]
    labels = {
        "whole_crop_total_power": "Whole-image total\ndynamic power",
        "rf_local_total_power": "Receptive-field-local\ntotal dynamic power",
        "rf_local_sf_tf_h2": "Local power weighted by squared\nspatial- and temporal-frequency tuning",
        "rf_local_radial_direct_f0": "Local power weighted by phase-averaged\nspatial- and temporal-frequency response",
        "rf_local_oriented_direct_f0": "Local power weighted by phase-averaged spatial-\nfrequency, orientation, and temporal-frequency response",
        NESTED_FORMULA: "Orientation-collapsed model plus separately\nfitted orientation contribution",
    }
    colors = ["#999999", "#56B4E9", "#E69F00", "#0072B2", "#D55E00", "#CC79A7"]
    figure = plt.figure(figsize=(21, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    for column, (target, title) in enumerate(
        [
            ("recorded", "Prediction of recorded neural firing rates"),
            ("full_twin", "Prediction of digital-twin firing rates"),
        ]
    ):
        axis = figure.add_subplot(grid[0, column])
        subset = metrics.loc[metrics.target.eq(target)].set_index(
            ["rr100_index", "formula"]
        )
        units = sorted(metrics.rr100_index.unique())
        x = np.arange(len(formulas))
        for unit in units:
            values = [subset.loc[(unit, formula), "heldout_cv_r2"] for formula in formulas]
            axis.plot(x, values, color="0.72", linewidth=1, marker="o", markersize=3)
        means = [
            metrics.loc[
                metrics.target.eq(target) & metrics.formula.eq(formula),
                "heldout_cv_r2",
            ].mean()
            for formula in formulas
        ]
        axis.scatter(x, means, color=colors, edgecolor="black", s=75, zorder=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[value] for value in formulas], rotation=15, ha="right")
        axis.set(
            ylabel="Held-out fraction of firing-rate variance explained (R²)",
            title=title,
        )

    delta_axis = figure.add_subplot(grid[0, 2])
    recorded = metrics.loc[metrics.target.eq("recorded")].pivot(
        index="rr100_index", columns="formula", values="heldout_cv_r2"
    )
    delta = recorded[NESTED_FORMULA] - recorded["rf_local_radial_direct_f0"]
    delta_axis.bar(np.arange(len(delta)), delta, color=np.where(delta >= 0, "#009E73", "#D55E00"))
    delta_axis.axhline(0, color="black", linewidth=0.8)
    delta_axis.set_xticks(np.arange(len(delta)), [str(value) for value in delta.index])
    delta_axis.set(
        xlabel="Unit identifier in the analyzed cohort",
        ylabel="Change in held-out R²",
        title=(
            "Does measured orientation tuning improve prediction beyond\n"
            "phase-averaged spatial- and temporal-frequency responses?"
        ),
    )

    for column, unit in enumerate(selected.rr100_index.astype(int).tolist()[:3]):
        axis = figure.add_subplot(grid[1, column])
        subset = predictions.loc[
            predictions.rr100_index.eq(unit)
            & predictions.target.eq("recorded")
            & predictions.formula.isin(
                ["rf_local_radial_direct_f0", NESTED_FORMULA]
            )
        ].sort_values("start_index_120hz")
        observed = subset.drop_duplicates(KEYS).sort_values("start_index_120hz")
        axis.plot(
            observed.start_index_120hz,
            observed.observed_rate_hz,
            color="black",
            linewidth=1.2,
            label="recorded",
        )
        for formula, color in [
            ("rf_local_radial_direct_f0", "#0072B2"),
            (NESTED_FORMULA, "#CC79A7"),
        ]:
            rows = subset.loc[subset.formula.eq(formula)]
            axis.plot(
                rows.start_index_120hz,
                rows.predicted_rate_hz,
                color=color,
                linewidth=1,
                label=labels[formula].replace("\n", " "),
            )
        role = selected.loc[selected.rr100_index.eq(unit), "selection_role"].iloc[0]
        role = reader_facing_roles.get(str(role), str(role))
        axis.set(
            xlabel="Time in the held-out grating recording (120-Hz sample index)",
            ylabel="Mean firing rate during each 333-ms window (spikes/s)",
            title=f"Unit {unit}: {role}",
        )
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "How well do retinal-power models predict firing rates during recorded grating trials?\n"
        "Five units from one recording session; all models use identical receptive-field-local spectra and response windows;\n"
        "entire experimental trials are held out for evaluation, and calibration uses training trials only",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    joined, join_audit = load_joined_rows(args.response_dir, args.orientation_dir)
    predictions, metrics, fold_fits = score_formulas(joined)
    summary = summarize_metrics(metrics)
    differences = formula_differences(metrics)
    selected = select_units(differences)

    joined.to_csv(args.out_dir / "joined_window_unit_inputs_and_responses.csv", index=False)
    predictions.to_csv(args.out_dir / "heldout_formula_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "unit_formula_metrics.csv", index=False)
    fold_fits.to_csv(args.out_dir / "formula_fold_fits.csv", index=False)
    summary.to_csv(args.out_dir / "formula_summary.csv", index=False)
    differences.to_csv(args.out_dir / "unit_formula_differences.csv", index=False)
    selected.to_csv(args.out_dir / "selected_units.csv", index=False)
    figure_base = args.out_dir / "recorded_grating_power_formula_response_checkpoint"
    plot_checkpoint(metrics, predictions, selected, figure_base, int(args.dpi))

    recorded_differences = differences.loc[differences.target.eq("recorded")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_power_formula_response_checkpoint",
        "status": "one_session_human_checkpoint_complete",
        "supersedes": (
            "rr100_recorded_grating_power_formula_response_checkpoint_v1, whose nested "
            "model did not impose the radial comparator's nonnegative-slope constraint"
        ),
        "scope": {
            "session": "Logan_2020-02-29",
            "n_units": int(joined.rr100_index.nunique()),
            "n_trials": int(joined.trial_index.nunique()),
            "n_joined_window_unit_rows": int(len(joined)),
        },
        "join_audit": join_audit,
        "contracts": {
            "cross_validation": "exact trial folds frozen by the RF-local three-way response checkpoint",
            "single_predictor_fit": "training-fold intercept plus nonnegative slope",
            "nested_orientation_fit": (
                "training-fold standardized radial direct-F0 drive constrained nonnegative, "
                "plus a signed orientation-minus-radial correction"
            ),
            "targets": "recorded rate and full-twin rate scored separately",
            "claim_scope": "one-session diagnostic; not a population conclusion",
        },
        "recorded_orientation_increment": {
            "mean_nested_minus_radial_direct_f0_cv_r2": float(
                recorded_differences.nested_minus_radial_direct_f0_cv_r2.mean()
            ),
            "median_nested_minus_radial_direct_f0_cv_r2": float(
                recorded_differences.nested_minus_radial_direct_f0_cv_r2.median()
            ),
            "fraction_units_positive": float(
                np.mean(recorded_differences.nested_minus_radial_direct_f0_cv_r2 > 0)
            ),
        },
        "inputs": {
            "response_predictions": file_identity(
                args.response_dir / "window_unit_three_way_predictions.csv"
            ),
            "response_manifest": file_identity(args.response_dir / "manifest.json"),
            "orientation_metrics": file_identity(
                args.orientation_dir / "window_unit_oriented_power_metrics.csv"
            ),
            "orientation_arrays": file_identity(
                args.orientation_dir / "rf_local_oriented_power_arrays.npz"
            ),
            "orientation_manifest": file_identity(args.orientation_dir / "manifest.json"),
        },
        "artifacts": {
            "figure_png": figure_base.with_suffix(".png").name,
            "figure_pdf": figure_base.with_suffix(".pdf").name,
            "joined_rows": "joined_window_unit_inputs_and_responses.csv",
            "predictions": "heldout_formula_predictions.csv",
            "unit_metrics": "unit_formula_metrics.csv",
            "formula_summary": "formula_summary.csv",
            "unit_differences": "unit_formula_differences.csv",
            "selected_units": "selected_units.csv",
            "fold_fits": "formula_fold_fits.csv",
        },
        "next_checkpoint": (
            "inspect the selected units and formula comparison before deciding whether to "
            "scale the orientation-response test to all eligible sessions"
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

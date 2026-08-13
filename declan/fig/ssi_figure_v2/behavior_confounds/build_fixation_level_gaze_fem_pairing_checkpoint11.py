#!/usr/bin/env python3
"""Checkpoint 11: held-out fixation-position to drift-covariance pairing test.

One row represents one BackImage fixation/trial. Models are fit separately by
animal and evaluated on sessions withheld from fitting. Image structure and
timing enter before gaze position. Null predictions retain each held-out
session's gaze support but shuffle which fixation receives which gaze position.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "fixation_level_gaze_fem_pairing_checkpoint11_v1"
)
SUBJECTS = ("Allen", "Logan")
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
OUTCOMES = ("log_scale", "screen_a", "screen_b")
OUTCOME_LABELS = {
    "log_scale": "log total drift scale",
    "screen_a": "horizontal allocation",
    "screen_b": "oblique covariance",
}
IMAGE_FEATURES = (
    "coherence", "log_gradient", "background_fraction", "edge_x", "edge_y",
    "late_fraction", "log_time",
)
RADIAL_FEATURES = ("eccentricity", "eccentricity_sq")
FULL_GAZE_FEATURES = ("gaze_x", "gaze_y", "gaze_x_sq", "gaze_xy", "gaze_y_sq")
MODEL_FEATURES = {
    "image_only": IMAGE_FEATURES,
    "image_plus_radial": IMAGE_FEATURES + RADIAL_FEATURES,
    "image_plus_full_position": IMAGE_FEATURES + FULL_GAZE_FEATURES,
}
MODEL_LABELS = {
    "image_only": "image + timing",
    "image_plus_radial": "+ eccentricity",
    "image_plus_full_position": "+ full gaze position",
}
ALPHAS = np.asarray([0.1, 1.0, 10.0, 100.0])
N_NULL = 1000
N_BOOTSTRAP = 5000
SEED = 20260810
MAP_EDGES = np.arange(-12.0, 12.01, 4.0)
MIN_MAP_TRIALS = 5
GRID = "#D8DDE3"
INK = "#202428"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_trial_table() -> pd.DataFrame:
    values = pd.read_csv(SOURCE).copy()
    trace = values["cov_xx_deg2"].to_numpy(dtype=float) + values["cov_yy_deg2"].to_numpy(dtype=float)
    coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
    edge_angle = np.radians(values["image_edge_axis_deg"].to_numpy(dtype=float))
    derived = pd.DataFrame(
        {
            "log_scale_window": np.log(60.0 * np.sqrt(np.maximum(trace, 1e-12))),
            "screen_a_window": (
                values["cov_xx_deg2"].to_numpy(dtype=float)
                - values["cov_yy_deg2"].to_numpy(dtype=float)
            ) / trace,
            "screen_b_window": 2.0 * values["cov_xy_deg2"].to_numpy(dtype=float) / trace,
            "log_gradient_window": np.log1p(values["image_gradient_energy"].to_numpy(dtype=float)),
            "edge_x_window": coherence * np.cos(2.0 * edge_angle),
            "edge_y_window": coherence * np.sin(2.0 * edge_angle),
            "late_window": values["phase"].astype(str).eq("late_fixation").astype(float),
            "log_time_window": np.log1p(values["samples_since_event"].to_numpy(dtype=float)),
        }
    )
    values = pd.concat([values.reset_index(drop=True), derived], axis=1)
    trials = values.groupby(["subject", "session", "trial_idx"], as_index=False).agg(
        gaze_x=("mean_x_deg", "median"),
        gaze_y=("mean_y_deg", "median"),
        log_scale=("log_scale_window", "median"),
        screen_a=("screen_a_window", "median"),
        screen_b=("screen_b_window", "median"),
        coherence=("image_orientation_coherence", "median"),
        log_gradient=("log_gradient_window", "median"),
        background_fraction=("image_patch_fraction_background", "median"),
        edge_x=("edge_x_window", "median"),
        edge_y=("edge_y_window", "median"),
        late_fraction=("late_window", "mean"),
        log_time=("log_time_window", "median"),
        n_windows=("trial_idx", "size"),
    )
    trials["eccentricity"] = np.hypot(trials["gaze_x"], trials["gaze_y"])
    trials["eccentricity_sq"] = trials["eccentricity"] ** 2
    trials["gaze_x_sq"] = trials["gaze_x"] ** 2
    trials["gaze_xy"] = trials["gaze_x"] * trials["gaze_y"]
    trials["gaze_y_sq"] = trials["gaze_y"] ** 2
    required = list(OUTCOMES) + list(IMAGE_FEATURES) + list(FULL_GAZE_FEATURES) + list(RADIAL_FEATURES)
    ok = trials["subject"].isin(SUBJECTS)
    for column in required:
        ok &= np.isfinite(trials[column].to_numpy(dtype=float))
    return trials.loc[ok].copy().reset_index(drop=True)


def choose_alpha(train: pd.DataFrame, features: tuple[str, ...], outcome: str) -> float:
    groups = train["session"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    losses = np.zeros(len(ALPHAS), dtype=float)
    counts = np.zeros(len(ALPHAS), dtype=int)
    x_all = train.loc[:, features].to_numpy(dtype=float)
    y_all = train[outcome].to_numpy(dtype=float)
    for train_index, validation_index in splitter.split(x_all, y_all, groups):
        x_scaler = StandardScaler().fit(x_all[train_index])
        y_mean = float(np.mean(y_all[train_index]))
        y_scale = float(np.std(y_all[train_index])) or 1.0
        x_train = x_scaler.transform(x_all[train_index])
        x_validation = x_scaler.transform(x_all[validation_index])
        y_train = (y_all[train_index] - y_mean) / y_scale
        y_validation = (y_all[validation_index] - y_mean) / y_scale
        for alpha_index, alpha in enumerate(ALPHAS):
            prediction = Ridge(alpha=float(alpha)).fit(x_train, y_train).predict(x_validation)
            losses[alpha_index] += float(np.sum((y_validation - prediction) ** 2))
            counts[alpha_index] += len(validation_index)
    return float(ALPHAS[int(np.argmin(losses / counts))])


def fit_outer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    outcome: str,
) -> dict[str, object]:
    alpha = choose_alpha(train, features, outcome)
    x_scaler = StandardScaler().fit(train.loc[:, features].to_numpy(dtype=float))
    y_train_raw = train[outcome].to_numpy(dtype=float)
    y_mean = float(np.mean(y_train_raw))
    y_scale = float(np.std(y_train_raw)) or 1.0
    model = Ridge(alpha=alpha).fit(
        x_scaler.transform(train.loc[:, features].to_numpy(dtype=float)),
        (y_train_raw - y_mean) / y_scale,
    )
    x_test_scaled = x_scaler.transform(test.loc[:, features].to_numpy(dtype=float))
    prediction = y_mean + y_scale * model.predict(x_test_scaled)
    return {
        "alpha": alpha,
        "features": features,
        "x_scaler": x_scaler,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "model": model,
        "prediction": prediction,
        "x_test_scaled": x_test_scaled,
    }


def feature_contribution(fit: dict[str, object], feature_names: tuple[str, ...]) -> np.ndarray:
    features = tuple(fit["features"])
    indices = [features.index(feature) for feature in feature_names]
    coefficients = np.asarray(fit["model"].coef_, dtype=float)[indices]
    x_scaled = np.asarray(fit["x_test_scaled"], dtype=float)[:, indices]
    return float(fit["y_scale"]) * (x_scaled @ coefficients)


def contribution_for_rows(
    fit: dict[str, object], rows: pd.DataFrame, feature_names: tuple[str, ...]
) -> np.ndarray:
    features = tuple(fit["features"])
    indices = [features.index(feature) for feature in feature_names]
    scaler: StandardScaler = fit["x_scaler"]
    raw = rows.loc[:, feature_names].to_numpy(dtype=float)
    means = scaler.mean_[indices]
    scales = scaler.scale_[indices]
    standardized = (raw - means) / scales
    coefficients = np.asarray(fit["model"].coef_, dtype=float)[indices]
    return float(fit["y_scale"]) * (standardized @ coefficients)


def eccentricity_bins(eccentricity: np.ndarray) -> np.ndarray:
    return np.asarray(
        pd.qcut(
            eccentricity, q=min(5, len(np.unique(eccentricity))), labels=False, duplicates="drop"
        ),
        dtype=int,
    )


def run_cross_validation(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    prediction_rows = []
    null_rows = []
    integration_rows = []
    alpha_rows = []
    for subject in SUBJECTS:
        subject_values = trials[trials["subject"].eq(subject)].copy()
        for session, test in subject_values.groupby("session", sort=True):
            train = subject_values[~subject_values["session"].eq(session)].copy()
            test = test.copy().reset_index(drop=True)
            fold_predictions: dict[tuple[str, str], np.ndarray] = {}
            fold_fits: dict[tuple[str, str], dict[str, object]] = {}
            for outcome in OUTCOMES:
                for model_name, features in MODEL_FEATURES.items():
                    fit = fit_outer(train, test, features, outcome)
                    fold_predictions[(outcome, model_name)] = np.asarray(fit["prediction"], dtype=float)
                    fold_fits[(outcome, model_name)] = fit
                    alpha_rows.append(
                        {
                            "subject": subject,
                            "held_out_session": session,
                            "outcome": outcome,
                            "model": model_name,
                            "selected_alpha": fit["alpha"],
                        }
                    )
            for row_index, row in test.iterrows():
                record = row.to_dict()
                for outcome in OUTCOMES:
                    for model_name in MODEL_FEATURES:
                        record[f"pred_{outcome}_{model_name}"] = fold_predictions[(outcome, model_name)][row_index]
                prediction_rows.append(record)

            bins = eccentricity_bins(test["eccentricity"].to_numpy(dtype=float))
            for outcome in OUTCOMES:
                observed = test[outcome].to_numpy(dtype=float)
                for model_name, gaze_features, null_name in (
                    ("image_plus_radial", RADIAL_FEATURES, "radial_unrestricted"),
                    ("image_plus_full_position", FULL_GAZE_FEATURES, "full_unrestricted"),
                    ("image_plus_full_position", FULL_GAZE_FEATURES, "full_eccentricity_matched"),
                ):
                    fit = fold_fits[(outcome, model_name)]
                    prediction = fold_predictions[(outcome, model_name)]
                    gaze_contribution = feature_contribution(fit, gaze_features)
                    base_prediction = prediction - gaze_contribution
                    real_mse = float(np.mean((observed - prediction) ** 2))
                    for null_index in range(N_NULL):
                        if null_name == "full_eccentricity_matched":
                            permutation = np.arange(len(test))
                            for bin_index in np.unique(bins):
                                positions = np.flatnonzero(bins == bin_index)
                                permutation[positions] = rng.permutation(positions)
                        else:
                            permutation = rng.permutation(len(test))
                        null_prediction = base_prediction + gaze_contribution[permutation]
                        null_rows.append(
                            {
                                "subject": subject,
                                "session": session,
                                "outcome": outcome,
                                "null": null_name,
                                "null_index": null_index,
                                "real_mse": real_mse,
                                "null_mse": float(np.mean((observed - null_prediction) ** 2)),
                            }
                        )

                full_fit = fold_fits[(outcome, "image_plus_full_position")]
                full_prediction = fold_predictions[(outcome, "image_plus_full_position")]
                own_gaze_contribution = feature_contribution(full_fit, FULL_GAZE_FEATURES)
                image_base_mean = float(np.mean(full_prediction - own_gaze_contribution))
                observed_mean = float(np.mean(observed))
                for donor_session, donor in subject_values.groupby("session", sort=True):
                    donor_contribution = contribution_for_rows(full_fit, donor, FULL_GAZE_FEATURES)
                    integrated_prediction = image_base_mean + float(np.mean(donor_contribution))
                    integration_rows.append(
                        {
                            "subject": subject,
                            "target_session": session,
                            "donor_session": donor_session,
                            "is_actual_gaze_distribution": donor_session == session,
                            "outcome": outcome,
                            "observed_session_mean": observed_mean,
                            "predicted_session_mean": integrated_prediction,
                            "squared_error": (observed_mean - integrated_prediction) ** 2,
                        }
                    )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(null_rows),
        pd.DataFrame(integration_rows),
        pd.DataFrame(alpha_rows),
    )


def cross_validated_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for model_name in MODEL_FEATURES:
            for subject in SUBJECTS:
                block = predictions[predictions["subject"].eq(subject)]
                observed = block[outcome].to_numpy(dtype=float)
                predicted = block[f"pred_{outcome}_{model_name}"].to_numpy(dtype=float)
                rows.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "scope": subject,
                        "n_trials": len(block),
                        "r2": r2_score(observed, predicted),
                        "pearson_r": pearsonr(observed, predicted).statistic,
                    }
                )
            subject_rows = rows[-len(SUBJECTS):]
            rows.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "scope": "equal-animal",
                    "n_trials": sum(row["n_trials"] for row in subject_rows),
                    "r2": float(np.mean([row["r2"] for row in subject_rows])),
                    "pearson_r": float(np.mean([row["pearson_r"] for row in subject_rows])),
                }
            )
    return pd.DataFrame(rows)


def summarize_pairing(nulls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_null = (
        nulls.groupby(["subject", "session", "outcome", "null"], as_index=False)
        .agg(real_mse=("real_mse", "first"), median_null_mse=("null_mse", "median"))
    )
    session_null["fractional_error_reduction"] = (
        session_null["median_null_mse"] - session_null["real_mse"]
    ) / session_null["median_null_mse"]
    rng = np.random.default_rng(SEED + 50)
    rows = []
    for (outcome, null_name), block in session_null.groupby(["outcome", "null"], sort=True):
        subject_points = {
            subject: float(np.median(block[block["subject"].eq(subject)]["fractional_error_reduction"]))
            for subject in SUBJECTS
        }
        point = float(np.mean(list(subject_points.values())))
        draws = np.empty(N_BOOTSTRAP)
        for draw_index in range(N_BOOTSTRAP):
            subject_draws = []
            for subject in SUBJECTS:
                values = block[block["subject"].eq(subject)]["fractional_error_reduction"].to_numpy()
                subject_draws.append(float(np.median(rng.choice(values, size=len(values), replace=True))))
            draws[draw_index] = np.mean(subject_draws)
        rows.append(
            {
                "outcome": outcome,
                "null": null_name,
                "equal_animal_fractional_error_reduction": point,
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "allen_median": subject_points["Allen"],
                "logan_median": subject_points["Logan"],
                "n_sessions": len(block),
            }
        )
    return session_null, pd.DataFrame(rows)


def summarize_integration(integration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (subject, target, outcome), block in integration.groupby(
        ["subject", "target_session", "outcome"], sort=True
    ):
        actual = float(block[block["is_actual_gaze_distribution"]]["squared_error"].iloc[0])
        alternatives = block[~block["is_actual_gaze_distribution"]]["squared_error"].to_numpy()
        median_alternative = float(np.median(alternatives))
        rows.append(
            {
                "subject": subject,
                "target_session": target,
                "outcome": outcome,
                "actual_distribution_squared_error": actual,
                "median_other_distribution_squared_error": median_alternative,
                "fractional_error_reduction": (median_alternative - actual) / median_alternative,
                "actual_distribution_percentile": float(np.mean(alternatives <= actual)),
            }
        )
    session_summary = pd.DataFrame(rows)
    aggregate_rows = []
    for outcome, block in session_summary.groupby("outcome"):
        subject_medians = block.groupby("subject")["fractional_error_reduction"].median()
        aggregate_rows.append(
            {
                "outcome": outcome,
                "equal_animal_fractional_error_reduction": float(subject_medians.mean()),
                "allen_median": float(subject_medians["Allen"]),
                "logan_median": float(subject_medians["Logan"]),
                "median_actual_distribution_percentile": float(
                    block.groupby("subject")["actual_distribution_percentile"].median().mean()
                ),
            }
        )
    return session_summary, pd.DataFrame(aggregate_rows)


def binned_map(values: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)]
        xbin = pd.cut(block["gaze_x"], MAP_EDGES, labels=False, right=False)
        ybin = pd.cut(block["gaze_y"], MAP_EDGES, labels=False, right=False)
        temporary = block.assign(xbin=xbin, ybin=ybin).dropna(subset=["xbin", "ybin"])
        for (xb, yb), cell in temporary.groupby(["xbin", "ybin"]):
            rows.append(
                {
                    "subject": subject,
                    "column": column,
                    "xbin": int(xb),
                    "ybin": int(yb),
                    "x_center_deg": 0.5 * (MAP_EDGES[int(xb)] + MAP_EDGES[int(xb) + 1]),
                    "y_center_deg": 0.5 * (MAP_EDGES[int(yb)] + MAP_EDGES[int(yb) + 1]),
                    "n_trials": len(cell),
                    "median_value": float(cell[column].median()),
                }
            )
    return pd.DataFrame(rows)


def build_map_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("log_scale", "screen_a"):
        temporary = predictions.copy()
        temporary[f"pred_{outcome}"] = temporary[f"pred_{outcome}_image_plus_full_position"]
        temporary[f"residual_{outcome}"] = temporary[outcome] - temporary[f"pred_{outcome}"]
        for kind, column in (
            ("observed", outcome), ("held_out_prediction", f"pred_{outcome}"),
            ("residual", f"residual_{outcome}"),
        ):
            table = binned_map(temporary, column)
            table["outcome"] = outcome
            table["map_kind"] = kind
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def plot_spatial_maps(map_values: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 3, figsize=(9.3, 11.0), sharex=True, sharey=True)
    row_specs = [
        ("Allen", "log_scale"), ("Logan", "log_scale"),
        ("Allen", "screen_a"), ("Logan", "screen_a"),
    ]
    kinds = ("observed", "held_out_prediction", "residual")
    kind_labels = ("Observed", "Held-out prediction", "Observed − prediction")
    images = []
    for row_index, (subject, outcome) in enumerate(row_specs):
        outcome_block = map_values[map_values["outcome"].eq(outcome)]
        non_residual = outcome_block[~outcome_block["map_kind"].eq("residual")]
        main_low = float(np.nanquantile(non_residual["median_value"], 0.02))
        main_high = float(np.nanquantile(non_residual["median_value"], 0.98))
        limit_main = float(np.nanquantile(np.abs(non_residual["median_value"]), 0.98))
        residual_block = outcome_block[outcome_block["map_kind"].eq("residual")]
        limit_residual = float(np.nanquantile(np.abs(residual_block["median_value"]), 0.98))
        for column_index, (kind, label) in enumerate(zip(kinds, kind_labels, strict=True)):
            ax = axes[row_index, column_index]
            block = map_values[
                map_values["subject"].eq(subject)
                & map_values["outcome"].eq(outcome)
                & map_values["map_kind"].eq(kind)
                & map_values["n_trials"].ge(MIN_MAP_TRIALS)
            ]
            grid = np.full((len(MAP_EDGES) - 1, len(MAP_EDGES) - 1), np.nan)
            for row in block.itertuples(index=False):
                grid[int(row.ybin), int(row.xbin)] = row.median_value
            if outcome == "log_scale" and kind != "residual":
                cmap, vmin, vmax = "viridis", main_low, main_high
            else:
                limit = limit_residual if kind == "residual" else limit_main
                cmap, vmin, vmax = "coolwarm", -limit, limit
            image = ax.imshow(
                grid, origin="lower", extent=[MAP_EDGES[0], MAP_EDGES[-1], MAP_EDGES[0], MAP_EDGES[-1]],
                cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal",
            )
            images.append(image)
            ax.axhline(0, color="#666C72", lw=0.5)
            ax.axvline(0, color="#666C72", lw=0.5)
            if row_index == 0:
                ax.set_title(label, weight="semibold")
            if column_index == 0:
                ax.set_ylabel(
                    f"{subject}\n{OUTCOME_LABELS[outcome]}\nscreen y (deg)",
                    color=SUBJECT_COLORS[subject], weight="semibold",
                )
            if row_index == len(row_specs) - 1:
                ax.set_xlabel("screen x (deg)")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        "Checkpoint 11A: fixation-level gaze-to-drift maps\n"
        "Held-out within-animal predictions; image structure enters before gaze position",
        y=0.995, fontsize=11.6, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_summary(
    cv_scores: pd.DataFrame,
    pairing: pd.DataFrame,
    session_pairing: pd.DataFrame,
    integration_summary: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.3))
    x = np.arange(len(OUTCOMES))

    ax = axes[0, 0]
    width = 0.24
    for model_index, model_name in enumerate(MODEL_FEATURES):
        block = cv_scores[
            cv_scores["scope"].eq("equal-animal") & cv_scores["model"].eq(model_name)
        ].set_index("outcome").loc[list(OUTCOMES)]
        ax.bar(x + (model_index - 1) * width, block["r2"], width=width, label=MODEL_LABELS[model_name])
    ax.axhline(0, color="#8E959C", lw=0.7)
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=15, ha="right")
    ax.set_ylabel("leave-session-out R²")
    ax.set_title("A  Held-out prediction", loc="left", weight="semibold")
    ax.legend(frameon=False, fontsize=6.6)

    ax = axes[0, 1]
    null_styles = (
        ("radial_unrestricted", "eccentricity; unrestricted shuffle", "#8B9299", "s"),
        ("full_unrestricted", "full position; unrestricted shuffle", "#1B7F5C", "o"),
        ("full_eccentricity_matched", "full position; eccentricity-matched shuffle", "#7A5195", "D"),
    )
    for null_index, (null_name, label, color, marker) in enumerate(null_styles):
        block = pairing[pairing["null"].eq(null_name)].set_index("outcome").loc[list(OUTCOMES)]
        estimate = 100 * block["equal_animal_fractional_error_reduction"].to_numpy()
        low = 100 * block["ci95_low"].to_numpy()
        high = 100 * block["ci95_high"].to_numpy()
        offset = (null_index - 1) * 0.13
        ax.errorbar(
            x + offset, estimate, yerr=[estimate - low, high - estimate],
            fmt=marker, color=color, capsize=2.5, ms=4.5, label=label,
        )
    ax.axhline(0, color="#8E959C", lw=0.7, ls=":")
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=15, ha="right")
    ax.set_ylabel("error reduction from correct pairing (%)")
    ax.set_title("B  Does the exact fixation pairing help?", loc="left", weight="semibold")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[1, 0]
    matched = session_pairing[session_pairing["null"].eq("full_eccentricity_matched")]
    for subject_index, subject in enumerate(SUBJECTS):
        block = matched[matched["subject"].eq(subject)]
        for outcome_index, outcome in enumerate(OUTCOMES):
            values = 100 * block[block["outcome"].eq(outcome)]["fractional_error_reduction"].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
            xpos = outcome_index + (subject_index - 0.5) * 0.15
            ax.scatter(xpos + jitter, values, s=13, color=SUBJECT_COLORS[subject], alpha=0.65)
            ax.plot([xpos - 0.10, xpos + 0.10], [np.median(values), np.median(values)], color=SUBJECT_COLORS[subject], lw=2)
    ax.axhline(0, color="#8E959C", lw=0.7, ls=":")
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=15, ha="right")
    ax.set_ylabel("session error reduction (%)")
    ax.set_title("C  Eccentricity-matched pairing by animal", loc="left", weight="semibold")

    ax = axes[1, 1]
    block = integration_summary.set_index("outcome").loc[list(OUTCOMES)]
    estimates = 100 * block["equal_animal_fractional_error_reduction"].to_numpy()
    ax.bar(x, estimates, color=("#4C78A8", "#72B7B2", "#B279A2"), alpha=0.82)
    ax.axhline(0, color="#8E959C", lw=0.7)
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=15, ha="right")
    ax.set_ylabel("session-mean error reduction (%)")
    ax.set_title("D  Actual versus another session's gaze distribution", loc="left", weight="semibold")

    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Checkpoint 11B: held-out fixation-position test\n"
        "Positive pairing gain means the real gaze–drift pairing predicts better than support-preserving shuffles",
        y=0.995, fontsize=11.8, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def select_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used: set[tuple[str, str, int]] = set()
    variances = predictions.groupby("subject")[list(OUTCOMES)].var()
    for role, outcome, direction in (
        ("scale_positive", "log_scale", "max"),
        ("screen_allocation_positive", "screen_a", "max"),
        ("gaze_prediction_failure", None, "min"),
    ):
        for subject in SUBJECTS:
            block = predictions[predictions["subject"].eq(subject)].copy()
            improvements = []
            for candidate_outcome in OUTCOMES[:2]:
                image_error = (
                    block[candidate_outcome]
                    - block[f"pred_{candidate_outcome}_image_only"]
                ) ** 2
                full_error = (
                    block[candidate_outcome]
                    - block[f"pred_{candidate_outcome}_image_plus_full_position"]
                ) ** 2
                improvements.append((image_error - full_error) / variances.loc[subject, candidate_outcome])
            score = improvements[0] if outcome == "log_scale" else improvements[1] if outcome == "screen_a" else improvements[0] + improvements[1]
            block["selection_score"] = score
            block = block.sort_values("selection_score", ascending=(direction == "min"))
            for _index, row in block.iterrows():
                key = (subject, str(row["session"]), int(row["trial_idx"]))
                if key not in used:
                    used.add(key)
                    record = row.to_dict()
                    record["example_role"] = role
                    record["selection_rule"] = (
                        "largest normalized held-out squared-error reduction from adding gaze"
                        if direction == "max"
                        else "largest normalized held-out error increase from adding gaze across scale and screen allocation"
                    )
                    record["criterion_value"] = float(row["selection_score"])
                    rows.append(record)
                    break
    return pd.DataFrame(rows)


def plot_examples(selected: pd.DataFrame, predictions: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.2))
    roles = ("scale_positive", "screen_allocation_positive", "gaze_prediction_failure")
    role_labels = {
        "scale_positive": "Scale positive",
        "screen_allocation_positive": "Horizontal-allocation positive",
        "gaze_prediction_failure": "Prediction failure",
    }
    for role_index, role in enumerate(roles):
        for subject_index, subject in enumerate(SUBJECTS):
            row = selected[selected["example_role"].eq(role) & selected["subject"].eq(subject)].iloc[0]
            location_ax = axes[role_index, subject_index * 2]
            value_ax = axes[role_index, subject_index * 2 + 1]
            support = predictions[predictions["subject"].eq(subject)]
            location_ax.scatter(support["gaze_x"], support["gaze_y"], s=5, color="#C7CCD1", alpha=0.35)
            location_ax.scatter(row["gaze_x"], row["gaze_y"], s=50, color=SUBJECT_COLORS[subject], marker="*", zorder=4)
            location_ax.axhline(0, color="#AEB4BA", lw=0.5)
            location_ax.axvline(0, color="#AEB4BA", lw=0.5)
            location_ax.set_xlim(-12, 12)
            location_ax.set_ylim(-12, 12)
            location_ax.set_aspect("equal")
            location_ax.grid(color=GRID, lw=0.45)
            location_ax.set_title(f"{row['session']} | trial {int(row['trial_idx'])}\nr={row['eccentricity']:.1f}°")
            if role_index == 2:
                location_ax.set_xlabel("screen x (deg)")
            location_ax.set_ylabel("screen y (deg)")

            z_values = []
            labels = []
            subject_means = support[list(OUTCOMES)].mean()
            subject_stds = support[list(OUTCOMES)].std()
            for outcome in OUTCOMES:
                labels.append(OUTCOME_LABELS[outcome])
                z_values.append(
                    [
                        (row[outcome] - subject_means[outcome]) / subject_stds[outcome],
                        (row[f"pred_{outcome}_image_only"] - subject_means[outcome]) / subject_stds[outcome],
                        (row[f"pred_{outcome}_image_plus_full_position"] - subject_means[outcome]) / subject_stds[outcome],
                    ]
                )
            z_values = np.asarray(z_values)
            xx = np.arange(len(OUTCOMES))
            for column, label, color, marker in (
                (0, "observed", INK, "o"), (1, "image-only", "#8B9299", "s"),
                (2, "image + gaze", "#1B7F5C", "D"),
            ):
                value_ax.plot(xx, z_values[:, column], marker=marker, color=color, lw=1.0, ms=4, label=label)
            value_ax.axhline(0, color="#AEB4BA", lw=0.6)
            value_ax.set_xticks(xx, ["scale", "H−V", "oblique"])
            value_ax.set_ylabel("within-animal z score")
            value_ax.grid(axis="y", color=GRID, lw=0.5)
            value_ax.spines[["top", "right"]].set_visible(False)
            if role_index == 0 and subject_index == 1:
                value_ax.legend(frameon=False, fontsize=6.5)
            if subject_index == 0:
                location_ax.text(
                    -0.34, 0.5, role_labels[role], transform=location_ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=8.5, weight="semibold",
                )
    fig.suptitle(
        "Checkpoint 11C: auditable fixation-level examples\n"
        "Positive and failure roles are selected from held-out prediction errors, not from visual appearance",
        y=0.995, fontsize=11.6, weight="bold",
    )
    fig.tight_layout(rect=(0.05, 0, 1, 0.94))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trial_table()
    predictions, nulls, integration, alphas = run_cross_validation(trials)
    cv_scores = cross_validated_scores(predictions)
    session_pairing, pairing_summary = summarize_pairing(nulls)
    session_integration, integration_summary = summarize_integration(integration)
    map_values = build_map_table(predictions)
    selected = select_examples(predictions)

    outputs = {
        "maps": save_figure(plot_spatial_maps(map_values), "fixation_level_gaze_drift_spatial_maps"),
        "summary": save_figure(
            plot_summary(cv_scores, pairing_summary, session_pairing, integration_summary),
            "fixation_level_gaze_drift_pairing_summary",
        ),
        "examples": save_figure(
            plot_examples(selected, predictions), "fixation_level_gaze_drift_selected_examples"
        ),
    }
    trials.to_csv(OUT_DIR / "trial_level_inputs.csv", index=False)
    predictions.to_csv(OUT_DIR / "held_out_trial_predictions.csv", index=False)
    nulls.to_csv(OUT_DIR / "pairing_null_draws.csv", index=False)
    session_pairing.to_csv(OUT_DIR / "session_pairing_values.csv", index=False)
    pairing_summary.to_csv(OUT_DIR / "pairing_summary.csv", index=False)
    integration.to_csv(OUT_DIR / "session_distribution_donor_values.csv", index=False)
    session_integration.to_csv(OUT_DIR / "session_distribution_pairing_values.csv", index=False)
    integration_summary.to_csv(OUT_DIR / "session_distribution_summary.csv", index=False)
    cv_scores.to_csv(OUT_DIR / "cross_validated_scores.csv", index=False)
    alphas.to_csv(OUT_DIR / "selected_ridge_alphas.csv", index=False)
    map_values.to_csv(OUT_DIR / "spatial_map_values.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_examples.csv", index=False)

    full_matched = pairing_summary[
        pairing_summary["null"].eq("full_eccentricity_matched")
    ].set_index("outcome").loc[list(OUTCOMES)]
    full_unrestricted = pairing_summary[
        pairing_summary["null"].eq("full_unrestricted")
    ].set_index("outcome").loc[list(OUTCOMES)]
    score_table = cv_scores[cv_scores["scope"].eq("equal-animal")].pivot(
        index="outcome", columns="model", values="r2"
    ).loc[list(OUTCOMES)]
    report = [
        "# Fixation-level gaze-position to drift-covariance pairing: checkpoint 11",
        "",
        "One row represents one BackImage trial/fixation. Outcomes separate total drift scale",
        "from the two scale-free screen-frame covariance components. Models are animal-specific",
        "and tested on entire sessions excluded from fitting. Image orientation, image energy,",
        "background fraction, and fixation timing enter before gaze position.",
        "",
        "## Held-out prediction",
        "",
    ]
    for outcome in OUTCOMES:
        report.append(
            f"- {OUTCOME_LABELS[outcome]}: image-only R2={score_table.loc[outcome, 'image_only']:+.3f}; "
            f"image+eccentricity={score_table.loc[outcome, 'image_plus_radial']:+.3f}; "
            f"image+full-position={score_table.loc[outcome, 'image_plus_full_position']:+.3f}."
        )
    report.extend(["", "## Correct fixation pairing versus shuffled gaze support", ""])
    for outcome in OUTCOMES:
        unrestricted = full_unrestricted.loc[outcome]
        matched = full_matched.loc[outcome]
        report.append(
            f"- {OUTCOME_LABELS[outcome]}: unrestricted error reduction "
            f"{100*unrestricted.equal_animal_fractional_error_reduction:+.1f}% "
            f"[{100*unrestricted.ci95_low:+.1f}, {100*unrestricted.ci95_high:+.1f}]; "
            f"eccentricity-matched {100*matched.equal_animal_fractional_error_reduction:+.1f}% "
            f"[{100*matched.ci95_low:+.1f}, {100*matched.ci95_high:+.1f}]."
        )
    report.extend(
        [
            "",
            "A positive eccentricity-matched gain means the exact polar screen location predicts",
            "drift geometry beyond local image variables and beyond coarse gaze eccentricity.",
            "This remains an observational result: stable position-dependent tracker geometry",
            "would produce the same pattern as a biological gaze-position mechanism.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 11; fixation-level held-out gaze-position pairing test",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "analysis_unit": "one median-aggregated BackImage trial/fixation",
        "outer_validation": "leave one entire session out, fit within animal",
        "inner_validation": "five-fold grouped-by-session ridge alpha selection",
        "ridge_alphas": ALPHAS.tolist(),
        "image_features": list(IMAGE_FEATURES),
        "radial_features": list(RADIAL_FEATURES),
        "full_gaze_features": list(FULL_GAZE_FEATURES),
        "outcomes": list(OUTCOMES),
        "n_null": N_NULL,
        "n_bootstrap": N_BOOTSTRAP,
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["maps"]["png"])
    print(ROOT / outputs["summary"]["png"])
    print(ROOT / outputs["examples"]["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Broad within-session gaze-position models for supplemental Figure 4 audit.

Checkpoint 2 follows the descriptive gaze-position maps.  The primary outcome
is anisotropy divided by the total drift-cloud RMS, then translated back to
arcmin at the median movement scale of Figure 4F's high-coherence windows.
This separates a change in cloud shape/allocation from the known increase in
the overall drift scale.  Absolute-outcome WLS fits are retained as a labeled
sensitivity analysis because their heavy tails make them substantially less
stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import t as student_t
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[4]
SOURCE_WINDOWS = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
FIGURE_F_CONTRASTS = (
    ROOT
    / "outputs"
    / "fig"
    / "ssi_figure_v2"
    / "behavior_confounds_map_first_v1"
    / "panel_f_descriptive_hierarchical_profiles_v1"
    / "panel_f_parallel_minus_orthogonal.csv"
)
OUT_DIR = (
    ROOT
    / "outputs"
    / "fig"
    / "ssi_figure_v2"
    / "behavior_confounds_map_first_v1"
    / "supp_gaze_position_anisotropy_broad_model_checkpoint2_v1"
)

SUBJECTS = ("Allen", "Logan")
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
INK = "#202124"
GRID = "#D8DDE3"
ECC_EDGES = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.01])
ECC_LABELS = ("0–2", "2–4", "4–6", "6–8", "8–10", "10–14")
MAP_EDGES = np.arange(-12.0, 12.01, 2.0)
MIN_MAP_WINDOWS = 10

OUTCOMES = {
    "screen": {
        "raw": "screen_h_minus_v_arcmin",
        "normalized": "screen_h_minus_v_fraction",
        "label": "Screen H−V",
    },
    "gaze": {
        "raw": "gaze_t_minus_r_arcmin",
        "normalized": "gaze_t_minus_r_fraction",
        "label": "Gaze-frame T−R",
    },
    "axis_free": {
        "raw": "axis_free_arcmin",
        "normalized": "axis_free_fraction",
        "label": "Axis-free major−minor",
    },
}

ECC_TERM = 'cr(eccentricity_deg, df=4, constraints="center")'
SCALE_TERM = 'cr(log_scale, df=3, constraints="center")'
POLAR_TERMS = "gaze_cos1 + gaze_sin1 + gaze_cos2 + gaze_sin2"
IMAGE_TERMS = (
    "image_orientation_coherence + image_edge_x + image_edge_y + "
    "z_log_gradient_energy + image_patch_fraction_background"
)
TIMING_TERMS = "C(phase) + z_log_samples_since_event"
SESSION_TERM = "C(session)"

MODEL_SPECS = {
    "within_session": f"{ECC_TERM} + {SESSION_TERM}",
    "plus_scale": f"{ECC_TERM} + {SCALE_TERM} + {SESSION_TERM}",
    "plus_polar_angle": (
        f"{ECC_TERM} + {SCALE_TERM} + {POLAR_TERMS} + {SESSION_TERM}"
    ),
    "broad_additive": (
        f"{ECC_TERM} + {SCALE_TERM} + {POLAR_TERMS} + {IMAGE_TERMS} + "
        f"{TIMING_TERMS} + {SESSION_TERM}"
    ),
    "interaction_sensitivity": (
        f"{ECC_TERM} * ({POLAR_TERMS} + {SCALE_TERM}) + {IMAGE_TERMS} + "
        f"{TIMING_TERMS} + {SESSION_TERM}"
    ),
}
PRIMARY_SPEC = "broad_additive"


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


def load_and_derive() -> tuple[pd.DataFrame, dict[str, float]]:
    values = pd.read_csv(SOURCE_WINDOWS)
    required = {
        "subject", "session", "trial_idx", "phase", "mean_x_deg", "mean_y_deg",
        "cov_xx_deg2", "cov_xy_deg2", "cov_yy_deg2", "samples_since_event",
        "image_orientation_coherence", "image_edge_axis_deg", "image_gradient_energy",
        "image_patch_fraction_background",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing model columns: {missing}")

    numeric = sorted(required.difference({"subject", "session", "phase"}))
    ok = values["subject"].isin(SUBJECTS) & values["session"].notna()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        ok &= np.isfinite(values[column])
    values = values.loc[ok].copy().reset_index(drop=True)

    x = values["mean_x_deg"].to_numpy(dtype=float)
    y = values["mean_y_deg"].to_numpy(dtype=float)
    eccentricity = np.hypot(x, y)
    gaze_angle = np.arctan2(y, x)
    radial_x = np.divide(x, eccentricity, out=np.ones_like(x), where=eccentricity > 1e-12)
    radial_y = np.divide(y, eccentricity, out=np.zeros_like(y), where=eccentricity > 1e-12)
    tangent_x = -radial_y
    tangent_y = radial_x

    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)
    trace = cxx + cyy
    scale = 60.0 * np.sqrt(np.maximum(trace, 0.0))

    def projected_rms(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        variance = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
        return 60.0 * np.sqrt(np.maximum(variance, 0.0))

    horizontal = projected_rms(np.ones_like(x), np.zeros_like(x))
    vertical = projected_rms(np.zeros_like(x), np.ones_like(x))
    radial = projected_rms(radial_x, radial_y)
    tangential = projected_rms(tangent_x, tangent_y)
    discriminant = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy**2, 0.0))
    major = 60.0 * np.sqrt(np.maximum(0.5 * (trace + discriminant), 0.0))
    minor = 60.0 * np.sqrt(np.maximum(0.5 * (trace - discriminant), 0.0))

    log_time = np.log1p(values["samples_since_event"].to_numpy(dtype=float))
    log_gradient = np.log1p(values["image_gradient_energy"].to_numpy(dtype=float))
    coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
    edge_angle = np.radians(values["image_edge_axis_deg"].to_numpy(dtype=float))

    derived = pd.DataFrame(
        {
            "eccentricity_deg": eccentricity,
            "gaze_polar_angle_deg": np.degrees(gaze_angle),
            "gaze_cos1": np.cos(gaze_angle),
            "gaze_sin1": np.sin(gaze_angle),
            "gaze_cos2": np.cos(2.0 * gaze_angle),
            "gaze_sin2": np.sin(2.0 * gaze_angle),
            "scale_arcmin": scale,
            "log_scale": np.log(scale),
            "screen_h_minus_v_arcmin": horizontal - vertical,
            "gaze_t_minus_r_arcmin": tangential - radial,
            "axis_free_arcmin": major - minor,
            "screen_h_minus_v_fraction": (horizontal - vertical) / scale,
            "gaze_t_minus_r_fraction": (tangential - radial) / scale,
            "axis_free_fraction": (major - minor) / scale,
            "image_edge_x": coherence * np.cos(2.0 * edge_angle),
            "image_edge_y": coherence * np.sin(2.0 * edge_angle),
            "z_log_samples_since_event": (log_time - np.mean(log_time)) / np.std(log_time),
            "z_log_gradient_energy": (log_gradient - np.mean(log_gradient)) / np.std(log_gradient),
        }
    )
    values = pd.concat([values, derived], axis=1).copy()
    values["eccentricity_bin"] = pd.cut(
        values["eccentricity_deg"], ECC_EDGES, labels=ECC_LABELS,
        include_lowest=True, right=False,
    )

    n_trials_in_session = values.groupby(["subject", "session"])["trial_idx"].transform("nunique")
    n_windows_in_trial = values.groupby(["subject", "session", "trial_idx"])["trial_idx"].transform("size")
    values["hierarchical_weight"] = 1.0 / (n_trials_in_session * n_windows_in_trial)
    values["hierarchical_weight"] /= values.groupby("subject")["hierarchical_weight"].transform("mean")

    central = values["eccentricity_deg"] < 4.0
    peripheral = values["eccentricity_deg"] >= 8.0
    high_coherence = values["image_orientation_coherence"] >= 0.5
    references = {
        "central_eccentricity_deg": float(np.median(values.loc[central, "eccentricity_deg"])),
        "peripheral_eccentricity_deg": float(np.median(values.loc[peripheral, "eccentricity_deg"])),
        "reference_scale_arcmin": float(np.median(values.loc[high_coherence, "scale_arcmin"])),
    }
    return values, references


def figure_f_reference() -> dict[str, float]:
    table = pd.read_csv(FIGURE_F_CONTRASTS)
    row = table[
        table["scope"].eq("grand_equal_subject")
        & table["coherence_band"].astype(str).eq("0.5–1")
    ].iloc[0]
    return {
        "estimate": float(row["parallel_minus_orthogonal_arcmin"]),
        "ci95_low": float(row["ci95_low"]),
        "ci95_high": float(row["ci95_high"]),
    }


def fit_one(values: pd.DataFrame, outcome: str, rhs: str):
    model = smf.wls(
        f"{outcome} ~ {rhs}", data=values, weights=values["hierarchical_weight"]
    )
    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": values["session"], "use_correction": True},
    )


def counterfactual_design_mean(
    model, values: pd.DataFrame, eccentricity: float, reference_scale: float
) -> np.ndarray:
    counterfactual = values.copy()
    counterfactual["eccentricity_deg"] = float(eccentricity)
    counterfactual["log_scale"] = float(np.log(reference_scale))
    design = np.asarray(
        build_design_matrices(
            [model.model.data.design_info], counterfactual, return_type="dataframe"
        )[0],
        dtype=float,
    )
    return np.average(
        design, axis=0, weights=values["hierarchical_weight"].to_numpy(dtype=float)
    )


def estimate_from_contrast(
    model, contrast: np.ndarray, n_sessions: int, multiplier: float
) -> dict[str, float]:
    params = np.asarray(model.params, dtype=float)
    covariance = np.asarray(model.cov_params(), dtype=float)
    estimate = float(contrast @ params) * multiplier
    variance = float(contrast @ covariance @ contrast) * multiplier**2
    se = float(np.sqrt(max(variance, 0.0)))
    critical = float(student_t.ppf(0.975, max(n_sessions - 1, 1)))
    return {
        "estimate_arcmin": estimate,
        "se_arcmin": se,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "variance": variance,
    }


def fit_models(
    values: pd.DataFrame, references: dict[str, float], figure_f: dict[str, float]
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models: dict[tuple[str, str, str, str], object] = {}
    effect_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    curve_rows: list[dict] = []
    central = references["central_eccentricity_deg"]
    peripheral = references["peripheral_eccentricity_deg"]
    reference_scale = references["reference_scale_arcmin"]
    curve_grid = np.linspace(0.5, 12.0, 48)

    for outcome_id, outcome_spec in OUTCOMES.items():
        for model_scale, specs in (
            ("normalized_at_reference_scale", MODEL_SPECS),
            ("absolute_wls", {
                "broad_additive": MODEL_SPECS["broad_additive"],
                "interaction_sensitivity": MODEL_SPECS["interaction_sensitivity"],
            }),
        ):
            outcome = outcome_spec["normalized"] if model_scale.startswith("normalized") else outcome_spec["raw"]
            multiplier = reference_scale if model_scale.startswith("normalized") else 1.0
            for spec_name, rhs in specs.items():
                subject_estimates = []
                subject_variances = []
                for subject in SUBJECTS:
                    block = values[values["subject"].eq(subject)].copy()
                    model = fit_one(block, outcome, rhs)
                    models[(outcome_id, model_scale, spec_name, subject)] = model
                    n_sessions = int(block["session"].nunique())

                    x_central = counterfactual_design_mean(model, block, central, reference_scale)
                    x_peripheral = counterfactual_design_mean(model, block, peripheral, reference_scale)
                    effect = estimate_from_contrast(
                        model, x_peripheral - x_central, n_sessions, multiplier
                    )
                    subject_estimates.append(effect["estimate_arcmin"])
                    subject_variances.append(effect["variance"])
                    effect_rows.append(
                        {
                            "outcome": outcome_id,
                            "outcome_label": outcome_spec["label"],
                            "analysis_scale": model_scale,
                            "model_spec": spec_name,
                            "scope": subject,
                            **{k: v for k, v in effect.items() if k != "variance"},
                            "ratio_to_figure4f": effect["estimate_arcmin"] / figure_f["estimate"],
                            "central_eccentricity_deg": central,
                            "peripheral_eccentricity_deg": peripheral,
                            "reference_scale_arcmin": reference_scale,
                        }
                    )

                    params = pd.Series(model.params)
                    bse = pd.Series(model.bse, index=params.index)
                    pvalues = pd.Series(model.pvalues, index=params.index)
                    for term in params.index:
                        coefficient_rows.append(
                            {
                                "outcome": outcome_id,
                                "analysis_scale": model_scale,
                                "model_spec": spec_name,
                                "subject": subject,
                                "term": str(term),
                                "estimate": float(params.loc[term]),
                                "cluster_se": float(bse.loc[term]),
                                "cluster_p": float(pvalues.loc[term]),
                            }
                        )
                    diagnostic_rows.append(
                        {
                            "outcome": outcome_id,
                            "analysis_scale": model_scale,
                            "model_spec": spec_name,
                            "subject": subject,
                            "n_windows": int(len(block)),
                            "n_sessions": n_sessions,
                            "n_parameters": int(len(model.params)),
                            "design_rank": int(model.model.rank),
                            "condition_number": float(np.linalg.cond(model.model.exog)),
                            "weighted_r_squared": float(model.rsquared),
                        }
                    )

                    if model_scale == "normalized_at_reference_scale" and spec_name == PRIMARY_SPEC:
                        for eccentricity in curve_grid:
                            x_mean = counterfactual_design_mean(
                                model, block, float(eccentricity), reference_scale
                            )
                            point = estimate_from_contrast(
                                model, x_mean, n_sessions, multiplier
                            )
                            curve_rows.append(
                                {
                                    "outcome": outcome_id,
                                    "scope": subject,
                                    "eccentricity_deg": float(eccentricity),
                                    **{k: v for k, v in point.items() if k != "variance"},
                                    "variance": point["variance"],
                                }
                            )

                equal_estimate = float(np.mean(subject_estimates))
                equal_variance = float(np.sum(subject_variances) / 4.0)
                equal_se = float(np.sqrt(max(equal_variance, 0.0)))
                equal_critical = float(student_t.ppf(0.975, min(
                    values[values["subject"].eq(subject)]["session"].nunique() - 1
                    for subject in SUBJECTS
                )))
                effect_rows.append(
                    {
                        "outcome": outcome_id,
                        "outcome_label": outcome_spec["label"],
                        "analysis_scale": model_scale,
                        "model_spec": spec_name,
                        "scope": "grand_equal_subject",
                        "estimate_arcmin": equal_estimate,
                        "se_arcmin": equal_se,
                        "ci95_low": equal_estimate - equal_critical * equal_se,
                        "ci95_high": equal_estimate + equal_critical * equal_se,
                        "ratio_to_figure4f": equal_estimate / figure_f["estimate"],
                        "central_eccentricity_deg": central,
                        "peripheral_eccentricity_deg": peripheral,
                        "reference_scale_arcmin": reference_scale,
                    }
                )

    curves = pd.DataFrame(curve_rows)
    grand_rows = []
    for (outcome, eccentricity), block in curves.groupby(["outcome", "eccentricity_deg"], sort=True):
        estimate = float(block["estimate_arcmin"].mean())
        variance = float(block["variance"].sum() / 4.0)
        se = float(np.sqrt(max(variance, 0.0)))
        critical = float(student_t.ppf(0.975, 13))
        grand_rows.append(
            {
                "outcome": outcome,
                "scope": "grand_equal_subject",
                "eccentricity_deg": eccentricity,
                "estimate_arcmin": estimate,
                "se_arcmin": se,
                "ci95_low": estimate - critical * se,
                "ci95_high": estimate + critical * se,
                "variance": variance,
            }
        )
    curves = pd.concat([curves, pd.DataFrame(grand_rows)], ignore_index=True)
    return (
        models,
        pd.DataFrame(effect_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(diagnostic_rows),
        curves,
    )


def descriptive_bin_summary(values: pd.DataFrame, reference_scale: float) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        subject_values = values[values["subject"].eq(subject)]
        for order, label in enumerate(ECC_LABELS):
            block = subject_values[subject_values["eccentricity_bin"].astype(str).eq(label)]
            if block.empty:
                continue
            row = {
                "subject": subject,
                "eccentricity_bin": label,
                "eccentricity_bin_order": order,
                "eccentricity_deg": float(np.median(block["eccentricity_deg"])),
                "scale_arcmin": float(np.median(block["scale_arcmin"])),
                "image_coherence": float(np.median(block["image_orientation_coherence"])),
                "n_windows": int(len(block)),
            }
            for outcome_id, spec in OUTCOMES.items():
                row[f"{outcome_id}_equivalent_arcmin"] = float(
                    np.median(block[spec["normalized"]]) * reference_scale
                )
            rows.append(row)
    summary = pd.DataFrame(rows)
    grand = []
    for (label, order), block in summary.groupby(
        ["eccentricity_bin", "eccentricity_bin_order"], sort=True
    ):
        row = {
            "subject": "grand_equal_subject",
            "eccentricity_bin": label,
            "eccentricity_bin_order": order,
            "eccentricity_deg": float(block["eccentricity_deg"].mean()),
            "scale_arcmin": float(block["scale_arcmin"].mean()),
            "image_coherence": float(block["image_coherence"].mean()),
            "n_windows": int(block["n_windows"].sum()),
        }
        for outcome_id in OUTCOMES:
            row[f"{outcome_id}_equivalent_arcmin"] = float(
                block[f"{outcome_id}_equivalent_arcmin"].mean()
            )
        grand.append(row)
    return pd.concat([summary, pd.DataFrame(grand)], ignore_index=True)


def add_primary_residuals(values: pd.DataFrame, models: dict, reference_scale: float) -> pd.DataFrame:
    result = values.copy()
    for outcome_id, spec in OUTCOMES.items():
        residual = np.full(len(result), np.nan)
        for subject in SUBJECTS:
            mask = result["subject"].eq(subject).to_numpy()
            block = result.loc[mask]
            model = models[(outcome_id, "normalized_at_reference_scale", PRIMARY_SPEC, subject)]
            fitted = np.asarray(model.predict(block), dtype=float)
            residual[mask] = (
                block[spec["normalized"]].to_numpy(dtype=float) - fitted
            ) * reference_scale
        result[f"{outcome_id}_primary_residual_arcmin"] = residual
    return result


def plot_design_audit(
    values: pd.DataFrame, descriptive: pd.DataFrame, references: dict[str, float]
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    hb = ax.hexbin(
        values["eccentricity_deg"], values["scale_arcmin"], gridsize=(30, 26),
        bins="log", mincnt=1, cmap="viridis", extent=(0, 14, 2.0, 8.0),
    )
    ax.axvline(references["central_eccentricity_deg"], color="white", lw=0.8, ls="--")
    ax.axvline(references["peripheral_eccentricity_deg"], color="white", lw=0.8, ls="--")
    ax.axhline(references["reference_scale_arcmin"], color="white", lw=0.8, ls=":")
    ax.set_ylim(2.0, 8.0)
    tail_fraction = float(np.mean(values["scale_arcmin"] > 8.0))
    ax.text(
        0.02, 0.04, f"display zoom; {tail_fraction:.1%} exceed 8 arcmin",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=6.2, color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_title("A  Eccentricity and drift-scale support", loc="left", weight="semibold")
    ax.set_xlabel("gaze eccentricity (deg)")
    ax.set_ylabel("covariance RMS radius (arcmin)")
    plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.8, label="log window count")

    ax = axes[0, 1]
    session_ranges = values.groupby(["subject", "session"])["eccentricity_deg"].agg(["min", "max", "median"]).reset_index()
    y_position = 0
    for subject in SUBJECTS:
        block = session_ranges[session_ranges["subject"].eq(subject)].sort_values("median")
        for row in block.itertuples(index=False):
            ax.plot([row.min, row.max], [y_position, y_position], color=SUBJECT_COLORS[subject], lw=1.0)
            ax.plot(row.median, y_position, ".", color=SUBJECT_COLORS[subject], ms=3)
            y_position += 1
        y_position += 1
    ax.axvline(4.0, color="#7D858C", lw=0.8, ls=":")
    ax.axvline(8.0, color="#7D858C", lw=0.8, ls=":")
    ax.set_title("B  Every session spans both endpoints", loc="left", weight="semibold")
    ax.set_xlabel("within-session eccentricity range (deg)")
    ax.set_ylabel("session")
    ax.set_yticks([])

    ax = axes[0, 2]
    counts, xedges, yedges = np.histogram2d(
        values["eccentricity_deg"], values["gaze_polar_angle_deg"],
        bins=[np.arange(0, 14.1, 1.0), np.arange(-180, 180.1, 20.0)],
    )
    mesh = ax.pcolormesh(xedges, yedges, counts.T, norm=LogNorm(vmin=1, vmax=max(2, counts.max())), cmap="magma")
    ax.set_title("C  Gaze-angle composition changes", loc="left", weight="semibold")
    ax.set_xlabel("gaze eccentricity (deg)")
    ax.set_ylabel("gaze polar angle (deg)")
    plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.8, label="windows")

    plot_specs = [
        ("scale_arcmin", "D  Total drift scale", "RMS radius (arcmin)"),
        ("screen_equivalent_arcmin", "E  Scale-normalized screen allocation", "H−V at reference scale (arcmin)"),
        ("gaze_equivalent_arcmin", "F  Scale-normalized gaze frame", "T−R at reference scale (arcmin)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes[1], plot_specs, strict=True):
        for subject in SUBJECTS:
            block = descriptive[descriptive["subject"].eq(subject)].sort_values("eccentricity_bin_order")
            ax.plot(block["eccentricity_deg"], block[metric], "o-", ms=3, lw=1.1,
                    color=SUBJECT_COLORS[subject], alpha=0.8, label=subject)
        grand = descriptive[descriptive["subject"].eq("grand_equal_subject")].sort_values("eccentricity_bin_order")
        ax.plot(grand["eccentricity_deg"], grand[metric], "o-", ms=3.5, lw=1.7,
                color=INK, label="equal-animal median")
        if metric != "scale_arcmin":
            ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        ax.set_title(title, loc="left", weight="semibold")
        ax.set_xlabel("gaze eccentricity (deg; bin median)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[1, 0].legend(frameon=False, fontsize=6.5)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Broad-model checkpoint 2A: design and normalization audit", fontsize=13, weight="bold")
    return fig


def plot_adjusted_curves(
    curves: pd.DataFrame, descriptive: pd.DataFrame, references: dict[str, float]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    for ax, (outcome_id, spec) in zip(axes, OUTCOMES.items(), strict=True):
        grand = curves[
            curves["outcome"].eq(outcome_id)
            & curves["scope"].eq("grand_equal_subject")
        ].sort_values("eccentricity_deg")
        ax.fill_between(
            grand["eccentricity_deg"], grand["ci95_low"], grand["ci95_high"],
            color="#AEB4BA", alpha=0.28, lw=0,
        )
        for subject in SUBJECTS:
            block = curves[
                curves["outcome"].eq(outcome_id) & curves["scope"].eq(subject)
            ].sort_values("eccentricity_deg")
            ax.plot(block["eccentricity_deg"], block["estimate_arcmin"],
                    color=SUBJECT_COLORS[subject], lw=1.1, alpha=0.8, label=subject)
        ax.plot(grand["eccentricity_deg"], grand["estimate_arcmin"], color=INK,
                lw=2.0, label="equal-animal adjusted")
        raw = descriptive[descriptive["subject"].eq("grand_equal_subject")].sort_values("eccentricity_bin_order")
        ax.plot(raw["eccentricity_deg"], raw[f"{outcome_id}_equivalent_arcmin"],
                "o--", color="#7D858C", ms=3, lw=0.9, label="descriptive median")
        ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        ax.axvline(references["central_eccentricity_deg"], color=GRID, lw=0.8)
        ax.axvline(references["peripheral_eccentricity_deg"], color=GRID, lw=0.8)
        ax.set_title(chr(65 + list(OUTCOMES).index(outcome_id)) + "  " + spec["label"], loc="left", weight="semibold")
        ax.set_xlabel("counterfactual gaze eccentricity (deg)")
        ax.set_ylabel("equivalent effect at reference scale (arcmin)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.2)
    fig.suptitle("Broad additive model: adjusted eccentricity curves", fontsize=12.5, weight="bold")
    return fig


def plot_specification_effects(effects: pd.DataFrame, figure_f: dict[str, float]) -> plt.Figure:
    specs = list(MODEL_SPECS)
    labels = ["within\nsession", "+ scale", "+ polar\nangle", "+ image, phase,\nevent timing", "interaction\nsensitivity"]
    x = np.arange(len(specs), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    subset = effects[effects["analysis_scale"].eq("normalized_at_reference_scale")]
    for ax, (outcome_id, outcome_spec) in zip(axes, OUTCOMES.items(), strict=True):
        for subject, offset, color in (
            ("Allen", -0.10, SUBJECT_COLORS["Allen"]),
            ("Logan", 0.10, SUBJECT_COLORS["Logan"]),
        ):
            block = subset[
                subset["outcome"].eq(outcome_id) & subset["scope"].eq(subject)
            ].set_index("model_spec").loc[specs]
            ax.plot(x + offset, block["estimate_arcmin"], "o-", color=color,
                    lw=0.9, ms=3, alpha=0.78, label=subject)
        grand = subset[
            subset["outcome"].eq(outcome_id)
            & subset["scope"].eq("grand_equal_subject")
        ].set_index("model_spec").loc[specs]
        ax.errorbar(
            x, grand["estimate_arcmin"],
            yerr=np.vstack([
                grand["estimate_arcmin"] - grand["ci95_low"],
                grand["ci95_high"] - grand["estimate_arcmin"],
            ]),
            fmt="o-", color=INK, ecolor=INK, capsize=2.5, lw=1.5, ms=4,
            label="equal-animal 95% CI",
        )
        ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
        if outcome_id == "screen":
            ax.axhspan(figure_f["ci95_low"], figure_f["ci95_high"], color="#AEB4BA", alpha=0.22)
            ax.axhline(figure_f["estimate"], color="#6B6F75", lw=1.0, ls="--",
                       label="Figure 4F")
        ax.set_xticks(x, labels)
        ax.set_title(outcome_spec["label"], loc="left", weight="semibold")
        ax.set_ylabel("peripheral − central (arcmin at reference scale)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.2)
    fig.suptitle("What survives incremental covariate adjustment?", fontsize=12.5, weight="bold")
    return fig


def plot_residual_maps(values: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    maps: dict[tuple[str, str], np.ndarray] = {}
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        block["x_bin"] = pd.cut(block["mean_x_deg"], MAP_EDGES, labels=False, include_lowest=True, right=False)
        block["y_bin"] = pd.cut(block["mean_y_deg"], MAP_EDGES, labels=False, include_lowest=True, right=False)
        for outcome_id in OUTCOMES:
            array = np.full((len(MAP_EDGES) - 1, len(MAP_EDGES) - 1), np.nan)
            for (iy, ix), cell in block.dropna(subset=["x_bin", "y_bin"]).groupby(["y_bin", "x_bin"]):
                if len(cell) >= MIN_MAP_WINDOWS:
                    array[int(iy), int(ix)] = float(np.median(cell[f"{outcome_id}_primary_residual_arcmin"]))
            maps[(subject, outcome_id)] = array

    limits = {}
    for outcome_id in OUTCOMES:
        finite = np.concatenate([
            maps[(subject, outcome_id)][np.isfinite(maps[(subject, outcome_id)])]
            for subject in SUBJECTS
        ])
        limits[outcome_id] = max(float(np.quantile(np.abs(finite), 0.92)), 0.05)

    for row, subject in enumerate(SUBJECTS):
        for col, (outcome_id, spec) in enumerate(OUTCOMES.items()):
            ax = axes[row, col]
            limit = limits[outcome_id]
            mesh = ax.pcolormesh(
                MAP_EDGES, MAP_EDGES, maps[(subject, outcome_id)], cmap="coolwarm",
                vmin=-limit, vmax=limit, shading="flat",
            )
            ax.axhline(0, color="white", lw=0.55, alpha=0.6)
            ax.axvline(0, color="white", lw=0.55, alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_title(f"{subject}: {spec['label']}", loc="left", weight="semibold")
            ax.set_xlabel("mean horizontal gaze (deg)")
            ax.set_ylabel("mean vertical gaze (deg)")
            plt.colorbar(mesh, ax=ax, shrink=0.76, pad=0.02, label="median residual (arcmin)")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Broad additive model residual spatial maps", fontsize=12.5, weight="bold")
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 260}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def write_report(
    effects: pd.DataFrame, diagnostics: pd.DataFrame, references: dict[str, float],
    figure_f: dict[str, float], values: pd.DataFrame,
) -> None:
    primary = effects[
        effects["analysis_scale"].eq("normalized_at_reference_scale")
        & effects["model_spec"].eq(PRIMARY_SPEC)
    ].set_index(["outcome", "scope"])
    interaction = effects[
        effects["analysis_scale"].eq("normalized_at_reference_scale")
        & effects["model_spec"].eq("interaction_sensitivity")
        & effects["scope"].eq("grand_equal_subject")
    ].set_index("outcome")
    lines = [
        "# Supplemental gaze-position anisotropy: broad-model checkpoint 2",
        "",
        "## Model contract",
        "",
        "Separate within-animal weighted least-squares models were combined with equal animal weight.",
        "Trials and sessions contribute equally within animal. Cluster-robust covariance is computed",
        "at the session level. The primary outcomes are anisotropy components divided by total",
        "sample-covariance RMS and translated to arcmin at the median high-coherence Figure 4F",
        f"movement scale ({references['reference_scale_arcmin']:.4f} arcmin).",
        "",
        "The primary broad additive model includes a nonlinear eccentricity curve, nonlinear drift",
        "scale, first- and second-harmonic gaze polar angle, local image coherence and coherence-",
        "weighted image axis, image gradient energy/background fraction, fixation phase, time since",
        "the last detected event, and session fixed effects. An interaction-rich specification is",
        "reported as a sensitivity analysis rather than silently selected after seeing the result.",
        "",
        "## Primary adjusted peripheral-minus-central contrasts",
        "",
        f"Central and peripheral reference eccentricities are {references['central_eccentricity_deg']:.3f}",
        f"and {references['peripheral_eccentricity_deg']:.3f} deg (the pooled medians of <4 and >=8 deg windows).",
        "",
        "| outcome | Allen | Logan | equal animal (95% CI) | ratio to 4F |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome_id in OUTCOMES:
        allen = primary.loc[(outcome_id, "Allen")]
        logan = primary.loc[(outcome_id, "Logan")]
        grand = primary.loc[(outcome_id, "grand_equal_subject")]
        lines.append(
            f"| {OUTCOMES[outcome_id]['label']} | {allen.estimate_arcmin:+.3f} | "
            f"{logan.estimate_arcmin:+.3f} | {grand.estimate_arcmin:+.3f} "
            f"[{grand.ci95_low:+.3f}, {grand.ci95_high:+.3f}] | "
            f"{grand.ratio_to_figure4f:+.2f}x |"
        )
    lines.extend(
        [
            "",
            f"Figure 4F reference: {figure_f['estimate']:+.3f} arcmin, 95% CI "
            f"[{figure_f['ci95_low']:+.3f}, {figure_f['ci95_high']:+.3f}].",
            "",
            "## Specification sensitivity",
            "",
            "Allowing eccentricity to interact flexibly with drift scale and gaze polar angle gives",
        ]
    )
    for outcome_id in OUTCOMES:
        row = interaction.loc[outcome_id]
        lines.append(
            f"- {OUTCOMES[outcome_id]['label']}: {row.estimate_arcmin:+.3f} "
            f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] arcmin."
        )
    lines.extend(
        [
            "",
            "The additive and interaction-rich models therefore disagree most strongly for the",
            "screen-frame effect. That dependence must be localized over movement scale and gaze",
            "angle before the additive estimate is promoted as the final supplemental effect size.",
            "",
            "Absolute-outcome fits are retained in `adjusted_effect_size_comparison.csv`. They are",
            "much less stable because a small number of large drift clouds dominate an arcmin-scale",
            "least-squares objective; they are not the primary inference. In this table, "
            f"{100.0 * np.mean(values['scale_arcmin'] > 8.0):.1f}% of windows exceed 8 arcmin and "
            f"the maximum covariance RMS radius is {values['scale_arcmin'].max():.1f} arcmin.",
            "",
            "## Provenance",
            "",
            f"Windows: {len(values)}; sessions: {values['session'].nunique()}; session-trials: "
            f"{values[['session', 'trial_idx']].drop_duplicates().shape[0]}.",
            f"All fitted design matrices were full rank: "
            f"{bool((diagnostics['n_parameters'] == diagnostics['design_rank']).all())}.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values, references = load_and_derive()
    figure_f = figure_f_reference()
    models, effects, coefficients, diagnostics, curves = fit_models(
        values, references, figure_f
    )
    descriptive = descriptive_bin_summary(values, references["reference_scale_arcmin"])
    values = add_primary_residuals(values, models, references["reference_scale_arcmin"])

    keep = [
        "subject", "session", "trial_idx", "phase", "mean_x_deg", "mean_y_deg",
        "eccentricity_deg", "gaze_polar_angle_deg", "scale_arcmin",
        "screen_h_minus_v_arcmin", "gaze_t_minus_r_arcmin", "axis_free_arcmin",
        "screen_h_minus_v_fraction", "gaze_t_minus_r_fraction", "axis_free_fraction",
        "image_orientation_coherence", "image_edge_axis_deg", "samples_since_event",
        "hierarchical_weight", "screen_primary_residual_arcmin",
        "gaze_primary_residual_arcmin", "axis_free_primary_residual_arcmin",
    ]
    values[keep].to_csv(
        OUT_DIR / "broad_model_window_values.csv.gz", index=False, compression="gzip"
    )
    descriptive.to_csv(OUT_DIR / "descriptive_normalized_eccentricity_curves.csv", index=False)
    effects.to_csv(OUT_DIR / "adjusted_effect_size_comparison.csv", index=False)
    coefficients.to_csv(OUT_DIR / "broad_model_coefficients.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "broad_model_diagnostics.csv", index=False)
    curves.to_csv(OUT_DIR / "broad_model_adjusted_eccentricity_curves.csv", index=False)

    outputs = {
        "design_audit": save_figure(
            plot_design_audit(values, descriptive, references),
            "broad_model_design_and_normalization_audit",
        ),
        "adjusted_curves": save_figure(
            plot_adjusted_curves(curves, descriptive, references),
            "broad_model_adjusted_eccentricity_curves",
        ),
        "specification_effects": save_figure(
            plot_specification_effects(effects, figure_f),
            "broad_model_incremental_specification_effects",
        ),
        "residual_maps": save_figure(
            plot_residual_maps(values), "broad_model_residual_spatial_maps"
        ),
    }
    write_report(effects, diagnostics, references, figure_f, values)

    metadata = {
        "stage": "map-first checkpoint 2; broad model with incremental specifications",
        "source_windows": str(SOURCE_WINDOWS.relative_to(ROOT)),
        "figure4f_reference": str(FIGURE_F_CONTRASTS.relative_to(ROOT)),
        "n_windows": int(len(values)),
        "n_sessions": int(values["session"].nunique()),
        "n_session_trials": int(values[["session", "trial_idx"]].drop_duplicates().shape[0]),
        "references": references,
        "primary_specification": PRIMARY_SPEC,
        "model_specifications": MODEL_SPECS,
        "inference": (
            "separate animal WLS with equal session/trial weights; session-clustered CR1 covariance; "
            "equal-animal estimates average animal effects and variances"
        ),
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for label, paths in outputs.items():
        print(label, ROOT / paths["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checkpoint 5: axial-orientation validation of the Figure 4F audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import t as student_t
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig.ssi_figure_v2.behavior_confounds import (  # noqa: E402
    build_panel_f_gaze_attenuation_checkpoint4 as prior,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (  # noqa: E402
    build_supp_gaze_position_anisotropy_broad_model as broad,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (  # noqa: E402
    build_supp_gaze_position_covariance_contrasts as covariance,
)


OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_axial_orientation_audit_checkpoint5_v1"
)
SUBJECTS = broad.SUBJECTS
SUBJECT_COLORS = broad.SUBJECT_COLORS
INK = broad.INK
GRID = broad.GRID
N_BOOTSTRAP = 1500
SEED = 20260810
CANONICAL_LABELS = ("horizontal", "45°", "vertical", "135°")


def load_values() -> tuple[pd.DataFrame, dict[str, float]]:
    values, references = prior.load_values()
    theta = np.radians(values["absolute_contour_axis_deg"].to_numpy(dtype=float))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)

    def projected_rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        variance = a * a * cxx + 2.0 * a * b * cxy + b * b * cyy
        return 60.0 * np.sqrt(np.maximum(variance, 0.0))

    values["parallel_rms_arcmin"] = projected_rms(ux, uy)
    values["orthogonal_rms_arcmin"] = projected_rms(vx, vy)
    values["alignment_delta_arcmin"] = (
        values["parallel_rms_arcmin"] - values["orthogonal_rms_arcmin"]
    )
    for harmonic in range(1, 5):
        values[f"axis_cos{2 * harmonic}"] = np.cos(2.0 * harmonic * theta)
        values[f"axis_sin{2 * harmonic}"] = np.sin(2.0 * harmonic * theta)
    values["canonical_axis_bin"] = axial_bin_index(
        values["absolute_contour_axis_deg"].to_numpy(dtype=float), 4, 0.0
    )
    return values, references


def axial_bin_index(theta_deg: np.ndarray, n_bins: int, center_phase_deg: float) -> np.ndarray:
    """Assign axial angles to bins whose first center is center_phase_deg."""
    width = 180.0 / float(n_bins)
    return np.floor(
        np.mod(np.asarray(theta_deg, dtype=float) - center_phase_deg + width / 2.0, 180.0)
        / width
    ).astype(int)


def session_trial_values(block: pd.DataFrame, outcome: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for session, session_block in block.groupby("session", sort=True):
        trial_values = (
            session_block.groupby("trial_idx", sort=True)[outcome]
            .median()
            .to_numpy(dtype=float)
        )
        trial_values = trial_values[np.isfinite(trial_values)]
        if trial_values.size:
            result[str(session)] = trial_values
    return result


def hierarchical_point_draws(
    block: pd.DataFrame,
    outcome: str,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, np.ndarray]:
    nested = session_trial_values(block, outcome)
    sessions = list(nested)
    if not sessions:
        return np.nan, np.full(n_bootstrap, np.nan)
    point = float(np.median([np.median(nested[session]) for session in sessions]))
    draws = np.empty(n_bootstrap, dtype=float)
    for draw_index in range(n_bootstrap):
        selected_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_points = []
        for selected in selected_sessions:
            trials = nested[sessions[int(selected)]]
            selected_trials = rng.integers(0, len(trials), size=len(trials))
            session_points.append(float(np.median(trials[selected_trials])))
        draws[draw_index] = float(np.median(session_points))
    return point, draws


def binned_standardization(
    values: pd.DataFrame,
    *,
    n_bins: int,
    phase_fraction: float,
    seed_offset: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[dict[str, float], pd.DataFrame]:
    width = 180.0 / float(n_bins)
    phase_deg = phase_fraction * width
    work = values.copy()
    work["audit_bin"] = axial_bin_index(
        work["absolute_contour_axis_deg"].to_numpy(dtype=float), n_bins, phase_deg
    )
    subject_points: dict[str, float] = {}
    subject_draws: dict[str, np.ndarray] = {}
    cell_rows = []
    for subject_index, subject in enumerate(SUBJECTS):
        block = work[work["subject"].eq(subject)]
        bin_points = []
        bin_draws = []
        for bin_index in range(n_bins):
            cell = block[block["audit_bin"].eq(bin_index)]
            rng = np.random.default_rng(SEED + seed_offset + 100 * subject_index + bin_index)
            point, draws = hierarchical_point_draws(
                cell, "alignment_delta_arcmin", rng, n_bootstrap=n_bootstrap
            )
            bin_points.append(point)
            bin_draws.append(draws)
            center = np.mod(phase_deg + bin_index * width, 180.0)
            finite_draws = draws[np.isfinite(draws)]
            cell_rows.append(
                {
                    "subject": subject,
                    "n_bins": n_bins,
                    "phase_fraction": phase_fraction,
                    "phase_deg": phase_deg,
                    "bin_index": bin_index,
                    "bin_center_deg": center,
                    "n_windows": int(len(cell)),
                    "n_trials": int(cell.groupby(["session", "trial_idx"]).ngroups),
                    "n_sessions": int(cell["session"].nunique()),
                    "effect_arcmin": point,
                    "ci95_low": (
                        float(np.quantile(finite_draws, 0.025))
                        if finite_draws.size else np.nan
                    ),
                    "ci95_high": (
                        float(np.quantile(finite_draws, 0.975))
                        if finite_draws.size else np.nan
                    ),
                }
            )
        complete = bool(np.isfinite(bin_points).all())
        subject_points[subject] = float(np.mean(bin_points)) if complete else np.nan
        subject_draws[subject] = (
            np.mean(np.stack(bin_draws), axis=0)
            if complete
            else np.full(n_bootstrap, np.nan)
        )
    complete_support = bool(np.isfinite([subject_points[s] for s in SUBJECTS]).all())
    grand_draws = (
        np.mean(np.stack([subject_draws[s] for s in SUBJECTS]), axis=0)
        if complete_support
        else np.full(n_bootstrap, np.nan)
    )
    result = {
        "n_bins": n_bins,
        "phase_fraction": phase_fraction,
        "phase_deg": phase_deg,
        "Allen": subject_points["Allen"],
        "Logan": subject_points["Logan"],
        "complete_support": complete_support,
        "grand_equal_subject": (
            float(np.mean([subject_points[s] for s in SUBJECTS]))
            if complete_support else np.nan
        ),
        "ci95_low": float(np.quantile(grand_draws, 0.025)) if complete_support else np.nan,
        "ci95_high": float(np.quantile(grand_draws, 0.975)) if complete_support else np.nan,
    }
    return result, pd.DataFrame(cell_rows)


def raw_reproduction(values: pd.DataFrame) -> dict[str, float]:
    points = {}
    draws = {}
    for subject_index, subject in enumerate(SUBJECTS):
        rng = np.random.default_rng(SEED + 9000 + subject_index)
        point, subject_draws = hierarchical_point_draws(
            values[values["subject"].eq(subject)], "alignment_delta_arcmin", rng
        )
        points[subject] = point
        draws[subject] = subject_draws
    grand_draws = np.mean(np.stack([draws[s] for s in SUBJECTS]), axis=0)
    return {
        "Allen": points["Allen"],
        "Logan": points["Logan"],
        "grand_equal_subject": float(np.mean([points[s] for s in SUBJECTS])),
        "ci95_low": float(np.quantile(grand_draws, 0.025)),
        "ci95_high": float(np.quantile(grand_draws, 0.975)),
    }


def continuous_uniform_standardization(
    values: pd.DataFrame,
    max_harmonic: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    """Median-regress the paired contrast and average it over uniform axial angle.

    Modeling parallel-minus-orthogonal directly is algebraically the interaction
    between endpoint relation and the doubled-angle terms. Median regression is
    used because Figure 4F is itself built from hierarchical medians and the
    window-level RMS differences have long tails.
    """
    harmonic_terms = []
    for harmonic in range(1, max_harmonic + 1):
        harmonic_terms.extend([f"axis_cos{2 * harmonic}", f"axis_sin{2 * harmonic}"])
    harmonic_rhs = " + ".join(harmonic_terms)
    formula = f"alignment_delta_arcmin ~ {harmonic_rhs} + C(session)"
    subject_rows = []
    subject_variances = []
    subject_estimates = []
    fitted_models: dict[str, object] = {}
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        model = smf.quantreg(formula, data=block).fit(q=0.5, max_iter=5000)
        fitted_models[subject] = model
        base = block.copy()
        for term in harmonic_terms:
            base[term] = 0.0
        design = np.asarray(
            build_design_matrices([model.model.data.design_info], base, return_type="dataframe")[0],
            dtype=float,
        )
        weights = block["hierarchical_weight"].to_numpy(dtype=float)
        contrast = np.average(design, axis=0, weights=weights)
        estimate = float(contrast @ np.asarray(model.params, dtype=float))

        # A leave-one-session-out jackknife supplies the hierarchy-aware interval
        # for the prespecified two-harmonic headline. Other harmonic orders are
        # point-estimate sensitivity checks only.
        if max_harmonic == 2:
            jackknife = []
            for heldout in sorted(block["session"].unique()):
                train = block[block["session"].ne(heldout)].copy()
                fit = smf.quantreg(formula, data=train).fit(q=0.5, max_iter=5000)
                counterfactual = train.copy()
                for term in harmonic_terms:
                    counterfactual[term] = 0.0
                xj = np.asarray(
                    build_design_matrices(
                        [fit.model.data.design_info], counterfactual, return_type="dataframe"
                    )[0],
                    dtype=float,
                )
                xj = np.average(
                    xj, axis=0,
                    weights=train["hierarchical_weight"].to_numpy(dtype=float),
                )
                jackknife.append(float(xj @ np.asarray(fit.params, dtype=float)))
            jackknife = np.asarray(jackknife, dtype=float)
            variance = float(
                (len(jackknife) - 1.0) / len(jackknife)
                * np.sum((jackknife - np.mean(jackknife)) ** 2)
            )
            se = float(np.sqrt(max(variance, 0.0)))
            critical = float(student_t.ppf(0.975, max(len(jackknife) - 1, 1)))
            ci_low = estimate - critical * se
            ci_high = estimate + critical * se
        else:
            variance = np.nan
            se = np.nan
            ci_low = np.nan
            ci_high = np.nan
        subject_rows.append(
            {
                "max_harmonic": max_harmonic,
                "scope": subject,
                "estimate_arcmin": estimate,
                "se_arcmin": se,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_windows": int(len(block)),
                "n_sessions": int(block["session"].nunique()),
            }
        )
        subject_estimates.append(estimate)
        subject_variances.append(variance)
    estimate = float(np.mean(subject_estimates))
    variance = float(np.nansum(subject_variances) / 4.0) if max_harmonic == 2 else np.nan
    se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
    critical = float(student_t.ppf(0.975, 13))
    grand = {
        "max_harmonic": max_harmonic,
        "scope": "grand_equal_subject",
        "estimate_arcmin": estimate,
        "se_arcmin": se,
        "ci95_low": estimate - critical * se if np.isfinite(se) else np.nan,
        "ci95_high": estimate + critical * se if np.isfinite(se) else np.nan,
    }
    subject_rows.append(grand)
    return grand, pd.DataFrame(subject_rows), fitted_models


def adjusted_uniform_models(values: pd.DataFrame, max_harmonic: int = 2) -> pd.DataFrame:
    harmonic_terms = " + ".join(
        term
        for harmonic in range(1, max_harmonic + 1)
        for term in (f"axis_cos{2 * harmonic}", f"axis_sin{2 * harmonic}")
    )
    specs = {
        "orientation only": harmonic_terms,
        "+ gaze": f"{harmonic_terms} + {broad.ECC_TERM} + {broad.POLAR_TERMS}",
        "+ gaze + total RMS": (
            f"{harmonic_terms} + {broad.ECC_TERM} + {broad.POLAR_TERMS} + {broad.SCALE_TERM}"
        ),
        "+ image/timing": (
            f"{harmonic_terms} + {broad.ECC_TERM} + {broad.POLAR_TERMS} + {broad.SCALE_TERM} + "
            "image_orientation_coherence + z_log_gradient_energy + "
            f"image_patch_fraction_background + {broad.TIMING_TERMS}"
        ),
    }
    rows = []
    for spec_name, rhs in specs.items():
        estimates = []
        variances = []
        for subject in SUBJECTS:
            block = values[values["subject"].eq(subject)].copy()
            model = smf.wls(
                f"alignment_delta_arcmin ~ {rhs} + C(session)",
                data=block,
                weights=block["hierarchical_weight"],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": block["session"], "use_correction": True},
            )
            counterfactual = block.copy()
            for harmonic in range(1, max_harmonic + 1):
                counterfactual[f"axis_cos{2 * harmonic}"] = 0.0
                counterfactual[f"axis_sin{2 * harmonic}"] = 0.0
            design = np.asarray(
                build_design_matrices(
                    [model.model.data.design_info], counterfactual, return_type="dataframe"
                )[0],
                dtype=float,
            )
            x = np.average(
                design, axis=0, weights=block["hierarchical_weight"].to_numpy(dtype=float)
            )
            estimate = float(x @ np.asarray(model.params, dtype=float))
            variance = float(x @ np.asarray(model.cov_params()) @ x)
            estimates.append(estimate)
            variances.append(variance)
        estimate = float(np.mean(estimates))
        variance = float(np.sum(variances) / 4.0)
        se = float(np.sqrt(max(variance, 0.0)))
        critical = float(student_t.ppf(0.975, 13))
        rows.append(
            {
                "model_spec": spec_name,
                "estimate_arcmin": estimate,
                "se_arcmin": se,
                "ci95_low": estimate - critical * se,
                "ci95_high": estimate + critical * se,
            }
        )
    return pd.DataFrame(rows)


def session_heldout_cross_validation(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare gaze models on entirely unseen sessions, without session fixed effects."""
    rows = []
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        sessions = sorted(block["session"].unique())
        for outcome_id, outcome_spec in covariance.OUTCOMES.items():
            outcome = outcome_spec["column"]
            for spec_name in ("broad_additive", "interaction_sensitivity"):
                rhs = broad.MODEL_SPECS[spec_name].replace(" + C(session)", "")
                for heldout in sessions:
                    train = block[block["session"].ne(heldout)]
                    test = block[block["session"].eq(heldout)]
                    model = smf.wls(
                        f"{outcome} ~ {rhs}", data=train, weights=train["hierarchical_weight"]
                    ).fit()
                    observed = test[outcome].to_numpy(dtype=float)
                    predicted = np.asarray(model.predict(test), dtype=float)
                    weights = test["hierarchical_weight"].to_numpy(dtype=float)
                    rows.append(
                        {
                            "subject": subject,
                            "outcome": outcome_id,
                            "model_spec": spec_name,
                            "heldout_session": heldout,
                            "n_test_windows": int(len(test)),
                            "weighted_mse": float(np.average((observed - predicted) ** 2, weights=weights)),
                            "weighted_mae": float(np.average(np.abs(observed - predicted), weights=weights)),
                            "test_weight": float(np.sum(weights)),
                        }
                    )
    folds = pd.DataFrame(rows)
    summary_rows = []
    for (subject, outcome, spec_name), group in folds.groupby(
        ["subject", "outcome", "model_spec"], sort=True
    ):
        weights = group["test_weight"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "subject": subject,
                "outcome": outcome,
                "model_spec": spec_name,
                "n_heldout_sessions": int(len(group)),
                "weighted_rmse": float(
                    np.sqrt(np.average(group["weighted_mse"], weights=weights))
                ),
                "weighted_mae": float(np.average(group["weighted_mae"], weights=weights)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    additive = summary[summary["model_spec"].eq("broad_additive")].set_index(
        ["subject", "outcome"]
    )
    interaction = summary[
        summary["model_spec"].eq("interaction_sensitivity")
    ].set_index(["subject", "outcome"])
    comparison = additive[["n_heldout_sessions", "weighted_rmse", "weighted_mae"]].join(
        interaction[["weighted_rmse", "weighted_mae"]],
        lsuffix="_additive",
        rsuffix="_interaction",
    )
    comparison["interaction_minus_additive_rmse_percent"] = 100.0 * (
        comparison["weighted_rmse_interaction"] / comparison["weighted_rmse_additive"] - 1.0
    )
    comparison["interaction_minus_additive_mae_percent"] = 100.0 * (
        comparison["weighted_mae_interaction"] / comparison["weighted_mae_additive"] - 1.0
    )
    return comparison.reset_index(), folds


def plot_axial_audit(
    values: pd.DataFrame,
    raw: dict[str, float],
    canonical: dict[str, float],
    canonical_cells: pd.DataFrame,
    sensitivity: pd.DataFrame,
    continuous: pd.DataFrame,
    adjusted: pd.DataFrame,
    figure_f: dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.7), constrained_layout=True)

    ax = axes[0, 0]
    bins = np.linspace(0, 180, 73)
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)]
        ax.hist(
            block["absolute_contour_axis_deg"], bins=bins,
            weights=np.ones(len(block)) / len(block), histtype="step", lw=1.4,
            color=SUBJECT_COLORS[subject], label=subject,
        )
    for boundary in (22.5, 67.5, 112.5, 157.5):
        ax.axvline(boundary, color="#7D858C", lw=0.7, ls=":")
    ax.annotate("same horizontal bin", xy=(179, 0.018), xytext=(139, 0.038),
                arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=6.2)
    ax.annotate("", xy=(1, 0.018), xytext=(41, 0.038),
                arrowprops={"arrowstyle": "->", "lw": 0.7})
    ax.set_title("A  Axial wrap and canonical boundaries", loc="left", weight="semibold")
    ax.set_xlabel("absolute contour axis (deg; 0° = 180°)")
    ax.set_ylabel("fraction of windows")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[0, 1]
    fine_edges = np.arange(0.0, 180.001, 15.0)
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)].copy()
        block["fine_bin"] = pd.cut(
            block["absolute_contour_axis_deg"], fine_edges, labels=False,
            include_lowest=True, right=False,
        )
        points = []
        centers = []
        for index in range(len(fine_edges) - 1):
            cell = block[block["fine_bin"].eq(index)]
            if len(cell) < 8:
                continue
            points.append(float(cell.groupby(["session", "trial_idx"])["alignment_delta_arcmin"].median().median()))
            centers.append((fine_edges[index] + fine_edges[index + 1]) / 2.0)
        ax.plot(centers, points, "o-", ms=3, lw=1.0, color=SUBJECT_COLORS[subject], label=subject)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_title("B  Paired contrast across axial orientation", loc="left", weight="semibold")
    ax.set_xlabel("absolute contour axis (deg)")
    ax.set_ylabel("parallel - orthogonal RMS (arcmin)")

    ax = axes[0, 2]
    x = np.arange(4)
    for subject in SUBJECTS:
        block = canonical_cells[canonical_cells["subject"].eq(subject)].sort_values("bin_index")
        ax.plot(x, block["effect_arcmin"], "o-", ms=3.5, lw=1.1,
                color=SUBJECT_COLORS[subject], label=subject)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(x, CANONICAL_LABELS)
    ax.set_title("C  Canonical wrapped-bin contrasts", loc="left", weight="semibold")
    ax.set_ylabel("parallel - orthogonal RMS (arcmin)")

    ax = axes[1, 0]
    cont2 = continuous[continuous["max_harmonic"].eq(2)].iloc[0]
    labels = ["reported\nFigure 4F", "raw exact\nreproduction", "canonical\n4 bins", "median model\n2 harmonics"]
    estimates = np.asarray([
        figure_f["estimate"], raw["grand_equal_subject"],
        canonical["grand_equal_subject"], cont2["estimate_arcmin"],
    ])
    lows = np.asarray([
        figure_f["ci95_low"], raw["ci95_low"], canonical["ci95_low"], cont2["ci95_low"],
    ])
    highs = np.asarray([
        figure_f["ci95_high"], raw["ci95_high"], canonical["ci95_high"], cont2["ci95_high"],
    ])
    ax.errorbar(np.arange(4), estimates, yerr=np.vstack([estimates - lows, highs - estimates]),
                fmt="o", color=INK, capsize=2.5, lw=1.2)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(np.arange(4), labels)
    ax.set_title("D  Primary axial-orientation audit", loc="left", weight="semibold")
    ax.set_ylabel("high-coherence contour contrast (arcmin)")

    ax = axes[1, 1]
    supported_sensitivity = sensitivity[sensitivity["complete_support"]].copy()
    phase_colors = plt.cm.viridis(np.linspace(0.12, 0.88, 4))
    for color, (phase, block) in zip(phase_colors, supported_sensitivity.groupby("phase_fraction", sort=True)):
        block = block.sort_values("n_bins")
        ax.plot(block["n_bins"], block["grand_equal_subject"], "o-", ms=3.2, lw=1.0,
                color=color, label=f"phase {phase:.2f} bin")
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(sorted(sensitivity["n_bins"].unique()))
    ax.set_title("E  Bin-count and boundary sensitivity", loc="left", weight="semibold")
    ax.set_xlabel("number of axial bins")
    ax.set_ylabel("uniform-bin estimate (arcmin)")
    ax.legend(frameon=False, fontsize=5.7, ncol=2)

    ax = axes[1, 2]
    x = np.arange(len(adjusted))
    estimates = adjusted["estimate_arcmin"].to_numpy(dtype=float)
    lows = adjusted["ci95_low"].to_numpy(dtype=float)
    highs = adjusted["ci95_high"].to_numpy(dtype=float)
    ax.errorbar(x, estimates, yerr=np.vstack([estimates - lows, highs - estimates]),
                fmt="o-", color=INK, capsize=2.5, lw=1.1)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(x, [label.replace(" + ", "\n+ ") for label in adjusted["model_spec"]], rotation=10, ha="right")
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_title("F  Mean regression after added controls", loc="left", weight="semibold")
    ax.set_ylabel("uniform-orientation contrast (arcmin)")
    ax.text(0.02, 0.03, "Total RMS adjustment conditions on cloud scale", transform=ax.transAxes,
            fontsize=5.8, color="#6B6F75")

    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Checkpoint 5: axial-orientation validation of Figure 4F", fontsize=12.6, weight="bold")
    return fig


def plot_session_cv(comparison: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3), constrained_layout=True)
    x = np.arange(len(covariance.OUTCOMES), dtype=float)
    width = 0.34
    for ax, metric, title in (
        (axes[0], "interaction_minus_additive_rmse_percent", "A  Held-out-session RMSE"),
        (axes[1], "interaction_minus_additive_mae_percent", "B  Held-out-session MAE"),
    ):
        for index, subject in enumerate(SUBJECTS):
            block = comparison[comparison["subject"].eq(subject)].set_index("outcome").loc[list(covariance.OUTCOMES)]
            ax.bar(x + (index - 0.5) * width, block[metric], width=width,
                   color=SUBJECT_COLORS[subject], alpha=0.86, label=subject)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [covariance.OUTCOMES[key]["label"] for key in covariance.OUTCOMES], rotation=15, ha="right")
        ax.set_ylabel("interaction - additive error (%)")
        ax.set_title(title, loc="left", weight="semibold")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Leave-one-session-out comparison (positive favors additive model)",
        fontsize=11.5,
        weight="bold",
    )
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 260}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def main() -> None:
    broad.configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values, references = load_values()
    figure_f = broad.figure_f_reference()
    raw = raw_reproduction(values)

    sensitivity_rows = []
    all_cells = []
    canonical = None
    canonical_cells = None
    seed_offset = 0
    for n_bins in (4, 6, 8, 12):
        for phase_fraction in (0.0, 0.25, 0.5, 0.75):
            result, cells = binned_standardization(
                values,
                n_bins=n_bins,
                phase_fraction=phase_fraction,
                seed_offset=seed_offset,
                n_bootstrap=(N_BOOTSTRAP if n_bins == 4 and np.isclose(phase_fraction, 0.0) else 100),
            )
            sensitivity_rows.append(result)
            all_cells.append(cells)
            if n_bins == 4 and np.isclose(phase_fraction, 0.0):
                canonical = result
                canonical_cells = cells
            seed_offset += 1000
    assert canonical is not None and canonical_cells is not None
    sensitivity = pd.DataFrame(sensitivity_rows)
    cell_table = pd.concat(all_cells, ignore_index=True)

    continuous_rows = []
    continuous_subject_rows = []
    for max_harmonic in (1, 2, 3, 4):
        grand, subjects, _models = continuous_uniform_standardization(values, max_harmonic)
        continuous_rows.append(grand)
        continuous_subject_rows.append(subjects)
    continuous = pd.DataFrame(continuous_rows)
    continuous_subjects = pd.concat(continuous_subject_rows, ignore_index=True)
    adjusted = adjusted_uniform_models(values, max_harmonic=2)
    session_cv, session_cv_folds = session_heldout_cross_validation(values)

    outputs = {
        "axial_audit": save_figure(
            plot_axial_audit(
                values, raw, canonical, canonical_cells, sensitivity,
                continuous, adjusted, figure_f,
            ),
            "panel_f_axial_orientation_audit",
        ),
        "session_cv": save_figure(
            plot_session_cv(session_cv), "session_heldout_model_specification_cv"
        ),
    }

    pd.DataFrame([{"estimate": "raw_exact_reproduction", **raw},
                  {"estimate": "canonical_wrapped_four_bins", **canonical}]).to_csv(
        OUT_DIR / "axial_primary_estimates.csv", index=False
    )
    sensitivity.to_csv(OUT_DIR / "axial_bin_count_phase_sensitivity.csv", index=False)
    cell_table.to_csv(OUT_DIR / "axial_bin_cells.csv", index=False)
    continuous.to_csv(OUT_DIR / "axial_continuous_uniform_estimates.csv", index=False)
    continuous_subjects.to_csv(OUT_DIR / "axial_continuous_uniform_subject_estimates.csv", index=False)
    adjusted.to_csv(OUT_DIR / "axial_conditional_sensitivity.csv", index=False)
    session_cv.to_csv(OUT_DIR / "session_heldout_model_specification_cv.csv", index=False)
    session_cv_folds.to_csv(OUT_DIR / "session_heldout_model_specification_cv_folds.csv", index=False)
    values[[
        "subject", "session", "trial_idx", "absolute_contour_axis_deg",
        "parallel_rms_arcmin", "orthogonal_rms_arcmin", "alignment_delta_arcmin",
        "eccentricity_deg", "gaze_polar_angle_deg", "scale_arcmin",
        "image_orientation_coherence", "hierarchical_weight",
    ]].to_csv(OUT_DIR / "axial_audit_window_values.csv.gz", index=False, compression="gzip")

    cont2 = continuous[continuous["max_harmonic"].eq(2)].iloc[0]
    supported_sensitivity = sensitivity[sensitivity["complete_support"]]
    sensitivity_min = float(supported_sensitivity["grand_equal_subject"].min())
    sensitivity_max = float(supported_sensitivity["grand_equal_subject"].max())
    cv_lines = []
    for _, row in session_cv.iterrows():
        favored = "additive" if row["interaction_minus_additive_rmse_percent"] > 0 else "interaction"
        cv_lines.append(
            f"- {row['subject']} {row['outcome']}: {favored} by "
            f"{abs(row['interaction_minus_additive_rmse_percent']):.2f}% RMSE."
        )
    report = [
        "# Axial-orientation Figure 4F audit: checkpoint 5",
        "",
        f"The exact high-coherence reproduction is {raw['grand_equal_subject']:+.3f} arcmin "
        f"[{raw['ci95_low']:+.3f}, {raw['ci95_high']:+.3f}], compared with the reported "
        f"Figure 4F value {figure_f['estimate']:+.3f} arcmin.",
        "",
        "The corrected four bins are centered on the canonical axial orientations. The horizontal",
        "bin wraps across 180/0 degrees, so contours near 0 and 180 degrees receive one combined",
        "weight rather than two independent weights.",
        "",
        f"Canonical four-bin standardization gives {canonical['grand_equal_subject']:+.3f} arcmin "
        f"[{canonical['ci95_low']:+.3f}, {canonical['ci95_high']:+.3f}].",
        f"A doubled-angle median model with two harmonics gives "
        f"{cont2['estimate_arcmin']:+.3f} arcmin [{cont2['ci95_low']:+.3f}, "
        f"{cont2['ci95_high']:+.3f}].",
        f"Across the 4, 6, 8, and 12 bin/phase combinations with complete support in both animals, estimates range from "
        f"{sensitivity_min:+.3f} to {sensitivity_max:+.3f} arcmin.",
        "",
        "Panel C in the prior checkpoint was an empirical reweighting of the high-coherence paired",
        "parallel-minus-orthogonal outcome. Its only intended change was the absolute-orientation",
        "distribution. Panel F used a covariance-contrast regression and additionally fixed or",
        "conditioned on gaze, total drift-cloud RMS radius, image variables, phase, and event timing.",
        "Those are different estimands. Total RMS is part of the drift-cloud behavior and is not an",
        "innocent nuisance variable; conditioning on it can reverse the sign. The conditional series",
        "is retained only as sensitivity analysis, not as the primary Figure 4F correction.",
        "",
        "## Leave-one-session-out model comparison",
        "",
        "Session fixed effects are omitted for this prediction test because the held-out session has",
        "no fitted session coefficient. Positive percentages favor the additive model.",
        "",
        *cv_lines,
        "",
        "## Interpretation",
        "",
        "The displayed pooled Figure 4F contrast is strongly dependent on the absolute contour-",
        "orientation distribution. Both corrected axial estimates are near zero and their intervals",
        "include zero. Evidence for orientation-independent local contour-drift alignment is weak.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 5; axial-orientation validation of Figure 4F",
        "n_high_coherence_windows": int(len(values)),
        "n_bootstrap": N_BOOTSTRAP,
        "canonical_bin_centers_deg": [0.0, 45.0, 90.0, 135.0],
        "canonical_bin_boundaries_deg": [22.5, 67.5, 112.5, 157.5],
        "bin_counts": [4, 6, 8, 12],
        "phase_fractions": [0.0, 0.25, 0.5, 0.75],
        "continuous_harmonics": [1, 2, 3, 4],
        "reference_scale_arcmin": references["reference_scale_arcmin"],
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["axial_audit"]["png"])
    print(ROOT / outputs["session_cv"]["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

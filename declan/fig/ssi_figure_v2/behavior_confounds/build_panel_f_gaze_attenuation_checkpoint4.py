#!/usr/bin/env python3
"""Checkpoint 4: direct attenuation audit of the descriptive Figure 4F contrast."""

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
    build_supp_gaze_position_anisotropy_broad_model as broad,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (  # noqa: E402
    build_supp_gaze_position_covariance_contrasts as covariance,
)


OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_gaze_attenuation_checkpoint4_v1"
)
SUBJECTS = broad.SUBJECTS
SUBJECT_COLORS = broad.SUBJECT_COLORS
INK = broad.INK
GRID = broad.GRID
ORIENTATION_EDGES = np.asarray([0.0, 45.0, 90.0, 135.0, 180.0001])
ORIENTATION_LABELS = ("0–45", "45–90", "90–135", "135–180")
GAZE_EDGES = np.asarray([0.0, 4.0, 8.0, 14.01])
GAZE_LABELS = ("central <4°", "middle 4–8°", "peripheral ≥8°")
N_BOOTSTRAP = 2000
SEED = 20260809


def load_values():
    values, references = covariance.load_and_derive()
    values = values[values["image_orientation_coherence"] >= 0.5].copy().reset_index(drop=True)
    theta = np.radians(values["image_edge_axis_deg"].to_numpy(dtype=float))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)

    def variance(a, b):
        return a * a * cxx + 2.0 * a * b * cxy + b * b * cyy

    values["contour_covariance_contrast"] = (
        variance(ux, uy) - variance(vx, vy)
    ) / (cxx + cyy)
    absolute_axis = np.mod(values["image_edge_axis_deg"].to_numpy(dtype=float), 180.0)
    values["absolute_contour_axis_deg"] = absolute_axis
    values["contour_cos2"] = np.cos(2.0 * theta)
    values["contour_sin2"] = np.sin(2.0 * theta)
    values["orientation_bin"] = pd.cut(
        absolute_axis,
        ORIENTATION_EDGES,
        labels=ORIENTATION_LABELS,
        include_lowest=True,
        right=False,
    )
    values["gaze_bin"] = pd.cut(
        values["eccentricity_deg"],
        GAZE_EDGES,
        labels=GAZE_LABELS,
        include_lowest=True,
        right=False,
    )
    return values, references


def hierarchical_point_and_draws(block: pd.DataFrame, rng: np.random.Generator):
    session_trials = {}
    for session, session_block in block.groupby("session"):
        trial_values = session_block.groupby("trial_idx")["contour_covariance_contrast"].median().to_numpy(dtype=float)
        if trial_values.size:
            session_trials[str(session)] = trial_values
    sessions = list(session_trials)
    if not sessions:
        return np.nan, np.full(N_BOOTSTRAP, np.nan)
    point = float(np.median([np.median(session_trials[session]) for session in sessions]))
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for draw_index in range(N_BOOTSTRAP):
        selected_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_points = []
        for selected in selected_sessions:
            trials = session_trials[sessions[int(selected)]]
            selected_trials = rng.integers(0, len(trials), size=len(trials))
            session_points.append(float(np.median(trials[selected_trials])))
        draws[draw_index] = float(np.median(session_points))
    return point, draws


def nonparametric_attenuation(values: pd.DataFrame, references: dict[str, float]):
    rng = np.random.default_rng(SEED)
    scale = references["reference_scale_arcmin"]
    cell_rows = []
    subject_raw = {}
    subject_raw_draws = {}
    subject_balanced = {}
    subject_balanced_draws = {}

    for subject in SUBJECTS:
        subject_values = values[values["subject"].eq(subject)]
        raw_point, raw_draws = hierarchical_point_and_draws(subject_values, rng)
        subject_raw[subject] = float(covariance.rms_difference_from_contrast(raw_point, scale))
        subject_raw_draws[subject] = covariance.rms_difference_from_contrast(raw_draws, scale)
        bin_points = []
        bin_draws = []
        for label in ORIENTATION_LABELS:
            block = subject_values[subject_values["orientation_bin"].astype(str).eq(label)]
            point, draws = hierarchical_point_and_draws(block, rng)
            arc_point = float(covariance.rms_difference_from_contrast(point, scale))
            arc_draws = covariance.rms_difference_from_contrast(draws, scale)
            bin_points.append(arc_point)
            bin_draws.append(arc_draws)
            cell_rows.append(
                {
                    "subject": subject,
                    "gaze_bin": "all",
                    "orientation_bin": label,
                    "n_windows": int(len(block)),
                    "n_sessions": int(block["session"].nunique()),
                    "effect_arcmin": arc_point,
                    "ci95_low": float(np.nanquantile(arc_draws, 0.025)),
                    "ci95_high": float(np.nanquantile(arc_draws, 0.975)),
                }
            )
        subject_balanced[subject] = float(np.mean(bin_points))
        subject_balanced_draws[subject] = np.mean(np.stack(bin_draws), axis=0)

        for gaze_label in GAZE_LABELS:
            gaze_values = subject_values[subject_values["gaze_bin"].astype(str).eq(gaze_label)]
            for orientation_label in ORIENTATION_LABELS:
                block = gaze_values[
                    gaze_values["orientation_bin"].astype(str).eq(orientation_label)
                ]
                point, draws = hierarchical_point_and_draws(block, rng)
                arc_draws = covariance.rms_difference_from_contrast(draws, scale)
                cell_rows.append(
                    {
                        "subject": subject,
                        "gaze_bin": gaze_label,
                        "orientation_bin": orientation_label,
                        "n_windows": int(len(block)),
                        "n_sessions": int(block["session"].nunique()),
                        "effect_arcmin": float(covariance.rms_difference_from_contrast(point, scale)),
                        "ci95_low": float(np.nanquantile(arc_draws, 0.025)) if np.isfinite(arc_draws).any() else np.nan,
                        "ci95_high": float(np.nanquantile(arc_draws, 0.975)) if np.isfinite(arc_draws).any() else np.nan,
                    }
                )

    raw_draws = np.mean(np.stack([subject_raw_draws[s] for s in SUBJECTS]), axis=0)
    balanced_draws = np.mean(
        np.stack([subject_balanced_draws[s] for s in SUBJECTS]), axis=0
    )
    summary = pd.DataFrame(
        [
            {
                "estimate": "raw_covariance_reproduction",
                "Allen": subject_raw["Allen"],
                "Logan": subject_raw["Logan"],
                "grand_equal_subject": float(np.mean(list(subject_raw.values()))),
                "ci95_low": float(np.quantile(raw_draws, 0.025)),
                "ci95_high": float(np.quantile(raw_draws, 0.975)),
            },
            {
                "estimate": "absolute_orientation_equal_weight",
                "Allen": subject_balanced["Allen"],
                "Logan": subject_balanced["Logan"],
                "grand_equal_subject": float(np.mean(list(subject_balanced.values()))),
                "ci95_low": float(np.quantile(balanced_draws, 0.025)),
                "ci95_high": float(np.quantile(balanced_draws, 0.975)),
            },
        ]
    )
    return summary, pd.DataFrame(cell_rows)


MODEL_SPECS = {
    "session_only": broad.SESSION_TERM,
    "absolute_axis_balanced": (
        f"contour_cos2 + contour_sin2 + {broad.SESSION_TERM}"
    ),
    "plus_gaze_and_scale": (
        f"contour_cos2 + contour_sin2 + {broad.ECC_TERM} + {broad.SCALE_TERM} + "
        f"{broad.POLAR_TERMS} + {broad.SESSION_TERM}"
    ),
    "full_additive": (
        f"contour_cos2 + contour_sin2 + {broad.ECC_TERM} + {broad.SCALE_TERM} + "
        f"{broad.POLAR_TERMS} + image_orientation_coherence + "
        f"z_log_gradient_energy + image_patch_fraction_background + "
        f"{broad.TIMING_TERMS} + {broad.SESSION_TERM}"
    ),
    "interaction_model": (
        f"{broad.ECC_TERM} * ({broad.SCALE_TERM} + {broad.POLAR_TERMS} + "
        f"contour_cos2 + contour_sin2) + image_orientation_coherence + "
        f"z_log_gradient_energy + image_patch_fraction_background + "
        f"{broad.TIMING_TERMS} + {broad.SESSION_TERM}"
    ),
}


def model_attenuation(values: pd.DataFrame, references: dict[str, float]):
    rows = []
    scale = references["reference_scale_arcmin"]
    coherence_reference = float(np.median(values["image_orientation_coherence"]))
    background_reference = float(np.median(values["image_patch_fraction_background"]))
    for spec_name, rhs in MODEL_SPECS.items():
        subject_estimates = []
        subject_variances = []
        for subject in SUBJECTS:
            block = values[values["subject"].eq(subject)].copy()
            model = smf.wls(
                f"contour_covariance_contrast ~ {rhs}",
                data=block,
                weights=block["hierarchical_weight"],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": block["session"], "use_correction": True},
            )
            counterfactual = block.copy()
            if spec_name != "session_only":
                counterfactual[["contour_cos2", "contour_sin2"]] = 0.0
            if spec_name in ("plus_gaze_and_scale", "full_additive", "interaction_model"):
                counterfactual["log_scale"] = np.log(scale)
                counterfactual[["gaze_cos1", "gaze_sin1", "gaze_cos2", "gaze_sin2"]] = 0.0
            if spec_name in ("full_additive", "interaction_model"):
                counterfactual["image_orientation_coherence"] = coherence_reference
                counterfactual["z_log_gradient_energy"] = 0.0
                counterfactual["image_patch_fraction_background"] = background_reference
                counterfactual["z_log_samples_since_event"] = 0.0
            design = np.asarray(
                build_design_matrices(
                    [model.model.data.design_info], counterfactual, return_type="dataframe"
                )[0],
                dtype=float,
            )
            x_mean = np.average(
                design, axis=0, weights=block["hierarchical_weight"].to_numpy(dtype=float)
            )
            point = covariance.transformed_point(
                model, x_mean, int(block["session"].nunique()), scale
            )
            subject_estimates.append(point["estimate_arcmin"])
            subject_variances.append(point["variance"])
            rows.append(
                {
                    "model_spec": spec_name,
                    "scope": subject,
                    **{k: v for k, v in point.items() if k != "variance"},
                }
            )
        estimate = float(np.mean(subject_estimates))
        variance = float(np.sum(subject_variances) / 4.0)
        se = float(np.sqrt(max(variance, 0.0)))
        critical = float(student_t.ppf(0.975, 13))
        rows.append(
            {
                "model_spec": spec_name,
                "scope": "grand_equal_subject",
                "predicted_covariance_contrast": np.nan,
                "estimate_arcmin": estimate,
                "se_arcmin": se,
                "ci95_low": estimate - critical * se,
                "ci95_high": estimate + critical * se,
            }
        )
    return pd.DataFrame(rows)


def plot_overview(
    values: pd.DataFrame,
    nonparametric: pd.DataFrame,
    cells: pd.DataFrame,
    model: pd.DataFrame,
    figure_f: dict[str, float],
):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    bins = np.linspace(0, 180, 19)
    for subject in SUBJECTS:
        block = values[values["subject"].eq(subject)]
        weights = np.ones(len(block)) / max(len(block), 1)
        ax.hist(block["absolute_contour_axis_deg"], bins=bins, weights=weights,
                histtype="step", lw=1.7, color=SUBJECT_COLORS[subject], label=subject)
    ax.set_title("A  High-coherence absolute contour axes", loc="left", weight="semibold")
    ax.set_xlabel("absolute contour axis (deg)")
    ax.set_ylabel("fraction of windows")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    all_cells = cells[cells["gaze_bin"].eq("all")]
    x = np.arange(len(ORIENTATION_LABELS))
    for subject in SUBJECTS:
        block = all_cells[all_cells["subject"].eq(subject)].set_index("orientation_bin").loc[list(ORIENTATION_LABELS)]
        ax.plot(x, block["effect_arcmin"], "o-", ms=3.5, lw=1.2,
                color=SUBJECT_COLORS[subject], label=subject)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(x, ORIENTATION_LABELS)
    ax.set_title("B  Figure 4F contrast within axis bins", loc="left", weight="semibold")
    ax.set_xlabel("absolute contour-axis bin (deg)")
    ax.set_ylabel("parallel − orthogonal (arcmin)")

    ax = axes[0, 2]
    points = nonparametric.set_index("estimate")
    labels = ["reported\nFigure 4F", "covariance\nreproduction", "equal-weight\naxis bins"]
    estimates = [
        figure_f["estimate"],
        points.loc["raw_covariance_reproduction", "grand_equal_subject"],
        points.loc["absolute_orientation_equal_weight", "grand_equal_subject"],
    ]
    lows = [figure_f["ci95_low"], points.loc["raw_covariance_reproduction", "ci95_low"], points.loc["absolute_orientation_equal_weight", "ci95_low"]]
    highs = [figure_f["ci95_high"], points.loc["raw_covariance_reproduction", "ci95_high"], points.loc["absolute_orientation_equal_weight", "ci95_high"]]
    ax.errorbar(np.arange(3), estimates, yerr=np.vstack([np.asarray(estimates)-lows, np.asarray(highs)-estimates]),
                fmt="o", color=INK, capsize=3, lw=1.4)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.set_xticks(np.arange(3), labels)
    ax.set_title("C  Direct attenuation", loc="left", weight="semibold")
    ax.set_ylabel("high-coherence contour contrast (arcmin)")

    for col, subject in enumerate(SUBJECTS):
        ax = axes[1, col]
        block = cells[(cells["subject"].eq(subject)) & cells["gaze_bin"].isin(GAZE_LABELS)].copy()
        support = block.pivot(index="gaze_bin", columns="orientation_bin", values="n_windows").reindex(index=GAZE_LABELS, columns=ORIENTATION_LABELS)
        image = ax.imshow(support.to_numpy(dtype=float), cmap="Blues", aspect="auto")
        for row in range(len(GAZE_LABELS)):
            for column in range(len(ORIENTATION_LABELS)):
                value = support.iloc[row, column]
                ax.text(column, row, f"{int(value) if np.isfinite(value) else 0}", ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and value > np.nanmax(support.to_numpy()) * 0.55 else INK)
        ax.set_xticks(range(len(ORIENTATION_LABELS)), ORIENTATION_LABELS)
        ax.set_yticks(range(len(GAZE_LABELS)), GAZE_LABELS)
        ax.set_title(f"{chr(68+col)}  {subject} support", loc="left", weight="semibold")
        ax.set_xlabel("absolute contour axis (deg)")
        plt.colorbar(image, ax=ax, shrink=0.75, pad=0.02, label="windows")

    ax = axes[1, 2]
    grand = model[model["scope"].eq("grand_equal_subject")].set_index("model_spec").loc[list(MODEL_SPECS)]
    labels = ["session\nonly", "balance\nabsolute axis", "+ gaze and\ntotal RMS", "+ image, phase,\nevent timing", "interaction\nmodel"]
    x = np.arange(len(labels))
    low = grand["ci95_low"].to_numpy(dtype=float).copy()
    high = grand["ci95_high"].to_numpy(dtype=float).copy()
    raw_row = nonparametric.set_index("estimate").loc["raw_covariance_reproduction"]
    low[0] = float(raw_row.ci95_low)
    high[0] = float(raw_row.ci95_high)
    estimate_array = grand["estimate_arcmin"].to_numpy(dtype=float)
    ax.errorbar(x, estimate_array, yerr=np.vstack([
        estimate_array - low,
        high - estimate_array,
    ]), fmt="o-", color=INK, capsize=2.5, lw=1.4)
    ax.axhline(0, color="#7D858C", lw=0.7, ls=":")
    ax.axhline(figure_f["estimate"], color="#6B6F75", lw=1.0, ls="--", label="reported Figure 4F")
    ax.set_xticks(x, labels, rotation=14, ha="right")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_title("F  Model-based attenuation", loc="left", weight="semibold")
    ax.set_ylabel("standardized contour contrast (arcmin)")
    ax.legend(frameon=False, fontsize=6.3)
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Checkpoint 4: does gaze/absolute-axis structure attenuate Figure 4F?", fontsize=12.6, weight="bold")
    return fig


def save_figure(fig, stem):
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 260}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def main():
    broad.configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values, references = load_values()
    figure_f = broad.figure_f_reference()
    nonparametric, cells = nonparametric_attenuation(values, references)
    model = model_attenuation(values, references)
    figure = plot_overview(values, nonparametric, cells, model, figure_f)
    outputs = {"overview": save_figure(figure, "panel_f_gaze_attenuation_overview")}

    nonparametric.to_csv(OUT_DIR / "panel_f_nonparametric_attenuation.csv", index=False)
    cells.to_csv(OUT_DIR / "panel_f_orientation_gaze_cells.csv", index=False)
    model.to_csv(OUT_DIR / "panel_f_model_attenuation.csv", index=False)
    values[[
        "subject", "session", "trial_idx", "eccentricity_deg", "gaze_polar_angle_deg",
        "scale_arcmin", "absolute_contour_axis_deg", "image_orientation_coherence",
        "contour_covariance_contrast", "orientation_bin", "gaze_bin",
    ]].to_csv(OUT_DIR / "panel_f_attenuation_window_values.csv.gz", index=False, compression="gzip")

    raw = nonparametric.set_index("estimate").loc["raw_covariance_reproduction"]
    balanced = nonparametric.set_index("estimate").loc["absolute_orientation_equal_weight"]
    attenuation = 1.0 - balanced.grand_equal_subject / raw.grand_equal_subject
    report = [
        "# Direct Figure 4F attenuation checkpoint",
        "",
        f"The covariance reconstruction is {raw.grand_equal_subject:+.3f} arcmin, compared with the",
        f"reported Figure 4F value {figure_f['estimate']:+.3f} arcmin.",
        "",
        f"Equal-weighting four absolute contour-axis bins gives {balanced.grand_equal_subject:+.3f}",
        f"arcmin [{balanced.ci95_low:+.3f}, {balanced.ci95_high:+.3f}], an attenuation of",
        f"{100.0 * attenuation:.1f}% relative to the covariance reconstruction.",
        "",
        "The model-based estimates also cross or approach zero after absolute contour orientation",
        "is balanced. This shows that the prior numerical ratio between the gaze-position effect",
        "and Figure 4F was not evidence that gaze position biased Figure 4F. Instead, the absolute",
        "screen-axis marginals themselves explain much of the displayed contour contrast.",
        "",
        "Peripheral gaze by orientation cells are sparse, so this checkpoint does not supply a",
        "precise gaze-specific attenuation coefficient. It does establish that absolute contour",
        "orientation must be controlled before Figure 4F is interpreted as local image/trajectory",
        "matching.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 4; direct Figure 4F attenuation",
        "n_high_coherence_windows": int(len(values)),
        "orientation_bins_deg": ORIENTATION_EDGES.tolist(),
        "gaze_bins_deg": GAZE_EDGES.tolist(),
        "n_bootstrap": N_BOOTSTRAP,
        "reference_scale_arcmin": references["reference_scale_arcmin"],
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["overview"]["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

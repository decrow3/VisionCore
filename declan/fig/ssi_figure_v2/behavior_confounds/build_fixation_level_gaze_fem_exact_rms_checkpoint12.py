#!/usr/bin/env python3
"""Checkpoint 12: repeat the fixation-level gaze pairing test in RMS units.

This preserves checkpoint 11 and changes only the response variables.  The
primary response is the same per-window parallel-minus-orthogonal RMS contrast
used to construct Figure 4F.  Total RMS and screen-horizontal-minus-vertical
RMS are retained as scale and screen-frame reference outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_fixation_level_gaze_fem_pairing_checkpoint11 as base,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = base.SOURCE
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "fixation_level_gaze_fem_exact_rms_checkpoint12_v1"
)
PANEL_F_CONTRASTS = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_descriptive_hierarchical_profiles_v1"
    / "panel_f_parallel_minus_orthogonal.csv"
)

OUTCOMES = (
    "total_rms_arcmin",
    "screen_h_minus_v_arcmin",
    "contour_parallel_minus_orthogonal_arcmin",
)
OUTCOME_LABELS = {
    "total_rms_arcmin": "total RMS",
    "screen_h_minus_v_arcmin": "screen H−V RMS",
    "contour_parallel_minus_orthogonal_arcmin": "contour P−O RMS (4F)",
}
SEED = base.SEED + 1
SUBJECTS = base.SUBJECTS
SUBJECT_COLORS = base.SUBJECT_COLORS
GRID = base.GRID
INK = base.INK


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_window_metrics() -> pd.DataFrame:
    values = pd.read_csv(SOURCE).copy()
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)
    trace = cxx + cyy
    coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
    edge_angle = np.radians(values["image_edge_axis_deg"].to_numpy(dtype=float))
    co, si = np.cos(edge_angle), np.sin(edge_angle)
    parallel_var = cxx * co * co + 2.0 * cxy * co * si + cyy * si * si
    orthogonal_var = cxx * si * si - 2.0 * cxy * co * si + cyy * co * co

    values["total_rms_arcmin_window"] = 60.0 * np.sqrt(np.maximum(trace, 0.0))
    values["screen_h_minus_v_arcmin_window"] = 60.0 * (
        np.sqrt(np.maximum(cxx, 0.0)) - np.sqrt(np.maximum(cyy, 0.0))
    )
    values["parallel_rms_arcmin_window"] = 60.0 * np.sqrt(np.maximum(parallel_var, 0.0))
    values["orthogonal_rms_arcmin_window"] = 60.0 * np.sqrt(np.maximum(orthogonal_var, 0.0))
    values["contour_parallel_minus_orthogonal_arcmin_window"] = (
        values["parallel_rms_arcmin_window"] - values["orthogonal_rms_arcmin_window"]
    )
    values["screen_a_window"] = (cxx - cyy) / trace
    values["contour_a_window"] = (parallel_var - orthogonal_var) / trace
    values["log_gradient_window"] = np.log1p(
        values["image_gradient_energy"].to_numpy(dtype=float)
    )
    values["edge_x_window"] = coherence * np.cos(2.0 * edge_angle)
    values["edge_y_window"] = coherence * np.sin(2.0 * edge_angle)
    values["late_window"] = values["phase"].astype(str).eq("late_fixation").astype(float)
    values["log_time_window"] = np.log1p(values["samples_since_event"].to_numpy(dtype=float))
    return values


def build_trial_table(
    values: pd.DataFrame | None = None,
    *,
    coherence_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    if values is None:
        values = load_window_metrics()
    else:
        values = values.copy()
    if coherence_range is not None:
        low, high = coherence_range
        coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
        values = values[(coherence >= low) & (coherence <= high)].copy()
    trials = values.groupby(["subject", "session", "trial_idx"], as_index=False).agg(
        gaze_x=("mean_x_deg", "median"),
        gaze_y=("mean_y_deg", "median"),
        total_rms_arcmin=("total_rms_arcmin_window", "median"),
        screen_h_minus_v_arcmin=("screen_h_minus_v_arcmin_window", "median"),
        contour_parallel_minus_orthogonal_arcmin=(
            "contour_parallel_minus_orthogonal_arcmin_window", "median"
        ),
        parallel_rms_arcmin=("parallel_rms_arcmin_window", "median"),
        orthogonal_rms_arcmin=("orthogonal_rms_arcmin_window", "median"),
        normalized_screen_a=("screen_a_window", "median"),
        normalized_contour_a=("contour_a_window", "median"),
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
    required = (
        list(OUTCOMES)
        + list(base.IMAGE_FEATURES)
        + list(base.FULL_GAZE_FEATURES)
        + list(base.RADIAL_FEATURES)
    )
    ok = trials["subject"].isin(SUBJECTS)
    for column in required:
        ok &= np.isfinite(trials[column].to_numpy(dtype=float))
    return trials.loc[ok].copy().reset_index(drop=True)


def panel_f_reproduction(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bands = ((0.0, 0.2, "0–0.2"), (0.2, 0.5, "0.2–0.5"), (0.5, 1.000001, "0.5–1"))
    for low, high, label in bands:
        coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
        block = values[(coherence >= low) & (coherence < high)].copy()
        trial = block.groupby(["subject", "session", "trial_idx"], as_index=False).agg(
            parallel_rms_arcmin=("parallel_rms_arcmin_window", "median"),
            orthogonal_rms_arcmin=("orthogonal_rms_arcmin_window", "median"),
            scalar_contrast_arcmin=(
                "contour_parallel_minus_orthogonal_arcmin_window", "median"
            ),
        )
        session = (
            trial.groupby(["subject", "session"], as_index=False)[
                ["parallel_rms_arcmin", "orthogonal_rms_arcmin"]
            ]
            .median()
        )
        subject = session.groupby("subject")[["parallel_rms_arcmin", "orthogonal_rms_arcmin"]].median()
        component_then_difference = float(
            (subject.loc[list(SUBJECTS), "parallel_rms_arcmin"]
             - subject.loc[list(SUBJECTS), "orthogonal_rms_arcmin"]).mean()
        )
        scalar_session = (
            trial.groupby(["subject", "session"])["scalar_contrast_arcmin"]
            .median().groupby("subject").median()
        )
        rows.append(
            {
                "coherence_band": label,
                "component_then_difference_arcmin": component_then_difference,
                "scalar_contrast_hierarchy_arcmin": float(scalar_session.loc[list(SUBJECTS)].mean()),
                "n_trials": len(trial),
            }
        )
    result = pd.DataFrame(rows)
    if PANEL_F_CONTRASTS.exists():
        panel = pd.read_csv(PANEL_F_CONTRASTS)
        panel = panel[panel["scope"].eq("grand_equal_subject")][
            ["coherence_band", "parallel_minus_orthogonal_arcmin"]
        ].rename(columns={"parallel_minus_orthogonal_arcmin": "panel_f_reported_arcmin"})
        result = result.merge(panel, on="coherence_band", how="left")
    return result


def plot_metric_audit(trials: pd.DataFrame, reproduction: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    sample = trials.sample(min(4000, len(trials)), random_state=SEED)
    ax = axes[0]
    for subject in SUBJECTS:
        block = sample[sample["subject"].eq(subject)]
        ax.scatter(
            block["normalized_screen_a"], block["screen_h_minus_v_arcmin"],
            s=7, alpha=0.28, color=SUBJECT_COLORS[subject], label=subject,
        )
    ax.axhline(0, color="#92989E", lw=0.7)
    ax.axvline(0, color="#92989E", lw=0.7)
    ax.set_xlabel("normalized screen H−V allocation")
    ax.set_ylabel("screen H−V RMS (arcmin)")
    ax.set_title("A  Same sign, scale now retained", loc="left", weight="semibold")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    for subject in SUBJECTS:
        block = sample[sample["subject"].eq(subject)]
        ax.scatter(
            block["normalized_contour_a"],
            block["contour_parallel_minus_orthogonal_arcmin"],
            s=7, alpha=0.28, color=SUBJECT_COLORS[subject],
        )
    ax.axhline(0, color="#92989E", lw=0.7)
    ax.axvline(0, color="#92989E", lw=0.7)
    ax.set_xlabel("normalized contour P−O allocation")
    ax.set_ylabel("contour P−O RMS (arcmin)")
    ax.set_title("B  Exact 4F units retain cloud size", loc="left", weight="semibold")

    ax = axes[2]
    x = np.arange(len(reproduction))
    width = 0.25
    ax.bar(x - width, reproduction["panel_f_reported_arcmin"], width, label="displayed 4F")
    ax.bar(x, reproduction["component_then_difference_arcmin"], width, label="trial-table components")
    ax.bar(x + width, reproduction["scalar_contrast_hierarchy_arcmin"], width, label="scalar trial contrast")
    ax.axhline(0, color="#92989E", lw=0.7)
    ax.set_xticks(x, reproduction["coherence_band"])
    ax.set_xlabel("edge coherence")
    ax.set_ylabel("parallel − orthogonal (arcmin)")
    ax.set_title("C  Aggregation-order check", loc="left", weight="semibold")
    ax.legend(frameon=False, fontsize=5.9)
    for axis in axes:
        axis.grid(axis="y", color=GRID, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Checkpoint 12A: translating normalized covariance into the original RMS metrics",
        y=1.01, fontsize=11.2, weight="bold",
    )
    fig.tight_layout()
    return fig


def select_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        block = predictions[predictions["subject"].eq(subject)].copy()
        for role, outcome, best in (
            ("4f_pairing_positive", "contour_parallel_minus_orthogonal_arcmin", True),
            ("screen_rms_pairing_positive", "screen_h_minus_v_arcmin", True),
            ("4f_pairing_failure", "contour_parallel_minus_orthogonal_arcmin", False),
        ):
            image_error = (block[outcome] - block[f"pred_{outcome}_image_only"]) ** 2
            full_error = (block[outcome] - block[f"pred_{outcome}_image_plus_full_position"]) ** 2
            block["selection_score"] = image_error - full_error
            row = block.sort_values("selection_score", ascending=not best).iloc[0].to_dict()
            row["example_role"] = role
            row["criterion_name"] = "held-out squared-error reduction from adding gaze"
            row["criterion_value"] = float(row["selection_score"])
            rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def run_model(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictions, nulls, integration, alphas = base.run_cross_validation(trials)
    cv_scores = base.cross_validated_scores(predictions)
    session_pairing, pairing_summary = base.summarize_pairing(nulls)
    session_integration, integration_summary = base.summarize_integration(integration)
    return {
        "trials": trials,
        "predictions": predictions,
        "nulls": nulls,
        "integration": integration,
        "alphas": alphas,
        "cv_scores": cv_scores,
        "session_pairing": session_pairing,
        "pairing_summary": pairing_summary,
        "session_integration": session_integration,
        "integration_summary": integration_summary,
        "selected": select_examples(predictions),
    }


def save_model_tables(result: dict[str, pd.DataFrame], prefix: str) -> None:
    names = {
        "trials": "trial_level_exact_rms_inputs.csv",
        "predictions": "held_out_trial_predictions.csv",
        "nulls": "pairing_null_draws.csv",
        "session_pairing": "session_pairing_values.csv",
        "pairing_summary": "pairing_summary.csv",
        "integration": "session_distribution_donor_values.csv",
        "session_integration": "session_distribution_pairing_values.csv",
        "integration_summary": "session_distribution_summary.csv",
        "cv_scores": "cross_validated_scores.csv",
        "alphas": "selected_ridge_alphas.csv",
        "selected": "selected_examples.csv",
    }
    for key, filename in names.items():
        result[key].to_csv(OUT_DIR / f"{prefix}{filename}", index=False)


def report_rows(result: dict[str, pd.DataFrame]) -> list[str]:
    pairing = result["pairing_summary"]
    scores_table = result["cv_scores"]
    matched = pairing[pairing["null"].eq("full_eccentricity_matched")].set_index("outcome")
    unrestricted = pairing[pairing["null"].eq("full_unrestricted")].set_index("outcome")
    scores = scores_table[scores_table["scope"].eq("equal-animal")].pivot(
        index="outcome", columns="model", values="r2"
    )
    rows = []
    for outcome in OUTCOMES:
        row = matched.loc[outcome]
        unr = unrestricted.loc[outcome]
        rows.append(
            f"- {OUTCOME_LABELS[outcome]}: full-position held-out R2 "
            f"{scores.loc[outcome, 'image_plus_full_position']:+.3f}; correct-pairing error reduction "
            f"{100*row.equal_animal_fractional_error_reduction:+.1f}% "
            f"[{100*row.ci95_low:+.1f}, {100*row.ci95_high:+.1f}] after eccentricity-matched shuffling "
            f"and {100*unr.equal_animal_fractional_error_reduction:+.1f}% unrestricted."
        )
    return rows


def main() -> None:
    base.configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the already-audited leave-session-out and shuffle machinery while
    # replacing only its outcome definitions and output labels.
    base.OUT_DIR = OUT_DIR
    base.OUTCOMES = OUTCOMES
    base.OUTCOME_LABELS = OUTCOME_LABELS
    base.SEED = SEED

    windows = load_window_metrics()
    all_trials = build_trial_table(windows)
    high_trials = build_trial_table(windows, coherence_range=(0.5, 1.0))
    reproduction = panel_f_reproduction(windows)
    all_result = run_model(all_trials)
    high_result = run_model(high_trials)

    metric_outputs = save_figure(
        plot_metric_audit(all_trials, reproduction), "exact_rms_metric_translation"
    )
    all_summary_figure = base.plot_summary(
        all_result["cv_scores"], all_result["pairing_summary"],
        all_result["session_pairing"], all_result["integration_summary"],
    )
    all_summary_figure._suptitle.set_text(
        "Checkpoint 12B: held-out gaze pairing in the original RMS units\n"
        "All fixations; same support as checkpoint 11"
    )
    all_summary_outputs = save_figure(all_summary_figure, "all_fixations_exact_rms_gaze_pairing_summary")
    high_summary_figure = base.plot_summary(
        high_result["cv_scores"], high_result["pairing_summary"],
        high_result["session_pairing"], high_result["integration_summary"],
    )
    high_summary_figure._suptitle.set_text(
        "Checkpoint 12C: direct high-coherence Figure 4F support\n"
        "Only 0.5–1 coherence windows; original RMS outcomes"
    )
    high_summary_outputs = save_figure(
        high_summary_figure, "high_coherence_exact_rms_gaze_pairing_summary"
    )

    save_model_tables(all_result, "all_fixations_")
    save_model_tables(high_result, "high_coherence_")
    reproduction.to_csv(OUT_DIR / "panel_f_metric_reproduction.csv", index=False)
    reproduction_lines = [
        "| coherence | displayed 4F | component-first trial table | scalar trial contrast | trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reproduction.itertuples(index=False):
        reproduction_lines.append(
            f"| {row.coherence_band} | {row.panel_f_reported_arcmin:+.4f} | "
            f"{row.component_then_difference_arcmin:+.4f} | "
            f"{row.scalar_contrast_hierarchy_arcmin:+.4f} | {row.n_trials} |"
        )
    report = [
        "# Fixation-level gaze pairing in exact RMS units: checkpoint 12",
        "",
        "This repeats checkpoint 11 without replacing it. The model, held-out sessions, image/timing",
        "controls, and gaze shuffles are unchanged. Outcomes are changed to RMS quantities in",
        "arcminutes. A second run uses only the high-coherence windows that support the strongest",
        "Figure 4F curve. The primary outcome is contour-parallel minus contour-orthogonal RMS.",
        "",
        "## Aggregation-order audit",
        "",
        *reproduction_lines,
        "",
        "The displayed panel subtracts parallel and orthogonal profiles after hierarchical aggregation.",
        "A prediction model needs a fixation-level scalar, so the primary model uses the median per-window",
        "RMS difference within each fixation. The table exposes the difference caused by this ordering.",
        "",
        "## All fixations: same support as checkpoint 11",
        "",
        *report_rows(all_result),
        "",
        "## High-coherence support: direct Figure 4F subset",
        "",
        *report_rows(high_result),
    ]
    report.extend(
        [
            "",
            "A positive primary gain means that the real fixation location helps predict the exact RMS",
            "contrast used by Figure 4F beyond the included image variables and coarse eccentricity.",
            "It remains observational because a stable screen-position-dependent measurement artifact",
            "would also survive this test.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 12; exact RMS metric replication of checkpoint 11",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "parent_analysis": "fixation_level_gaze_fem_pairing_checkpoint11_v1",
        "outcomes": list(OUTCOMES),
        "primary_outcome": "contour_parallel_minus_orthogonal_arcmin",
        "analysis_units": {
            "all_fixations": "one median-aggregated BackImage trial/fixation",
            "high_coherence": "one BackImage trial/fixation using only coherence 0.5-1 windows",
        },
        "validation": "leave one entire session out within animal",
        "n_null": base.N_NULL,
        "n_bootstrap": base.N_BOOTSTRAP,
        "seed": SEED,
        "outputs": {
            "metric_translation": metric_outputs,
            "all_fixations_pairing_summary": all_summary_outputs,
            "high_coherence_pairing_summary": high_summary_outputs,
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / metric_outputs["png"])
    print(ROOT / all_summary_outputs["png"])
    print(ROOT / high_summary_outputs["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

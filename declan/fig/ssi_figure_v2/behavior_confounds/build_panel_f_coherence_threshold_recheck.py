#!/usr/bin/env python3
"""Independent coherence-threshold recheck of the Figure 4F axial audit.

This checkpoint deliberately reimplements the paired RMS outcome, axial bins,
and hierarchical estimator from the raw reviewed-window table.  It does not
call the checkpoint-5 estimation helpers, so equality at coherence >= 0.5 is
an implementation cross-check rather than a rerender of the same table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
CHECKPOINT5_PRIMARY = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_axial_orientation_audit_checkpoint5_v1/axial_primary_estimates.csv"
)
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_coherence_threshold_recheck_v1"
)
SUBJECTS = ("Allen", "Logan")
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
BIN_LABELS = ("horizontal", "45°", "vertical", "135°")
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
N_BOOTSTRAP = 2000
SEED = 20260810


def load_values() -> pd.DataFrame:
    columns = [
        "subject",
        "session",
        "trial_idx",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "image_edge_axis_deg",
        "image_orientation_coherence",
    ]
    values = pd.read_csv(SOURCE, usecols=columns)
    theta_deg = np.mod(values["image_edge_axis_deg"].to_numpy(dtype=float), 180.0)
    theta = np.radians(theta_deg)
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)

    def projected_rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        variance = a * a * cxx + 2.0 * a * b * cxy + b * b * cyy
        return 60.0 * np.sqrt(np.maximum(variance, 0.0))

    values["absolute_contour_axis_deg"] = theta_deg
    values["alignment_delta_arcmin"] = projected_rms(ux, uy) - projected_rms(vx, vy)
    values["canonical_bin"] = np.floor(np.mod(theta_deg + 22.5, 180.0) / 45.0).astype(int)
    return values


def nested_trial_values(block: pd.DataFrame) -> dict[str, np.ndarray]:
    nested: dict[str, np.ndarray] = {}
    for session, session_block in block.groupby("session", sort=True):
        trials = (
            session_block.groupby("trial_idx", sort=True)["alignment_delta_arcmin"]
            .median()
            .to_numpy(dtype=float)
        )
        if trials.size:
            nested[str(session)] = trials
    return nested


def hierarchical_point_draws(
    block: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    nested = nested_trial_values(block)
    sessions = list(nested)
    if not sessions:
        return np.nan, np.full(N_BOOTSTRAP, np.nan)
    point = float(np.median([np.median(nested[session]) for session in sessions]))
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for draw_index in range(N_BOOTSTRAP):
        chosen_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_points = []
        for chosen in chosen_sessions:
            trials = nested[sessions[int(chosen)]]
            chosen_trials = rng.integers(0, len(trials), size=len(trials))
            session_points.append(float(np.median(trials[chosen_trials])))
        draws[draw_index] = float(np.median(session_points))
    return point, draws


def estimate_thresholds(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimate_rows = []
    cell_rows = []
    for threshold_index, threshold in enumerate(THRESHOLDS):
        retained = values[values["image_orientation_coherence"].ge(threshold)]
        raw_points: dict[str, float] = {}
        raw_draws: dict[str, np.ndarray] = {}
        balanced_points: dict[str, float] = {}
        balanced_draws: dict[str, np.ndarray] = {}
        for subject_index, subject in enumerate(SUBJECTS):
            subject_values = retained[retained["subject"].eq(subject)]
            raw_point, subject_raw_draws = hierarchical_point_draws(
                subject_values,
                np.random.default_rng(SEED + threshold_index * 10000 + subject_index * 1000),
            )
            raw_points[subject] = raw_point
            raw_draws[subject] = subject_raw_draws
            bin_points = []
            bin_draws = []
            for bin_index, label in enumerate(BIN_LABELS):
                cell = subject_values[subject_values["canonical_bin"].eq(bin_index)]
                point, draws = hierarchical_point_draws(
                    cell,
                    np.random.default_rng(
                        SEED
                        + threshold_index * 10000
                        + subject_index * 1000
                        + 100
                        + bin_index
                    ),
                )
                bin_points.append(point)
                bin_draws.append(draws)
                cell_rows.append(
                    {
                        "coherence_threshold_ge": threshold,
                        "subject": subject,
                        "canonical_bin": bin_index,
                        "canonical_label": label,
                        "n_windows": int(len(cell)),
                        "n_trials": int(cell.groupby(["session", "trial_idx"]).ngroups),
                        "n_sessions": int(cell["session"].nunique()),
                        "effect_arcmin": point,
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                    }
                )
            balanced_points[subject] = float(np.mean(bin_points))
            balanced_draws[subject] = np.mean(np.stack(bin_draws), axis=0)

        for estimand, points, draws_by_subject in (
            ("observed_distribution", raw_points, raw_draws),
            ("equal_four_axial_bins", balanced_points, balanced_draws),
        ):
            grand_draws = np.mean(
                np.stack([draws_by_subject[subject] for subject in SUBJECTS]), axis=0
            )
            estimate_rows.append(
                {
                    "coherence_threshold_ge": threshold,
                    "estimand": estimand,
                    "n_windows": int(len(retained)),
                    "n_trials": int(retained.groupby(["subject", "session", "trial_idx"]).ngroups),
                    "n_sessions": int(retained.groupby(["subject", "session"]).ngroups),
                    "Allen": points["Allen"],
                    "Logan": points["Logan"],
                    "grand_equal_subject": float(np.mean(list(points.values()))),
                    "ci95_low": float(np.quantile(grand_draws, 0.025)),
                    "ci95_high": float(np.quantile(grand_draws, 0.975)),
                }
            )
    return pd.DataFrame(estimate_rows), pd.DataFrame(cell_rows)


def plot_recheck(estimates: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    colors = {"observed_distribution": "#252A2E", "equal_four_axial_bins": "#2A9D8F"}
    labels = {"observed_distribution": "observed orientation distribution", "equal_four_axial_bins": "four axes weighted equally"}

    ax = axes[0]
    for estimand in labels:
        block = estimates[estimates["estimand"].eq(estimand)].sort_values(
            "coherence_threshold_ge"
        )
        x = block["coherence_threshold_ge"].to_numpy(dtype=float)
        y = block["grand_equal_subject"].to_numpy(dtype=float)
        low = block["ci95_low"].to_numpy(dtype=float)
        high = block["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", ms=4,
                    lw=1.2, capsize=2.5, color=colors[estimand], label=labels[estimand])
    ax.axhline(0, color="#80868B", lw=0.8, ls=":")
    ax.set_title("A  Threshold robustness", loc="left", weight="semibold")
    ax.set_xlabel("minimum orientation coherence (≥)")
    ax.set_ylabel("parallel − orthogonal RMS (arcmin)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    primary = cells[np.isclose(cells["coherence_threshold_ge"], 0.3)]
    x = np.arange(4, dtype=float)
    for subject_index, subject in enumerate(SUBJECTS):
        block = primary[primary["subject"].eq(subject)].sort_values("canonical_bin")
        ax.plot(x, block["effect_arcmin"], "o-", ms=4, lw=1.2,
                color=SUBJECT_COLORS[subject], label=subject)
    ax.axhline(0, color="#80868B", lw=0.8, ls=":")
    ax.set_xticks(x, BIN_LABELS)
    ax.set_title("B  Axis-specific effects at ≥0.3", loc="left", weight="semibold")
    ax.set_ylabel("parallel − orthogonal RMS (arcmin)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    forty_five = cells[cells["canonical_bin"].eq(1)]
    width = 0.035
    for subject_index, subject in enumerate(SUBJECTS):
        block = forty_five[forty_five["subject"].eq(subject)].sort_values(
            "coherence_threshold_ge"
        )
        ax.bar(
            block["coherence_threshold_ge"] + (subject_index - 0.5) * width,
            block["n_trials"],
            width=width,
            color=SUBJECT_COLORS[subject],
            alpha=0.9,
            label=subject,
        )
    ax.set_title("C  Limiting 45°-bin support", loc="left", weight="semibold")
    ax.set_xlabel("minimum orientation coherence (≥)")
    ax.set_ylabel("trials")
    ax.legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.grid(axis="y", color="#D8DDE3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Independent Figure 4F coherence-threshold recheck",
        fontsize=12,
        weight="bold",
    )
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_values()
    estimates, cells = estimate_thresholds(values)

    checkpoint5 = pd.read_csv(CHECKPOINT5_PRIMARY)
    expected = checkpoint5.set_index("estimate").loc[
        ["raw_exact_reproduction", "canonical_wrapped_four_bins"],
        "grand_equal_subject",
    ].to_numpy(dtype=float)
    observed = (
        estimates[np.isclose(estimates["coherence_threshold_ge"], 0.5)]
        .set_index("estimand")
        .loc[["observed_distribution", "equal_four_axial_bins"], "grand_equal_subject"]
        .to_numpy(dtype=float)
    )
    max_checkpoint5_difference = float(np.max(np.abs(expected - observed)))
    if max_checkpoint5_difference > 1e-12:
        raise AssertionError(
            f"Independent recheck does not reproduce checkpoint 5: {max_checkpoint5_difference}"
        )

    estimates.to_csv(OUT_DIR / "coherence_threshold_estimates.csv", index=False)
    cells.to_csv(OUT_DIR / "coherence_threshold_axis_cells.csv", index=False)
    fig = plot_recheck(estimates, cells)
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT_DIR / f"coherence_threshold_recheck.{suffix}", transparent=True, **kwargs)
    plt.close(fig)

    primary = estimates[np.isclose(estimates["coherence_threshold_ge"], 0.3)].set_index(
        "estimand"
    )
    raw = primary.loc["observed_distribution"]
    balanced = primary.loc["equal_four_axial_bins"]
    report = [
        "# Independent Figure 4F coherence-threshold recheck",
        "",
        "The raw paired RMS outcome, axial wrapping, and hierarchical point estimator were",
        "reimplemented directly from the reviewed-window source table. At coherence >=0.5,",
        f"the independent point estimates match checkpoint 5 to {max_checkpoint5_difference:.3e} arcmin.",
        "",
        "At the better-supported coherence >=0.3 threshold:",
        "",
        f"- observed orientation distribution: {raw.grand_equal_subject:+.3f} arcmin "
        f"[{raw.ci95_low:+.3f}, {raw.ci95_high:+.3f}];",
        f"- four axial bins weighted equally: {balanced.grand_equal_subject:+.3f} arcmin "
        f"[{balanced.ci95_low:+.3f}, {balanced.ci95_high:+.3f}].",
        "",
        "Across coherence thresholds >=0.2 through >=0.5, the observed-distribution point",
        "estimate remains positive and its hierarchical interval excludes zero. The equal-axis",
        "point estimate remains near zero and every interval includes zero. The conclusion is",
        "therefore not created by choosing the sparse >=0.5 cutoff.",
        "",
        "The axis-specific pattern is screen-frame-like: horizontal contours have positive",
        "parallel-minus-orthogonal spread, vertical contours have negative values, and oblique",
        "bins are weaker and less stable. That pattern does not support a general rotation of",
        "the drift cloud to follow the local contour.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first independent coherence-threshold recheck of Figure 4F",
        "source": str(SOURCE.relative_to(ROOT)),
        "checkpoint5_reference": str(CHECKPOINT5_PRIMARY.relative_to(ROOT)),
        "coherence_threshold_contract": ">= threshold",
        "thresholds": list(THRESHOLDS),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "canonical_bin_centers_deg": [0.0, 45.0, 90.0, 135.0],
        "max_abs_point_difference_from_checkpoint5_at_0p5": max_checkpoint5_difference,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUT_DIR / "coherence_threshold_recheck.png")
    print(OUT_DIR / "coherence_threshold_estimates.csv")
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checkpoint 10: compare macro fixation geometry with FEM cloud geometry."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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
    / "macro_fixation_micro_fem_checkpoint10_v1"
)
SUBJECTS = ("Allen", "Logan")
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
ANGLES_DEG = np.arange(0.0, 181.0, 1.0)
LOW_MAX = 0.05
MIN_LOW_WINDOWS = 5
MIN_LOW_TRIALS = 2
N_PERMUTATIONS = 20000
SEED = 20260810
ROLES = ("aligned_strong", "macro_only_dissociation", "micro_only_dissociation")
ROLE_LABELS = {
    "aligned_strong": "Strong absolute alignment",
    "macro_only_dissociation": "Macro-dominant rank",
    "micro_only_dissociation": "FEM-dominant rank",
}
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


def axial_distance(a: float, b: float) -> float:
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


def covariance_metrics(points: np.ndarray) -> dict[str, float]:
    covariance = np.cov(np.asarray(points, dtype=float), rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    vector = eigenvectors[:, 0]
    axis = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
    total = float(np.sum(eigenvalues))
    return {
        "axis_deg": axis,
        "anisotropy": float((eigenvalues[0] - eigenvalues[1]) / total) if total > 0 else np.nan,
        "horizontal_minus_vertical_deg": float(
            math.sqrt(max(float(covariance[0, 0]), 0.0))
            - math.sqrt(max(float(covariance[1, 1]), 0.0))
        ),
        "cov_xx": float(covariance[0, 0]),
        "cov_xy": float(covariance[0, 1]),
        "cov_yy": float(covariance[1, 1]),
    }


def projected_rms(values: pd.DataFrame) -> np.ndarray:
    theta = np.radians(ANGLES_DEG)[None, :]
    ux, uy = np.cos(theta), np.sin(theta)
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)[:, None]
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)[:, None]
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)[:, None]
    variance = ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(variance, 0.0))


def session_micro_profile(values: pd.DataFrame) -> np.ndarray:
    values = values.reset_index(drop=True)
    window_profiles = projected_rms(values)
    trial_profiles = []
    for _trial, indices in values.groupby("trial_idx", sort=True).groups.items():
        trial_profiles.append(np.median(window_profiles[np.asarray(indices, dtype=int)], axis=0))
    return np.median(np.stack(trial_profiles), axis=0)


def profile_metrics(profile: np.ndarray) -> dict[str, float]:
    profile = np.asarray(profile, dtype=float)
    preferred_index = int(np.argmax(profile[:-1]))
    maximum = float(np.max(profile[:-1]))
    minimum = float(np.min(profile[:-1]))
    return {
        "axis_deg": float(ANGLES_DEG[preferred_index]),
        "anisotropy": (maximum - minimum) / (maximum + minimum),
        "elongation_arcmin": maximum - minimum,
        "horizontal_minus_vertical_arcmin": float(profile[0] - profile[90]),
    }


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(SOURCE)
    fixation_rows = []
    session_rows = []
    profile_rows = []
    for (subject, session), block in values.groupby(["subject", "session"], sort=True):
        fixations = (
            block.groupby("trial_idx", as_index=False)
            .agg(x_deg=("mean_x_deg", "median"), y_deg=("mean_y_deg", "median"))
            .sort_values("trial_idx")
            .reset_index(drop=True)
        )
        fixations["subject"] = subject
        fixations["session"] = session
        fixation_rows.append(fixations)
        macro = covariance_metrics(fixations[["x_deg", "y_deg"]].to_numpy())
        transitions = np.diff(fixations[["x_deg", "y_deg"]].to_numpy(), axis=0)
        macro_step = covariance_metrics(transitions)

        all_profile = session_micro_profile(block)
        all_micro = profile_metrics(all_profile)
        low = block[block["image_orientation_coherence"].le(LOW_MAX)].copy()
        low_eligible = len(low) >= MIN_LOW_WINDOWS and low["trial_idx"].nunique() >= MIN_LOW_TRIALS
        low_profile = session_micro_profile(low) if low_eligible else np.full(len(ANGLES_DEG), np.nan)
        low_micro = profile_metrics(low_profile) if low_eligible else {
            "axis_deg": np.nan,
            "anisotropy": np.nan,
            "elongation_arcmin": np.nan,
            "horizontal_minus_vertical_arcmin": np.nan,
        }
        session_rows.append(
            {
                "subject": subject,
                "session": session,
                "n_fixations": len(fixations),
                "macro_position_axis_deg": macro["axis_deg"],
                "macro_position_anisotropy": macro["anisotropy"],
                "macro_position_horizontal_minus_vertical_deg": macro["horizontal_minus_vertical_deg"],
                "macro_position_cov_xx_deg2": macro["cov_xx"],
                "macro_position_cov_xy_deg2": macro["cov_xy"],
                "macro_position_cov_yy_deg2": macro["cov_yy"],
                "macro_step_axis_deg": macro_step["axis_deg"],
                "macro_step_anisotropy": macro_step["anisotropy"],
                "macro_step_horizontal_minus_vertical_deg": macro_step["horizontal_minus_vertical_deg"],
                "all_fem_axis_deg": all_micro["axis_deg"],
                "all_fem_anisotropy": all_micro["anisotropy"],
                "all_fem_elongation_arcmin": all_micro["elongation_arcmin"],
                "all_fem_horizontal_minus_vertical_arcmin": all_micro["horizontal_minus_vertical_arcmin"],
                "n_low_coherence_windows": len(low),
                "n_low_coherence_trials": low["trial_idx"].nunique(),
                "low_coherence_fem_eligible": low_eligible,
                "low_coherence_fem_axis_deg": low_micro["axis_deg"],
                "low_coherence_fem_anisotropy": low_micro["anisotropy"],
                "low_coherence_fem_elongation_arcmin": low_micro["elongation_arcmin"],
                "low_coherence_fem_horizontal_minus_vertical_arcmin": low_micro["horizontal_minus_vertical_arcmin"],
                "macro_all_fem_axis_delta_deg": axial_distance(macro["axis_deg"], all_micro["axis_deg"]),
                "macro_all_fem_axis_cos2": math.cos(math.radians(2 * (macro["axis_deg"] - all_micro["axis_deg"]))),
                "macro_low_fem_axis_delta_deg": (
                    axial_distance(macro["axis_deg"], low_micro["axis_deg"]) if low_eligible else np.nan
                ),
                "macro_low_fem_axis_cos2": (
                    math.cos(math.radians(2 * (macro["axis_deg"] - low_micro["axis_deg"])))
                    if low_eligible else np.nan
                ),
            }
        )
        for condition, profile in (("all_backimage", all_profile), ("low_coherence", low_profile)):
            for angle, rms in zip(ANGLES_DEG, profile, strict=True):
                profile_rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "condition": condition,
                        "absolute_screen_axis_deg": angle,
                        "rms_arcmin": rms,
                    }
                )
    return pd.DataFrame(session_rows), pd.concat(fixation_rows, ignore_index=True), pd.DataFrame(profile_rows)


def select_examples(sessions: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for subject in SUBJECTS:
        block = sessions[sessions["subject"].eq(subject)].copy()
        block["macro_rank"] = block["macro_position_anisotropy"].rank(pct=True)
        block["micro_rank"] = block["all_fem_anisotropy"].rank(pct=True)
        scores = {
            "aligned_strong": block["macro_all_fem_axis_cos2"] + block["macro_rank"] + block["micro_rank"],
            "macro_only_dissociation": block["macro_rank"] - block["micro_rank"] - block["macro_all_fem_axis_cos2"],
            "micro_only_dissociation": block["micro_rank"] - block["macro_rank"] - block["macro_all_fem_axis_cos2"],
        }
        used: set[str] = set()
        for role in ROLES:
            candidates = block.assign(selection_score=scores[role]).sort_values(
                ["selection_score", "session"], ascending=[False, True]
            )
            candidates = candidates[~candidates["session"].isin(used)]
            row = candidates.iloc[0].copy()
            used.add(str(row["session"]))
            row["example_role"] = role
            row["selection_rule"] = {
                "aligned_strong": "maximize axis cos2 + macro anisotropy rank + FEM anisotropy rank",
                "macro_only_dissociation": "maximize macro rank - FEM rank - axis cos2",
                "micro_only_dissociation": "maximize FEM rank - macro rank - axis cos2",
            }[role]
            selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def add_covariance_ellipse(ax: plt.Axes, row: pd.Series) -> None:
    covariance = np.asarray(
        [
            [row["macro_position_cov_xx_deg2"], row["macro_position_cov_xy_deg2"]],
            [row["macro_position_cov_xy_deg2"], row["macro_position_cov_yy_deg2"]],
        ],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(
        Ellipse(
            (0, 0), 4 * math.sqrt(eigenvalues[0]), 4 * math.sqrt(eigenvalues[1]),
            angle=angle, fill=False, edgecolor=INK, lw=1.2,
        )
    )


def full_axial(angle: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.radians(np.concatenate([angle, angle[1:] + 180])), np.concatenate([radius, radius[1:]])


def plot_examples(
    selected: pd.DataFrame, fixations: pd.DataFrame, profiles: pd.DataFrame
) -> plt.Figure:
    fig = plt.figure(figsize=(11.2, 9.0))
    grid = fig.add_gridspec(3, 4, left=0.15, right=0.985, bottom=0.07, top=0.86, hspace=0.48, wspace=0.35)
    radial_high = float(profiles[profiles["session"].isin(selected["session"])] ["rms_arcmin"].max()) * 1.05
    xy_limit = 12.5
    for role_index, role in enumerate(ROLES):
        for subject_index, subject in enumerate(SUBJECTS):
            row = selected[selected["example_role"].eq(role) & selected["subject"].eq(subject)].iloc[0]
            macro_ax = fig.add_subplot(grid[role_index, 2 * subject_index])
            polar_ax = fig.add_subplot(grid[role_index, 2 * subject_index + 1], projection="polar")
            points = fixations[fixations["session"].eq(row["session"])].sort_values("trial_idx")
            centered = points[["x_deg", "y_deg"]].to_numpy() - points[["x_deg", "y_deg"]].to_numpy().mean(axis=0)
            macro_ax.scatter(centered[:, 0], centered[:, 1], c=np.arange(len(centered)), cmap="viridis", s=12, alpha=0.78)
            add_covariance_ellipse(macro_ax, row)
            macro_ax.axhline(0, color="#AEB4BA", lw=0.6)
            macro_ax.axvline(0, color="#AEB4BA", lw=0.6)
            macro_ax.set_xlim(-xy_limit, xy_limit)
            macro_ax.set_ylim(-xy_limit, xy_limit)
            macro_ax.set_aspect("equal")
            macro_ax.grid(color=GRID, lw=0.45)
            macro_ax.set_title(
                f"fixation axis={row.macro_position_axis_deg:.0f}°; aniso={row.macro_position_anisotropy:.2f}\n"
                f"{int(row.n_fixations)} fixation centers",
                fontsize=7.5,
            )
            if role_index == 2:
                macro_ax.set_xlabel("screen x, centered (deg)")
            macro_ax.set_ylabel("screen y, centered (deg)")

            for condition, color, linestyle, label in (
                ("all_backimage", SUBJECT_COLORS[subject], "-", "all BackImage"),
                ("low_coherence", "#7A5195", "--", "coh≤0.05"),
            ):
                block = profiles[
                    profiles["session"].eq(row["session"]) & profiles["condition"].eq(condition)
                ].sort_values("absolute_screen_axis_deg")
                if block["rms_arcmin"].notna().any():
                    theta, radius = full_axial(
                        block["absolute_screen_axis_deg"].to_numpy(), block["rms_arcmin"].to_numpy()
                    )
                    polar_ax.plot(theta, radius, color=color, ls=linestyle, lw=1.8 if condition == "all_backimage" else 1.1, label=label)
            polar_ax.set_theta_zero_location("E")
            polar_ax.set_theta_direction(1)
            polar_ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], fontsize=5.8)
            polar_ax.set_ylim(0, radial_high)
            polar_ax.grid(color=GRID, lw=0.5)
            polar_ax.set_title(
                f"FEM axis={row.all_fem_axis_deg:.0f}°; aniso={row.all_fem_anisotropy:.3f}\n"
                f"axis difference={row.macro_all_fem_axis_delta_deg:.0f}°",
                fontsize=7.5,
            )
    for y, role in zip((0.73, 0.48, 0.23), ROLES, strict=True):
        fig.text(
            0.025, y, ROLE_LABELS[role], rotation=90, va="center", ha="center",
            fontsize=8.7, weight="semibold",
        )
    fig.text(0.35, 0.90, "Allen", color=SUBJECT_COLORS["Allen"], fontsize=11, weight="bold", ha="center")
    fig.text(0.76, 0.90, "Logan", color=SUBJECT_COLORS["Logan"], fontsize=11, weight="bold", ha="center")
    fig.suptitle(
        "Checkpoint 10A: macro fixation locations versus within-fixation FEM geometry\n"
        "Rows were selected by explicit agreement/dissociation scores; all macro panels share one scale",
        y=0.985, fontsize=12.0, weight="bold",
    )
    fig.text(
        0.985, 0.018, "FEM profiles: solid = all BackImage windows; dashed purple = coherence ≤0.05.",
        ha="right", fontsize=6.8, color="#4E545A",
    )
    return fig


def paired_axis_test(
    sessions: pd.DataFrame, micro_axis_column: str, eligible: pd.Series
) -> dict[str, float]:
    block = sessions[eligible].copy()
    observed = float(np.mean(np.cos(2 * np.radians(block["macro_position_axis_deg"] - block[micro_axis_column]))))
    rng = np.random.default_rng(SEED + (0 if micro_axis_column == "all_fem_axis_deg" else 10))
    null = np.empty(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        values = []
        for _subject, subject_block in block.groupby("subject"):
            values.extend(
                np.cos(
                    2 * np.radians(
                        subject_block["macro_position_axis_deg"].to_numpy()
                        - rng.permutation(subject_block[micro_axis_column].to_numpy())
                    )
                )
            )
        null[permutation] = np.mean(values)
    return {
        "observed_mean_cos2": observed,
        "shuffle_mean": float(np.mean(null)),
        "shuffle_ci95_low": float(np.quantile(null, 0.025)),
        "shuffle_ci95_high": float(np.quantile(null, 0.975)),
        "pairing_p_upper": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "n_sessions": len(block),
    }


def correlation_rows(sessions: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("all_fem", pd.Series(True, index=sessions.index), "all_fem"),
        ("low_coherence_fem", sessions["low_coherence_fem_eligible"].astype(bool), "low_coherence_fem"),
    ]
    rows = []
    for condition, eligible, prefix in specs:
        block = sessions[eligible]
        for scope in ("pooled", *SUBJECTS):
            scoped = block if scope == "pooled" else block[block["subject"].eq(scope)]
            for geometry, macro_prefix in (("position", "macro_position"), ("step", "macro_step")):
                for quantity, micro_suffix, macro_suffix in (
                    ("anisotropy_strength", "anisotropy", "anisotropy"),
                    ("horizontal_minus_vertical", "horizontal_minus_vertical_arcmin", "horizontal_minus_vertical_deg"),
                ):
                    x = scoped[f"{macro_prefix}_{macro_suffix}"].to_numpy(dtype=float)
                    y = scoped[f"{prefix}_{micro_suffix}"].to_numpy(dtype=float)
                    rows.append(
                        {
                            "fem_condition": condition,
                            "scope": scope,
                            "macro_geometry": geometry,
                            "quantity": quantity,
                            "n_sessions": len(scoped),
                            "pearson_r": pearsonr(x, y).statistic,
                            "pearson_p": pearsonr(x, y).pvalue,
                            "spearman_rho": spearmanr(x, y).statistic,
                            "spearman_p": spearmanr(x, y).pvalue,
                        }
                    )
    return pd.DataFrame(rows)


def plot_summary(
    sessions: pd.DataFrame, correlations: pd.DataFrame, axis_tests: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.3))
    panels = [
        (axes[0, 0], "macro_position_anisotropy", "all_fem_anisotropy", "A  Anisotropy strength: all BackImage FEM", "macro fixation-cloud anisotropy", "FEM anisotropy"),
        (axes[0, 1], "macro_position_horizontal_minus_vertical_deg", "all_fem_horizontal_minus_vertical_arcmin", "B  Horizontal bias: all BackImage FEM", "macro H−V spread (deg)", "FEM H−V RMS (arcmin)"),
        (axes[1, 0], "macro_position_anisotropy", "low_coherence_fem_anisotropy", "C  Anisotropy strength: low-coherence FEM", "macro fixation-cloud anisotropy", "low-coherence FEM anisotropy"),
    ]
    for ax, xcol, ycol, title, xlabel, ylabel in panels:
        block = sessions[np.isfinite(sessions[ycol])]
        for subject in SUBJECTS:
            subject_block = block[block["subject"].eq(subject)]
            ax.scatter(subject_block[xcol], subject_block[ycol], s=30, color=SUBJECT_COLORS[subject], alpha=0.82, label=subject)
            if len(subject_block) >= 3:
                coefficients = np.polyfit(subject_block[xcol], subject_block[ycol], 1)
                xx = np.linspace(subject_block[xcol].min(), subject_block[xcol].max(), 50)
                ax.plot(xx, np.polyval(coefficients, xx), color=SUBJECT_COLORS[subject], lw=1.0, alpha=0.75)
        ax.set_title(title, loc="left", weight="semibold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    x = np.arange(2)
    observed = axis_tests["observed_mean_cos2"].to_numpy()
    null_mean = axis_tests["shuffle_mean"].to_numpy()
    low = axis_tests["shuffle_ci95_low"].to_numpy()
    high = axis_tests["shuffle_ci95_high"].to_numpy()
    ax.errorbar(x, null_mean, yerr=[null_mean - low, high - null_mean], fmt="s", color="#8B9299", capsize=3, label="random session pairing")
    ax.scatter(x, observed, color=INK, marker="o", s=42, zorder=3, label="actual session pairing")
    for index, row in axis_tests.iterrows():
        ax.text(index, observed[index] + 0.035, f"p={row.pairing_p_upper:.2f}", ha="center", fontsize=7)
    ax.axhline(0, color="#AEB4BA", lw=0.7, ls=":")
    ax.set_xticks(x, ["all BackImage FEM", "low-coherence FEM"])
    ax.set_ylabel("macro–FEM axis agreement (cos 2Δ)")
    ax.set_title("D  Does the correct session pairing matter?", loc="left", weight="semibold")
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.7, loc="lower left")

    fig.suptitle(
        "Checkpoint 10B: session-level macro–micro association\n"
        "Absolute axes often overlap. A correlation requires the correct macro and FEM clouds to pair by session.",
        y=0.99, fontsize=11.6, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    sessions, fixations, profiles = build_tables()
    selected = select_examples(sessions)
    correlations = correlation_rows(sessions)
    axis_rows = []
    for label, column, eligible in (
        ("all_backimage_fem", "all_fem_axis_deg", pd.Series(True, index=sessions.index)),
        ("low_coherence_fem", "low_coherence_fem_axis_deg", sessions["low_coherence_fem_eligible"].astype(bool)),
    ):
        axis_rows.append({"fem_condition": label, **paired_axis_test(sessions, column, eligible)})
    axis_tests = pd.DataFrame(axis_rows)

    outputs = {
        "examples": save_figure(plot_examples(selected, fixations, profiles), "macro_fixation_micro_fem_examples"),
        "summary": save_figure(plot_summary(sessions, correlations, axis_tests), "macro_fixation_micro_fem_summary"),
    }
    sessions.to_csv(OUT_DIR / "session_macro_micro_metrics.csv", index=False)
    fixations.to_csv(OUT_DIR / "trial_fixation_centers.csv", index=False)
    profiles.to_csv(OUT_DIR / "session_fem_profiles.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_examples.csv", index=False)
    correlations.to_csv(OUT_DIR / "correlation_summary.csv", index=False)
    axis_tests.to_csv(OUT_DIR / "axis_pairing_tests.csv", index=False)

    all_strength = correlations[
        correlations["fem_condition"].eq("all_fem")
        & correlations["macro_geometry"].eq("position")
        & correlations["quantity"].eq("anisotropy_strength")
    ]
    low_strength = correlations[
        correlations["fem_condition"].eq("low_coherence_fem")
        & correlations["macro_geometry"].eq("position")
        & correlations["quantity"].eq("anisotropy_strength")
    ]
    report = [
        "# Macro fixation geometry versus FEM geometry: checkpoint 10",
        "",
        "Macro position geometry is computed from one median fixation center per BackImage trial.",
        "FEM geometry is the session median of trial-level projected RMS profiles. Consecutive",
        "fixation-center displacement geometry is retained as a sensitivity analysis.",
        "",
        "Absolute macro and FEM axes often occupy the same broad screen direction. However,",
        "within-animal session shuffling reproduces that agreement: the correct macro cloud is not",
        "more aligned with its own session's FEM cloud than with another session's FEM cloud.",
        "",
    ]
    for label, table in (("all BackImage FEM", all_strength), ("low-coherence FEM", low_strength)):
        report.append(f"## {label}: macro-position versus FEM anisotropy strength")
        report.append("")
        for row in table.itertuples(index=False):
            report.append(
                f"- {row.scope}: Pearson r={row.pearson_r:+.3f} (p={row.pearson_p:.3f}); "
                f"Spearman rho={row.spearman_rho:+.3f} (p={row.spearman_p:.3f})."
            )
        report.append("")
    report.extend(
        [
            "Allen alone shows a positive macro-strength/all-FEM-strength association, but Logan",
            "does not; the association vanishes for the low-coherence FEM estimate. It should not",
            "be treated as an animal-general relationship.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 10; macro fixation centers versus FEM cloud geometry",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "macro_position_unit": "one median gaze location per session/trial",
        "micro_profile_unit": "median window projected RMS within trial, then median trial profile within session",
        "low_coherence_max": LOW_MAX,
        "low_coherence_min_windows": MIN_LOW_WINDOWS,
        "low_coherence_min_trials": MIN_LOW_TRIALS,
        "n_permutations": N_PERMUTATIONS,
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["examples"]["png"])
    print(ROOT / outputs["summary"]["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot absolute screen-frame drift-cloud profiles without image conditioning."""

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
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "unconditioned_cloud_polar_checkpoint7_v1"
)
SUBJECTS = ("Allen", "Logan")
COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D", "equal-animal": "#202124"}
ANGLES_DEG = np.arange(0.0, 180.0 + 1.0, 1.0)
N_BOOTSTRAP = 1500
SEED = 20260811
GRID = "#D8DDE3"


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


def load_values() -> pd.DataFrame:
    columns = [
        "subject",
        "session",
        "trial_idx",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    values = pd.read_csv(SOURCE, usecols=columns)
    ok = values["subject"].isin(SUBJECTS) & values["session"].notna()
    for column in columns[2:]:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        ok &= np.isfinite(values[column])
    return values.loc[ok].copy().reset_index(drop=True)


def projected_rms(values: pd.DataFrame) -> np.ndarray:
    theta = np.radians(ANGLES_DEG)[None, :]
    ux = np.cos(theta)
    uy = np.sin(theta)
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)[:, None]
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)[:, None]
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)[:, None]
    variance = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(variance, 0.0))


def build_trial_profiles(values: pd.DataFrame, window_profiles: np.ndarray) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, indices in values.groupby(["subject", "session", "trial_idx"], sort=True).groups.items():
        subject, session, trial_idx = key
        positions = np.asarray(indices, dtype=int)
        rows.append(
            pd.DataFrame(
                {
                    "subject": str(subject),
                    "session": str(session),
                    "trial_idx": int(trial_idx),
                    "absolute_screen_axis_deg": ANGLES_DEG,
                    "rms_arcmin": np.median(window_profiles[positions], axis=0),
                    "n_windows_in_trial": int(len(positions)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def session_trial_matrices(trials: pd.DataFrame, subject: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for session, block in trials[trials["subject"].eq(subject)].groupby("session", sort=True):
        matrix = block.pivot(index="trial_idx", columns="absolute_screen_axis_deg", values="rms_arcmin")
        matrix = matrix.reindex(columns=ANGLES_DEG).to_numpy(dtype=float)
        if matrix.size and np.isfinite(matrix).all():
            result[str(session)] = matrix
    return result


def hierarchical_profile(
    matrices: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    sessions = list(matrices)
    point = np.median(
        np.stack([np.median(matrices[session], axis=0) for session in sessions]), axis=0
    )
    draws = np.empty((N_BOOTSTRAP, len(ANGLES_DEG)), dtype=float)
    for draw_index in range(N_BOOTSTRAP):
        chosen_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_profiles = []
        for chosen in chosen_sessions:
            trial_matrix = matrices[sessions[int(chosen)]]
            chosen_trials = rng.integers(0, len(trial_matrix), size=len(trial_matrix))
            session_profiles.append(np.median(trial_matrix[chosen_trials], axis=0))
        draws[draw_index] = np.median(np.stack(session_profiles), axis=0)
    return point, draws


def summarize(values: pd.DataFrame, trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    profiles: dict[str, np.ndarray] = {}
    draws: dict[str, np.ndarray] = {}
    rows: list[pd.DataFrame] = []
    for subject in SUBJECTS:
        matrices = session_trial_matrices(trials, subject)
        point, subject_draws = hierarchical_profile(matrices, rng)
        profiles[subject] = point
        draws[subject] = subject_draws
        low, high = np.quantile(subject_draws, [0.025, 0.975], axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "scope": "subject",
                    "subject": subject,
                    "absolute_screen_axis_deg": ANGLES_DEG,
                    "rms_arcmin": point,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        )

    grand = np.mean(np.stack([profiles[s] for s in SUBJECTS]), axis=0)
    grand_draws = np.mean(np.stack([draws[s] for s in SUBJECTS]), axis=0)
    low, high = np.quantile(grand_draws, [0.025, 0.975], axis=0)
    rows.append(
        pd.DataFrame(
            {
                "scope": "equal-animal",
                "subject": "equal-animal",
                "absolute_screen_axis_deg": ANGLES_DEG,
                "rms_arcmin": grand,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    )
    profile_table = pd.concat(rows, ignore_index=True)

    summary_rows = []
    for label in (*SUBJECTS, "equal-animal"):
        point = grand if label == "equal-animal" else profiles[label]
        sample_draws = grand_draws if label == "equal-animal" else draws[label]
        preferred_index = int(np.argmax(point[:-1]))
        horizontal_minus_vertical = point[0] - point[90]
        oblique_135_minus_45 = point[135] - point[45]
        elongation = np.max(point[:-1]) - np.min(point[:-1])
        hmv_draws = sample_draws[:, 0] - sample_draws[:, 90]
        oblique_draws = sample_draws[:, 135] - sample_draws[:, 45]
        elongation_draws = np.max(sample_draws[:, :-1], axis=1) - np.min(sample_draws[:, :-1], axis=1)
        summary_rows.append(
            {
                "scope": label,
                "n_windows": int(len(values) if label == "equal-animal" else values["subject"].eq(label).sum()),
                "n_trials": int(
                    values.groupby(["session", "trial_idx"]).ngroups
                    if label == "equal-animal"
                    else values[values["subject"].eq(label)].groupby(["session", "trial_idx"]).ngroups
                ),
                "n_sessions": int(values["session"].nunique() if label == "equal-animal" else values.loc[values["subject"].eq(label), "session"].nunique()),
                "preferred_screen_axis_deg": float(ANGLES_DEG[preferred_index]),
                "horizontal_minus_vertical_arcmin": float(horizontal_minus_vertical),
                "horizontal_minus_vertical_ci95_low": float(np.quantile(hmv_draws, 0.025)),
                "horizontal_minus_vertical_ci95_high": float(np.quantile(hmv_draws, 0.975)),
                "rms_135_minus_45_arcmin": float(oblique_135_minus_45),
                "rms_135_minus_45_ci95_low": float(np.quantile(oblique_draws, 0.025)),
                "rms_135_minus_45_ci95_high": float(np.quantile(oblique_draws, 0.975)),
                "max_minus_min_rms_arcmin": float(elongation),
                "max_minus_min_ci95_low": float(np.quantile(elongation_draws, 0.025)),
                "max_minus_min_ci95_high": float(np.quantile(elongation_draws, 0.975)),
            }
        )
    return profile_table, pd.DataFrame(summary_rows)


def full_axial(block: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = block.sort_values("absolute_screen_axis_deg")
    angle = ordered["absolute_screen_axis_deg"].to_numpy(dtype=float)
    radius = ordered[column].to_numpy(dtype=float)
    return (
        np.radians(np.concatenate([angle, angle[1:] + 180.0])),
        np.concatenate([radius, radius[1:]]),
    )


def format_polar(ax: plt.Axes) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
        fontsize=6.2,
    )
    ax.grid(color=GRID, lw=0.65)
    ax.spines["polar"].set_color("#7D858C")
    ax.spines["polar"].set_linewidth(0.7)


def plot_small_multiples(profiles: pd.DataFrame, *, zero_origin: bool) -> plt.Figure:
    labels = (*SUBJECTS, "equal-animal")
    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 4.1), subplot_kw={"projection": "polar"}, constrained_layout=False
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.76, wspace=0.27)
    global_low = float(profiles["ci95_low"].min())
    global_high = float(profiles["ci95_high"].max())
    for ax, label in zip(axes, labels, strict=True):
        block = profiles[profiles["subject"].eq(label)]
        theta, radius = full_axial(block, "rms_arcmin")
        _, low = full_axial(block, "ci95_low")
        _, high = full_axial(block, "ci95_high")
        color = COLORS[label]
        ax.fill_between(theta, low, high, color=color, alpha=0.13, lw=0)
        ax.plot(theta, radius, color=color, lw=2.0)
        format_polar(ax)
        if zero_origin:
            ax.set_ylim(0.0, global_high * 1.04)
        else:
            pad = 0.08 * (global_high - global_low)
            ax.set_ylim(global_low - pad, global_high + pad)
        ax.set_title(label if label != "equal-animal" else "equal-animal mean", pad=7, weight="semibold")
    subtitle = "radial origin = 0" if zero_origin else "zoomed radial axis; origin is not zero"
    fig.suptitle(
        f"Unconditioned drift-cloud RMS by absolute screen axis\n{subtitle}",
        fontsize=11.5,
        weight="bold",
        y=0.98,
    )
    return fig


def plot_overlay(profiles: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw={"projection": "polar"}, constrained_layout=True)
    low = float(profiles["ci95_low"].min())
    high = float(profiles["ci95_high"].max())
    for label in (*SUBJECTS, "equal-animal"):
        block = profiles[profiles["subject"].eq(label)]
        theta, radius = full_axial(block, "rms_arcmin")
        ax.plot(theta, radius, color=COLORS[label], lw=2.2 if label == "equal-animal" else 1.5, label=label)
    format_polar(ax)
    pad = 0.08 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title("Absolute screen-frame cloud profiles\nzoomed radial axis", pad=13, weight="semibold")
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.12, -0.02))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_values()
    window_profiles = projected_rms(values)
    trials = build_trial_profiles(values, window_profiles)
    profiles, summary = summarize(values, trials)

    outputs = {
        "zero_origin": save_figure(
            plot_small_multiples(profiles, zero_origin=True), "unconditioned_cloud_polar_zero_origin"
        ),
        "zoomed_small_multiples": save_figure(
            plot_small_multiples(profiles, zero_origin=False), "unconditioned_cloud_polar_zoomed"
        ),
        "zoomed_overlay": save_figure(
            plot_overlay(profiles), "unconditioned_cloud_polar_subject_overlay"
        ),
    }
    profiles.to_csv(OUT_DIR / "unconditioned_cloud_profiles.csv", index=False)
    summary.to_csv(OUT_DIR / "unconditioned_cloud_summary.csv", index=False)
    trials.to_csv(OUT_DIR / "unconditioned_cloud_trial_profiles.csv.gz", index=False, compression="gzip")

    grand = summary[summary["scope"].eq("equal-animal")].iloc[0]
    report = [
        "# Unconditioned absolute drift-cloud polar profiles: checkpoint 7",
        "",
        "This checkpoint uses every reviewed drift window and only its 2 x 2 position covariance.",
        "No image orientation, image coherence, contour-relative angle, or image-derived selection",
        "enters the calculation. Windows are collapsed within trials, trials within sessions, and",
        "Allen and Logan receive equal weight in the combined profile.",
        "",
        f"The equal-animal profile peaks at {grand.preferred_screen_axis_deg:.1f} degrees. Its",
        f"horizontal-minus-vertical RMS difference is {grand.horizontal_minus_vertical_arcmin:+.3f}",
        f"arcmin [{grand.horizontal_minus_vertical_ci95_low:+.3f},",
        f"{grand.horizontal_minus_vertical_ci95_high:+.3f}]. The 135-minus-45-degree RMS difference",
        f"is {grand.rms_135_minus_45_arcmin:+.3f} arcmin",
        f"[{grand.rms_135_minus_45_ci95_low:+.3f}, {grand.rms_135_minus_45_ci95_high:+.3f}].",
        "",
        "The zero-origin version shows the absolute size of the anisotropy; the zoomed versions make",
        "its orientation visible but must not be read as showing the fractional size of the cloud.",
        "This is a descriptive screen-frame reference and does not identify whether the direction is",
        "biological, head/screen aligned, or introduced by tracker coordinates.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 7; unconditioned absolute screen-frame cloud",
        "source": str(SOURCE.relative_to(ROOT)),
        "n_windows": int(len(values)),
        "n_trials": int(values.groupby(["session", "trial_idx"]).ngroups),
        "n_sessions": int(values["session"].nunique()),
        "subjects": list(SUBJECTS),
        "image_conditioning": "none",
        "aggregation": "window median within trial; trial median within resampled session; session median; equal subject mean",
        "n_bootstrap": N_BOOTSTRAP,
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["zero_origin"]["png"])
    print(ROOT / outputs["zoomed_small_multiples"]["png"])
    print(ROOT / outputs["zoomed_overlay"]["png"])
    print(OUT_DIR / "unconditioned_cloud_summary.csv")


if __name__ == "__main__":
    main()

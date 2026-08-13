#!/usr/bin/env python3
"""Build the updated descriptive Figure 4F contour-relative spread panel.

The displayed profiles are recomputed from individual reviewed BackImage
windows. Repeated windows are collapsed within trials, the point estimate is
hierarchical across trials and sessions, and Allen and Logan receive equal
weight. The panel intentionally contains no orientation-randomization or
spatial-offset reference: those remain robustness analyses rather than part of
this descriptive main-figure estimand.
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
SOURCE_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
)
SOURCE_WINDOWS = SOURCE_DIR / "contour_motion_component_windows.csv"
OLD_DISPLAY_VALUES = (
    ROOT
    / "outputs"
    / "fig"
    / "ssi_figure_v2"
    / "panels"
    / "panel_h_unwrapped_edge_coherence_values.csv"
)
OUT_DIR = (
    ROOT
    / "outputs"
    / "fig"
    / "ssi_figure_v2"
    / "behavior_confounds_map_first_v1"
    / "panel_f_descriptive_hierarchical_profiles_v1"
)

COHERENCE_BANDS = (
    (0.0, 0.2, "0–0.2", "0-0.2"),
    (0.2, 0.5, "0.2–0.5", "0.2-0.5"),
    (0.5, 1.0, "0.5–1", "0.5-1"),
)
ANGLES_DEG = np.arange(0.0, 180.0 + 1.875, 3.75, dtype=float)
SUBJECT_ORDER = ("Allen", "Logan")
N_BOOTSTRAP = 1000
SEED = 47

COLORS = ("#8DAF8C", "#4E8C68", "#0E4E3D")
GRID = "#D8DDE3"
INK = "#202124"


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


def load_windows() -> pd.DataFrame:
    values = pd.read_csv(SOURCE_WINDOWS)
    required = {
        "subject",
        "session",
        "trial_idx",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing required columns from {SOURCE_WINDOWS}: {missing}")

    numeric = sorted(required.difference({"subject", "session"}))
    ok = values["subject"].isin(SUBJECT_ORDER) & values["session"].notna()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        ok &= np.isfinite(values[column])
    values = values.loc[ok].copy().reset_index(drop=True)

    coherence = values["image_orientation_coherence"].to_numpy(dtype=float)
    band_index = np.full(len(values), -1, dtype=int)
    for index, (low, high, _label, _id) in enumerate(COHERENCE_BANDS):
        upper_ok = coherence <= high if np.isclose(high, 1.0) else coherence < high
        band_index[(coherence >= low) & upper_ok] = index
    values["coherence_band_index"] = band_index
    values = values[values["coherence_band_index"] >= 0].copy().reset_index(drop=True)
    return values


def directional_rms(values: pd.DataFrame) -> np.ndarray:
    theta = np.radians(
        values["image_edge_axis_deg"].to_numpy(dtype=float)[:, None]
        + ANGLES_DEG[None, :]
    )
    ux = np.cos(theta)
    uy = np.sin(theta)
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)[:, None]
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)[:, None]
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)[:, None]
    variance = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(variance, 0.0))


def build_trial_profiles(values: pd.DataFrame, window_profiles: np.ndarray) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_columns = ["subject", "session", "trial_idx", "coherence_band_index"]
    for key, index in values.groupby(group_columns, sort=True).groups.items():
        subject, session, trial_idx, band_index = key
        positions = np.asarray(index, dtype=int)
        profile = np.nanmedian(window_profiles[positions], axis=0)
        low, high, label, band_id = COHERENCE_BANDS[int(band_index)]
        rows.append(
            pd.DataFrame(
                {
                    "subject": str(subject),
                    "session": str(session),
                    "trial_idx": int(trial_idx),
                    "coherence_band_index": int(band_index),
                    "coherence_band": label,
                    "coherence_band_id": band_id,
                    "coherence_low": low,
                    "coherence_high": high,
                    "relative_angle_deg": ANGLES_DEG,
                    "rms_arcmin": profile,
                    "n_windows_in_trial_band": int(len(positions)),
                }
            )
        )
    if not rows:
        raise ValueError("No trial profiles were produced")
    return pd.concat(rows, ignore_index=True)


def _session_trial_matrices(trial_profiles: pd.DataFrame, subject: str, band_index: int) -> dict[str, np.ndarray]:
    block = trial_profiles[
        trial_profiles["subject"].eq(subject)
        & trial_profiles["coherence_band_index"].eq(band_index)
    ]
    matrices: dict[str, np.ndarray] = {}
    for session, session_block in block.groupby("session", sort=True):
        matrix = session_block.pivot(
            index="trial_idx", columns="relative_angle_deg", values="rms_arcmin"
        )
        matrix = matrix.reindex(columns=ANGLES_DEG)
        array = matrix.to_numpy(dtype=float)
        if array.size and np.isfinite(array).all():
            matrices[str(session)] = array
    return matrices


def _hierarchical_point(session_trials: dict[str, np.ndarray]) -> np.ndarray:
    session_profiles = [np.median(trials, axis=0) for trials in session_trials.values()]
    if not session_profiles:
        return np.full(ANGLES_DEG.size, np.nan)
    return np.median(np.stack(session_profiles, axis=0), axis=0)


def _hierarchical_draws(
    session_trials: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sessions = list(session_trials)
    if not sessions:
        return np.full((n_bootstrap, ANGLES_DEG.size), np.nan)
    draws = np.empty((n_bootstrap, ANGLES_DEG.size), dtype=float)
    for draw_index in range(n_bootstrap):
        selected_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_profiles = []
        for selected in selected_sessions:
            trials = session_trials[sessions[int(selected)]]
            selected_trials = rng.integers(0, len(trials), size=len(trials))
            session_profiles.append(np.median(trials[selected_trials], axis=0))
        draws[draw_index] = np.median(np.stack(session_profiles, axis=0), axis=0)
    return draws


def summarize_profiles(
    values: pd.DataFrame,
    window_profiles: np.ndarray,
    trial_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    profile_rows: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, float | int | str]] = []
    support_rows: list[dict[str, float | int | str]] = []

    for band_index, (low, high, label, band_id) in enumerate(COHERENCE_BANDS):
        band_window_mask = values["coherence_band_index"].eq(band_index).to_numpy()
        pooled_profile = np.nanmedian(window_profiles[band_window_mask], axis=0)
        subject_points: dict[str, np.ndarray] = {}
        subject_draws: dict[str, np.ndarray] = {}

        for subject in SUBJECT_ORDER:
            matrices = _session_trial_matrices(trial_profiles, subject, band_index)
            point = _hierarchical_point(matrices)
            draws = _hierarchical_draws(matrices, n_bootstrap=N_BOOTSTRAP, rng=rng)
            subject_points[subject] = point
            subject_draws[subject] = draws
            lo_ci, hi_ci = np.nanquantile(draws, [0.025, 0.975], axis=0)
            profile_rows.append(
                pd.DataFrame(
                    {
                        "scope": "subject",
                        "subject": subject,
                        "coherence_band_index": band_index,
                        "coherence_band": label,
                        "coherence_band_id": band_id,
                        "coherence_low": low,
                        "coherence_high": high,
                        "relative_angle_deg": ANGLES_DEG,
                        "rms_arcmin": point,
                        "ci95_low": lo_ci,
                        "ci95_high": hi_ci,
                        "pooled_window_median_rms_arcmin": pooled_profile,
                    }
                )
            )

            parallel = 0.5 * (point[0] + point[-1])
            delta_draws = 0.5 * (draws[:, 0] + draws[:, -1]) - draws[:, ANGLES_DEG.tolist().index(90.0)]
            delta_ci = np.nanquantile(delta_draws, [0.025, 0.975])
            contrast_rows.append(
                {
                    "scope": "subject",
                    "subject": subject,
                    "coherence_band_index": band_index,
                    "coherence_band": label,
                    "parallel_rms_arcmin": float(parallel),
                    "orthogonal_rms_arcmin": float(point[ANGLES_DEG.tolist().index(90.0)]),
                    "parallel_minus_orthogonal_arcmin": float(
                        parallel - point[ANGLES_DEG.tolist().index(90.0)]
                    ),
                    "ci95_low": float(delta_ci[0]),
                    "ci95_high": float(delta_ci[1]),
                }
            )

            subject_windows = values[
                values["subject"].eq(subject)
                & values["coherence_band_index"].eq(band_index)
            ]
            subject_trials = trial_profiles[
                trial_profiles["subject"].eq(subject)
                & trial_profiles["coherence_band_index"].eq(band_index)
            ][["session", "trial_idx"]].drop_duplicates()
            support_rows.append(
                {
                    "subject": subject,
                    "coherence_band_index": band_index,
                    "coherence_band": label,
                    "n_windows": int(len(subject_windows)),
                    "n_trials": int(len(subject_trials)),
                    "n_sessions": int(subject_trials["session"].nunique()),
                }
            )

        grand_point = np.mean(np.stack([subject_points[s] for s in SUBJECT_ORDER]), axis=0)
        grand_draws = np.mean(np.stack([subject_draws[s] for s in SUBJECT_ORDER]), axis=0)
        grand_lo, grand_hi = np.nanquantile(grand_draws, [0.025, 0.975], axis=0)
        profile_rows.append(
            pd.DataFrame(
                {
                    "scope": "grand_equal_subject",
                    "subject": "equal_subject_mean",
                    "coherence_band_index": band_index,
                    "coherence_band": label,
                    "coherence_band_id": band_id,
                    "coherence_low": low,
                    "coherence_high": high,
                    "relative_angle_deg": ANGLES_DEG,
                    "rms_arcmin": grand_point,
                    "ci95_low": grand_lo,
                    "ci95_high": grand_hi,
                    "pooled_window_median_rms_arcmin": pooled_profile,
                }
            )
        )
        orthogonal_index = ANGLES_DEG.tolist().index(90.0)
        parallel = 0.5 * (grand_point[0] + grand_point[-1])
        grand_delta_draws = 0.5 * (grand_draws[:, 0] + grand_draws[:, -1]) - grand_draws[:, orthogonal_index]
        grand_delta_ci = np.nanquantile(grand_delta_draws, [0.025, 0.975])
        contrast_rows.append(
            {
                "scope": "grand_equal_subject",
                "subject": "equal_subject_mean",
                "coherence_band_index": band_index,
                "coherence_band": label,
                "parallel_rms_arcmin": float(parallel),
                "orthogonal_rms_arcmin": float(grand_point[orthogonal_index]),
                "parallel_minus_orthogonal_arcmin": float(parallel - grand_point[orthogonal_index]),
                "ci95_low": float(grand_delta_ci[0]),
                "ci95_high": float(grand_delta_ci[1]),
            }
        )

    return (
        pd.concat(profile_rows, ignore_index=True),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(support_rows),
    )


def old_vs_exact_diagnostic(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    if not OLD_DISPLAY_VALUES.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_DISPLAY_VALUES)
    exact = profiles[profiles["scope"].eq("grand_equal_subject")][
        ["coherence_band_id", "relative_angle_deg", "pooled_window_median_rms_arcmin"]
    ].copy()
    merged = old.merge(
        exact,
        left_on=["wide_coherence_bin", "relative_angle_deg"],
        right_on=["coherence_band_id", "relative_angle_deg"],
        how="inner",
    )
    merged["old_minus_exact_arcmin"] = (
        merged["rms_arcmin"] - merged["pooled_window_median_rms_arcmin"]
    )
    return merged


def draw_panel(ax: plt.Axes, profiles: pd.DataFrame) -> None:
    grand = profiles[profiles["scope"].eq("grand_equal_subject")]
    for band_index, color in enumerate(COLORS):
        block = grand[grand["coherence_band_index"].eq(band_index)].sort_values(
            "relative_angle_deg"
        )
        if block.empty:
            continue
        x = block["relative_angle_deg"].to_numpy(dtype=float)
        y = block["rms_arcmin"].to_numpy(dtype=float)
        lo = block["ci95_low"].to_numpy(dtype=float)
        hi = block["ci95_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.055, linewidth=0, zorder=1)
        ax.plot(
            x,
            y,
            color=color,
            lw=1.8,
            label=str(block["coherence_band"].iloc[0]),
            zorder=2,
        )

    ax.axvline(90.0, color="#7D858C", lw=0.75, ls=":", zorder=0)
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    labels = ax.get_xticklabels()
    labels[0].set_ha("left")
    labels[-1].set_ha("right")
    ax.set_xlabel("Angle from local edge")
    ax.set_ylabel("Position spread RMS (arcmin)")
    ax.set_title(
        "Spread anisotropy increases\nwith edge coherence",
        loc="left",
        color=INK,
        fontweight="semibold",
        linespacing=1.15,
        pad=8,
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.78,
        fontsize=6.2,
        title="edge coherence",
        title_fontsize=6.5,
        loc="upper left",
        handlelength=1.35,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    ax.text(
        0.98,
        0.02,
        "95% hierarchical bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color="#6B6F75",
    )
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def write_report(contrasts: pd.DataFrame, support: pd.DataFrame, path: Path) -> None:
    grand = contrasts[contrasts["scope"].eq("grand_equal_subject")].sort_values(
        "coherence_band_index"
    )
    lines = [
        "# Figure 4F descriptive hierarchical profile",
        "",
        "The panel shows only real fixation-centered contour-relative position-spread profiles.",
        "Every wide coherence band is recomputed from individual windows. Windows are collapsed",
        "within trials; sessions and trials are hierarchically resampled; Allen and Logan are",
        "held fixed and equally weighted.",
        "",
        "## Parallel-minus-orthogonal spread",
        "",
        "| coherence | difference (arcmin) | 95% CI |",
        "|---|---:|---|",
    ]
    for row in grand.itertuples(index=False):
        lines.append(
            f"| {row.coherence_band} | {row.parallel_minus_orthogonal_arcmin:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Support by animal",
            "",
            "| animal | coherence | windows | trials | sessions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in support.sort_values(["coherence_band_index", "subject"]).itertuples(index=False):
        lines.append(
            f"| {row.subject} | {row.coherence_band} | {row.n_windows} | "
            f"{row.n_trials} | {row.n_sessions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a descriptive association between local edge coherence and the shape of the",
            "fixation position cloud. It is not a matched-pair test and does not by itself separate",
            "local image-contingent behavior from shared global image and movement statistics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = load_windows()
    window_profiles = directional_rms(windows)
    trial_profiles = build_trial_profiles(windows, window_profiles)
    profiles, contrasts, support = summarize_profiles(windows, window_profiles, trial_profiles)
    diagnostic = old_vs_exact_diagnostic(profiles)

    trial_profiles.to_csv(OUT_DIR / "panel_f_trial_profiles.csv.gz", index=False, compression="gzip")
    profiles.to_csv(OUT_DIR / "panel_f_hierarchical_profiles.csv", index=False)
    contrasts.to_csv(OUT_DIR / "panel_f_parallel_minus_orthogonal.csv", index=False)
    support.to_csv(OUT_DIR / "panel_f_support.csv", index=False)
    if not diagnostic.empty:
        diagnostic.to_csv(OUT_DIR / "panel_f_old_vs_exact_wide_band_diagnostic.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.15, 3.0), constrained_layout=True)
    draw_panel(ax, profiles)
    outputs: dict[str, str] = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        output_path = OUT_DIR / f"panel_f_descriptive_hierarchical_profiles.{suffix}"
        fig.savefig(output_path, transparent=True, **kwargs)
        outputs[suffix] = str(output_path.relative_to(ROOT))
    plt.close(fig)

    write_report(contrasts, support, OUT_DIR / "summary_report.md")
    provenance = {
        "panel": "Figure 4F descriptive candidate",
        "source_windows": str(SOURCE_WINDOWS.relative_to(ROOT)),
        "n_windows": int(len(windows)),
        "n_trials": int(windows[["session", "trial_idx"]].drop_duplicates().shape[0]),
        "n_sessions": int(windows["session"].nunique()),
        "subjects": list(SUBJECT_ORDER),
        "coherence_bands": [
            {"low": low, "high": high, "label": label}
            for low, high, label, _band_id in COHERENCE_BANDS
        ],
        "angles_deg": ANGLES_DEG.tolist(),
        "point_estimator": (
            "median windows within trial; median trials within session; median sessions within "
            "subject; arithmetic mean of the two fixed subjects"
        ),
        "uncertainty": (
            f"{N_BOOTSTRAP} hierarchical bootstrap draws resampling sessions within subject and "
            "trials within selected session; fixed subjects equally weighted"
        ),
        "seed": SEED,
        "controls_intentionally_not_displayed": [
            "uniform orientation randomization",
            "matched real-pair reassignment",
            "same-image offset patches",
        ],
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for output in outputs.values():
        print(ROOT / output)
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

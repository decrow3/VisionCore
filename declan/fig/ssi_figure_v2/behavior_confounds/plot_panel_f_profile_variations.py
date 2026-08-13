#!/usr/bin/env python3
"""Render map-first visual variations of the updated Figure 4F profiles."""

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
    / "fig"
    / "ssi_figure_v2"
    / "behavior_confounds_map_first_v1"
    / "panel_f_descriptive_hierarchical_profiles_v1"
)
PROFILE_CSV = SOURCE_DIR / "panel_f_hierarchical_profiles.csv"
CONTRAST_CSV = SOURCE_DIR / "panel_f_parallel_minus_orthogonal.csv"
TRIAL_CSV = SOURCE_DIR / "panel_f_trial_profiles.csv.gz"
OUT_DIR = SOURCE_DIR / "visual_variations_v1"

COLORS = {"0–0.2": "#8DAF8C", "0.2–0.5": "#4E8C68", "0.5–1": "#0E4E3D"}
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
SUBJECT_ORDER = ("Allen", "Logan")
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


def load_values() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_csv(PROFILE_CSV)
    contrasts = pd.read_csv(CONTRAST_CSV)
    trials = pd.read_csv(TRIAL_CSV)
    expected_bands = set(COLORS)
    if set(profiles["coherence_band"].unique()) != expected_bands:
        raise ValueError("Profile coherence bands do not match the visual contract")
    return profiles, contrasts, trials


def grand_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    return profiles[profiles["scope"].eq("grand_equal_subject")].copy()


def _full_axial_profile(block: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = block.sort_values("relative_angle_deg")
    angle = ordered["relative_angle_deg"].to_numpy(dtype=float)
    radius = ordered[value].to_numpy(dtype=float)
    full_angle = np.concatenate([angle, angle[1:] + 180.0])
    full_radius = np.concatenate([radius, radius[1:]])
    return np.radians(full_angle), full_radius


def _format_unwrapped(ax: plt.Axes, *, ylabel: str = "Position spread RMS (arcmin)") -> None:
    ax.axvline(90.0, color="#7D858C", lw=0.75, ls=":", zorder=0)
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_ha("left")
    ticklabels[-1].set_ha("right")
    ax.set_xlabel("Angle from local edge")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_unwrapped_absolute(ax: plt.Axes, profiles: pd.DataFrame, *, legend: bool = True) -> None:
    grand = grand_profiles(profiles)
    for band, color in COLORS.items():
        block = grand[grand["coherence_band"].eq(band)].sort_values("relative_angle_deg")
        x = block["relative_angle_deg"].to_numpy(dtype=float)
        y = block["rms_arcmin"].to_numpy(dtype=float)
        lo = block["ci95_low"].to_numpy(dtype=float)
        hi = block["ci95_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.055, lw=0)
        ax.plot(x, y, color=color, lw=1.8, label=band)
    _format_unwrapped(ax)
    ax.set_title("Absolute contour-relative profiles", loc="left", weight="semibold")
    if legend:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.8,
            fontsize=6.2,
            title="edge coherence",
            title_fontsize=6.5,
            loc="upper left",
        )


def _format_polar(ax: plt.Axes) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=["parallel", "orthogonal", "parallel", "orthogonal"],
        fontsize=6.5,
    )
    ax.grid(color=GRID, lw=0.7)
    ax.spines["polar"].set_color("#7D858C")
    ax.spines["polar"].set_linewidth(0.7)


def draw_polar_overlay(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    *,
    zero_origin: bool,
    legend: bool = True,
) -> None:
    grand = grand_profiles(profiles)
    all_lo: list[np.ndarray] = []
    all_hi: list[np.ndarray] = []
    for band, color in COLORS.items():
        block = grand[grand["coherence_band"].eq(band)]
        theta, y = _full_axial_profile(block, "rms_arcmin")
        _, lo = _full_axial_profile(block, "ci95_low")
        _, hi = _full_axial_profile(block, "ci95_high")
        all_lo.append(lo)
        all_hi.append(hi)
        ax.fill_between(theta, lo, hi, color=color, alpha=0.05, lw=0)
        ax.plot(theta, y, color=color, lw=1.65, label=band)
    _format_polar(ax)
    lo_value = float(np.nanmin(np.concatenate(all_lo)))
    hi_value = float(np.nanmax(np.concatenate(all_hi)))
    if zero_origin:
        ax.set_ylim(0.0, hi_value * 1.05)
        ax.set_title("Polar profiles (zero origin)", fontsize=9, weight="semibold", pad=11)
    else:
        pad = max(0.04, 0.06 * (hi_value - lo_value))
        ax.set_ylim(lo_value - pad, hi_value + pad)
        ax.set_title("Polar profiles (zoomed radius)", fontsize=9, weight="semibold", pad=11)
        ax.text(
            0.5,
            -0.10,
            "nonzero radial origin",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.2,
            color="#6B6F75",
        )
    if legend:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.8,
            fontsize=6.0,
            loc="lower left",
            bbox_to_anchor=(-0.08, -0.05),
        )


def draw_orthogonal_centered(ax: plt.Axes, profiles: pd.DataFrame) -> None:
    grand = grand_profiles(profiles)
    for band, color in COLORS.items():
        block = grand[grand["coherence_band"].eq(band)].sort_values("relative_angle_deg")
        x = block["relative_angle_deg"].to_numpy(dtype=float)
        y = block["rms_arcmin"].to_numpy(dtype=float)
        orthogonal = float(block.loc[np.isclose(block["relative_angle_deg"], 90.0), "rms_arcmin"].iloc[0])
        ax.plot(x, y - orthogonal, color=color, lw=1.8, label=band)
    ax.axhline(0.0, color="#7D858C", lw=0.75, ls=":")
    _format_unwrapped(ax, ylabel="Spread relative to orthogonal (arcmin)")
    ax.set_title("Shape after removing orthogonal level", loc="left", weight="semibold")
    ax.legend(frameon=False, fontsize=6.2, title="edge coherence", title_fontsize=6.5)
    ax.text(
        0.98,
        0.03,
        "derived point-estimate view; no CI shown",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.7,
        color="#6B6F75",
    )


def draw_subject_profiles(ax: plt.Axes, profiles: pd.DataFrame, band: str) -> None:
    color = COLORS[band]
    for subject, linestyle in zip(SUBJECT_ORDER, ("-", "--"), strict=True):
        block = profiles[
            profiles["scope"].eq("subject")
            & profiles["subject"].eq(subject)
            & profiles["coherence_band"].eq(band)
        ].sort_values("relative_angle_deg")
        ax.plot(
            block["relative_angle_deg"],
            block["rms_arcmin"],
            color=SUBJECT_COLORS[subject],
            lw=1.35,
            ls=linestyle,
            label=subject,
        )
    grand = profiles[
        profiles["scope"].eq("grand_equal_subject")
        & profiles["coherence_band"].eq(band)
    ].sort_values("relative_angle_deg")
    ax.plot(grand["relative_angle_deg"], grand["rms_arcmin"], color=color, lw=2.1, label="equal mean")
    _format_unwrapped(ax)
    ax.set_title(f"coherence {band}", fontsize=8.5, weight="semibold")


def session_delta_values(trials: pd.DataFrame) -> pd.DataFrame:
    endpoints = trials[np.isclose(trials["relative_angle_deg"], 0.0) | np.isclose(trials["relative_angle_deg"], 90.0) | np.isclose(trials["relative_angle_deg"], 180.0)]
    wide = endpoints.pivot(
        index=["subject", "session", "trial_idx", "coherence_band_index", "coherence_band"],
        columns="relative_angle_deg",
        values="rms_arcmin",
    ).reset_index()
    wide["parallel_minus_orthogonal_arcmin"] = 0.5 * (wide[0.0] + wide[180.0]) - wide[90.0]
    session = (
        wide.groupby(
            ["subject", "session", "coherence_band_index", "coherence_band"],
            as_index=False,
        )
        .agg(
            parallel_minus_orthogonal_arcmin=("parallel_minus_orthogonal_arcmin", "median"),
            n_trials=("trial_idx", "nunique"),
        )
    )
    return session


def draw_session_deltas(
    ax: plt.Axes,
    session_values: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(11)
    bands = list(COLORS)
    for band_index, band in enumerate(bands):
        for subject_index, subject in enumerate(SUBJECT_ORDER):
            block = session_values[
                session_values["coherence_band"].eq(band)
                & session_values["subject"].eq(subject)
            ]
            center = band_index + (-0.13 if subject_index == 0 else 0.13)
            jitter = rng.uniform(-0.045, 0.045, size=len(block))
            ax.scatter(
                center + jitter,
                block["parallel_minus_orthogonal_arcmin"],
                s=12,
                alpha=0.62,
                color=SUBJECT_COLORS[subject],
                edgecolor="white",
                linewidth=0.25,
                label=subject if band_index == 0 else None,
                zorder=2,
            )
        grand = contrasts[
            contrasts["scope"].eq("grand_equal_subject")
            & contrasts["coherence_band"].eq(band)
        ].iloc[0]
        ax.errorbar(
            band_index,
            grand["parallel_minus_orthogonal_arcmin"],
            yerr=np.array(
                [
                    [grand["parallel_minus_orthogonal_arcmin"] - grand["ci95_low"]],
                    [grand["ci95_high"] - grand["parallel_minus_orthogonal_arcmin"]],
                ]
            ),
            fmt="D",
            ms=4.2,
            color=COLORS[band],
            mec="white",
            mew=0.45,
            elinewidth=1.1,
            zorder=4,
        )
    ax.axhline(0.0, color="#7D858C", lw=0.75, ls=":", zorder=0)
    ax.set_xticks(range(len(bands)), bands)
    ax.set_xlabel("Edge coherence")
    ax.set_ylabel("Parallel − orthogonal spread (arcmin)")
    ax.set_title("Session heterogeneity", loc="left", weight="semibold")
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")


def draw_endpoint_summary(ax: plt.Axes, profiles: pd.DataFrame) -> None:
    grand = grand_profiles(profiles)
    bands = list(COLORS)
    x = np.arange(len(bands), dtype=float)
    for angle, label, marker, offset in (
        (0.0, "parallel", "o", -0.05),
        (90.0, "orthogonal", "s", 0.05),
    ):
        points = []
        lows = []
        highs = []
        for band in bands:
            row = grand[
                grand["coherence_band"].eq(band)
                & np.isclose(grand["relative_angle_deg"], angle)
            ].iloc[0]
            points.append(float(row["rms_arcmin"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))
        points_array = np.asarray(points)
        ax.errorbar(
            x + offset,
            points_array,
            yerr=np.vstack([points_array - lows, np.asarray(highs) - points_array]),
            marker=marker,
            markersize=4.2,
            lw=1.2,
            capsize=0,
            color="#3F4A52" if angle == 0.0 else "#9BA5AC",
            label=label,
        )
    ax.set_xticks(x, bands)
    ax.set_xlabel("Edge coherence")
    ax.set_ylabel("Position spread RMS (arcmin)")
    ax.set_title("Parallel and orthogonal endpoints", loc="left", weight="semibold")
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.2)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 240) -> dict[str, Path]:
    paths = {}
    for suffix, kwargs in (("png", {"dpi": dpi}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = path
    plt.close(fig)
    return paths


def render_all(
    profiles: pd.DataFrame,
    contrasts: pd.DataFrame,
    trials: pd.DataFrame,
) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    session_values = session_delta_values(trials)
    session_values.to_csv(OUT_DIR / "panel_f_session_delta_values.csv", index=False)

    for zero_origin, stem in (
        (True, "panel_f_polar_overlay_zero_origin"),
        (False, "panel_f_polar_overlay_zoomed"),
    ):
        fig, ax = plt.subplots(figsize=(3.45, 3.45), subplot_kw={"projection": "polar"}, constrained_layout=True)
        draw_polar_overlay(ax, profiles, zero_origin=zero_origin)
        outputs[stem] = save_figure(fig, stem)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.1, 3.15),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    grand = grand_profiles(profiles)
    max_hi = float(grand["ci95_high"].max()) * 1.05
    for ax, (band, color) in zip(axes, COLORS.items(), strict=True):
        block = grand[grand["coherence_band"].eq(band)]
        theta, y = _full_axial_profile(block, "rms_arcmin")
        _, lo = _full_axial_profile(block, "ci95_low")
        _, hi = _full_axial_profile(block, "ci95_high")
        ax.fill_between(theta, lo, hi, color=color, alpha=0.08, lw=0)
        ax.plot(theta, y, color=color, lw=1.8)
        _format_polar(ax)
        ax.set_ylim(0.0, max_hi)
        ax.set_title(f"coherence {band}", fontsize=8.5, weight="semibold", pad=10)
    outputs["panel_f_polar_small_multiples_zero_origin"] = save_figure(
        fig, "panel_f_polar_small_multiples_zero_origin"
    )

    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    draw_orthogonal_centered(ax, profiles)
    outputs["panel_f_unwrapped_orthogonal_centered"] = save_figure(
        fig, "panel_f_unwrapped_orthogonal_centered"
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True, constrained_layout=True)
    for ax, band in zip(axes, COLORS, strict=True):
        draw_subject_profiles(ax, profiles, band)
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper left")
    for ax in axes[1:]:
        ax.set_ylabel("")
    outputs["panel_f_subject_profiles"] = save_figure(fig, "panel_f_subject_profiles")

    fig, ax = plt.subplots(figsize=(3.55, 3.0), constrained_layout=True)
    draw_session_deltas(ax, session_values, contrasts)
    outputs["panel_f_session_delta_distributions"] = save_figure(
        fig, "panel_f_session_delta_distributions"
    )

    fig, ax = plt.subplots(figsize=(3.55, 3.0), constrained_layout=True)
    draw_endpoint_summary(ax, profiles)
    outputs["panel_f_endpoint_summary"] = save_figure(fig, "panel_f_endpoint_summary")

    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], projection="polar")
    ax_c = fig.add_subplot(grid[0, 2], projection="polar")
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])
    draw_unwrapped_absolute(ax_a, profiles)
    draw_polar_overlay(ax_b, profiles, zero_origin=True)
    draw_polar_overlay(ax_c, profiles, zero_origin=False)
    draw_orthogonal_centered(ax_d, profiles)
    draw_endpoint_summary(ax_e, profiles)
    draw_session_deltas(ax_f, session_values, contrasts)
    for label, ax in zip("ABCDEF", (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f), strict=True):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    fig.suptitle("Figure 4F visual audit: absolute scale, shape, and heterogeneity", fontsize=13, weight="bold")
    outputs["panel_f_visual_compendium"] = save_figure(fig, "panel_f_visual_compendium", dpi=260)
    return outputs


def write_report(contrasts: pd.DataFrame, session_values: pd.DataFrame, path: Path) -> None:
    grand = contrasts[contrasts["scope"].eq("grand_equal_subject")].sort_values(
        "coherence_band_index"
    )
    lines = [
        "# Figure 4F visual-variation checkpoint",
        "",
        "This directory contains multiple views of the same hierarchical descriptive profiles.",
        "The variations change only the visualization, except for the explicitly labeled",
        "orthogonal-centered diagnostic and session-level summaries.",
        "",
        "## Visible readout",
        "",
        "- The zero-origin polar plots show that contour anisotropy is modest relative to total spread.",
        "- The zoomed polar and orthogonal-centered views make the parallel elongation most visible",
        "  in the 0.5-1 coherence band.",
        "- Allen shows a substantially deeper contour-relative profile than Logan at intermediate",
        "  and high coherence; the equal-animal curve lies between them.",
        "- Session-level contrasts overlap substantially across coherence bands and contain",
        "  heterogeneous positive and negative sessions.",
        "",
        "## Equal-animal profile contrasts",
        "",
        "| coherence | parallel - orthogonal (arcmin) | 95% CI |",
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
            "## View-specific guardrails",
            "",
            "- The zoomed polar plot has a nonzero radial origin and is only a shape diagnostic.",
            "- The orthogonal-centered plot subtracts the point estimate at 90 degrees and does not",
            "  display a confidence interval for that derived curve.",
            "- Polar plots repeat the axial 0-180 degree profile over 180-360 degrees; they do not",
            "  add independent observations.",
            "- Each session dot is the median trial-level parallel-minus-orthogonal contrast within",
            "  that session. The experiment still contains only two fixed animals.",
            "- None of these views is a matched-pair or causal locality test.",
            "",
            f"Session-value rows: {len(session_values)}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles, contrasts, trials = load_values()
    outputs = render_all(profiles, contrasts, trials)
    session_values = pd.read_csv(OUT_DIR / "panel_f_session_delta_values.csv")
    write_report(contrasts, session_values, OUT_DIR / "summary_report.md")
    provenance = {
        "purpose": "Map-first visual variations of the updated descriptive Figure 4F profiles",
        "source_profiles": str(PROFILE_CSV.relative_to(ROOT)),
        "source_contrasts": str(CONTRAST_CSV.relative_to(ROOT)),
        "source_trial_profiles": str(TRIAL_CSV.relative_to(ROOT)),
        "polar_zero_origin_note": "Radial origin is zero; preserves absolute scale.",
        "polar_zoomed_note": "Radial origin is nonzero; diagnostic view emphasizing profile shape.",
        "orthogonal_centered_note": (
            "Subtracts each point-estimate profile's value at 90 degrees; derived visualization with no CI."
        ),
        "session_delta_note": (
            "Each dot is the median trial-level parallel-minus-orthogonal profile contrast within a session."
        ),
        "outputs": {
            stem: {suffix: str(path.relative_to(ROOT)) for suffix, path in paths.items()}
            for stem, paths in outputs.items()
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for stem, paths in outputs.items():
        print(stem, paths["png"])


if __name__ == "__main__":
    main()

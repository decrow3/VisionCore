#!/usr/bin/env python3
"""Checkpoint 1 maps for gaze-position dependence of drift anisotropy.

This is a map-first descriptive render. It uses the same reviewed, event-free
BackImage drift windows as Figure 4F and keeps screen-frame, gaze-frame, and
axis-free anisotropy distinct. Central-to-peripheral summaries are preliminary
and are not a substitute for the planned hierarchical matched analysis.
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
    / "supp_gaze_position_anisotropy_checkpoint1_v1"
)

GRID_EDGES_DEG = np.arange(-12.0, 12.0 + 2.0, 2.0)
ECC_EDGES_DEG = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.01])
ECC_LABELS = ("0–2", "2–4", "4–6", "6–8", "8–10", "10–14")
MIN_GRID_WINDOWS = 20

SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
INK = "#202124"
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


def load_and_derive() -> pd.DataFrame:
    values = pd.read_csv(SOURCE_WINDOWS)
    required = {
        "subject",
        "session",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "rms_radius_deg",
        "anisotropy",
        "phase",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    numeric = sorted(required.difference({"subject", "session", "phase"}))
    ok = values["subject"].isin(["Allen", "Logan"]) & values["session"].notna()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        ok &= np.isfinite(values[column])
    values = values.loc[ok].copy().reset_index(drop=True)

    x = values["mean_x_deg"].to_numpy(dtype=float)
    y = values["mean_y_deg"].to_numpy(dtype=float)
    eccentricity = np.hypot(x, y)
    radial_x = np.divide(x, eccentricity, out=np.ones_like(x), where=eccentricity > 1e-12)
    radial_y = np.divide(y, eccentricity, out=np.zeros_like(y), where=eccentricity > 1e-12)
    tangent_x = -radial_y
    tangent_y = radial_x

    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)

    def projected_rms(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        variance = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
        return 60.0 * np.sqrt(np.maximum(variance, 0.0))

    horizontal = projected_rms(np.ones_like(x), np.zeros_like(x))
    vertical = projected_rms(np.zeros_like(x), np.ones_like(x))
    radial = projected_rms(radial_x, radial_y)
    tangential = projected_rms(tangent_x, tangent_y)

    trace = cxx + cyy
    discriminant = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4.0 * cxy**2, 0.0))
    lambda_major = 0.5 * (trace + discriminant)
    lambda_minor = 0.5 * (trace - discriminant)
    major_rms = 60.0 * np.sqrt(np.maximum(lambda_major, 0.0))
    minor_rms = 60.0 * np.sqrt(np.maximum(lambda_minor, 0.0))
    major_axis_rad = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)

    values["gaze_eccentricity_deg"] = eccentricity
    values["gaze_polar_angle_deg"] = np.degrees(np.arctan2(y, x))
    values["horizontal_rms_arcmin"] = horizontal
    values["vertical_rms_arcmin"] = vertical
    values["horizontal_minus_vertical_arcmin"] = horizontal - vertical
    values["radial_rms_arcmin"] = radial
    values["tangential_rms_arcmin"] = tangential
    values["tangential_minus_radial_arcmin"] = tangential - radial
    values["major_rms_arcmin"] = major_rms
    values["minor_rms_arcmin"] = minor_rms
    values["axis_free_anisotropy_arcmin"] = major_rms - minor_rms
    values["major_axis_deg"] = np.degrees(major_axis_rad)
    # Figure 4F's directional RMS values are projections of the stored sample
    # covariance. Use sqrt(trace(covariance)) here so total scale obeys the
    # same convention; retain the source population-RMS value for provenance.
    values["source_population_rms_radius_arcmin"] = (
        60.0 * values["rms_radius_deg"].to_numpy(dtype=float)
    )
    values["drift_rms_radius_arcmin"] = 60.0 * np.sqrt(np.maximum(trace, 0.0))
    values["eccentricity_bin"] = pd.cut(
        eccentricity,
        ECC_EDGES_DEG,
        labels=ECC_LABELS,
        include_lowest=True,
        right=False,
    )
    values = values[values["eccentricity_bin"].notna()].copy().reset_index(drop=True)
    return values


def build_grid(values: pd.DataFrame) -> pd.DataFrame:
    x_bin = pd.cut(values["mean_x_deg"], GRID_EDGES_DEG, labels=False, include_lowest=True, right=False)
    y_bin = pd.cut(values["mean_y_deg"], GRID_EDGES_DEG, labels=False, include_lowest=True, right=False)
    work = values.assign(x_bin=x_bin, y_bin=y_bin).dropna(subset=["x_bin", "y_bin"]).copy()
    work["x_bin"] = work["x_bin"].astype(int)
    work["y_bin"] = work["y_bin"].astype(int)
    rows = []
    metric_columns = [
        "drift_rms_radius_arcmin",
        "axis_free_anisotropy_arcmin",
        "horizontal_minus_vertical_arcmin",
        "tangential_minus_radial_arcmin",
    ]
    for (iy, ix), block in work.groupby(["y_bin", "x_bin"], sort=True):
        axis = np.radians(block["major_axis_deg"].to_numpy(dtype=float))
        weight = np.clip(block["anisotropy"].to_numpy(dtype=float), 0.0, 1.0)
        z = np.sum(weight * np.exp(2j * axis)) / max(float(np.sum(weight)), 1e-12)
        row = {
            "x_bin": int(ix),
            "y_bin": int(iy),
            "x_center_deg": float(0.5 * (GRID_EDGES_DEG[ix] + GRID_EDGES_DEG[ix + 1])),
            "y_center_deg": float(0.5 * (GRID_EDGES_DEG[iy] + GRID_EDGES_DEG[iy + 1])),
            "n_windows": int(len(block)),
            "n_sessions": int(block["session"].nunique()),
            "weighted_axis_deg": float(0.5 * np.degrees(np.angle(z))),
            "weighted_axis_resultant": float(np.abs(z)),
        }
        for metric in metric_columns:
            row[f"median_{metric}"] = float(np.nanmedian(block[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_eccentricity_summary(values: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "drift_rms_radius_arcmin",
        "axis_free_anisotropy_arcmin",
        "horizontal_minus_vertical_arcmin",
        "tangential_minus_radial_arcmin",
    ]
    rows = []
    scopes = [("pooled", "pooled", values)] + [
        ("subject", subject, values[values["subject"].eq(subject)])
        for subject in ("Allen", "Logan")
    ]
    for scope, subject, scope_values in scopes:
        for order, label in enumerate(ECC_LABELS):
            block = scope_values[scope_values["eccentricity_bin"].astype(str).eq(label)]
            if block.empty:
                continue
            row = {
                "scope": scope,
                "subject": subject,
                "eccentricity_bin_order": order,
                "eccentricity_bin": label,
                "eccentricity_center_deg": float(np.nanmedian(block["gaze_eccentricity_deg"])),
                "n_windows": int(len(block)),
                "n_trials": int(block[["session", "trial_idx"]].drop_duplicates().shape[0]),
                "n_sessions": int(block["session"].nunique()),
            }
            for metric in metrics:
                row[f"median_{metric}"] = float(np.nanmedian(block[metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def figure_f_reference() -> dict[str, float]:
    contrasts = pd.read_csv(FIGURE_F_CONTRASTS)
    row = contrasts[
        contrasts["scope"].eq("grand_equal_subject")
        & contrasts["coherence_band"].astype(str).eq("0.5–1")
    ].iloc[0]
    return {
        "estimate": float(row["parallel_minus_orthogonal_arcmin"]),
        "ci95_low": float(row["ci95_low"]),
        "ci95_high": float(row["ci95_high"]),
    }


def preliminary_effect_sizes(values: pd.DataFrame, reference: dict[str, float]) -> pd.DataFrame:
    metrics = {
        "screen_horizontal_minus_vertical": "horizontal_minus_vertical_arcmin",
        "gaze_tangential_minus_radial": "tangential_minus_radial_arcmin",
        "axis_free_major_minus_minor": "axis_free_anisotropy_arcmin",
        "drift_rms_radius": "drift_rms_radius_arcmin",
    }
    rows = []
    for subject in ("pooled", "Allen", "Logan"):
        block = values if subject == "pooled" else values[values["subject"].eq(subject)]
        central = block[block["gaze_eccentricity_deg"] < 4.0]
        peripheral = block[block["gaze_eccentricity_deg"] >= 8.0]
        for label, metric in metrics.items():
            central_value = float(np.nanmedian(central[metric]))
            peripheral_value = float(np.nanmedian(peripheral[metric]))
            rows.append(
                {
                    "subject": subject,
                    "metric": label,
                    "central_lt4deg": central_value,
                    "peripheral_ge8deg": peripheral_value,
                    "peripheral_minus_central": peripheral_value - central_value,
                    "n_central_windows": int(len(central)),
                    "n_peripheral_windows": int(len(peripheral)),
                    "figure4f_high_coherence_contour_effect_arcmin": reference["estimate"],
                    "signed_ratio_to_figure4f": (
                        peripheral_value - central_value
                    ) / reference["estimate"],
                    "status": "preliminary pooled-window descriptive; not hierarchical or movement-scale matched",
                }
            )
    return pd.DataFrame(rows)


def _grid_array(grid: pd.DataFrame, column: str) -> np.ndarray:
    n = len(GRID_EDGES_DEG) - 1
    array = np.full((n, n), np.nan)
    for row in grid.itertuples(index=False):
        if row.n_windows >= MIN_GRID_WINDOWS:
            array[int(row.y_bin), int(row.x_bin)] = float(getattr(row, column))
    return array


def _plot_map(
    ax: plt.Axes,
    grid: pd.DataFrame,
    column: str,
    title: str,
    colorbar_label: str,
    *,
    cmap: str,
    symmetric: bool = False,
) -> None:
    array = _grid_array(grid, column)
    finite = array[np.isfinite(array)]
    if symmetric:
        limit = float(np.nanquantile(np.abs(finite), 0.92)) if finite.size else 1.0
        vmin, vmax = -limit, limit
    else:
        vmin = float(np.nanquantile(finite, 0.05)) if finite.size else None
        vmax = float(np.nanquantile(finite, 0.95)) if finite.size else None
    mesh = ax.pcolormesh(GRID_EDGES_DEG, GRID_EDGES_DEG, array, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_title(title, loc="left", weight="semibold")
    ax.set_aspect("equal")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.axhline(0, color="white", lw=0.55, alpha=0.55)
    ax.axvline(0, color="white", lw=0.55, alpha=0.55)
    plt.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02, label=colorbar_label)


def plot_mechanism_maps(values: pd.DataFrame, grid: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.1, 7.0), constrained_layout=True)
    count = _grid_array(grid, "n_windows")
    count_masked = np.ma.masked_invalid(count)
    positive = count[np.isfinite(count) & (count > 0)]
    mesh = axes[0, 0].pcolormesh(
        GRID_EDGES_DEG,
        GRID_EDGES_DEG,
        count_masked,
        cmap="Greys",
        norm=LogNorm(vmin=max(MIN_GRID_WINDOWS, float(np.min(positive))), vmax=float(np.max(positive))),
        shading="flat",
    )
    axes[0, 0].set_title("A  Reviewed drift-window support", loc="left", weight="semibold")
    plt.colorbar(mesh, ax=axes[0, 0], shrink=0.78, pad=0.02, label="windows / 2° bin")

    _plot_map(
        axes[0, 1], grid, "median_drift_rms_radius_arcmin",
        "B  Total drift-cloud scale", "RMS radius (arcmin)", cmap="viridis",
    )
    _plot_map(
        axes[0, 2], grid, "median_axis_free_anisotropy_arcmin",
        "C  Axis-free anisotropy", "major − minor RMS (arcmin)", cmap="magma",
    )
    _plot_map(
        axes[1, 0], grid, "median_horizontal_minus_vertical_arcmin",
        "D  Screen-frame allocation", "horizontal − vertical RMS (arcmin)", cmap="coolwarm", symmetric=True,
    )
    _plot_map(
        axes[1, 1], grid, "median_tangential_minus_radial_arcmin",
        "E  Gaze-position frame", "tangential − radial RMS (arcmin)", cmap="PuOr", symmetric=True,
    )

    ax = axes[1, 2]
    supported = grid[grid["n_windows"] >= MIN_GRID_WINDOWS]
    ax.scatter(supported["x_center_deg"], supported["y_center_deg"], s=6, color="#D7DCE0", zorder=0)
    for row in supported.itertuples(index=False):
        theta = np.radians(row.weighted_axis_deg)
        half_length = 0.25 + 0.75 * row.weighted_axis_resultant
        dx = half_length * np.cos(theta)
        dy = half_length * np.sin(theta)
        ax.plot(
            [row.x_center_deg - dx, row.x_center_deg + dx],
            [row.y_center_deg - dy, row.y_center_deg + dy],
            color=INK,
            lw=0.55 + 1.25 * row.weighted_axis_resultant,
            alpha=0.45 + 0.5 * row.weighted_axis_resultant,
            solid_capstyle="round",
        )
    ax.set_title("F  Absolute drift-cloud axes", loc="left", weight="semibold")
    ax.set_aspect("equal")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.axhline(0, color=GRID, lw=0.65)
    ax.axvline(0, color=GRID, lw=0.65)
    ax.text(0.03, 0.03, "line angle: anisotropy-weighted mean axis\nline length: axial consistency", transform=ax.transAxes, fontsize=6.2, color="#6B6F75")

    for ax in axes.flat:
        ax.set_xlabel("mean horizontal gaze (deg)")
        ax.set_ylabel("mean vertical gaze (deg)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Supplemental checkpoint 1: where gaze-position effects could enter", fontsize=13, weight="bold")
    return fig


def plot_eccentricity_curves(summary: pd.DataFrame) -> plt.Figure:
    specs = [
        ("median_drift_rms_radius_arcmin", "A  Drift-cloud scale", "RMS radius (arcmin)"),
        ("median_axis_free_anisotropy_arcmin", "B  Axis-free anisotropy", "major − minor RMS (arcmin)"),
        ("median_horizontal_minus_vertical_arcmin", "C  Screen frame", "horizontal − vertical RMS (arcmin)"),
        ("median_tangential_minus_radial_arcmin", "D  Gaze-position frame", "tangential − radial RMS (arcmin)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True, constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs, strict=True):
        pooled = summary[summary["scope"].eq("pooled")].sort_values("eccentricity_bin_order")
        for subject in ("Allen", "Logan"):
            block = summary[summary["subject"].eq(subject)].sort_values("eccentricity_bin_order")
            ax.plot(
                block["eccentricity_center_deg"],
                block[metric],
                marker="o",
                ms=3.2,
                lw=1.15,
                color=SUBJECT_COLORS[subject],
                alpha=0.78,
                label=subject,
            )
        ax.plot(
            pooled["eccentricity_center_deg"],
            pooled[metric],
            marker="o",
            ms=4.0,
            lw=1.7,
            color=INK,
            label="pooled windows",
        )
        if "minus" in metric:
            ax.axhline(0.0, color="#7D858C", lw=0.75, ls=":")
        ax.set_title(title, loc="left", weight="semibold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("gaze eccentricity (deg; bin median)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Descriptive gaze-eccentricity curves (no hierarchical CI yet)", fontsize=12.5, weight="bold")
    return fig


def plot_effect_size_comparison(
    preliminary: pd.DataFrame, reference: dict[str, float]
) -> plt.Figure:
    """Compare raw gaze endpoint changes with the hierarchical Figure 4F contrast."""
    metric_order = [
        "screen_horizontal_minus_vertical",
        "gaze_tangential_minus_radial",
        "axis_free_major_minus_minor",
        "drift_rms_radius",
    ]
    labels = {
        "screen_horizontal_minus_vertical": "Screen H−V (raw)",
        "gaze_tangential_minus_radial": "Gaze-frame T−R (raw)",
        "axis_free_major_minus_minor": "Axis-free major−minor (raw)",
        "drift_rms_radius": "Total drift RMS radius (raw)",
    }
    y = np.arange(len(metric_order), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.45), constrained_layout=True)
    ax.axvline(0.0, color="#7D858C", lw=0.8, ls=":")

    offsets = {"Allen": 0.13, "pooled": 0.0, "Logan": -0.13}
    colors = {**SUBJECT_COLORS, "pooled": INK}
    for subject in ("Allen", "pooled", "Logan"):
        block = preliminary[preliminary["subject"].eq(subject)].set_index("metric")
        x = [float(block.loc[metric, "peripheral_minus_central"]) for metric in metric_order]
        ax.scatter(
            x,
            y + offsets[subject],
            s=30 if subject == "pooled" else 22,
            color=colors[subject],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label="pooled windows" if subject == "pooled" else subject,
        )

    f_y = len(metric_order) + 0.65
    ax.errorbar(
        reference["estimate"],
        f_y,
        xerr=np.asarray(
            [[reference["estimate"] - reference["ci95_low"]],
             [reference["ci95_high"] - reference["estimate"]]]
        ),
        fmt="D",
        ms=5.2,
        color="#6B6F75",
        ecolor="#6B6F75",
        capsize=3,
        lw=1.25,
        zorder=4,
    )
    ax.axhspan(f_y - 0.34, f_y + 0.34, color="#EEF0F2", zorder=0)
    ax.set_yticks(
        np.concatenate([y, [f_y]]),
        [labels[metric] for metric in metric_order]
        + ["Figure 4F contour parallel−orthogonal\n(hierarchical 95% CI)"],
    )
    ax.invert_yaxis()
    ax.set_xlabel("effect size (arcmin RMS)\n"
                  "gaze rows: peripheral (≥8°) − central (<4°)")
    ax.set_title(
        "Preliminary size comparison: gaze-position changes versus Figure 4F",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="x", color=GRID, lw=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 260}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def write_report(values: pd.DataFrame, preliminary: pd.DataFrame, reference: dict[str, float]) -> None:
    pooled = preliminary[preliminary["subject"].eq("pooled")].set_index("metric")
    lines = [
        "# Supplemental gaze-position anisotropy: checkpoint 1",
        "",
        "## Contract",
        "",
        "These are descriptive maps of the same 11,749 event-free drift windows used by Figure 4F.",
        "Detected high-speed events were removed before the 128-sample windows were extracted, so",
        "`events_in_trial` must not be used to label a window as microsaccade-containing.",
        "",
        "The views keep three questions distinct:",
        "",
        "1. screen-horizontal versus vertical spread;",
        "2. tangential versus radial spread relative to each window's mean gaze position;",
        "3. axis-free major-minus-minor spread and total drift scale.",
        "All displayed RMS components, including total scale, use the same sample-covariance",
        "convention as the updated Figure 4F directional profiles.",
        "",
        "## Preliminary pooled-window endpoint differences",
        "",
        "Central means gaze eccentricity <4 deg; peripheral means >=8 deg. These values are not",
        "hierarchical, not movement-scale matched, and not the final effect-size comparison.",
        "",
        "| metric | central | peripheral | peripheral - central |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "screen_horizontal_minus_vertical",
        "gaze_tangential_minus_radial",
        "axis_free_major_minus_minor",
        "drift_rms_radius",
    ]:
        row = pooled.loc[metric]
        lines.append(
            f"| {metric} | {row.central_lt4deg:+.4f} | {row.peripheral_ge8deg:+.4f} | "
            f"{row.peripheral_minus_central:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Figure 4F high-coherence contour-relative reference: {reference['estimate']:+.4f} "
            f"arcmin, 95% CI [{reference['ci95_low']:+.4f}, {reference['ci95_high']:+.4f}].",
            "The raw pooled peripheral-minus-central changes are "
            f"{pooled.loc['screen_horizontal_minus_vertical'].signed_ratio_to_figure4f:+.2f}× (screen H−V), "
            f"{pooled.loc['gaze_tangential_minus_radial'].signed_ratio_to_figure4f:+.2f}× (gaze T−R), "
            f"{pooled.loc['axis_free_major_minus_minor'].signed_ratio_to_figure4f:+.2f}× (axis-free anisotropy), and "
            f"{pooled.loc['drift_rms_radius'].signed_ratio_to_figure4f:+.2f}× (total drift scale) "
            "that reference. These ratios compare units, not equivalent biological contrasts.",
            "",
            "## Interpretation guardrails",
            "",
            "- A change with gaze eccentricity can reflect screen position, gaze polar angle, subject",
            "  composition, or the accompanying change in drift-cloud scale.",
            "- Tangential retinal displacement from torsion is not present in these two-dimensional",
            "  eye-position measurements; a null tangential-radial result does not test for torsion.",
            "- The human horizontal prior and rhesus vertical prior are not imported into marmosets;",
            "  the absolute-axis maps measure the prior empirically.",
            "- The next checkpoint should construct hierarchical central/peripheral contrasts while",
            "  matching or stratifying movement scale and gaze polar angle, then compare those draws",
            "  directly with the Figure 4F contrast.",
            "",
            f"Windows: {len(values)}; sessions: {values['session'].nunique()}; trials: "
            f"{values[['session', 'trial_idx']].drop_duplicates().shape[0]}.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_and_derive()
    grid = build_grid(values)
    summary = build_eccentricity_summary(values)
    reference = figure_f_reference()
    preliminary = preliminary_effect_sizes(values, reference)

    keep_columns = [
        "subject", "session", "trial_idx", "phase", "mean_x_deg", "mean_y_deg",
        "gaze_eccentricity_deg", "gaze_polar_angle_deg", "eccentricity_bin",
        "drift_rms_radius_arcmin", "source_population_rms_radius_arcmin",
        "axis_free_anisotropy_arcmin",
        "horizontal_rms_arcmin", "vertical_rms_arcmin", "horizontal_minus_vertical_arcmin",
        "radial_rms_arcmin", "tangential_rms_arcmin", "tangential_minus_radial_arcmin",
        "major_rms_arcmin", "minor_rms_arcmin", "major_axis_deg", "anisotropy",
    ]
    values[keep_columns].to_csv(
        OUT_DIR / "gaze_position_window_values.csv.gz", index=False, compression="gzip"
    )
    grid.to_csv(OUT_DIR / "gaze_position_grid_values.csv", index=False)
    summary.to_csv(OUT_DIR / "gaze_eccentricity_descriptive_values.csv", index=False)
    preliminary.to_csv(OUT_DIR / "preliminary_effect_size_reference.csv", index=False)

    outputs = {
        "mechanism_maps": save_figure(plot_mechanism_maps(values, grid), "gaze_position_mechanism_maps"),
        "eccentricity_curves": save_figure(
            plot_eccentricity_curves(summary), "gaze_eccentricity_descriptive_curves"
        ),
        "effect_size_comparison": save_figure(
            plot_effect_size_comparison(preliminary, reference),
            "gaze_position_effect_size_comparison",
        ),
    }
    write_report(values, preliminary, reference)

    metadata = {
        "stage": "map-first checkpoint 1; descriptive input/mechanism views",
        "source_windows": str(SOURCE_WINDOWS.relative_to(ROOT)),
        "figure4f_reference": str(FIGURE_F_CONTRASTS.relative_to(ROOT)),
        "n_windows": int(len(values)),
        "n_trials": int(values[["session", "trial_idx"]].drop_duplicates().shape[0]),
        "n_sessions": int(values["session"].nunique()),
        "subjects": values.groupby("subject")["session"].nunique().to_dict(),
        "window_contract": (
            "128-sample contiguous clean windows after detected high-speed event samples were removed; "
            "mid- and late-fixation phases only"
        ),
        "grid_edges_deg": GRID_EDGES_DEG.tolist(),
        "minimum_windows_per_grid_cell": MIN_GRID_WINDOWS,
        "eccentricity_edges_deg": ECC_EDGES_DEG.tolist(),
        "preliminary_endpoint_definition": {"central": "eccentricity <4 deg", "peripheral": "eccentricity >=8 deg"},
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for group, paths in outputs.items():
        print(group, ROOT / paths["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()

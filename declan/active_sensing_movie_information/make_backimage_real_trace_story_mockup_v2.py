"""Make a simplified five-panel story mockup for real-trace SSI results.

This version emphasizes a shared visual language: all data panels plot percent
change from the counterfactually stabilized movie on a compressed, broken-log
movement axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
SUMMARY_DIR = (
    MATRIX_DIR
    / "phase1_phase2_conditioning_v1"
    / "schematic_pathlength_summary_v1"
    / "unit_first_and_population_v1"
)
OUT_DIR = SUMMARY_DIR / "figures"

SF_COLORS = {
    "low_sf": "#0072B2",
    "middle_sf": "#009E73",
    "high_sf": "#D55E00",
}
SF_LABELS = {
    "low_sf": "Low SF",
    "middle_sf": "Middle SF",
    "high_sf": "High SF",
}
RELATION_STYLES = {
    "strong_contours_no_osi": ("all high-SF units", "#5F5F5F", "-", 2.3),
    "contour_matched": ("aligned", SF_COLORS["high_sf"], "-", 2.0),
    "contour_intermediate": ("intermediate", SF_COLORS["high_sf"], "-.", 2.0),
    "contour_orthogonal": ("orthogonal", SF_COLORS["high_sf"], "--", 2.0),
}
COMPONENT_STYLES = {
    "across_path_arcmin": ("across contour", "-", "o"),
    "along_path_arcmin": ("along contour", "--", "s"),
}


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    total = pd.read_csv(SUMMARY_DIR / "spike_weighted_population_summary.csv")
    comp = pd.read_csv(SUMMARY_DIR / "spike_weighted_population_component_summary.csv")
    return _add_percent_change(total), _add_percent_change(comp)


def _add_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["relation", "sf_group"]
    if "component_metric" in df.columns:
        group_cols.append("component_metric")
    for _, group in df.groupby(group_cols, sort=False):
        baseline = group.loc[
            group["context"].eq("stabilized"), "population_ssi_bits_per_spike"
        ].iloc[0]
        copy = group.copy()
        copy["population_ssi_percent_vs_stabilized"] = (
            100.0 * copy["population_ssi_delta_vs_stabilized"] / baseline
        )
        out.append(copy)
    return pd.concat(out, ignore_index=True)


def _x_broken_log(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Map zero to a left anchor and positive path lengths to a log axis."""
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    min_pos = 45.0
    max_pos = 185.0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(
        max_pos / min_pos
    )
    return mapped


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def _format_movement_axis(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_xlim(-0.12, 6.25)
    ax.set_xticks(_x_broken_log(ticks))
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.text(
        0.52,
        -0.065,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _plot_series(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x_col: str,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    include_microsaccade: bool = True,
) -> None:
    zero = rows[rows["context"].eq("stabilized")]
    drift = rows[rows["context"].eq("drift_only")].sort_values(x_col)
    ms = rows[rows["context"].eq("microsaccade")].sort_values(x_col)

    if not zero.empty and not drift.empty:
        drift_with_zero = pd.concat([zero.iloc[:1], drift], ignore_index=True)
        ax.plot(
            _x_broken_log(drift_with_zero[x_col]),
            drift_with_zero["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
    elif not drift.empty:
        ax.plot(
            _x_broken_log(drift[x_col]),
            drift["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    plot_groups = [(drift, False)]
    if include_microsaccade:
        plot_groups.append((ms, True))
    for plot_rows, filled in plot_groups:
        if plot_rows.empty:
            continue
        ax.plot(
            _x_broken_log(plot_rows[x_col]),
            plot_rows["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.scatter(
            _x_broken_log(plot_rows[x_col]),
            plot_rows["population_ssi_percent_vs_stabilized"],
            marker=marker,
            s=30,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.5,
            alpha=alpha,
            zorder=3,
        )

    if not zero.empty:
        ax.scatter(
            [0],
            [0],
            marker=marker,
            s=32,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            alpha=alpha,
            zorder=3,
        )


def _panel_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(plt.Rectangle((0.13, 0.18), 0.31, 0.62, color="0.15"))
    ax.add_patch(plt.Rectangle((0.44, 0.18), 0.31, 0.62, color="0.86"))
    ax.plot([0.44, 0.44], [0.18, 0.80], color="white", lw=2.0)
    ax.text(0.44, 0.83, "strong contour", ha="center", va="bottom", fontsize=10)

    ax.annotate(
        "",
        xy=(0.70, 0.49),
        xytext=(0.44, 0.49),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=SF_COLORS["high_sf"]),
    )
    ax.annotate(
        "",
        xy=(0.44, 0.73),
        xytext=(0.44, 0.49),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=SF_COLORS["high_sf"]),
    )
    ax.text(0.72, 0.49, "across", color=SF_COLORS["high_sf"], va="center", fontsize=10)
    ax.text(0.46, 0.74, "along", color=SF_COLORS["high_sf"], va="bottom", fontsize=10)

    t = np.linspace(0, 1, 40)
    ax.plot(
        0.50 + 0.035 * np.sin(9 * t),
        0.28 + 0.34 * t,
        color="#0072B2",
        lw=2.0,
    )
    ax.scatter([0.53], [0.28], s=38, facecolors="white", edgecolors="#0072B2", lw=1.6)
    ax.scatter([0.47], [0.62], s=38, facecolors="#0072B2", edgecolors="#0072B2", lw=1.6)

    ax.text(
        0.08,
        0.06,
        "Curves report percent change from the no-motion movie.\n"
        "Open points: drift-only snippets. Filled points: snippets with microsaccades where shown.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
    )


def _panel_strong_total(ax: plt.Axes, total: pd.DataFrame) -> None:
    subset = total[total["relation"].eq("strong_contours_no_osi")]
    for sf_group in ("low_sf", "middle_sf", "high_sf"):
        rows = subset[subset["sf_group"].eq(sf_group)]
        _plot_series(
            ax,
            rows,
            x_col="path_median_arcmin",
            color=SF_COLORS[sf_group],
            label=SF_LABELS[sf_group],
            linewidth=2.2,
        )
    ax.axhline(0, color="0.35", lw=1.0, ls=":")
    ax.set_title("Total path: strong contour images, all units", fontsize=11)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("trajectory path length (arcmin; log scale after break)")
    ax.set_ylim(-5, 62)
    _format_movement_axis(ax, [0, 90, 105, 120, 150, 175])
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=3)
    _style_axis(ax)


def _panel_high_sf_total_by_relation(ax: plt.Axes, total: pd.DataFrame) -> None:
    for relation, (label, color, linestyle, linewidth) in RELATION_STYLES.items():
        rows = total[
            total["relation"].eq(relation) & total["sf_group"].eq("high_sf")
        ]
        _plot_series(
            ax,
            rows,
            x_col="path_median_arcmin",
            color=color,
            label=label,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.86 if relation == "strong_contours_no_osi" else 1.0,
            include_microsaccade=False,
        )
    ax.axhline(0, color="0.35", lw=1.0, ls=":")
    ax.set_title("High SF: total path hides geometry-specific effects", fontsize=11)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("trajectory path length (arcmin; log scale after break)")
    ax.set_ylim(-28, 20)
    _format_movement_axis(ax, [0, 90, 105, 120, 150, 175])
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=2)
    _style_axis(ax)


def _panel_component(
    ax: plt.Axes,
    comp: pd.DataFrame,
    *,
    relation: str,
    title: str,
) -> None:
    for metric, (label, linestyle, marker) in COMPONENT_STYLES.items():
        rows = comp[
            comp["relation"].eq(relation)
            & comp["sf_group"].eq("high_sf")
            & comp["component_metric"].eq(metric)
        ]
        _plot_series(
            ax,
            rows,
            x_col="component_median_arcmin",
            color=SF_COLORS["high_sf"],
            label=label,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.1,
            include_microsaccade=False,
        )
    ax.axhline(0, color="0.35", lw=1.0, ls=":")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("component path length (arcmin; log scale after break)")
    ax.set_ylim(-24, 14)
    _format_movement_axis(ax, [0, 50, 65, 80, 105, 120])
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style_axis(ax)


def main() -> None:
    total, comp = _load_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 9.5))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=(0.9, 1.0, 1.0),
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.08,
        hspace=0.55,
        wspace=0.28,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    _panel_schematic(axes[0])
    _panel_strong_total(axes[1], total)
    _panel_high_sf_total_by_relation(axes[2], total)
    _panel_component(
        axes[3],
        comp,
        relation="contour_matched",
        title="High-SF aligned units: across and along paths diverge",
    )
    _panel_component(
        axes[4],
        comp,
        relation="contour_orthogonal",
        title="High-SF orthogonal units: the signs reverse",
    )

    for label, ax in zip("ABCDE", axes):
        ax.text(
            -0.07,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "Real fixational motion changes SSI according to contour geometry",
        fontsize=15,
        y=0.97,
    )
    png = OUT_DIR / "five_panel_real_trace_geometry_strategy_mockup_v2.png"
    pdf = OUT_DIR / "five_panel_real_trace_geometry_strategy_mockup_v2.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

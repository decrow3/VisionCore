"""Make a compact five-panel story mockup for the real-trace SSI analysis."""

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
REL_LABELS = {
    "contour_matched": "aligned",
    "contour_intermediate": "intermediate",
    "contour_orthogonal": "orthogonal",
}


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace = pd.read_csv(MATRIX_DIR / "trace_feature_table.csv")
    total = pd.read_csv(SUMMARY_DIR / "spike_weighted_population_summary.csv")
    comp = pd.read_csv(SUMMARY_DIR / "spike_weighted_population_component_summary.csv")
    return trace, total, comp


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def _add_zero_break(ax: plt.Axes) -> None:
    ax.text(
        0.085,
        -0.055,
        "//",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _plot_population_curves(
    ax: plt.Axes,
    total: pd.DataFrame,
    relation: str,
    title: str,
) -> None:
    subset = total[total["relation"].eq(relation)]
    for sf_group in ("low_sf", "middle_sf", "high_sf"):
        sf = subset[subset["sf_group"].eq(sf_group)].sort_values("path_bin_order")
        if sf.empty:
            continue

        color = SF_COLORS[sf_group]
        last_xy = None
        zero = sf[sf["context"].eq("stabilized")]
        drift = sf[sf["context"].eq("drift_only")]
        ms = sf[sf["context"].eq("microsaccade")]

        for rows, filled in ((drift, False), (ms, True)):
            if rows.empty:
                continue
            ax.plot(
                rows["path_median_arcmin"],
                rows["population_ssi_bits_per_spike"],
                color=color,
                lw=2.0,
                alpha=0.95,
            )
            ax.scatter(
                rows["path_median_arcmin"],
                rows["population_ssi_bits_per_spike"],
                s=30,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=1.6,
                zorder=3,
            )
            if last_xy is None or rows["path_median_arcmin"].iloc[-1] > last_xy[0]:
                last_xy = (
                    rows["path_median_arcmin"].iloc[-1],
                    rows["population_ssi_bits_per_spike"].iloc[-1],
                )
        if not zero.empty:
            ax.scatter(
                [0],
                [zero["population_ssi_bits_per_spike"].iloc[0]],
                s=34,
                facecolors="white",
                edgecolors=color,
                linewidths=1.6,
                zorder=3,
            )
        if last_xy is not None:
            y_offset = {"low_sf": 0.0000, "middle_sf": 0.0010, "high_sf": -0.0010}[sf_group]
            ax.text(
                0.965,
                last_xy[1] + y_offset,
                SF_LABELS[sf_group],
                transform=ax.get_yaxis_transform(),
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="right",
                va="center",
            )

    ax.set_title(title, fontsize=11, pad=5)
    ax.set_xlabel("trajectory path length (arcmin)")
    ax.set_ylabel("population SSI\n(bits/spike)")
    ax.set_xticks([0, 90, 120, 150, 175])
    ax.set_xlim(-10, 188)
    _add_zero_break(ax)
    _style_axis(ax)


def _first_bin_percent(
    df: pd.DataFrame,
    relation: str,
    sf_group: str,
    *,
    component_metric: str | None = None,
) -> float:
    subset = df[df["relation"].eq(relation) & df["sf_group"].eq(sf_group)]
    if component_metric is not None:
        subset = subset[subset["component_metric"].eq(component_metric)]
        order_col = "component_bin_order"
    else:
        order_col = "path_bin_order"
    baseline = subset[subset["context"].eq("stabilized")][
        "population_ssi_bits_per_spike"
    ].iloc[0]
    first = subset[subset["context"].eq("drift_only") & subset[order_col].eq(1)][
        "population_ssi_bits_per_spike"
    ].iloc[0]
    return 100.0 * (first - baseline) / baseline


def _plot_component_percent_by_sf(ax: plt.Axes, comp: pd.DataFrame) -> None:
    x = np.arange(3)
    width = 0.36
    for i, metric in enumerate(("across_path_arcmin", "along_path_arcmin")):
        vals = [
            _first_bin_percent(comp, "strong_contours_no_osi", sf, component_metric=metric)
            for sf in ("low_sf", "middle_sf", "high_sf")
        ]
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=["white", "white", "white"] if i == 0 else "0.78",
            edgecolor=[SF_COLORS[sf] for sf in ("low_sf", "middle_sf", "high_sf")],
            linewidth=2.0,
            hatch=None if i == 0 else "///",
            label="across" if i == 0 else "along",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.9,
                f"{val:+.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0, color="0.35", lw=1.0)
    ax.set_title("Strong contours: component size matters", fontsize=11, pad=5)
    ax.set_xticks(x, [SF_LABELS[sf] for sf in ("low_sf", "middle_sf", "high_sf")])
    ax.set_ylabel("first drift bin vs no motion (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(-4, 32)
    _style_axis(ax)


def _plot_high_sf_geometry(ax: plt.Axes, comp: pd.DataFrame) -> None:
    relations = ("contour_matched", "contour_intermediate", "contour_orthogonal")
    x = np.arange(len(relations))
    width = 0.34
    vals_across = [
        _first_bin_percent(comp, rel, "high_sf", component_metric="across_path_arcmin")
        for rel in relations
    ]
    vals_along = [
        _first_bin_percent(comp, rel, "high_sf", component_metric="along_path_arcmin")
        for rel in relations
    ]
    ax.bar(
        x - width / 2,
        vals_across,
        width=width,
        facecolor="white",
        edgecolor=SF_COLORS["high_sf"],
        linewidth=2.0,
        label="across contour",
    )
    ax.bar(
        x + width / 2,
        vals_along,
        width=width,
        facecolor="0.78",
        edgecolor=SF_COLORS["high_sf"],
        linewidth=2.0,
        hatch="///",
        label="along contour",
    )
    for xpos, val in zip(x - width / 2, vals_across):
        ax.text(xpos, val + (1.0 if val >= 0 else -1.8), f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    for xpos, val in zip(x + width / 2, vals_along):
        ax.text(xpos, val + (1.0 if val >= 0 else -1.8), f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.axhline(0, color="0.25", lw=1.0)
    ax.set_title("High-SF response depends on unit-contour geometry", fontsize=11, pad=5)
    ax.set_xticks(x, [REL_LABELS[r] for r in relations])
    ax.set_ylabel("first component bin vs no motion (%)")
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="upper center")
    ax.set_ylim(-16, 14)
    _style_axis(ax)


def _plot_trace_reference(ax: plt.Axes, trace: pd.DataFrame) -> None:
    path = trace["rendered_path_length_arcmin"].to_numpy()
    has_ms = trace["rendered_n_microsaccade_events"].to_numpy() >= 1
    bins = np.linspace(np.nanmin(path), np.nanpercentile(path, 99.5), 34)
    ax.hist(
        path[~has_ms],
        bins=bins,
        density=True,
        color="0.78",
        edgecolor="white",
        alpha=0.9,
        label="drift only",
    )
    ax.hist(
        path[has_ms],
        bins=bins,
        density=True,
        histtype="step",
        color="0.2",
        linewidth=2.0,
        label="with microsaccade",
    )
    q = np.nanpercentile(path[~has_ms], [25, 50, 75])
    ax.axvspan(q[0], q[2], color="0.45", alpha=0.12)
    ax.axvline(q[1], color="0.25", ls="--", lw=1.0)
    ax.set_title("Real fixation snippets define movement scale", fontsize=11, pad=5)
    ax.set_xlabel("trajectory path length (arcmin)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)


def main() -> None:
    trace, total, comp = _load_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.9),
        left=0.07,
        right=0.98,
        top=0.91,
        bottom=0.08,
        hspace=0.52,
        wspace=0.28,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]

    _plot_trace_reference(axes[0], trace)
    _plot_population_curves(
        axes[1],
        total,
        "all_images_no_osi",
        "All images: motion increases population SSI",
    )
    _plot_population_curves(
        axes[2],
        total,
        "strong_contours_no_osi",
        "Strong contours: high-SF boost is not simply monotonic",
    )
    _plot_component_percent_by_sf(axes[3], comp)
    _plot_high_sf_geometry(axes[4], comp)

    for label, ax in zip("ABCDE", axes):
        ax.text(
            -0.12,
            1.09,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "Mockup: real fixational motion, contour geometry, and population SSI",
        fontsize=15,
        y=0.975,
    )
    fig.text(
        0.5,
        0.035,
        "Open markers: drift-only snippets. Filled markers: snippets containing >=1 detected microsaccade. "
        "Bars show the first drift-only component bin relative to the counterfactually stabilized movie.",
        ha="center",
        fontsize=9,
        color="0.25",
    )

    png = OUT_DIR / "five_panel_real_trace_geometry_strategy_mockup.png"
    pdf = OUT_DIR / "five_panel_real_trace_geometry_strategy_mockup.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

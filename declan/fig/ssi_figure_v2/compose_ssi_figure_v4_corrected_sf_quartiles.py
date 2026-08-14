#!/usr/bin/env python3
"""Compose the two-row Figure 4 with corrected-cache SF quartiles.

Panels A/C retain the established explanatory schematics. Population panels
B/D/E are recomputed from a frozen set of complete balanced corrected-cache
rounds. The recorded-response fit gate is applied before preferred-SF
quartiles are assigned.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from pypdf import PdfWriter

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_sf_halves as base
from declan.fig4_active_sensing.make_rr100_corrected_quartile_population_figure import (
    COLORS,
    GROUPS,
    LABELS,
    assign_validated_quartiles,
)


ROOT = Path(__file__).resolve().parents[3]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_011_n012_quartile_snapshot_v1"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
UNIT_TABLE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)
TRACE_XY = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "input_cache/corrected_trace_segments.npz"
)
MICROSACCADE_LABELS = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected_microsaccade_audit_v1/"
    "corrected_scored_microsaccade_labels.csv"
)
SCHEMATIC_PANELS = ROOT / (
    "outputs/fig/ssi_figure_v2/sf_outer_thirds_recorded_validated_r0p5_v1/panels"
)
OUT = ROOT / "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_rounds000_011_v2"
PANELS = OUT / "panels"
STEM = "ssi_figure_v4_corrected_cache_sf_quartiles_no_bottom_row_rounds000_011_v2"

COHERENCE_MIN = 0.20
OSI_MIN = 0.05
ORIENTATION_MATCH_MAX_DEG = 22.5
N_BOOTSTRAP = 4000
SEED = 20260813
INK = "#111111"
FIGURE_SCOPE_LABEL = "corrected cache · 12 complete balanced rounds (interim)"
COMPONENT_GROUP = "sf_q4"
GROUPING_DESCRIPTION = "61 recorded-validated units split 16/15/15/15 after the r >= 0.5 gate"
PATH_PANEL_TITLE = "Path length separates\npreferred-SF quartiles"
MATCH_PANEL_TITLE = "Contour alignment reshapes\nSF dependence"
COMPONENT_PANEL_TITLE = "Across-contour spread limits\nhighest-SF-quartile benefit"
PATH_MIN_POS = 55.0
PATH_MAX_POS = 160.0
PATH_TICKS = (0.0, 60.0, 75.0, 90.0, 120.0, 150.0)
COMPONENT_MIN_POS = 1.0
COMPONENT_MAX_POS = 9.0
COMPONENT_TICKS = (0.0, 1.0, 1.5, 2.0, 4.0, 8.0)


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def axis_delta_deg(first: np.ndarray, second: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(first, dtype=float) - np.asarray(second, dtype=float) + 90.0) % 180.0 - 90.0)


def path_broken_log(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Map zero to its own anchor and positive paths onto a log segment."""
    raw = np.asarray(values, dtype=float)
    mapped = np.zeros_like(raw, dtype=float)
    positive = raw > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(raw[positive] / PATH_MIN_POS) / np.log(
        PATH_MAX_POS / PATH_MIN_POS
    )
    return mapped


def format_broken_path_axis(ax: plt.Axes) -> None:
    # The mapping reaches 6.1 at PATH_MAX_POS. Keep the full positive segment
    # visible so terminal high-path markers and their error bars are not
    # clipped at the right boundary.
    ax.set_xlim(-0.12, 6.25)
    ax.set_xticks(path_broken_log(list(PATH_TICKS)))
    ax.set_xticklabels([str(int(tick)) for tick in PATH_TICKS])
    ax.text(
        0.52, -0.075, "//", transform=ax.get_xaxis_transform(),
        ha="center", va="center", fontsize=15, fontweight="bold",
        rotation=-20, clip_on=False,
    )


def component_broken_log(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Map stabilized zero separately from positive component excursions."""
    raw = np.asarray(values, dtype=float)
    mapped = np.zeros_like(raw, dtype=float)
    positive = raw > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(raw[positive] / COMPONENT_MIN_POS) / np.log(
        COMPONENT_MAX_POS / COMPONENT_MIN_POS
    )
    return mapped


def format_broken_component_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.12, 6.25)
    ax.set_xticks(component_broken_log(list(COMPONENT_TICKS)))
    ax.set_xticklabels([f"{tick:g}" for tick in COMPONENT_TICKS])
    ax.text(
        0.52, -0.075, "//", transform=ax.get_xaxis_transform(),
        ha="center", va="center", fontsize=15, fontweight="bold",
        rotation=-20, clip_on=False,
    )


def add_segmented_zero_anchor(ax: plt.Axes, colors: list[str]) -> None:
    """Show coincident stabilized anchors without giving them x offsets."""
    diameter = 6.8
    pad = 0.8
    area = DrawingArea(diameter + 2 * pad, diameter + 2 * pad, 0, 0, clip=False)
    center = (diameter / 2 + pad, diameter / 2 + pad)
    step = 360.0 / len(colors)
    for index, color in enumerate(colors):
        area.add_artist(
            patches.Wedge(
                center, diameter / 2, index * step, (index + 1) * step,
                facecolor=color, edgecolor="none",
            )
        )
    area.add_artist(
        patches.Circle(center, diameter / 2, facecolor="none", edgecolor="0.25", linewidth=0.65)
    )
    ax.add_artist(
        AnnotationBbox(
            area, (0.0, 0.0), xycoords="data", frameon=False,
            box_alignment=(0.5, 0.5), pad=0.0, annotation_clip=False,
        )
    )


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def condition_sufficient_statistics(
    moving_info: np.ndarray,
    moving_spikes: np.ndarray,
    baseline_info: np.ndarray,
    baseline_spikes: np.ndarray,
    image_ids: np.ndarray,
    unit_mask_by_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(image_ids)
    out = [np.full(n, np.nan, dtype=float) for _ in range(4)]
    valid = np.zeros(n, dtype=bool)
    for image in np.unique(image_ids):
        rows = np.flatnonzero(image_ids == image)
        units = np.flatnonzero(unit_mask_by_image[int(image)])
        if units.size == 0:
            continue
        out[0][rows] = np.asarray(moving_info[np.ix_(rows, units)], dtype=float).sum(axis=1)
        out[1][rows] = np.asarray(moving_spikes[np.ix_(rows, units)], dtype=float).sum(axis=1)
        out[2][rows] = float(baseline_info[int(image), units].sum())
        out[3][rows] = float(baseline_spikes[int(image), units].sum())
        valid[rows] = True
    return out[0], out[1], out[2], out[3], valid


def ratio_delta_percent(info: np.ndarray, spikes: np.ndarray, base_info: np.ndarray, base_spikes: np.ndarray) -> float:
    moving = float(np.sum(info) / np.maximum(np.sum(spikes), 1e-12))
    baseline = float(np.sum(base_info) / np.maximum(np.sum(base_spikes), 1e-12))
    return 100.0 * (moving / baseline - 1.0)


def summarize_binned(
    *,
    x: np.ndarray,
    context: np.ndarray,
    image_ids: np.ndarray,
    info: np.ndarray,
    spikes: np.ndarray,
    base_info: np.ndarray,
    base_spikes: np.ndarray,
    valid: np.ndarray,
    group: str,
    relation: str,
    bins_by_context: dict[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context_name, n_bins in bins_by_context.items():
        use_context = valid & (context == context_name) & np.isfinite(x)
        rank = pd.Series(x[use_context]).rank(method="first")
        labels = pd.qcut(rank, n_bins, labels=False).to_numpy(int)
        selected_rows = np.flatnonzero(use_context)
        for path_bin in range(n_bins):
            use = selected_rows[labels == path_bin]
            images = np.unique(image_ids[use])
            totals = np.zeros((len(images), 4), dtype=float)
            for ordinal, image in enumerate(images):
                image_rows = use[image_ids[use] == image]
                totals[ordinal] = [
                    info[image_rows].sum(),
                    spikes[image_rows].sum(),
                    base_info[image_rows].sum(),
                    base_spikes[image_rows].sum(),
                ]
            estimate = ratio_delta_percent(*totals.sum(axis=0))
            sample = rng.integers(0, len(images), size=(N_BOOTSTRAP, len(images)))
            sampled = totals[sample].sum(axis=1)
            moving = sampled[:, 0] / np.maximum(sampled[:, 1], 1e-12)
            baseline = sampled[:, 2] / np.maximum(sampled[:, 3], 1e-12)
            boot = 100.0 * (moving / baseline - 1.0)
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "relation": relation,
                    "sf_quartile": group,
                    "context": context_name,
                    "bin": path_bin,
                    "x_median": float(np.median(x[use])),
                    "delta_percent": estimate,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "n_conditions": int(len(use)),
                    "n_images": int(len(images)),
                }
            )
    return pd.DataFrame(rows)


def build_inputs() -> dict[str, Any]:
    manifest = json.loads((ASSEMBLED / "manifest.json").read_text(encoding="utf-8"))
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    expected = int(manifest["n_complete_rounds"]) * int(manifest.get("conditions_per_round", 1000))
    if len(condition) != expected:
        raise ValueError(f"Expected {expected} conditions from the assembly manifest, found {len(condition)}")
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv").sort_values("trace_index")
    microsaccades = pd.read_csv(MICROSACCADE_LABELS)
    condition = condition.merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]],
        on="trace_index",
        validate="many_to_one",
    )
    condition = condition.merge(
        microsaccades[["trace_index", "scored_n_microsaccade_events"]],
        on="trace_index",
        validate="many_to_one",
    )
    condition["context"] = np.where(
        condition.scored_n_microsaccade_events.gt(0), "microsaccade", "drift_only"
    )
    quartiles = assign_validated_quartiles(pd.read_csv(ASSIGNMENTS))
    units = pd.read_csv(UNIT_TABLE)[
        ["unit_index", "prior_preferred_orientation_deg", "prior_orientation_selectivity_index"]
    ].merge(
        quartiles[["rr100_index", "preferred_sf_cpd", "sf_quartile", "sf_quartile_label"]],
        left_on="unit_index",
        right_on="rr100_index",
        validate="one_to_one",
    )
    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_spikes = np.asarray(archive["expected_spikes"], dtype=float)
        baseline_info = np.asarray(archive["movie_ssi_bits_per_spike"], dtype=float) * baseline_spikes
    return {
        "manifest": manifest,
        "condition": condition,
        "images": images,
        "traces": traces,
        "quartiles": quartiles,
        "units": units,
        "moving_info": moving_info,
        "moving_spikes": moving_spikes,
        "baseline_info": baseline_info,
        "baseline_spikes": baseline_spikes,
    }


def make_unit_masks(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    images = data["images"]
    units = data["units"]
    coherence = images.corrected_reconstruction_orientation_coherence.to_numpy(float)
    image_axis = images.corrected_reconstruction_contour_axis_deg.to_numpy(float)
    strong = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= COHERENCE_MIN)
    strong_masks: dict[str, np.ndarray] = {}
    matched_masks: dict[str, np.ndarray] = {}
    for group in GROUPS:
        membership = units.sf_quartile.eq(group).to_numpy()
        strong_mask = np.zeros((100, 100), dtype=bool)
        matched_mask = np.zeros((100, 100), dtype=bool)
        strong_mask[strong] = membership
        for row in units[membership].itertuples(index=False):
            pref = float(row.prior_preferred_orientation_deg)
            osi = float(row.prior_orientation_selectivity_index)
            if not np.isfinite(pref) or not np.isfinite(osi) or osi < OSI_MIN:
                continue
            matched = strong & (axis_delta_deg(image_axis, pref) <= ORIENTATION_MATCH_MAX_DEG)
            matched_mask[matched, int(row.unit_index)] = True
        strong_masks[group] = strong_mask
        matched_masks[group] = matched_mask
    return strong_masks, matched_masks


def compute_path_tables(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    condition = data["condition"]
    image_ids = condition.image_index.to_numpy(int)
    path = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    context = condition.context.to_numpy(str)
    strong_masks, matched_masks = make_unit_masks(data)
    rng = np.random.default_rng(SEED)
    output: dict[str, list[pd.DataFrame]] = {"strong_contours": [], "contour_matched": []}
    matched_stats: dict[str, np.ndarray] = {}
    for relation, masks in (("strong_contours", strong_masks), ("contour_matched", matched_masks)):
        for group in GROUPS:
            stats = condition_sufficient_statistics(
                data["moving_info"], data["moving_spikes"], data["baseline_info"],
                data["baseline_spikes"], image_ids, masks[group],
            )
            output[relation].append(
                summarize_binned(
                    x=path, context=context, image_ids=image_ids, info=stats[0], spikes=stats[1],
                    base_info=stats[2], base_spikes=stats[3], valid=stats[4], group=group,
                    relation=relation, bins_by_context={"drift_only": 7, "microsaccade": 3}, rng=rng,
                )
            )
            if relation == "contour_matched" and group == COMPONENT_GROUP:
                matched_stats = {"info": stats[0], "spikes": stats[1], "base_info": stats[2], "base_spikes": stats[3], "valid": stats[4]}
    return pd.concat(output["strong_contours"], ignore_index=True), pd.concat(output["contour_matched"], ignore_index=True), matched_stats


def component_rms(condition: pd.DataFrame, images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with np.load(TRACE_XY, allow_pickle=False) as archive:
        xy = np.asarray(archive["score_xy_deg"], dtype=float)
    centered = xy - xy.mean(axis=1, keepdims=True)
    image_axis = images.corrected_reconstruction_contour_axis_deg.to_numpy(float)
    angles = np.deg2rad(image_axis[condition.image_index.to_numpy(int)])
    vectors_along = np.column_stack([np.cos(angles), np.sin(angles)])
    vectors_across = np.column_stack([-np.sin(angles), np.cos(angles)])
    traces = centered[condition.trace_index.to_numpy(int)]
    along = np.sqrt(np.mean(np.sum(traces * vectors_along[:, None, :], axis=2) ** 2, axis=1)) * 60.0
    across = np.sqrt(np.mean(np.sum(traces * vectors_across[:, None, :], axis=2) ** 2, axis=1)) * 60.0
    return across, along


def compute_component_table(data: dict[str, Any], stats: dict[str, np.ndarray]) -> pd.DataFrame:
    condition = data["condition"]
    image_ids = condition.image_index.to_numpy(int)
    across, along = component_rms(condition, data["images"])
    rng = np.random.default_rng(SEED + 1)
    tables = []
    context = np.full(len(condition), "all_traces", dtype=object)
    for component, values in (("across", across), ("along", along)):
        table = summarize_binned(
            x=values,
            context=context,
            image_ids=image_ids,
            info=stats["info"],
            spikes=stats["spikes"],
            base_info=stats["base_info"],
            base_spikes=stats["base_spikes"],
            valid=stats["valid"],
            group=COMPONENT_GROUP,
            relation=f"component_{component}",
            bins_by_context={"all_traces": 6},
            rng=rng,
        )
        table["component"] = component
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def panel_header(fig: plt.Figure, label: str, title: str) -> None:
    fig.text(0.01, 0.985, label, ha="left", va="top", fontsize=10, weight="bold", color=INK)
    fig.text(0.12, 0.985, title, ha="left", va="top", fontsize=9, weight="bold", color=INK, linespacing=1.05)


def draw_path_panel(table: pd.DataFrame, *, relation: str, out_path: Path, figsize: tuple[float, float], label: str, title: str) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.19, 0.15, 0.77, 0.72])
    panel_header(fig, label, title)
    fig.text(
        0.12, 0.895, FIGURE_SCOPE_LABEL,
        ha="left", va="top", fontsize=5.7, color="0.38",
    )
    # Stabilization is the matched zero-motion reference for every SF group.
    # For the half split, a single split-color marker keeps both coincident
    # group anchors visible without assigning either a false nonzero path.
    if len(GROUPS) == 2:
        ax.plot(
            [0.0], [0.0], marker="o", markersize=4.6, linestyle="none",
            fillstyle="left", markerfacecolor=COLORS[GROUPS[0]],
            markerfacecoloralt=COLORS[GROUPS[1]], markeredgecolor="0.25",
            markeredgewidth=0.7, zorder=6,
        )
    else:
        add_segmented_zero_anchor(ax, [COLORS[group] for group in GROUPS])
    for group in GROUPS:
        for context_name, filled in (("drift_only", False), ("microsaccade", True)):
            sub = table[table.sf_quartile.eq(group) & table.context.eq(context_name)].sort_values("bin")
            yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
            ax.errorbar(
                path_broken_log(sub.x_median), sub.delta_percent, yerr=yerr, color=COLORS[group], marker="o",
                mfc=COLORS[group] if filled else "white", mec=COLORS[group], ms=3.5,
                lw=1.3, ls="-", capsize=1.5,
                label=LABELS[group] if context_name == "drift_only" else None,
            )
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    format_broken_path_axis(ax)
    ax.set_xlabel("corrected retinal path length (arcmin)")
    ax.set_ylabel("SSI change (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="best", handlelength=1.5)
    ax.text(0.98, 0.02, "open: drift only\nfilled: microsaccade", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def draw_component_panel(table: pd.DataFrame, *, out_path: Path, figsize: tuple[float, float]) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.20, 0.15, 0.76, 0.72])
    panel_header(fig, "E", COMPONENT_PANEL_TITLE)
    ax.scatter(
        [0.0], [0.0], marker="o", s=18, facecolors="white",
        edgecolors="#D55E00", linewidths=1.1, zorder=5,
    )
    styles = {"across": ("#D55E00", "-", "o"), "along": ("#D55E00", "--", "s")}
    for component in ("across", "along"):
        sub = table[table.component.eq(component)].sort_values("bin")
        color, linestyle, marker = styles[component]
        yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
        ax.errorbar(
            component_broken_log(sub.x_median), sub.delta_percent, yerr=yerr, color=color, ls=linestyle,
            marker=marker, mfc="white", mec=color, ms=3.5, lw=1.4, capsize=1.5, label=component,
        )
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    format_broken_component_axis(ax)
    ax.set_xlabel("component RMS excursion (arcmin)")
    ax.set_ylabel("SSI change (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="best")
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def compose(panel_paths: dict[str, Path]) -> Path:
    page_w_in = base.layout.PAGE_SIZE_IN[0]
    page_h_in = 7.85
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0
    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    page = writer.pages[0]
    for key in ("A", "BC", "D", "EF", "G"):
        x_in, y_in, _width, _height = base.PLACEMENT_BOXES[key]
        base._place(writer, page, panel_paths[key], x_in, y_in, page_h_pt)
    output = OUT / f"{STEM}.pdf"
    with output.open("wb") as handle:
        writer.write(handle)
    return output


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=False)
    PANELS.mkdir(parents=True)
    data = build_inputs()
    n_rounds = int(data["manifest"]["n_complete_rounds"])
    n_conditions = int(len(data["condition"]))
    strong, matched, q4_stats = compute_path_tables(data)
    components = compute_component_table(data, q4_stats)
    strong.to_csv(OUT / f"{STEM}_panel_b_values.csv", index=False)
    matched.to_csv(OUT / f"{STEM}_panel_d_values.csv", index=False)
    components.to_csv(OUT / f"{STEM}_panel_e_values.csv", index=False)
    data["quartiles"].to_csv(OUT / f"{STEM}_unit_assignments.csv", index=False)

    panel_paths = {
        "A": SCHEMATIC_PANELS / "panel_a.pdf",
        "D": SCHEMATIC_PANELS / "panel_d.pdf",
        "BC": PANELS / "panel_b_corrected_sf_quartiles.pdf",
        "EF": PANELS / "panel_d_corrected_sf_quartiles.pdf",
        "G": PANELS / "panel_e_corrected_sf_q4_components.pdf",
    }
    draw_path_panel(
        strong,
        relation="strong_contours",
        out_path=panel_paths["BC"],
        figsize=base.PLACEMENT_BOXES["BC"][2:4],
        label="B",
        title=PATH_PANEL_TITLE,
    )
    draw_path_panel(
        matched,
        relation="contour_matched",
        out_path=panel_paths["EF"],
        figsize=base.PLACEMENT_BOXES["EF"][2:4],
        label="D",
        title=MATCH_PANEL_TITLE,
    )
    draw_component_panel(components, out_path=panel_paths["G"], figsize=base.PLACEMENT_BOXES["G"][2:4])
    pdf = compose(panel_paths)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "two_row_corrected_sf_quartile_figure_complete",
        "scope": (
            f"Population panels use {n_rounds} complete balanced corrected-cache rounds "
            f"({n_conditions:,} movies; interim, not a complete half-bank/full-bank result). "
            "Explanatory schematics A/C are unchanged."
        ),
        "quartiles": GROUPING_DESCRIPTION,
        "corrected_population_contract": {
            "history": "32 genuine recorded 120-Hz prehistory frames; 40 scored frames",
            "trajectory": "corrected dpi_pix crop trajectory and retinal sign",
            "strong_contour": f"corrected reconstruction coherence >= {COHERENCE_MIN}",
            "contour_match": f"prior OSI >= {OSI_MIN} and axial orientation delta <= {ORIENTATION_MATCH_MAX_DEG} deg",
            "uncertainty": f"{N_BOOTSTRAP} image-cluster bootstrap resamples; seed {SEED}",
        },
        "sources": {
            "assembly": identity(ASSEMBLED / "manifest.json"),
            "assignments": identity(ASSIGNMENTS),
            "images": identity(COHORT / "corrected100_images.csv"),
            "traces": identity(COHORT / "corrected1000_traces.csv"),
            "microsaccade_labels": identity(MICROSACCADE_LABELS),
            "trace_xy": identity(TRACE_XY),
            "panel_a": identity(panel_paths["A"]),
            "panel_c": identity(panel_paths["D"]),
        },
        "outputs": {"pdf": str(pdf.resolve()), "panels": {k: str(v.resolve()) for k, v in panel_paths.items()}},
    }
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

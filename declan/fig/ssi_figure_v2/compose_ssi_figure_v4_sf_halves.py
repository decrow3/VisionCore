#!/usr/bin/env python3
"""Build a versioned SSI Figure 4 with the new parametric-SF median halves.

Panels A and C and the final row (F-H) are retained from ``ssi_figure_v4``.
Panels B, D, and E are refreshed as follows:

* B: strong-contour population path curves for low/high SF halves;
* D: contour-aligned population path curves for low/high SF halves;
* E: aligned high-SF across/along curves on the existing RMS-excursion axis.

The original compositor, panels, and ``ssi_figure_v4.pdf`` are never written.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

from panels import panel_a_motion_schematic  # noqa: E402
from panels import panel_bcef_path_bins as path_panel  # noqa: E402
from panels import panel_d_contour_relative_stimulus  # noqa: E402
from panels import panel_g_alternative_x_axes_diagnostic as rms_analysis  # noqa: E402
from panels import panel_g_option_sheet as dose_plot  # noqa: E402
from panels import panel_h_unwrapped_edge_coherence  # noqa: E402
from panels import panel_header  # noqa: E402
from panels import panel_j_match_advantage  # noqa: E402
from panels import panel_k_patch_radius_alignment_slope  # noqa: E402
from panels import reference_layout_v3 as layout  # noqa: E402


HALF_ROOT = ROOT / "outputs" / "fig4_active_sensing" / "backimage_real_trace_sf_half_checks_v1"
ASSIGNMENTS_CSV = HALF_ROOT / "sf_half_unit_assignments.csv"
OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "sf_halves_v1"
PANELS_DIR = OUT_DIR / "panels"
OUTPUT_STEM = "ssi_figure_v4_sf_halves_v1"
RMS_PANEL_TITLE = "Across-contour spread limits\nhigh-SF benefit"

GROUPS = ("sf_low_half", "sf_high_half")
GROUP_LABELS = {"sf_low_half": "low-SF", "sf_high_half": "high-SF"}
GROUP_TO_LEGACY_KEY = {"sf_low_half": "low_lt0p5", "sf_high_half": "high_ge0p75"}
COLORS = {"sf_low_half": path_panel.BLUE, "sf_high_half": path_panel.ORANGE}

PLACEMENT_BOXES = {
    "A": (0.0944, 0.1200, 5.6000, 3.9939),
    "BC": (5.7600, 0.1200, 2.5000, 3.9939),
    "D": (0.0944, 3.9250, 2.9000, 3.8800),
    "EF": (3.0444, 3.9250, 2.6000, 3.8800),
    "G": (5.6849, 3.9250, 2.5962, 3.8800),
    "I": (0.0944, 7.8656, 2.6704, 3.0306),
    "J": (2.9148, 7.8656, 2.6704, 3.0306),
    "K": (5.7352, 7.8656, 2.6704, 3.0306),
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_percent_path_values(relation: str) -> pd.DataFrame:
    relation_dir = HALF_ROOT / relation
    summary = pd.read_csv(relation_dir / "spike_weighted_population_summary.csv")
    selection = pd.read_csv(relation_dir / "unit_image_selection.csv")
    rows: list[pd.DataFrame] = []
    for group in GROUPS:
        frame = summary[summary["sf_group"].eq(group)].copy()
        baseline_rows = frame[frame["context"].eq("stabilized")]
        if len(baseline_rows) != 1:
            raise ValueError(f"Expected one stabilized row for {relation}/{group}, found {len(baseline_rows)}")
        baseline = float(baseline_rows["population_ssi_bits_per_spike"].iloc[0])
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"Invalid stabilized SSI for {relation}/{group}: {baseline}")
        moving = frame[frame["context"].isin(["drift_only", "microsaccade"])].copy()
        moving["ssi_percent_vs_cell_baseline"] = 100.0 * moving["population_ssi_delta_vs_stabilized"] / baseline
        moving["ssi_percent_ci95_low_image_boot"] = (
            100.0 * moving["population_delta_ci95_low_image_boot"] / baseline
        )
        moving["ssi_percent_ci95_high_image_boot"] = (
            100.0 * moving["population_delta_ci95_high_image_boot"] / baseline
        )
        selected = selection[selection["sf_group"].eq(group)]
        n_pairs = int(pd.to_numeric(selected["n_selected_images"], errors="coerce").fillna(0).sum())
        moving["n_selected_units"] = int(moving["n_units"].iloc[0])
        moving["n_selected_unit_image_pairs"] = n_pairs
        moving["sf_group"] = GROUP_TO_LEGACY_KEY[group]
        moving["sf_group_label"] = GROUP_LABELS[group]
        moving["relation"] = relation
        moving["source_sf_half"] = group
        moving["stabilized_population_ssi_bits_per_spike"] = baseline
        rows.append(moving)
    return pd.concat(rows, ignore_index=True)


def _build_path_pair_panel(
    relation: str,
    *,
    labels: tuple[str, str],
    figsize: tuple[float, float],
    panel_label: str,
    panel_title: str,
    axes_box: tuple[float, float, float, float],
    separate_header: bool,
    xlabel: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    values = _load_percent_path_values(relation)
    path_panel.configure_matplotlib()
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = fig.add_axes(list(axes_box))
    ylim = path_panel.shared_ylim(values, pad_low=0.055 if relation == "contour_matched" else 0.12,
                                  pad_high=0.055 if relation == "contour_matched" else 0.14)
    path_panel.draw_pair_panel(
        ax,
        labels=labels,
        values=values,
        ylim=ylim,
        show_microsaccade_legend=(relation == "strong_contours_no_osi"),
        panel_label=panel_label,
        panel_title=panel_title,
        xlabel=xlabel,
        ylabel_x=panel_header.MIDDLE_ROW_YLABEL_X if relation == "contour_matched" else None,
        use_middle_header=True,
        draw_title=not separate_header,
    )
    if separate_header:
        header_ax = fig.add_axes([0, 0, 1, 1], zorder=20)
        header_ax.set_axis_off()
        panel_header.draw_panel_header(
            header_ax,
            panel_label,
            panel_title,
            y=path_panel.TOP_ROW_PAIR_HEADER_Y,
            letter_x=path_panel.TOP_ROW_PAIR_LETTER_X,
            letter_y_offset_pt=path_panel.TOP_ROW_PAIR_LETTER_Y_OFFSET_PT,
            title_linespacing=panel_header.PANEL_TITLE_LINESPACING,
            title_y_offset_pt=panel_header.TOP_ROW_TITLE_Y_OFFSET_PT,
        )
    else:
        panel_header.align_middle_row_xlabel(ax)
    out_path = PANELS_DIR / ("panel_b_sf_halves.pdf" if relation == "strong_contours_no_osi" else "panel_d_sf_halves.pdf")
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path, values


def _selected_high_half_aligned_images(data: dict[str, Any]) -> dict[int, np.ndarray]:
    selection_path = HALF_ROOT / "contour_matched" / "unit_image_selection.csv"
    selection = pd.read_csv(selection_path)
    selection = selection[selection["sf_group"].eq("sf_high_half")].copy()
    available_units = set(data["unit"]["unit_index"].astype(int))
    selected: dict[int, np.ndarray] = {}
    for row in selection.itertuples(index=False):
        unit_index = int(row.unit_index)
        if unit_index not in available_units:
            raise ValueError(f"Contour-matched selection contains unknown unit {unit_index}")
        text = str(row.selected_image_indices).strip()
        images = np.asarray([int(value) for value in text.split()], dtype=int) if text else np.asarray([], dtype=int)
        if images.size:
            selected[unit_index] = images
    return selected


def _compute_rms_values() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = rms_analysis.panel_c.load_dataset(rms_analysis.panel_c.MATRIX_DIR)
    metrics = rms_analysis._compute_extended_component_metrics(data)
    reference_map = rms_analysis._reference_context_by_family(metrics)
    family = next(item for item in rms_analysis.FAMILIES if item["key"] == "component_rms")
    population = {
        "key": dose_plot.POPULATION_KEY,
        "title": "Aligned High-SF Half",
        "subtitle": "upper half of valid parametric preferred SF; audited contour-matched image selection",
        "sf_group": "high_half",
        "relation": "aligned",
        "requires_orientation_tuning": True,
    }
    unit_to_images = _selected_high_half_aligned_images(data)
    values, meta = rms_analysis._compute_family(
        data,
        metrics,
        family,
        population=population,
        population_index=1,
        family_index=1,
        unit_to_images=unit_to_images,
    )
    contrasts = pd.DataFrame(meta["contrast"])
    populations = pd.DataFrame(
        [{
            "population_key": dose_plot.POPULATION_KEY,
            "population_title": population["title"],
            "population_subtitle": population["subtitle"],
            "sf_group": "sf_high_half",
            "relation": "aligned",
            "requires_orientation_tuning": True,
            "n_selected_units": int(len(unit_to_images)),
            "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        }]
    )
    reference = pd.DataFrame(
        [{"metric_family": key, **context} for key, context in reference_map.items()]
    )
    return values, contrasts, populations, reference


def _build_rms_panel() -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values, contrasts, populations, reference = _compute_rms_values()
    dose_plot.configure_matplotlib()
    fig = plt.figure(figsize=PLACEMENT_BOXES["G"][2:4], constrained_layout=False)
    ax = panel_header.add_middle_row_axes(fig)
    frame = values[values["metric_family"].eq("component_rms")]
    keep_through = int(frame["component_bin_order"].max()) - 1
    visible = frame[frame["component_bin_order"] <= keep_through]
    y_values = [0.0]
    for column in (
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ):
        array = pd.to_numeric(visible[column], errors="coerce").to_numpy(dtype=float)
        y_values.extend(array[np.isfinite(array)].tolist())
    lo, hi = min(y_values), max(y_values)
    span = max(hi - lo, 1.0)
    ylim = (lo - 0.08 * span, hi + 0.20 * span)
    dropped_note = dose_plot._draw_dose_panel(
        ax,
        metric_family="component_rms",
        values=values,
        reference=reference,
        populations=populations,
        last_bin_contrasts=contrasts,
        ylim=ylim,
        exclude_last_bins=1,
        axis_override={"min_pos": 0.9, "max_pos": 3.3, "ticks": [0, 1, 2, 3], "zero_gap": 0.5},
        final_bracket_x_offset=0.32,
    )
    panel_header.draw_middle_row_header(
        ax,
        "E",
        RMS_PANEL_TITLE,
        title_linespacing=panel_header.MIDDLE_ROW_TITLE_LINESPACING,
        color=dose_plot.INK,
    )
    ax.set_ylabel("SSI change (%)", labelpad=2.0)
    panel_header.align_middle_row_ylabel(ax)
    if dropped_note:
        ax.set_xlabel(f"{ax.get_xlabel()}\n{dropped_note}")
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    panel_header.align_middle_row_xlabel(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in label else "along" for label in labels]
        ax.legend(handles, short_labels, frameon=False, fontsize=5.9, loc="lower left",
                  handlelength=1.8, borderaxespad=0.2)
    out_path = PANELS_DIR / "panel_e_rms_sf_high_half.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path, values, contrasts, populations, reference


def _build_panel_set() -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}
    paths["A"] = panel_a_motion_schematic.build_panel(
        figsize=PLACEMENT_BOXES["A"][2:4], out_dir=PANELS_DIR,
        panel_label="A", panel_title="FEMs sharpen spatial coding",
    )
    paths["D"] = panel_d_contour_relative_stimulus.build_panel(
        figsize=PLACEMENT_BOXES["D"][2:4], out_dir=PANELS_DIR,
        panel_label="C", panel_title="Local contours define the\nrelevant image axis",
    )
    paths["BC"], tables["panel_b"] = _build_path_pair_panel(
        "strong_contours_no_osi", labels=("B", "C"), figsize=PLACEMENT_BOXES["BC"][2:4],
        panel_label="B", panel_title="Path length separates low- and\nhigh-SF benefit",
        axes_box=path_panel.TOP_ROW_PAIR_AXES_BOX, separate_header=True,
    )
    paths["EF"], tables["panel_d"] = _build_path_pair_panel(
        "contour_matched", labels=("E", "F"), figsize=PLACEMENT_BOXES["EF"][2:4],
        panel_label="D", panel_title="Contour alignment exposes a\nhigh-SF limit",
        axes_box=panel_header.MIDDLE_ROW_AXES_BOX, separate_header=False,
        xlabel="path length (arcmin; irrespective of\nspatial footprint)",
    )
    paths["G"], tables["panel_e"], tables["panel_e_contrasts"], tables["panel_e_populations"], tables["panel_e_reference"] = _build_rms_panel()

    # The last row is deliberately retained without analytical changes.
    paths["I"] = panel_h_unwrapped_edge_coherence.build_panel(
        out_dir=PANELS_DIR, figsize=PLACEMENT_BOXES["I"][2:4],
        label="F", title="Real FEM spread is contour-aligned",
    )["pdf"]
    paths["J"] = panel_j_match_advantage.build_panel(
        out_dir=PANELS_DIR, figsize=PLACEMENT_BOXES["J"][2:4],
        label="G", title="Contour-matched FEMs beat\nrotations for aligned high-SF units",
    )["pdf"]
    paths["K"] = panel_k_patch_radius_alignment_slope.build_panel(
        out_dir=PANELS_DIR, figsize=PLACEMENT_BOXES["K"][2:4],
        label="H", title="Edge following saturates near\nfoveal scale",
    )["pdf"]
    return paths, tables


def _place(writer: PdfWriter, base_page, source_pdf: Path, x_in: float, y_in: float, page_h_pt: float) -> None:
    panel_page = PdfReader(str(source_pdf)).pages[0]
    tx = x_in * 72.0
    ty = page_h_pt - y_in * 72.0 - float(panel_page.mediabox.height)
    base_page.merge_transformed_page(panel_page, Transformation().translate(tx=tx, ty=ty))


def compose() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_paths, tables = _build_panel_set()
    page_w_in, page_h_in = layout.PAGE_SIZE_IN
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0
    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    base_page = writer.pages[0]
    for key in ("A", "BC", "D", "EF", "G", "I", "J", "K"):
        x_in, y_in, _width, _height = PLACEMENT_BOXES[key]
        _place(writer, base_page, panel_paths[key], x_in, y_in, page_h_pt)
    out_pdf = OUT_DIR / f"{OUTPUT_STEM}.pdf"
    with out_pdf.open("wb") as handle:
        writer.write(handle)

    table_paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = OUT_DIR / f"{OUTPUT_STEM}_{name}_values.csv"
        table.to_csv(path, index=False)
        table_paths[name] = path
    provenance_path = OUT_DIR / f"{OUTPUT_STEM}_provenance.json"
    assignment = pd.read_csv(ASSIGNMENTS_CSV)
    valid = assignment[assignment["sf_half"].isin(GROUPS)]
    provenance = {
        "figure": OUTPUT_STEM,
        "source_figure": "outputs/fig/ssi_figure_v2/ssi_figure_v4.pdf",
        "source_compositor": "declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py",
        "updated_compositor": str(Path(__file__).relative_to(ROOT)),
        "non_overwrite_contract": "All outputs are under outputs/fig/ssi_figure_v2/sf_halves_v1.",
        "updated_panels": {
            "B": "strong-contour total-path SSI, low/high halves",
            "D": "contour-aligned total-path SSI, low/high halves",
            "E": "contour-aligned high-half component RMS excursion",
        },
        "unchanged_panels": ["A", "C", "F", "G", "H"],
        "sf_half_contract": {
            "source": str(ASSIGNMENTS_CSV.relative_to(ROOT)),
            "metric": "preferred_sf_cpd from the joint parametric SF/TF fit",
            "median_threshold_cpd": float(valid["median_threshold_cpd"].iloc[0]),
            "low_rule": "preferred_sf_cpd <= median",
            "high_rule": "preferred_sf_cpd > median",
            "counts": valid["sf_half"].value_counts().to_dict(),
        },
        "panel_paths": panel_paths,
        "table_paths": table_paths,
        "output_pdf": out_pdf,
    }
    provenance_path.write_text(json.dumps(_json_ready(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pdf": out_pdf, "provenance": provenance_path, **table_paths}


def main() -> None:
    for path in compose().values():
        print(path)


if __name__ == "__main__":
    main()

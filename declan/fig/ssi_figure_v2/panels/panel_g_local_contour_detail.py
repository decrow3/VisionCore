#!/usr/bin/env python3
"""Standalone build for Panel G (reference crop + zoomed aperture + the
across/along RMS-excursion explainer).

v3 architecture: every panel is its own independently-rendered figure,
composited onto the final page at a measured position (see
compose_ssi_figure_v3.py). The drawing logic is unchanged and still lives
in generate_ssi_figure_v2.draw_contour_components_panel.

AX_BOX reserves headroom for the panel title (matplotlib's native
ax.set_title, which draws above the axes' own y=1 edge) inside a *fixed*
figsize page, instead of letting bbox_inches="tight" grow the saved page
to fit (it needed about 0.21in/6% of this panel's own height) -- same fix
as Panel A/D, needed so panel_g_layout_boxes.py's box export/import has a
deterministic axes-fraction -> page-point mapping to invert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

import generate_ssi_figure_v2 as figure  # noqa: E402

from panels import reference_layout_v3 as layout  # noqa: E402

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels_v3"
DEFAULT_FIGSIZE = layout.PANEL_BOXES["G"][2:4]

LAYOUT_OVERRIDES_JSON = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache" / "panel_g_layout_overrides.json"

AX_BOX = (0.0, 0.0, 1.0, 0.925)  # left, bottom, width, height (figure-fraction)


def data_frac_to_page_pt(x_frac: float, y_frac: float, figsize: tuple[float, float]) -> tuple[float, float]:
    """See panels/panel_a_motion_schematic.py's function of the same name --
    identical purpose, just against this panel's own AX_BOX."""
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_pt = ax_left * panel_w_pt + x_frac * ax_w * panel_w_pt
    y_pt = ax_bottom * panel_h_pt + y_frac * ax_h * panel_h_pt
    return x_pt, y_pt


def page_pt_to_data_frac(x_pt: float, y_pt: float, figsize: tuple[float, float]) -> tuple[float, float]:
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_frac = (x_pt - ax_left * panel_w_pt) / (ax_w * panel_w_pt)
    y_frac = (y_pt - ax_bottom * panel_h_pt) / (ax_h * panel_h_pt)
    return x_frac, y_frac


def load_layout_overrides() -> dict[str, tuple[float, float, float, float]] | None:
    if not LAYOUT_OVERRIDES_JSON.exists():
        return None
    raw = json.loads(LAYOUT_OVERRIDES_JSON.read_text(encoding="utf-8"))
    return {name: tuple(box) for name, box in raw.items()}


def build_panel(figsize: tuple[float, float] = DEFAULT_FIGSIZE, out_dir: Path = OUT_DIR) -> Path:
    figure.configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    schematic_payload = figure.read_schematic_payload()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(AX_BOX))
    ax.set_axis_off()
    figure.draw_contour_components_panel(
        ax,
        schematic_payload=schematic_payload,
        layout_overrides=load_layout_overrides(),
    )

    out_path = out_dir / "panel_g.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def compute_current_boxes(figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> dict[str, tuple[float, float, float, float]]:
    """The resolved (default-merged-with-override) boxes, without a full
    build_panel() side effect -- used by panel_g_layout_boxes.py's exporter,
    which calls build_panel() separately to get a fresh background render."""
    figure.configure_matplotlib()
    schematic_payload = figure.read_schematic_payload()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(AX_BOX))
    ax.set_axis_off()
    boxes = figure.draw_contour_components_panel(
        ax,
        schematic_payload=schematic_payload,
        layout_overrides=load_layout_overrides(),
    )
    plt.close(fig)
    return boxes


def main() -> None:
    path = build_panel()
    print(path)


if __name__ == "__main__":
    main()

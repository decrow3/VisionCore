"""Create a compact multipanel highlight figure from key-PDF source figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
FIG_DIR = (
    BASE
    / "phase1_phase2_conditioning_v1"
    / "schematic_pathlength_summary_v1"
    / "unit_first_and_population_v1"
    / "figures"
)
OUT_DIR = BASE / "phase1_phase2_conditioning_v1" / "plot_collections"

PANEL_SOURCES = {
    "strong_total": FIG_DIR
    / "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
    "strong_components": FIG_DIR
    / "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
    "aligned_components": FIG_DIR
    / "contour_matched_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
    "intermediate_components": FIG_DIR
    / "contour_intermediate_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
    "orthogonal_components": FIG_DIR
    / "contour_orthogonal_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
}

SF_COLORS = {
    "Low SF": "#0072B2",
    "Middle SF": "#009E73",
    "High SF": "#D55E00",
}


def _open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _concat_h(left: Image.Image, right: Image.Image, gap: int = 70) -> Image.Image:
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def _concat_v(images: list[Image.Image], gap: int = 42) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    return canvas


def _strong_total_modulation_crop(sf_group: str) -> Image.Image:
    """Right-column movement-modulation crop from the six-panel figure."""
    img = _open(PANEL_SOURCES["strong_total"])
    # Row-specific crops from the movement-modulation column. These remove the
    # absolute-SSI panels so baseline SF-group differences do not dominate.
    y_bounds = {
        "low": (190, 655),
        "middle": (720, 1185),
        "high": (1245, 1810),
    }
    y0, y1 = y_bounds[sf_group]
    return img.crop((1430, y0, 2818, y1))


def _strong_total_modulation_stack() -> Image.Image:
    return _concat_v(
        [
            _strong_total_modulation_crop("low"),
            _strong_total_modulation_crop("middle"),
            _strong_total_modulation_crop("high"),
        ],
        gap=34,
    )


def _high_sf_component_pair(source_key: str) -> Image.Image:
    """High-SF row, across/along movement-modulation columns from a 12-panel figure."""
    img = _open(PANEL_SOURCES[source_key])
    width, _ = img.size
    # Source figures are either 4447 or 4429 px wide. Fractions keep crops stable.
    y0, y1 = 1240, 1810
    across = img.crop((int(0.255 * width), y0, int(0.488 * width), y1))
    along = img.crop((int(0.760 * width), y0, int(0.995 * width), y1))
    return _concat_h(across, along, gap=55)


def _imshow(ax: plt.Axes, image: Image.Image) -> None:
    ax.imshow(image)
    ax.set_axis_off()


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.00,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.08,
        1.08,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def _add_component_headers(ax: plt.Axes) -> None:
    ax.text(
        0.25,
        1.015,
        "across-contour component",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        0.75,
        1.015,
        "along-contour component",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def _add_total_path_sf_labels(ax: plt.Axes) -> None:
    for text, y in (("Low SF", 0.84), ("Middle SF", 0.51), ("High SF", 0.17)):
        ax.text(
            0.92,
            y,
            text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=SF_COLORS[text],
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crops = {
        "A": _strong_total_modulation_stack(),
        "B": _high_sf_component_pair("strong_components"),
        "C": _high_sf_component_pair("aligned_components"),
        "D": _high_sf_component_pair("intermediate_components"),
        "E": _high_sf_component_pair("orthogonal_components"),
    }

    fig = plt.figure(figsize=(16, 12.0))
    gs = fig.add_gridspec(
        4,
        5,
        left=0.035,
        right=0.99,
        top=0.90,
        bottom=0.06,
        hspace=0.38,
        wspace=0.20,
        width_ratios=(1.0, 1.0, 1.08, 1.08, 1.08),
    )
    axes = {
        "A": fig.add_subplot(gs[:, 0:2]),
        "B": fig.add_subplot(gs[0, 2:5]),
        "C": fig.add_subplot(gs[1, 2:5]),
        "D": fig.add_subplot(gs[2, 2:5]),
        "E": fig.add_subplot(gs[3, 2:5]),
    }

    _imshow(axes["A"], crops["A"])
    _imshow(axes["B"], crops["B"])
    _imshow(axes["C"], crops["C"])
    _imshow(axes["D"], crops["D"])
    _imshow(axes["E"], crops["E"])
    axes["A"].set_anchor("N")

    _panel_label(axes["A"], "A", "Total-path modulation by SF")
    _panel_label(axes["B"], "B", "All high-SF units: component-path modulation")
    _panel_label(axes["C"], "C", "Aligned high-SF units")
    _panel_label(axes["D"], "D", "Intermediate high-SF units")
    _panel_label(axes["E"], "E", "Orthogonal high-SF units")

    _add_total_path_sf_labels(axes["A"])
    for key in ("B", "C", "D", "E"):
        _add_component_headers(axes[key])

    fig.suptitle(
        "Real fixational motion changes SSI according to local contour geometry",
        fontsize=17,
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "Composite made from crops of the key-figures PDF source plots. "
        "Open points: drift-only snippets. Filled points: snippets containing >=1 detected microsaccade.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
    )

    png = OUT_DIR / "backimage_real_trace_key_figures_highlight_multipanel.png"
    pdf = OUT_DIR / "backimage_real_trace_key_figures_highlight_multipanel.pdf"
    manifest = OUT_DIR / "backimage_real_trace_key_figures_highlight_multipanel_manifest.json"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    manifest.write_text(
        json.dumps(
            {
                "source_pdf": str(OUT_DIR / "backimage_real_trace_key_figures_multipage.pdf"),
                "panel_sources": {key: str(value) for key, value in PANEL_SOURCES.items()},
                "output_png": str(png),
                "output_pdf": str(pdf),
                "notes": (
                    "Panel A stacks the low-, middle-, and high-SF movement-modulation rows from the strong-contour six-panel figure. "
                    "Panels B-E use high-SF rows and movement-modulation component columns from 12-panel figures."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(png)
    print(pdf)
    print(manifest)


if __name__ == "__main__":
    main()

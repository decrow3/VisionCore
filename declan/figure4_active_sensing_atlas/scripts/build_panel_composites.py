"""Build LLM-friendly composite sheets from Figure 4 atlas subpanels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "composites"


@dataclass(frozen=True)
class PanelSpec:
    label: str
    title: str
    path: Path


MODULES: dict[str, list[PanelSpec]] = {
    "A": [
        PanelSpec("A1", "Screen image to retinal crop", FIGURES / "panel_A" / "A1_retinal_movie_transform.png"),
        PanelSpec("A2", "Stabilized vs FEM movie QC", FIGURES / "panel_A" / "A2_movie_transform_qc.png"),
        PanelSpec("A3", "Local gradient sampling", FIGURES / "panel_A" / "A3_gradient_sampling_cartoon.png"),
        PanelSpec("A4", "BackImage V1-twin pipeline", FIGURES / "panel_A" / "A4_backimage_pipeline_bridge.png"),
        PanelSpec("A5", "Covariance bridge guardrail", FIGURES / "panel_A" / "A5_covariance_bridge_guardrail.png"),
    ],
    "B": [
        PanelSpec("B1", "Feature-decoding task", FIGURES / "panel_B" / "B1_task_schematic.png"),
        PanelSpec("B2", "Motion family QC", FIGURES / "panel_B" / "B2_motion_family_qc.png"),
        PanelSpec("B3", "Gain over static", FIGURES / "panel_B" / "B3_empirical_gain_vs_static.png"),
        PanelSpec("B4", "Empirical minus controls", FIGURES / "panel_B" / "B4_empirical_minus_controls.png"),
        PanelSpec("B5", "Absolute gain guardrail", FIGURES / "panel_B" / "B5_absolute_gain_guardrail.png"),
    ],
    "C": [
        PanelSpec("C1", "Latent eye-image observer", FIGURES / "panel_C" / "C1_observer_schematic.png"),
        PanelSpec("C2", "Accuracy ordering", FIGURES / "panel_C" / "C2_accuracy_ordering.png"),
        PanelSpec("C3", "Matched-static rescue", FIGURES / "panel_C" / "C3_matched_static_rescue.png"),
        PanelSpec("C4", "Posterior concentration", FIGURES / "panel_C" / "C4_posterior_concentration.png"),
        PanelSpec("C5", "Scale rescue guardrail", FIGURES / "panel_C" / "C5_scale_gap_guardrail.png"),
        PanelSpec("C6", "Compact mechanism guardrail", FIGURES / "panel_C" / "C6_compact_mechanism_guardrail.png"),
    ],
    "D": [
        PanelSpec("D1", "Local edge and motion axes", FIGURES / "panel_D" / "D1_local_axis_schematic.png"),
        PanelSpec("D2", "Axis-conditioned observer", FIGURES / "panel_D" / "D2_axis_conditioned_accuracy.png"),
        PanelSpec("D3", "Axis preference guardrail", FIGURES / "panel_D" / "D3_axis_preference_guardrail.png"),
        PanelSpec("D4", "Edge-parallel preservation", FIGURES / "panel_D" / "D4_edge_parallel_stability.png"),
        PanelSpec("D5", "Objective alignment guardrail", FIGURES / "panel_D" / "D5_objective_alignment_guardrail.png"),
    ],
    "E": [
        PanelSpec("E1", "Behavior setup example", FIGURES / "panel_E" / "E1_behavior_setup_example.png"),
        PanelSpec("E2", "Behavior alignment strength", FIGURES / "panel_E" / "E2_behavior_alignment_strength.png"),
        PanelSpec("E3", "Endpoint-zone enrichment", FIGURES / "panel_E" / "E3_parallel_zone_enrichment.png"),
        PanelSpec(
            "E6",
            "Full distribution/session diagnostic",
            FIGURES / "panel_E" / "E6_full_distribution_session_diagnostic.png",
        ),
        PanelSpec(
            "E7",
            "Confidence and signed-delta diagnostic",
            FIGURES / "panel_E" / "E7_confidence_signed_delta_diagnostic.png",
        ),
        PanelSpec("E8", "Endpoint/null diagnostic", FIGURES / "panel_E" / "E8_endpoint_null_diagnostic.png"),
        PanelSpec("E4", "Metric convention guardrail", FIGURES / "panel_E" / "E4_metric_convention_guardrail.png"),
        PanelSpec("E5", "Scope summary", FIGURES / "panel_E" / "E5_scope_summary.png"),
    ],
}

MODULE_E_CONTOUR_DIAGNOSTICS = [
    MODULES["E"][1],
    MODULES["E"][2],
    MODULES["E"][3],
    MODULES["E"][4],
    MODULES["E"][5],
]

MAIN_SPINE = [
    MODULES["A"][0],
    MODULES["A"][1],
    MODULES["A"][3],
    MODULES["B"][2],
    MODULES["B"][3],
    MODULES["C"][1],
    MODULES["C"][2],
    MODULES["D"][0],
    MODULES["D"][3],
    MODULES["E"][1],
    MODULES["E"][2],
]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_panel(spec: PanelSpec, image_box: tuple[int, int]) -> Image.Image:
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing panel image for {spec.label}: {spec.path}")
    image = Image.open(spec.path).convert("RGBA")
    return ImageOps.contain(image, image_box, Image.Resampling.LANCZOS)


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, max_width: int) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_height = int(font.size * 1.18) if hasattr(font, "size") else 18
    for line in lines:
        draw.text((x, y), line, fill=(32, 35, 38), font=font)
        y += line_height
    return y


def _make_cell(spec: PanelSpec, cell_size: tuple[int, int]) -> Image.Image:
    cell_w, cell_h = cell_size
    pad = 22
    title_h = 72
    cell = Image.new("RGB", cell_size, "white")
    draw = ImageDraw.Draw(cell)
    draw.rounded_rectangle((0, 0, cell_w - 1, cell_h - 1), radius=12, outline=(194, 199, 204), width=2)
    label_font = _font(27, bold=True)
    title_font = _font(22)
    draw.text((pad, pad - 2), spec.label, fill=(10, 75, 135), font=label_font)
    _draw_wrapped(draw, (pad + 62, pad + 1), spec.title, title_font, cell_w - pad * 2 - 62)

    image_box = (cell_w - pad * 2, cell_h - title_h - pad * 2)
    image = _load_panel(spec, image_box)
    x = pad + (image_box[0] - image.width) // 2
    y = title_h + pad + (image_box[1] - image.height) // 2
    cell.paste(Image.new("RGB", image.size, "white"), (x, y))
    cell.paste(image, (x, y), image)
    return cell


def make_grid(
    specs: list[PanelSpec],
    title: str,
    out_path: Path,
    *,
    columns: int = 2,
    cell_size: tuple[int, int] = (1320, 780),
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gap = 34
    margin = 44
    header_h = 96
    rows = (len(specs) + columns - 1) // columns
    width = margin * 2 + columns * cell_size[0] + (columns - 1) * gap
    height = margin * 2 + header_h + rows * cell_size[1] + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), title, fill=(18, 24, 31), font=_font(40, bold=True))
    draw.line((margin, margin + 64, width - margin, margin + 64), fill=(176, 184, 192), width=2)
    for i, spec in enumerate(specs):
        row = i // columns
        col = i % columns
        x = margin + col * (cell_size[0] + gap)
        y = margin + header_h + row * (cell_size[1] + gap)
        sheet.paste(_make_cell(spec, cell_size), (x, y))
    sheet.save(out_path, optimize=True)
    return out_path


def main() -> None:
    for module, specs in MODULES.items():
        make_grid(specs, f"Figure 4 Module {module} Composite", OUT_DIR / f"module_{module}_composite.png")
    make_grid(
        MODULE_E_CONTOUR_DIAGNOSTICS,
        "Figure 4 Module E Contour-Following Diagnostics",
        OUT_DIR / "module_E_contour_following_diagnostics.png",
        columns=1,
        cell_size=(1800, 1250),
    )
    make_grid(MAIN_SPINE, "Figure 4 Main Spine Composite Candidate", OUT_DIR / "main_spine_composite.png", columns=3)
    print(f"Wrote composites to {OUT_DIR}")


if __name__ == "__main__":
    main()

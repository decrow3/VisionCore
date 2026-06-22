"""Build the selected provisional Figure 4 composite.

This assembles the current single-panel choices into one A-E draft figure.
It intentionally uses the promotion-candidate PNGs directly so the composite
is a layout draft, not a hidden reanalysis step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "composites"


@dataclass(frozen=True)
class SelectedPanel:
    label: str
    title: str
    path: Path
    read: str


SELECTED_PANELS = (
    SelectedPanel(
        "A",
        "One image becomes a movie",
        FIGURES / "panel_A/promotion_candidates/4A_candidate_3_real_high_contrast_positive.png",
        "recorded eye drift samples different retinal views",
    ),
    SelectedPanel(
        "B",
        "Motion adds feature information",
        FIGURES / "panel_B/promotion_candidates/4B_candidate_3_absolute_gain_guardrail.png",
        "recorded drift improves the model response over static input",
    ),
    SelectedPanel(
        "C",
        "Features survive hidden eye position",
        FIGURES / "panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery.png",
        "joint inference recovers local features without the eye trace",
    ),
    SelectedPanel(
        "D",
        "Along-edge motion preserves structure",
        FIGURES / "panel_D/promotion_candidates/4D_candidate_1_edge_parallel_preservation.png",
        "across-edge shifts disrupt pixels and model responses more",
    ),
    SelectedPanel(
        "E",
        "Real drift follows coherent edges",
        FIGURES / "panel_E/promotion_candidates/4E_candidate_3a_image_coherence_focus.png",
        "behavioral alignment strengthens when the local edge is clear",
    ),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    fill: tuple[int, int, int],
) -> int:
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
    line_height = int(getattr(font, "size", 18) * 1.16)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def _load_panel(path: Path, image_box: tuple[int, int]) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGBA")
    image = ImageOps.exif_transpose(image)
    return ImageOps.contain(image, image_box, Image.Resampling.LANCZOS)


def _draw_panel(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    panel: SelectedPanel,
    box: tuple[int, int, int, int],
) -> None:
    x, y, w, h = box
    label_font = _font(56, bold=True)
    title_font = _font(28, bold=True)
    note_font = _font(20)
    title_x = x + 70

    draw.text((x, y - 5), panel.label, fill=(14, 66, 112), font=label_font)
    _draw_wrapped(
        draw,
        (title_x, y + 6),
        panel.title,
        title_font,
        w - 70,
        fill=(18, 24, 31),
    )
    draw.text((title_x, y + 42), panel.read, fill=(86, 95, 104), font=note_font)

    image_y = y + 78
    image_h = h - 78
    image_box = (w, image_h)
    image = _load_panel(panel.path, image_box)
    image_x = x + (w - image.width) // 2
    paste_y = image_y + (image_h - image.height) // 2
    sheet.paste(Image.new("RGB", image.size, "white"), (image_x, paste_y))
    sheet.paste(image, (image_x, paste_y), image)


def _write_manifest(out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["panel", "title", "selected_asset", "read"],
            lineterminator="\n",
        )
        writer.writeheader()
        for panel in SELECTED_PANELS:
            writer.writerow(
                {
                    "panel": panel.label,
                    "title": panel.title,
                    "selected_asset": panel.path.relative_to(ROOT).as_posix(),
                    "read": panel.read,
                }
            )


def _write_caption(out_path: Path) -> None:
    caption = """# Figure 4 Selected Composite v3

Status: provisional selected-panel draft, 2026-06-21.

Selected assets:

- 4A: `figures/panel_A/promotion_candidates/4A_candidate_3_real_high_contrast_positive.png`
- 4B: `figures/panel_B/promotion_candidates/4B_candidate_3_absolute_gain_guardrail.png`
- 4C: `figures/panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery.png`
- 4D: `figures/panel_D/promotion_candidates/4D_candidate_1_edge_parallel_preservation.png`
- 4E: `figures/panel_E/promotion_candidates/4E_candidate_3a_image_coherence_focus.png`

Draft legend:

Figure 4. Small eye movements turn a static natural image into an informative
retinal movie. (A) During a recorded fixation, one image becomes a sequence of
shifted retinal views. (B) Adding recorded drift to the V1-twin response
increases recoverable feature information relative to a static input. (C) That
feature information remains recoverable when the observer must infer features
without being given the eye trace. (D) A local reason is visible in the image:
motion along an edge changes pixels and model responses less than matched
motion across the edge. (E) Measured eye drift shows the same contour-following
geometry most clearly when the local image supplies a coherent edge axis. The
figure argues for convergence between useful retinal-movie geometry and
measured behavior, not for a completed proof of behavioral optimality.
"""
    out_path.write_text(caption, encoding="utf-8")


def build() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width = 3000
    margin = 80
    gap = 58
    header_h = 150
    row_a_h = 850
    row_h = 850
    col_w = (width - 2 * margin - gap) // 2
    span_w = width - 2 * margin
    height = margin + header_h + row_a_h + gap + row_h + gap + row_h + margin

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 35),
        "Figure 4. Small eye movements turn images into informative retinal movies",
        fill=(18, 24, 31),
        font=_font(44, bold=True),
    )
    draw.text(
        (margin, 94),
        "Story draft: movie input -> feature gain -> hidden-eye recovery -> edge mechanism -> behavior.",
        fill=(86, 95, 104),
        font=_font(24),
    )
    draw.line((margin, 132, width - margin, 132), fill=(184, 190, 197), width=2)

    y0 = margin + header_h
    _draw_panel(sheet, draw, SELECTED_PANELS[0], (margin, y0, span_w, row_a_h))

    y1 = y0 + row_a_h + gap
    _draw_panel(sheet, draw, SELECTED_PANELS[1], (margin, y1, col_w, row_h))
    _draw_panel(sheet, draw, SELECTED_PANELS[2], (margin + col_w + gap, y1, col_w, row_h))

    y2 = y1 + row_h + gap
    _draw_panel(sheet, draw, SELECTED_PANELS[3], (margin, y2, col_w, row_h))
    _draw_panel(sheet, draw, SELECTED_PANELS[4], (margin + col_w + gap, y2, col_w, row_h))

    png = OUT_DIR / "figure4_selected_v3.png"
    pdf = OUT_DIR / "figure4_selected_v3.pdf"
    manifest = OUT_DIR / "figure4_selected_v3_manifest.csv"
    caption = OUT_DIR / "figure4_selected_v3_caption.md"

    sheet.save(png, optimize=True)
    sheet.save(pdf, "PDF", resolution=300.0)
    _write_manifest(manifest)
    _write_caption(caption)
    return [png, pdf, manifest, caption]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()

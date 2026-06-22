"""Build draft option sheets for provisional Figure 4A.

The sheets are review artifacts: they reuse existing Panel A subpanels and
compose a few story/layout choices without changing the underlying evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PANEL_A = ROOT / "figures" / "panel_A"
OUT_DIR = PANEL_A / "options"


@dataclass(frozen=True)
class Asset:
    label: str
    title: str
    filename: str

    @property
    def path(self) -> Path:
        return PANEL_A / self.filename


@dataclass(frozen=True)
class Slot:
    asset: Asset
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Option:
    slug: str
    title: str
    recommendation: str
    use_when: str
    includes: str
    boundary: str
    slots: tuple[Slot, ...]


ASSETS = {
    "A1": Asset("A1", "Retinal movie transform", "A1_retinal_movie_transform.png"),
    "A2": Asset("A2", "Retinal-motion QC", "A2_movie_transform_qc.png"),
    "A3": Asset("A3", "Gradient sampling mechanism", "A3_gradient_sampling_cartoon.png"),
    "A4": Asset("A4", "BackImage / V1-twin pipeline", "A4_backimage_pipeline_bridge.png"),
    "A5": Asset("A5", "Covariance bridge guardrail", "A5_covariance_bridge_guardrail.png"),
}


OPTIONS: tuple[Option, ...] = (
    Option(
        slug="4A_option_1_main_spine_recommended",
        title="Option 1: Main-Spine Premise + QC",
        recommendation="Recommended default for the compressed main figure.",
        use_when="4A needs to teach the physical retinal-movie premise, verify the rendering, and hand off quickly to the downstream BackImage panels.",
        includes="A1 large premise, A2 quantitative stabilized-vs-FEM QC, A4 compact downstream provenance.",
        boundary="Does not claim active-sensing optimality; it only establishes that the fixed screen image becomes a structured retinal movie.",
        slots=(
            Slot(ASSETS["A1"], (70, 320, 1635, 960)),
            Slot(ASSETS["A2"], (1690, 320, 2530, 960)),
            Slot(ASSETS["A4"], (70, 1030, 2530, 1600)),
        ),
    ),
    Option(
        slug="4A_option_2_mechanism_teaching",
        title="Option 2: Premise + Local Geometry",
        recommendation="Good if Figure 4 needs 4A to set up the contour-axis logic used later.",
        use_when="readers need the extra step from retinal shift to image-dependent response change before seeing edge-parallel preservation and behavior.",
        includes="A1 retinal crop premise, A3 gradient/axis cartoon, A4 downstream V1-twin pipeline.",
        boundary="The gradient cartoon is explanatory, not a result; quantitative rendering QC should move to supplement or caption.",
        slots=(
            Slot(ASSETS["A1"], (70, 320, 2530, 890)),
            Slot(ASSETS["A3"], (70, 965, 1445, 1600)),
            Slot(ASSETS["A4"], (1515, 965, 2530, 1600)),
        ),
    ),
    Option(
        slug="4A_option_3_qc_provenance_compact",
        title="Option 3: QC + Provenance Compact",
        recommendation="Space-saving alternative if the main figure can assume the movie premise visually.",
        use_when="the paper already has a strong retinal-movie visual elsewhere and 4A mainly needs to certify the input and provenance.",
        includes="A2 rendering/QC large, A4 pipeline/provenance large, A1 small visual anchor.",
        boundary="Less pedagogical; it may be too abrupt for readers who have not already internalized the retinal movie transform.",
        slots=(
            Slot(ASSETS["A2"], (70, 320, 1280, 1110)),
            Slot(ASSETS["A4"], (1350, 320, 2530, 1110)),
            Slot(ASSETS["A1"], (70, 1180, 2530, 1600)),
        ),
    ),
    Option(
        slug="4A_option_4_bridge_supplement",
        title="Option 4: Premise + Covariance Bridge",
        recommendation="Best as supplement or bridge, not the main default.",
        use_when="4A should explicitly connect retinal movies to earlier FEM-linked covariance evidence.",
        includes="A1 premise, A2 rendering/QC, A5 covariance bridge guardrail, A4 provenance inset.",
        boundary="A5 has mixed-denominator caveats, so this sheet should not be the main claim unless the caption foregrounds that limitation.",
        slots=(
            Slot(ASSETS["A1"], (70, 320, 1280, 900)),
            Slot(ASSETS["A2"], (1350, 320, 2530, 900)),
            Slot(ASSETS["A5"], (70, 975, 1445, 1600)),
            Slot(ASSETS["A4"], (1515, 975, 2530, 1600)),
        ),
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


def _wrapped_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            if _text_width(candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    if hasattr(font, "getbbox"):
        box = font.getbbox(text)
        return int(box[2] - box[0])
    return int(font.getsize(text)[0])


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    fill: tuple[int, int, int] = (38, 43, 48),
    line_scale: float = 1.22,
) -> int:
    x, y = xy
    line_h = int(getattr(font, "size", 18) * line_scale)
    for line in _wrapped_lines(text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _paste_asset(sheet: Image.Image, draw: ImageDraw.ImageDraw, slot: Slot) -> None:
    x0, y0, x1, y1 = slot.box
    pad = 22
    header_h = 48
    _panel_box(draw, (x0, y0, x1, y1))
    label_font = _font(28, bold=True)
    title_font = _font(24)
    draw.text((x0 + pad, y0 + 16), slot.asset.label, font=label_font, fill=(24, 82, 132))
    draw.text((x0 + pad + 58, y0 + 20), slot.asset.title, font=title_font, fill=(42, 47, 52))

    if not slot.asset.path.exists():
        raise FileNotFoundError(slot.asset.path)
    image = Image.open(slot.asset.path).convert("RGBA")
    image_box = (x1 - x0 - pad * 2, y1 - y0 - pad * 2 - header_h)
    image = _contain(image, image_box)
    ix = x0 + pad + (image_box[0] - image.width) // 2
    iy = y0 + pad + header_h + (image_box[1] - image.height) // 2
    sheet.paste(image, (ix, iy), image)


def _panel_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=18, outline=(191, 199, 207), width=2, fill=(255, 255, 255))
    else:
        draw.rectangle(box, outline=(191, 199, 207), fill=(255, 255, 255))


def _contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    max_w, max_h = box
    scale = min(max_w / image.width, max_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
    return image.resize(new_size, resample)


def _make_option(option: Option) -> Path:
    width, height = 2600, 1700
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    margin = 70
    title_font = _font(48, bold=True)
    rec_font = _font(26, bold=True)
    body_font = _font(24)
    small_font = _font(22)

    draw.text((margin, 52), option.title, font=title_font, fill=(20, 26, 32))
    draw.text((margin, 122), option.recommendation, font=rec_font, fill=(21, 91, 139))
    y = _draw_wrapped(draw, (margin, 170), f"Use when: {option.use_when}", body_font, 1160)
    _draw_wrapped(draw, (margin, y + 10), f"Includes: {option.includes}", body_font, 1160)
    _draw_wrapped(
        draw,
        (1390, 170),
        f"Claim boundary: {option.boundary}",
        small_font,
        1110,
        fill=(101, 75, 35),
    )
    draw.line((margin, 305, width - margin, 305), fill=(183, 190, 198), width=2)

    for slot in option.slots:
        x0, y0, x1, y1 = slot.box
        shifted = Slot(slot.asset, (x0, y0 + 35, x1, y1 + 35))
        _paste_asset(sheet, draw, shifted)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{option.slug}.png"
    sheet.save(out_path, optimize=True)
    return out_path


def _make_contact_sheet(paths: list[Path]) -> Path:
    width, height = 2600, 2200
    margin, gap = 70, 44
    thumb_w, thumb_h = 1208, 790
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 52), "Figure 4A Draft Panel Option Sheets", font=_font(48, bold=True), fill=(20, 26, 32))
    draw.text(
        (margin, 122),
        "Compare story weight: premise, QC/provenance, mechanism teaching, and supplement bridge.",
        font=_font(27),
        fill=(72, 80, 88),
    )
    draw.line((margin, 185, width - margin, 185), fill=(183, 190, 198), width=2)

    for i, path in enumerate(paths):
        option = OPTIONS[i]
        row = i // 2
        col = i % 2
        x = margin + col * (thumb_w + gap)
        y = 235 + row * (thumb_h + 220)
        _panel_box(draw, (x, y, x + thumb_w, y + thumb_h))
        image = Image.open(path).convert("RGBA")
        image = _contain(image, (thumb_w - 34, thumb_h - 34))
        sheet.paste(image, (x + 17 + (thumb_w - 34 - image.width) // 2, y + 17), image)
        _draw_wrapped(
            draw,
            (x, y + thumb_h + 24),
            f"{i + 1}. {option.title.replace('Option ' + str(i + 1) + ': ', '')}: {option.recommendation}",
            _font(24, bold=True),
            thumb_w,
        )
        _draw_wrapped(draw, (x, y + thumb_h + 88), option.boundary, _font(21), thumb_w, fill=(92, 79, 60))

    out_path = OUT_DIR / "4A_option_sheet_contact.png"
    sheet.save(out_path, optimize=True)
    return out_path


def _write_markdown(paths: list[Path], contact: Path) -> Path:
    md_path = OUT_DIR / "README.md"
    lines = [
        "# Figure 4A Draft Panel Option Sheets",
        "",
        "Status: draft review sheets generated from existing Panel A assets.",
        "",
        "![Contact sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_A/options/4A_option_sheet_contact.png)",
        "",
        "## Recommendation",
        "",
        "Use Option 1 as the default compressed main-figure 4A. It teaches the retinal-movie premise, shows the stabilized-vs-FEM QC, and hands off to the canonical BackImage/V1-twin pipeline without importing the mixed-denominator covariance bridge into the main claim.",
        "",
    ]
    for i, (option, path) in enumerate(zip(OPTIONS, paths), start=1):
        rel = path.relative_to(ROOT)
        lines.extend(
            [
                f"## Option {i}: {option.title.split(': ', 1)[1]}",
                "",
                f"![{option.title}](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/{rel})",
                "",
                f"Recommendation: {option.recommendation}",
                "",
                f"Use when: {option.use_when}",
                "",
                f"Includes: {option.includes}",
                "",
                f"Claim boundary: {option.boundary}",
                "",
            ]
        )
    lines.extend(
        [
            "## Source Assets",
            "",
            *[
                f"- `{asset.label}`: `figures/panel_A/{asset.filename}`"
                for asset in ASSETS.values()
            ],
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    paths = [_make_option(option) for option in OPTIONS]
    contact = _make_contact_sheet(paths)
    md_path = _write_markdown(paths, contact)
    for path in [*paths, contact, md_path]:
        print(path)


if __name__ == "__main__":
    main()

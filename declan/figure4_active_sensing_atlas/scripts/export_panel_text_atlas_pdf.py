"""Export the panel text atlas Markdown to a PDF with embedded images.

This intentionally avoids external PDF dependencies. It renders enough
Markdown for the atlas contact sheet: headings, paragraphs, fenced text blocks,
bullets, and local image links.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MD = REPO_ROOT / "declan/figure4_active_sensing_atlas/panel_text_atlas.md"
DEFAULT_PDF = REPO_ROOT / "declan/figure4_active_sensing_atlas/panel_text_atlas.pdf"

PAGE_W = 1275
PAGE_H = 1650
MARGIN_X = 90
MARGIN_Y = 82
GAP = 18
TEXT_W = PAGE_W - 2 * MARGIN_X
BG = "white"
INK = "#242a2f"
MUTED = "#65717a"
RULE = "#d8dde3"
CODE_BG = "#f5f7f9"
LINK_BLUE = "#244f7a"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    elif bold:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    else:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size=size)


FONTS = {
    "h1": _font(32, bold=True),
    "h2": _font(25, bold=True),
    "h3": _font(20, bold=True),
    "body": _font(15),
    "body_bold": _font(15, bold=True),
    "small": _font(13),
    "code": _font(13, mono=True),
}


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    src: str = ""
    alt: str = ""


class PdfCanvas:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y

    def _new_page(self) -> Image.Image:
        return Image.new("RGB", (PAGE_W, PAGE_H), BG)

    def add_page(self) -> None:
        self.pages.append(self.page)
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y

    def ensure(self, height: int) -> None:
        if self.y + height > PAGE_H - MARGIN_Y:
            self.add_page()

    def line(self) -> None:
        self.ensure(20)
        self.draw.line((MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y), fill=RULE, width=2)
        self.y += 18


def _clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    text = _clean_inline(text).strip()
    if not text:
        return []
    lines: list[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _text_width(draw, candidate, font) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_text_block(
    canvas: PdfCanvas,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    color: str = INK,
    x: int = MARGIN_X,
    width: int = TEXT_W,
    line_gap: int = 5,
    before: int = 0,
    after: int = GAP,
) -> None:
    lines = _wrap_text(canvas.draw, text, font, width)
    if not lines:
        canvas.y += after
        return
    line_h = font.size + line_gap
    height = before + line_h * len(lines) + after
    canvas.ensure(height)
    canvas.y += before
    for line in lines:
        canvas.draw.text((x, canvas.y), line, font=font, fill=color)
        canvas.y += line_h
    canvas.y += after


def _parse_markdown(md_path: Path) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    code: list[str] | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(Block("paragraph", " ".join(line.strip() for line in paragraph)))
            paragraph = []

    for raw in md_path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if code is None:
                flush_paragraph()
                code = []
            else:
                blocks.append(Block("code", "\n".join(code)))
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            flush_paragraph()
            blocks.append(Block("image", alt=image_match.group(1), src=image_match.group(2)))
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_paragraph()
            blocks.append(Block("heading", heading_match.group(2), level=len(heading_match.group(1))))
            continue
        if line.startswith("- "):
            flush_paragraph()
            blocks.append(Block("bullet", line[2:].strip()))
            continue
        paragraph.append(line)
    flush_paragraph()
    return blocks


def _draw_heading(canvas: PdfCanvas, block: Block) -> None:
    if block.level == 1:
        font = FONTS["h1"]
        before, after = 0, 24
    elif block.level == 2:
        font = FONTS["h2"]
        before, after = 20, 16
        canvas.ensure(70)
        if canvas.y > MARGIN_Y + 10:
            canvas.line()
    else:
        font = FONTS["h3"]
        before, after = 16, 12
    _draw_text_block(canvas, block.text, font=font, color=INK, before=before, after=after, line_gap=6)


def _draw_code(canvas: PdfCanvas, text: str) -> None:
    font = FONTS["code"]
    lines: list[str] = []
    for raw_line in text.splitlines():
        wrapped = _wrap_text(canvas.draw, raw_line, font, TEXT_W - 32)
        lines.extend(wrapped or [""])
    line_h = font.size + 5
    height = line_h * len(lines) + 30
    canvas.ensure(height)
    x0 = MARGIN_X
    y0 = canvas.y
    canvas.draw.rounded_rectangle(
        (x0, y0, PAGE_W - MARGIN_X, y0 + height - 12),
        radius=8,
        fill=CODE_BG,
        outline=RULE,
        width=1,
    )
    canvas.y += 12
    for line in lines:
        canvas.draw.text((MARGIN_X + 16, canvas.y), line, font=font, fill=INK)
        canvas.y += line_h
    canvas.y += GAP


def _draw_image(canvas: PdfCanvas, md_path: Path, block: Block) -> None:
    img_path = (md_path.parent / block.src).resolve()
    if not img_path.exists():
        _draw_text_block(canvas, f"[missing image: {block.src}]", font=FONTS["body"], color="#b00020")
        return
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        max_w = TEXT_W
        max_h = 620
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
    caption_h = FONTS["small"].size + 18
    canvas.ensure(resized.height + caption_h + GAP)
    x = MARGIN_X + (TEXT_W - resized.width) // 2
    canvas.page.paste(resized, (x, canvas.y))
    canvas.y += resized.height + 8
    if block.alt:
        _draw_text_block(canvas, block.alt, font=FONTS["small"], color=MUTED, before=0, after=GAP, line_gap=4)
    else:
        canvas.y += GAP


def export(md_path: Path, pdf_path: Path) -> None:
    blocks = _parse_markdown(md_path)
    canvas = PdfCanvas()
    for block in blocks:
        if block.kind == "heading":
            _draw_heading(canvas, block)
        elif block.kind == "paragraph":
            _draw_text_block(canvas, block.text, font=FONTS["body"], after=GAP)
        elif block.kind == "bullet":
            _draw_text_block(canvas, f"- {block.text}", font=FONTS["body"], x=MARGIN_X + 24, width=TEXT_W - 24, after=8)
        elif block.kind == "code":
            _draw_code(canvas, block.text)
        elif block.kind == "image":
            _draw_image(canvas, md_path, block)
    canvas.pages.append(canvas.page)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = canvas.pages
    first.save(pdf_path, "PDF", resolution=150.0, save_all=True, append_images=rest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    export(args.markdown, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

"""Build single-panel promotion candidates for Figure 4D.

Each generated PNG is a candidate for the one promoted Panel D. Existing
axis/preservation/guardrail subpanels are reused directly so the review surface
chooses among current evidence, not a new composite.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
PANEL_D = ATLAS / "figures" / "panel_D"
OUT_DIR = PANEL_D / "promotion_candidates"


@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    source_png: str
    values_csv: str
    recommendation: str
    boundary: str


CANDIDATES = (
    Candidate(
        slug="4D_candidate_1_axis_feature_recovery",
        title="Along-edge feature recovery",
        source_png="D2_axis_feature_recovery.png",
        values_csv="panel_D_axis_feature_recovery_values.csv",
        recommendation="Selected revised 4D: along-edge trajectory priors recover more matched-static feature signal than across-edge priors when eye trajectory is latent.",
        boundary="Matched-static feature-posterior endpoint; hard-negative controls remain the explicit guardrail.",
    ),
    Candidate(
        slug="4D_candidate_2_image_identity_rescue",
        title="Image-identity rescue",
        source_png="D2_axis_conditioned_accuracy.png",
        values_csv="panel_D_axis_conditioned_values.csv",
        recommendation="Context view: local image-axis priors rescue image-identity decoding above zero-eye when eye position is latent.",
        boundary="Image-identity endpoint is supporting context; feature recovery is the promoted Figure 4 readout.",
    ),
    Candidate(
        slug="4D_candidate_3_edge_parallel_preservation",
        title="Edge-parallel preservation",
        source_png="D4_edge_parallel_stability.png",
        values_csv="panel_D_edge_stability_values.csv",
        recommendation="Use as mechanism support: edge-parallel displacement preserves pixels and V1-twin responses better than matched orthogonal displacement.",
        boundary="Supporting preservation result, not the promoted D readout story.",
    ),
    Candidate(
        slug="4D_candidate_4_axis_preference_guardrail",
        title="Axis preference guardrail",
        source_png="D3_axis_preference_guardrail.png",
        values_csv="panel_D_axis_preference_values.csv",
        recommendation="Best if the main pressure is preventing a universal edge-parallel policy overread.",
        boundary="Caveat-forward and less direct as the promoted positive result.",
    ),
    Candidate(
        slug="4D_candidate_5_raw_edge_objective_guardrail",
        title="Raw-edge objective guardrail",
        source_png="D5_objective_alignment_guardrail.png",
        values_csv="panel_D_objective_guardrail_values.csv",
        recommendation="Use if Panel D should foreground that response-objective models do not yet beat raw edge geometry.",
        boundary="Negative guardrail; it does not show edge-parallel preservation by itself.",
    ),
)


def _copy_candidate(candidate: Candidate) -> tuple[Path, pd.DataFrame]:
    source = PANEL_D / candidate.source_png
    if not source.exists():
        raise FileNotFoundError(source)
    out = OUT_DIR / f"{candidate.slug}.png"
    shutil.copy2(source, out)
    pdf = source.with_suffix(".pdf")
    if pdf.exists():
        shutil.copy2(pdf, out.with_suffix(".pdf"))

    values = pd.read_csv(PANEL_D / candidate.values_csv).assign(candidate=candidate.slug)
    return out, values


def _make_contact_sheet(paths: list[Path]) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
        scale = min(box[0] / image.width, box[1] / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
        return image.resize(size, resample)

    def wrap(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, max_width: int) -> int:
        x, y = xy
        line = ""
        line_h = int(getattr(fnt, "size", 18) * 1.18)
        for word in text.split():
            candidate = word if not line else f"{line} {word}"
            width = fnt.getbbox(candidate)[2] - fnt.getbbox(candidate)[0] if hasattr(fnt, "getbbox") else fnt.getsize(candidate)[0]
            if width <= max_width:
                line = candidate
            else:
                if line:
                    draw.text((x, y), line, font=fnt, fill=(45, 49, 54))
                    y += line_h
                line = word
        if line:
            draw.text((x, y), line, font=fnt, fill=(45, 49, 54))
            y += line_h
        return y

    width, height = 2600, 3220
    margin, gap = 62, 36
    thumb_w, thumb_h = 1210, 650
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 45), "Figure 4D Single-Panel Promotion Candidates", font=font(46, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Choose the one panel that should carry the along-vs-across readout story.",
        font=font(26),
        fill=(73, 80, 88),
    )
    draw.line((margin, 170, width - margin, 170), fill=(183, 190, 198), width=2)

    for i, (candidate, path) in enumerate(zip(CANDIDATES, paths)):
        row = i // 2
        col = i % 2
        x = margin + col * (thumb_w + gap)
        y = 215 + row * (thumb_h + 330)
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(191, 199, 207), fill="white")
        image = Image.open(path).convert("RGBA")
        image = contain(image, (thumb_w - 34, thumb_h - 34))
        sheet.paste(image, (x + 17 + (thumb_w - 34 - image.width) // 2, y + 17), image)
        draw.text((x, y + thumb_h + 24), f"{i + 1}. {candidate.title}", font=font(25, True), fill=(20, 26, 32))
        wrap(draw, (x, y + thumb_h + 64), candidate.recommendation, font(22), thumb_w)
        wrap(draw, (x, y + thumb_h + 136), f"Boundary: {candidate.boundary}", font(20), thumb_w)

    out = OUT_DIR / "4D_single_panel_candidate_sheet.png"
    sheet.save(out, optimize=True)
    return out


def _write_readme(paths: list[Path]) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Figure 4D Single-Panel Promotion Candidates",
        "",
        "Status: candidate 1 selected provisionally for the promoted local image-axis readout panel.",
        "",
        "![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_D/promotion_candidates/4D_single_panel_candidate_sheet.png)",
        "",
        "## Recommendation",
        "",
        "Selected provisional 4D: candidate 1, `4D_candidate_1_axis_feature_recovery.png`. It gives the corrected readout story: along-edge trajectory priors recover more matched-static feature signal than across-edge priors when eye trajectory is latent. Candidate 2 is retained as image-identity context, candidate 3 as preservation support, and candidates 4-5 as guardrails.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "- `4D_single_panel_candidate_sheet.png`",
            "- `4D_single_panel_candidate_values.csv`",
            "",
            "## Claim Boundary",
            "",
            "Panel D supports image-conditioned useful motion axes through a matched-static feature-recovery result. It does not support a universal edge-parallel policy or a settled response objective that beats raw image geometry.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    value_rows: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        path, values = _copy_candidate(candidate)
        paths.append(path)
        value_rows.append(values)

    pd.concat(value_rows, ignore_index=True, sort=False).to_csv(OUT_DIR / "4D_single_panel_candidate_values.csv", index=False)
    sheet = _make_contact_sheet(paths)
    _write_readme(paths)
    for path in paths + [sheet, OUT_DIR / "README.md", OUT_DIR / "4D_single_panel_candidate_values.csv"]:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
